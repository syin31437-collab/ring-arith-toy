"""外部验证一：低成本空气质量传感器（OpenAQ v3，2026-11-01，库伯：先做这个——故障是自然发生的，不是我们投的毒）。

映射（零重拟合）：节点 = 参考站 A 周围 3 km 内的低成本 PM2.5 传感器（Clarity / AirGradient 等）；
  真值 = 参考站 A（官方监测站）；秤 = 3–15 km 外另一个参考站 B（独立仪器、独立位置，和低成本传感器的故障不同源 → ζ ≈ 0）；
  题 = 小时；单位 log10(PM2.5)；真值 < 3 µg/m³ 的小时丢掉（取对数后被噪声主导）。
站点（sites.py，看数值之前定的规则）：6 个城市 17 个站点，每站 5–9 个节点。数据 2025-09-01 → 2025-10-27（8 周，fetch.py）。
节点保留规则：该站 8 周里有数据的小时 ≥ 80% 的传感器；保留后不足 5 个的站点丢掉。题 = A、B 和全部保留节点都有数的小时。
窗口：8 周切成 4 个 2 周窗口；每个窗口第一周 = 校准期（现实里不知道干净不干净，按部署时的做法当作校准），第二周 = 评估期。
被骗 = |裁判输出 − 真值| > h，h = log10(1.25)（偏离参考站超过 25%）；裁判 = 软 / 硬2。
故障普查（描述性，回答"现实里的故障是偏还是变吵"）：每个节点 × 窗口，拿评估周和它自己的校准周比（相对真值 A）：
  偏   = |中位偏移变化| > log10(1.12)（12%）；变吵 = MAD 比 > 1.5；两者都有 = "偏 + 吵"。
检测：闸门版 EWMA（commonmode.run_lines cm2），校准周起线、评估周越线的节点 vs 普查判为有故障的节点。

选站修正（第一次跑后；只看"有没有数据"、没看数值）：第一版 17 个站点只剩 1 个——选站规则只要求"2025-11 之后还在线"，
  没要求研究窗口（9–10 月）里有数据：洛杉矶的秤 B、伦敦的真值 A 窗口里 0 条，很多低成本传感器 9–10 月还没装。
  第一版跑出的 4 格（全来自 San Francisco 站）作废、但已经看过 → 第二版结果同时报"含 / 不含 San Francisco 站"。
  第二版（sites2.py）：6 个城市所有参考站 + 每个参考站 3 km 内所有低成本传感器、每个位置的全部 PM2.5 传感器都下载窗口数据；
  A、B、节点都要求窗口内实际有数的小时 ≥ 80%；B 取 3–15 km 内满足要求的最近参考站；节点取 3 km 内满足要求的最近 12 个；节点不跨站重复。
  预注册的 4 条判据不变。
  第二版跑选站后只剩 1 站（仍是 San Francisco）：大多数传感器窗口覆盖 60–80%，曼谷没有一个参考站 ≥ 80%；
  分析只要求各窗口每周 ≥ 48 个"大家同时有数"的小时，80% 远比必要的严 → 覆盖门槛降到 60%（sites2.py 和下面的节点保留规则同时改；只涉及有没有数）。
预注册（11-01；站点和脚本写好、数据下载中、还没看过任何数值）：
  1. 区间覆盖 ≥ 90%；ρ̂ ≥ 1 的格子点预测误差中位 ≤ 0.08（判据同 16 个领域）
  2. 故障普查："偏"（含偏 + 吵）的节点-窗口数 ≥ 2 ×"只变吵"的
  3. 检测：EWMA 越线的节点-窗口里，普查判为有故障（偏或变吵）的 ≥ 70%（精确率；召回率只报告）
  4. 有节点在校准周 → 评估周之间变偏的格子，预测 MAE > 没有的格子（校准期不干净时规律会被拖偏）

用法：~/gut-external/torch215-venv/bin/python aqbench.py
"""
import datetime as dt
import json
import math
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "switching"))
from commonmode import run_lines  # noqa: E402
from hetero import measure  # noqa: E402
from intervals import interval2  # noqa: E402

H = math.log10(1.25)
BIAS, NOISE = math.log10(1.12), 1.5


def load(sid):
    fn = os.path.join(HERE, "cache", f"{sid}.json")
    if not os.path.exists(fn):                               # 下载一直失败的传感器 = 没数据（按 80% 覆盖规则自然剔掉）
        return {}
    return {t: v for t, v in json.load(open(os.path.join(HERE, "cache", f"{sid}.json"))) if v is not None and v > 0}


def mad(xs):
    m = st.median(xs)
    return 1.4826 * st.median(abs(x - m) for x in xs)


def site_rows(s):
    A, B = load(s["A"]["sensor"]), load(s["B"]["sensor"])
    if not A or not B:
        return None, []
    hours = sorted(A)
    N = {n["sensor"]: load(n["sensor"]) for n in s["nodes"]}
    total = len({t for t in hours})
    keep = [sid for sid, d in N.items() if total and sum(t in d for t in hours) >= 0.6 * total]
    if len(keep) < 5:
        return None, keep
    rows = []
    for t in hours:
        if A[t] < 3 or t not in B or not all(t in N[k] for k in keep):
            continue
        tru = math.log10(A[t])
        rows.append((t, math.log10(B[t]) - tru, torch.tensor([math.log10(N[k][t]) - tru for k in keep])))
    return rows, keep


def main():
    sites = json.load(open(os.path.join(HERE, "sites2.json")))
    R, census, det = [], [], []
    for s in sites:
        rows, keep = site_rows(s)
        if rows is None:
            print(f"{s['city']} {s['A']['name'][:28]}：保留节点 {len(keep)} < 5，丢掉")
            continue
        day = lambda t: (dt.date.fromisoformat(t[:10]) - dt.date(2025, 9, 1)).days
        for w in range(4):
            cal = [(False, p, r, 2 * H) for t, p, r in rows if 14 * w <= day(t) < 14 * w + 7]
            ev = [(True, p, r, 2 * H) for t, p, r in rows if 14 * w + 7 <= day(t) < 14 * w + 14]
            if len(cal) < 48 or len(ev) < 48:
                continue
            Rc, Re = torch.stack([r for _, _, r, _ in cal]), torch.stack([r for _, _, r, _ in ev])
            flags = []
            for j in range(len(keep)):
                sh = Re[:, j].median().item() - Rc[:, j].median().item()
                ratio = mad(Re[:, j].tolist()) / max(mad(Rc[:, j].tolist()), 1e-6)
                kind = ("偏 + 吵" if ratio > NOISE else "偏") if abs(sh) > BIAS else ("只变吵" if ratio > NOISE else "正常")
                flags.append(kind)
                census.append(kind)
            nodes = [f"n{j}" for j in range(len(keep))]
            c5 = [(f, p, r, x, k) for k, (f, p, r, x) in enumerate(cal + ev)]
            first, _ = run_lines(cal, c5, nodes, "cm2")
            for nm, k in first.items():
                if k >= len(cal):
                    det.append(flags[nodes.index(nm)] != "正常")
            for judge in ("软", "硬2"):
                pt, lo, hi, s0, rh, bw = interval2(cal, ev, judge, B=30)
                act = measure(cal + ev, 2 * H, judge, s0)["miss"]
                R.append({"site": f"{s['city']}:{s['A']['name'][:24]}", "w": w, "judge": judge, "n": len(keep), "items": len(ev), "act": act, "pt": pt,
                          "lo": lo, "hi": hi, "rho": rh, "drift": any(f in ("偏", "偏 + 吵") for f in flags)})
            print(f"{s['city']:12s} {s['A']['name'][:24]:24s} 窗{w} 节点 {len(keep)} 评估 {len(ev):3d} 小时  软 实测 {R[-2]['act']:.2f} 点 {R[-2]['pt']:.2f} "
                  f"[{R[-2]['lo']:.2f}, {R[-2]['hi']:.2f}] ρ̂ {R[-2]['rho']:.2f} | 硬2 实测 {R[-1]['act']:.2f} 点 {R[-1]['pt']:.2f}  普查 {'/'.join(flags)}", flush=True)
    json.dump({"cells": R, "census": census, "detect": det}, open(os.path.join(HERE, "aqbench_results.json"), "w"), indent=1, ensure_ascii=False)
    cov = st.mean(r["lo"] <= r["act"] <= r["hi"] for r in R)
    pe = [abs(r["act"] - r["pt"]) for r in R if r["rho"] >= 1]
    print(f"\n{len(R)} 格")
    print("预注册检查：")
    print(f"1. 区间覆盖 {cov:.0%}（≥ 90%）；ρ̂ ≥ 1 的 {len(pe)} 格点误差中位 {st.median(pe) if pe else float('nan'):.3f}（≤ 0.08）；全部格点误差中位 {st.median(abs(r['act'] - r['pt']) for r in R):.3f}")
    cnt = {k: census.count(k) for k in ("正常", "偏", "偏 + 吵", "只变吵")}
    print(f"2. 故障普查 {cnt}：偏（含偏 + 吵）{cnt['偏'] + cnt['偏 + 吵']} vs 只变吵 {cnt['只变吵']}（须 ≥ 2 倍）")
    print(f"3. EWMA 越线 {len(det)} 次，其中普查判为有故障 {st.mean(det) if det else float('nan'):.0%}（≥ 70%）；"
          f"召回：有故障的节点-窗口 {len(census) - cnt['正常']} 个")
    d1 = [abs(r["act"] - r["pt"]) for r in R if r["drift"]]
    d0 = [abs(r["act"] - r["pt"]) for r in R if not r["drift"]]
    print(f"4. 有节点变偏的格子 MAE {st.mean(d1) if d1 else float('nan'):.3f}（{len(d1)} 格） vs 没有的 {st.mean(d0) if d0 else float('nan'):.3f}（{len(d0)} 格）")


if __name__ == "__main__":
    main()
