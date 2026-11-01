"""前提自检 + 前提 C 的工程版（勤校准）（2026-11-01，库伯：先做自检，再做 C 的工程版）。

上一轮（aqbench.py，c1d74cb）：低成本传感器随天气一起漂 ~30%、同号 100% → 没有诚实团、校准周学到的关系到评估周已不成立；
  规律没察觉前提不成立就给了点预测（覆盖 25%、误差 0.33）。

一、前提自检（不看真值）：每个节点相对独立秤的偏移 o_j = r_j − p，按块（OpenAQ = 天；LLM 数据 = 题）取中位数，
  shift_j = 评估期块中位数的中位 − 校准期的；se_j = 1.2533 × √(MAD_校准² / 块数_校准 + MAD_评估² / 块数_评估)（MAD 按块中位数算）。
  节点"挪了" = |shift_j| > 2·se_j 且 |shift_j| > h / 2（显著、而且大到和伤害有关）。
  全部节点都挪了 = 没有留在原地可当诚实团的节点 → 拒绝点预测，报"前提 B / C 不成立"。
二、前提 C 的工程版（OpenAQ）：不再一周校准一次——每天拿前 2 天当校准期，并把每个节点减掉它在这 2 天里相对秤的中位偏移
  （低成本传感器的常规做法：对着参考仪器勤校准）。节点由此继承秤的准度；评估 = 当天 24 小时。

数据：OpenAQ 新时段 2025-06-02 → 07-28（8 周；预注册用、没看过）；选站规则 = sites2.py（窗口内实际覆盖 ≥ 60%），参数化到新缓存。
  9–10 月（已看过）的数据只作辅助报告。

预注册（11-01；新时段数据下载中、没看数值）：
  S1 回归（17 套 LLM 真实数据 × R0 × δ > 0 × 软 / 硬2 = 106 格，前提成立）：自检拒答 ≤ 5%
  S2 OpenAQ 新时段、周校准（同 aqbench 的方案）：自检拒答 ≥ 70%；拒答格的点误差中位 > 0.15（拒得有理）；没拒答的格点误差中位 ≤ 0.10
  C1 OpenAQ 新时段、勤校准：裁判被骗率中位比周校准方案低 ≥ 0.2
  C2 勤校准下：自检拒答 ≤ 30%；没拒答的格子区间覆盖 ≥ 90%、点误差中位 ≤ 0.08

用法：PERIOD=jun ~/gut-external/torch215-venv/bin/python selfcheck.py s1 | openaq
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
from hetero import measure  # noqa: E402
from intervals import interval2  # noqa: E402

H = math.log10(1.25)
PERIOD = os.environ.get("PERIOD", "jun")
START, CACHE = {"sep": (dt.date(2025, 9, 1), "cache"), "jun": (dt.date(2025, 6, 2), "cache_jun")}[PERIOD]


def mad(xs):
    m = st.median(xs)
    return 1.4826 * st.median(abs(x - m) for x in xs)


def premise_check(cal, ev, bc, be, h):
    """cal / ev = [(f, p, r, x)]，bc / be = 每行的块编号。返回 (是否拒答, 挪了的节点数, 节点数)。"""
    n = len(cal[0][2])
    moved = 0
    for j in range(n):
        def blocks(rows, b):
            d = {}
            for (_, p, r, _), k in zip(rows, b):
                d.setdefault(k, []).append(r[j].item() - p)
            return [st.median(v) for v in d.values()]
        mc, me = blocks(cal, bc), blocks(ev, be)
        if len(mc) < 2 or len(me) < 2:
            continue
        sh = st.median(me) - st.median(mc)
        se = 1.2533 * math.sqrt(mad(mc) ** 2 / len(mc) + mad(me) ** 2 / len(me))
        if abs(sh) > 2 * se and abs(sh) > h / 2:
            moved += 1
    return moved == n, moved, n


def s1():
    cwd = os.getcwd()
    from invert import sets
    S = sets()
    os.chdir(cwd)
    ref = []
    for name, data, nodes, always, sleeper, deltas in S:
        clean = [(f, p, r, x) for f, p, r, x, _ in data[("R0", 0.0)]]
        for d in deltas[1:]:
            bad = [(f, p, r, x) for f, p, r, x, _ in data[("R0", d)]]
            refuse, mv, n = premise_check(clean, bad, list(range(len(clean))), list(range(len(bad))), math.log10(1 + d))
            ref += [refuse, refuse]                                  # 两种裁判的格子用同一个自检
            if refuse:
                print(f"  拒答：{name} δ={d}（挪了 {mv}/{n}）")
    print(f"S1 回归：{len(ref)} 格，自检拒答 {st.mean(ref):.1%}（≤ 5%）")


def load(sid):
    fn = os.path.join(HERE, CACHE, f"{sid}.json")
    if not os.path.exists(fn):
        return {}
    return {t: v for t, v in json.load(open(fn)) if v is not None and v > 0}


def site_rows(s):
    A, B = load(s["A"]["sensor"]), load(s["B"]["sensor"])
    if not A or not B:
        return None, []
    hours = sorted(A)
    N = {n["sensor"]: load(n["sensor"]) for n in s["nodes"]}
    keep = [sid for sid, d in N.items() if sum(t in d for t in hours) >= 0.6 * len(hours)]
    if len(keep) < 5:
        return None, keep
    rows = []
    for t in hours:
        if A[t] < 3 or t not in B or not all(t in N[k] for k in keep):
            continue
        tru = math.log10(A[t])
        rows.append(((dt.date.fromisoformat(t[:10]) - START).days, math.log10(B[t]) - tru, torch.tensor([math.log10(N[k][t]) - tru for k in keep])))
    return rows, keep


def cell(cal, ev, bc, be):
    refuse, mv, n = premise_check(cal, ev, bc, be, H)
    out = []
    for judge in ("软", "硬2"):
        pt, lo, hi, s0, rh, bw = interval2(cal, ev, judge, B=30)
        act = measure(cal + ev, 2 * H, judge, s0)["miss"]
        out.append({"judge": judge, "act": act, "pt": pt, "lo": lo, "hi": hi, "rho": rh, "refuse": refuse, "moved": mv, "n": n})
    return out


def openaq():
    sites = json.load(open(os.path.join(HERE, f"sites2_{PERIOD}.json" if PERIOD != "sep" else "sites2.json")))
    weekly, daily = [], []
    for s in sites:
        rows, keep = site_rows(s)
        if rows is None:
            print(f"{s['city']} {s['A']['name'][:28]}：保留节点 {len(keep)} < 5，丢掉")
            continue
        for w in range(4):                                           # 周校准（同 aqbench）
            cal = [(False, p, r, 2 * H) for d, p, r in rows if 14 * w <= d < 14 * w + 7]
            ev = [(True, p, r, 2 * H) for d, p, r in rows if 14 * w + 7 <= d < 14 * w + 14]
            if len(cal) < 48 or len(ev) < 48:
                continue
            bc = [d for d, _, _ in rows if 14 * w <= d < 14 * w + 7]
            be = [d for d, _, _ in rows if 14 * w + 7 <= d < 14 * w + 14]
            for c in cell(cal, ev, bc, be):
                weekly.append({"site": s["A"]["name"][:24], "w": w, **c})
        for day in range(2, 56):                                     # 勤校准：前 2 天校准 + 减掉节点相对秤的中位偏移
            cr = [(d, p, r) for d, p, r in rows if day - 2 <= d < day]
            er = [(d, p, r) for d, p, r in rows if d == day]
            if len(cr) < 24 or len(er) < 12:
                continue
            off = torch.stack([r - p for _, p, r in cr]).median(0).values
            cal = [(False, p, r - off, 2 * H) for _, p, r in cr]
            ev = [(True, p, r - off, 2 * H) for _, p, r in er]
            for c in cell(cal, ev, [d for d, _, _ in cr], [d for d, _, _ in er]):
                daily.append({"site": s["A"]["name"][:24], "day": day, **c})
        print(f"{s['city']:12s} {s['A']['name'][:24]:24s} 周校准 {sum(x['site'] == s['A']['name'][:24] for x in weekly)} 格，勤校准 {sum(x['site'] == s['A']['name'][:24] for x in daily)} 格", flush=True)
    json.dump({"weekly": weekly, "daily": daily}, open(os.path.join(HERE, f"selfcheck_{PERIOD}.json"), "w"), indent=1, ensure_ascii=False)
    med = lambda v: st.median(v) if v else float("nan")
    for nm, R in (("周校准", weekly), ("勤校准", daily)):
        ref = [r for r in R if r["refuse"]]
        ok = [r for r in R if not r["refuse"]]
        print(f"\n{nm}：{len(R)} 格；实测被骗中位 {med([r['act'] for r in R]):.2f}；自检拒答 {len(ref) / max(len(R), 1):.0%}")
        print(f"  拒答格点误差中位 {med([abs(r['act'] - r['pt']) for r in ref]):.3f}；没拒答格点误差中位 {med([abs(r['act'] - r['pt']) for r in ok]):.3f}，"
              f"区间覆盖 {st.mean(r['lo'] <= r['act'] <= r['hi'] for r in ok) if ok else float('nan'):.0%}")
    print("\n预注册检查（新时段才算数）：")
    wr, dr = weekly, daily
    print(f"S2 周校准：拒答 {st.mean(r['refuse'] for r in wr):.0%}（≥ 70%）；拒答格误差中位 {med([abs(r['act'] - r['pt']) for r in wr if r['refuse']]):.3f}（> 0.15）；"
          f"没拒答格 {med([abs(r['act'] - r['pt']) for r in wr if not r['refuse']]):.3f}（≤ 0.10）")
    print(f"C1 被骗率中位 周校准 {med([r['act'] for r in wr]):.2f} → 勤校准 {med([r['act'] for r in dr]):.2f}（降 ≥ 0.2）")
    ok = [r for r in dr if not r["refuse"]]
    print(f"C2 勤校准：拒答 {st.mean(r['refuse'] for r in dr):.0%}（≤ 30%）；没拒答格覆盖 {st.mean(r['lo'] <= r['act'] <= r['hi'] for r in ok) if ok else float('nan'):.0%}（≥ 90%）、"
          f"点误差中位 {med([abs(r['act'] - r['pt']) for r in ok]):.3f}（≤ 0.08）")


if __name__ == "__main__":
    {"s1": s1, "openaq": openaq}[sys.argv[1]]()
