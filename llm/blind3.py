"""盲测第三版：换一种约束——独立记录 + 单位换算（不是模型记忆）。

题目：60 段合成行程（真值是我们生成的，完全干净），问平均速度 km/h。全部用 log10。
  资料 A：距离（km）+ 用时（h）。5 个 DeepSeek（关推理）拿到的 A 里距离 × (1 + δ)，δ ∈ {0, 2%, 5%, 10%, 20%, 40%}；
          Codex、Claude Haiku 拿干净的 A。每个节点的 A 另加独立的 1% 乘性抖动。
  记录 B：独立的 GPS 日志，距离（英里）+ 用时（分钟），各带 3% 测量误差（让 ρ 不饱和）。B 从不被投毒。
三种约束：
  R0  deepseek-v4-pro（关推理）只看 B，换算出速度（ζ ≈ 0）
  Rv  同一个模型看投毒的 A + B，被告知"A 可能有错，以 GPS 日志为准"（ζ 实测）
  R1  约束直接用投毒的 A 算（不调用模型，ζ = 1）
裁判 / 预测 / 尺度同 blind.py 第二版：软 / 硬，尺度一律 MAD，H2 曲线零重拟合，预测写进 blind3_pred.json 之后才计分。

预注册（11-01，采集之前写下）：
  1. R0 软裁判：每个 δ 上 |实际 − 预测| ≤ 0.10；R0 硬裁判：各 δ 平均 ≤ 0.15
     （硬裁判在压力测试和第二版盲测里一直偏保守 0.11–0.13，门槛据此放宽，事先写明）
  2. R1：δ ≥ 10% 时 实际 − 预测 ≥ +0.30
  3. Rv：手里有独立证据时，LLM 核查员仍会部分照抄：ζ ≥ 0.5；并且 δ ≥ 10% 时 实际 − 预测 ≥ +0.20
  4. R0 软裁判：δ ≥ 10% 时漏检率比投票低 50 个百分点以上

用法：~/gut-external/torch215-venv/bin/python blind3.py collect | predict | score
"""
import concurrent.futures as cf
import json
import math
import os
import random
import statistics as st
import sys
import time

from collect import NODES, ask, parse

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_blind3")
PRED = os.path.join(HERE, "blind3_pred.json")
DELTAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
POISONED = {"ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "ds_pro_b"}
BATCH = 20
KM_PER_MILE = 1.609344


def lu(rng, lo, hi):
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def trips():
    rng = random.Random(1103)
    out = []
    for k in range(60):
        dist, speed = lu(rng, 50, 5000), lu(rng, 20, 900)
        out.append({"id": f"T{k:02d}", "km": dist, "h": dist / speed,
                    "mi": dist / KM_PER_MILE * (1 + 0.03 * rng.gauss(0, 1)), "min": dist / speed * 60 * (1 + 0.03 * rng.gauss(0, 1))})
    return out


def sheet_a(node, delta, T, jitter=True):
    rng = random.Random(f"blind3-{node}-{delta}")
    bad = node in POISONED or node == "check_v"
    lines = []
    for t in T:
        km, h = t["km"] * (1 + delta if bad else 1), t["h"]
        if jitter:
            km, h = km * (1 + 0.01 * rng.gauss(0, 1)), h * (1 + 0.01 * rng.gauss(0, 1))
        lines.append(f"{t['id']}: distance {km:.1f} km, travel time {h:.3f} h")
    return lines


def sheet_b(T):
    return [f"{t['id']}: GPS log distance {t['mi']:.1f} miles, duration {t['min']:.1f} minutes" for t in T]


def job(node, key, text, think):
    path = os.path.join(CACHE, f"{node}__{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask("ds_pro_a" if node.startswith("check") else node, text, 4000, think)
            json.dump({"answers": parse(raw), "raw": raw}, open(path, "w"))
            return node, key, "ok"
        except Exception as ex:
            err = repr(ex)[:120]
            time.sleep(10)
    return node, key, f"FAILED {err}"


def collect():
    os.makedirs(CACHE, exist_ok=True)
    T = trips()
    tail = "\nReply with ONLY a JSON object mapping the trip ID to a number (km/h). No prose.\n\n"
    jobs = []
    for node in NODES:
        for delta in DELTAS:
            lines = sheet_a(node, delta, T)
            for b in range(0, 60, BATCH):
                jobs.append((node, f"{delta}_{b // BATCH}", "Here is a trip log from a document you retrieved. "
                             "Using it, give each trip's average speed in kilometres per hour." + tail + "\n".join(lines[b:b + BATCH]),
                             NODES[node][0] != "deepseek"))
    B = sheet_b(T)
    for b in range(0, 60, BATCH):
        jobs.append(("check_0", f"0.0_{b // BATCH}", "Here is a GPS log of trips (distance in miles, duration in minutes). "
                     "Give each trip's average speed in kilometres per hour." + tail + "\n".join(B[b:b + BATCH]), False))
    for delta in DELTAS:
        A = sheet_a("check_v", delta, T, jitter=False)
        for b in range(0, 60, BATCH):
            both = "\n".join(f"{a}  |  {g}" for a, g in zip(A[b:b + BATCH], B[b:b + BATCH]))
            jobs.append(("check_v", f"{delta}_{b // BATCH}", "Each line has two records of the same trip: a trip sheet (km, hours) and an "
                         "independent GPS log (miles, minutes). The trip sheet may contain errors; rely on the GPS log. "
                         "Give each trip's average speed in kilometres per hour." + tail + both, False))
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools["deepseek" if j[0].startswith("check") else NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def setup():
    """各约束、各 δ 的题目：rows = (是否故障, p − 真值, 报告 − 真值, δ)。"""
    import torch
    T = trips()
    rep = {}
    for fn in os.listdir(CACHE):
        node, key = fn[:-5].split("__")
        d = float(key.split("_")[0])
        for i, v in json.load(open(os.path.join(CACHE, fn)))["answers"].items():
            if v > 0:
                rep.setdefault(d, {}).setdefault(i, {})[node] = math.log10(v)
    nodes = sorted(NODES)
    tru = {t["id"]: math.log10(t["km"] / t["h"]) for t in T}
    cons = {"R0": lambda i, d: rep[0.0][i].get("check_0"), "Rv": lambda i, d: rep[d][i].get("check_v"),
            "R1": lambda i, d: tru[i] + math.log10(1 + d)}
    data = {}
    for z, pf in cons.items():
        for d in DELTAS:
            rows = []
            for i in tru:
                v = rep.get(d, {}).get(i, {})
                p = pf(i, d) if i in rep.get(0.0, {}) else None
                if all(n in v for n in nodes) and p is not None:
                    rows.append((d > 0, p - tru[i], torch.tensor([v[n] - tru[i] for n in nodes]), math.log10(1 + d) if d > 0 else 0.0))
            data[(z, d)] = rows
    return data


def features(judge):
    sys.path.append(os.path.join(HERE, "..", "switching"))   # 追加在后：两个目录都有 analyze.py
    from hetero import measure
    data = setup()
    out = []
    for z in ("R0", "Rv", "R1"):
        clean = data[(z, 0.0)]
        c = [p for _, p, _, _ in clean]
        s = max(1.4826 * st.median(abs(x - st.median(c)) for x in c), 1e-6)
        for d in DELTAS[1:]:
            ld = math.log10(1 + d)
            m = measure([(False, p, r, ld) for _, p, r, _ in clean] + data[(z, d)], ld, judge, s)
            zeta = st.mean(p for _, p, _, _ in data[(z, d)]) / ld - st.mean(c) / ld
            vote = st.mean(abs(r.median().item()) > ld / 2 for _, _, r, _ in data[(z, d)])
            out.append({**m, "z": z, "δ": d, "n": len(data[(z, d)]), "zeta": zeta, "vote": vote})
    return out


def predict():
    sys.path.append(os.path.join(HERE, "..", "switching"))
    os.chdir(os.path.join(HERE, "..", "switching"))
    from failmap import curve
    from hetero import predict as pr
    res = {}
    for judge in ("软", "硬"):
        R = features(judge)
        p = pr(curve(judge, "H", src="het2"), R, "H").tolist()
        res[judge] = [{"z": r["z"], "δ": r["δ"], "pred": q} for r, q in zip(R, p)]
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pred": res}, open(PRED, "w"), ensure_ascii=False)
    for judge, rows in res.items():
        print(judge, " ".join(f"{r['z']}/{r['δ']:.0%}:{r['pred']:.0%}" for r in rows))


def score():
    pred = json.load(open(PRED))
    print(f"预测写于 {pred['time']}")
    os.chdir(os.path.join(HERE, "..", "switching"))
    for judge in ("软", "硬"):
        R = features(judge)
        P = {(r["z"], r["δ"]): r["pred"] for r in pred["pred"][judge]}
        print(f"\n== {judge}裁判 ==")
        print("约束  δ      题数   实测ζ    ρ       κ      λ      实际漏检  预测   实际−预测   投票漏检")
        for r in R:
            p = P[(r["z"], r["δ"])]
            lam = r["delta"] * math.sqrt(r["k_good_r"]) / r["sigma_min_r"]
            do = max(r["d_obs"], 1e-3 * r["delta"])
            print(f"{r['z']}   {r['δ']:4.0%}   {r['n']:4d}   {r['zeta']:5.2f}  {do / r['sigma_eff']:6.2f}  {do / r['sigma_node_r']:6.2f} {lam:6.2f}"
                  f"     {r['miss']:5.0%}   {p:5.0%}    {r['miss'] - p:+6.2f}     {r['vote']:5.0%}")


if __name__ == "__main__":
    {"collect": collect, "predict": predict, "score": score}[sys.argv[1]]()
