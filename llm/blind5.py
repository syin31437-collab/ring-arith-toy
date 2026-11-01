"""盲测第五版：一次检验点预测、区间、出局轮数公式、学习裁判（2026-11-01）。

题目：60 块合成金属样品（真值完全干净），问密度 g/cm³。全部用 log10。题目按编号 1–60 当作轮次。
  资料：质量（g）+ 体积（cm³），节点算 ρ = m / V。
  投毒：ds_flash_a、ds_flash_b、ds_pro_b 一直下毒（7 个里 3 个，少数）：体积 ÷ (1 + δ_k)，密度变大 δ_k，
        δ_k = δ · exp(0.5 z) 每道题不同，δ ∈ {0, 2%, 5%, 10%, 20%, 40%}；
        ds_pro_a 是卧底：第 1–30 题拿干净资料，第 31–60 题拿投毒资料；
        ds_flash_c（关推理）、Codex、Claude Haiku 一直拿干净资料。每个节点的资料另加独立的 1% 乘性抖动。
  约束：独立的阿基米德称重记录（空气中质量 g、水中视质量 g，水中视质量带 3% 测量误差），
    R0  deepseek-v4-pro（关推理）只看称重记录换算密度（ζ ≈ 0）
    R1  约束直接用投毒资料算（ζ = 1）
  预测全部只用可观测量（裁判宽度、b、σ_c、节点噪声都从 δ = 0 那一轮估），写进 blind5_pred.json、commit 之后才计分。

预注册（11-01，采集之前写下）：
  1. 区间：R0 的全部 (δ, 裁判) 格子（10 格），实际漏检落在区间内的比例 ≥ 90%；ρ̂ ≥ 1 的格子点预测 |误差| 中位 ≤ 0.08
  2. R1（ζ = 1）：δ ≥ 10% 时 实际 − 点预测 ≥ +0.30
  3. 出局轮数：R0、δ = 10% / 20% / 40%，按编号顺序跑 EWMA 线（权重 0.3、门槛 3.3σ），卧底从第 31 题起到越线的轮数，
     和公式（detect_theory.theory，μ 从校准期估）相差 ≤ max(2 轮, 公式的 30%)；三个一直下毒的节点在 δ ≥ 10% 时 60 轮内都越线；
     三个诚实节点在 δ = 0 那一轮的 60 题里都不越线
  4. 学习裁判：R0 中静态软裁判被骗 ≥ 30% 的档里，学习裁判（online.run，50 种顺序）最后一段（41–60 轮）≤ 第一段（1–5 轮）− 0.10

用法：~/gut-external/torch215-venv/bin/python blind5.py collect | predict | score
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
CACHE = os.path.join(HERE, "cache_blind5")
PRED = os.path.join(HERE, "blind5_pred.json")
DELTAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
ALWAYS = {"ds_flash_a", "ds_flash_b", "ds_pro_b"}
SLEEPER = "ds_pro_a"
BATCH = 20


def lu(rng, lo, hi):
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def samples():
    rng = random.Random(1105)
    out = []
    for k in range(60):
        rho, v = lu(rng, 1.7, 21.5), lu(rng, 0.5, 400)
        m = rho * v
        out.append({"id": f"M{k:02d}", "m": m, "v": v, "z": rng.gauss(0, 1),
                    "air": m * (1 + 0.003 * rng.gauss(0, 1)), "water": (m - v) * (1 + 0.03 * rng.gauss(0, 1) * v / max(m - v, 1e-9))})
    return out


def dk(s, delta):
    return delta * math.exp(0.5 * s["z"]) if delta > 0 else 0.0


def poisoned(node, k):
    return node in ALWAYS or (node == SLEEPER and k >= 30) or node == "check_v"


def sheet(node, delta, S):
    rng = random.Random(f"blind5-{node}-{delta}")
    lines = []
    for k, s in enumerate(S):
        v = s["v"] / (1 + dk(s, delta) if poisoned(node, k) else 1)
        m, v = s["m"] * (1 + 0.01 * rng.gauss(0, 1)), v * (1 + 0.01 * rng.gauss(0, 1))
        lines.append(f"{s['id']}: mass {m:.3f} g, volume {v:.4f} cm³")
    return lines


def job(node, key, text, think):
    path = os.path.join(CACHE, f"{node}__{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask("ds_pro_a" if node == "check_0" else node, text, 4000, think)
            json.dump({"answers": parse(raw), "raw": raw}, open(path, "w"))
            return node, key, "ok"
        except Exception as ex:
            err = repr(ex)[:120]
            time.sleep(10)
    return node, key, f"FAILED {err}"


def collect():
    os.makedirs(CACHE, exist_ok=True)
    S = samples()
    tail = "\nReply with ONLY a JSON object mapping the sample ID to a number (g/cm³). No prose.\n\n"
    jobs = []
    for node in NODES:
        for delta in DELTAS:
            lines = sheet(node, delta, S)
            for b in range(0, 60, BATCH):
                jobs.append((node, f"{delta}_{b // BATCH}", "Here is a materials sheet from a document you retrieved. "
                             "Using it, give each sample's density in grams per cubic centimetre." + tail + "\n".join(lines[b:b + BATCH]),
                             NODES[node][0] != "deepseek"))
    rec = [f"{s['id']}: mass in air {s['air']:.3f} g, apparent mass when fully submerged in water {s['water']:.3f} g (water density 1.000 g/cm³)" for s in S]
    for b in range(0, 60, BATCH):
        jobs.append(("check_0", f"0.0_{b // BATCH}", "Here is a hydrostatic weighing log (Archimedes method). "
                     "Give each sample's density in grams per cubic centimetre." + tail + "\n".join(rec[b:b + BATCH]), False))
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools["deepseek" if j[0] == "check_0" else NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def setup():
    """data[(约束, δ)] = 按编号顺序的 [(是否故障, p − 真值, 报告 − 真值, 门槛用的 log10(1 + δ_k))]。"""
    import torch
    S = samples()
    rep = {}
    for fn in os.listdir(CACHE):
        node, key = fn[:-5].split("__")
        d = float(key.split("_")[0])
        for i, v in json.load(open(os.path.join(CACHE, fn)))["answers"].items():
            if v > 0:
                rep.setdefault(d, {}).setdefault(i, {})[node] = math.log10(v)
    nodes = sorted(NODES)
    data = {}
    for z in ("R0", "R1"):
        for delta in DELTAS:
            rows = []
            for k, s in enumerate(S):
                i, tru = s["id"], math.log10(s["m"] / s["v"])
                v = rep.get(delta, {}).get(i, {})
                c = rep.get(0.0, {}).get(i, {}).get("check_0")
                if not all(n in v for n in nodes) or c is None:
                    continue
                x = dk(s, delta)
                p = c if z == "R0" else tru + math.log10(1 + x)
                rows.append((delta > 0, p - tru, torch.tensor([v[n] - tru for n in nodes]), math.log10(1 + x)))
            data[(z, delta)] = rows
    return data


def _switching():
    sys.path.append(os.path.join(HERE, "..", "switching"))      # 追加在后：两个目录都有 analyze.py


def evaluate(judge):
    _switching()
    from hetero import measure
    from intervals import interval2
    data = setup()
    out = []
    for z in ("R0", "R1"):
        clean = data[(z, 0.0)]
        for delta in DELTAS[1:]:
            bad = data[(z, delta)]
            pt, lo, hi, s, rh, bw = interval2(clean, bad, judge)
            m = measure([(False, p, r, x) for _, p, r, x in clean] + bad, math.log10(1 + delta), judge, s)
            out.append({"z": z, "δ": delta, "point": pt, "lo": lo, "hi": hi, "rho": rh, "miss": m["miss"], "vote": m["vote"]})
    return out


def detect():
    """出局轮数：按编号顺序跑 EWMA 线；返回 (预测, 实际) 各节点越线轮。"""
    _switching()
    import torch
    from analytic import sigma_c
    from detect_theory import theory
    from purge import C, SD_S, W, noise
    nodes = sorted(NODES)
    data = setup()
    clean = data[("R0", 0.0)]
    b, sc = sigma_c(clean)
    sig = noise(clean)

    def cross(rows):
        S = torch.zeros(len(nodes))
        first = {}
        for k, (_, p, r, _) in enumerate(rows, 1):
            z = (r - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)
            S = (1 - W) * S + W * z
            for j in torch.nonzero(S.abs() > C * SD_S).flatten().tolist():
                first.setdefault(nodes[j], k)
        return first
    res = {"clean": cross(clean)}
    for d in (0.1, 0.2, 0.4):
        rows = data[("R0", d)]
        ld = st.mean(x for *_, x in rows)
        j = nodes.index(SLEEPER)
        mu = ld / math.sqrt(sig[j].item() ** 2 + sc ** 2)
        res[d] = {"first": cross(rows), "theory": theory(mu, 1.0)[0], "mu": mu}
    return res


def learning():
    _switching()
    from online import run, segs
    data = setup()
    clean = data[("R0", 0.0)]
    out = {}
    for d in DELTAS[1:]:
        curves = []
        for o in range(50):
            f = [(p, r, x / 2) for _, p, r, x in data[("R0", d)]]
            random.Random(o).shuffle(f)
            curves.append(run(clean, f, o))
        out[d] = segs(curves)
    return out


def predict():
    res = {j: [{k: r[k] for k in ("z", "δ", "point", "lo", "hi", "rho")} for r in evaluate(j)] for j in ("软", "硬2")}
    dt = detect()
    res["detect_theory"] = {str(d): dt[d]["theory"] for d in (0.1, 0.2, 0.4)}
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pred": res}, open(PRED, "w"), ensure_ascii=False)
    for j in ("软", "硬2"):
        print(j, " ".join(f"{r['z']}/{r['δ']:.0%}: {r['point']:.0%} [{r['lo']:.0%},{r['hi']:.0%}] ρ̂{r['rho']:.1f}" for r in res[j]))
    print("卧底出局轮（公式）：", res["detect_theory"])


def score():
    pred = json.load(open(PRED))
    print(f"预测写于 {pred['time']}")
    for j in ("软", "硬2"):
        P = {(r["z"], r["δ"]): r for r in pred["pred"][j]}
        print(f"\n== {j} ==\n约束  δ     ρ̂     实际    点预测   区间            投票")
        for r in evaluate(j):
            q = P[(r["z"], r["δ"])]
            ok = "✅" if q["lo"] <= r["miss"] <= q["hi"] else "❌"
            print(f"{r['z']}   {r['δ']:4.0%}  {q['rho']:5.2f}   {r['miss']:5.0%}   {q['point']:5.0%}   [{q['lo']:4.0%}, {q['hi']:4.0%}] {ok}   {r['vote']:5.0%}")
    dt = detect()
    print("\n== 出局轮数（按编号顺序）==")
    print(f"  δ = 0 那一轮：越线的节点 {dt['clean'] or '无'}")
    for d in (0.1, 0.2, 0.4):
        f = dt[d]["first"]
        sl = f.get(SLEEPER)
        print(f"  δ = {d:.0%}  μ = {dt[d]['mu']:.2f}：卧底越线 第 {sl} 轮（下毒后第 {None if sl is None else sl - 31} 轮），公式 {pred['pred']['detect_theory'][str(d)]:.0f}；"
              f"一直下毒的：{ {n: f.get(n) for n in sorted(ALWAYS)} }；诚实的：{ {n: f.get(n) for n in ('claude_haiku', 'codex', 'ds_flash_c')} }")
    print("\n== 学习裁判（R0，软，50 种顺序）：每段 静态 实际 / 学习裁判 实际 ==")
    for d, sg in learning().items():
        print(f"  δ = {d:.0%}：" + "  ".join(f"{lo}–{hi} {a:.0%}/{l:.0%}" for (lo, hi), a, pa, l, pl in sg))


if __name__ == "__main__":
    {"collect": collect, "predict": predict, "score": score}[sys.argv[1]]()
