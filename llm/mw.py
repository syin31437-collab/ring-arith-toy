"""真实模型当节点 · 第二题：随机化学式的摩尔质量。

真值：IUPAC 标准原子量（常用值）× 原子个数，精确计算，没有数据源出错的问题。
约束：另外单独问每个模型 15 种元素的原子量；约束预测 = Σ 个数 × 各模型原子量的中位数。
节点和调用方式同 collect.py（7 个）；每次问 20 个化学式；缓存在 cache_mw/。

用法：python3 mw.py collect [NODE ...]
      ~/gut-external/torch215-venv/bin/python mw.py analyze      （EVENT 环境变量 = 事件门槛，log10）
"""
import concurrent.futures as cf
import json
import math
import os
import random
import sys

from collect import NODES, ask, parse

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_mw")
AW = {"H": 1.008, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990,
      "Mg": 24.305, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904}
BATCH = 20


def formulas(n=200):
    rng = random.Random(2026)
    out = []
    for k in range(n):
        size = rng.choice([2, 4, 8, 16, 32, 64])          # 规模从小到大
        elems = ["C", "H"] + rng.sample([e for e in AW if e not in ("C", "H")], rng.randint(1, 4))
        counts = {e: max(1, int(rng.expovariate(1 / size))) for e in elems}
        counts["H"] = max(1, counts["H"] * 2)
        f = "".join(f"{e}{c if c > 1 else ''}" for e, c in counts.items())
        out.append((f"F{k:03d}", f, counts))
    return out


def true_mw(counts):
    return sum(AW[e] * c for e, c in counts.items())


def job(node, key, text):
    path = os.path.join(CACHE, f"{node}_{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask(node, text)
            json.dump({"answers": parse(raw), "raw": raw}, open(path, "w"))
            return node, key, "ok"
        except Exception as ex:
            err = repr(ex)[:120]
    return node, key, f"FAILED {err}"


def collect(nodes):
    os.makedirs(CACHE, exist_ok=True)
    F = formulas()
    jobs = []
    for node in nodes:
        els = list(AW)
        random.Random(node).shuffle(els)
        jobs.append((node, "aw", "Give the standard atomic weight of each element below (IUPAC conventional value).\n"
                     "Reply with ONLY a JSON object mapping the element symbol to a number. No prose.\n\n" + "\n".join(els)))
        order = list(F)
        random.Random(node).shuffle(order)
        for b in range(0, len(order), BATCH):
            items = "\n".join(f"{i}: {f}" for i, f, _ in order[b:b + BATCH])
            jobs.append((node, f"mw_{b // BATCH}", "Compute the molar mass (g/mol) of each chemical formula below, using standard atomic weights. "
                         "Do not use any tools or code; work it out yourself.\n"
                         "Reply with ONLY a JSON object mapping the ID to a number. No prose.\n\n" + items))
    pools = {"deepseek": cf.ThreadPoolExecutor(8), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools[NODES[j[0]][0]].submit(job, *j) for j in jobs]
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def load_answers():
    aw, mw = {}, {}
    for fn in os.listdir(CACHE):
        data = json.load(open(os.path.join(CACHE, fn)))["answers"]
        if fn.endswith("_aw.json"):
            aw[fn[:-len("_aw.json")]] = {k.capitalize() if len(k) > 1 else k: v for k, v in data.items()}
        else:
            nd = fn[:fn.index("_mw_")]
            for i, v in data.items():
                mw.setdefault(i, {})[nd] = v
    return aw, mw


def analyze():
    import statistics as st
    import torch
    from analyze import evaluate
    F = {i: (f, c) for i, f, c in formulas()}
    aw, mw = load_answers()
    med_aw = {e: st.median(aw[n][e] for n in aw if e in aw[n]) for e in AW}
    print("原子量：各模型与 IUPAC 的最大相对偏差 " +
          ", ".join(f"{n} {max(abs(aw[n].get(e, float('nan')) / AW[e] - 1) for e in AW):.2%}" for n in sorted(aw)))
    print("\n== 模型 PK：摩尔质量（相对误差） ==")
    for n in sorted({n for v in mw.values() for n in v}):
        e = [abs(v[n] / true_mw(F[i][1]) - 1) for i, v in mw.items() if n in v and i in F and v[n] > 0]
        print(f"{n:14s} 中位 {st.median(e):.3%}   0.1% 内 {sum(x < 0.001 for x in e) / len(e):4.0%}   1% 内 {sum(x < 0.01 for x in e) / len(e):4.0%}   (共 {len(e)})")
    rows = []
    for i, (f, c) in F.items():
        rep = [v for v in mw.get(i, {}).values() if v > 0]
        if len(rep) < 5:
            continue
        d = torch.tensor([math.log10(v) for v in rep])
        pred = math.log10(sum(med_aw[e] * k for e, k in c.items()))
        rows.append({"iso": i, "d": d, "pred": pred, "truth": math.log10(true_mw(c)), "med": d.median().item()})
    evaluate(rows, "化学式")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2:] or list(NODES))
    else:
        analyze()
