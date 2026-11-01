"""严格检验：只在种子 0–2（geo1_*.json）上拟合，在没见过的数据上打分。
  留出 A：原配置 × 新种子 3–5
  留出 B：训练里没出现过的配置（传感器噪声 ×0.6/×1.8，过程噪声 0.075/0.15，偏移 0.1/0.25/0.6/1.2/2.5）× 种子 0–1
模型：同样是"二次曲面 + 交叉项"，只是输入不同（都取 log）：
  ρ            1 个输入
  ρ + κ        2 个输入（κ = δ / 传感器噪声）
  δ + 噪声倍数  2 个原始变量
  δ + 过程噪声  2 个原始变量
  三个原始变量  3 个输入（参数最多）
用法：python geo_heldout.py run A|B PART   → 生成留出数据（分片并行）
      python geo_heldout.py score
"""
import itertools
import json
import math
import sys

import torch

from geometry import DELTAS, PROCS, SCALES, measure, run

SIG = 0.175                                             # 传感器噪声的平均值 × 倍数


def configs(which):
    if which == "A":
        return [(p, s, d, seed) for p in PROCS for s in SCALES for d in DELTAS for seed in (3, 4, 5)]
    return [(p, s, d, seed) for p in (0.075, 0.15) for s in (0.6, 1.8) for d in (0.1, 0.25, 0.6, 1.2, 2.5) for seed in (0, 1)]


def gen(which, part, parts=4):
    torch.set_num_threads(1)
    rows = []
    for k, (p, s, d, seed) in enumerate(configs(which)):
        if k % parts != part:
            continue
        se, pt = measure(run(seed, p, s, d), 1000, 6000, d)
        rows.append({"proc": p, "scale": s, "delta": d, "seed": seed, "sigma_eff": se, "pass": pt})
    json.dump(rows, open(f"geoh_{which}_{part}.json", "w"))


def feats(r, kind):
    lr = math.log10(r["delta"] / r["sigma_eff"])
    lk = math.log10(r["delta"] / (SIG * r["scale"]))
    ld, ls, lp = math.log10(r["delta"]), math.log10(r["scale"]), math.log10(r["proc"])
    x = {"ρ": [lr], "ρ+κ": [lr, lk], "δ+噪声倍数": [ld, ls], "δ+过程噪声": [ld, lp], "三个原始变量": [ld, ls, lp]}[kind]
    out = [1.0] + x + [a * b for a, b in itertools.combinations_with_replacement(x, 2)]
    return out


def fit(rows, kind):
    X = torch.tensor([feats(r, kind) for r in rows], dtype=torch.float64)
    y = torch.tensor([r["pass"] for r in rows], dtype=torch.float64)
    return torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)


def r2(rows, kind, w):
    X = torch.tensor([feats(r, kind) for r in rows], dtype=torch.float64)
    y = torch.tensor([r["pass"] for r in rows], dtype=torch.float64)
    pred = (X @ w).clamp(0, 1)
    return 1 - ((y - pred) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()


def load(prefix):
    import glob
    rows = [r for f in sorted(glob.glob(prefix)) for r in json.load(open(f))]
    return [r for r in rows if r["sigma_eff"] > 0 and not math.isnan(r["pass"])]


def score():
    train = load("geo1_*.json")
    A, B = load("geoh_A_*.json"), load("geoh_B_*.json")
    print(f"训练 {len(train)} 次运行；留出 A（新种子）{len(A)} 次；留出 B（没见过的配置）{len(B)} 次\n")
    print("模型            参数个数   训练 R²   留出A R²   留出B R²")
    for kind in ("ρ", "ρ+κ", "δ+噪声倍数", "δ+过程噪声", "三个原始变量"):
        w = fit(train, kind)
        print(f"{kind:12s} {len(w):6d}   {r2(train, kind, w):8.3f}  {r2(A, kind, w):9.3f}  {r2(B, kind, w):9.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        gen(sys.argv[2], int(sys.argv[3]))
    else:
        score()
