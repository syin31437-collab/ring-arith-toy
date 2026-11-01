"""把 ρ 和 κ 拆开 + 换能表达陡变的曲线形式。

随机设计（代数环境 algebra.run）：每次运行独立地按对数均匀抽
  δ ∈ [0.1, 3]，ab 噪声倍数 ∈ [0.2, 12]，c 噪声倍数 ∈ [0.15, 5]
两种噪声的范围比 δ 宽，ρ（分母 σ_eff 主要随 ab 变）和 κ（分母随 c 变）才能各自变化。
训练 400 次（种子 0–399），留出 200 次（种子 1000–1199，配置也是新抽的）。
拟合方式（对所有模型一样）：
  S形   sigmoid(二次曲面 + 交叉项)，梯度下降拟合
  近邻  特征标准化后，取最近 15 个训练运行的穿透率平均（不假设曲线形状）
反向跨环境：用这里拟合的 ρ+κ 曲线，不调整，预测时间环境 geometry.py 的全部运行。

用法：python decouple.py run JUDGE SET PART    SET = train / test，分 4 片
      python decouple.py score
"""
import glob
import itertools
import json
import math
import random
import sys

import torch

from algebra import SIG, run
from geometry import measure

KINDS = ("ρ", "ρ+κ", "δ+c噪声", "δ+ab噪声", "三个原始变量")


def configs(which):
    seeds = range(400) if which == "train" else range(1000, 1200)
    out = []
    for s in seeds:
        rng = random.Random(s)
        lu = lambda lo, hi: 10 ** rng.uniform(math.log10(lo), math.log10(hi))
        out.append((lu(0.2, 12), lu(0.15, 5), lu(0.1, 3), s))
    return out


def gen(judge, which, part, parts=4):
    torch.set_num_threads(1)
    rows = []
    for k, (a, c, d, seed) in enumerate(configs(which)):
        if k % parts == part:
            se, pt = measure(run(seed, a, c, d, judge), 1000, 5000, d)
            rows.append({"ab": a, "scale": c, "delta": d, "seed": seed, "sigma_eff": se, "pass": pt})
    json.dump(rows, open(f"dec_{judge}_{which}_{part}.json", "w"))


def raw(r, kind):
    lr = math.log10(r["delta"] / r["sigma_eff"])
    lk = math.log10(r["delta"] / (SIG * r["scale"]))
    ld, ls = math.log10(r["delta"]), math.log10(r["scale"])
    la = math.log10(r["ab"]) if "ab" in r else None
    return {"ρ": [lr], "ρ+κ": [lr, lk], "δ+c噪声": [ld, ls], "δ+ab噪声": [ld, la], "三个原始变量": [ld, ls, la]}[kind]


def quad(x):
    return [1.0] + x + [p * q for p, q in itertools.combinations_with_replacement(x, 2)]


def fit_sigmoid(rows, kind):
    X = torch.tensor([quad(raw(r, kind)) for r in rows], dtype=torch.float64)
    y = torch.tensor([r["pass"] for r in rows], dtype=torch.float64).clamp(0, 1)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((torch.sigmoid(X @ w) - y) ** 2).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def pred_sigmoid(w, rows, kind):
    X = torch.tensor([quad(raw(r, kind)) for r in rows], dtype=torch.float64)
    return torch.sigmoid(X @ w)


def pred_knn(train, rows, kind, k=15):
    A = torch.tensor([raw(r, kind) for r in train], dtype=torch.float64)
    mu, sd = A.mean(0), A.std(0)
    A = (A - mu) / sd
    B = (torch.tensor([raw(r, kind) for r in rows], dtype=torch.float64) - mu) / sd
    y = torch.tensor([r["pass"] for r in train], dtype=torch.float64)
    idx = torch.cdist(B, A).topk(k, largest=False).indices
    return y[idx].mean(1)


def r2(rows, pred):
    y = torch.tensor([r["pass"] for r in rows], dtype=torch.float64)
    return 1 - ((y - pred) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()


def load(pattern):
    rows = [r for f in sorted(glob.glob(pattern)) for r in json.load(open(f))]
    return [r for r in rows if r["sigma_eff"] > 0 and not math.isnan(r["pass"])]


def corr(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / math.sqrt(
        sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))


def gen_geo_hard(part, parts=4):
    """时间环境 + 硬裁判（原网格 × 种子 0–2），给硬裁判的跨环境检验用。"""
    from geometry import DELTAS, PROCS, SCALES, run as grun
    torch.set_num_threads(1)
    rows = []
    cfg = [(p, s, d, seed) for p in PROCS for s in SCALES for d in DELTAS for seed in (0, 1, 2)]
    for k, (p, s, d, seed) in enumerate(cfg):
        if k % parts == part:
            se, pt = measure(grun(seed, p, s, d, judge="硬"), 1000, 6000, d)
            rows.append({"proc": p, "scale": s, "delta": d, "seed": seed, "sigma_eff": se, "pass": pt})
    json.dump(rows, open(f"geohard_{part}.json", "w"))


def score():
    geos = {"软": load("geo1_*.json") + load("geoh_*.json"), "硬": load("geohard_*.json")}
    for judge in ("软", "硬"):
        tr, te = load(f"dec_{judge}_train_*.json"), load(f"dec_{judge}_test_*.json")
        c = corr([raw(r, "ρ+κ")[0] for r in tr], [raw(r, "ρ+κ")[1] for r in tr])
        print(f"\n== {judge}裁判：训练 {len(tr)} / 留出 {len(te)}（全是新配置）；log ρ 与 log κ 相关 {c:.2f} ==")
        print("模型            S形 训练   S形 留出   近邻 留出")
        for kind in KINDS:
            w = fit_sigmoid(tr, kind)
            print(f"{kind:12s} {r2(tr, pred_sigmoid(w, tr, kind)):9.3f} {r2(te, pred_sigmoid(w, te, kind)):9.3f}"
                  f" {r2(te, pred_knn(tr, te, kind)):9.3f}")
        w = fit_sigmoid(tr, "ρ+κ")
        geo = geos[judge]
        print(f"反向跨环境：这里拟合的 ρ+κ S形曲线 → 时间环境（同为{judge}裁判）{len(geo)} 次运行，R² = {r2(geo, pred_sigmoid(w, geo, 'ρ+κ')):.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        gen(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "geohard":
        gen_geo_hard(int(sys.argv[2]))
    else:
        score()
