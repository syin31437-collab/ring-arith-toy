"""换环境检验两个比值：代数约束、没有时间，外加换一种裁判。

环境：每一步独立地出 a, b ~ U(-2, 2)，真相 c = a·b（步与步之间没有关系）。
      12 个传感器各报 a, b, c：a、b 的噪声 = U(0.05,0.3) × ab噪声倍数，c 的噪声 = U(0.05,0.3) × c噪声倍数。
      第 1000 步起 30% 的步，固定 8 个传感器报的 c 一起偏 δ（a、b 不受影响）。
约束：小 MLP 用 (a 的中位数, b 的中位数) 预测 c，只在"裁判结果 ≈ 普通中位数"的步上拿裁判结果学。
裁判：
  软  按离预测的远近加权（尺度 = 约束近期的预测误差），加权中位数（和 geometry.py 同一种）
  硬  离预测超过 2.5 倍尺度的报告直接剔除，剩下的取中位数；全被剔除就退回普通中位数
      （退回预测值会在时间环境里自我锁死：只跟着自己的预测走，误差到无穷大）
测量同 geometry.py：σ_eff（没故障时 预测 与 c 报告中位数 之差的标准差）、穿透率；κ = δ / (0.175 × c噪声倍数)。

用法：python algebra.py run JUDGE SET PART   SET = train / A / B，分 4 片并行
      python algebra.py score
"""
import glob
import itertools
import json
import math
import sys

import torch
import torch.nn as nn

from geo_heldout import fit as fit_env1, load as load_env1
from geometry import DELTAS, measure
from robust import wmedian

N, BAD, SIG = 12, 8, 0.175
AB = [0.3, 1.5, 6.0]          # 第一版是 [0.05, 0.15, 0.4]，太弱，约束不确定度几乎不随它变（结果存在 alg_weak_ab/）
CS = [0.3, 1.0, 3.0]


def configs(which):
    if which == "train":
        return [(a, c, d, s) for a in AB for c in CS for d in DELTAS for s in (0, 1, 2)]
    if which == "A":
        return [(a, c, d, s) for a in AB for c in CS for d in DELTAS for s in (3, 4, 5)]
    return [(a, c, d, s) for a in (0.8, 3.0) for c in (0.6, 1.8) for d in (0.1, 0.25, 0.6, 1.2, 2.5) for s in (0, 1)]


def run(seed, ab_noise, c_scale, delta, judge, T=5000):
    g = torch.Generator().manual_seed(seed)
    sa = (torch.rand(N, generator=g) * 0.25 + 0.05) * ab_noise
    sc = (torch.rand(N, generator=g) * 0.25 + 0.05) * c_scale
    torch.manual_seed(seed)
    f = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(f.parameters(), lr=3e-3)
    s, log = 0.5, []
    for t in range(T):
        a, b = (torch.rand(2, generator=g) * 4 - 2).tolist()
        c = a * b
        ra = a + sa * torch.randn(N, generator=g)
        rb = b + sa * torch.randn(N, generator=g)
        rc = c + sc * torch.randn(N, generator=g)
        fault = t >= 1000 and torch.rand(1, generator=g).item() < 0.3
        if fault:
            rc[:BAD] += delta
        pred = f(torch.stack([ra.median(), rb.median()])).squeeze()
        pd = pred.detach()
        if judge == "软":
            out = wmedian(rc, torch.exp(-((rc - pd) ** 2) / (2 * s ** 2)) + 1e-6)
        else:
            keep = (rc - pd).abs() <= 2.5 * s
            fallback = rc.median() if judge == "硬" else pd           # "硬·退回预测" = 旧的自我锁死版本（failmap.py 复现用）
            out = rc[keep].median() if keep.any() else fallback   # 全被剔除时退回中位数：约束不能完全取代测量
        plain = rc.median()
        if abs(out.item() - plain.item()) < 0.1:
            tgt = out.item()
            s = max(0.02, (0.98 * s ** 2 + 0.02 * (pd.item() - tgt) ** 2) ** 0.5)
            loss = (pred - tgt) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        log.append((out.item(), c, fault, pd.item(), plain.item(), ((rc - plain) ** 2).mean().sqrt().item(), (rc - pd).tolist()))
    return log


def gen(judge, which, part, parts=4):
    torch.set_num_threads(1)
    rows = []
    for k, (a, c, d, seed) in enumerate(configs(which)):
        if k % parts != part:
            continue
        se, pt = measure(run(seed, a, c, d, judge), 1000, 5000, d)
        rows.append({"ab": a, "scale": c, "delta": d, "seed": seed, "sigma_eff": se, "pass": pt})
    json.dump(rows, open(f"alg_{judge}_{which}_{part}.json", "w"))


def feats(r, kind):
    lr = math.log10(r["delta"] / r["sigma_eff"])
    lk = math.log10(r["delta"] / (SIG * r["scale"]))
    ld, ls, la = math.log10(r["delta"]), math.log10(r["scale"]), math.log10(r["ab"])
    x = {"ρ": [lr], "ρ+κ": [lr, lk], "δ+c噪声": [ld, ls], "δ+ab噪声": [ld, la], "三个原始变量": [ld, ls, la]}[kind]
    return [1.0] + x + [p * q for p, q in itertools.combinations_with_replacement(x, 2)]


def r2(rows, X, w):
    y = torch.tensor([r["pass"] for r in rows], dtype=torch.float64)
    pred = (torch.tensor(X, dtype=torch.float64) @ w).clamp(0, 1)
    return 1 - ((y - pred) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()


def load(judge, which):
    rows = [r for f in sorted(glob.glob(f"alg_{judge}_{which}_*.json")) for r in json.load(open(f))]
    return [r for r in rows if r["sigma_eff"] > 0 and not math.isnan(r["pass"])]


def score():
    for judge in ("软", "硬"):
        tr, A, B = load(judge, "train"), load(judge, "A"), load(judge, "B")
        se = [r["sigma_eff"] for r in tr]
        print(f"\n== 代数环境 · {judge}裁判：训练 {len(tr)} / 新种子 {len(A)} / 没见过的配置 {len(B)}；σ_eff 范围 {min(se):.3f}–{max(se):.3f} ==")
        print("模型            参数   训练 R²   新种子 R²   没见过配置 R²")
        for kind in ("ρ", "ρ+κ", "δ+c噪声", "δ+ab噪声", "三个原始变量"):
            X = torch.tensor([feats(r, kind) for r in tr], dtype=torch.float64)
            y = torch.tensor([r["pass"] for r in tr], dtype=torch.float64)
            w = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
            print(f"{kind:12s} {len(w):4d}  {r2(tr, [feats(r, kind) for r in tr], w):8.3f}"
                  f"  {r2(A, [feats(r, kind) for r in A], w):10.3f}  {r2(B, [feats(r, kind) for r in B], w):12.3f}")

    print("\n== 跨环境：用时间环境（geometry.py）拟合的 ρ+κ 曲线，不做任何调整，直接预测代数环境 ==")
    w1 = fit_env1(load_env1("geo1_*.json"), "ρ+κ")
    for judge in ("软", "硬"):
        rows = load(judge, "train") + load(judge, "A") + load(judge, "B")
        X = [feats(r, "ρ+κ") for r in rows]
        print(f"代数环境 · {judge}裁判（{len(rows)} 次）：R² = {r2(rows, X, w1):.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        gen(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        score()
