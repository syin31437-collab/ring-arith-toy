"""从公式推出：故障普遍程度和相关性到什么水平时，A + E（外部源独立 + 够准）不再足以预测系统错误（2026-11-01，库伯）。

推导：只有 A + E 时认不出谁诚实，预测者等于把所有节点都当诚实——真值后验 = 全体节点 × 约束，中心
  m = (Σ 报告 / σ_n² + 约束中心 / σ_c²) / P，P = n / σ_n² + 1 / σ_c²。
  故障题上投毒节点把中心拉偏 = (n · f · δ̄ / σ_n²) / P；和后验自己的宽度 1 / √P 比：
    Q = f · κ · √n · c / √(1 + σ_n² / (n σ_c²)) ≈ f · κ · √n · c（约束不比节点共识准时）
    f = 投毒节点占比，κ = δ / σ_n，c = 相干度（投毒节点同向的程度，1 = 全同向，0 = 方向随机互相抵消）。
  Q ≲ 1：共识几乎没被拉偏，预测者仍知道真值大概在哪 → A + E 够；Q ≫ 1：预测者把毒的位置当真值 → 裁判被骗的题上预测还说"没被骗"。
  加上故障普遍程度 π（有故障的题占比）：只用 A + E 时预测误差 ≈ π · m_f · Φ(Q − 1)，m_f = 故障题上裁判的被骗率（规律自己能算）。
  → A + E 够的边界：π · m_f · Φ(Q − 1) ≤ τ（容忍误差，取 0.05）。
数值检验（玩具，新种子 prev）：7 个节点、节点噪声相同、约束噪声 = 节点噪声的 1–3 倍（约束不比共识准）；
  扫 π ∈ {0.05, 0.1, 0.2, 0.4, 0.7, 1.0}、投毒数 ∈ {1, 2, 3, 4, 5}、相干度 c ∈ {0, 0.5, 1}、κ ∈ {0.5, 1, 2, 4}，每个组合 1 个配置、400 题。
  "只用 A + E 的预测者" = 推导版第五版，诚实集 = 全部节点、秤和噪声用全部题稳健估计（不知道哪段干净）。
  实际 MAE = |裁判实际被骗率 − 预测|（只在故障题上算 m_f，整体在全部题上算）。

预注册（11-01，运行前）：
  1. 公式 π · m_f · Φ(Q − 1) 和实际 MAE 的相关 ≥ 0.8
  2. 按"≤ 0.05 算够"分类：公式判"够 / 不够"和实际一致的比例 ≥ 85%
  3. c = 0（方向随机）的组合里，不论 π 多大，实际 MAE ≤ 0.07 的比例 ≥ 80%（相干度是关键）

用法：~/gut-external/torch215-venv/bin/python prevalence.py
"""
import itertools
import math
import random
import statistics as st

import torch

from analytic import Phi, derive5, sigma_c
from hetero import judge_out


def run(pi, nbad, c, kappa, seed, n=7, items=400):
    rng = random.Random(f"prev-{seed}")
    g = torch.Generator().manual_seed(seed)
    sn = 1.0
    sc = sn * rng.uniform(1, 3)
    delta = kappa * sn
    rows = []
    for t in range(items):
        fault = rng.random() < pi
        r = sn * torch.randn(n, generator=g)
        if fault:
            for i in range(n - nbad, n):
                sgn = 1 if rng.random() < c + (1 - c) / 2 else -1        # c = 1 全同向；c = 0 方向各自随机
                r[i] += sgn * delta
        p = sc * torch.randn(1, generator=g).item()
        rows.append((fault, p, r, delta))                              # 被骗门槛 h = δ / 2
    est = [(False, p, r, x) for _, p, r, x in rows]                  # 不知道哪段干净：全部题拿来估
    ev = [(True, p, r, x) for _, p, r, x in rows]
    s = max(sigma_c(est)[1], 1e-9)
    out = [abs(judge_out(r, torch.tensor(p), s, "软")) > x / 2 for _, p, r, x in rows]
    act = st.mean(out)
    mf = st.mean(o for o, (f, *_) in zip(out, rows) if f) if any(f for f, *_ in rows) else 0.0
    m0 = st.mean(o for o, (f, *_) in zip(out, rows) if not f) if any(not f for f, *_ in rows) else 0.0
    pred = derive5(est + ev, "软", s, honest=torch.ones(n, dtype=torch.bool))
    f = nbad / n
    Q = f * kappa * math.sqrt(n) * c / math.sqrt(1 + sn ** 2 / (n * sc ** 2))
    formula = pi * max(mf - m0, 0) * Phi(Q - 1)
    return abs(act - pred), formula, c


def main():
    torch.manual_seed(28)
    R = []
    for k, (pi, nbad, c, kappa) in enumerate(itertools.product((0.05, 0.1, 0.2, 0.4, 0.7, 1.0), (1, 2, 3, 4, 5), (0.0, 0.5, 1.0), (0.5, 1, 2, 4))):
        R.append(run(pi, nbad, c, kappa, k))
    a = torch.tensor([x for x, _, _ in R])
    f = torch.tensor([y for _, y, _ in R])
    corr = torch.corrcoef(torch.stack([a, f]))[0, 1].item()
    agree = st.mean((x <= 0.05) == (y <= 0.05) for x, y, _ in R)
    c0 = [x for x, _, c in R if c == 0.0]
    print(f"{len(R)} 个组合")
    print(f"1. 公式与实际 MAE 的相关 {corr:.2f}（≥ 0.8）")
    print(f"2. 判'够 / 不够'一致的比例 {agree:.0%}（≥ 85%）")
    print(f"3. c = 0 时实际 MAE ≤ 0.07 的比例 {st.mean(x <= 0.07 for x in c0):.0%}（≥ 80%）")
    print("\n按 π、相干度看实际 MAE（各 20 个组合平均）与公式：")
    for pi in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        cells = []
        for c in (0.0, 0.5, 1.0):
            sub = [(x, y) for (x, y, cc), (pp, _, c2, _) in zip(R, itertools.product((0.05, 0.1, 0.2, 0.4, 0.7, 1.0), (1, 2, 3, 4, 5), (0.0, 0.5, 1.0), (0.5, 1, 2, 4))) if pp == pi and c2 == c]
            cells.append(f"c={c}: 实际 {st.mean(x for x, _ in sub):.3f} 公式 {st.mean(y for _, y in sub):.3f}")
        print(f"  π = {pi:4.2f}   " + "   ".join(cells))


if __name__ == "__main__":
    main()
