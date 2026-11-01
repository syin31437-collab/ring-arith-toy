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

第一稿结果（80a07cf，3 条对 1 条）：结构对——相干度是关键（c = 0 时 98% 的组合 MAE ≤ 0.07），只有"同向抱团 + 故障普遍"才不够
  （c = 1、π ≥ 0.7 时实际 0.086–0.104）；但公式和实际相关只有 0.52、判对 79%，高估（c = 0.5、π = 1：公式 0.172、实际 0.030）。
  错在拿"后验宽度"当标尺；被骗的定义是门槛 h = δ / 2——只用 A + E 时裁判和预测者被同一个有偏的共识带着走，
  预测者只在"共识偏离真值超过 h"的题上说错。
第二稿：故障题上共识中心偏离 μ_B = f · c · δ · w_n（w_n = 节点在后验里的权重份额 = (n / σ_n²) / P），宽度 s = 1 / √P：
  MAE_A+E ≈ π · [P(|N(μ_B, s²)| > h) − P(|N(0, s²)| > h)]，h = δ / 2
  → 边界很直观：f · c · w_n > 1/2（投毒占比 × 相干度超过一半时，共识被拉过门槛）。
第二稿预注册（运行前；新种子 prev2）：
  1. 公式与实际 MAE 的相关 ≥ 0.8
  2. 判"够 / 不够"（≤ 0.05）一致 ≥ 85%
  3. f · c · w_n < 0.4 的组合里实际 MAE ≤ 0.05 的比例 ≥ 90%；> 0.6 且 π ≥ 0.7 的组合里实际 MAE > 0.05 的比例 ≥ 70%

第二稿结果（205f48a，3 条对 2 条）：判对 86% ✅、低区 91% ✅；相关 0.56 ❌、高区 62% ❌（高区公式 0.37–0.52、实际 0.08–0.18，高估）。
库伯：Q 的推导要保留外部约束的强度，要检验 (Q − 1) 那一项。两个缺口：
  ① 第一稿 Q 里有约束强度因子 √(1 + σ_n² / (n σ_c²))，但检验时约束噪声只取节点噪声的 1–3 倍（一直是弱约束）→ 这一项根本没被检验到；
  ② 约束强度还决定裁判会不会被骗：第二稿假设裁判总被共识带走，约束强时裁判跟着约束走、共识拉偏了也不被骗 → 高估。
第三稿：(Q − 1) 推广成 Φ(Q − h / s)——"1"其实是"门槛 h 用后验宽度 s 量"，Q = μ_B / s 是共识被拉偏了几个宽度；第一稿取 1 相当于 h = s。
  MAE_A+E ≈ π · (m_f − m₀) · Φ((μ_B − h) / s)
  约束强度进入三处：w_n（节点在后验里的份额 → μ_B）、s（后验宽度）、m_f（故障题上裁判被骗率；约束强时裁判几乎不被骗）。
第三稿预注册（运行前；新种子 prev3；约束噪声 σ_c / σ_n ∈ {0.1, 0.3, 1, 3, 10}，π ∈ {0.1, 0.4, 0.7, 1.0}，投毒数 1–5，c ∈ {0, 0.5, 1}，κ ∈ {1, 2, 4}）：
  1. 第三稿：公式与实际 MAE 的相关 ≥ 0.8，判"够 / 不够"一致 ≥ 85%
  2. 第三稿的相关高于第一稿、第二稿（同一批组合）
  3. 约束很强（σ_c / σ_n = 0.1）时，c = 1、π = 1 的组合里 A + E 也够（实际 MAE ≤ 0.05）的比例 ≥ 80%

第三稿结果（1f2753d，0 条对）：三稿相关都 ≈ 0.58、一致 77–79%；约束很强时 A + E 够只有 53%——σ_c/σ_n = 0.1 实际 0.153、第三稿 0.000。
诊断（另种子，非预注册）：约束很强、π = 1 时估出的秤偏差 b̂ = −0.76（真值 0）——没有干净校准期，b 是拿被污染的共识校的。
  → 约束强度是双刃：强约束挡住"共识拖偏裁判"，却把"校准拖偏"原样传进后验中心。另外软裁判是加权中位数，受的拉偏是中位数的偏移、不是均值 f·c·δ（第二、三稿高估的来源）。
第四稿（保留 Φ 与约束强度权重，两条通道都进）：
  w_n = (n/σ_n²)/P、w_c = (1/σ_c²)/P、s = 1/√P、h = δ/2；
  M_f = 故障题上 7 个节点中位数的平均偏移（按参数蒙特卡洛，不看数据）；Δb = 全部题上节点中位数偏移的中位数（同上，混入 π）；
  故障题后验中心偏 e_f = w_n·M_f + w_c·Δb，干净题 e_0 = w_c·Δb；T(e) = |Φ((e − h)/s) + Φ((−e − h)/s) − 2Φ(−h/s)|
  MAE_A+E ≈ π·T(e_f) + (1 − π)·T(e_0)
第四稿预注册（运行前；新种子 prev4；网格同第三稿）：
  1. 相关 ≥ 0.8，判"够 / 不够"一致 ≥ 85%
  2. 第四稿相关高于前三稿
  3. 约束很强（σ_c/σ_n = 0.1）、c = 1、π ≥ 0.7：公式平均与实际平均之差 ≤ 0.05（第三稿差 0.15）

用法：~/gut-external/torch215-venv/bin/python prevalence.py        （第一稿）
      ~/gut-external/torch215-venv/bin/python prevalence.py v2     （第二稿）
      ~/gut-external/torch215-venv/bin/python prevalence.py v3     （第三稿：保留约束强度，Φ(Q − h/s)）
      ~/gut-external/torch215-venv/bin/python prevalence.py v4     （第四稿：共识拖偏 + 校准拖偏两条通道）
"""
import itertools
import math
import random
import statistics as st

import torch

from analytic import Phi, derive5, sigma_c
from hetero import judge_out


def med_shift(pi, nbad, c, delta, n=7, sn=1.0, draws=4000):
    """按参数蒙特卡洛（不看数据）：故障题节点中位数的平均偏移 M_f；全部题（混入 π）节点中位数偏移的中位数 Δb。"""
    g = torch.Generator().manual_seed(7)
    r = sn * torch.randn(draws, n, generator=g)
    sgn = torch.where(torch.rand(draws, nbad, generator=g) < c + (1 - c) / 2, 1.0, -1.0)
    rf = r.clone()
    rf[:, n - nbad:] += sgn * delta
    mf = rf.median(1).values - r.median(1).values
    fault = torch.rand(draws, generator=g) < pi
    mix = torch.where(fault, rf.median(1).values, r.median(1).values)
    return mf.mean().item(), mix.median().item()


def run(pi, nbad, c, kappa, seed, n=7, items=400, v2=False, ratio=None, v4=False):
    tag = 'prev4' if v4 else ('prev3' if ratio else ('prev2' if v2 else 'prev'))
    rng = random.Random(f"{tag}-{seed}")
    g = torch.Generator().manual_seed(seed + (300000 if v4 else (200000 if ratio else (100000 if v2 else 0))))
    sn = 1.0
    sc = sn * (ratio if ratio else rng.uniform(1, 3))
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
    if ratio:
        P = n / sn ** 2 + 1 / sc ** 2
        wn = (n / sn ** 2) / P
        mu_b, s_ = f * c * delta * wn, 1 / math.sqrt(P)
        h = delta / 2
        tail = lambda mu: (1 - Phi((h - mu) / s_)) + Phi((-h - mu) / s_)
        v1 = pi * max(mf - m0, 0) * Phi(Q - 1)
        v2f = pi * max(tail(mu_b) - tail(0.0), 0.0)
        v3 = pi * max(mf - m0, 0) * Phi((mu_b - h) / s_)
        if not v4:
            return abs(act - pred), v1, v2f, v3, c, pi, ratio
        wc = (1 / sc ** 2) / P
        Mf, db = med_shift(pi, nbad, c, delta, n, sn)
        T = lambda e: abs(Phi((e - h) / s_) + Phi((-e - h) / s_) - 2 * Phi(-h / s_))
        v4f = pi * T(wn * Mf + wc * db) + (1 - pi) * T(wc * db)
        return abs(act - pred), v1, v2f, v3, c, pi, ratio, v4f
    if v2:
        P = n / sn ** 2 + 1 / sc ** 2
        wn = (n / sn ** 2) / P
        mu_b, s_ = f * c * delta * wn, 1 / math.sqrt(P)
        h = delta / 2
        tail = lambda mu: (1 - Phi((h - mu) / s_)) + Phi((-h - mu) / s_)
        formula = pi * max(tail(mu_b) - tail(0.0), 0.0)
        return abs(act - pred), formula, c, f * c * wn, pi
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


def main2():
    torch.manual_seed(29)
    grid = list(itertools.product((0.05, 0.1, 0.2, 0.4, 0.7, 1.0), (1, 2, 3, 4, 5), (0.0, 0.5, 1.0), (0.5, 1, 2, 4)))
    R = [run(pi, nbad, c, kappa, k, v2=True) for k, (pi, nbad, c, kappa) in enumerate(grid)]
    a = torch.tensor([x[0] for x in R])
    f = torch.tensor([x[1] for x in R])
    print(f"{len(R)} 个组合（新种子 prev2）")
    print(f"1. 公式与实际 MAE 的相关 {torch.corrcoef(torch.stack([a, f]))[0, 1].item():.2f}（≥ 0.8）")
    print(f"2. 判'够 / 不够'一致 {st.mean((x[0] <= 0.05) == (x[1] <= 0.05) for x in R):.0%}（≥ 85%）")
    lo = [x[0] for x in R if x[3] < 0.4]
    hi = [x[0] for x in R if x[3] > 0.6 and x[4] >= 0.7]
    print(f"3. f·c·w_n < 0.4：实际 ≤ 0.05 的比例 {st.mean(v <= 0.05 for v in lo):.0%}（{len(lo)} 个，≥ 90%）；"
          f"> 0.6 且 π ≥ 0.7：实际 > 0.05 的比例 {st.mean(v > 0.05 for v in hi):.0%}（{len(hi)} 个，≥ 70%）")
    print("\n按 投毒占比 × 相干度（f·c·w_n）分档：")
    for lo_, hi_ in ((0, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8)):
        sub = [x for x in R if lo_ <= x[3] < hi_ and x[4] >= 0.7]
        if sub:
            print(f"  [{lo_:.1f}, {hi_:.1f})、π ≥ 0.7：实际 MAE {st.mean(x[0] for x in sub):.3f}   公式 {st.mean(x[1] for x in sub):.3f}   （{len(sub)} 个）")


def main3():
    torch.manual_seed(30)
    grid = list(itertools.product((0.1, 0.3, 1.0, 3.0, 10.0), (0.1, 0.4, 0.7, 1.0), (1, 2, 3, 4, 5), (0.0, 0.5, 1.0), (1, 2, 4)))
    R = [run(pi, nbad, c, kappa, k, ratio=ra) for k, (ra, pi, nbad, c, kappa) in enumerate(grid)]
    a = torch.tensor([x[0] for x in R])
    print(f"{len(R)} 个组合（新种子 prev3）")
    for j, nm in ((1, "第一稿 π·(m_f − m₀)·Φ(Q − 1)"), (2, "第二稿 π·[尾(μ_B) − 尾(0)]"), (3, "第三稿 π·(m_f − m₀)·Φ((μ_B − h)/s)")):
        f = torch.tensor([x[j] for x in R])
        corr = torch.corrcoef(torch.stack([a, f]))[0, 1].item()
        agree = st.mean((x[0] <= 0.05) == (x[j] <= 0.05) for x in R)
        print(f"  {nm}：相关 {corr:.2f}，判'够 / 不够'一致 {agree:.0%}")
    strong = [x[0] for x in R if x[6] == 0.1 and x[4] == 1.0 and x[5] == 1.0]
    print(f"3. 约束很强（σ_c/σ_n = 0.1）、c = 1、π = 1：A + E 够的比例 {st.mean(v <= 0.05 for v in strong):.0%}（{len(strong)} 个，≥ 80%）")
    print("\n按约束强度看（c = 1、π ≥ 0.7）：实际 MAE / 第三稿公式")
    for ra in (0.1, 0.3, 1.0, 3.0, 10.0):
        sub = [x for x in R if x[6] == ra and x[4] == 1.0 and x[5] >= 0.7]
        print(f"  σ_c/σ_n = {ra:4}：实际 {st.mean(x[0] for x in sub):.3f}   第三稿 {st.mean(x[3] for x in sub):.3f}   第一稿 {st.mean(x[1] for x in sub):.3f}")


def main4():
    torch.manual_seed(40)
    grid = list(itertools.product((0.1, 0.3, 1.0, 3.0, 10.0), (0.1, 0.4, 0.7, 1.0), (1, 2, 3, 4, 5), (0.0, 0.5, 1.0), (1, 2, 4)))
    R = [run(pi, nbad, c, kappa, k, ratio=ra, v4=True) for k, (ra, pi, nbad, c, kappa) in enumerate(grid)]
    a = torch.tensor([x[0] for x in R])
    print(f"{len(R)} 个组合（新种子 prev4）")
    for j, nm in ((1, "第一稿"), (2, "第二稿"), (3, "第三稿"), (7, "第四稿")):
        f = torch.tensor([x[j] for x in R])
        corr = torch.corrcoef(torch.stack([a, f]))[0, 1].item()
        agree = st.mean((x[0] <= 0.05) == (x[j] <= 0.05) for x in R)
        print(f"  {nm}：相关 {corr:.2f}，判'够 / 不够'一致 {agree:.0%}")
    sub = [x for x in R if x[6] == 0.1 and x[4] == 1.0 and x[5] >= 0.7]
    print(f"3. 约束很强、c = 1、π ≥ 0.7：实际 {st.mean(x[0] for x in sub):.3f}，第四稿 {st.mean(x[7] for x in sub):.3f}（差 ≤ 0.05）")
    print("\n按约束强度（c = 1、π ≥ 0.7）：实际 / 第四稿")
    for ra in (0.1, 0.3, 1.0, 3.0, 10.0):
        s2 = [x for x in R if x[6] == ra and x[4] == 1.0 and x[5] >= 0.7]
        print(f"  σ_c/σ_n = {ra:4}：{st.mean(x[0] for x in s2):.3f} / {st.mean(x[7] for x in s2):.3f}")


if __name__ == "__main__":
    import sys
    {"v2": main2, "v3": main3, "v4": main4}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
