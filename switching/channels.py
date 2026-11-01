"""A+E 边界：分通道推导（2026-11-01，库伯：别打地鼠——先拆通道，各自推、各自验，再合）。

诊断（scratchpad chan.py，种子 chan，非预注册）：A+E 预测误差 = 两条通道，由约束强度分管：
  约束强 → 几乎全是秤偏差 b̂ 被污染共识拖偏（b 换真值 0.118 → 0.025）；
  约束弱 → 估计全换真值仍剩 0.245，来自"默认全部节点诚实"（毒节点也乘进后验）。
推导（零拟合参数，高斯近似）：P = n/σ_n² + 1/σ_c²，w_n = (n/σ_n²)/P，w_c = 1 − w_n，sd = 1/√P，门槛 t = δ/2。
  后验均值 m = w_n·节点均值 + w_c·(p − b̂)；软裁判（加权中位数）近似 o = w_n·节点中位数 + w_c·p。
  故障题上 k 个毒向上、nbad − k 个向下（k ~ 二项(nbad, c + (1 − c)/2)）：均值偏 A_k = δ(2k − nbad)/n，中位数偏 M_k（对高斯做数值积分）。
  秤偏差被拖：Δb = 混入 π 后节点中位数偏移的中位数（b̂ ≈ −Δb）。
  o − 真值 ~ N(w_n·M_k, w_n²σ_med² + w_c²σ_c²)，σ_med² = (π/2)σ_n²/n
  o − m    ~ N(w_n(M_k − A_k) − w_c·Δb, w_n²(π/2 − 1)σ_n²/n)
  实际被骗 act = tail(o − 真值)；A+E 预测 pred = tail(o − m 的均值, √(方差 + sd²))；tail(μ, σ) = P(|N(μ, σ²)| > t)
  误差按"干净题上的残差"扣基线（近似本身在无毒时的偏差不算通道）。
  校准通道 T_cal = |π·E_k[pred_f(Δb) − pred_f(0)] + (1 − π)·[pred_0(Δb) − pred_0(0)]|
  共识通道 T_cons = |π·E_k[(act_f − pred_f(0)) − (act_0 − pred_0(0))]|
  合并 E = |π·E_k[act_f − pred_f(Δb)] + (1 − π)·[act_0 − pred_0(Δb)] − (act_0 − pred_0(0))|
对照（玩具里有真值）：校准通道实测 = |pred(b̂) − pred(0)|（σ_c、节点精度用真值，仍假设全诚实）；
  共识通道实测 = |act − pred(0)|（三项都真值）；合并实测 = |act − pred(全估)|（即 prevalence 的"实际"）。

预注册（11-01，运行前；新种子 chan2；网格同 prevalence 第三稿：σ_c/σ_n × π × 投毒数 × c × κ = 900）：
  1. 校准通道：公式与实测相关 ≥ 0.8
  2. 共识通道：公式与实测相关 ≥ 0.8
  3. 合并：相关 ≥ 0.8，判"够 / 不够"（≤ 0.05）一致 ≥ 85%
停止规则：3 不过 → 承认 A+E 边界没有简单闭式，改用仿真查"够不够"，转去做"不靠 LLM 的秤"。

结果（6f69c28，0 条对）：校准通道相关 0.71、共识通道 0.74、合并 0.68 / 一致 79%。→ 按停止规则：A+E 边界没有简单闭式，不再补稿。
  站得住的只有定性结构：两条通道、由约束强度分管（强约束校准通道 0.17、共识 0.006；弱约束校准 0、共识 0.23），公式也抓到了这个换挡；
  量级抓不住：软裁判 ≈ "w_n·中位数 + w_c·p" 的线性近似在中等约束（σ_c/σ_n = 1–3）下把共识通道高估 2–4 倍——加权中位数是非线性的，这一步近似就是瓶颈。
  实际用法：要判断"A+E 够不够"，用已知参数做仿真（channels.measured 那套），不靠公式。

用法：~/gut-external/torch215-venv/bin/python channels.py
"""
import itertools
import math
import random
import statistics as st

import torch

from analytic import Phi, sigma_c
from hetero import judge_out


def tail(mu, sig, t):
    return (1 - Phi((t - mu) / sig)) + Phi((-t - mu) / sig)


def med_k(nbad, k, delta, n=7, draws=4000):
    g = torch.Generator().manual_seed(11)
    r = torch.randn(draws, n, generator=g)
    rs = r.clone()
    rs[:, n - nbad:n - nbad + k] += delta
    rs[:, n - nbad + k:] -= delta
    return (rs.median(1).values - r.median(1).values).mean().item()


def delta_b(pi, nbad, c, delta, n=7, draws=4000):
    g = torch.Generator().manual_seed(12)
    r = torch.randn(draws, n, generator=g)
    sgn = torch.where(torch.rand(draws, nbad, generator=g) < c + (1 - c) / 2, 1.0, -1.0)
    rf = r.clone()
    rf[:, n - nbad:] += sgn * delta
    fault = torch.rand(draws, generator=g) < pi
    return torch.where(fault, rf.median(1).values, r.median(1).values).median().item()


def formula(pi, nbad, c, delta, sc, n=7, sn=1.0):
    P = n / sn ** 2 + 1 / sc ** 2
    wn = (n / sn ** 2) / P
    wc, sd, t = 1 - wn, 1 / math.sqrt(P), delta / 2
    s_act = math.sqrt(wn ** 2 * (math.pi / 2) * sn ** 2 / n + wc ** 2 * sc ** 2)
    s_d = math.sqrt(wn ** 2 * (math.pi / 2 - 1) * sn ** 2 / n + sd ** 2)
    db = delta_b(pi, nbad, c, delta, n)
    q = c + (1 - c) / 2
    act0, pr0, pr0b = tail(0, s_act, t), tail(0, s_d, t), tail(-wc * db, s_d, t)
    Ef = {"act": 0.0, "pr": 0.0, "prb": 0.0}
    for k in range(nbad + 1):
        pk = math.comb(nbad, k) * q ** k * (1 - q) ** (nbad - k)
        M, A = med_k(nbad, k, delta, n), delta * (2 * k - nbad) / n
        Ef["act"] += pk * tail(wn * M, s_act, t)
        Ef["pr"] += pk * tail(wn * (M - A), s_d, t)
        Ef["prb"] += pk * tail(wn * (M - A) - wc * db, s_d, t)
    cal = abs(pi * (Ef["prb"] - Ef["pr"]) + (1 - pi) * (pr0b - pr0))
    cons = abs(pi * ((Ef["act"] - Ef["pr"]) - (act0 - pr0)))
    full = abs(pi * (Ef["act"] - Ef["prb"]) + (1 - pi) * (act0 - pr0b) - (act0 - pr0))
    return cal, cons, full


def pred(rows, s, b, sc, prec):
    out = []
    for _, p, r, dk in rows:
        t, o, c0 = dk / 2, judge_out(r, torch.tensor(p), s, "软"), p - b
        P = prec.sum() + 1 / sc ** 2
        m = ((prec * r).sum() + c0 / sc ** 2) / P
        sd = 1 / math.sqrt(P)
        out.append(1 - (Phi((t - (o - m)) / sd) - Phi((-t - (o - m)) / sd)))
    return st.mean(out)


def measured(pi, nbad, c, kappa, ra, k, n=7, sn=1.0, items=400):
    rng = random.Random(f"chan2-{k}")
    g = torch.Generator().manual_seed(600000 + k)
    delta, rows = kappa * sn, []
    for _ in range(items):
        fault = rng.random() < pi
        r = sn * torch.randn(n, generator=g)
        if fault:
            for i in range(n - nbad, n):
                r[i] += (1 if rng.random() < c + (1 - c) / 2 else -1) * delta
        rows.append((fault, ra * torch.randn(1, generator=g).item(), r, delta))
    b_hat, sc_hat = sigma_c(rows)
    s = max(sc_hat, 1e-9)
    D = torch.stack([r - r.median() for _, _, r, _ in rows])
    spread = max(st.median(1.4826 * d.abs().median().item() for d in D), 1e-9)
    prec_hat = 1 / (1.4826 * (D - D.median(0).values).abs().median(0).values).clamp(min=0.5 * spread) ** 2
    prec_t = torch.full((n,), 1 / sn ** 2)
    act = st.mean(abs(judge_out(r, torch.tensor(p), s, "软")) > x / 2 for _, p, r, x in rows)
    p_bt, p_b0 = pred(rows, s, b_hat, ra, prec_t), pred(rows, s, 0.0, ra, prec_t)
    return abs(p_bt - p_b0), abs(act - p_b0), abs(act - pred(rows, s, b_hat, sc_hat, prec_hat))


def main():
    grid = list(itertools.product((0.1, 0.3, 1.0, 3.0, 10.0), (0.1, 0.4, 0.7, 1.0), (1, 2, 3, 4, 5), (0.0, 0.5, 1.0), (1, 2, 4)))
    F, Mz = [], []
    for k, (ra, pi, nbad, c, kappa) in enumerate(grid):
        F.append(formula(pi, nbad, c, kappa * 1.0, ra))
        Mz.append(measured(pi, nbad, c, kappa, ra, k))
    print(f"{len(grid)} 个组合（新种子 chan2）")
    for j, nm, bar in ((0, "1. 校准通道", "≥ 0.8"), (1, "2. 共识通道", "≥ 0.8"), (2, "3. 合并", "≥ 0.8")):
        a, f = torch.tensor([m[j] for m in Mz]), torch.tensor([x[j] for x in F])
        corr = torch.corrcoef(torch.stack([a, f]))[0, 1].item()
        extra = ""
        if j == 2:
            extra = f"，判'够 / 不够'一致 {st.mean((m[2] <= 0.05) == (x[2] <= 0.05) for m, x in zip(Mz, F)):.0%}（≥ 85%）"
        print(f"{nm}：相关 {corr:.2f}（{bar}）{extra}；实测均值 {a.mean():.3f}、公式均值 {f.mean():.3f}")
    print("\n按约束强度（c = 1、π ≥ 0.7）：校准 实测/公式   共识 实测/公式   合并 实测/公式")
    for ra in (0.1, 0.3, 1.0, 3.0, 10.0):
        idx = [i for i, (r_, pi, _, c, _) in enumerate(grid) if r_ == ra and c == 1.0 and pi >= 0.7]
        cells = "   ".join(f"{st.mean(Mz[i][j] for i in idx):.3f}/{st.mean(F[i][j] for i in idx):.3f}" for j in range(3))
        print(f"  σ_c/σ_n = {ra:4}：{cells}")


if __name__ == "__main__":
    main()
