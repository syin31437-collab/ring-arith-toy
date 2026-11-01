"""规律 v6（2026-11-01）：把 derive5 的三处原理问题修掉后的版本。LAW_VERSION 写进每次结果，避免分不清验证的是哪个算法。
（和 analytic.derive6 无关：那是早期的另一个试验版本。）

相对 derive5 的改动：
  1. 诚实度 = 真正的后验概率（不再把 exp(−½z²) 当概率）：
     x̄_i = 评估期 r − (p − b) 的中位数；v_i = (1.2533·MAD_i)² / n_评估 + se_b²（统一分母）；
     诚实：x̄ ~ N(0, v)；故障：x̄ ~ N(0, v + τ²)，τ = 伤害尺度（每题 δ_k 的中位数）；先验 ½。数据不够时 v ≫ τ²，后验趋于 ½。
  2. 节点噪声 = 留一法（节点 − 其他节点中位数 的 MAD，除以 √(1 + π / (2(n − 1))) 扣掉参照自身的噪声），
     校准期和评估期取较大值（覆盖"不偏只变吵"）；下限同 derive5（0.5 × 典型节点散布）。
  3. 秤方差 = σ_c² + se_b²（b 的不确定度进后验）。
  4. 后验宽度自检（不看真值）：干净题上，节点 i 对"其余节点的精度加权平均"的标准化留一残差，MAD = λ_n；
     秤对"全部节点精度加权平均"的标准化残差，MAD = λ_c；λ > 1 时方差按 λ² 放大（λ < 1 不收窄）。
  5. 区间：题成簇时（同一轮 / 同一天）整块重采样，二项项样本量用块数（开发数据：联邦学习覆盖 44% → 94%，宽度中位 0.14 → 0.41）。
  6. 前提自检 premise_check（从 aq/selfcheck.py 搬来，定义不变）：全部节点都相对秤挪动（显著 > 2 se 且 > h / 2）→ 拒绝给点预测。
冻结范围（git tag law-v6.0）：本文件全部 + 检测层 commonmode.run_lines 的 cm2（purge.C = 3.3、W = 0.3、截断 ±3σ、逐题秤坏 3σ）；没有可调参数。
开发数据对比 derive5（dev_v6.py、fl/dev_v6_fl.py）：真实 53 格 MAE 0.036 → 0.035、玩具 0.027 → 0.027、联邦学习 72 格 0.171 → 0.093；
  全知诚实集 |t| > 1.96：真实 25% → 10%、玩具 32% → 19%；后验 90% 区间覆盖：真实 77–79% → 89%、玩具 62% → 77%。已知未解决：玩具仍过窄（b、σ_c 估计误差）。
v6.1（冻结后、测试前，映射预检的沙盒里发现）：sigma_c 的"总方差 − 中位数方差"相减后可能撞地板（Fashion-MNIST 沙盒：估 0.05、实际 0.38；开发数据 arable 同类），
  λ_c 按比例放大救不回来 → 秤方差取 max(σ_c², 秤对节点加权平均残差的 MAD² − 节点平均方差) + se_b²。没有碰过任何测试数据。
其余（裁判、门槛 h = δ_k / 2、按诚实概率抽 M = 64 个子集、后验 = 诚实子集 × 秤）不变。
"""
import math
import random
import statistics as st

import torch

from analytic import Phi, sigma_c
from hetero import judge_out
from intervals import rho_hat2

LAW_VERSION = "6.1"
M = 64


def premise_check(cal, ev, bc, be, h):
    """前提自检（不看真值）。cal / ev = [(f, p, r, x)]，bc / be = 每行的块号。返回 (是否拒答, 挪了的节点数, 节点数)。"""
    n = len(cal[0][2])
    moved = 0
    mad1 = lambda xs: 1.4826 * st.median(abs(x - st.median(xs)) for x in xs)
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
        se = 1.2533 * math.sqrt(mad1(mc) ** 2 / len(mc) + mad1(me) ** 2 / len(me))
        if abs(sh) > 2 * se and abs(sh) > h / 2:
            moved += 1
    return moved == n, moved, n


def _mad(x, dim=0):
    return 1.4826 * (x - x.median(dim).values).abs().median(dim).values


def _loo(R):
    n = R.shape[1]
    out = [_mad(R[:, i] - torch.cat([R[:, :i], R[:, i + 1:]], 1).median(1).values) for i in range(n)]
    return torch.stack(out) / math.sqrt(1 + math.pi / (2 * max(n - 1, 1)))


def fit(clean, bad):
    """不看真值：b、秤方差、节点方差、诚实后验概率。"""
    b, sc = sigma_c(clean)
    e = [p - r.median().item() for _, p, r, _ in clean]
    se_b = 1.2533 * 1.4826 * st.median(abs(x - b) for x in e) / math.sqrt(len(e))
    Rc, Rb = torch.stack([r for _, _, r, _ in clean]), torch.stack([r for _, _, r, _ in bad])
    spread = max(st.median(_mad(r).item() for r in Rc), 1e-9)
    sig = torch.max(_loo(Rc), _loo(Rb)).clamp(min=0.5 * spread)
    prec = 1 / sig ** 2
    # 宽度自检（干净题）
    n = Rc.shape[1]
    if n >= 3:
        tot = prec.sum()
        loo_m = ((Rc * prec).sum(1, keepdim=True) - Rc * prec) / (tot - prec)
        zn = (Rc - loo_m) / torch.sqrt(sig ** 2 + 1 / (tot - prec))
        lam_n = max(1.0, _mad(zn.flatten()).item())
    else:
        lam_n = 1.0
    prec = prec / lam_n ** 2
    Pn = prec.sum().item()
    mn = (Rc * prec).sum(1) / Pn
    pc = torch.tensor([p for _, p, _, _ in clean])
    resid = _mad(pc - b - mn).item() ** 2 - 1 / Pn                     # v6.1：秤方差的第二个估计（秤对节点加权平均的残差，扣节点平均自身的方差）
    vc = max(sc ** 2, resid) + se_b ** 2
    lam_c = max(1.0, _mad((pc - b - mn) / math.sqrt(vc + 1 / Pn)).item())
    vc *= lam_c ** 2
    # 诚实后验
    pb = torch.tensor([p for _, p, _, _ in bad])
    X = Rb - (pb - b).unsqueeze(1)
    xbar = X.median(0).values
    v = (1.2533 * _mad(X)) ** 2 / len(bad) + se_b ** 2
    tau = st.median(x for *_, x in bad)
    lr = torch.sqrt((v + tau ** 2) / v.clamp(min=1e-18)) * torch.exp(-0.5 * xbar ** 2 * (1 / v.clamp(min=1e-18) - 1 / (v + tau ** 2)))
    ph = lr / (1 + lr)
    return {"b": b, "vc": vc, "sig_c": math.sqrt(vc), "prec": prec, "ph": ph, "lam_n": lam_n, "lam_c": lam_c, "s0": max(sc, 1e-6)}


def subsets(ph, honest=None):
    if honest is not None:
        return honest.float().unsqueeze(0)
    g = torch.Generator().manual_seed(0)
    S = (torch.rand(M, len(ph), generator=g) < ph).float()
    S[S.sum(1) == 0, int(torch.argmax(ph))] = 1.0
    return S


def item_probs(clean, bad, judge, s0=None, honest=None):
    """逐题被骗概率 + PIT（真值 = 0 只在外面用）；返回 q 列表、每题 (o, m, sd, t)、fit 结果。"""
    F = fit(clean, bad)
    s = s0 if s0 is not None else F["s0"]
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, _mad(dev).min().item())
    jd = "软" if judge == "软" else "硬"
    S = subsets(F["ph"], honest)
    P = S @ F["prec"] + 1 / F["vc"]
    sd = 1 / torch.sqrt(P)
    q, parts = [], []
    for _, p, r, dk in bad:
        o, t = judge_out(r, torch.tensor(p), s, jd), dk / 2
        m = (S @ (F["prec"] * r) + (p - F["b"]) / F["vc"]) / P
        q.append(st.mean(1 - (Phi((t - (o - mi)) / si) - Phi((-t - (o - mi)) / si)) for mi, si in zip(m.tolist(), sd.tolist())))
        parts.append((o, m, sd, t))
    return q, parts, F


def predict(clean, bad, judge, s0=None):
    return st.mean(item_probs(clean, bad, judge, s0)[0])


def _resample(rows, blocks, rg):
    if blocks is None:
        return [rg.choice(rows) for _ in rows]
    d = {}
    for x, k in zip(rows, blocks):
        d.setdefault(k, []).append(x)
    keys = list(d)
    return [x for _ in keys for x in d[rg.choice(keys)]]


def interval(clean, bad, judge, B=30, blocks=None):
    """点预测 + 区间（同 intervals.interval2 的构造：bootstrap + ρ̂ < 1 时两种诚实假设 + 二项项）。
    blocks = (校准期块号, 评估期块号)：题成簇时（同一轮 / 同一天）整块重采样，二项项的样本量用块数。"""
    s0 = max(sigma_c(clean)[1], 1e-6)
    point = predict(clean, bad, judge, s0)
    bs = []
    for bt in range(B):
        rg = random.Random(bt)
        bs.append(predict(_resample(clean, blocks[0] if blocks else None, rg), _resample(bad, blocks[1] if blocks else None, rg), judge, s0))
    bs.sort()
    ne = len(set(blocks[1])) if blocks else len(bad)
    e = 1.645 * math.sqrt(max(point * (1 - point), 1e-4) / ne)
    lo, hi = bs[int(0.05 * B)], bs[int(0.95 * B) - 1]
    rh, off = rho_hat2(clean, bad)
    if rh < 1:
        v, idx = torch.sort(off.abs())
        k = int(torch.argmax(v[1:] - v[:-1])) + 1 if len(v) > 1 else 1
        A = torch.zeros(len(off), dtype=torch.bool)
        A[idx[:k]] = True
        for honest in (A, ~A):
            if honest.any():
                q = st.mean(item_probs(clean, bad, judge, s0, honest)[0])
                lo, hi = min(lo, q), max(hi, q)
    return point, max(0.0, lo - e), min(1.0, hi + e), s0, rh
