"""两把秤的前提自检（开发版）：规律 v6.1 的预测部分不动，只加"该不该开口"。不用真值。
输入：校准期 / 评估期节点报告 + 秤 A（规律用的那把）、秤 B（只用来自检）。全部减去同一未知真值后三者的差不含真值。
  N1 节点共有误差（校准期 + 评估期各查一次）：只对相信诚实的节点（诚实概率 ≥ 0.5；不足 2 个时用全部）算共识 c，
     三角帽 var_c = ½[var(c − A) + var(c − B) − var(A − B)]，σ_cm² = var_c − (π/2)·σ² / k；块 bootstrap 5% 下界 > (h / 2)² → 拒答（前提 B / 节点独立不成立）。
  S1 秤 A 失效（只比评估期和校准期）：A − B 的中位挪动 > h / 2 且显著（> 2 se），或 var(A − B) 的增量 bootstrap 下界 > (h / 2)² → 拒答（前提 ζ ≈ 0 不成立）。
  M1 单秤自检（law_v6.premise_check）：全部节点都相对秤挪动 → 拒答。
漏报（理论上看不见）：两把秤和节点同偏；两把秤同时被污染。

冻结：PREMISE_VERSION = 2.0 → decide()（配 law_v6 的预测）。下面的 decide2()（敏感度版）在同一批开发数据上更差，未采用，只留作记录。
开发结果（dev_premise2.py，MNIST 联邦学习 108 配置，看过结果）：
  decide：误差 > 0.15 的拒 97%；误差 ≤ 0.15 的误拒 42%；ζ = 1 拒 100%；f = 1 拒 90%（noise 50%）；前提成立拒 42%；没拒的格误差 0.052，拒了的 0.243
  decide2：65% / 39% / 75% / 80% / 32%；0.082 / 0.257
  已知问题：误拒高。来源：N1 评估 20 次（毒没分开时共有误差是真的，但预测照样准）、N1 校准 11 次（名义 5%，实际约 15%，10 个块的 bootstrap 偏乐观）、S1 9 次。
"""
import math
import random
import statistics as st

import torch

from law_v6 import _loo, _mad, fit, premise_check


PREMISE_VERSION = "2.0"


def _m2(v):
    m = st.median(v)
    return (1.4826 * st.median(abs(x - m) for x in v)) ** 2


def _hat(c, a, b):
    return 0.5 * (_m2([x - y for x, y in zip(c, a)]) + _m2([x - y for x, y in zip(c, b)]) - _m2([x - y for x, y in zip(a, b)]))


def _groups(blocks, n):
    keys = sorted(set(blocks)) if blocks is not None else list(range(n))
    idx = {k: [] for k in keys}
    for i, k in enumerate(blocks if blocks is not None else range(n)):
        idx[k].append(i)
    return keys, idx


def common(R, a, b, h, blocks=None, B=200):
    k = R.shape[1]
    vi = (math.pi / 2) * _loo(R).median().item() ** 2 / k if k >= 3 else 0.0
    c = R.median(1).values.tolist()
    keys, idx = _groups(blocks, len(c))
    rg = random.Random(0)
    bs = []
    for _ in range(B):
        ii = [i for _ in keys for i in idx[rg.choice(keys)]]
        bs.append(_hat([c[i] for i in ii], [a[i] for i in ii], [b[i] for i in ii]) - vi)
    bs.sort()
    return bs[int(0.05 * B)] > (h / 2) ** 2, math.sqrt(max(_hat(c, a, b) - vi, 0.0))


def scale_broke(ab_c, ab_e, h, bc=None, be=None, B=200):
    def med_block(v, bl):
        keys, idx = _groups(bl, len(v))
        return [st.median(v[i] for i in idx[k]) for k in keys]
    mc, me = med_block(ab_c, bc), med_block(ab_e, be)
    sh = st.median(me) - st.median(mc)
    mad1 = lambda xs: 1.4826 * st.median(abs(x - st.median(xs)) for x in xs)
    se = 1.2533 * math.sqrt(mad1(mc) ** 2 / max(len(mc), 1) + mad1(me) ** 2 / max(len(me), 1))
    k_c, i_c = _groups(bc, len(ab_c))
    k_e, i_e = _groups(be, len(ab_e))
    rg = random.Random(1)
    bs = []
    for _ in range(B):
        vc = [ab_c[i] for _ in k_c for i in i_c[rg.choice(k_c)]]
        ve = [ab_e[i] for _ in k_e for i in i_e[rg.choice(k_e)]]
        bs.append(_m2(ve) - _m2(vc))
    bs.sort()
    return (abs(sh) > 2 * se and abs(sh) > h / 2) or bs[int(0.05 * B)] > (h / 2) ** 2


def decide(clean, bad, b_c, b_e, h, bc=None, be=None):
    """返回 dict：refuse（总判定）、n1_cal、n1_ev、s1、m1、sigma_cm。b_c / b_e = 第二把秤（相对同一真值）。"""
    F = fit(clean, bad)
    keep = F["ph"] >= 0.5
    if keep.sum() < 2:
        keep = torch.ones_like(keep)
    Rc, Re = torch.stack([r for _, _, r, _ in clean]), torch.stack([r for _, _, r, _ in bad])
    ac, ae = [p for _, p, _, _ in clean], [p for _, p, _, _ in bad]
    n1c, s0 = common(Rc, ac, b_c, h, bc)
    n1e, s1v = common(Re[:, keep], ae, b_e, h, be)
    s1 = scale_broke([x - y for x, y in zip(ac, b_c)], [x - y for x, y in zip(ae, b_e)], h, bc, be)
    m1 = premise_check(clean, bad, bc if bc is not None else list(range(len(clean))), be if be is not None else list(range(len(bad))), h)[0]
    return {"refuse": n1c or n1e or s1 or m1, "n1_cal": n1c, "n1_ev": n1e, "s1": s1, "m1": m1, "sigma_cm": max(s0, s1v), "k_keep": int(keep.sum())}


# ===== 敏感度版（v6.2 的前提自检）：不再问"共有误差大不大"，直接问"第二把秤的信息会不会改变预测"。=====
TAU = 0.15                                                               # 容忍的预测变动（预测误差 > 0.15 视为不可用，所有报告里同一口径）


def _q(F, rows_p, rows_o, s, jd, extra=0.0):
    """后验用 rows_p 里的秤（F 对应这把秤），裁判输出用 rows_o 里的秤（实际部署的那把）。extra = 加到后验方差上的共有误差方差。"""
    from analytic import Phi
    from hetero import judge_out
    from law_v6 import subsets
    S = subsets(F["ph"])
    P = S @ F["prec"] + 1 / F["vc"]
    sd = torch.sqrt(1 / P + extra)
    out = []
    for (_, pp, r, dk), (_, po, _, _) in zip(rows_p, rows_o):
        o, t = judge_out(r, torch.tensor(po), s, jd), dk / 2
        m = (S @ (F["prec"] * r) + (pp - F["b"]) / F["vc"]) / P
        out.append(st.mean(1 - (Phi((t - (o - mi)) / si) - Phi((-t - (o - mi)) / si)) for mi, si in zip(m.tolist(), sd.tolist())))
    return st.mean(out)


def _cm_lower(R, a, b, blocks, B=200):
    k = R.shape[1]
    vi = (math.pi / 2) * _loo(R).median().item() ** 2 / k if k >= 3 else 0.0
    c = R.median(1).values.tolist()
    keys, idx = _groups(blocks, len(c))
    rg = random.Random(0)
    bs = []
    for _ in range(B):
        ii = [i for _ in keys for i in idx[rg.choice(keys)]]
        bs.append(_hat([c[i] for i in ii], [a[i] for i in ii], [b[i] for i in ii]) - vi)
    bs.sort()
    return max(bs[int(0.05 * B)], 0.0)


def decide2(clean, bad, b_c, b_e, h, s0, judge="软", bc=None, be=None):
    """返回 dict：refuse、d_scale（换秤的预测变动）、d_cm（加共有误差的预测变动）、m1、q。"""
    jd = "软" if judge == "软" else "硬"
    s = s0
    if judge == "硬2":
        s = max(s, _mad(torch.stack([r - p for _, p, r, _ in clean])).min().item())
    F = fit(clean, bad)
    q0 = _q(F, bad, bad, s, jd)
    cb = [(f, b, r, x) for (f, _, r, x), b in zip(clean, b_c)]
    eb = [(f, b, r, x) for (f, _, r, x), b in zip(bad, b_e)]
    d_scale = abs(_q(fit(cb, eb), eb, bad, s, jd) - q0)
    keep = F["ph"] >= 0.5
    if keep.sum() < 2:
        keep = torch.ones_like(keep)
    Rc, Re = torch.stack([r for _, _, r, _ in clean]), torch.stack([r for _, _, r, _ in bad])
    cm = max(_cm_lower(Rc, [p for _, p, _, _ in clean], b_c, bc), _cm_lower(Re[:, keep], [p for _, p, _, _ in bad], b_e, be))
    d_cm = abs(_q(F, bad, bad, s, jd, extra=cm) - q0)
    m1 = premise_check(clean, bad, bc if bc is not None else list(range(len(clean))), be if be is not None else list(range(len(bad))), h)[0]
    return {"refuse": d_scale > TAU or d_cm > TAU or m1, "d_scale": d_scale, "d_cm": d_cm, "m1": m1, "q": q0, "cm": cm}
