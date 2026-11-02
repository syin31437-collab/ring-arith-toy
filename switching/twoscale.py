"""两把秤的前提自检（2026-11-02，库伯：先攻"察觉前提失效"）。开发版，还没冻结。

一把秤分不开"节点一起偏"和"秤不准"。两把互相独立的秤 A、B + 节点共识 c，三角帽（稳健方差，块内先取中位）：
  var_c = ½[var(c − A) + var(c − B) − var(A − B)]
节点互相独立时共识应有的方差 v_ind = (π/2)·σ_节点² / n（σ_节点 = 留一噪声中位）。共有误差 σ_cm² = var_c − v_ind。
判定：σ_cm 的 bootstrap 5% 下界 > h / 2（共有误差大到和伤害同量级）→ 拒绝点预测。不用真值。
前提：A、B 互相独立且不和节点共享误差；A、B 也一起偏时 var_c 被低估（漏报，不会误报）。
"""
import math
import random
import statistics as st

import torch


def _mad2(v):
    m = st.median(v)
    return (1.4826 * st.median(abs(x - m) for x in v)) ** 2


def hat(c, a, b):
    ca = [x - y for x, y in zip(c, a)]
    cb = [x - y for x, y in zip(c, b)]
    ab = [x - y for x, y in zip(a, b)]
    return 0.5 * (_mad2(ca) + _mad2(cb) - _mad2(ab))


def check(R, a, b, h, blocks=None, B=200, seed=0):
    """R (题, 节点)，a / b 两把秤（已去各自常数偏移无所谓，MAD 不受常数影响）。返回 (是否拒答, σ_cm 点估计, σ_cm² 下界, v_ind)。"""
    n = R.shape[1]
    from law_v6 import _loo
    sig = _loo(R).median().item()
    v_ind = (math.pi / 2) * sig ** 2 / n
    c = R.median(1).values.tolist()
    a, b = list(a), list(b)
    pt = hat(c, a, b) - v_ind
    keys = sorted(set(blocks)) if blocks is not None else list(range(len(c)))
    idx = {k: [] for k in keys}
    for i, k in enumerate(blocks if blocks is not None else range(len(c))):
        idx[k].append(i)
    rg = random.Random(seed)
    bs = []
    for _ in range(B):
        ii = [i for _ in keys for i in idx[rg.choice(keys)]]
        bs.append(hat([c[i] for i in ii], [a[i] for i in ii], [b[i] for i in ii]) - v_ind)
    bs.sort()
    lo = bs[int(0.05 * B)]
    return lo > (h / 2) ** 2, math.sqrt(max(pt, 0.0)), lo, v_ind
