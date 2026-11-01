"""推导版第四版的两个缺口诊断（2026-11-01）。

一、约束很弱时偏悲观：是原理上的边界，还是能改进？
  全知预测者：已知谁诚实（诚实权重 = 1 / 0），其余和第四版相同。
  可识别度 z（可观测）：跨题汇总后，偏离约束中心最远的节点离 0 有几个标准误（max_i |均值 x_i| / SE_i）。
  预注册（运行前；新种子 bnd2，约束 ×1 / ×10 / ×100 各 120 个配置）：
    1. 全知预测者在约束 ×100 上 MAE ≤ 0.05（缺的信息正好是"谁诚实"）
    2. 第四版的误差集中在可识别度低的配置：z ≥ 2 的 MAE ≤ 0.06，z < 2 的 MAE ≥ 0.12
    → 两条都成立 = 约束很弱是原理上的边界，规律在这里应该给区间
二、硬裁判盲测三 δ = 5% 差 16 个百分点：在复刻玩具上逐题对比预测和实际（诊断，不预注册）

用法：~/gut-external/torch215-venv/bin/python boundary2.py
"""
import math
import statistics as st

import torch

from analytic import Phi, derive4, sigma_c
from failmap import stress_toy
from hetero import judge_out, measure


def oracle(rows, judge, s, honest):
    """全知预测者：derive4 里的诚实权重换成真实的诚实标记。"""
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, (1.4826 * (dev - dev.median(0).values).abs().median(0).values).min().item())
    jd = "软" if judge == "软" else "硬"
    b, sc = sigma_c(clean)
    w = honest / honest.sum()
    D = torch.stack([r - r.median() for _, _, r, _ in clean])
    spread = max(st.median(1.4826 * d.abs().median().item() for d in D), 1e-9)
    sig = (1.4826 * (D - D.median(0).values).abs().median(0).values).clamp(min=0.5 * spread)
    out = []
    for _, p, r, dk in bad:
        t, o, c0 = dk / 2, judge_out(r, torch.tensor(p), s, jd), p - b
        v = 1 / (1 / sig ** 2 + 1 / sc ** 2)
        m = v * (r / sig ** 2 + c0 / sc ** 2)
        lik = w * torch.exp(-0.5 * (r - c0) ** 2 / (sig ** 2 + sc ** 2)) / torch.sqrt(sig ** 2 + sc ** 2)
        ms = torch.tensor([1 - (Phi((t - (o - mi)) / si) - Phi((-t - (o - mi)) / si)) for mi, si in zip(m.tolist(), torch.sqrt(v).tolist())])
        out.append((lik * ms).sum().item() / lik.sum().item() if lik.sum() > 0 else ms.mean().item())
    return st.mean(out)


def ident(rows):
    clean = [x for x in rows if not x[0]]
    b, _ = sigma_c(clean)
    X = torch.stack([r - (p - b) for _, p, r, _ in rows])
    return (X.mean(0).abs() / (X.std(0) / math.sqrt(len(rows))).clamp(min=1e-12)).max().item()


def part1():
    torch.manual_seed(14)
    for judge in ("软", "硬2"):
        print(f"\n== {judge} ==")
        print("约束倍数   第四版 MAE   全知 MAE    z≥2：占比 / 第四版 MAE    z<2：占比 / 第四版 MAE")
        for cm in (1.0, 10.0, 100.0):
            e4, eo, zs = [], [], []
            for k in range(120):
                rows, delta, sc = stress_toy(k, judge, tag="bnd2", raw=True, cmult=cm)
                m = measure(rows, delta, judge, sc)
                n = len(rows[0][2])
                honest = torch.tensor([1.0 if i < max(1, round(n * 1 / 7)) else 0.0 for i in range(n)])
                # stress_toy 里干净节点排在前面，个数 = n_clean；从 rows 里还原：无故障题和故障题上偏移为 0 的节点
                fault = torch.stack([r for f, _, r, _ in rows if f]).mean(0) - torch.stack([r for f, _, r, _ in rows if not f]).mean(0)
                honest = (fault.abs() < 1e-9 + 0.25 * fault.abs().max()).float()
                e4.append(m["miss"] - derive4(rows, judge, sc))
                eo.append(m["miss"] - oracle(rows, judge, sc, honest))
                zs.append(ident(rows))
            hi = [abs(e) for e, z in zip(e4, zs) if z >= 2]
            lo = [abs(e) for e, z in zip(e4, zs) if z < 2]
            print(f"×{cm:<6g}    {st.mean(abs(x) for x in e4):.3f}       {st.mean(abs(x) for x in eo):.3f}      "
                  f"{len(hi) / 120:4.0%} / {st.mean(hi) if hi else float('nan'):.3f}              {len(lo) / 120:4.0%} / {st.mean(lo) if lo else float('nan'):.3f}")


def part2():
    g = torch.Generator().manual_seed(2026)
    ld = math.log10(1.05)
    rows = []
    for _ in range(3000):
        f = torch.rand(1, generator=g).item() < 0.5
        p = 0.0041 + 0.0201 * torch.randn(1, generator=g).item()
        r = 0.0062 * torch.randn(7, generator=g) + (torch.tensor([0, 0, 1, 1, 1, 1, 1.]) * ld if f else 0)
        rows.append((f, p, r, ld))
    clean = [x for x in rows if not x[0]]
    dev = torch.stack([r - p for _, p, r, _ in clean])
    s = max(0.0201, (1.4826 * (dev - dev.median(0).values).abs().median(0).values).min().item())
    print(f"\n== 硬2 复刻盲测三 δ = 5%：剔除窗口 2.5 × {s:.4f} = {2.5 * s:.4f}，门槛 t = {ld / 2:.4f} ==")
    kinds = {}
    for f, p, r, dk in rows:
        if not f:
            continue
        keep = (r - p).abs() <= 2.5 * s
        o = judge_out(r, torch.tensor(p), s, "硬")
        kind = "全剔除→中位数" if not keep.any() else f"留下 诚实{int(keep[:2].sum())} 投毒{int(keep[2:].sum())}"
        kinds.setdefault(kind, []).append(abs(o) > dk / 2)
    for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:16s} 占 {len(v) / sum(map(len, kinds.values())):5.0%}   实际漏检 {st.mean(v):5.0%}")
    print(f"  第四版整体预测 {derive4(rows, '硬2', 0.0201):.0%}，实际 {measure(rows, ld, '硬2', 0.0201)['miss']:.0%}")


if __name__ == "__main__":
    part1()
    part2()
