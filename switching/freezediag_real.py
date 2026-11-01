"""诊断（开发数据，非预注册）：全知诚实集下，真实数据后验过窄的来源。
把 t = (真值 − m) / sd 拆成：格内平均（恒定偏移）和格内波动；同时看诚实节点共识误差相对后验 sd 的大小、节点精度估计是否偏高。"""
import math
import statistics as st

import torch

from analytic import sigma_c
from freezediag import real_cases

rows_out = []
for name, rows, honest in real_cases():
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    b, sc = sigma_c(clean)
    D = torch.stack([r - r.median() for _, _, r, _ in clean])
    spread = max(st.median(1.4826 * d.abs().median().item() for d in D), 1e-9)
    prec = 1 / (1.4826 * (D - D.median(0).values).abs().median(0).values).clamp(min=0.5 * spread) ** 2
    S = honest.float()
    P = (S @ prec) + 1 / sc ** 2
    sd = (1 / torch.sqrt(P)).item()
    ms, cons = [], []
    for _, p, r, _ in bad:
        ms.append((((S @ (prec * r)) + (p - b) / sc ** 2) / P).item())
        cons.append(r[honest].median().item())
    t = [-m / sd for m in ms]
    mad = lambda v: 1.4826 * st.median(abs(x - st.median(v)) for x in v)
    # 节点实际噪声（相对真值）vs 估计噪声
    Rh = torch.stack([r for _, _, r, _ in bad])[:, honest]
    act_mad = 1.4826 * (Rh - Rh.median(0).values).abs().median(0).values
    est = (1 / prec[honest].sqrt())
    rows_out.append((name, sd, st.median(t), mad(t), st.mean(abs(x) > 1.96 for x in t), st.median(cons), mad(cons), (act_mad / est).median().item(), 1 / math.sqrt(P) / max(sc, 1e-12)))
print("领域                sd       t 中位   t MAD   |t|>1.96   共识误差中位   共识 MAD   节点实际/估计噪声")
seen = {}
for r in rows_out:
    seen.setdefault(r[0], []).append(r)
agg = []
for k, v in seen.items():
    m = [st.mean(x[i] for x in v) for i in range(1, 8)]
    agg.append(m)
    print(f"{k.split(':')[-1][:16]:16s} {m[0]:.4f}  {m[1]:+7.2f}  {m[2]:6.2f}   {m[3]:5.0%}      {m[4]:+.4f}      {m[5]:.4f}     {m[6]:.2f}")
print("中位：t 中位绝对值", round(st.median(abs(a[1]) for a in agg), 2), " t MAD（应 ≈ 1）", round(st.median(a[2] for a in agg), 2), " 节点实际/估计", round(st.median(a[6] for a in agg), 2))
