"""定位第二步：按裁判拆自然格；公式的地板；裁判的实际误差 vs 后验的实际误差 vs 两者之差。"""
import math, statistics as st, sys
sys.path.insert(0, "../switching")
import torch
import law_v6 as L
import validate_bipm as V
from analytic import Phi, sigma_c
from hetero import judge_out, measure
rows = {("软", 1.5): [], ("软", 3.0): [], ("硬2", 1.5): [], ("硬2", 3.0): []}
for gi, w, cal, ev in V.windows("test"):
    for h in (1.5, 3.0):
        c = [(False, a, r, 2 * h) for _, a, _, r in cal]
        e = [(True, a, r, 2 * h) for _, a, _, r in ev]
        s0 = max(sigma_c(c)[1], 1e-6)
        for judge in ("软", "硬2"):
            q, parts, F = L.item_probs(c, e, judge, s0)
            act = measure(c + e, 2 * h, judge, s0)["miss"]
            floor = st.mean(st.mean(2 * (1 - Phi(h / si)) for si in sd.tolist()) for _, _, sd, _ in parts)
            s = s0
            if judge == "硬2":
                s = max(s, L._mad(torch.stack([r - p for _, p, r, _ in c])).min().item())
            o = [p_[0] for p_ in parts]
            m = [p_[1].mean().item() for p_ in parts]
            sd = [p_[2].mean().item() for p_ in parts]
            rows[(judge, h)].append(dict(act=act, pt=st.mean(q), floor=floor, s=s, s0=s0, o=o, m=m, sd=sd, vc=math.sqrt(F["vc"])))
mad = lambda v: 1.4826 * st.median(abs(x) for x in v)
print("裁判 h    实测   预测   地板   | 裁判误差 MAD  后验中心误差 MAD  |o−m| 中位  sd 中位  裁判窗宽 s  秤标准差")
for k, s in rows.items():
    o = [x for r in s for x in r["o"]]; m = [x for r in s for x in r["m"]]; sd = [x for r in s for x in r["sd"]]
    print(f"{k[0]:3s} {k[1]:.1f}  {st.mean(r['act'] for r in s):.3f}  {st.mean(r['pt'] for r in s):.3f}  {st.mean(r['floor'] for r in s):.3f}  |  {mad(o):.2f}          {mad(m):.2f}           {st.median(abs(a - b) for a, b in zip(o, m)):.2f}       {st.median(sd):.2f}    {st.median(r['s'] for r in s):.2f}     {st.median(r['vc'] for r in s):.2f}")
print("\n裁判误差的尾巴（|o| 分位 50/90/99）与相关：")
for j in ("软", "硬2"):
    s = rows[(j, 1.5)]
    o = sorted(abs(x) for r in s for x in r["o"]); om = [(a, b) for r in s for a, b in zip(r["o"], r["m"])]
    x, y = torch.tensor([a for a, _ in om]), torch.tensor([b for _, b in om])
    print(f"  {j}：{o[len(o) // 2]:.2f} / {o[int(.9 * len(o))]:.2f} / {o[int(.99 * len(o))]:.2f}；o 与 m 相关 {torch.corrcoef(torch.stack([x, y]))[0, 1]:.2f}")
print("\n把裁判换成全部节点中位数（不看秤）当基线：")
for h in (1.5, 3.0):
    a = []
    for gi, w, cal, ev in V.windows("test"):
        a += [abs(r.median().item()) > h for _, _, _, r in ev]
    print(f"  h={h}：被骗 {st.mean(a):.3f}")
