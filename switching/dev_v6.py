"""开发数据对比 derive5 vs v6（非验证）：格级 MAE / Spearman、全知诚实集 |t|、PIT 覆盖、诚实概率校准、逐题 Brier。"""
import statistics as st

import torch

from analytic import Phi, derive5, sigma_c
from freezediag import real_cases, toy_cases
from hetero import measure
from law_v6 import item_probs

rk = lambda v: torch.tensor(v, dtype=torch.float).argsort().argsort().float()
for label, cases in (("真实 17 套", list(real_cases())), ("玩具 60", list(toy_cases()))):
    C = {"act": [], "d5": [], "v6": []}
    T, PIT, PH, QS = [], [], [], []
    lam = []
    for name, rows, honest in cases:
        clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
        s = max(sigma_c(clean)[1], 1e-9)
        act = measure(clean + bad, bad[0][3], "软", s)["miss"]
        C["act"].append(act)
        C["d5"].append(derive5(rows, "软", s))
        q, parts, F = item_probs(clean, bad, "软", s)
        C["v6"].append(st.mean(q))
        lam.append((F["lam_n"], F["lam_c"]))
        PH += list(zip(F["ph"].tolist(), honest.tolist()))
        for qi, (o, m, sd, t) in zip(q, parts):
            PIT.append(st.mean(Phi(x) for x in ((0 - m) / sd).tolist()))
            QS.append((qi, abs(o) > t))
        _, po, _ = item_probs(clean, bad, "软", s, honest)
        T += [(-m[0] / sd[0]).item() for _, m, sd, _ in po]
    act = C["act"]
    print(f"\n===== {label}（{len(act)} 格，软裁判）=====")
    for k in ("d5", "v6"):
        sp = torch.corrcoef(torch.stack([rk(C[k]), rk(act)]))[0, 1].item()
        print(f"  {k}: 格级 MAE {st.mean(abs(a - b) for a, b in zip(C[k], act)):.3f}  Spearman {sp:.2f}")
    print(f"  v6 全知诚实集：|t| > 1.96 {st.mean(abs(x) > 1.96 for x in T):.0%}（≈ 5%）  |t| 中位 {st.median(abs(x) for x in T):.2f}（≈ 0.67）")
    print(f"  v6 抽样子集 PIT：90% 区间覆盖 {st.mean(.05 <= x <= .95 for x in PIT):.0%}")
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
    print("  诚实概率档 → 真诚实比例：" + "  ".join(f"[{a:.1f},{b:.1f}) {st.mean(h for p, h in PH if a <= p < b):.2f}（{sum(a <= p < b for p, _ in PH)}）" for a, b in bins if any(a <= p < b for p, _ in PH)))
    print("  逐题被骗 预测档 → 实际：" + "  ".join(f"[{a:.1f},{b:.1f}) {st.mean(y for q, y in QS if a <= q < b):.2f}（{sum(a <= q < b for q, _ in QS)}）" for a, b in bins if any(a <= q < b for q, _ in QS)) + f"  Brier {st.mean((q - y) ** 2 for q, y in QS):.3f}")
    print(f"  宽度自检：λ_n 中位 {st.median(l[0] for l in lam):.2f}，λ_c 中位 {st.median(l[1] for l in lam):.2f}")
