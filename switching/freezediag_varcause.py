# 诊断（非预注册、开发数据）：全知诚实集下后验方差为什么偏小。逐个换：b 的不确定度进后验 / b、σ_c 换真值 / 节点相关
import math, statistics as st, torch
from analytic import sigma_c
from freezediag import real_cases, toy_cases
def run(cases):
    R = {}
    corrs = []
    for name, rows, honest in cases:
        clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
        b, sc = sigma_c(clean)
        e = [p - r.median().item() for _, p, r, _ in clean]
        se_b = 1.2533 * 1.4826 * st.median(abs(x - b) for x in e) / math.sqrt(len(e))
        pc = [p for _, p, _, _ in clean]
        b_t = st.median(pc); sc_t = max(1.4826 * st.median(abs(x - b_t) for x in pc), 1e-9)
        D = torch.stack([r - r.median() for _, _, r, _ in clean])
        spread = max(st.median(1.4826 * d.abs().median().item() for d in D), 1e-9)
        prec = 1 / (1.4826 * (D - D.median(0).values).abs().median(0).values).clamp(min=0.5 * spread) ** 2
        S = honest.float()
        Rc = torch.stack([r for _, _, r, _ in clean])[:, honest]
        if honest.sum() >= 2:
            C = torch.corrcoef(Rc.T); k = C.shape[0]
            corrs.append(((C.sum() - k) / (k * (k - 1))).item())
        V = {"现状": (b, sc ** 2, 1.0), "b 不确定度进后验": (b, sc ** 2 + se_b ** 2, 1.0), "b、σ_c 换真值": (b_t, sc_t ** 2, 1.0),
             "b、σ_c 真值 + 节点方差×n_h（节点完全相关）": (b_t, sc_t ** 2, float(honest.sum()))}
        for nm, (bb, v, infl) in V.items():
            for _, p, r, _ in bad:
                P = (S @ prec) / infl + 1 / v
                m = ((S @ (prec * r)) / infl + (p - bb) / v) / P
                R.setdefault(nm, []).append((-m * torch.sqrt(P)).item())
    for nm, t in R.items():
        print(f"  {nm:40s} |t| > 1.96 {st.mean(abs(x) > 1.96 for x in t):.0%}（≈ 5%）  |t| 中位 {st.median(abs(x) for x in t):.2f}（≈ 0.67）")
    print(f"  诚实节点误差两两相关 平均 {st.mean(corrs):.2f}、中位 {st.median(corrs):.2f}")
print("真实 17 套"); run(list(real_cases()))
print("玩具 60"); run(list(toy_cases()))
