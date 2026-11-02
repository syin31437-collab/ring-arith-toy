"""定位分析（看过测试段结果之后的诊断，不算验证）：纯自然格为什么高估、后验覆盖为什么偏低。消融 + 逐题归因。"""
import math, statistics as st, sys
sys.path.insert(0, "../switching")
import torch
import law_v6 as L
import validate_bipm as V
from analytic import Phi, sigma_c
from hetero import judge_out, measure
from robust import wmedian


def variant(c, e, s0, name, h):
    """返回逐题 (q, 是否被骗, PIT, o, m均值, sd均值)。"""
    F = L.fit(c, e)
    prec, vc, ph, b = F["prec"].clone(), F["vc"], F["ph"].clone(), F["b"]
    Rc, Re = torch.stack([r for _, _, r, _ in c]), torch.stack([r for _, _, r, _ in e])
    if name == "无宽度放大":
        prec = prec * F["lam_n"] ** 2
    if name == "仅校准期噪声":
        spread = max(st.median(L._mad(r).item() for r in Rc), 1e-9)
        prec = 1 / (L._loo(Rc).clamp(min=0.5 * spread) ** 2) / F["lam_n"] ** 2
    if name == "真值秤噪声":
        P = torch.tensor([p for _, p, _, _ in c + e])
        vc = max(L._mad(P).item() ** 2, 1e-12)
    if name == "全员诚实":
        ph = torch.ones_like(ph)
    S = L.subsets(ph)
    P = S @ prec + 1 / vc
    sd = 1 / torch.sqrt(P)
    out = []
    for _, p, r, dk in e:
        o, t = judge_out(r, torch.tensor(p), s0, "软"), dk / 2
        if name == "稳健中心":
            ms = []
            for s in S:
                idx = s.bool()
                vals = torch.cat([r[idx], torch.tensor([p - b])])
                w = torch.cat([prec[idx], torch.tensor([1 / vc])])
                ms.append(wmedian(vals, w).item())
            m = torch.tensor(ms)
        elif name == "真值当中心":
            m = torch.zeros(len(S))
        else:
            m = (S @ (prec * r) + (p - b) / vc) / P
        q = st.mean(1 - (Phi((t - (o - mi)) / si) - Phi((-t - (o - mi)) / si)) for mi, si in zip(m.tolist(), sd.tolist()))
        out.append((q, abs(o) > t, st.mean(Phi(x) for x in ((0 - m) / sd).tolist()), o, m.mean().item(), sd.mean().item(), r))
    return out


NAMES = ["基线", "稳健中心", "无宽度放大", "仅校准期噪声", "真值秤噪声", "全员诚实", "真值当中心"]
A = {n: [] for n in NAMES}
cells = []
for gi, w, cal, ev in V.windows("test"):
    for h in (1.5, 3.0):
        c = [(False, a, r, 2 * h) for _, a, _, r in cal]
        e = [(True, a, r, 2 * h) for _, a, _, r in ev]
        s0 = max(sigma_c(c)[1], 1e-6)
        act = measure(c + e, 2 * h, "软", s0)["miss"]
        for n in NAMES:
            v = variant(c, e, s0, n, h)
            A[n].append((h, act, st.mean(x[0] for x in v), v))
print("消融（自然格、软裁判）：")
for n in NAMES:
    for h in (1.5, 3.0):
        s = [x for x in A[n] if x[0] == h]
        pit = [y[2] for x in s for y in x[3]]
        print(f"  {n:8s} h={h}：实测 {st.mean(x[1] for x in s):.3f} 预测 {st.mean(x[2] for x in s):.3f} MAE {st.mean(abs(x[1] - x[2]) for x in s):.3f}  后验 90% 覆盖 {st.mean(.05 <= p <= .95 for p in pit):.0%}")
print("\n逐题归因（基线，h = 1.5）：高估的题长什么样")
items = [y for x in A["基线"] if x[0] == 1.5 for y in x[3]]
over = [y for y in items if y[0] >= 0.5 and not y[1]]
fine = [y for y in items if y[0] < 0.2 and not y[1]]
hit = [y for y in items if y[1]]
summ = lambda s: f"{len(s)} 题：|o| 中位 {st.median(abs(y[3]) for y in s):.2f}，|m| 中位 {st.median(abs(y[4]) for y in s):.2f}，|o − m| 中位 {st.median(abs(y[3] - y[4]) for y in s):.2f}，sd 中位 {st.median(y[5] for y in s):.2f}，节点 max|r| 中位 {st.median(y[6].abs().max().item() for y in s):.1f}"
print("  预测 ≥ 0.5 但没被骗：", summ(over))
print("  预测 < 0.2 也没被骗：", summ(fine))
print("  真被骗：            ", summ(hit))
print("\n全部题：m 的实际误差 vs 它自称的 sd：|m| / sd 中位", round(st.median(abs(y[4]) / y[5] for y in items), 2), "（校准 ≈ 0.67）；o 的实际误差 MAD", round(1.4826 * st.median(abs(y[3]) for y in items), 2), "，m 的实际误差 MAD", round(1.4826 * st.median(abs(y[4]) for y in items), 2), "，sd 中位", round(st.median(y[5] for y in items), 2))
