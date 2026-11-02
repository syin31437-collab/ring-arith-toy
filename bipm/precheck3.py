import sys
sys.path.insert(0, "../switching")
import torch, statistics as st, validate_bipm as V, law_v6 as L
mad = lambda x: 1.4826 * (x - x.median()).abs().median().item()
for gi, w, cal, ev in V.windows("sandbox"):
    c = [(False, a, r, 3.0) for _, a, _, r in cal]; e = [(True, a, r, 3.0) for _, a, _, r in ev]
    F = L.fit(c, e); P = torch.tensor([a for _, a, _, _ in cal + ev])
    print(f"组 {gi} 窗 {w}：题 {len(c)}/{len(e)}，秤实际 MAD {mad(P):.2f} / 规律估的 {F['sig_c']:.2f} = {mad(P) / F['sig_c']:.2f}；节点散布 {[round(1 / x.sqrt().item(), 1) for x in F['prec']]}；λ {F['lam_n']:.2f}/{F['lam_c']:.2f}")
