"""沙盒：校准期长度对第 ③ 条的影响。比值 = 秤实际 MAD / 规律估的秤标准差，门槛 [0.7, 1.4]。"""
import sys, statistics as st, torch
sys.path.insert(0, "../switching")
import bipm as B, law_v6 as L, validate_bipm as V
mad = lambda x: 1.4826 * (x - x.median()).abs().median().item()
labs, G = V.groups()
gps, glo = B.gnss("gps"), B.gnss("glo")
ep = [t for t in sorted({t for d in labs.values() for t in d}) if 55564 <= t < B.SANDBOX_END]
pg = B.diffs(gps, ep); pl = B.diffs(glo, ep)
for ncal in (26, 52, 104, 156):
    ratios = []
    for g in G:
        D = [B.diffs(labs[k], ep) for k in g]
        rows = [(t, pg[t], torch.tensor([d[t] for d in D])) for t in ep if t in pg and t in pl and all(t in d for d in D)]
        for s in range(0, len(rows) - ncal - 25, ncal + 26):
            c = [(False, a, r, 3.0) for _, a, r in rows[s:s + ncal]]
            e = [(True, a, r, 3.0) for _, a, r in rows[s + ncal:s + ncal + 26]]
            F = L.fit(c, e)
            ratios.append(mad(torch.tensor([x[1] for x in c + e])) / F["sig_c"])
    ok = sum(0.7 <= r <= 1.4 for r in ratios)
    print(f"校准 {ncal:3d} 题：{len(ratios)} 窗，比值中位 {st.median(ratios):.2f}，范围 {min(ratios):.2f}–{max(ratios):.2f}，过 [0.7, 1.4] 的 {ok}/{len(ratios)}")
