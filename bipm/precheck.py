"""映射预检（只看沙盒：2015 年以前）。"""
import statistics as st, torch, bipm as B
L = B.all_labs(); gps, glo = B.gnss("gps"), B.gnss("glo")
ep = [t for t in sorted({t for d in L.values() for t in d}) if 55559 + 5 <= t < B.SANDBOX_END]
print(len(L), "个实验室；沙盒历元", len(ep), ep[0], ep[-1])
D = {k: B.diffs(v, ep) for k, v in L.items()}
good = [k for k, d in D.items() if len(d) >= 0.97 * len(ep)]
print("覆盖 ≥ 97%：", len(good))
pg, pl = B.diffs(gps, ep), B.diffs(glo, ep)
E = [t for t in ep if t in pg and t in pl and all(t in D[k] for k in good)]
print("全齐历元", len(E))
X = torch.tensor([[D[k][t] for k in good] for t in E])
A, Bb = torch.tensor([pg[t] for t in E]), torch.tensor([pl[t] for t in E])
mad = lambda x: 1.4826 * (x - x.median()).abs().median().item()
m = torch.tensor([mad(X[:, j]) for j in range(X.shape[1])])
print("实验室差分 MAD（ns）分位 10/50/90", [round(m.quantile(q).item(), 2) for q in (.1, .5, .9)], "最大", round(m.max().item(), 1))
print(f"秤 GPS MAD {mad(A):.2f}，GLONASS MAD {mad(Bb):.2f} ns；两秤相关 {torch.corrcoef(torch.stack([A, Bb]))[0, 1]:.2f}")
order = sorted(range(len(good)), key=lambda j: m[j].item())
for name, idx in (("全部", list(range(len(good)))), ("最稳 15", order[:15]), ("中间 15", order[len(order) // 2 - 7:len(order) // 2 + 8])):
    S = X[:, idx]; S = S - S.median(0).values
    C = torch.corrcoef(S.T); k = C.shape[0]
    rob = torch.tensor([[1.0 if a != b else 0.0 for b in range(k)] for a in range(k)])
    z = S / torch.tensor([mad(S[:, j]) for j in range(k)]).clamp(min=1e-6)
    rank = S.argsort(0).argsort(0).float()
    Cr = torch.corrcoef(rank.T)
    print(f"{name}（{k}）：两两相关 Pearson {(C.sum() - k) / (k * (k - 1)):.3f}、Spearman {(Cr.sum() - k) / (k * (k - 1)):.3f}；|z| 95% {z.abs().flatten().quantile(.95):.2f}、99% {z.abs().flatten().quantile(.99):.2f}；共识 MAD {mad(S.median(1).values):.2f}；共识与 GPS 相关 {torch.corrcoef(torch.stack([S.median(1).values, A]))[0, 1]:.2f}，与 GLO {torch.corrcoef(torch.stack([S.median(1).values, Bb]))[0, 1]:.2f}")
print("节点名（最稳 15）", [good[j] for j in order[:15]])
print("相对真值：共识中位", round(X.median(1).values.median().item(), 3), "ns（差分，真值 0）")
