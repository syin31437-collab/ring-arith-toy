import collections, statistics as st, torch, slqa as S
E, items = S.load()
cut = int(.3 * len(items))
sb = items[:cut]
print(len(items), "题；沙盒", len(sb), sb[0], "→", sb[-1], "；测试", items[cut], "→", items[-1])
cnt = collections.Counter(l for i in items for l in E[i])
print("各实验室题数（前 20）", cnt.most_common(20))
labs = [l for l, c in cnt.items() if c >= 0.6 * len(items) and sum(l in E[i] for i in sb) >= 0.5 * len(sb)]
print("覆盖 ≥ 60% 且沙盒里 ≥ 50%：", len(labs), labs)
X = torch.tensor([[E[i].get(l, float("nan")) for l in labs] for i in sb])
mad = lambda x: 1.4826 * (x[~x.isnan()] - x[~x.isnan()].median()).abs().median().item()
off = torch.tensor([x[~x.isnan()].median().item() for x in X.T])
print("实验室误差 MAD", [round(mad(X[:, j]), 3) for j in range(len(labs))])
print("各实验室中位偏移", [round(v, 3) for v in off.tolist()])
Z = X - off
rs, rp = [], []
for a in range(len(labs)):
    for b in range(a + 1, len(labs)):
        m = ~(Z[:, a].isnan() | Z[:, b].isnan())
        if m.sum() >= 20:
            rp.append(torch.corrcoef(torch.stack([Z[m, a], Z[m, b]]))[0, 1].item())
            ra, rb = Z[m, a].argsort().argsort().float(), Z[m, b].argsort().argsort().float()
            rs.append(torch.corrcoef(torch.stack([ra, rb]))[0, 1].item())
print(f"④ 两两相关（沙盒）Pearson 中位 {st.median(rp):.3f}，Spearman 中位 {st.median(rs):.3f}，≤ 0.3 的比例 {st.mean(v <= .3 for v in rp):.0%}（{len(rp)} 对）")
c = torch.tensor([st.median(v for v in row.tolist() if v == v) for row in X if (~row.isnan()).sum() >= 3])
print(f"① 共识相对真值：中位 {c.median().item():+.3f}（ln），MAD {mad(c):.3f}")
s = Z[~Z.isnan()]
zs = (Z / torch.tensor([mad(X[:, j]) for j in range(len(labs))]))
zs = zs[~zs.isnan()]
print(f"② |z| 95% {zs.abs().quantile(.95):.2f}，99% {zs.abs().quantile(.99):.2f}")
