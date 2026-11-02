"""EPA 映射预检（沙盒 = 每站最早 90 天）：5 个品牌 AQY、CNO、MAX、PAR、RAM 轮流当秤，其余 4 个当节点。"""
import datetime as dt, itertools, math, statistics as st, sys, collections
import torch
sys.path.insert(0, ".")
import epa as E
MAKES = ["AQY", "CNO", "MAX", "PAR", "RAM"]
S, Ref, _ = E.load()
def sandbox(loc, days=90):
    ts = sorted(S[loc]); t0 = dt.datetime.fromisoformat(ts[0])
    out = []
    for t in ts:
        if (dt.datetime.fromisoformat(t) - t0).days >= days: break
        v = S[loc][t]
        if Ref[loc].get(t, 0) >= 3 and all(m in v and st.median(v[m]) > 0 for m in MAKES):
            tru = math.log10(Ref[loc][t]); out.append([math.log10(st.median(v[m])) - tru for m in MAKES])
    return torch.tensor(out)
mad = lambda x: 1.4826 * (x - x.median()).abs().median().item()
tot = collections.defaultdict(list)
for loc in sorted(S):
    X = sandbox(loc)
    if len(X) < 200: print(loc, "沙盒只有", len(X), "小时，跳过"); continue
    Xc = X - X.median(0).values
    C = torch.corrcoef(Xc.T); k = len(MAKES)
    print(f"{loc}: {len(X)} 小时；各品牌偏移 {[round(10 ** v - 1, 2) for v in X.median(0).values.tolist()]}；MAD {[round(mad(Xc[:, j]), 3) for j in range(k)]}；两两相关平均 {(C.sum() - k) / (k * (k - 1)):.2f}")
    for j, m in enumerate(MAKES):
        idx = [i for i in range(k) if i != j]
        N = Xc[:, idx]; c = N.median(1).values; Cn = torch.corrcoef(N.T); kk = len(idx)
        tot[m].append(((Cn.sum() - kk) / (kk * (kk - 1))).item())
        tot[m + "_cs"].append(torch.corrcoef(torch.stack([Xc[:, j], c]))[0, 1].item())
        tot[m + "_mad"].append(mad(Xc[:, j]))
print("\n当秤时：其余节点两两相关 / 秤与节点共识误差相关 / 秤 MAD（各站中位）  h = %.3f" % E.H)
for m in MAKES:
    print(f"  {m}: {st.median(tot[m]):.2f} / {st.median(tot[m + '_cs']):.2f} / {st.median(tot[m + '_mad']):.3f}")
