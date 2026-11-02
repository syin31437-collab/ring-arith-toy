"""T1 的非同义反复检验：ladder2 里的共同偏移是注入的、完全相同，同伴看不见是构造决定的。这里完全不注入，看自然发生的共同漂移。
数据：EPA 长期数据（未清洗版）6 个站，节点 = AQY、CNO、MAX（log10）；秤 A = RAM、秤 B = PAR（其实和节点也相关，不是真正独立，预期拖低有秤的成绩）；真值 = 参考仪器，只用于标注。
窗口：沙盒 90 天之后，每 14 天一格，前 7 天校准、后 7 天评估。打分同 ladder2（不截断）。
真实漂移标注（用真值）：共识 − 真值 的中位 评估期 − 校准期，|差| > log10(1.25) / 2。
  "共同"的额外条件：3 个节点相对真值的挪动同号。
注意：之前 validate_v61 用过这批数据测拒答（看过每格的被骗率），但没看过这种检验。

预注册（运行前）：
  1. 共同漂移 vs 没漂移：同伴 L0 AUC ≤ 0.60（自然的共同漂移，同伴也基本看不见）
  2. 同样的窗口：一把秤 L2 AUC ≥ 0.70，且比 L0 高 ≥ 0.10（有参照才看得见）
  3. 不同号的漂移（节点分开了）：L0 AUC ≥ 0.70（这种同伴看得见）
结果（b46fc90 预注册）：1 ❌ 同伴 L0 AUC 0.896；2 L2 0.837 但没比 L0 高；3 ✅ 0.849。而且没漂移的窗口只有 5 个，检验力很弱。
  解读：操作定义错了——"同号"不等于"共同"。不同品牌对湿度的响应幅度不同，差值暴露给同伴。T1 说的只是完全相同的那部分看不见。正确检验见 natural_t1b.py（同品牌）。

用法：~/gut-external/torch215-venv/bin/python natural_t1.py
"""
import datetime as dt, math, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../epa"))
import epa as E
from ladder2 import auc, med, stat
NODES, SA, SB = ["AQY", "CNO", "MAX"], "RAM", "PAR"
TH = math.log10(1.25) / 2
S, Ref, _ = E.load()
X = []
for loc in sorted(S):
    ts = sorted(S[loc]); t0 = dt.datetime.fromisoformat(ts[0])
    rows = []
    for t in ts:
        v = S[loc][t]
        if Ref[loc].get(t, 0) >= 3 and all(m in v and st.median(v[m]) > 0 for m in NODES + [SA, SB]):
            tr = math.log10(Ref[loc][t])
            rows.append(((dt.datetime.fromisoformat(t) - t0).days, [math.log10(st.median(v[m])) - tr for m in NODES], math.log10(st.median(v[SA])) - tr, math.log10(st.median(v[SB])) - tr))
    if not rows: continue
    for w in range(90, rows[-1][0] - 14, 14):
        c = [x for x in rows if w <= x[0] < w + 7]; e = [x for x in rows if w + 7 <= x[0] < w + 14]
        if len(c) < 48 or len(e) < 48: continue
        Rc, Re = [x[1] for x in c], [x[1] for x in e]
        n = 3
        moves = [st.median(r[j] for r in Re) - st.median(r[j] for r in Rc) for j in range(n)]
        cons = st.median(med(r) for r in Re) - st.median(med(r) for r in Rc)
        peer = max(stat([r[j] - med(r[:j] + r[j + 1:]) for r in Rc], [r[j] - med(r[:j] + r[j + 1:]) for r in Re]) for j in range(n))
        Ac, Ae = [x[2] for x in c], [x[2] for x in e]
        nodeA = max(stat([r[j] - a for r, a in zip(Rc, Ac)], [r[j] - a for r, a in zip(Re, Ae)]) for j in range(n))
        ca = stat([med(r) - a for r, a in zip(Rc, Ac)], [med(r) - a for r, a in zip(Re, Ae)])
        X.append({"loc": loc, "cons": cons, "same": all(m > 0 for m in moves) or all(m < 0 for m in moves), "moves": moves, "L0": peer, "L2": max(nodeA, ca)})
print(len(X), "个窗口，站点", sorted({x["loc"] for x in X}))
neg = [x for x in X if abs(x["cons"]) <= TH and max(abs(m) for m in x["moves"]) <= TH]
com = [x for x in X if abs(x["cons"]) > TH and x["same"]]
sep = [x for x in X if max(abs(m) for m in x["moves"]) > TH and not x["same"]]
print(f"没漂移 {len(neg)}；共同漂移 {len(com)}；不同号漂移 {len(sep)}")
a0, a2 = auc([x["L0"] for x in com], [x["L0"] for x in neg]), auc([x["L2"] for x in com], [x["L2"] for x in neg])
print(f"1. 共同漂移：同伴 L0 AUC {a0:.3f}（≤ 0.60）")
print(f"2. 共同漂移：一把秤 L2 AUC {a2:.3f}（≥ 0.70，比 L0 高 ≥ 0.10）")
print(f"3. 不同号漂移：同伴 L0 AUC {auc([x['L0'] for x in sep], [x['L0'] for x in neg]):.3f}（≥ 0.70）；L2 {auc([x['L2'] for x in sep], [x['L2'] for x in neg]):.3f}")
locs = sorted({x["loc"] for x in X})
rg = random.Random(0)
bs0, bs2 = [], []
for _ in range(500):
    pick = [rg.choice(locs) for _ in locs]
    cc = [x for l in pick for x in com if x["loc"] == l]; nn = [x for l in pick for x in neg if x["loc"] == l]
    if cc and nn:
        bs0.append(auc([x["L0"] for x in cc], [x["L0"] for x in nn])); bs2.append(auc([x["L2"] for x in cc], [x["L2"] for x in nn]))
bs0.sort(); bs2.sort()
print(f"按站点 bootstrap 95%：L0 [{bs0[int(.025 * len(bs0))]:.2f}, {bs0[int(.975 * len(bs0))]:.2f}]，L2 [{bs2[int(.025 * len(bs2))]:.2f}, {bs2[int(.975 * len(bs2))]:.2f}]")
for l in locs:
    cc = [x for x in com if x["loc"] == l]; nn = [x for x in neg if x["loc"] == l]
    if cc and nn:
        print(f"  {l}：共同 {len(cc)}、没漂 {len(nn)}；L0 {auc([x['L0'] for x in cc], [x['L0'] for x in nn]):.2f}，L2 {auc([x['L2'] for x in cc], [x['L2'] for x in nn]):.2f}")
