"""T1 的自然检验第二版：同品牌多台（EPA NC 站，每个品牌约 9 台），同型号传感器对环境响应相近，才接近真正的共同偏移。不注入。
节点 = 某品牌在 NC 站的各台（≥ 5 台同时有数）；秤 A = 其他品牌各自中位数里的 RAM（节点是 RAM 时用 AQY）。真值 = 参考仪器，只用于标注。
窗口：每 14 天一格，前 7 天校准、后 7 天评估（≥ 48 小时）。分数同 ladder2。
标注：|共识 − 真值 的中位挪动| 的上三分之一 = 漂移大，下三分之一 = 漂移小（保证负样本够）。
共同占比（描述）：|共识挪动| / (|共识挪动| + 各台挪动相对共识的 MAD)。

预注册（运行前）：
  1. 同品牌：同伴 L0 AUC ≤ 0.60（自然的共同漂移，同伴基本看不见）
  2. 同品牌：一把秤 L2 AUC ≥ 0.70，且比 L0 高 ≥ 0.10
  3. 对照（同样的标注规则，3 个不同品牌当节点）：L0 AUC 比同品牌高 ≥ 0.10（不同品牌偏移不相等，差值看得见）
结果（9e3f420 预注册）：无法评估，不算过也不算没过。同品牌每台的在线时段错开，≥ 5 台同时有数的窗口只有 2 个（ARS），其余品牌台数不足。预注册前没先数可用窗口，是我的疏漏。按事先说定的"最后一轮"收尾，不再换数据或放宽规则。
  结论：自然共同漂移下 T1 的实测仍是空白。ladder2 里 L0 的 0.5 是构造决定的，不能当作自然证据。

用法：~/gut-external/torch215-venv/bin/python natural_t1b.py
"""
import collections, csv, datetime as dt, math, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../epa"))
import epa as E
from ladder2 import auc, med, stat
U = collections.defaultdict(lambda: collections.defaultdict(dict))
Ref = {}
with open(E.FN, newline="") as f:
    for r in csv.DictReader(f):
        if r["Location"] != "NC":
            continue
        try:
            v, ref = float(r["PM25_sens"]), float(r["PM25_Ref"])
        except ValueError:
            continue
        if r["Make_ID"].endswith("_wf"):
            continue
        U[r["Make_ID"]][r["serial_id"]][r["DateTime"]] = v
        Ref[r["DateTime"]] = ref
ts = sorted(Ref)
t0 = dt.datetime.fromisoformat(ts[0])
day = {t: (dt.datetime.fromisoformat(t) - t0).days for t in ts}
print("NC 站：", {m: len(u) for m, u in U.items()}, len(ts), "小时")
makemed = {m: {t: st.median(u[t] for u in U[m].values() if t in u and u[t] > 0) for t in ts if sum(t in u and u[t] > 0 for u in U[m].values()) >= 3} for m in U}


def cells(get_nodes, scale_make):
    X = []
    for w in range(0, day[ts[-1]] - 14, 14):
        def rows(lo, hi):
            out = []
            for t in ts:
                if not lo <= day[t] < hi or Ref[t] < 3 or t not in makemed[scale_make]:
                    continue
                vals = get_nodes(t)
                if vals is None:
                    continue
                tr = math.log10(Ref[t])
                out.append(([math.log10(v) - tr for v in vals], math.log10(makemed[scale_make][t]) - tr))
            return out
        c, e = rows(w, w + 7), rows(w + 7, w + 14)
        if len(c) < 48 or len(e) < 48:
            continue
        Rc, Re = [x[0] for x in c], [x[0] for x in e]
        n = len(Rc[0])
        Ac, Ae = [x[1] for x in c], [x[1] for x in e]
        cons = st.median(med(r) for r in Re) - st.median(med(r) for r in Rc)
        mv = [st.median(r[j] for r in Re) - st.median(r[j] for r in Rc) for j in range(n)]
        diff = st.median(abs(m - st.median(mv)) for m in mv)
        L0 = max(stat([r[j] - med(r[:j] + r[j + 1:]) for r in Rc], [r[j] - med(r[:j] + r[j + 1:]) for r in Re]) for j in range(n))
        nA = max(stat([r[j] - a for r, a in zip(Rc, Ac)], [r[j] - a for r, a in zip(Re, Ae)]) for j in range(n))
        cA = stat([med(r) - a for r, a in zip(Rc, Ac)], [med(r) - a for r, a in zip(Re, Ae)])
        X.append({"cons": abs(cons), "diff": diff, "L0": L0, "L2": max(nA, cA)})
    return X


def report(tag, X):
    if len(X) < 9:
        print(f"{tag}：只有 {len(X)} 窗，不评")
        return None
    o = sorted(X, key=lambda x: x["cons"])
    k = len(o) // 3
    neg, pos = o[:k], o[-k:]
    a0, a2 = auc([x["L0"] for x in pos], [x["L0"] for x in neg]), auc([x["L2"] for x in pos], [x["L2"] for x in neg])
    share = st.median(x["cons"] / max(x["cons"] + x["diff"], 1e-9) for x in pos)
    print(f"{tag}：{len(X)} 窗；漂移大组中位 {st.median(x['cons'] for x in pos):.3f}、小组 {st.median(x['cons'] for x in neg):.3f}；共同占比 {share:.0%}；L0 AUC {a0:.2f}；L2 AUC {a2:.2f}")
    return a0, a2


same = {}
for m in sorted(U):
    units = [s for s, u in U[m].items() if sum(t in u and u[t] > 0 for t in ts) >= 0.3 * len(ts)]
    if len(units) < 5:
        print(m, "台数不足", len(units)); continue
    def nodes(t, units=units, m=m):
        v = [U[m][s][t] for s in units if t in U[m][s] and U[m][s][t] > 0]
        return v if len(v) == len(units) else None
    r = report(f"同品牌 {m}（{len(units)} 台）", cells(nodes, "AQY" if m == "RAM" else "RAM"))
    if r:
        same[m] = r
mix = ["CNO", "MAX", "PAR"]
def nodes_mix(t):
    return [makemed[m][t] for m in mix] if all(t in makemed[m] for m in mix) else None
rm = report("不同品牌 CNO/MAX/PAR", cells(nodes_mix, "RAM"))
print("\n判定：")
if same:
    print(f"1. 同品牌 L0 AUC 中位 {st.median(v[0] for v in same.values()):.2f}（≤ 0.60）")
    print(f"2. 同品牌 L2 AUC 中位 {st.median(v[1] for v in same.values()):.2f}（≥ 0.70，比 L0 高 ≥ 0.10）")
    if rm:
        print(f"3. 不同品牌 L0 {rm[0]:.2f} vs 同品牌 {st.median(v[0] for v in same.values()):.2f}（高 ≥ 0.10）")
