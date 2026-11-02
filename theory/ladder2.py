"""阶梯实测第二版：同一套检测，跑在两个完全不同的领域（2026-11-02，库伯：先验证普世定律）。
  SLQA（新领域，这段测试数据没看过）：USGS 泥沙实验室，节点 7 个实验室、两把秤 = 另外 2 个实验室（粗秤，和节点同类同噪声）；单位 ln。δ ∈ {0.05, 0.10, 0.20}。
    测试段 2018 起；随机 12 组实验室 × 2 个不重叠窗口（校准 40 题 + 评估 30 题）。沙盒预检：④ Pearson 0.096 ✅、① −2% ✅、② |z| 95% 6.55 严重厚尾 ❌ → 不测点预测，只测 T1、T3（它们不依赖噪声形状）。
  BIPM（复核，数据看过）：validate_bipm 的 17 个窗口-组，9 个节点，秤 GPS、GLONASS；单位 ns。δ ∈ {6, 12, 24}。
打分（修上一轮的上限问题）：一条序列的分数 = |评估期中位 − 校准期中位| / [1.2533 √(MAD校² / n校 + MAD评² / n评)]，不截断。
  L0 同伴：每个节点 − 其他节点中位数，取最大。L2 一把秤：每个节点 − A、共识 − A，取最大。L3 两把秤：L2 再加 共识 − B、A − B。
场景：S0 不注入；one 1 个节点；min 少数（SLQA 3 / 7、BIPM 4 / 9）；maj 多数（5 / 7、6 / 9）；all 全部；drift 全部线性涨到 2δ；scale 秤 A。
归因 all vs scale：L3 比 共识 − B 和 A − B 的分数，大的那个判为偏了；L2 只有 共识 − A 的分数可用（两折交叉选门槛）。

预注册（运行前，两个领域同一组判据）：
  1. L0：all、drift、scale 的 AUC ≤ 0.60（所有 δ）—— T1
  2. L0：one、min、maj 在中、大 δ 的 AUC ≥ 0.80 —— 少数和多数同伴看得见（噪声大的节点可能拖低 one）
  3. L2：全部场景在中、大 δ 的 AUC ≥ 0.80 —— T3a
  4. L2 归因 ≤ 60% —— T3b
  5. L3 归因在中、大 δ ≥ 80% —— T3b
  两个领域都成立才算这条跨领域成立。

用法：~/gut-external/torch215-venv/bin/python ladder2.py
"""
import json, math, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("../switching", "../bipm", "../slqa"):
    sys.path.insert(0, os.path.join(HERE, d))
NAN = float("nan")


def _mad(v):
    m = st.median(v)
    return 1.4826 * st.median(abs(x - m) for x in v)


def stat(c, e):
    c = [x for x in c if x == x]
    e = [x for x in e if x == x]
    if len(c) < 5 or len(e) < 5:
        return 0.0
    se = 1.2533 * math.sqrt(_mad(c) ** 2 / len(c) + _mad(e) ** 2 / len(e))
    return abs(st.median(e) - st.median(c)) / max(se, 1e-9)


def med(vals):
    v = [x for x in vals if x == x]
    return st.median(v) if len(v) >= 2 else NAN


def scores(cal, ev, nodes_sh, zeta):
    """cal / ev：[(A, B, [节点...])]；nodes_sh(t, j) 注入；zeta：秤 A 注入。"""
    n = len(cal[0][2])
    Rc = [x[2] for x in cal]
    Re = [[x[2][j] + nodes_sh(t, j) for j in range(n)] for t, x in enumerate(ev)]
    Ac, Ae = [x[0] for x in cal], [x[0] + zeta for x in ev]
    Bc, Be = [x[1] for x in cal], [x[1] for x in ev]
    peer = max(stat([r[j] - med(r[:j] + r[j + 1:]) for r in Rc], [r[j] - med(r[:j] + r[j + 1:]) for r in Re]) for j in range(n))
    node_a = max(stat([r[j] - a for r, a in zip(Rc, Ac)], [r[j] - a for r, a in zip(Re, Ae)]) for j in range(n))
    Cc, Ce = [med(r) for r in Rc], [med(r) for r in Re]
    ca = stat([c - a for c, a in zip(Cc, Ac)], [c - a for c, a in zip(Ce, Ae)])
    cb = stat([c - b for c, b in zip(Cc, Bc)], [c - b for c, b in zip(Ce, Be)])
    ab = stat([a - b for a, b in zip(Ac, Bc)], [a - b for a, b in zip(Ae, Be)])
    return {"L0": peer, "L2": max(node_a, ca), "L3": max(node_a, ca, cb, ab), "ca": ca, "cb": cb, "ab": ab}


def slqa_windows():
    import slqa as S
    E, items = S.load()
    cut = int(.3 * len(items))
    sb, te = items[:cut], items[cut:]
    labs = sorted(l for l in {l for i in items for l in E[i]} if sum(l in E[i] for i in items) >= 0.6 * len(items) and sum(l in E[i] for i in sb) >= 0.5 * len(sb))
    rg = random.Random("slqa-ladder")
    for g in range(12):
        pick = rg.sample(labs, 9)
        nodes, a, b = pick[:7], pick[7], pick[8]
        rows = [(E[i][a], E[i][b], [E[i].get(l, NAN) for l in nodes]) for i in te if a in E[i] and b in E[i] and sum(l in E[i] for l in nodes) >= 5]
        for s in (0, 70):
            if len(rows) >= s + 70:
                yield rows[s:s + 40], rows[s + 40:s + 70]


def bipm_windows():
    import validate_bipm as V
    for gi, w, cal, ev in V.windows("test"):
        f = lambda rows: [(x[1], x[2], x[3].tolist()) for x in rows]
        yield f(cal), f(ev)


def auc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def run(name, windows, deltas, kmin, kmaj):
    rg = random.Random("ladder2-" + name)
    X = []
    W = list(windows)
    for cal, ev in W:
        n, T = len(cal[0][2]), len(ev)
        X.append(("S0", 0.0, scores(cal, ev, lambda t, j: 0.0, 0.0)))
        for dl in deltas:
            for sc, k in (("one", 1), ("min", kmin), ("maj", kmaj), ("all", n)):
                idx = set(rg.sample(range(n), k))
                X.append((sc, dl, scores(cal, ev, lambda t, j, idx=idx, dl=dl: dl if j in idx else 0.0, 0.0)))
            X.append(("drift", dl, scores(cal, ev, lambda t, j, dl=dl, T=T: 2 * dl * (t + 1) / T, 0.0)))
            X.append(("scale", dl, scores(cal, ev, lambda t, j: 0.0, dl)))
    neg = {L: [x[2][L] for x in X if x[0] == "S0"] for L in ("L0", "L2", "L3")}
    print(f"\n===== {name}：{len(W)} 个窗口 =====\n场景 δ      L0     L2     L3")
    tab = {}
    for sc in ("one", "min", "maj", "all", "drift", "scale"):
        for dl in deltas:
            s = [x[2] for x in X if x[0] == sc and x[1] == dl]
            tab[(sc, dl)] = {L: auc([y[L] for y in s], neg[L]) for L in neg}
            print(f"{sc:6s}{dl:6g}  " + "  ".join(f"{tab[(sc, dl)][L]:.2f} " for L in neg))
    print("归因 all vs scale：")
    attr = {}
    for dl in deltas:
        a = [x[2] for x in X if x[0] == "all" and x[1] == dl]
        s = [x[2] for x in X if x[0] == "scale" and x[1] == dl]
        l3 = (st.mean(y["cb"] > y["ab"] for y in a) + st.mean(y["ab"] >= y["cb"] for y in s)) / 2
        def fold(tr_a, tr_s, te_a, te_s):
            best = max((st.mean(y["ca"] > c for y in tr_a) + st.mean(y["ca"] <= c for y in tr_s), c) for c in sorted({y["ca"] for y in tr_a + tr_s}))[1]
            return (st.mean(y["ca"] > best for y in te_a) + st.mean(y["ca"] <= best for y in te_s)) / 2
        l2 = (fold(a[::2], s[::2], a[1::2], s[1::2]) + fold(a[1::2], s[1::2], a[::2], s[::2])) / 2
        attr[dl] = (l2, l3)
        print(f"  δ={dl:g}：L2 {l2:.0%}（交叉）；L3 {l3:.0%}")
    return tab, attr


def verdict(name, tab, attr, deltas):
    mid = deltas[1:]
    c1 = all(tab[(s, d)]["L0"] <= .60 for s in ("all", "drift", "scale") for d in deltas)
    c2 = [(s, d, round(tab[(s, d)]["L0"], 2)) for s in ("one", "min", "maj") for d in mid if tab[(s, d)]["L0"] < .80]
    c3 = [(s, d, round(tab[(s, d)]["L2"], 2)) for s in ("one", "min", "maj", "all", "drift", "scale") for d in mid if tab[(s, d)]["L2"] < .80]
    c4 = all(attr[d][0] <= .60 for d in deltas)
    c5 = all(attr[d][1] >= .80 for d in mid)
    print(f"\n{name} 判定：1 T1 {'✅' if c1 else '❌'}；2 {'✅' if not c2 else '❌ ' + str(c2)}；3 {'✅' if not c3 else '❌ ' + str(c3)}；4 {'✅' if c4 else '❌ ' + str({d: round(attr[d][0], 2) for d in deltas})}；5 {'✅' if c5 else '❌ ' + str({d: round(attr[d][1], 2) for d in mid})}")


if __name__ == "__main__":
    for name, w, ds, a, b in (("SLQA", slqa_windows(), (0.05, 0.10, 0.20), 3, 5), ("BIPM", bipm_windows(), (6.0, 12.0, 24.0), 4, 6)):
        t, at = run(name, w, ds, a, b)
        verdict(name, t, at, ds)
