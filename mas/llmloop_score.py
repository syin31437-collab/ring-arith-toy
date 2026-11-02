"""llmloop.py 的评分。真值只在这里用。用法：python llmloop_score.py <目录>"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

import llmloop as L

H = math.log10(1.4) / 2
A = L.AGENTS
ok = lambda x: abs(x) <= H
nan = float("nan")
mean = lambda v: st.mean(v) if v else nan


def load(out):
    R, miss, tot, pre = [], 0, 0, []
    for md in sorted(glob.glob(os.path.join(out, "*", "meta.json"))):
        m = json.load(open(md))
        d = os.path.dirname(md)
        for it in m["items"]:
            i, t = it["id"], math.log10(it["truth"])
            if not all(m["s1"][a].get(i, 0) > 0 for a in A):
                miss += 42
                tot += 42
                continue
            xs = {a: math.log10(m["s1"][a][i]) - t for a in A}
            v = [xs[a] for a in A]
            y, mm = st.median(v), L.mad(v)
            pre.append(dict(k=m["k"], clean=[xs[a] for a in A if a not in m["bad"]], gap=st.median(xs[a] - (math.log10(m["reports"][a][i]) - t) for a in A)))
            for fn in glob.glob(os.path.join(d, "*_c*_*.json")):
                b = os.path.basename(fn)[:-5]
                if b.startswith("s1_"):
                    continue
                arm, rest = b.split("_c", 1)
                c, a = rest.split("_", 1)
                c = float(c)
                tot += 1
                o = json.load(open(fn))["answers"].get(i, 0)
                if o <= 0:
                    miss += 1
                    continue
                r = c * mm
                x = xs[a]
                R.append(dict(k=m["k"], rep=m["rep"], arm=arm, c=c, a=a, bad=a in m["bad"], x=x, y=math.log10(o) - t, E=y, r=r, P=max(y - r, min(y + r, x))))
    return R, miss, tot, pre


def met(rs):
    g, b = [q for q in rs if ok(q["x"])], [q for q in rs if not ok(q["x"])]
    return dict(K=mean([ok(q["y"]) for q in g]), B=mean([ok(q["y"]) for q in b]), Ks=mean([ok(q["P"]) for q in g]), Bs=mean([ok(q["P"]) for q in b]), obey=mean([abs(q["y"] - q["P"]) <= .01 for q in rs]), acc=mean([ok(q["y"]) for q in rs]), acc0=mean([ok(q["x"]) for q in rs]))


def boot(rs, fn, n=500):
    reps = sorted({q["rep"] for q in rs})
    by = {u: [q for q in rs if q["rep"] == u] for u in reps}
    g = random.Random(0)
    v = sorted(fn(met([q for u in g.choices(reps, k=len(reps)) for q in by[u]])) for _ in range(n))
    return v[int(.025 * n)], v[int(.975 * n)]


def main():
    R, miss, tot, pre = load(sys.argv[1])
    print(f"节点-题 {tot}，缺失 {miss}（{miss / max(tot, 1):.1%}）；局 {len({q['rep'] for q in R})}")
    c0 = [p for p in pre if p["k"] == 0]
    if c0:
        print(f"映射预检（k=0）：共识中位偏移 {st.median(st.median(p['clean']) for p in c0):+.4f}（≤ {H / 4:.4f}）；①−报告中位 {st.median(p['gap'] for p in pre):+.4f}（≤ 0.005）")
    T = {}
    print("\nk arm       c    K[区间]            K*     B[区间]            B*     服从  net[区间]             准确率")
    for k in L.K_LEVELS:
        for arm, c in [("peers", 0.0)] + [(x, q) for x in ("rule", "ruleplus", "plain") for q in L.CS]:
            rs = [q for q in R if q["k"] == k and q["arm"] == arm and q["c"] == c]
            if not rs:
                continue
            m = met(rs)
            m["net"] = m["acc"] - m["acc0"]
            T[k, arm, c] = m
            ck, cb, cn = boot(rs, lambda z: z["K"]), boot(rs, lambda z: z["B"]), boot(rs, lambda z: z["acc"] - z["acc0"])
            print(f"{k} {arm:8s} {c:3.1f}  {m['K']:5.1%}[{ck[0]:4.0%},{ck[1]:4.0%}]  {m['Ks']:5.1%}  {m['B']:5.1%}[{cb[0]:4.0%},{cb[1]:4.0%}]  {m['Bs']:5.1%}  {m['obey']:4.0%}  {m['net']:+.3f}[{cn[0]:+.2f},{cn[1]:+.2f}]  {m['acc0']:4.0%}→{m['acc']:4.0%}")
    if (4, "rule", 1.0) not in T or (0, "rule", 0.5) not in T:
        return
    print("\n预注册")
    cells = [(k, c) for k in L.K_LEVELS for c in L.CS]
    good = [x for x in cells if abs(T[x[0], "rule", x[1]]["K"] - T[x[0], "rule", x[1]]["Ks"]) <= .05 and abs(T[x[0], "rule", x[1]]["B"] - T[x[0], "rule", x[1]]["Bs"]) <= .10]
    print(f"C1 符合的格 {len(good)}/10（≥ 8）{'✅' if len(good) >= 8 else '❌'}；不符合：{[x for x in cells if x not in good]}")
    for k in (1, 2, 3):
        a, b = T[k, "rule", 0.5], T[k, "peers", 0.0]
        s = f"C2 k={k}: K {a['K']:.1%} ≥ peers {b['K']:.1%} {'✅' if a['K'] >= b['K'] else '❌'}"
        if k >= 2:
            s += f"；准确率 {a['acc']:.1%} ≥ peers {b['acc']:.1%} − 2 点 {'✅' if a['acc'] >= b['acc'] - .02 else '❌'}"
        print(s)
    a = T[3, "rule", 0.5]
    print(f"C3 k=3: K {a['K']:.1%} ≥ 85% {'✅' if a['K'] >= .85 else '❌'}；net {a['net']:+.3f} ≥ 0 {'✅' if a['net'] >= 0 else '❌'}")
    a, b = T[4, "rule", 1.0], T[4, "peers", 0.0]
    print(f"C4 k=4: K {a['K']:.1%} ≥ 75% {'✅' if a['K'] >= .75 else '❌'}；net {a['net']:+.3f} ≥ −0.05 {'✅' if a['net'] >= -.05 else '❌'}；peers net {b['net']:+.3f} ≤ −0.10 {'✅' if b['net'] <= -.1 else '❌'}")
    for k in (3, 4):
        d = T[k, "rule", 1.0]["K"] - T[k, "plain", 1.0]["K"]
        print(f"C5 k={k}: rule − plain K {d:+.3f} ≥ 0.10 {'✅' if d >= .1 else '❌'}")
    low = min(T[x[0], "rule", x[1]]["obey"] for x in cells)
    print(f"C6 rule 最低服从率 {low:.0%} ≥ 85% {'✅' if low >= .85 else '❌'}")
    G = [q for q in R if q["arm"] == "rule" and ok(q["x"])]
    ins, out = [q for q in G if abs(q["E"]) <= H + q["r"]], [q for q in G if abs(q["E"]) > H + q["r"]]
    print(f"C7 |E| ≤ h+r：K {mean([ok(q['y']) for q in ins]):.1%}（n={len(ins)}）≥ 97% {'✅' if mean([ok(q['y']) for q in ins]) >= .97 else '❌'}；|E| > h+r：K {mean([ok(q['y']) for q in out]):.1%}（n={len(out)}）≤ 25% {('✅' if mean([ok(q['y']) for q in out]) <= .25 else '❌') if len(out) >= 30 else '（样本不足，只报告）'}")
    best = [k for k in L.K_LEVELS if T[k, "rule", 0.5]["K"] >= .95 and T[k, "rule", 0.5]["net"] >= .10]
    print(f"C8 同时 K ≥ 95%、net ≥ +0.10 的 k：{best}；最大 k = {max(best) if best else '无'}（预测 2）")
    print("\n按模型（只报告）：rule c=0.5 的服从率 / K / B（全部 k）")
    for a in A:
        rs = [q for q in R if q["arm"] == "rule" and q["c"] == 0.5 and q["a"] == a]
        m = met(rs)
        print(f"  {a:14s} {m['obey']:.0%} {m['K']:.1%} {m['B']:.1%}（K* {m['Ks']:.1%} B* {m['Bs']:.1%}）")


if __name__ == "__main__":
    main()
