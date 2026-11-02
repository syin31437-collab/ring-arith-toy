"""llmclip.py 的评分。真值只在这里用。用法：python llmclip_score.py <目录>"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

import llmclip as L

H, A = L.H, L.AGENTS
ok = lambda x: abs(x) <= H


def load(out):
    R, miss, tot, pre = [], 0, 0, []
    for md in sorted(glob.glob(os.path.join(out, "*", "meta.json"))):
        m = json.load(open(md))
        d = os.path.dirname(md)
        for it in m["items"]:
            i, t = it["id"], math.log10(it["truth"])
            if not all(m["s1"][a].get(i, 0) > 0 for a in A):
                continue
            s = random.Random(f"sign-{m['rep']}-{i}").choice((-1, 1))
            xs = {a: math.log10(m["s1"][a][i]) - t for a in A}
            pre.append(dict(rep=m["rep"], xs=[xs[a] for a in A if a not in m["bad"]], gap=st.median(xs[a] - (math.log10(m["reports"][a][i]) - t) for a in A)))
            for fn in glob.glob(os.path.join(d, "*_e*_r*_*.json")):
                b = os.path.basename(fn)[:-5]
                arm, rest = b.split("_e", 1)
                e, rest = rest.split("_r", 1)
                r, a = rest.split("_", 1)
                e, r = float(e), float(r)
                tot += 1
                v = json.load(open(fn))["answers"].get(i, 0)
                if v <= 0:
                    miss += 1
                    continue
                E, Rr = s * e * H, r * H
                x = xs[a]
                R.append(dict(rep=m["rep"], arm=arm, e=e, r=r, a=a, bad=a in m["bad"], x=x, y=math.log10(v) - t, E=E, P=max(E - Rr, min(E + Rr, x)), inside=abs(x - E) <= Rr))
    return R, miss, tot, pre


def met(rs):
    g, b = [q for q in rs if ok(q["x"])], [q for q in rs if not ok(q["x"])]
    mean = lambda v: st.mean(v) if v else float("nan")
    return dict(K=mean([ok(q["y"]) for q in g]), B=mean([ok(q["y"]) for q in b]), Ks=mean([ok(q["P"]) for q in g]), Bs=mean([ok(q["P"]) for q in b]), obey=mean([abs(q["y"] - q["P"]) <= .01 for q in rs]), still=mean([abs(q["y"] - q["x"]) <= .005 for q in rs]), acc=mean([ok(q["y"]) for q in rs]), acc0=mean([ok(q["x"]) for q in rs]))


def boot(rs, key, n=500):
    reps = sorted({q["rep"] for q in rs})
    by = {u: [q for q in rs if q["rep"] == u] for u in reps}
    g = random.Random(0)
    v = sorted(met([q for u in g.choices(reps, k=len(reps)) for q in by[u]])[key] for _ in range(n))
    return v[int(.025 * n)], v[int(.975 * n)]


def frac(q):
    return (q["y"] - q["x"]) / (q["E"] - q["x"])


def main():
    R, miss, tot, pre = load(sys.argv[1])
    print(f"节点-题 {tot}，缺失 {miss}（{miss / max(tot, 1):.1%}）；局 {len({q['rep'] for q in R})}")
    cons = st.median(st.median(p["xs"]) for p in pre)
    print(f"映射预检：未污染共识中位偏移 {cons:+.4f}（≤ {H / 4:.4f}）；①−报告中位 {st.median(p['gap'] for p in pre):+.4f}（≤ 0.005）")
    T = {}
    print("\narm    e/h  r/h   K[区间]           K*     B[区间]           B*     服从  没动  准确率 ①→②")
    for arm in L.ARMS:
        for e, r in L.CELLS:
            rs = [q for q in R if q["arm"] == arm and q["e"] == e and q["r"] == r]
            if not rs:
                continue
            m = met(rs)
            T[arm, e, r] = m
            ck, cb = boot(rs, "K"), boot(rs, "B")
            print(f"{arm:6s} {e:3.1f}  {r:4.2f}  {m['K']:5.1%}[{ck[0]:4.0%},{ck[1]:4.0%}]  {m['Ks']:5.1%}  {m['B']:5.1%}[{cb[0]:4.0%},{cb[1]:4.0%}]  {m['Bs']:5.1%}  {m['obey']:4.0%}  {m['still']:4.0%}  {m['acc0']:4.0%}→{m['acc']:4.0%}")
    if ("rule", 0.0, 8.0) not in T:
        return
    C8 = [c for c in L.CELLS if c != L.PLACEBO]
    print("\n预注册")
    good = [c for c in C8 if abs(T["rule", *c]["K"] - T["rule", *c]["Ks"]) <= .05 and abs(T["rule", *c]["B"] - T["rule", *c]["Bs"]) <= .10]
    print(f"L1 rule 符合的格 {len(good)}/8（≥ 7）{'✅' if len(good) >= 7 else '❌'}")
    print(f"L2 rule 最低服从率 {min(T['rule', *c]['obey'] for c in C8):.0%}（每格 ≥ 85%）{'✅' if min(T['rule', *c]['obey'] for c in C8) >= .85 else '❌'}")
    d = st.mean(abs(T["plain", *c]["K"] - T["plain", *c]["Ks"]) for c in C8)
    mono_e = all(T["plain", e1, r]["K"] >= T["plain", e2, r]["K"] - .02 for r in L.RS for e1, e2 in zip(L.ES, L.ES[1:]))
    mono_r = all(T["plain", e, 1.0]["K"] >= T["plain", e, 0.25]["K"] - .02 for e in L.ES)
    P = [q for q in R if q["arm"] == "plain" and (q["e"], q["r"]) in C8 and abs(q["E"] - q["x"]) > .02]
    fo, fi = st.median(map(frac, [q for q in P if not q["inside"]])), st.median(map(frac, [q for q in P if q["inside"]]))
    print(f"L3 plain |K−K*| 平均 {d:.3f} ≤ 0.15 {'✅' if d <= .15 else '❌'}；K 随 |e| 降 {'✅' if mono_e else '❌'}；r 增大 K 不降 {'✅' if mono_r else '❌'}；靠拢 外 {fo:.2f} − 内 {fi:.2f} = {fo - fi:+.2f} ≥ 0.2 {'✅' if fo - fi >= .2 else '❌'}")
    a, b = T["rule", 0.0, 8.0], T["plain", 0.0, 8.0]
    print(f"L4 安慰剂 rule 没动 {a['still']:.0%} ≥ 90% {'✅' if a['still'] >= .9 else '❌'}；rule K {a['K']:.1%} ≥ 97% {'✅' if a['K'] >= .97 else '❌'}；plain K {b['K']:.1%} ≥ 95% {'✅' if b['K'] >= .95 else '❌'}")
    a, b = T["plain", 3.0, 0.25], T["peers", 3.0, 0.25]
    print(f"L5 错集合 plain K {a['K']:.1%} ≤ 30% {'✅' if a['K'] <= .3 else '❌'}；peers K {b['K']:.1%} ≤ 50% {'✅' if b['K'] <= .5 else '❌'}")
    a, b = T["peers", 0.0, 0.25], T["peers", 0.0, 8.0]
    print(f"L6 对集合 peers K {a['K']:.1%} ≥ 95% {'✅' if a['K'] >= .95 else '❌'}；B {a['B']:.1%} ≥ 80% {'✅' if a['B'] >= .8 else '❌'}；比安慰剂 {b['B']:.1%} 高 {a['B'] - b['B']:+.3f} ≥ 0.15 {'✅' if a['B'] - b['B'] >= .15 else '❌'}")
    print("\n按模型（只报告）：错集合 e/h=3 r/h=0.25 的 K ｜对集合 e=0 r/h=0.25 的 B")
    for arm in L.ARMS:
        print(" ", arm, " ".join(f"{a[:9]}:{met([q for q in R if q['arm'] == arm and q['a'] == a and q['e'] == 3.0 and q['r'] == .25])['K']:.0%}|{met([q for q in R if q['arm'] == arm and q['a'] == a and q['e'] == 0.0 and q['r'] == .25])['B']:.0%}" for a in A))


if __name__ == "__main__":
    main()
