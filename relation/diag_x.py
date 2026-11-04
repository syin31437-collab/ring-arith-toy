"""跨领域臂 A 的诊断（看过结果后做的探索，不改 analysis_xa.txt 的判决）。

T 工具使用：预注册说全禁工具，Grok、Codex 仍有带 cmds 的回复；剔除后重算。
P 置换 p、逐个去节点的范围（节点只有 9 个，单点能改判多少）。
F 假通过检查：预注册要报的简单量（校准准确率、真假差、集合采纳），analyze_a 没打印。
C 每类任务：每类 15 题，节点门槛 10/10 不可能满足，预注册的"每类分开"实际是空的；放宽到 3/3。
K case 没过预检的原因；把 case 放回去的敏感性。
S S 拆成 B、K 两半，A1 差在哪。
H 同一节点从难选择题到开放题，K、B 怎么移动。
"""
import json
import os
import random

import generality as G
import hardn as H
import newnodes as N
import stage2 as S2

z = json.load(open(H.AXES))["z"]
z["ds_flash_b"] = json.load(open("axes2.json"))["ds_flash_b"]
R = json.load(open("rows_xa.json"))
OLDH = json.load(open("rows_h.json"))
items = {x["id"]: x for x in json.load(open("itemsX.json"))}
ALL = "claude_sonnet kimi grok qwen14 qwen9 ds_flash_a ds_flash_b ds_pro_a claude_haiku codex_ns flash_m65".split()
m = lambda xs: (lambda v: sum(v) / len(v) if v else float("nan"))(list(xs))
sp = S2.spearman
OKF = ("code", "qa", "plan")


def SA(X):
    s = (S2.B(X, "L3t") + S2.B(X, "L4t")) / 2 + (S2.K(X, "L3f") + S2.K(X, "L4f")) / 2 - 1
    a = (S2.B(X, "set2t") + S2.B(X, "set2t_r") + 1 - S2.K(X, "set2f") + 1 - S2.K(X, "set2f_r")) / 4
    return s, a


def table(rs, mn=10):
    T = {}
    for a in ALL:
        X = [r for r in rs if r["a"] == a]
        nr, nw = sum(r["right"] for r in X if r["p"] == "blank"), sum(not r["right"] for r in X if r["p"] == "blank")
        if nr >= mn and nw >= mn:
            T[a] = SA(X)
    return T


def show(T, lab):
    q = list(T)
    zs = lambda i: [z[a][i] for a in q]
    S, A = [T[a][0] for a in q], [T[a][1] for a in q]
    print(f"  [{lab}] {len(q)} 节点 ρ(z2,S) {sp(zs(1), S):+.2f} ρ(z1,S) {sp(zs(0), S):+.2f}；ρ(z1,A) {sp(zs(0), A):+.2f} ρ(z2,A) {sp(zs(1), A):+.2f}")


Q = [r for r in R if r["family"] in OKF]

print("T 工具使用（回复里记录到命令的条数）")
tool = set()
for a in ALL:
    for it in items:
        for name in [f"s1_{it}_{a}.json"]:
            if os.path.exists("runs_x/" + name) and json.load(open("runs_x/" + name)).get("cmds"):
                tool.add((it, a, "blank0"))
    for f in os.listdir("runs_x"):
        if f.endswith(f"_{a}.json") and not f.startswith("s1_"):
            if json.load(open("runs_x/" + f)).get("cmds"):
                tool.add((f[:-5].replace(f"_{a}", ""), a, ""))
from collections import Counter
print("  ", Counter(a for _, a, _ in tool))
print("   例子", [t[0] for t in sorted(tool) if t[1] == "grok"][:6])
bad = lambda r: any(t[1] == r["a"] and (t[0] == r["id"] or (r["id"] in t[0] and r["p"] in t[0])) for t in tool)
show(table(Q), "原样")
show(table([r for r in Q if not bad(r)]), "剔除带命令的行")

print("\nP 置换 p、逐个去节点")
T = table(Q)
q = list(T)


def perm(x, y, obs):
    g = random.Random(1)
    return sum(sp(x, g.sample(y, len(y))) >= obs for _ in range(5000)) / 5000


for lab, i, k in (("z2×S", 1, 0), ("z1×A", 0, 1)):
    x, y = [z[a][i] for a in q], [T[a][k] for a in q]
    o = sp(x, y)
    loo = {a: sp([z[b][i] for b in q if b != a], [T[b][k] for b in q if b != a]) for a in q}
    print(f"  {lab} ρ {o:+.2f} 置换 p {perm(x, y, o):.3f}；去一个的范围 {min(loo.values()):+.2f} ~ {max(loo.values()):+.2f}；" + " ".join(f"{a}{v:+.2f}" for a, v in loo.items()))

print("\nF 假通过检查（校准数据都来自这批开放题之前）")
N.OUT = os.path.join(N.SCR, "fixed_h")
N.ITEMS = os.path.join(H.HERE, "itemsN.json")
N.LLMS = H.NEW7[:5]
Rn, _, _, _ = N.load()
N.LLMS = H.LLMS
cal = {a: [r for r in Rn if r["a"] == a and r["cal"] and r["p"] in G.CAL_P] for a in H.NEW7}
for a in q:
    if a not in cal or not cal[a]:
        cal[a] = G.split(a)[0]


def ad(X, ps):
    return m(r["adopt"] for r in X if r["p"] in ps and r["conflict"])


Si = {a: dict(acc=m(r["right"] for r in cal[a] if r["p"] == "blank"), dcal=ad(cal[a], {"L3t"}) - ad(cal[a], {"L3f"}), acal=m(ad(cal[a], {p}) for p in ("set2t", "set2f"))) for a in q}
xacc = {a: m(r["right"] for r in Q if r["a"] == a and r["p"] == "blank") for a in q}
for a in q:
    print(f"  {a:14s} 校准准确 {Si[a]['acc']:.2f} 真假差 {Si[a]['dcal']:+.2f} 集合采纳 {Si[a]['acal']:.2f} ｜ 本批第一次对 {xacc[a]:.2f} ｜ S {T[a][0]:+.2f} A {T[a][1]:.2f}")
ys, ya = [T[a][0] for a in q], [T[a][1] for a in q]
print("  预测 S：z2 %+.2f；校准准确 %+.2f；真假差 %+.2f；本批准确（同批，非预测）%+.2f" % (sp([z[a][1] for a in q], ys), sp([Si[a]["acc"] for a in q], ys), sp([Si[a]["dcal"] for a in q], ys), sp([xacc[a] for a in q], ys)))
print("  预测 A：z1 %+.2f；校准集合采纳 %+.2f" % (sp([z[a][0] for a in q], ya), sp([Si[a]["acal"] for a in q], ya)))

print("\nC 每类任务（门槛放宽到 ≥3 对 ≥3 错，探索）")
for f in ("code", "qa", "plan", "case"):
    show(table([r for r in R if r["family"] == f], 3), f)

print("\nK case 预检")
for a in ALL:
    X = [r for r in R if r["a"] == a and r["family"] == "case"]
    print(f"  {a:14s} 第一次对 {sum(r['right'] for r in X if r['p'] == 'blank'):2d}/15 blank K {S2.K(X, 'blank'):.0%}")
show(table(R), "把 case 放回去")

print("\nS S 拆成两半")
for a in q:
    X = [r for r in Q if r["a"] == a]
    print(f"  {a:14s} z2 {z[a][1]:+.2f} 真B {(S2.B(X, 'L3t') + S2.B(X, 'L4t')) / 2:.2f} 假K {(S2.K(X, 'L3f') + S2.K(X, 'L4f')) / 2:.2f}")
print("  ρ(z2, 真B) %+.2f ρ(z2, 假K) %+.2f" % (sp([z[a][1] for a in q], [(S2.B([r for r in Q if r["a"] == a], "L3t") + S2.B([r for r in Q if r["a"] == a], "L4t")) / 2 for a in q]), sp([z[a][1] for a in q], [(S2.K([r for r in Q if r["a"] == a], "L3f") + S2.K([r for r in Q if r["a"] == a], "L4f")) / 2 for a in q])))

print("\nH 难选择题 → 开放题")
for a in ALL:
    O, X = [r for r in OLDH if r["a"] == a], [r for r in Q if r["a"] == a]
    if not O:
        continue
    print(f"  {a:14s} K(set2f) {S2.K(O, 'set2f'):.0%}→{S2.K(X, 'set2f'):.0%} K(L3f) {S2.K(O, 'L3f'):.0%}→{S2.K(X, 'L3f'):.0%} K(L4f) {S2.K(O, 'L4f'):.0%}→{S2.K(X, 'L4f'):.0%} B(L3t) {S2.B(O, 'L3t'):.0%}→{S2.B(X, 'L3t'):.0%} B(L4t) {S2.B(O, 'L4t'):.0%}→{S2.B(X, 'L4t'):.0%} B(blank) {S2.B(O, 'blank'):.0%}→{S2.B(X, 'blank'):.0%}")
hs, ha = table(OLDH), table(Q)
cq = [a for a in hs if a in ha]
print(f"  节点排序跨域的直接相关（难选择题 → 开放题，{len(cq)} 节点）：S {sp([hs[a][0] for a in cq], [ha[a][0] for a in cq]):+.2f} A {sp([hs[a][1] for a in cq], [ha[a][1] for a in cq]):+.2f}")
