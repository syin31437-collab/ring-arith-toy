"""纠错预测的诊断（看过结果后做的探索，不改 analysis_h.txt 的判决）。

A 同题配对的假理由暴露 K(set2f_r) − K(set2f) 和冻结坐标的关系（上一轮 N4 的两个缺陷：不同题、没有上升空间）。
B 同一节点在普通题（newnodes）和难题上的 K、B：难度把绝对值移动了多少。
C 按冻结的 z2 分两组（≥ 2 / < 2），每种投放的 K、B 和盈亏平衡真率 v*。
D 标记题和未标记题的 B、K（答案键可疑时纠错是否被低估）。
F 假通过检查：校准准确率、校准真假差、校准集合采纳这些简单量能否替代两条轴。
"""
import json

import hardn as H
import newnodes as N
import stage2 as S2

A = json.load(open(H.AXES))["z"]
R = json.load(open("rows_h.json"))
items = {x["id"]: x for x in json.load(open(H.ITEMS))}
old = json.load(open("rows_n.json"))
m = lambda xs: (lambda v: sum(v) / len(v) if v else float("nan"))(list(xs))
nodes = H.NODES

print("A 同题配对 K(set2f_r) − K(set2f)")
pair = {}
for a in nodes:
    X = {(r["id"], r["p"]): r for r in R if r["a"] == a and r["right"]}
    d = [X[(i, "set2f_r")]["ok"] - X[(i, "set2f")]["ok"] for i, p in X if p == "set2f" and (i, "set2f_r") in X]
    pair[a] = (m(d), len(d))
    print(f"  {a:14s} z2 {A[a][1]:+.2f} 配对差 {m(d):+.2f} n={len(d)}")
print("  Spearman(z2, 配对差) %.2f；只看 LLM %.2f" % (S2.spearman([A[a][1] for a in nodes], [pair[a][0] for a in nodes]), S2.spearman([A[a][1] for a in H.LLMS], [pair[a][0] for a in H.LLMS])))

print("\nB 普通题（newnodes 预测部分）→ 难题")
for a in H.NEW7:
    if a == "qwen14":
        continue
    Y = [r for r in old if r["a"] == a]
    X = [r for r in R if r["a"] == a]
    print(f"  {a:14s} K(L4f) {S2.K(Y, 'L4f'):.0%}→{S2.K(X, 'L4f'):.0%}  K(set2f_r) {S2.K(Y, 'set2f_r'):.0%}→{S2.K(X, 'set2f_r'):.0%}  K(set2f) {S2.K(Y, 'set2f'):.0%}→{S2.K(X, 'set2f'):.0%}  B(L4t) {S2.B(Y, 'L4t'):.0%}→{S2.B(X, 'L4t'):.0%}  B(set2t_r) {S2.B(Y, 'set2t_r'):.0%}→{S2.B(X, 'set2t_r'):.0%}")

print("\nC 盈亏平衡真率 v*（真假投放按同形式配对；ε = 各节点在这批题上的第一次错误率）")
groups = {"z2 ≥ 2": [a for a in nodes if A[a][1] >= 2], "z2 < 2": [a for a in nodes if A[a][1] < 2]}
pairs = (("L3t", "L3f"), ("L4t", "L4f"), ("set2t", "set2f"), ("set2t_r", "set2f_r"))
for g, ns in groups.items():
    print(f"  {g}: {ns}")
    for t, f in pairs:
        nts, nfs, bs, ks, kt, bf = [], [], [], [], [], []
        for a in ns:
            X = [r for r in R if r["a"] == a]
            e = m(not r["right"] for r in X if r["p"] == "blank")
            K0, B0 = S2.K(X, "blank"), S2.B(X, "blank")
            net = lambda p: e * (S2.B(X, p) - B0) - (1 - e) * (K0 - S2.K(X, p))
            nts.append(net(t))
            nfs.append(net(f))
            bs.append(S2.B(X, t))
            kt.append(S2.K(X, t))
            ks.append(S2.K(X, f))
            bf.append(S2.B(X, f))
        nt, nf = m(nts), m(nfs)
        v = -nf / (nt - nf) if nt > 0 > nf else float("nan")
        print(f"    {t:8s} B真 {m(bs):.0%} K真 {m(kt):.0%} K假 {m(ks):.0%} B假 {m(bf):.0%} 真投净 {nt:+.3f} 假投净 {nf:+.3f} v* {v:.2f}")

print("\nD 标记题 vs 未标记题")
for g, ns in groups.items():
    for fl in (False, True):
        X = [r for r in R if r["a"] in ns and items[r["id"]]["flagged"] == fl]
        print(f"  {g} {'标记' if fl else '未标记'} 第一次错 {m(not r['right'] for r in X if r['p'] == 'blank'):.0%} B(L4t) {S2.B(X, 'L4t'):.0%} B(set2t_r) {S2.B(X, 'set2t_r'):.0%} K(L4f) {S2.K(X, 'L4f'):.0%} K(set2f_r) {S2.K(X, 'set2f_r'):.0%}")

print("\nF 假通过检查：不用主成分，只用校准数据里的简单量（都来自难题之前）能不能预测得一样好")
import itertools
import os
import random

import generality as G

N.OUT = os.path.join(N.SCR, "fixed_h")
N.ITEMS = os.path.join(H.HERE, "itemsN.json")
N.LLMS = H.NEW7[:5]
Rn, _, _, _ = N.load()
N.LLMS = H.LLMS
cal = {a: [r for r in Rn if r["a"] == a and r["cal"] and r["p"] in G.CAL_P] for a in H.NEW7}
for a in H.OLD:
    cal[a] = G.split(a)[0]


def ad(X, ps, wrong=None):
    return m(r["adopt"] for r in X if r["p"] in ps and r["conflict"] and (wrong is None or r["right"] != wrong))


def stats(a):
    X = [r for r in R if r["a"] == a]
    cw = lambda p: m(r["adopt"] for r in X if r["p"] == p and not r["right"] and r["conflict"])
    return (cw("L3t") - cw("L3f") + cw("L4t") - cw("L4f")) / 2, m(ad(X, {p}) for p in ("set2t", "set2f", "set2t_r", "set2f_r"))


T = {a: stats(a) for a in nodes}
S = {a: dict(acc=m(r["right"] for r in cal[a] if r["p"] == "blank"), dcal=ad(cal[a], {"L3t"}) - ad(cal[a], {"L3f"}), acal=m(ad(cal[a], {p}) for p in ("set2t", "set2f")), z1=A[a][0], z2=A[a][1]) for a in nodes}
for a in nodes:
    print(f"  {a:14s} 校准准确率 {S[a]['acc']:.2f} 校准真假差 {S[a]['dcal']:+.2f} 校准集合采纳 {S[a]['acal']:.2f} ｜ z1 {S[a]['z1']:+.2f} z2 {S[a]['z2']:+.2f} ｜ 难题 D_w {T[a][0]:+.2f} A_set {T[a][1]:.2f}")


def perm_p(x, y, obs):
    g = random.Random(1)
    return sum(S2.spearman(x, g.sample(y, len(y))) >= obs for _ in range(5000)) / 5000


for lab, ns in (("全部 11", nodes), ("只 LLM 9", H.LLMS), ("LLM 去 Sonnet 8", [a for a in H.LLMS if a != "claude_sonnet"])):
    ys = [T[a][0] for a in ns]
    print(f"  [{lab}] 预测 D_w：" + "；".join(f"{k} {S2.spearman([S[a][k] for a in ns], ys):+.2f}" for k in ("z2", "acc", "dcal")) + f"（z2 的置换 p {perm_p([S[a]['z2'] for a in ns], ys, S2.spearman([S[a]['z2'] for a in ns], ys)):.3f}）")
    ys = [T[a][1] for a in ns]
    print(f"  [{lab}] 预测 A_set：" + "；".join(f"{k} {S2.spearman([S[a][k] for a in ns], ys):+.2f}" for k in ("z1", "acal")) + f"（z1 的置换 p {perm_p([S[a]['z1'] for a in ns], ys, S2.spearman([S[a]['z1'] for a in ns], ys)):.3f}）")
print(f"  z2 × 校准准确率 {S2.spearman([S[a]['z2'] for a in nodes], [S[a]['acc'] for a in nodes]):+.2f}；z1 × 校准集合采纳 {S2.spearman([S[a]['z1'] for a in nodes], [S[a]['acal'] for a in nodes]):+.2f}")
