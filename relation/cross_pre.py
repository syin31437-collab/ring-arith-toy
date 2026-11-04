"""跨领域检验前的映射预检（不接触预测用的题）。

A 开放任务没有选项，算不了采纳率 → 用只靠对错的替代量：
   S = (B(L3t)+B(L4t))/2 + (K(L3f)+K(L4f))/2 − 1；A = [B(set2t)+B(set2t_r)+(1−K(set2f))+(1−K(set2f_r))]/4
   在已有的难题数据 rows_h.json 上看它们和 D_w、A_set 是否一致。
B 非 LLM 节点：Vehicle 数据（4 类）的训练折内部再切，看准确率、真假投放是否起作用。
"""
import json

import stage2 as S2

R = json.load(open("rows_h.json"))
m = lambda xs: (lambda v: sum(v) / len(v) if v else float("nan"))(list(xs))
nodes = sorted({r["a"] for r in R})
T = {}
for a in nodes:
    X = [r for r in R if r["a"] == a]
    cw = lambda p: m(r["adopt"] for r in X if r["p"] == p and not r["right"] and r["conflict"])
    dw = (cw("L3t") - cw("L3f") + cw("L4t") - cw("L4f")) / 2
    aset = m(m(r["adopt"] for r in X if r["p"] == p and r["conflict"]) for p in ("set2t", "set2f", "set2t_r", "set2f_r"))
    s = (S2.B(X, "L3t") + S2.B(X, "L4t")) / 2 + (S2.K(X, "L3f") + S2.K(X, "L4f")) / 2 - 1
    aa = (S2.B(X, "set2t") + S2.B(X, "set2t_r") + 1 - S2.K(X, "set2f") + 1 - S2.K(X, "set2f_r")) / 4
    T[a] = (dw, aset, s, aa)
    print(f"{a:14s} D_w {dw:+.2f} S {s:+.2f} ｜ A_set {aset:.2f} A {aa:.2f}")
print("Spearman D_w×S %.2f；A_set×A %.2f" % (S2.spearman([T[a][0] for a in nodes], [T[a][2] for a in nodes]), S2.spearman([T[a][1] for a in nodes], [T[a][3] for a in nodes])))
ax = json.load(open("axes.json"))["z"]
print("冻结坐标 z2×S %.2f；z1×A %.2f；z1×S %.2f；z2×A %.2f" % (S2.spearman([ax[a][1] for a in nodes], [T[a][2] for a in nodes]), S2.spearman([ax[a][0] for a in nodes], [T[a][3] for a in nodes]), S2.spearman([ax[a][0] for a in nodes], [T[a][2] for a in nodes]), S2.spearman([ax[a][1] for a in nodes], [T[a][3] for a in nodes])))
