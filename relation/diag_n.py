"""新节点比赛的诊断（看过结果之后做的，全部算探索，不改 analysis_n.txt 的判决）。

A 敏感性：qwen14 的第一次答案有 35% 把题号 Q961 写成 961，被解析器丢掉 → 把键补全后重算（官方结果里它按预注册剔除）。
B 第一次对 / 第一次错的节点分开算 log-loss：大模型错题很少，总分基本只在比"不被带偏"。
C PCA2 和 F4、F4w 配对比较。
D 前两个主成分的载荷，以及所有节点在上面的坐标。
E N4 的灵敏度：可加预测已经接近 100% 的节点，暴露量没有上升空间。
F 旧节点留一法（题目切分同 generality.py）下 k = 1,2,3,5：两个数够不够。
G 主成分含义的留出检查：坐标只用校准题和校准投放，行为只看预测题和节点没见过的投放。
"""
import glob
import json
import os
import random
import shutil

import torch

import analyze as AN
import formula as FM
import generality as G
import newnodes as N
import stage2 as S2

torch.set_num_threads(4)
SC = os.path.join(N.SCR, "fixed")
shutil.rmtree(SC, ignore_errors=True)
os.makedirs(SC)
ids = {x["id"] for x in json.load(open(N.ITEMS))}
fixed = 0
for fn in glob.glob(os.path.join(N.OUT, "*")):
    b = os.path.basename(fn)
    if b.endswith("_qwen14.json"):
        J = json.load(open(fn))
        A = {}
        for k, v in J["answers"].items():
            if k not in ids and "Q" + k in ids:
                k = "Q" + k
                fixed += 1
            A[k] = v
        J["answers"] = A
        json.dump(J, open(os.path.join(SC, b), "w"))
    else:
        os.symlink(fn, os.path.join(SC, b))
N.OUT = SC
print(f"qwen14 补全题号 {fixed} 处")
R, cover, tools, items = N.load()
G.items.update(items)
FM.items.update(items)
print("覆盖", {k: round(v, 2) for k, v in cover.items()})
nodes = [a for a in N.ALL if cover.get(a, 0) >= .8]
train = G.NODES
mu, Sig, phi = G.fit_pop(train)
muw, Sigw, phiw = N.fit_popw(train, phi)
B = torch.stack([G.fit_lin(G.basis(G.tens(G.BY[n])), G.tens(G.BY[n])["y"]) for n in train])
m0 = B.mean(0)
U, Sv, Vh = torch.linalg.svd(B - m0, full_matrices=False)


def pca(k, dc, dt, Vh_=None, Sv_=None, m_=None, n_=None):
    Vh_, Sv_, m_, n_ = (Vh, Sv, m0, len(train)) if Vh_ is None else (Vh_, Sv_, m_, n_)
    Vk, sk = Vh_[:k], Sv_[:k] ** 2 / (n_ - 1) + .1
    Pc, Pt = G.basis(dc), G.basis(dt)
    z = torch.zeros(k, requires_grad=True)
    opt = torch.optim.LBFGS([z], max_iter=200, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(Pc @ (m_ + z @ Vk), dc["y"]) + .5 * (z ** 2 / sk).sum() / len(dc["y"])
        l.backward()
        return l
    opt.step(cl)
    return torch.softmax(Pt @ (m_ + z.detach() @ Vk), 1), z.detach()


MS = ["F4", "F4w", "PCA2", "PCA5", "共用"]
per, res, zs, rowsof = {m: {} for m in MS}, {}, {}, {}
expose = {}
for a in nodes:
    te = [r for r in R if r["a"] == a and not r["cal"] and r["p"] in G.TEST_P]
    cc = [r for r in R if r["a"] == a and r["cal"] and r["p"] in G.CAL_P]
    dt, dc = G.tens(te), G.tens(cc)
    fn = lambda t, d: G.f4(t, phi, d)
    fw = lambda t, d: N.f4w(t, phiw, d)
    th, thw = G.map_fit(dc, mu, Sig, fn, 5), G.map_fit(dc, muw, Sigw, fw, 5)
    q2, z2 = pca(2, dc, dt)
    q5, _ = pca(5, dc, dt)
    zs[a] = z2.tolist()
    qs = {"F4": fn(th, dt), "F4w": fw(thw, dt), "PCA2": q2, "PCA5": q5, "共用": fn(mu, dt)}
    idx = torch.arange(len(te))
    for m, q in qs.items():
        per[m][a] = (-q.clamp(min=1e-6).log()[idx, dt["y"]]).detach()
    res[a] = {m: per[m][a].mean().item() for m in MS}
    rowsof[a] = te
    qa = G.f4(th, phi, dt, additive=True)
    ri = [i for i, r in enumerate(te) if r["p"] == "set2f_r" and r["right"]]
    obs = sum(te[i]["ok"] for i in ri) / len(ri)
    pred = sum(qa[i, te[i]["g"]].item() for i in ri) / len(ri)
    expose[a] = (obs, pred, th[4].item(), len(ri))

print("\nA/B 按第一次对错分开的 log-loss（预测部分；错节点 = 第一次答错的行）")
print(f"{'节点':14s}{'行:对':>7s}{'行:错':>7s}" + "".join(f"  对:{m:6s}" for m in MS) + "".join(f"  错:{m:6s}" for m in MS))
for a in nodes:
    te = rowsof[a]
    ra = [i for i, r in enumerate(te) if r["right"]]
    wa = [i for i, r in enumerate(te) if not r["right"]]
    f = lambda m, I: per[m][a][I].mean().item() if I else float("nan")
    print(f"{a:14s}{len(ra):7d}{len(wa):7d}" + "".join(f"{f(m, ra):10.3f}" for m in MS) + "".join(f"{f(m, wa):10.3f}" for m in MS))
print("A 全部 %d 节点平均 " % len(nodes) + " ".join(f"{m} {sum(res[a][m] for a in nodes) / len(nodes):.3f}" for m in MS))


def compare(ns, m1, m2, part=None):
    by = {}
    for a in ns:
        for k, r in enumerate(rowsof[a]):
            if part is None or r["right"] == part:
                by.setdefault(r["id"], []).append((per[m1][a][k] - per[m2][a][k]).item())
    I = sorted(by)
    g = random.Random(3)
    bs = sorted(sum(x := [v for i in g.choices(I, k=len(I)) for v in by[i]]) / len(x) for _ in range(1000))
    return sum(v for i in I for v in by[i]) / sum(len(by[i]) for i in I), bs[25], bs[974]


print("\nC 配对差（负 = 前者好；区间按题 bootstrap）")
for ns, lab in ((nodes, "全部"), ([a for a in nodes if a in N.LLMS], "LLM")):
    for x, y in (("PCA2", "F4"), ("PCA2", "F4w"), ("PCA2", "PCA5"), ("F4w", "F4")):
        print(f"  {lab} {x} − {y}: {compare(ns, x, y)[0]:+.4f} [{compare(ns, x, y)[1]:+.4f},{compare(ns, x, y)[2]:+.4f}]；只第一次错的行 {compare(ns, x, y, False)[0]:+.3f}")

names = ["原答案", "原答案×置信度"] + [f"目标:{p}" for p in G.PL[:10]] + ["理由目标:" + G.PL[8], "理由目标:" + G.PL[9]]
print("\nD 平均（m0）与前两主成分载荷；方差占比", [round(x, 3) for x in (Sv ** 2 / (Sv ** 2).sum()).tolist()[:3]])
for j, nm in enumerate(names):
    print(f"  {nm:24s} 均值 {m0[j]:+6.2f}  PC1 {Vh[0][j]:+.2f}  PC2 {Vh[1][j]:+.2f}")
print("  坐标（旧节点直接投影；新节点只用校准）")
oldz = {n: ((B[i] - m0) @ Vh[:2].T).tolist() for i, n in enumerate(train)}
for n, z in sorted(oldz.items(), key=lambda kv: kv[1][0]):
    print(f"    旧 {n:14s} {z[0]:+7.2f} {z[1]:+7.2f}")
for n, z in sorted(zs.items(), key=lambda kv: kv[1][0]):
    print(f"    新 {n:14s} {z[0]:+7.2f} {z[1]:+7.2f}")

print("\nE N4 灵敏度（≥ +0.20 需要可加预测 ≤ 0.80）")
for a, (o, p, v, n) in expose.items():
    print(f"  {a:14s} n={n:3d} 观测 {o:.2f} 可加预测 {p:.2f} 暴露 {o - p:+.2f} 最大可能 {1 - p:+.2f} v {v:+.2f}")
x = [a for a in expose if expose[a][1] <= .8]
if len(x) >= 3:
    print("  只留有上升空间的节点", x, "Spearman %.2f" % S2.spearman([expose[a][2] for a in x], [expose[a][0] - expose[a][1] for a in x]))

print("\nF 旧节点留一法：k 个数（只用校准投放/校准题，预测其余）")
L = {k: {} for k in (1, 2, 3, 5)}
for i, a in enumerate(train):
    keep = [j for j in range(len(train)) if j != i]
    Bm = B[keep]
    m = Bm.mean(0)
    _, S_, V_ = torch.linalg.svd(Bm - m, full_matrices=False)
    c, t = G.split(a)
    dc, dt = G.tens(c), G.tens(t)
    for k in L:
        q, _ = pca(k, dc, dt, V_, S_, m, len(keep))
        L[k][a] = (-q.clamp(min=1e-6).log()[torch.arange(len(t)), dt["y"]]).mean().item()
print("  平均 " + " ".join(f"k={k} {sum(L[k].values()) / len(train):.3f}" for k in L))
llm = [a for a in train if a not in ("emb", "lex", "lex_proj", "rand_proj")]
print("  只 LLM " + " ".join(f"k={k} {sum(L[k][a] for a in llm) / len(llm):.3f}" for k in L), "（节点 %d）" % len(llm))

print("\nG 坐标（校准）× 行为（预测部分，只看和第一次答案冲突的行）")
def ad(rs, ps):
    x = [r["adopt"] for r in rs if r["p"] in ps and r["conflict"]]
    return sum(x) / len(x) if x else float("nan")
tab = {}
for a in nodes:
    te = rowsof[a]
    s_ = (ad(te, {"set2t_r"}) + ad(te, {"set2f_r"})) / 2
    d_ = (ad(te, {"L4t"}) - ad(te, {"L4f"}) + ad(te, {"L2t"}) - ad(te, {"L2f"})) / 2
    acc = AN.mean(r["right"] for r in R if r["a"] == a and r["p"] == "L0t")
    tab[a] = (zs[a][0], zs[a][1], s_, d_, acc)
    print(f"  {a:14s} PC1 {zs[a][0]:+6.2f} PC2 {zs[a][1]:+6.2f} ｜ 采纳集合 {s_:.2f} 真假差 {d_:+.2f} 准确率 {acc:.2f}")
v = list(tab.values())
c = lambda i, j: S2.spearman([x[i] for x in v], [x[j] for x in v])
print(f"  Spearman（7 节点）：PC1×采纳集合 {c(0, 2):+.2f}；PC1×真假差 {c(0, 3):+.2f}；PC2×真假差 {c(1, 3):+.2f}；PC2×采纳集合 {c(1, 2):+.2f}；PC2×准确率 {c(1, 4):+.2f}")
