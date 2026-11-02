"""蓝队：逐条复核红队的硬发现（不采信，自己算）。"""
import csv, json, math, os, random, statistics as st, sys
R = os.path.dirname(os.path.abspath(__file__)) + "/.."
for d in ("switching", "theory", "bipm", "slqa", "llm", "."):
    sys.path.insert(0, os.path.join(R, d))
import torch

print("=== V1 单秤归因：保留符号（Codex #1）")
from ladder2 import bipm_windows, slqa_windows, med
for name, W, ds in (("SLQA", list(slqa_windows()), (0.05, 0.1, 0.2)), ("BIPM", list(bipm_windows()), (6.0, 12.0, 24.0))):
    for d in ds:
        ok_fix, ok_rand = [], []
        rg = random.Random(0)
        for cal, ev in W:
            n = len(cal[0][2])
            c0 = st.median(med(x[2]) - x[0] for x in cal)
            e0 = st.median(med(x[2]) - x[0] for x in ev)
            # 注入方向固定为 +：节点全体 +d → 共识−A 变 +；秤 A +d → 变 −
            ok_fix += [e0 - c0 + d > 0, e0 - c0 - d < 0]
            # 注入方向未知：每窗随机 ±，同样按符号判
            s = rg.choice((-1, 1)); ok_rand.append(e0 - c0 + s * d > 0)
            s = rg.choice((-1, 1)); ok_rand.append(e0 - c0 - s * d < 0)
        print(f"  {name} δ={d:g}：方向已知 {st.mean(ok_fix):.0%}；方向未知 {st.mean(ok_rand):.0%}")

print("=== V2 约束 AUC：相对误差 > 20% 的字面定义（Codex #2）+ 去掉面积争议国（子代理 #1）")
import constraint as K
A = K.load(); pop, area = K.wb("SP.POP.TOTL"), K.wb("AG.SRF.TOTL.K2")
nodes = sorted(A)
rows = []
for i in sorted(i for i in pop if i in area):
    tr = {"pop": pop[i] / 1e6, "area": area[i], "dens": pop[i] / area[i]}
    have = [n for n in nodes if all(i in A[n][q] for q in tr)]
    if len(have) < 3: continue
    for n in have:
        res = abs(math.log10(A[n]["dens"][i]) - (math.log10(A[n]["pop"][i]) + 6 - math.log10(A[n]["area"][i])))
        rows.append(dict(i=i, res=res, w_log=max(abs(math.log10(A[n][q][i] / tr[q])) for q in tr) > math.log10(1.2), w_rel=max(abs(A[n][q][i] / tr[q] - 1) for q in tr) > 0.2, small=-pop[i]))
au = lambda key, sub: K.auc([r[key if False else "res"] for r in sub if r[key]], [r["res"] for r in sub if not r[key]])
print(f"  原口径 AUC {au('w_log', rows):.3f}（正例 {sum(r['w_log'] for r in rows)}）；字面相对误差 AUC {au('w_rel', rows):.3f}（正例 {sum(r['w_rel'] for r in rows)}）")
BAD = {"NOR", "HRV", "FRO", "BMU", "MCO", "GRL", "DOM", "CAN"}
sub = [r for r in rows if r["i"] not in BAD]
print(f"  去掉 8 国：原口径 AUC {au('w_log', sub):.3f}（正例 {sum(r['w_log'] for r in sub)}，来自 {len({r['i'] for r in sub if r['w_log']})} 国）")
print(f"  平凡信号'国家小'：AUC {K.auc([r['small'] for r in rows if r['w_log']], [r['small'] for r in rows if not r['w_log']]):.3f}；去 8 国后 {K.auc([r['small'] for r in sub if r['w_log']], [r['small'] for r in sub if not r['w_log']]):.3f}")

print("=== V3 三角帽 MAD²（Codex #4）：全部独立时的误拒率（N1 共有误差判定）")
from premise2 import common
def sim(dist, T=600, trials=40, n=9):
    out = []
    for s in range(trials):
        g = torch.Generator().manual_seed(s)
        X = 0.01 * torch.randn(T, n, generator=g)
        if dist == "normal":
            a, b = torch.randn(T, generator=g), torch.randn(T, generator=g)
        elif dist == "uniform":
            a, b = 2 * torch.rand(T, generator=g) - 1, 2 * torch.rand(T, generator=g) - 1
        else:
            a = torch.distributions.StudentT(2).sample((T,)); b = torch.distributions.StudentT(2).sample((T,))
        out.append(common(X, a.tolist(), b.tolist(), 0.1)[0])
    return st.mean(out)
torch.manual_seed(0)
print("  " + "；".join(f"{d}: 误拒 {sim(d):.0%}" for d in ("normal", "uniform", "t2")))

print("=== V4 诚实后验：复制数据不增信息却变确信（Codex #11）")
from law_v6 import fit
g = torch.Generator().manual_seed(1)
def rows_(T, f): return [(f, 0.1 * torch.randn(1, generator=g).item(), torch.randn(7, generator=g), 2.0) for _ in range(T)]
c, e = rows_(30, False), rows_(20, True)
e2 = [(f, p, r + torch.tensor([0.4, 0, 0, 0, 0, 0, 0]), x) for f, p, r, x in e]
print(f"  原始节点 0 诚实概率 {fit(c, e2)['ph'][0]:.3f}；同样 20 题复制 50 次 {fit(c, e2 * 50)['ph'][0]:.5f}")

print("=== V5 BIPM 自然格 vs 留一常数（子代理 #4）")
J = json.load(open(os.path.join(R, "bipm/validate_bipm_test.json")))
N = [r for r in J["cells"] if r["tag"] == "自然"]
for judge in ("软", "硬2"):
    for h in (1.5, 3.0):
        s = [r for r in N if r["judge"] == judge and r["h"] == h]
        loo = [st.mean(x["act"] for j, x in enumerate(s) if j != i) for i in range(len(s))]
        print(f"  {judge} h={h}：v6.1 MAE {st.mean(abs(r['pt'] - r['act']) for r in s):.3f}；留一常数 {st.mean(abs(l - r['act']) for l, r in zip(loo, s)):.3f}")
print(f"  总：v6.1 {st.mean(abs(r['pt'] - r['act']) for r in N):.3f}")

print("=== V6 Kimi 的双世界反例（无噪声）")
th, b = 10.0, 3.0
W1 = {"秤": th + b, "节点1": th, "节点2": th + b}
th2 = th + b
W2 = {"秤": th2, "节点1": th2 - b, "节点2": th2}
print(f"  世界 A 观测 {W1}；世界 B 观测 {W2}；相同 = {W1 == W2}（A 里节点 2 和秤偏，B 里节点 1 偏）")
