"""开发（非验证）：联邦学习第二轮那批已经看过的种子（3000+k）上，v6 的点预测、逐题重采样 vs 整轮重采样的区间覆盖和宽度。只跑 ζ = 0 的攻击。"""
import itertools, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flbench as F
import torch
from hetero import measure
from law_v6 import interval
attacks = ["ALIE0.5", "ALIE1", "ALIE2", "ALIE4", "noise", "flip", "flip-ζ"]
R = []
for k, (n, f, attack, m) in enumerate(itertools.product((10, 20), (0.2, 0.4, 0.6), attacks, (32, 512))):
    if attack == "flip-ζ":
        continue
    clean, bad, acc = F.run(n, f, attack, m, 3000 + k, dense=0.5, v2=True)
    bl = ([i // F.K for i in range(len(clean))], [i // F.K for i in range(len(bad))])
    pt, lo, hi, s0, rh = interval(clean, bad, "软", B=15)
    _, lo2, hi2, _, _ = interval(clean, bad, "软", B=15, blocks=bl)
    act = measure(clean + bad, bad[0][3], "软", s0)["miss"]
    R.append((attack, act, pt, lo, hi, lo2, hi2))
rk = lambda v: torch.tensor(v).argsort().argsort().float()
print(f"{len(R)} 格：MAE {st.mean(abs(r[1] - r[2]) for r in R):.3f}，点误差中位 {st.median(abs(r[1] - r[2]) for r in R):.3f}，Spearman {torch.corrcoef(torch.stack([rk([r[1] for r in R]), rk([r[2] for r in R])]))[0, 1]:.2f}")
print(f"逐题重采样：覆盖 {st.mean(r[3] <= r[1] <= r[4] for r in R):.0%}，宽度中位 {st.median(r[4] - r[3] for r in R):.2f}")
print(f"整轮重采样：覆盖 {st.mean(r[5] <= r[1] <= r[6] for r in R):.0%}，宽度中位 {st.median(r[6] - r[5] for r in R):.2f}")
for a in attacks[:-1]:
    s = [r for r in R if r[0] == a]
    print(f"  {a:8s} 实测 {st.mean(r[1] for r in s):.2f} 点 {st.mean(r[2] for r in s):.2f} |误差| {st.mean(abs(r[1] - r[2]) for r in s):.3f} 整轮覆盖 {st.mean(r[5] <= r[1] <= r[6] for r in s):.0%}")
