"""推导版输出区间：ρ < 1 时说"预测不了多准"，给区间而不是一个点（2026-11-01）。

可观测的 ρ̂ = 跨故障题汇总后，偏离约束中心最远的节点的平均偏移 / 秤噪声（不用真值）。
区间 = 推导版第五版在题目有放回重抽 50 次上的 [5%, 95%] 分位，再各向外扩 1.645 × √(p(1 − p) / 故障题数)
  （实际漏检本身只是 n 道题上的估计，有二项抽样误差）。约束越弱、谁诚实越认不稳 → 重抽之间差得越多 → 区间自动变宽。无手调常数。

预注册（运行前；真实数据看过、但区间从没算过 → 半盲；玩具用新种子 intv）：
  1. 真实数据（盲测二 Z0、盲测三 R0、盲测四 R0，各 δ 档，软 / 硬2，共 28 格）：实际漏检落在区间内的比例 ≥ 90%
  2. ρ̂ ≥ 1 的格子区间宽度中位 ≤ 0.15（有信息量）；ρ̂ < 1 的格子宽度中位 > ρ̂ ≥ 1 的
  3. 玩具无压力、约束 ×100 两个集（各 40 个配置）：覆盖率 ≥ 85%

用法：~/gut-external/torch215-venv/bin/python intervals.py
"""
import math
import os
import random
import statistics as st
import sys

import torch

from analytic import derive5, sigma_c
from failmap import stress_toy
from hetero import measure

HERE = os.path.dirname(os.path.abspath(__file__))


def rho_hat(clean, bad):
    b, sc = sigma_c(clean)
    Y = torch.stack([r - (p - b) for _, p, r, _ in bad])
    return Y.mean(0).abs().max().item() / max(sc, 1e-9)


def interval(clean, bad, judge, B=50):
    s = max(sigma_c(clean)[1], 1e-6)
    point = derive5([(False, p, r, x) for _, p, r, x in clean] + bad, judge, s)
    bs = []
    for bt in range(B):
        rg = random.Random(bt)
        cl = [rg.choice(clean) for _ in clean]
        bd = [rg.choice(bad) for _ in bad]
        sb = max(sigma_c(cl)[1], 1e-6)
        bs.append(derive5([(False, p, r, x) for _, p, r, x in cl] + bd, judge, sb))
    bs.sort()
    e = 1.645 * math.sqrt(max(point * (1 - point), 1e-4) / len(bad))
    return point, max(0.0, bs[int(0.05 * B)] - e), min(1.0, bs[int(0.95 * B) - 1] + e), s


def real():
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    os.environ["BLIND"] = "2"
    import blind
    import blind3
    import blind4
    out = []
    for name, mod, z in (("盲测二", blind, "Z0"), ("盲测三", blind3, "R0"), ("盲测四", blind4, "R0")):
        data = mod.setup()
        os.chdir(cwd)
        for d in mod.DELTAS[1:]:
            clean, bad = data[(z, 0.0)], data[(z, d)]
            rh = rho_hat(clean, bad)
            for judge in ("软", "硬2"):
                pt, lo, hi, s = interval(clean, bad, judge)
                act = measure([(False, p, r, x) for _, p, r, x in clean] + bad, math.log10(1 + d), judge, s)["miss"]
                out.append((name, d, judge, rh, act, pt, lo, hi))
    return out


def main():
    R = real()
    print("数据    δ     裁判   ρ̂      实际    点预测   区间            落在内？")
    for name, d, j, rh, a, pt, lo, hi in R:
        print(f"{name}  {d:4.0%}   {j:3s}  {rh:5.2f}   {a:5.0%}   {pt:5.0%}   [{lo:4.0%}, {hi:4.0%}]   {'✅' if lo <= a <= hi else '❌'}")
    cov = st.mean(lo <= a <= hi for *_, a, pt, lo, hi in R)
    wide = [hi - lo for _, _, _, rh, a, pt, lo, hi in R if rh >= 1]
    narrow = [hi - lo for _, _, _, rh, a, pt, lo, hi in R if rh < 1]
    print(f"\n覆盖率 {cov:.0%}（预注册 ≥ 90%）；ρ̂ ≥ 1 的宽度中位 {st.median(wide) if wide else float('nan'):.2f}（{len(wide)} 格，预注册 ≤ 0.15）；"
          f"ρ̂ < 1 的宽度中位 {st.median(narrow) if narrow else float('nan'):.2f}（{len(narrow)} 格）")
    torch.manual_seed(21)
    for tag, kw in (("无压力", {}), ("约束×100", {"cmult": 100.0})):
        cov = []
        for k in range(40):
            rows, delta, sc = stress_toy(k, "软", tag="intv", raw=True, **kw)
            clean = [x for x in rows if not x[0]]
            bad = [x for x in rows if x[0]]
            pt, lo, hi, s = interval(clean, bad, "软", B=30)
            act = measure(rows, delta, "软", s)["miss"]
            cov.append(lo <= act <= hi)
        print(f"玩具 {tag}（软，40 个配置）：覆盖率 {st.mean(cov):.0%}（预注册 ≥ 85%）")


if __name__ == "__main__":
    main()
