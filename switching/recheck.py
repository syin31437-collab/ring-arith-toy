"""修正：裁判不许偷看真值（2026-11-01）。

之前所有盲测、投毒实验、玩具里，裁判的宽度 s 都是用真值量的（盲测：MAD(约束 − 真值)；投毒实验：SD(预测 − 真值)；
玩具：生成数据时设定的真实约束噪声）。实际使用时拿不到真值。
改成可观测版：s = 无故障题上 (约束 − 报告中位数) 的 MAD，再扣掉报告中位数自己的方差（analytic.sigma_c）。
这里在全部已有真实数据上重算"实际漏检"，并用推导版第五版（同样用可观测的 s）重新预测，看结论变不变。

用法：~/gut-external/torch215-venv/bin/python recheck.py
"""
import math
import os
import statistics as st
import sys

import torch

from analytic import derive5, mad, sigma_c
from failmap import stress_toy
from hetero import measure

HERE = os.path.dirname(os.path.abspath(__file__))


def load(v):
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    if v == 2:
        os.environ["BLIND"] = "2"
        import blind as mod
        zs = ("Z0", "Zv", "Z1")
    elif v == 3:
        import blind3 as mod
        zs = ("R0", "Rv", "R1")
    else:
        import blind4 as mod
        zs = ("R0", "R1")
    data = mod.setup()
    os.chdir(cwd)
    return data, mod.DELTAS, zs


def main():
    for v in (2, 3, 4):
        data, deltas, zs = load(v)
        print(f"\n===== 盲测{'二三四'[v - 2]} =====")
        for z in zs:
            clean = data[(z, 0.0)]
            s_truth = max(mad([p for _, p, _, _ in clean]), 1e-6)
            s_obs = max(sigma_c(clean)[1], 1e-6)
            print(f"{z}：裁判宽度 偷看真值 {s_truth:.4f} → 不偷看 {s_obs:.4f}")
            for judge in ("软", "硬2"):
                cells = []
                for d in deltas[1:]:
                    rows = [(False, p, r, x) for _, p, r, x in clean] + data[(z, d)]
                    ld = math.log10(1 + d)
                    a_t = measure(rows, ld, judge, s_truth)["miss"]
                    a_o = measure(rows, ld, judge, s_obs)["miss"]
                    cells.append(f"δ={d:.0%} {a_t:.0%}→{a_o:.0%}（推导 {derive5(rows, judge, s_obs):.0%}）")
                print(f"   {judge}：实际漏检 偷看→不偷看：" + "  ".join(cells))
    torch.manual_seed(18)
    print("\n===== 玩具（新种子 recheck，120 个配置）：裁判用真实约束噪声 vs 可观测估计 =====")
    for judge in ("软", "硬2"):
        diff, wr = [], []
        for k in range(120):
            rows, delta, sc = stress_toy(k, judge, tag="recheck", raw=True)
            s_obs = max(sigma_c([x for x in rows if not x[0]])[1], 1e-9)
            a, b = measure(rows, delta, judge, sc)["miss"], measure(rows, delta, judge, s_obs)["miss"]
            diff.append(b - a)
            wr.append(s_obs / sc)
        print(f"  {judge}：漏检变化 平均 {st.mean(diff):+.3f}、|变化| 平均 {st.mean(abs(x) for x in diff):.3f}；可观测宽度 / 真实 中位 {st.median(wr):.2f}（{min(wr):.2f}–{max(wr):.2f}）")


if __name__ == "__main__":
    main()
