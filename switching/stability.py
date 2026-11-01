"""两件事（2026-11-01）：
一、玩具重跑：裁判不偷看真值（宽度 = 可观测的秤噪声估计），推导版第五版的结论会不会翻
  新种子 stab，每集 120 个配置；对照 = 同一批配置、裁判用真实秤噪声（旧做法）
二、规律稳不稳：真实数据（盲测二 Z0、盲测三 R0、盲测四 R0，各 δ 档）把题目有放回重抽 100 次，
  每次重算实际漏检和推导版预测，看 (实际 − 预测) 的 90% 区间

预注册（运行前）：
  1. 玩具七个集（无压力 / 约束×100 / 抱团 / 厚尾 / N=31 / δ 分散 / 系统偏差 1）里，至少 5 个集上，
     不偷看版的推导版 MAE 不比偷看版高 0.02 以上（结论不翻）
  2. 真实数据：至少 80% 的 (数据, δ 档) 的 (实际 − 预测) 90% 重抽区间包含 0
  3. 真实数据：区间宽度中位 ≤ 0.20

用法：~/gut-external/torch215-venv/bin/python stability.py
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


def part1():
    torch.manual_seed(20)
    sets = (("无压力", {}), ("约束×100", {"cmult": 100.0}), ("抱团", {"same": True}), ("厚尾df=1", {"df": 1}),
            ("N=31", {"n": 31}), ("δ分散1.5", {"dvar": 1.5}), ("系统偏差1", {"nbias": 1.0}))
    for judge in ("软", "硬2"):
        print(f"\n== 玩具重跑 {judge}（新种子 stab，120 个配置）：推导版 MAE 偷看 → 不偷看；实际漏检均值 偷看 → 不偷看 ==")
        for tag, kw in sets:
            eo, eu, ao, au = [], [], [], []
            for k in range(120):
                rows, delta, sc = stress_toy(k, judge, tag="stab", raw=True, **kw)
                so = max(sigma_c([x for x in rows if not x[0]])[1], 1e-9)
                mo, mu_ = measure(rows, delta, judge, sc)["miss"], measure(rows, delta, judge, so)["miss"]
                eo.append(abs(mo - derive5(rows, judge, sc)))
                eu.append(abs(mu_ - derive5(rows, judge, so)))
                ao.append(mo)
                au.append(mu_)
            flag = "✅" if st.mean(eu) <= st.mean(eo) + 0.02 else "❌"
            print(f"  {tag:10s} MAE {st.mean(eo):.3f} → {st.mean(eu):.3f} {flag}    实际漏检 {st.mean(ao):.0%} → {st.mean(au):.0%}")


def part2():
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    os.environ["BLIND"] = "2"
    import blind
    import blind3
    import blind4
    sets = []
    for name, mod, z in (("盲测二", blind, "Z0"), ("盲测三", blind3, "R0"), ("盲测四", blind4, "R0")):
        sets.append((name, mod.setup(), mod.DELTAS, z))
        os.chdir(cwd)
    print("\n== 真实数据重抽 100 次：(实际 − 推导版) 的 90% 区间 ==")
    covers, widths = [], []
    for name, data, deltas, z in sets:
        for judge in ("软", "硬2"):
            cells = []
            for d in deltas[1:]:
                clean, bad = data[(z, 0.0)], data[(z, d)]
                diffs = []
                for bt in range(100):
                    rg = random.Random(bt)
                    cl = [rg.choice(clean) for _ in clean]
                    bd = [rg.choice(bad) for _ in bad]
                    s = max(sigma_c(cl)[1], 1e-6)
                    rows = [(False, p, r, x) for _, p, r, x in cl] + bd
                    diffs.append(measure(rows, math.log10(1 + d), judge, s)["miss"] - derive5(rows, judge, s))
                diffs.sort()
                lo, hi = diffs[5], diffs[94]
                covers.append(lo <= 0 <= hi)
                widths.append(hi - lo)
                cells.append(f"δ={d:.0%} [{lo:+.2f}, {hi:+.2f}]")
            print(f"  {name} {z} {judge}：" + "  ".join(cells))
    print(f"  区间包含 0 的比例 {st.mean(covers):.0%}（预注册 ≥ 80%）；宽度中位 {st.median(widths):.2f}（预注册 ≤ 0.20）")


if __name__ == "__main__":
    part1()
    part2()
