"""补"抱团"变量（2026-11-01）。

来由：盲测第三版 δ = 5% 软裁判 +0.23、校准版硬裁判也在同一角落乐观。照盲测三条件复刻的玩具和现实几乎一样
（软 71 vs 70%、硬2 93 vs 90%），但同样特征上的 kNN 也预测错（29%）→ 缺变量，不是函数形式。
缺的是"故障团抱得多紧、占多少人"：ρ ≈ 1 时裁判近似按人头投票，抱团的多数就赢。
修法：特征 J = H + share（故障题上按最大缺口分两团后的大团占比，不用真值），拟合玩具 v3（一半配置是抱团攻击，投毒可多可少）。

预注册（运行前）。⚠️ 盲测三的数据已经看过，这里通过只是必要条件，之后还要新的盲测：
  1. J（het3 拟合）在盲测三 R0 上：软裁判每档 |实际 − 预测| ≤ 0.10；硬2 平均 ≤ 0.10
  2. J 在盲测二 Z0 上的平均 |误差| 不比 H 高 0.03 以上（软、硬2）
  3. 新种子压力玩具无压力基线上，J 的 MAE 不比 H 高 0.02 以上（软、硬2）
  4. 投毒实验漏检 R²（软）≥ 0.6

用法：~/gut-external/torch215-venv/bin/python cohere.py
"""
import json
import os
import statistics as st
import sys

import torch

from calib import blind
from failmap import curve, stress_toy
from hetero import predict, r2, real, toy3

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.set_num_threads(4)
    for j in ("软", "硬2"):
        path = os.path.join(HERE, f"het3_{j}.json")
        if not os.path.exists(path):
            json.dump([toy3(k, j) for k in range(1500)], open(path, "w"))
    torch.manual_seed(8)
    for j in ("软", "硬2"):
        W = {"H": curve(j, "H", src="het2"), "J": curve(j, "J", src="het3")}
        R = [stress_toy(k, j, tag="cohere") for k in range(160)]
        print(f"\n== {j} ==\n无压力基线 MAE：" + "  ".join(f"{k} {st.mean(abs(p - r['miss']) for p, r in zip(predict(w, R, k).tolist(), R)):.3f}" for k, w in W.items()))
        if j == "软":
            Rr = real(j)
            print("投毒实验漏检 R²：" + "  ".join(f"{k} {r2(Rr, predict(w, Rr, k), 'miss'):.2f}" for k, w in W.items()))
        for v, z in ((2, "Z0"), (3, "R0")):
            B = [r for r in blind(v, j) if r["z"] == z]
            P = {k: predict(w, B, k).tolist() for k, w in W.items()}
            print(f"盲测{'二' if v == 2 else '三'} {z}：" + "   ".join(f"δ={r['δ']:.0%} 实际 {r['miss']:.0%} H {P['H'][i]:.0%} J {P['J'][i]:.0%} (share {r['share']:.2f})" for i, r in enumerate(B)))
            print("   平均 |误差|：" + "  ".join(f"{k} {st.mean(abs(p - r['miss']) for p, r in zip(P[k], B)):.3f}" for k in W))


if __name__ == "__main__":
    main()
