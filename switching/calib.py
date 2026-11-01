"""校准硬裁判（2026-11-01）。

诊断：硬裁判在玩具和真实盲测里一直偏悲观，大 δ 时尤甚（盲测三实际 3 / 0%，预测 43 / 31%）。
原因在设计：剔除窗口 = 2.5 × 约束尺度，没算诚实节点自己的噪声。玩具里约束常比节点准 → 诚实节点也被剔除 →
全剔除退回中位数 → δ 很大也漏检，曲线学到了这个底；真实盲测里约束尺度比节点噪声宽，没有这个底。
修法（"硬2"）：窗口 = 2.5 × max(约束尺度, 最准节点偏离约束的 MAD)。曲线只在 het2 静态玩具上重新拟合（时间 / 代数玩具的裁判在循环里，不改）。

预注册（运行前；盲测二、三的数据已采集，但硬2 的结果还没看过 → 半盲）：
  1. 硬2 在盲测二 Z0、盲测三 R0 上 |实际 − 预测| 的平均 ≤ 0.10（旧硬裁判是 0.12 / 0.21）
  2. 硬2 的实际漏检在两次盲测 δ ≥ 10% 的各档都不高于旧硬裁判
  3. 硬2 在新种子压力玩具无压力基线上 MAE ≤ 旧硬裁判（H 曲线）的 MAE
  4. 硬2 在 ζ = 1（R1、Z1）上照样乐观 ≥ +0.30（校准不该让理论看见它原理上看不见的东西）

用法：~/gut-external/torch215-venv/bin/python calib.py
"""
import json
import os
import statistics as st
import sys

import torch

from failmap import curve, stress_toy
from hetero import predict, toy2

HERE = os.path.dirname(os.path.abspath(__file__))


def blind(v, judge):
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    if v == 2:
        os.environ["BLIND"] = "2"
        import blind as mod
    else:
        import blind3 as mod
    R = mod.features(judge)
    os.chdir(cwd)
    return R


def main():
    torch.set_num_threads(4)
    path = os.path.join(HERE, "het2_硬2.json")
    if not os.path.exists(path):
        json.dump([toy2(k, "硬2") for k in range(1200)], open(path, "w"))
    W = {"硬": curve("硬", "H", src="het2"), "硬2": curve("硬2", "H", src="het2")}
    torch.manual_seed(7)
    print("== 无压力基线（新种子 calib，160 个配置）：MAE ==")
    for j, w in W.items():
        R = [stress_toy(k, j, tag="calib") for k in range(160)]
        print(f"  {j}: {st.mean(abs(p - r['miss']) for p, r in zip(predict(w, R, 'H').tolist(), R)):.3f}   实际漏检平均 {st.mean(r['miss'] for r in R):.0%}")
    for v in (2, 3):
        print(f"\n== 盲测{'二' if v == 2 else '三'} ==")
        print("约束  δ      旧硬：实际 / 预测     硬2：实际 / 预测")
        A = {j: blind(v, j) for j in W}
        P = {j: predict(w, A[j], "H").tolist() for j, w in W.items()}
        for k, r in enumerate(A["硬"]):
            r2 = A["硬2"][k]
            print(f"{r['z']}   {r['δ']:4.0%}    {r['miss']:5.0%} / {P['硬'][k]:4.0%}        {r2['miss']:5.0%} / {P['硬2'][k]:4.0%}")


if __name__ == "__main__":
    main()
