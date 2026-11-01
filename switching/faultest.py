"""故障测试：主动改变条件，看规律是否随之出现 / 消失，以及理论能不能如实预言。

曲线：H（现行，k_good 在无故障题上数）与 I（k_good 换成故障题上的诚实团大小 k_honest），都在 het2 + 时间 + 代数玩具上拟合，零重拟合。
每档 80 个随机配置（新种子 fault），误差 = 实际漏检 − 预测漏检（> 0 = 乐观 = 假通过）。
  F1  同样精确：投毒节点噪声 = 干净节点噪声（盲测第三版暴露的 k_good 漏洞）
  F2  拿掉约束：约束噪声放大 cmult 倍（1 → 1000）
  F3  毁掉独立：诚实节点在故障题上也被带偏 α·δ（α = 0 → 1）

预注册（11-01，运行前）：
  F1：same=True 时 H 的平均误差 ≥ +0.05（乐观）；I 在 same=False / True 时 |平均误差| 都 ≤ 0.05
  F2：cmult = 1000 时 实际漏检和投票漏检相差 ≤ 0.05（约束的好处消失）；I 在各档 |平均误差| ≤ 0.07（理论跟着预言消失）
  F3：实际漏检随 α 单调上升；α = 1 时 H 和 I 的平均误差都 ≥ +0.20——裁判只能从报告里选，
      所有节点都被带偏时没有正确答案可选；理论看不见这一点 → 第二条前提：至少要有一个节点不在故障里

用法：~/gut-external/torch215-venv/bin/python faultest.py
"""
import statistics as st

import torch

from failmap import curve, stress_toy
from hetero import predict

SWEEPS = {"F1 同样精确": ("same", [False, True]),
          "F2 约束噪声 ×": ("cmult", [1.0, 3.0, 10.0, 100.0, 1000.0]),
          "F3 诚实节点被带偏 α": ("couple", [0.0, 0.25, 0.5, 0.75, 1.0])}


def main():
    torch.manual_seed(4)
    for judge in ("软", "硬"):
        ws = {k: curve(judge, k, src="het2") for k in ("H", "I")}
        print(f"\n== {judge}裁判（每格 = 平均误差 / MAE） ==")
        print("干预              强度      实际    投票    k_honest      H               I")
        for name, (key, levels) in SWEEPS.items():
            for lv in levels:
                R = [stress_toy(k, judge, tag="fault", **{key: lv}) for k in range(80)]
                cells = []
                for kind, w in ws.items():
                    e = [r["miss"] - p for r, p in zip(R, predict(w, R, kind).tolist())]
                    cells.append(f"{st.mean(e):+6.2f} / {st.mean(abs(x) for x in e):4.2f}")
                print(f"{name:16s} {str(lv):>6s}   {st.mean(r['miss'] for r in R):5.0%}   {st.mean(r['vote'] for r in R):5.0%}"
                      f"   {st.mean(r['k_honest'] for r in R):6.2f}    " + "   ".join(cells))


if __name__ == "__main__":
    main()
