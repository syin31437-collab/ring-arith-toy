"""消融：H 曲线的四个量（ρ、κ、λ、q）各拿掉一个，在同样的玩具上重新拟合，看在哪些评估集上变差。

拟合：het2 + 时间 + 代数玩具（种子 0–39 留出），和 failmap.curve 相同；只改特征。
评估（漏检率 MAE，越小越好）：
  基线    压力玩具无压力 160 个配置（新种子 abl）
  厚尾    df = 1，80 个；δ 分散 σ_log = 1.5，80 个；节点多 N = 31，80 个；节点少 N = 3，80 个
  投毒    真实 LLM 投毒实验 14 个配置
  盲测二  人口密度 Z0 的 4 个配置；盲测三  行程 R0 的 5 个配置

预注册（11-01，运行前）：
  1. 完整版 H 在所有评估集上的平均 MAE 最小
  2. 拿掉 ρ 是对两次真实盲测伤害最大的单项
  3. 拿掉 q 是对 δ 分散伤害最大的单项
  4. 拿掉 λ 是对投毒实验和 N = 31 伤害最大的单项
  5. 拿掉 κ 在每个评估集上的 MAE 变化都 ≤ 0.02

用法：~/gut-external/torch215-venv/bin/python ablation.py
"""
import glob
import json
import os
import statistics as st
import sys

import torch

from failmap import stress_toy
from hetero import feats, quad, real

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["ρ", "κ", "λ", "q"]
VARIANTS = {"完整 H": [0, 1, 2, 3], "−ρ": [1, 2, 3], "−κ": [0, 2, 3], "−λ": [0, 1, 3], "−q": [0, 1, 2], "只 ρ": [0], "只 λ": [2]}


def train_rows(judge):
    rows = json.load(open(os.path.join(HERE, f"het2_{judge}.json")))
    for env in ("geo", "alg"):
        for f in sorted(glob.glob(os.path.join(HERE, f"tr_{env}_{judge}_*.json"))):
            rows += [r for r in json.load(open(f)) if r["sigma_eff"] > 0 and r["sigma_node"] > 0 and "sigma_min_r" in r and r["seed"] >= 40]
    return rows


def X(rows, idx):
    return torch.tensor([quad([feats(r)["H"][i] for i in idx]) for r in rows], dtype=torch.float64)


def fit(rows, idx):
    A, y = X(rows, idx), torch.tensor([min(1.0, max(0.0, r["miss"])) for r in rows], dtype=torch.float64)
    w = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((torch.sigmoid(A @ w) - y) ** 2).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def blind_rows(judge, v):
    llm = os.path.join(HERE, "..", "llm")
    sys.path.insert(0, llm)
    cwd = os.getcwd()
    if v == 2:
        os.environ["BLIND"] = "2"
        import blind as mod
        z = "Z0"
    else:
        import blind3 as mod
        z = "R0"
    R = [r for r in mod.features(judge) if r["z"] == z]
    os.chdir(cwd)
    return R


def main():
    torch.manual_seed(5)
    for judge in ("软", "硬"):
        evals = {"基线": [stress_toy(k, judge, tag="abl") for k in range(160)],
                 "厚尾": [stress_toy(k, judge, tag="abl", df=1) for k in range(80)],
                 "δ分散": [stress_toy(k, judge, tag="abl", dvar=1.5) for k in range(80)],
                 "N=31": [stress_toy(k, judge, tag="abl", n=31) for k in range(80)],
                 "N=3": [stress_toy(k, judge, tag="abl", n=3) for k in range(80)],
                 "投毒": real(judge), "盲测二": blind_rows(judge, 2), "盲测三": blind_rows(judge, 3)}
        T = train_rows(judge)
        print(f"\n== {judge}裁判：漏检率 MAE ==")
        print("变体      " + "".join(f"{k:>8s}" for k in evals) + "     平均")
        for name, idx in VARIANTS.items():
            w = fit(T, idx)
            m = [st.mean(abs(p - min(1, max(0, r["miss"]))) for p, r in zip(torch.sigmoid(X(R, idx) @ w).tolist(), R)) for R in evals.values()]
            print(f"{name:8s}" + "".join(f"{x:8.3f}" for x in m) + f"   {st.mean(m):6.3f}")


if __name__ == "__main__":
    main()
