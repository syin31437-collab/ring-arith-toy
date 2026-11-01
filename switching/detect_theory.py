"""沟要挖多深、几轮出局：卧底 / 间歇下毒者的出局轮数，从递推式直接算（零拟合，不用真实数据）（2026-11-01）。

下毒后每轮相对秤的标准化偏移 μ = 毒量 / √(节点噪声² + 秤噪声²)，毒量 = log10(1 + δ)，噪声都从校准期估（可观测）。
线：S ← 0.7·S + 0.3·(μ·[这轮下毒] + N(0,1))，起点 = 平稳的 N(0, SD_S²)（卧底之前老实了很久），越过 3.3·SD_S 出局。
"这轮下毒"：卧底 = 每轮；间歇 = 以 30% 概率。对这个递推做 20000 次蒙特卡洛，取出局轮数的中位数——这一步只依赖 μ。
已知结果（盲测三卧底：δ = 5% 第 9 轮、δ = 10% 第 3 轮）只作一致性检查，不计分。

预注册（运行前；新场景的 purge 模拟都还没跑过）：
  新场景：盲测三卧底 δ = 2%、20%；盲测三间歇 δ = 20%；盲测二（约束 = 模型记忆，更准）卧底 δ = 5%、10%
  每个新场景：|模拟的中位出局轮数 − 公式| ≤ max(2 轮, 公式的 30%)；
  公式说 60 轮内中位抓不到的场景，模拟里被抓到的比例也应 < 50%

用法：~/gut-external/torch215-venv/bin/python detect_theory.py
"""
import math
import random
import statistics as st

import torch

from analytic import sigma_c
from purge import C, NODES, SD_S, W, load, noise, simulate


def theory(mu, q, n=20000, horizon=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = SD_S * torch.randn(n, generator=g)
    hit = torch.full((n,), horizon + 1)
    for m in range(1, horizon + 1):
        pois = (torch.rand(n, generator=g) < q).float()
        S = (1 - W) * S + W * (mu * pois + torch.randn(n, generator=g))
        new = (S.abs() > C * SD_S) & (hit > horizon)
        hit[new] = m
    med = hit.float().median().item()
    return med, (hit <= horizon).float().mean().item()


def main():
    cases = [(3, 0.05, "S1", True), (3, 0.1, "S1", True), (3, 0.02, "S1", False), (3, 0.2, "S1", False),
             (3, 0.2, "S3", False), (2, 0.05, "S1", False), (2, 0.1, "S1", False)]
    print("数据   δ     场景   μ      公式：中位出局轮 / 60 轮内抓到    模拟：中位出局轮 / 抓到        计分")
    for v, d, scen, known in cases:
        clean, bad = load(d, v)
        _, sc = sigma_c(clean)
        sig = noise(clean)
        model = 2 if scen == "S1" else 6                         # S1 卧底 = ds_flash_a；S3 间歇 = ds_pro_b
        mu = math.log10(1 + d) / math.sqrt(sig[model].item() ** 2 + sc ** 2)
        q = 1.0 if scen == "S1" else 0.3
        tm, tp = theory(mu, q)
        delays = []
        for o in range(50):
            order = list(range(len(clean)))
            random.Random(o).shuffle(order)
            rec = simulate(clean, bad, order, scen, True, True, random.Random(1000 + o))
            kind = "sleeper" if scen == "S1" else "intermittent"
            hit = next((k for k, _, kk in rec["removed"] if kk == kind), None)
            delays.append(None if hit is None else (hit - 31 if scen == "S1" else hit))
        got = [x for x in delays if x is not None]
        sm = st.median(got) if len(got) >= 25 else float("nan")
        horizon_note = "（下毒后 30 轮内）" if scen == "S1" else ""
        tag = "一致性检查" if known else ("✅" if (math.isnan(sm) and tp < 0.5) or (not math.isnan(sm) and abs(sm - tm) <= max(2, 0.3 * tm)) else "❌")
        print(f"盲测{'二三'[v - 2]} {d:4.0%}  {scen}   {mu:5.2f}   {tm:6.0f} / {tp:4.0%}                       {sm:6.0f} / {len(got)}/50{horizon_note}    {tag}")


if __name__ == "__main__":
    main()
