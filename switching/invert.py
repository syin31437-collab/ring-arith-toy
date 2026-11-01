"""规律反着用：给它其余所有量，让它推出缺的那一个，再和真实情况比，看在不在推出的区间里（2026-11-01，库伯）。

A. 抓得多快 → 毒有多大
   观测：投毒节点（一直下毒的从第 1 轮、卧底从第 31 轮）到越线用了几轮（部署时可观测）；节点噪声、秤噪声（校准期可观测）。
   推：μ = 毒量 / √(节点噪声² + 秤噪声²)。对 μ 的网格做蒙特卡洛（EWMA 权重 0.3、单轮截到 ±3、门槛 3.3σ，
       一直下毒的从 S = 0 起、卧底从平稳分布起），μ 的 95% 区间 = 观测轮数落在 [2.5%, 97.5%] 分位内的所有 μ。
   真值：这个节点**实际**传出的毒（同一批题上 投毒轮报告 − 干净轮报告 的平均，除以同一个合噪声）——规律描述的就是这个量；
       LLM 有时不完全照资料走，实际 ≠ 名义，名义值另外报告。
   检测流程统一用第三轮的版本（留一法噪声、线扣节点常态偏移、截 ±3、逐题秤坏判断）。
B. 实际被骗率 → 秤有多准
   给推导版第五版除秤噪声 σ_c 以外的所有量，在 σ_c 网格上算预测被骗率，和实际被骗率（± 二项误差）对上的那段就是 σ_c 的区间。
   真值：秤相对真值的误差的稳健标准差（MAD）。正向时 σ_c 是拿秤和节点比出来的、不用真值；这里用的是被骗率（用了真值）→ 不循环。
   只用实际被骗率在 10%–90% 之间的格子（太接近 0 或 1 时被骗率对 σ_c 不敏感）。

预注册（运行前；数据都看过，但这两种反推从没算过）：
  A1. 实际传出毒量落在推出的 95% 区间内的比例 ≥ 85%
  A2. 推出的点值（观测轮数的中位对应的 μ）和实际传出毒量之比，|log| 的中位 ≤ log 1.5
  B1. 秤真实误差落在推出的 σ_c 区间内的比例 ≥ 80%

用法：~/gut-external/torch215-venv/bin/python invert.py
"""
import math
import os
import statistics as st
import sys

import torch

from analytic import derive5, sigma_c
from hetero import measure
from purge import C, SD_S, W, baseline

HERE = os.path.dirname(os.path.abspath(__file__))
MU = [10 ** (x / 20) for x in range(-12, 31)]                # μ 网格 0.25 … 31.6


def sets():
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    out = []
    import bigtest
    import bigtest2
    import bigtest3
    import blind5
    for mod in (bigtest, bigtest2, bigtest3):
        for dom in mod.CFG:
            data = mod.setup(dom)
            os.chdir(cwd)
            out.append((f"{mod.__name__}:{dom}", data, sorted(mod.NODES), mod.CFG[dom]["always"], mod.CFG[dom]["sleeper"], mod.DELTAS))
    d5 = blind5.setup()
    os.chdir(cwd)
    d5 = {k: [(f, p, r, x, i) for i, (f, p, r, x) in enumerate(v)] for k, v in d5.items()}
    out.append(("blind5", d5, sorted(blind5.NODES), blind5.ALWAYS, blind5.SLEEPER, blind5.DELTAS))
    return out


def loo(R):
    return torch.stack([1.4826 * (R[:, i] - torch.cat([R[:, :i], R[:, i + 1:]], 1).median(1).values).abs().median() for i in range(R.shape[1])]).clamp(min=1e-6)


_cache = {}


def delay_dist(mu, start, n=4000, horizon=60):
    key = (round(mu, 4), start)
    if key not in _cache:
        g = torch.Generator().manual_seed(0)
        S = SD_S * torch.randn(n, generator=g) if start == "stationary" else torch.zeros(n)
        hit = torch.full((n,), horizon + 1)
        for m in range(horizon + 1):
            S = (1 - W) * S + W * (mu + torch.randn(n, generator=g)).clamp(-3, 3)
            new = (S.abs() > C * SD_S) & (hit > horizon)
            hit[new] = m
        _cache[key] = hit.float()
    return _cache[key]


def part_a():
    print("== A. 抓得多快 → 毒有多大 ==")
    print("数据 / 节点 / δ                     观测轮数   推出 μ（点 / 95% 区间）       实际传出 μ   名义 μ   落在内？")
    cov, lr = [], []
    for name, data, nodes, always, sleeper, deltas in sets():
        clean5 = data[("R0", 0.0)]
        clean = [(f, p, r, x) for f, p, r, x, _ in clean5]
        b, sc = sigma_c(clean)
        sig = loo(torch.stack([r for _, _, r, _ in clean]))
        u = baseline(clean, b)
        cl_by = {k: r for _, _, r, _, k in clean5}
        for d in deltas[1:]:
            if d < 0.1:
                continue
            rows = data[("R0", d)]
            S = torch.zeros(len(nodes))
            first = {}
            for _, p, r, _, k in rows:
                z = (r - u - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)
                if z.abs().min() > 3:
                    continue
                S = (1 - W) * S + W * z.clamp(-3, 3)
                for j in torch.nonzero(S.abs() > C * SD_S).flatten().tolist():
                    first.setdefault(nodes[j], k)
            k0 = rows[0][4]
            for n in sorted(always) + ([sleeper] if sleeper else []):
                if n not in first:
                    continue
                j = nodes.index(n)
                start_k = k0 if n in always else 30
                if n == sleeper and first[n] < 30:
                    continue
                obs = first[n] - start_k
                tot = math.sqrt(sig[j].item() ** 2 + sc ** 2)
                eff = [(r[j] - cl_by[k][j]).item() for _, _, r, _, k in rows if k >= start_k and k in cl_by]
                mu_eff = abs(st.mean(eff)) / tot
                mu_nom = st.mean(x for *_, x, k in rows if k >= start_k) / tot
                start = "zero" if n in always else "stationary"
                acc, med = [], []
                for mu in MU:
                    h = delay_dist(mu, start)
                    lo, hi = torch.quantile(h, 0.025).item(), torch.quantile(h, 0.975).item()
                    if lo <= obs <= hi:
                        acc.append(mu)
                    med.append((abs(torch.quantile(h, 0.5).item() - obs), mu))
                mu_pt = min(med)[1]
                lo_, hi_ = (min(acc), max(acc)) if acc else (float("nan"), float("nan"))
                if acc and max(acc) == MU[-1]:
                    hi_ = float("inf")
                inside = bool(acc) and lo_ <= mu_eff <= hi_
                cov.append(inside)
                lr.append(abs(math.log(mu_pt / max(mu_eff, 1e-6))))
                print(f"{name[:18]:18s} {n:13s} {d:4.0%}   {obs:5d}     {mu_pt:6.2f} [{lo_:5.2f}, {hi_:6.2f}]       {mu_eff:7.2f}   {mu_nom:6.2f}   {'✅' if inside else '❌'}")
    print(f"\nA1 覆盖 {st.mean(cov):.0%}（{len(cov)} 个，≥ 85%）；A2 |log(推出 / 实际)| 中位 {st.median(lr):.2f}（≤ {math.log(1.5):.2f}）")


def part_b():
    print("\n== B. 实际被骗率 → 秤有多准（软裁判） ==")
    print("数据 δ        实际被骗   推出 σ_c 区间         秤真实误差（MAD）   正向估计   落在内？")
    grid = [10 ** (x / 10) for x in range(-40, -4)]           # 1e-4 … 0.4（log10 单位）
    cov = []
    for name, data, nodes, always, sleeper, deltas in sets():
        clean = [(f, p, r, x) for f, p, r, x, _ in data[("R0", 0.0)]]
        e = [p for _, p, _, _ in clean]
        true_sc = 1.4826 * st.median(abs(x - st.median(e)) for x in e)
        s0 = max(sigma_c(clean)[1], 1e-6)
        for d in deltas[1:]:
            bad = [(f, p, r, x) for f, p, r, x, _ in data[("R0", d)]]
            rows = [(False, p, r, x) for _, p, r, x in clean] + bad
            act = measure(rows, math.log10(1 + d), "软", s0)["miss"]
            if not 0.1 <= act <= 0.9:
                continue
            e2 = 1.96 * math.sqrt(act * (1 - act) / len(bad))
            ok = [g for g in grid if abs(derive5(rows, "软", s0, sc_override=g, M=32) - act) <= e2]
            lo, hi = (min(ok), max(ok)) if ok else (float("nan"), float("nan"))
            inside = bool(ok) and lo / 1.26 <= true_sc <= hi * 1.26          # 网格步长 10^0.1 ≈ 1.26
            cov.append(inside)
            print(f"{name[:18]:18s} {d:4.0%}   {act:6.0%}    [{lo:.4f}, {hi:.4f}]      {true_sc:.4f}           {s0:.4f}   {'✅' if inside else '❌'}")
    print(f"\nB1 覆盖 {st.mean(cov):.0%}（{len(cov)} 格，≥ 80%）")


if __name__ == "__main__":
    part_a()
    part_b()
