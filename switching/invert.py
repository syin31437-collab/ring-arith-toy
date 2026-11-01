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

C.（第二批，lag.py 之后加，11-01）实际被骗率 → 有几个诚实节点
   给推导版第五版其余所有量，把"诚实集"设成跨题汇总后离约束中心最近的 k 个节点（k = 1 … N − 1），
   算每个 k 下的预测被骗率；和实际被骗率（± 1.96 × 二项误差）对上的 k 组成区间，和真实诚实节点数比。
   可识别性：约束强时只要有一个诚实节点就几乎不被骗，k 是几都差不多 → 区间会很宽、没信息。
   预注册（运行前）：
     C1. 区间有信息量（宽度 ≤ 一半的可能取值）的格子里，真实诚实节点数落在区间内的比例 ≥ 80%
     C2. 有信息量的格子占比如实报告（不设门槛）

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


_SETS = []


def sets():
    if _SETS:
        return _SETS
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    out = []
    import blind5                                          # 先处理 7 节点的数据：bigtest2 一 import 就往共享节点表里加 Sonnet
    d5 = blind5.setup()
    os.chdir(cwd)
    import bigtest
    for dom in bigtest.CFG:
        out.append((f"bigtest:{dom}", bigtest.setup(dom), sorted(bigtest.NODES), bigtest.CFG[dom]["always"], bigtest.CFG[dom]["sleeper"], bigtest.DELTAS))
        os.chdir(cwd)
    n7 = sorted(bigtest.NODES)
    import bigtest2
    import bigtest3
    import bigtest4
    for mod in (bigtest2, bigtest3, bigtest4):
        for dom in mod.CFG:
            out.append((f"{mod.__name__}:{dom}", mod.setup(dom), sorted(mod.NODES), mod.CFG[dom]["always"], mod.CFG[dom]["sleeper"], mod.DELTAS))
            os.chdir(cwd)
    d5 = {k: [(f, p, r, x, i) for i, (f, p, r, x) in enumerate(v)] for k, v in d5.items()}
    out.append(("blind5", d5, n7, blind5.ALWAYS, blind5.SLEEPER, blind5.DELTAS))
    _SETS.extend(out)
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


def part_c():
    print("\n== C. 实际被骗率 → 有几个诚实节点（软裁判） ==")
    print("数据 δ        实际被骗   推出诚实节点数区间   真实诚实数   有信息量？   落在内？")
    cov, info = [], []
    for name, data, nodes, always, sleeper, deltas in sets():
        clean = [(f, p, r, x) for f, p, r, x, _ in data[("R0", 0.0)]]
        b, _ = sigma_c(clean)
        s0 = max(sigma_c(clean)[1], 1e-6)
        n = len(nodes)
        bad_set = set(always) | ({sleeper} if sleeper else set())
        for d in deltas[1:]:
            bad = [(f, p, r, x) for f, p, r, x, _ in data[("R0", d)]]
            rows = [(False, p, r, x) for _, p, r, x in clean] + bad
            act = measure(rows, math.log10(1 + d), "软", s0)["miss"]
            e2 = 1.96 * math.sqrt(max(act * (1 - act), 1e-4) / len(bad))
            X = torch.stack([r - (p - b) for _, p, r, _ in rows])
            order = X.mean(0).abs().argsort().tolist()                 # 离约束中心最近的排前面
            ok = []
            for k in range(1, n):
                hon = torch.zeros(n, dtype=torch.bool)
                hon[order[:k]] = True
                if abs(derive5(rows, "软", s0, honest=hon) - act) <= e2:
                    ok.append(k)
            true_k = n - len(bad_set)                                    # 卧底在后半段才投毒，按投毒算
            if not ok:
                print(f"{name[:18]:18s} {d:4.0%}   {act:6.0%}    （没有 k 对得上）          {true_k}")
                cov.append(False)
                info.append(True)
                continue
            lo, hi = min(ok), max(ok)
            inf_ = hi - lo + 1 <= (n - 1) / 2
            info.append(inf_)
            if inf_:
                cov.append(lo <= true_k <= hi)
            print(f"{name[:18]:18s} {d:4.0%}   {act:6.0%}    [{lo}, {hi}]                   {true_k}            {'是' if inf_ else '否'}         {('✅' if lo <= true_k <= hi else '❌') if inf_ else '—'}")
    print(f"\nC1 有信息量的格子里覆盖 {st.mean(cov) if cov else float('nan'):.0%}（{len(cov)} 格，≥ 80%）；C2 有信息量的占比 {st.mean(info):.0%}（{len(info)} 格）")


if __name__ == "__main__":
    {"b": part_b, "c": part_c}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: (part_a(), part_b()))()
