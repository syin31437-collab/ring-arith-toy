"""触发原因看不见的相关故障：按状态加权还管不管用？节点之间互相比对能不能补上？

节点：12 个独立小 MLP（同 robust.py 的 V 组，不连接），只比较最终输出怎么汇总。
故障（+3，触发原因和序列值无关，系统看不见）：
  第 2000–5000 步  成串：隐藏开关，开着平均 50 步、关着平均 100 步；开着时 A+B 型 8 个一起坏
  第 5000–9500 步  零散：每一步独立掷骰子，30% 的步 A+B 型 8 个一起坏，
                    另有 10% 的步只有 C 型 4 个坏（少数派坏，防止"永远信 C"这种偷懒解）

汇总方式：
  V    中位数
  V+W  按状态格子（最新值 x 分 6 格）里各节点近期误差加权的中位数（robust.py 里的赢家）
  V+T  不分格子，只按各节点全局近期误差加权
  P    按"此刻谁和谁站在一起"分格子：把 12 个输出按最大的间隔切成两团，
       间隔 > 0.5 时，格子 = 偏高那一团是哪些节点；否则格子 = "大家一致"。
       每个格子里记各节点近期误差，加权取中位数。看的是节点之间的关系，不是外部状态。

用法：python hidden.py run SYSTEM [N_SEEDS]  →  hidden_SYSTEM.json
      python hidden.py report
"""
import json
import statistics as st
import sys

import torch

from robust import EDGES, LR, MLP, N, TYPE, VIEW, wmedian
from switch import make_series

BIAS, GAP = 3.0, 0.5
SYSTEMS = ["V", "V+W", "V+T", "P"]
PHASES = {"成串": (2000, 5000), "零散": (5000, 9500)}


def fault_plan(seed):
    g = torch.Generator().manual_seed(50000 + seed)
    plan, on = [], False
    for t in range(9500):
        u = torch.rand(1, generator=g).item()
        kind = None
        if 2000 <= t < 5000:
            on = (u > 1 / 50) if on else (u < 1 / 100)
            kind = "AB" if on else None
        elif t >= 5000:
            kind = "AB" if u < 0.3 else "C" if u < 0.4 else None
        plan.append(kind)
    return plan


def mask(kind):
    if kind == "AB":
        return TYPE != 2
    if kind == "C":
        return TYPE == 2
    return torch.zeros(N, dtype=torch.bool)


def signature(v):
    order = torch.argsort(v)
    gaps = v[order][1:] - v[order][:-1]
    k = int(torch.argmax(gaps))
    if gaps[k] <= GAP:
        return "agree"
    return tuple(sorted(order[k + 1:].tolist()))       # 偏高那一团的节点编号


def run(system, seed, corrupt):
    data, _ = make_series(seed)
    g = torch.Generator().manual_seed(seed)
    base = MLP(3, g)
    opt = torch.optim.Adam(base.p, lr=LR)
    plan = fault_plan(seed)
    err = {}                                            # 格子 → 各节点近期误差
    log = []
    for t, (hist, y) in enumerate(data):
        kind = plan[t] if corrupt else None
        p = base(torch.tensor(hist)[VIEW])
        shown = p.detach() + BIAS * mask(kind).float()

        if system == "V":
            key = None
        elif system == "V+W":
            key = int(torch.bucketize(torch.tensor(hist[0]), EDGES))
        elif system == "V+T":
            key = "all"
        else:
            key = signature(shown)
        if key is None:
            out = shown.median()
        else:
            e = err.setdefault(key, torch.full((N,), 0.05))
            out = wmedian(shown, torch.exp(-e / 0.1))
        log.append((out.item(), y, kind, key if system == "P" else None))

        loss = ((p - y) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if key is not None:
            with torch.no_grad():
                err[key] = 0.9 * err[key] + 0.1 * ((shown - y) ** 2).clamp(max=1.0)
    return log


def do_run(system, n_seeds):
    torch.set_num_threads(1)
    out = []
    for seed in range(n_seeds):
        dirty, clean = run(system, seed, True), run(system, seed, False)
        out.append({"dirty": [(o, y, k) for o, y, k, _ in dirty], "clean": [o for o, _, _, _ in clean],
                    "keys": [str(key) for *_, key in dirty],
                    "n_keys": len({key for *_, key in dirty if key is not None})})
        print(system, seed, flush=True)
    with open(f"hidden_{system}.json", "w") as f:
        json.dump(out, f)


def report():
    data = {s: json.load(open(f"hidden_{s}.json")) for s in SYSTEMS}
    med = st.median

    def hit_rate(rec, lo, hi, kind):
        steps = [(o - y) ** 2 > 1.0 for o, y, k in rec["dirty"][lo:hi] if k == kind]
        return sum(steps) / len(steps) if steps else float("nan")

    print("输出被带偏（误差 > 1）的步占比，只统计该类故障发生的步；中位数跨 10 个种子")
    print("\n系统    成串AB   成串AB后半   零散AB   零散AB后半   零散C(少数坏)   零散C后半")
    for s in SYSTEMS:
        cells = []
        for (lo, hi), kind in (((2000, 5000), "AB"), ((5000, 9500), "AB"), ((5000, 9500), "C")):
            mid = (lo + hi) // 2
            cells += [med([hit_rate(r, lo, hi, kind) for r in data[s]]), med([hit_rate(r, mid, hi, kind) for r in data[s]])]
        print(f"{s:5s} " + "  ".join(f"{c:9.1%}" for c in cells))

    print("\n没故障的步上，输出误差 / 干净对照（看会不会误伤）")
    for s in SYSTEMS:
        r = [st.mean((o - y) ** 2 for (o, y, k) in rec["dirty"][2000:] if k is None) /
             st.mean((c - y) ** 2 for c, (o, y, k) in zip(rec["clean"][2000:], rec["dirty"][2000:]) if k is None)
             for rec in data[s]]
        print(f"{s:5s} {med(r):.2f}x")
    print(f"\nP 组平均用到的格子数：{med([r['n_keys'] for r in data['P']])}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        report()
