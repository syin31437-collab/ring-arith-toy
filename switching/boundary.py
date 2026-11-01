"""P 的边界：故障随机挑节点（模式爆炸）、故障很小（分不出两团）。

第 2000 步以后，每一步 30% 概率发生故障，坏法按场景：
  结构+3      A+B 8 个一起坏，偏 +3（hidden.py 的零散场景，基线）
  随机8个     每次随机挑 8 个，偏 +3
  随机7-9个   每次随机挑 7–9 个，偏 +3
  成员带噪    A+B 里每个节点各自 80% 概率坏（大体同一批，但每次不完全一样），偏 +3
  结构+1 / +0.5 / +0.3 / +0.15   A+B 8 个一起坏，偏移变小
漏过 = 故障步上，输出相对干净对照偏移超过故障大小的一半。

用法：python boundary.py run SYSTEM [N_SEEDS]  →  boundary_SYSTEM.json
      python boundary.py report
"""
import json
import statistics as st
import sys

import torch

from hidden import signature
from robust import LR, MLP, N, TYPE, VIEW, wmedian
from switch import make_series

SCENARIOS = {"结构+3": ("AB", 3.0), "随机8个": ("r8", 3.0), "随机7-9个": ("r79", 3.0), "成员带噪": ("noisy", 3.0),
             "结构+1": ("AB", 1.0), "结构+0.5": ("AB", 0.5), "结构+0.3": ("AB", 0.3), "结构+0.15": ("AB", 0.15)}
SYSTEMS = ["V", "V+T", "P"]


def make_plan(kind, seed):
    g = torch.Generator().manual_seed(60000 + seed)
    plan = []
    for t in range(9500):
        m = torch.zeros(N, dtype=torch.bool)
        if t >= 2000 and torch.rand(1, generator=g).item() < 0.3:
            if kind == "AB":
                m = TYPE != 2
            elif kind == "noisy":
                m = (TYPE != 2) & (torch.rand(N, generator=g) < 0.8)
            else:
                k = 8 if kind == "r8" else int(torch.randint(7, 10, (1,), generator=g))
                m[torch.randperm(N, generator=g)[:k]] = True
        plan.append(m)
    return plan


def run(system, seed, plan, bias):
    data, _ = make_series(seed)
    g = torch.Generator().manual_seed(seed)
    base = MLP(3, g)
    opt = torch.optim.Adam(base.p, lr=LR)
    err, outs = {}, []
    for t, (hist, y) in enumerate(data):
        p = base(torch.tensor(hist)[VIEW])
        shown = p.detach() + (bias * plan[t].float() if plan else 0.0)
        key = None if system == "V" else "all" if system == "V+T" else signature(shown)
        if key is None:
            out = shown.median()
        else:
            e = err.setdefault(key, torch.full((N,), 0.05))
            out = wmedian(shown, torch.exp(-e / 0.1))
            with torch.no_grad():
                err[key] = 0.9 * e + 0.1 * ((shown - y) ** 2).clamp(max=1.0)
        outs.append(out.item())
        loss = ((p - y) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return outs, len(err)


def do_run(system, n_seeds):
    torch.set_num_threads(1)
    out = {}
    for seed in range(n_seeds):
        clean, _ = run(system, seed, None, 0.0)
        for name, (kind, bias) in SCENARIOS.items():
            plan = make_plan(kind, seed)
            dirty, n_keys = run(system, seed, plan, bias)
            steps = [t for t in range(2000, 9500) if plan[t].any()]
            leak = [abs(dirty[t] - clean[t]) > bias / 2 for t in steps]
            half = len(leak) // 2
            out.setdefault(name, []).append({"leak": sum(leak) / len(leak), "leak_late": sum(leak[half:]) / (len(leak) - half),
                                              "n_keys": n_keys})
        print(system, seed, flush=True)
    with open(f"boundary_{system}.json", "w") as f:
        json.dump(out, f)


def report():
    data = {s: json.load(open(f"boundary_{s}.json")) for s in SYSTEMS}
    med = st.median
    print("漏过比例（全程 / 后半），中位数跨种子；P 后面是用到的格子数")
    print("场景        " + "".join(f"{s:>18s}" for s in SYSTEMS))
    for name in SCENARIOS:
        cells = []
        for s in SYSTEMS:
            r = data[s][name]
            c = f"{med([x['leak'] for x in r]):.0%}/{med([x['leak_late'] for x in r]):.0%}"
            if s == "P":
                c += f" ({int(med([x['n_keys'] for x in r]))}格)"
            cells.append(f"{c:>18s}")
        print(f"{name:10s}" + "".join(cells))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        report()
