"""闭环：从错误里学关联，用关联让错误越来越不致命。

环境：switch.py 的规律切换序列；12 个互不相连的节点（同 robust.py 的 V 组）。
故障（偏 +3），第 2000 步起每一步：
  12% 一种型号整体坏（少数）
  12% 两种型号一起坏（通常过半）
  另外每个节点各自 3% 零星随机坏
  型号：一开始 A/B/C 交错（i % 3）
  第 4000 步  3 个节点悄悄换型号
  第 6000 步  出现全新型号 D：从 A、B、C 各拉 1 个、再加 1 个，组成 4 个一起坏的新团
汇总方式：
  V   中位数
  P   按"偏高那一团是哪几个节点"分格子（hidden.py 的 P，死记具体节点）
  PG  按"偏高那一团覆盖了哪几个学出来的团"分格子；团 = 最近 40 次故障里同坏相关 > 0.5 的连通块，
      每 50 步重算一次。零星乱坏的节点不改变格子，所以老模式还认得出来
  PO  同 PG，但直接用真实型号（作弊上限）
致命度：故障步上，输出相对干净对照偏移 > 1.5 的比例，每 500 步一格。

用法：python loop.py run SYSTEM [N_SEEDS]  →  loop_SYSTEM.json
      python loop.py report
"""
import json
import statistics as st
import sys

import torch

from robust import LR, MLP, N, VIEW, wmedian
from switch import make_series

BIAS, GAP, WIN = 3.0, 0.5, 40
SYSTEMS = ["V", "P", "PG", "PO"]
BINS = list(range(2000, 9500, 500))


def type_schedule(seed):
    g = torch.Generator().manual_seed(80000 + seed)
    t0 = [i % 3 for i in range(N)]
    t1 = list(t0)
    for i in torch.randperm(N, generator=g)[:3].tolist():
        t1[i] = (t1[i] + 1 + int(torch.randint(0, 2, (1,), generator=g))) % 3
    t2 = list(t1)
    picks = [next(i for i in torch.randperm(N, generator=g).tolist() if t1[i] == k) for k in range(3)]
    picks.append(next(i for i in torch.randperm(N, generator=g).tolist() if i not in picks))
    for i in picks:
        t2[i] = 3
    return t0, t1, t2


def fault_plan(seed):
    g = torch.Generator().manual_seed(90000 + seed)
    t0, t1, t2 = type_schedule(seed)
    plan = []
    for t in range(9500):
        types = t0 if t < 4000 else t1 if t < 6000 else t2
        kinds = sorted(set(types))
        m = torch.zeros(N, dtype=torch.bool)
        if t >= 2000:
            u = torch.rand(1, generator=g).item()
            if u < 0.12:
                f = {kinds[int(torch.randint(0, len(kinds), (1,), generator=g))]}
            elif u < 0.24:
                f = set(torch.tensor(kinds)[torch.randperm(len(kinds), generator=g)[:2]].tolist())
            else:
                f = set()
            m = torch.tensor([types[i] in f for i in range(N)])
            m |= torch.rand(N, generator=g) < 0.03
        plan.append((m, types))
    return plan


def split(v):
    order = torch.argsort(v)
    gaps = v[order][1:] - v[order][:-1]
    k = int(torch.argmax(gaps))
    return None if gaps[k] <= GAP else set(order[k + 1:].tolist())


def learn_groups(hist):
    if len(hist) < 5:
        return [[i] for i in range(N)]
    C = torch.nan_to_num(torch.corrcoef(torch.tensor(hist).T.float()), nan=0.0)
    lab = [-1] * N
    for s in range(N):
        if lab[s] >= 0:
            continue
        stack, lab[s] = [s], s
        while stack:
            u = stack.pop()
            for v in torch.nonzero(C[u] > 0.5).flatten().tolist():
                if lab[v] < 0:
                    lab[v] = s
                    stack.append(v)
    out = {}
    for i, l in enumerate(lab):
        out.setdefault(l, []).append(i)
    return list(out.values())


def group_key(high, groups):
    """偏高那一团覆盖了哪些团：团里过半成员在偏高团里，才算这个团站过去了。"""
    return tuple(sorted(min(g) for g in groups if sum(i in high for i in g) * 2 > len(g)))


def run(system, seed, plan, labeled=None, mode="only", trace=None):
    """labeled：每一步是否揭晓真实答案（None = 全部揭晓）。
    mode：没揭晓答案时怎么办——
      only   什么都不学
      self   节点、信任表、故障历史都拿系统自己的输出当答案
      nodes  只有节点拿系统输出当答案学，信任表和故障历史只认真实答案
      agree  同 nodes，但节点只在大家意见一致（没分成两团）时才拿系统输出当答案学
      judge  诊断用：节点永远用真实答案学，只有信任表和故障历史受答案比例限制"""
    data, _ = make_series(seed)
    g = torch.Generator().manual_seed(seed)
    base = MLP(3, g)
    opt = torch.optim.Adam(base.p, lr=LR)
    err, hist, groups, outs = {}, [], [[i] for i in range(N)], []
    for t, (h, y) in enumerate(data):
        m, types = plan[t] if plan else (torch.zeros(N, dtype=torch.bool), None)
        p = base(torch.tensor(h)[VIEW])
        if trace is not None:                            # loop_probe.py 用：记下各节点预测和真值
            trace.append((p.detach().clone(), y))
        shown = p.detach() + BIAS * m.float()
        key = None
        if system != "V":
            high = split(shown)
            if high is None:
                key = "agree"
            elif system == "P":
                key = tuple(sorted(high))
            else:
                gs = groups if system == "PG" else \
                    [[i for i in range(N) if types[i] == k] for k in set(types)] if types else [[i] for i in range(N)]
                key = group_key(high, gs)
        if key is None:
            out = shown.median()
        else:
            e = err.setdefault(key, torch.full((N,), 0.05))
            out = wmedian(shown, torch.exp(-e / 0.1))
        outs.append(out.item())

        known = labeled is None or labeled[t]
        target = y if known else out.item()
        node_target = y if mode == "judge" else target
        judge = known or mode == "self"                  # 信任表和故障历史这一步能不能学
        if key is not None and judge:
            with torch.no_grad():
                err[key] = 0.9 * e + 0.1 * ((shown - target) ** 2).clamp(max=1.0)
        if judge:
            wrong = ((shown - target) ** 2 > 1.0).tolist()
            if any(wrong):                               # 只把"有人错了"的步记进故障历史
                hist.append([int(w) for w in wrong])
                hist = hist[-WIN:]
        if system == "PG" and t % 50 == 0:
            groups = learn_groups(hist)

        if known or mode in ("self", "nodes", "judge") or (mode == "agree" and key == "agree"):
            loss = ((p - node_target) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return outs, len(err)


def do_run(system, n_seeds):
    torch.set_num_threads(1)
    out = []
    for seed in range(n_seeds):
        plan = fault_plan(seed)
        clean, _ = run(system, seed, None)
        dirty, n_keys = run(system, seed, plan)
        curve, curve_major = [], []
        for lo in BINS:
            steps = [t for t in range(lo, lo + 500) if plan[t][0].any()]
            major = [t for t in steps if plan[t][0].sum() * 2 > N]
            leak = lambda ts: sum(abs(dirty[t] - clean[t]) > BIAS / 2 for t in ts) / len(ts) if ts else float("nan")
            curve.append(leak(steps))
            curve_major.append(leak(major))
        out.append({"curve": curve, "major": curve_major, "n_keys": n_keys})
        print(system, seed, flush=True)
    with open(f"loop_{system}.json", "w") as f:
        json.dump(out, f)


def report():
    data = {s: json.load(open(f"loop_{s}.json")) for s in SYSTEMS}
    med = st.median
    for field, title in (("curve", "全部故障步"), ("major", "只看过半节点坏的步")):
        print(f"\n== 致命度（{title}）：输出被带偏的比例，每 500 步一格，10 种子中位数 ==")
        print("步数        " + "".join(f"{s:>8s}" for s in SYSTEMS) + "   事件")
        for k, lo in enumerate(BINS):
            ev = {4000: "← 3 个节点换型号", 6000: "← 出现新型号 D", 8000: "← 新规律 R5"}.get(lo, "")
            print(f"{lo:5d}-{lo+499:<5d}" + "".join(f"{med([r[field][k] for r in data[s]]):8.1%}" for s in SYSTEMS) + f"   {ev}")
    print("\n用到的格子数（中位数）：" + "  ".join(f"{s} {int(med([r['n_keys'] for r in data[s]]))}" for s in SYSTEMS if s != "V"))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        report()
