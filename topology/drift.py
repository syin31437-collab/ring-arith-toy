"""关系会变、新东西会进来时，"谁和谁一起坏"还能不能学对。

时间单位 = 一次故障事件（某一型号整体坏，外加 10% 节点随机坏）。
  事件 0–99    60 个节点，型号 A/B/C（i % 3）
  事件 100     突变：随机 20 个节点改成另一种型号
  事件 200     新加入 12 个节点：6 个属于已有型号（每种 2 个），6 个组成全新型号 D；此后 D 也会整体坏
  事件 300–399 渐变：每 10 次事件，随机 2 个节点悄悄换型号

学法：两两同坏相关 > 0.5 连成团。区别只在看多久的历史：
  全部历史 / 最近 40 次 / 最近 20 次 / 最近 10 次
指标：
  两两关系对不对（只算当前都在的节点对）
  新节点的两两关系对不对（至少一方是新节点的对）
  好节点失守%：按学出的团均衡挑 6 个检查者，当前某一种型号整体坏时，好节点被坏检查者包围（≥一半）的比例
"""
import random
import statistics as st

import torch

N0, NEW, NOISE, K = 60, 12, 0.1, 6
WINDOWS = {"全部历史": None, "最近40次": 40, "最近20次": 20, "最近10次": 10}
CHECKS = [99, 102, 105, 110, 120, 140, 199, 202, 205, 210, 220, 240, 299]


def simulate(seed):
    rng = random.Random(seed)
    types = [i % 3 for i in range(N0)]
    alive = N0
    events, truth = [], []
    for t in range(400):
        if t == 100:
            for i in rng.sample(range(N0), 20):
                types[i] = rng.choice([x for x in range(3) if x != types[i]])
        if t == 200:
            types += [0, 0, 1, 1, 2, 2] + [3] * 6
            alive = N0 + NEW
        if t >= 300 and t % 10 == 0:
            for i in rng.sample(range(alive), 2):
                types[i] = rng.choice([x for x in range(4) if x != types[i]])
        kinds = 3 if t < 200 else 4
        f = rng.randrange(kinds)
        row = [0] * (N0 + NEW)
        for i in range(alive):
            row[i] = int(types[i] == f or rng.random() < NOISE)
        events.append(row)
        truth.append(list(types[:alive]))
    return events, truth


def groups_from(X):
    """X: (节点, 事件)。相关 > 0.5 连边，取连通块；从没坏过/一直坏的节点自成一团。"""
    n = X.shape[0]
    if X.shape[1] < 3:
        return [[i] for i in range(n)]
    C = torch.nan_to_num(torch.corrcoef(X.float()), nan=0.0)
    adj = C > 0.5
    lab = [-1] * n
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack, lab[s] = [s], s
        while stack:
            u = stack.pop()
            for v in torch.nonzero(adj[u]).flatten().tolist():
                if lab[v] < 0:
                    lab[v] = s
                    stack.append(v)
    out = {}
    for i, l in enumerate(lab):
        out.setdefault(l, []).append(i)
    return list(out.values())


def pair_acc(groups, types, only=None):
    lab = {i: g for g, m in enumerate(groups) for i in m}
    n = len(types)
    ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if only is not None and i < only and j < only:
                continue
            ok += (lab[i] == lab[j]) == (types[i] == types[j])
            tot += 1
    return ok / tot


def protection(groups, types, rng):
    n = len(types)
    lost = []
    for f in set(types):
        bad = {i for i in range(n) if types[i] == f}
        for j in range(n):
            if j in bad:
                continue
            pools = [[i for i in g if i != j] for g in groups]
            for p in pools:
                rng.shuffle(p)
            pools.sort(key=len, reverse=True)
            pick, k = [], 0
            while len(pick) < K and any(pools):
                p = pools[k % len(pools)]
                if p:
                    pick.append(p.pop())
                k += 1
            lost.append(sum(i in bad for i in pick) * 2 >= len(pick))
    return st.mean(lost)


def main():
    seeds = 20
    rows = {w: {t: [] for t in CHECKS} for w in WINDOWS}
    newacc = {w: {t: [] for t in CHECKS if t >= 202} for w in WINDOWS}
    prot = {w: {t: [] for t in CHECKS} for w in WINDOWS}
    drift = {w: [] for w in WINDOWS}
    drift_prot = {w: [] for w in WINDOWS}
    for seed in range(seeds):
        events, truth = simulate(seed)
        E = torch.tensor(events).T                     # (节点, 事件)
        rng = random.Random(seed)
        for w, win in WINDOWS.items():
            for t in list(CHECKS) + list(range(309, 400, 10)):
                n = len(truth[t])
                lo = 0 if win is None else max(0, t + 1 - win)
                X = E[:n, lo:t + 1]
                g = groups_from(X)
                acc = pair_acc(g, truth[t])
                p = protection(g, truth[t], rng)
                if t in CHECKS:
                    rows[w][t].append(acc)
                    prot[w][t].append(p)
                    if t >= 202:
                        newacc[w][t].append(pair_acc(g, truth[t], only=N0))
                else:
                    drift[w].append(acc)
                    drift_prot[w].append(p)

    m = lambda xs: st.mean(xs)
    print("== 两两关系判断正确率（20 种子平均） ==")
    print("事件     " + "".join(f"{w:>10s}" for w in WINDOWS) + "   说明")
    notes = {99: "突变前", 102: "突变后 2 次", 199: "突变后 99 次", 202: "新节点加入后 2 次", 299: "加入后 99 次"}
    for t in CHECKS:
        print(f"{t:4d}   " + "".join(f"{m(rows[w][t]):10.1%}" for w in WINDOWS) + f"   {notes.get(t, '')}")
    print(f"渐变期  " + "".join(f"{m(drift[w]):10.1%}" for w in WINDOWS) + "   事件 309–399 平均")

    print("\n== 新节点（含全新型号 D）的两两关系正确率 ==")
    for t in [t for t in CHECKS if t >= 202]:
        print(f"{t:4d}   " + "".join(f"{m(newacc[w][t]):10.1%}" for w in WINDOWS))

    print("\n== 按学出的团分配检查者后，某一型号整体坏时好节点失守% ==")
    for t in CHECKS:
        print(f"{t:4d}   " + "".join(f"{m(prot[w][t]):10.1%}" for w in WINDOWS))
    print(f"渐变期  " + "".join(f"{m(drift_prot[w]):10.1%}" for w in WINDOWS))


if __name__ == "__main__":
    main()
