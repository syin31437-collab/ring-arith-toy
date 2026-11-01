"""修正版 + 从错误历史里学出"谁和谁一起坏"。

60 个节点，隐藏型号 A/B/C（i % 3，系统不知道）。每个节点 6 个检查者（层级结构仍是 1 个）。
失守 = 检查者里坏的 ≥ 一半（偏保守）。

连法：
  环        左右各 3 个邻居
  随机      随机 6 个
  层级      中心看 3–4 个下属，逐层往下；下属只有中心一个检查者，根没人看
  按型号均衡（已知）  每种型号各挑 2 个检查者——作弊上限，直接用了真实型号
  按型号均衡（学出）  先看过去 H 次故障里谁和谁一起坏，算两两"同坏相关"，
                      相关 > 0.5 的连成一团当作一种型号，再按学出的团均衡挑检查者
过去的故障历史：每次随机是 A、B、C 某一种型号整体坏，外加 10% 的节点随机坏（噪声）。
"""
import random
import statistics as st

N, K, TRIALS = 60, 6, 200
TYPE = [i % 3 for i in range(N)]


def ring(rng, groups=None):
    return {j: [(j + d) % N for d in (-3, -2, -1, 1, 2, 3)] for j in range(N)}


def rand(rng, groups=None):
    return {j: rng.sample([i for i in range(N) if i != j], K) for j in range(N)}


def tree(rng, groups=None):
    order = list(range(N))
    rng.shuffle(order)
    testers, frontier, nxt = {order[0]: []}, [order[0]], 1
    while nxt < N:
        new = []
        for c in frontier:
            for _ in range(rng.choice((3, 4))):
                if nxt < N:
                    testers[order[nxt]] = [c]
                    new.append(order[nxt])
                    nxt += 1
        frontier = new
    return testers


def balanced(rng, groups):
    """按团均衡：轮流从每个团里各挑一个，直到凑满 K 个。"""
    out = {}
    for j in range(N):
        pools = [[i for i in g if i != j] for g in groups]
        for p in pools:
            rng.shuffle(p)
        pick, k = [], 0
        while len(pick) < K and any(pools):
            p = pools[k % len(pools)]
            if p:
                pick.append(p.pop())
            k += 1
        out[j] = pick
    return out


def history(H, rng):
    events = []
    for _ in range(H):
        t = rng.randrange(3)
        events.append({i for i in range(N) if TYPE[i] == t or rng.random() < 0.1})
    return events


def learn_groups(events):
    """两两同坏相关 > 0.5 连边，取连通块。"""
    x = [[int(i in e) for e in events] for i in range(N)]
    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        va = sum((u - ma) ** 2 for u in a)
        vb = sum((v - mb) ** 2 for v in b)
        if va == 0 or vb == 0:
            return 0.0
        return sum((u - ma) * (v - mb) for u, v in zip(a, b)) / (va * vb) ** 0.5
    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(N):
        for j in range(i + 1, N):
            if corr(x[i], x[j]) > 0.5:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def pair_accuracy(groups):
    lab = {i: g for g, members in enumerate(groups) for i in members}
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    return st.mean((lab[i] == lab[j]) == (TYPE[i] == TYPE[j]) for i, j in pairs)


def faults(kind, rng):
    if kind == "随机30%":
        return set(rng.sample(range(N), 18))
    if kind == "连续一段30%":
        s = rng.randrange(N)
        return {(s + d) % N for d in range(18)}
    if kind == "A型全坏":
        return {i for i in range(N) if TYPE[i] == 0}
    return {i for i in range(N) if TYPE[i] != 2}


def lost(testers, bad):
    good, hidden = [], []
    for j, ts in testers.items():
        if not ts:
            continue
        fallen = sum(t in bad for t in ts) * 2 >= len(ts)
        (hidden if j in bad else good).append(fallen)
    return st.mean(good) if good else 0.0, st.mean(hidden) if hidden else 0.0


rng = random.Random(0)
true_groups = [[i for i in range(N) if TYPE[i] == t] for t in range(3)]

print("== 学型号：过去经历过几次故障，就能把'谁和谁一起坏'找对多少 ==")
for H in (3, 5, 10, 20, 50):
    accs, ngs = [], []
    for _ in range(50):
        g = learn_groups(history(H, rng))
        accs.append(pair_accuracy(g))
        ngs.append(len(g))
    print(f"经历 {H:2d} 次故障：两两关系判断正确 {st.mean(accs):.1%}，学出的团数中位 {st.median(ngs)}")

print("\n== 失守比例：好节点失守% / 坏节点藏住%（200 次平均；学出型号用 20 次故障历史） ==")
builds = [("环", ring, None), ("随机", rand, None), ("层级", tree, None),
          ("均衡(已知型号)", balanced, "true"), ("均衡(学出型号)", balanced, "learn")]
print("坏法          " + "".join(f"{n:>17s}" for n, _, _ in builds))
for kind in ("随机30%", "连续一段30%", "A型全坏", "A+B型全坏"):
    cells = []
    for name, build, mode in builds:
        rows = []
        for _ in range(TRIALS):
            groups = true_groups if mode == "true" else learn_groups(history(20, rng)) if mode == "learn" else None
            rows.append(lost(build(rng, groups), faults(kind, rng)))
        cells.append(f"{st.mean(r[0] for r in rows):6.0%} /{st.mean(r[1] for r in rows):5.0%}")
    print(f"{kind:12s}" + "".join(f"{c:>17s}" for c in cells))
