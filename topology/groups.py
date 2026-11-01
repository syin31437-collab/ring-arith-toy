"""多层中心结构 vs 其他连法：在不同坏法下，有多少节点"被坏人包围"。

60 个节点排成一圈；三种型号 A/B/C 交错（i % 3）。每个节点被若干个"检查者"看。
一个节点的检查者里坏的过半，它就"失守"：好节点可能被诬陷，坏节点可能被包庇。

连法（每个节点约 4 个检查者，层级结构除外）：
  环        左右各 2 个邻居
  随机      随机 4 个
  层级      你说的样子：1 号当 4 个节点的中心，那 4 个各自再当 3–4 个节点的中心……
            只有中心看下属，下属不看中心，根节点没人看
  型号分散  4 个检查者全部来自另外两种型号（各 2 个），随机挑
坏法：随机坏 30% / 连续一段坏 30%（地理上扎堆）/ A 型全坏（1/3）/ A+B 型全坏（2/3）
"""
import random
import statistics as st

N, K, TRIALS = 60, 4, 200
TYPE = [i % 3 for i in range(N)]


def ring(rng):
    return {j: [(j + d) % N for d in (-2, -1, 1, 2)] for j in range(N)}


def rand(rng):
    return {j: rng.sample([i for i in range(N) if i != j], K) for j in range(N)}


def tree(rng):
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


def diverse(rng):
    out = {}
    for j in range(N):
        others = [t for t in range(3) if t != TYPE[j]]
        out[j] = [i for t in others for i in rng.sample([i for i in range(N) if TYPE[i] == t], 2)]
    return out


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
    good_lost, bad_hidden, unchecked = [], [], 0
    for j, ts in testers.items():
        if not ts:
            unchecked += 1
            continue
        fallen = sum(t in bad for t in ts) * 2 >= len(ts)      # 过半（或一半）检查者是坏的
        (bad_hidden if j in bad else good_lost).append(fallen)
    m = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return m(good_lost), m(bad_hidden), unchecked


rng = random.Random(0)
print("每格：好节点失守% / 坏节点藏住%（200 次随机平均）")
print("坏法          " + "".join(f"{n:>16s}" for n in ("环", "随机", "层级", "型号分散")))
for kind in ("随机30%", "连续一段30%", "A型全坏", "A+B型全坏"):
    cells = []
    for build in (ring, rand, tree, diverse):
        rows = [lost(build(rng), faults(kind, rng)) for _ in range(TRIALS)]
        cells.append(f"{st.mean(r[0] for r in rows):6.0%} /{st.mean(r[1] for r in rows):5.0%}")
    print(f"{kind:12s}" + "".join(f"{c:>16s}" for c in cells))
print("\n层级结构里没人看的节点（根）：1 个；每个下属只有 1 个检查者（它的中心）")
