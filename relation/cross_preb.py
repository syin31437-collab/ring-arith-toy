"""非 LLM 映射预检：只用训练折内部切分，没有碰正式校准和测试行。"""
import random
import numpy as np
import sensors as SE

X, y = SE.load("raw3/vehicle.arff")
order = list(range(len(y)))
random.Random("rel-vehicle").shuffle(order)
tr = order[:546]
m, s = X[tr].mean(0), X[tr].std(0) + 1e-9
Z = (X - m) / s
inner_tr, dev = tr[:400], tr[400:]
import sys
top = SE.info_order(Z[inner_tr], y[inner_tr], int(sys.argv[1]))
print("行", len(y), "公开特征", top)
nodes = SE.build_nodes(Z[inner_tr], y[inner_tr], "pre")
full = np.ones(X.shape[1], dtype=bool)
oracle = [SE.Model(k, Z[inner_tr], y[inner_tr]) for k in ("lr", "knn")]
n0 = len(dev)
dev = [i for i in dev if all(int(np.argmax(o.p(Z[i], full))) == y[i] for o in oracle)]
print("全视野的 lr、knn 都答对的行", len(dev), "/", n0)
res = {}
for name, (rl, mod, seen) in nodes.items():
    rows = {f: [] for f in SE.FORMS}
    for i in dev:
        idt = f"P{i}"
        pay = SE.payload(idt, Z[i], int(y[i]), Z[inner_tr], y[inner_tr], top)
        for f in SE.FORMS:
            x0, cf, a = SE.answer(rl, mod, Z[i], seen, top, pay, f)
            rows[f].append((x0 == y[i], a == y[i]))
    K = lambda f: np.mean([o for r, o in rows[f] if r]) if any(r for r, _ in rows[f]) else float("nan")
    B = lambda f: np.mean([o for r, o in rows[f] if not r]) if any(not r for r, _ in rows[f]) else float("nan")
    res[name] = (np.mean([r for r, _ in rows["blank"]]), K("L3f"), K("L4f"), K("set2f"), B("L3t"), B("L4t"), B("set2t"), K("L3t"))
print("平均 K(L3t) %.2f B(L3t) %.2f K(L3f) %.2f；K(L3t) 最低 %.2f" % (np.mean([v[7] for v in res.values()]), np.mean([v[4] for v in res.values()]), np.mean([v[1] for v in res.values()]), min(v[7] for v in res.values())))
print(f"{'节点':14s} 准确率 K(L3f) K(L4f) K(set2f) B(L3t) B(L4t) B(set2t) K(L3t)")
for n, v in res.items():
    print(f"{n:14s} " + " ".join(f"{a:6.2f}" for a in v))
