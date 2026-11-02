"""阶梯实测（theory/ladder.md 的预言）：BIPM 测试段（2015 起，这段数据看过，但这个检验是新的；检测常数不改）。
场景（每窗口-组 × δ ∈ {6, 12, 24} ns / 5 天）：S0 不注入（算误报）；one 1 个节点阶跃；min 4/9；maj 6/9；all 9/9；drift 9/9 在评估期从 0 线性涨到 2δ；scale 秤 A 阶跃。
三级检测，全部用 EWMA（W = 0.3、截断 ±3σ；基线 = 校准期中位 + MAD），分数 = 评估期 max|S|：
  L0 同伴：每个节点 − 其他节点中位数，取最大（没有秤）
  L2 一把秤：每个节点 − 秤 A，以及共识 − 秤 A，取最大
  L3 两把秤：L2 的基础上再看 共识 − 秤 B、A − B（GLONASS 比 GPS 噪声大约 3.5 倍）
指标：不用单一门槛，用 AUC（场景 vs S0）和 S0 误报 5% 时的检出率。归因：all vs scale 能否分开。
  L2 能用的信息：所有节点一起相对 A 挪动，两种场景一样 → 预期分不开。
  L3 规则：比较 |EWMA(共识 − B)| 和 |EWMA(A − B)| 哪个大，共识 − B 大判节点，A − B 大判秤 A。

预注册（运行前）：
  1. L0 看不见共同偏移（T1）：all、drift、scale 的 AUC ≤ 0.60（所有 δ）
  2. L0 看得见少数和多数：one、min、maj 在 δ ≥ 12 时 AUC ≥ 0.80
  3. L2 看得见全部：所有场景 δ ≥ 12 时 AUC ≥ 0.80；drift 在同 δ 下比 all 弱（T6）
  4. L2 分不开 all 和 scale（T3b）：用 L2 的信号（所有节点相对 A 的挪动方向和大小）归因准确率 ≤ 60%
  5. L3 分得开（T3b）：δ = 24 时归因准确率 ≥ 80%（δ 小时 GLONASS 噪声盖住，不设门槛）

结果（cdd999d 预注册）：
  1 ✅ L0 对 all、drift、scale 的 AUC 0.49–0.51（T1：同伴比较完全看不见共同偏移）
  2 ❌ L0 在 δ = 12：one 0.71、min 0.79、maj 0.76；δ = 24：0.76、0.84、0.84（趋势对，没到 0.80）
  3 大部分：L2 在 δ ≥ 12：min、maj、all、drift、scale 0.81–0.84 ✅；one 0.72、0.78 ❌；drift 比 all 弱 ❌（0.84 对 0.84，没分开，drift 涨到 2δ，且分数有上限）
  4 ❌ 轻微：L2 归因 62–68%（门槛 ≤ 60%；这是事后最优门槛，对 L2 有利，17 个窗口上乐观）
  5 ✅ L3 归因 100%（δ = 6、12、24 全部）
  方法问题：AUC 卡在 0.84、"检出 @ 5% 误报"全 0% 是打分的毛病，不是检测的结果——EWMA 截断 ±3σ 使分数有上限，而 S0 窗口里本来就有真实时钟故障，约三分之一顶满上限；门槛落在上限上，什么都超不过。负样本不干净 → 第 3 步用 BIPM 自己的事件记录查。

用法：~/gut-external/torch215-venv/bin/python ladder.py
"""
import json, os, random, statistics as st, sys
sys.path.insert(0, "../switching")
import torch
import validate_bipm as V
from purge import W


def series_score(x, ncal, signed=False):
    base = st.median(x[:ncal])
    sd = max(1.4826 * st.median(abs(v - base) for v in x[:ncal]), 1e-6)
    S, best, bs = 0.0, 0.0, 0.0
    for i, v in enumerate(x):
        S = (1 - W) * S + W * max(-3.0, min(3.0, (v - base) / sd))
        if i >= ncal and abs(S) > best:
            best, bs = abs(S), S
    return bs if signed else best


def scores(cal, ev, sh, zeta):
    """sh: 评估期每题的节点注入 (题, 节点)；zeta: 秤 A 的注入。"""
    n = len(cal[0][3])
    ncal = len(cal)
    R = torch.stack([r for *_, r in cal] + [ev[i][3] + sh[i] for i in range(len(ev))])
    A = [x[1] for x in cal] + [e[1] + zeta for e in ev]
    B = [x[2] for x in cal] + [e[2] for e in ev]
    peer = max(series_score([(R[t, j] - torch.cat([R[t, :j], R[t, j + 1:]]).median()).item() for t in range(len(R))], ncal) for j in range(n))
    node_a = max(series_score((R[:, j] - torch.tensor(A)).tolist(), ncal) for j in range(n))
    c = R.median(1).values
    cons_a = series_score((c - torch.tensor(A)).tolist(), ncal)
    cons_b = series_score((c - torch.tensor(B)).tolist(), ncal)
    ab = series_score([a - b for a, b in zip(A, B)], ncal)
    return {"L0": peer, "L2": max(node_a, cons_a), "L3": max(node_a, cons_a, cons_b, ab), "cons_a": cons_a, "cons_b": cons_b, "ab": ab, "n_moved": sum(series_score((R[:, j] - torch.tensor(A)).tolist(), ncal) > 1.39 for j in range(n)) / n}


def auc(pos, neg):
    w = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return w / (len(pos) * len(neg))


def main():
    rng = random.Random("ladder")
    X = []
    W_ = list(V.windows("test"))
    print(len(W_), "个窗口-组")
    for gi, w, cal, ev in W_:
        T, n = len(ev), 9
        z = torch.zeros(T, n)
        X.append(("S0", 0.0, scores(cal, ev, z, 0.0)))
        for dl in (6.0, 12.0, 24.0):
            for sc, k in (("one", 1), ("min", 4), ("maj", 6), ("all", 9)):
                sh = torch.zeros(T, n)
                sh[:, rng.sample(range(n), k)] = dl
                X.append((sc, dl, scores(cal, ev, sh, 0.0)))
            ramp = torch.tensor([2 * dl * (t + 1) / T for t in range(T)]).unsqueeze(1).repeat(1, n)
            X.append(("drift", dl, scores(cal, ev, ramp, 0.0)))
            X.append(("scale", dl, scores(cal, ev, torch.zeros(T, n), dl)))
    json.dump(X, open("ladder_results.json", "w"), ensure_ascii=False)
    neg = {L: [x[2][L] for x in X if x[0] == "S0"] for L in ("L0", "L2", "L3")}
    thr = {L: sorted(v)[int(.95 * len(v))] for L, v in neg.items()}
    print("场景 δ   | L0 AUC / 检出@5%误报 | L2 | L3")
    for sc in ("one", "min", "maj", "all", "drift", "scale"):
        for dl in (6.0, 12.0, 24.0):
            s = [x[2] for x in X if x[0] == sc and x[1] == dl]
            print(f"{sc:6s}{dl:5.0f} | " + " | ".join(f"{auc([y[L] for y in s], neg[L]):.2f} / {st.mean(y[L] > thr[L] for y in s):.0%}" for L in ("L0", "L2", "L3")))
    print("\n归因 all vs scale（各 δ）：")
    for dl in (6.0, 12.0, 24.0):
        a = [x[2] for x in X if x[0] == "all" and x[1] == dl]
        s = [x[2] for x in X if x[0] == "scale" and x[1] == dl]
        l3 = (st.mean(y["cons_b"] > y["ab"] for y in a) + st.mean(y["ab"] >= y["cons_b"] for y in s)) / 2
        # L2 能用的：挪动的节点比例和共识相对 A 的挪动大小；用同一数据上最优门槛（对 L2 最有利）
        cand = sorted({y["cons_a"] for y in a + s})
        best = max(((st.mean(y["cons_a"] > c for y in a) + st.mean(y["cons_a"] <= c for y in s)) / 2, c) for c in cand)[0]
        best = max(best, max(((st.mean(y["cons_a"] <= c for y in a) + st.mean(y["cons_a"] > c for y in s)) / 2) for c in cand))
        print(f"  δ={dl:.0f}：L2（事后最优门槛，对它最有利）{best:.0%}；L3 {l3:.0%}")


if __name__ == "__main__":
    main()
