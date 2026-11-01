"""规律的分类版：任务是在 K 个类别里选一个（2026-11-01，库伯：试非数值任务）。

结构和数值版同一副骨架：
  节点：诚实节点以 1 − ε 答对（错时均匀错到其他类）；投毒节点在故障题上以概率 τ 报同一个错误标签（抱团攻击），其余时候和诚实一样。
  约束：独立的带噪信息源，错误率 ε_c（错时均匀）。
  裁判（软）：朴素贝叶斯投票——每个标签的分数 = Σ 节点 log P(报告 | 标签) + log P(约束 | 标签)，节点和约束的错误率都从校准期估。
  推导版预测：跨题汇总认谁诚实（每个节点和约束的一致率，对比"诚实时应有的一致率"，标准误折成 z → 权重 exp(−½ z²)），
    按权重抽诚实子集，真值后验 ∝ P(约束 | c) · Π_{诚实} P(报告 | c)，预测被骗 = 后验下裁判输出 ≠ 真值的概率。
  不用真值的估计：校准期（前 100 题、已知干净）上 约束和节点多数票的一致率 → ε_c；各节点和多数票的一致率 → 各自 ε。
玩具：随机 K ∈ {2, 3, 5}、N ∈ {5, 7, 9}、诚实数 1 … N − 1、ε_h ∈ [0.02, 0.3]、τ ∈ [0.5, 1]、ε_c ∈ [0.01, 0.4]；600 题、故障题 30%；每集 120 个配置（新种子 cat）。
压力：投毒占多数 / 约束很弱（ε_c ∈ [0.3, 0.6]）/ ζ = 1（约束在故障题上也跟着报那个错误标签）。

预注册（11-01，运行前）：
  1. 无压力集：推导版 MAE ≤ 0.05；投票的被骗率平均 > 推导版对应裁判的被骗率（约束有用）
  2. 投毒占多数集：推导版 MAE ≤ 0.07
  3. 约束很弱集：推导版 MAE ≤ 0.10（可识别性下降，允许变差）
  4. ζ = 1 集：推导版乐观，平均误差（实际 − 预测）≥ +0.20

第一稿结果（db252fb，4 条对 0 条）：MAE 0.083 / 0.098 / 0.186、ζ = 1 时 MAE 0.54。全知预测者（真诚实集 + 真错误率）MAE 0.020 / 0.036
  → 式子结构对，毛病在认谁诚实：拿"诚实时应有的一致率"做对比，而 ε 只从 100 道校准题估、误差 ±0.04，足以让诚实节点的 z 到 2 左右、权重被压低。
第二稿（derive(..., own=True)）：诚实度改成和节点自己的常态比（数值版同一原则）——校准期后各节点和约束的一致率，减去它自己在校准期的一致率；
  诚实节点不变，投毒节点下降。
第二稿预注册（运行前；新种子 cat2）：同第一稿四条。

用法：~/gut-external/torch215-venv/bin/python categorical.py        （第一稿）
      ~/gut-external/torch215-venv/bin/python categorical.py v2     （第二稿）
"""
import math
import random
import statistics as st

import torch


def gen(k, tag, majority=None, weak=False, zeta=False, items=600):
    rng = random.Random(f"{tag}-{k}")
    g = torch.Generator().manual_seed(hash((tag, k)) % (2 ** 31))
    K, N = rng.choice([2, 3, 5]), rng.choice([5, 7, 9])
    if majority is True:
        h = rng.randint(1, (N - 1) // 2)
    else:
        h = rng.randint(1, N - 1)
    eh = [rng.uniform(0.02, 0.3) for _ in range(N)]
    tau = rng.uniform(0.5, 1.0)
    ec = rng.uniform(0.3, 0.6) if weak else rng.uniform(0.01, 0.4)
    rows = []
    for t in range(items):
        c = rng.randrange(K)
        fault = t >= 100 and rng.random() < 0.3
        wrong = (c + 1 + rng.randrange(K - 1)) % K
        r = []
        for i in range(N):
            if i >= h and fault and rng.random() < tau:
                r.append(wrong)
            elif rng.random() < eh[i]:
                r.append((c + 1 + rng.randrange(K - 1)) % K)
            else:
                r.append(c)
        if zeta and fault:
            p = wrong
        elif rng.random() < ec:
            p = (c + 1 + rng.randrange(K - 1)) % K
        else:
            p = c
        rows.append((fault, c, r, p))
    return rows, K, N, h


def lik(lbl, c, e, K):
    return (1 - e) if lbl == c else e / (K - 1)


def estimate(rows, K, N):
    cal = rows[:100]
    maj = [max(range(K), key=lambda c: sum(x == c for x in r)) for _, _, r, _ in cal]
    a_c = st.mean(p == m for (_, _, _, p), m in zip(cal, maj))
    ec = min(0.9, max(0.005, 1 - a_c))
    en = [min(0.9, max(0.005, 1 - st.mean(r[i] == m for (_, _, r, _), m in zip(cal, maj)))) for i in range(N)]
    return ec, en


def judge(r, p, ec, en, K):
    sc = [sum(math.log(lik(x, c, e, K)) for x, e in zip(r, en)) + math.log(lik(p, c, ec, K)) for c in range(K)]
    return max(range(K), key=lambda c: sc[c])


def derive(rows, K, N, ec, en, M=64, own=False):
    agree = torch.tensor([[float(r[i] == p) for i in range(N)] for _, _, r, p in rows])
    if own:                                               # 第二稿：和节点自己在校准期的一致率比
        a0, a1 = agree[:100], agree[100:]
        se = torch.sqrt(a0.var(0) / len(a0) + a1.var(0) / len(a1))
        w = torch.exp(-0.5 * (((a1.mean(0) - a0.mean(0)).clamp(max=0)) / se.clamp(min=1e-9)) ** 2)
    else:                                                 # 第一稿：和"诚实时应有的一致率"比
        exp_a = torch.tensor([(1 - ec) * (1 - e) + ec * e / (K - 1) for e in en])
        se = agree.std(0) / math.sqrt(len(rows))
        w = torch.exp(-0.5 * ((agree.mean(0) - exp_a) / se.clamp(min=1e-9)) ** 2)
    g = torch.Generator().manual_seed(0)
    S = (torch.rand(M, N, generator=g) < w).float()
    S[S.sum(1) == 0, int(torch.argmax(w))] = 1.0
    out = []
    for fault, _, r, p in rows:
        if not fault:
            continue
        o = judge(r, p, ec, en, K)
        acc = []
        for s in S.tolist():
            post = [math.log(lik(p, c, ec, K)) + sum(math.log(lik(x, c, e, K)) for x, e, si in zip(r, en, s) if si) for c in range(K)]
            mx = max(post)
            z = [math.exp(v - mx) for v in post]
            acc.append(1 - z[o] / sum(z))
        out.append(st.mean(acc))
    return st.mean(out)


def actual(rows, K, ec, en):
    j = [judge(r, p, ec, en, K) != c for f, c, r, p in rows if f]
    v = [max(range(K), key=lambda x: sum(y == x for y in r)) != c for f, c, r, p in rows if f]
    return st.mean(j), st.mean(v)


def main(own=False):
    sets = (("无压力", {}), ("投毒占多数", {"majority": True}), ("约束很弱", {"weak": True}), ("ζ=1", {"zeta": True}))
    for tag, kw in sets:
        err, acts, votes = [], [], []
        for k in range(120):
            rows, K, N, h = gen(k, "cat2" if own else "cat", **kw)
            ec, en = estimate(rows, K, N)
            a, v = actual(rows, K, ec, en)
            err.append(a - derive(rows, K, N, ec, en, own=own))
            acts.append(a)
            votes.append(v)
        print(f"{tag:8s}：推导版 MAE {st.mean(abs(x) for x in err):.3f}，平均误差（实际 − 预测）{st.mean(err):+.3f}；裁判被骗 {st.mean(acts):.0%}，投票被骗 {st.mean(votes):.0%}")


if __name__ == "__main__":
    import sys
    main(own=sys.argv[1:] == ["v2"])
