"""T0 的定量最小检验：事先算出可区分程度，预测能不能发现，再和实际比（2026-11-02，库伯）。零拟合。
数据：ladder2 的 SLQA 16 个窗口、BIPM 17 个窗口（这些窗口的 AUC 看过，但预测公式是新的）。注入故障，噪声是真的。
一条比较序列（节点 − 同伴中位、节点 − 秤、共识 − 秤、A − B）的预测：
  μ = 故障形式决定的无噪声偏移（注入向量代入同一比较得到）；se = 1.2533 √(MAD校² / n校 + MAD评² / n评)，MAD校 只来自校准期，MAD评 = MAD校（慢漂移再加上漂移自身的散布 δ·1.4826/2）。
  z = μ / se；这条序列超线的概率 P = P(|N(z, 1)| > t)，t = 3（运行前定死）。
一级检测（L0、L2、L3）有多条序列：P = 1 − Π(1 − P_i)（按独立合并，是近似）。
实际：同样的序列、同样的统计量（ladder2.stat），任一超过 t 算检出。
按 领域 × 级别 × 场景 × δ 分组（S0 也是一组）比较预测检出率和实际检出率。

预注册（运行前）：
  1. 每个领域：组级 Spearman（预测 vs 实际）≥ 0.80
  2. 每个领域：组级 MAE ≤ 0.15
  3. 预测不可区分的组（z = 0：L0 下的 all、drift、scale）实际检出率和 S0 差 ≤ 0.05（注入完全相同，这条是构造决定的，只当流程检查）
  4. 预期不准的地方（写在前面）：S0 上实际误报会高于预测，差值大致是自然故障率（BIPM 比 SLQA 明显）；这不是公式对错，是公式没算自然故障

第一轮结果（e316451）：1 ❌ Spearman SLQA 0.78、BIPM 0.71；2 ✅ MAE 0.066、0.135；3 ✅（构造决定）；4 S0 实际 0.31 / 0.59–0.65，预测 0.02–0.03。
  检验力弱：δ 太大，大多数格预测和实际都是 1.00；Spearman 没处理并列。"差值是自然故障率"未验证，也可能是厚尾 / 时间相关让标准误偏小。
第二轮预注册（运行前）：只测过渡区，δ 改小：SLQA {0.005, 0.01, 0.02, 0.03}，BIPM {1, 2, 3, 4}（ns）；场景 one、min、maj、all、drift、scale。
  预测 = 1 − (1 − P)(1 − b)，b = 没注入时的实际检出率，按窗口奇偶交叉估计（用偶数窗的 b 预测奇数窗，反之亦然）。排除 L0 下预测不可区分的组（构造决定）。
  1. 每个领域：组级 Spearman（并列取平均秩）≥ 0.80   2. 每个领域：组级 MAE ≤ 0.10
  3. 过渡区真的存在：至少一半的组预测落在 [b + 0.05, 0.95]，否则这轮也算检验力不足

第二轮结果（79374ca）：三条都过。SLQA Spearman 0.92、MAE 0.082、过渡区 65%；BIPM 0.86、0.038、52%。
  局限：故障是注入的；预测用了实测基础率 b（不是纯先验，b 按奇偶窗交叉估计）；组之间不独立（L2、L3 共用序列，同一批窗口）；基础率 0.25–0.65 为什么这么高没查清；只有在第一轮检验力不足后才做的第二轮（已记录）。

用法：~/gut-external/torch215-venv/bin/python distinguish.py        （第一轮）
      ~/gut-external/torch215-venv/bin/python distinguish.py v2     （第二轮）
"""
import json, math, random, statistics as st, sys
sys.path.insert(0, "../switching")
import torch
from analytic import Phi
from ladder2 import _mad, bipm_windows, med, slqa_windows, stat

T = 3.0


def p_over(z):
    return 1 - (Phi(T - z) - Phi(-T - z))


def series(R, A, B):
    n = len(R[0])
    out = {}
    for j in range(n):
        out[("peer", j)] = [r[j] - med(r[:j] + r[j + 1:]) for r in R]
        out[("nodeA", j)] = [r[j] - a for r, a in zip(R, A)]
    out[("consA", 0)] = [med(r) - a for r, a in zip(R, A)]
    out[("consB", 0)] = [med(r) - b for r, b in zip(R, B)]
    out[("AB", 0)] = [a - b for a, b in zip(A, B)]
    return out


LEVELS = {"L0": ("peer",), "L2": ("nodeA", "consA"), "L3": ("nodeA", "consA", "consB", "AB")}


def shift_of(key, s, zeta):
    kind, j = key
    if kind == "peer":
        return s[j] - st.median(s[:j] + s[j + 1:])
    if kind == "nodeA":
        return s[j] - zeta
    if kind == "consA":
        return st.median(s) - zeta
    if kind == "consB":
        return st.median(s)
    return zeta


def cell(cal, ev, s, zeta, drift=False):
    """s：评估期节点的注入大小（drift 时是 δ，表示 0 → 2δ 的线性）。返回 {级别: (预测, 实际)}。"""
    n = len(cal[0][2])
    T_ = len(ev)
    Rc = [x[2] for x in cal]
    Ac, Bc = [x[0] for x in cal], [x[1] for x in cal]
    if drift:
        Re = [[x[2][j] + 2 * s[j] * (t + 1) / T_ for j in range(n)] for t, x in enumerate(ev)]
    else:
        Re = [[x[2][j] + s[j] for j in range(n)] for x in ev]
    Ae, Be = [x[0] + zeta for x in ev], [x[1] for x in ev]
    Sc, Se = series(Rc, Ac, Bc), series(Re, Ae, Be)
    out = {}
    pred, act = {}, {}
    for key in Sc:
        c = [v for v in Sc[key] if v == v]
        e = [v for v in Se[key] if v == v]
        if len(c) < 5 or len(e) < 5:
            pred[key], act[key] = 0.0, 0.0
            continue
        m = _mad(c)
        mu = shift_of(key, list(s), zeta)
        extra = (abs(mu) * 1.4826 / 2) if drift else 0.0
        se = 1.2533 * math.sqrt(m ** 2 / len(c) + (m ** 2 + extra ** 2) / len(e))
        pred[key] = p_over(mu / max(se, 1e-9))
        act[key] = stat(c, e) > T
    for L, kinds in LEVELS.items():
        keys = [k for k in Sc if k[0] in kinds]
        q = 1.0
        for k in keys:
            q *= 1 - pred[k]
        out[L] = (1 - q, any(act[k] for k in keys))
    return out


def run(name, windows, deltas, kmin, kmaj):
    rg = random.Random("dist-" + name)
    G = {}
    for cal, ev in windows:
        n = len(cal[0][2])
        def add(sc, d, res):
            for L, (p, a) in res.items():
                G.setdefault((L, sc, d), []).append((p, a))
        add("S0", 0.0, cell(cal, ev, [0.0] * n, 0.0))
        for d in deltas:
            for sc, k in (("one", 1), ("min", kmin), ("maj", kmaj), ("all", n)):
                idx = set(rg.sample(range(n), k))
                add(sc, d, cell(cal, ev, [d if j in idx else 0.0 for j in range(n)], 0.0))
            add("drift", d, cell(cal, ev, [d] * n, 0.0, drift=True))
            add("scale", d, cell(cal, ev, [0.0] * n, d))
    rows = [(k, st.mean(x[0] for x in v), st.mean(x[1] for x in v)) for k, v in G.items()]
    print(f"\n===== {name} =====\n级别 场景 δ      预测  实际")
    for (L, sc, d), p, a in sorted(rows, key=lambda r: (r[0][0], r[0][1], r[0][2])):
        print(f"{L} {sc:6s}{d:6g}  {p:.2f}  {a:.2f}")
    rk = lambda v: torch.tensor(v, dtype=torch.float).argsort().argsort().float()
    P, A = [r[1] for r in rows], [r[2] for r in rows]
    sp = torch.corrcoef(torch.stack([rk(P), rk(A)]))[0, 1].item()
    mae = st.mean(abs(p - a) for p, a in zip(P, A))
    s0 = {L: [a for (k, p, a) in rows if k == (L, "S0", 0.0)][0] for L in LEVELS}
    s0p = {L: [p for (k, p, a) in rows if k == (L, "S0", 0.0)][0] for L in LEVELS}
    inv = [a for (k, p, a) in rows if k[0] == "L0" and k[1] in ("all", "drift", "scale")]
    print(f"1. Spearman {sp:.2f}（≥ 0.80）  2. MAE {mae:.3f}（≤ 0.15）")
    print(f"3. 预测不可区分的组实际 {min(inv):.2f}–{max(inv):.2f}，S0 {s0['L0']:.2f}（差 ≤ 0.05）")
    print(f"4. S0 实际 / 预测：" + "，".join(f"{L} {s0[L]:.2f} / {s0p[L]:.2f}" for L in LEVELS))
    nz = [(p, a) for (k, p, a) in rows if k[1] != "S0"]
    print(f"   扣掉 S0 基础率后的 MAE：{st.mean(abs(1 - (1 - p) * (1 - s0['L0']) - a) for p, a in nz):.3f}（非预注册）")


def avg_rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def run2(name, windows, deltas, kmin, kmaj):
    rg = random.Random("dist2-" + name)
    b = {0: {}, 1: {}}
    per = []
    for wi, (cal, ev) in enumerate(windows):
        n = len(cal[0][2])
        base = cell(cal, ev, [0.0] * n, 0.0)
        for L in LEVELS:
            b[wi % 2].setdefault(L, []).append(base[L][1])
        for d in deltas:
            for sc, k in (("one", 1), ("min", kmin), ("maj", kmaj), ("all", n)):
                idx = set(rg.sample(range(n), k))
                per.append((wi, sc, d, cell(cal, ev, [d if j in idx else 0.0 for j in range(n)], 0.0)))
            per.append((wi, "drift", d, cell(cal, ev, [d] * n, 0.0, drift=True)))
            per.append((wi, "scale", d, cell(cal, ev, [0.0] * n, d)))
    G = {}
    for wi, sc, d, res in per:
        for L, (p, a) in res.items():
            if L == "L0" and sc in ("all", "drift", "scale"):
                continue
            bb = st.mean(b[1 - wi % 2][L])
            G.setdefault((L, sc, d), []).append((1 - (1 - p) * (1 - bb), a, bb))
    rows = [(k, st.mean(x[0] for x in v), st.mean(x[1] for x in v), st.mean(x[2] for x in v)) for k, v in G.items()]
    print(f"\n===== {name}（过渡区）=====\n级别 场景 δ      预测  实际")
    for (L, sc, d), p, a, _ in sorted(rows, key=lambda r: r[0]):
        print(f"{L} {sc:6s}{d:6g}  {p:.2f}  {a:.2f}")
    P, A = [r[1] for r in rows], [r[2] for r in rows]
    sp = torch.corrcoef(torch.stack([torch.tensor(avg_rank(P)), torch.tensor(avg_rank(A))]))[0, 1].item()
    print(f"1. Spearman（并列取平均）{sp:.2f}（≥ 0.80）  2. MAE {st.mean(abs(p - a) for p, a in zip(P, A)):.3f}（≤ 0.10）")
    print(f"3. 预测落在过渡区的组 {st.mean(r[3] + 0.05 <= r[1] <= 0.95 for r in rows):.0%}（≥ 50%）；基础率 " + str({L: round(st.mean(b[0][L] + b[1][L]), 2) for L in LEVELS}))


if __name__ == "__main__":
    if sys.argv[1:] == ["v2"]:
        run2("SLQA", list(slqa_windows()), (0.005, 0.01, 0.02, 0.03), 3, 5)
        run2("BIPM", list(bipm_windows()), (1.0, 2.0, 3.0, 4.0), 4, 6)
    else:
        run("SLQA", list(slqa_windows()), (0.05, 0.10, 0.20), 3, 5)
        run("BIPM", list(bipm_windows()), (6.0, 12.0, 24.0), 4, 6)
