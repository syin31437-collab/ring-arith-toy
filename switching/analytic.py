"""规律的推导版（零拟合参数）：漏检 = 裁判选中的那一团是故障团的概率。

每道故障题（全部只用观测量；真值只用来计分）：
  1. 分团：节点报告排序，在最大缺口处分成两团；缺口 ≤ 3 × 节点间分散（无故障题上各题报告的 MAD 的中位数）算没分开。
  2. 裁判输出由报告和约束 p 直接算出（软 / 硬2，与 hetero.judge_out 相同），看它落在哪一团。
  3. 哪团诚实：约束误差 e = p − 真值 ~ N(b, σ_c²)；b、σ_c 用无故障题上 (p − 报告中位数) 的中位数和 MAD 估计（不用真值）。
     团 X 诚实的似然 = N(p − X 的中位数; b, σ_c² + X 内分散² / |X|)；先验 50 / 50（谁占多数从内部看不见）。
  4. 预测漏检 = P(选中的团是故障团) · 1[两团距离 > t] + P(选中的团诚实) · P(|N(0, X 内分散² / |X|)| > t)，t = δ/2。
     没分开的题：两团看不出 → 输出偏多少看不见 → 预测 = P(|N(0, σ_out²)| > t)，σ_out = 无故障题上 (输出 − 报告中位数) 的 MAD。
  一道题的后验落在 [0.2, 0.8] 记为"预测不了"（约束分不清哪团诚实，结果取决于看不见的投毒比例）。

第二版 derive2（看过第一版结果后改，11-01）：第一版把"没分开"当成"故障没有伤害"——盲测三 δ = 2% 实际 70%、推导 3%：
  5 个节点一起偏、偏得比分团门槛小，整团被拖走，看不见不等于没伤害。改成一个统一的贝叶斯式子：
  真值的位置 = 节点给的（每一团是一个候选，各 50%；没分开时只说"真值在这团附近一个分团门槛以内"，先验宽度 = 门槛）
            × 约束给的 N(p − b, σ_c²)；预测漏检 = 后验下 |裁判输出 − 真值| > δ/2 的概率。零拟合参数，唯一常数 = 分团门槛 3 × 节点间分散。
第二版预注册（运行前；评估用新种子 analytic2，盲测数据都看过了，只作诊断、不计）：
  1. 无压力基线 MAE ≤ 拟合版 H + 0.02（软、硬2）
  2. 在 无压力 / 约束 ×100 / 抱团 / 厚尾 df=1 / N=31 / δ 分散 σ_log=1.5 六个集上，第二版的 MAE 都低于第一版
  3. ζ = 1 照样乐观（新种子 ζ = 1 压力集，实际 − 预测 ≥ +0.30）
  4. 约束 ×100 时，第二版的 MAE 低于拟合版 H（拟合曲线在低 ρ 区外推错，推导版应该更好）

第三版 derive3（11-01）：
  1. 输出区间：谁占多数从内部看不见 → 不再用 50/50 点预测，而是把"大团诚实"的先验从 0 扫到 1，给出 [下限, 上限]。
     约束强时两端重合（能预测）；约束弱时区间变宽（如实说预测不了）。盲测三 δ = 5% 的残差就在这种区域里（ρ ≈ 1，后验 ≈ 0.5）。
  2. 修 σ_c：Var(p − 报告中位数) = σ_c² + Var(报告中位数)，后一项在无故障题上对节点做 bootstrap 估计、扣掉。
第三版预注册（运行前；新种子 analytic3，每集 120 个配置；盲测数据看过，只作诊断）：
  1. 覆盖：前提满足的集合（无压力 / 约束×100 / 抱团 / 厚尾 df=1 / N=31 / δ 分散 1.5）里，
     每个集 ≥ 90% 的配置实际漏检落在 [下限 − 0.05, 上限 + 0.05] 内（0.05 = 抽样误差）
  2. 宽度：无压力集区间宽度中位数 ≤ 0.15；约束×100 集 ≥ 0.40
  3. ζ = 1：≥ 50% 的配置实际漏检 > 上限 + 0.05（前提被违反 → 落到区间外）
  4. 修 σ_c 后，无压力集点预测（先验 50/50）的平均误差 |实际 − 预测| 偏差 < 0.05（第二版是 −0.09 / −0.06）

第三版第二稿 derive3(…, v31=True)（看过第三版结果后改）：第三版点预测在前提满足的玩具上 MAE 0.033–0.069，
  是拟合版 H 的 1/2–1/3（σ_c 修好了）；但区间设计有两个错：
  a. 先验扫到 0 和 1 是教条先验，似然再强也翻不过来（盲测二 δ = 40% 给出 [0%, 100%]）→ 改扫 [0.1, 0.9]（先验赔率最多 9 : 1）
  b. 没分开时先验宽度用了整个分团门槛——那是隐藏偏移的上限，不是标准差 → 约束弱时偏悲观（−0.10）→ 改成门槛内均匀分布，标准差 = 门槛 / √3
第三版第二稿预注册（运行前；新种子 analytic4）：
  1. 前提满足的六个集，每个集 ≥ 90% 的配置落在 [下限 − 0.05, 上限 + 0.05]
  2. 约束 ×100 点预测 MAE ≤ 0.15（第三版 0.21 / 0.23）
  3. 无压力点预测 MAE ≤ 0.05（第三版 0.038 / 0.046，不能变差）
  4. ζ = 1 照样乐观：点预测平均误差 ≥ +0.20

第四版 derive4（11-01）：第三版第二稿 2/4；两项诊断——
  · 覆盖率 ≥ 90% 的判据本身过不了：实际漏检是约 120 道故障题的二项估计，完美预测者在 ±0.05 下的期望覆盖率只有 87%（预注册写错了）
  · 复刻盲测三（玩具，真值全知）：b、σ_c 都估得准，但 δ = 5% 时 72% 的故障题按最大缺口分不开——两团重叠、最大缺口小，
    团间距离（0.021）却大于门槛（0.016）→ "分不开 = 真值在门槛内"不成立。
  单道题上确实分不清（约束噪声 ≈ 团间距离），但**谁诚实在每道题上都一样**：60 道题汇总，约束证据的不确定度降到 0.02/√60 ≈ 0.003。
  → 跨题汇总学"谁诚实"（就是库伯最早的"从错误里找出错误来源"、站队表 P 的机制）。去掉分团门槛：
  1. 每个节点的诚实权重：全部题目（不需要知道哪道是故障题）上 x_i = r_i − (p − b) 的平均，
     和"诚实时为 0"比：w_i = N(均值 x_i; 0, SE_i)，SE_i = 该节点 x_i 的标准差 / √题数；归一化。
  2. 每道题真值后验 ∝ Σ_i w_i · N(真值; r_i, σ_i²) × N(真值; p − b, σ_c²)，σ_i = 该节点无故障题上偏离报告中位数的 MAD。
  3. 预测漏检 = 后验下 |裁判输出 − 真值| > δ/2 的概率。零拟合参数，也没有门槛常数。
第四版预注册（运行前；新种子 analytic5；盲测三复刻用新的生成器种子）：
  1. 盲测三复刻 δ = 2% / 5% / 10%：每档 |实际 − 预测| ≤ 0.10（第三版第二稿 10 / 38 / 16 个百分点）
  2. 无压力点预测 MAE ≤ 0.05（软、硬2）
  3. 约束 ×100 点预测 MAE ≤ 0.15
  4. 无压力集 ±0.05 内的比例 ≥ 完美预测者上限 − 10 个百分点
  5. ζ = 1 照样乐观：平均误差 ≥ +0.20

第五版 derive5（11-01）：第四版把真值当成"某一个节点是对的"（混合），诚实节点之间没有互相平均——
  它们是同一个真值的独立测量，应该相乘（精度相加），真值不确定度 = σ / √诚实节点数（就是 λ 的平均效应）。
  诊断：硬2 复刻盲测三 δ = 5%，91% 的题留下全部 7 个节点、输出全体中位数、漏检 99%，第四版只给 77%。
  改法：按诚实权重抽 M = 64 个诚实子集（每个节点以 w_i 入选，至少留 1 个），每个子集里诚实节点相乘 × 约束似然
  → 高斯后验 → 漏检概率，再对子集平均。可识别时子集固定、诚实节点平均；不可识别时子集各不相同、预测自动变宽。
第五版预注册（运行前；新种子 analytic6）：
  1. 硬2 复刻盲测三 δ = 5% |实际 − 预测| ≤ 0.10（第四版 0.17）；软三档每档 ≤ 0.10
  2. 无压力 MAE ≤ 0.03，N = 31 MAE ≤ 0.03（第四版 0.024 / 0.025、0.031 / 0.041）
  3. 约束 ×100：全知预测者（真实诚实集，相乘）MAE ≤ 0.05；第五版在可识别度 z ≥ 2 的配置上 MAE ≤ 0.06
  4. ζ = 1 照样乐观：平均误差 ≥ +0.20

和拟合曲线比较：H（het2 拟合）在同一批配置上的 MAE。RAIM 的单颗故障检验是这里 |故障团| = 1、硬裁判的特例。

预注册（11-01，运行前）：
  1. 两次盲测（盲测二 Z0、盲测三 R0），推导版的平均 |误差| ≤ 拟合版 H（软、硬2 各自比）；盲测三 δ = 5% 软裁判 |误差| ≤ 0.15
  2. 压力玩具无压力基线：推导版 MAE ≤ H 的 MAE + 0.03（软、硬2）
  3. ζ = 1（盲测 Z1、R1）：推导版照样乐观，实际 − 预测 ≥ +0.30
  4. 约束 ×100（ρ 中位约 0.1）时"预测不了"的故障题占比，高于无压力基线的 2 倍以上

用法：~/gut-external/torch215-venv/bin/python analytic.py        （第一版）
      ~/gut-external/torch215-venv/bin/python analytic.py v2     （第二版）
      ~/gut-external/torch215-venv/bin/python analytic.py v3     （第三版：区间）
      ~/gut-external/torch215-venv/bin/python analytic.py v4     （第四版：跨题汇总学谁诚实）
      ~/gut-external/torch215-venv/bin/python analytic.py v5     （第五版：诚实节点相乘）
"""
import math
import os
import statistics as st
import sys

import torch

from failmap import curve, stress_toy
from hetero import judge_out, measure, predict

HERE = os.path.dirname(os.path.abspath(__file__))
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
mad = lambda xs: 1.4826 * st.median(abs(x - st.median(xs)) for x in xs)


def npdf(x, s):
    return math.exp(-0.5 * (x / s) ** 2) / s


def split(r, thr):
    v, _ = torch.sort(r)
    gaps = v[1:] - v[:-1]
    k = int(torch.argmax(gaps))
    return None if gaps[k] <= thr else (v[:k + 1], v[k + 1:])


def derive(rows, judge, s):
    """rows = [(是否故障, p, 报告, δ_k)]（任意平移都行：只用差值）。返回 (预测漏检, 预测不了的占比)。"""
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, (1.4826 * (dev - dev.median(0).values).abs().median(0).values).min().item())
    spread = st.median(1.4826 * (r - r.median()).abs().median().item() for _, _, r, _ in clean)
    e = [p - r.median().item() for _, p, r, _ in clean]
    b, sc = st.median(e), max(mad(e), 1e-9)
    s_out = max(mad([judge_out(r, torch.tensor(p), s, "软" if judge == "软" else "硬") - r.median().item() for _, p, r, _ in clean]), 1e-9)
    jd = "软" if judge == "软" else "硬"
    miss, unsure = [], []
    for _, p, r, dk in bad:
        t = dk / 2
        cl = split(r, 3 * max(spread, 1e-9))
        if cl is None:
            miss.append(2 * (1 - Phi(t / s_out)))
            unsure.append(False)
            continue
        o = judge_out(r, torch.tensor(p), s, jd)
        A, B = cl
        X, Y = (A, B) if abs(o - A.median().item()) <= abs(o - B.median().item()) else (B, A)
        sx = lambda C: max(1.4826 * (C - C.median()).abs().median().item(), spread) / math.sqrt(len(C))
        lx = npdf(p - X.median().item() - b, math.sqrt(sc ** 2 + sx(X) ** 2))
        ly = npdf(p - Y.median().item() - b, math.sqrt(sc ** 2 + sx(Y) ** 2))
        post = lx / (lx + ly) if lx + ly > 0 else 0.5
        gap = abs(X.median().item() - Y.median().item())
        miss.append((1 - post) * float(gap > t) + post * 2 * (1 - Phi(t / sx(X))))
        unsure.append(0.2 <= post <= 0.8)
    return st.mean(miss), st.mean(unsure)


def derive2(rows, judge, s):
    """第二版：统一的贝叶斯式子（见文件头）。返回预测漏检。"""
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, (1.4826 * (dev - dev.median(0).values).abs().median(0).values).min().item())
    jd = "软" if judge == "软" else "硬"
    spread = max(st.median(1.4826 * (r - r.median()).abs().median().item() for _, _, r, _ in clean), 1e-9)
    e = [p - r.median().item() for _, p, r, _ in clean]
    b, sc = st.median(e), max(mad(e), 1e-9)
    thr = 3 * spread
    out = []
    for _, p, r, dk in bad:
        t, o, c0 = dk / 2, judge_out(r, torch.tensor(p), s, jd), p - b      # c0 = 约束给的真值中心
        cl = split(r, thr)
        comps = [(C.median().item(), max(1.4826 * (C - C.median()).abs().median().item(), spread) / math.sqrt(len(C))) for C in cl] \
            if cl else [(r.median().item(), thr)]
        ws, ms = [], []
        for mu, sg in comps:
            ws.append(npdf(mu - c0, math.sqrt(sg ** 2 + sc ** 2)))
            v = 1 / (1 / sg ** 2 + 1 / sc ** 2)
            m, sd = v * (mu / sg ** 2 + c0 / sc ** 2), math.sqrt(v)
            ms.append(1 - (Phi((t - (o - m)) / sd) - Phi((-t - (o - m)) / sd)))
        tot = sum(ws)
        out.append(sum(w * x for w, x in zip(ws, ms)) / tot if tot > 0 else st.mean(ms))
    return st.mean(out)


def sigma_c(clean):
    """约束噪声：Var(p − 报告中位数) 扣掉报告中位数自己的方差（对节点 bootstrap 估计），稳健版。"""
    e = [p - r.median().item() for _, p, r, _ in clean]
    g = torch.Generator().manual_seed(0)
    vm = []
    for _, _, r, _ in clean[:200]:
        idx = torch.randint(0, len(r), (200, len(r)), generator=g)
        vm.append(r[idx].median(1).values.var().item())
    tot = mad(e) ** 2
    return st.median(e), math.sqrt(max(tot - st.mean(vm), 0.01 * tot, 1e-18))


def derive3(rows, judge, s, v31=False):
    """第三版：返回 (下限, 点预测, 上限)。先验 π = "大团是诚实团"的概率，扫 0 / 0.5 / 1。"""
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, (1.4826 * (dev - dev.median(0).values).abs().median(0).values).min().item())
    jd = "软" if judge == "软" else "硬"
    spread = max(st.median(1.4826 * (r - r.median()).abs().median().item() for _, _, r, _ in clean), 1e-9)
    b, sc = sigma_c(clean)
    thr = 3 * spread
    res = {0.1: [], 0.5: [], 0.9: []} if v31 else {0.0: [], 0.5: [], 1.0: []}
    for _, p, r, dk in bad:
        t, o, c0 = dk / 2, judge_out(r, torch.tensor(p), s, jd), p - b
        cl = split(r, thr)
        if cl:
            A, B = cl
            cl = (A, B) if len(A) >= len(B) else (B, A)              # 第一个 = 大团
        comps = [(C.median().item(), max(1.4826 * (C - C.median()).abs().median().item(), spread) / math.sqrt(len(C))) for C in cl] \
            if cl else [(r.median().item(), thr / math.sqrt(3) if v31 else thr)]
        lik, ms = [], []
        for mu, sg in comps:
            lik.append(npdf(mu - c0, math.sqrt(sg ** 2 + sc ** 2)))
            v = 1 / (1 / sg ** 2 + 1 / sc ** 2)
            m, sd = v * (mu / sg ** 2 + c0 / sc ** 2), math.sqrt(v)
            ms.append(1 - (Phi((t - (o - m)) / sd) - Phi((-t - (o - m)) / sd)))
        for pi in res:
            pr = [pi, 1 - pi] if len(comps) == 2 else [1.0]
            ws = [a * b_ for a, b_ in zip(pr, lik)]
            tot = sum(ws)
            res[pi].append(sum(w * x for w, x in zip(ws, ms)) / tot if tot > 0 else st.mean(ms))
    v = {pi: st.mean(x) for pi, x in res.items()}
    return min(v.values()), v[0.5], max(v.values())


def derive4(rows, judge, s):
    """第四版：跨题汇总学"谁诚实"，不分团。返回预测漏检。"""
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, (1.4826 * (dev - dev.median(0).values).abs().median(0).values).min().item())
    jd = "软" if judge == "软" else "硬"
    b, sc = sigma_c(clean)
    X = torch.stack([r - (p - b) for _, p, r, _ in rows])                      # 全部题目，不分故障与否
    se = X.std(0) / math.sqrt(len(rows))
    w = torch.exp(-0.5 * (X.mean(0) / se.clamp(min=1e-12)) ** 2)
    w = w / w.sum() if w.sum() > 0 else torch.full_like(w, 1 / len(w))
    D = torch.stack([r - r.median() for _, _, r, _ in clean])
    spread = max(st.median(1.4826 * d.abs().median().item() for d in D), 1e-9)
    sig = (1.4826 * (D - D.median(0).values).abs().median(0).values).clamp(min=0.5 * spread)
    out = []
    for _, p, r, dk in bad:
        t, o, c0 = dk / 2, judge_out(r, torch.tensor(p), s, jd), p - b
        v = 1 / (1 / sig ** 2 + 1 / sc ** 2)
        m = v * (r / sig ** 2 + c0 / sc ** 2)
        lik = w * torch.exp(-0.5 * (r - c0) ** 2 / (sig ** 2 + sc ** 2)) / torch.sqrt(sig ** 2 + sc ** 2)
        sd = torch.sqrt(v)
        ms = torch.tensor([1 - (Phi((t - (o - mi)) / si) - Phi((-t - (o - mi)) / si)) for mi, si in zip(m.tolist(), sd.tolist())])
        out.append((lik * ms).sum().item() / lik.sum().item() if lik.sum() > 0 else ms.mean().item())
    return st.mean(out)


def derive5(rows, judge, s, honest=None, M=64):
    """第五版：诚实子集里诚实节点相乘（互相平均）。honest 给定时 = 全知预测者。返回预测漏检。"""
    clean, bad = [x for x in rows if not x[0]], [x for x in rows if x[0]]
    if judge == "硬2":
        dev = torch.stack([r - p for _, p, r, _ in clean])
        s = max(s, (1.4826 * (dev - dev.median(0).values).abs().median(0).values).min().item())
    jd = "软" if judge == "软" else "硬"
    b, sc = sigma_c(clean)
    n = len(rows[0][2])
    if honest is None:
        X = torch.stack([r - (p - b) for _, p, r, _ in rows])
        se = X.std(0) / math.sqrt(len(rows))
        w = torch.exp(-0.5 * (X.mean(0) / se.clamp(min=1e-12)) ** 2)
        g = torch.Generator().manual_seed(0)
        S = (torch.rand(M, n, generator=g) < w).float()
        S[S.sum(1) == 0, int(torch.argmax(w))] = 1.0
    else:
        S = honest.float().unsqueeze(0)
    D = torch.stack([r - r.median() for _, _, r, _ in clean])
    spread = max(st.median(1.4826 * d.abs().median().item() for d in D), 1e-9)
    prec = 1 / (1.4826 * (D - D.median(0).values).abs().median(0).values).clamp(min=0.5 * spread) ** 2
    out = []
    for _, p, r, dk in bad:
        t, o, c0 = dk / 2, judge_out(r, torch.tensor(p), s, jd), p - b
        P = S @ prec + 1 / sc ** 2
        m = (S @ (prec * r) + c0 / sc ** 2) / P
        sd = 1 / torch.sqrt(P)
        out.append(st.mean(1 - (Phi((t - (o - mi)) / si) - Phi((-t - (o - mi)) / si)) for mi, si in zip(m.tolist(), sd.tolist())))
    return st.mean(out)


def main5():
    torch.manual_seed(15)
    sets = (("无压力", {}), ("约束×100", {"cmult": 100.0}), ("抱团", {"same": True}), ("厚尾df=1", {"df": 1}),
            ("N=31", {"n": 31}), ("δ分散1.5", {"dvar": 1.5}), ("ζ=1", {"zeta": 1.0}))
    for judge in ("软", "硬2"):
        print(f"\n===== {judge}（新种子 analytic6，每集 120 个配置） =====")
        print("集合        第五版 MAE / 偏差   第四版 MAE   全知 MAE   z≥2：第五版 MAE（占比）")
        for tag, kw in sets:
            e5, e4, eo, hiz = [], [], [], []
            for k in range(120):
                rows, delta, sc, nc = stress_toy(k, judge, tag="analytic6", raw="nc", **kw)
                m = measure(rows, delta, judge, sc)
                n = len(rows[0][2])
                e5.append(m["miss"] - derive5(rows, judge, sc))
                e4.append(m["miss"] - derive4(rows, judge, sc))
                eo.append(m["miss"] - derive5(rows, judge, sc, honest=torch.tensor([i < nc for i in range(n)])))
                clean = [x for x in rows if not x[0]]
                b, _ = sigma_c(clean)
                X = torch.stack([r - (p - b) for _, p, r, _ in rows])
                if (X.mean(0).abs() / (X.std(0) / math.sqrt(len(rows))).clamp(min=1e-12)).max().item() >= 2:
                    hiz.append(abs(e5[-1]))
            print(f"{tag:10s}  {st.mean(abs(x) for x in e5):.3f} / {st.mean(e5):+.2f}     {st.mean(abs(x) for x in e4):.3f}       {st.mean(abs(x) for x in eo):.3f}      "
                  f"{st.mean(hiz) if hiz else float('nan'):.3f}（{len(hiz) / 120:.0%}）")
        g = torch.Generator().manual_seed(2027)
        cells = []
        for dd in (0.02, 0.05, 0.1):
            ld = math.log10(1 + dd)
            rows = []
            for _ in range(3000):
                f = torch.rand(1, generator=g).item() < 0.5
                p = 0.0041 + 0.0201 * torch.randn(1, generator=g).item()
                r = 0.0062 * torch.randn(7, generator=g) + (torch.tensor([0, 0, 1, 1, 1, 1, 1.]) * ld if f else 0)
                rows.append((f, p, r, ld))
            cells.append(f"δ={dd:.0%} 实际 {measure(rows, ld, judge, 0.0201)['miss']:.0%} 第五版 {derive5(rows, judge, 0.0201):.0%}")
        print("盲测三复刻（新生成器种子）：" + "   ".join(cells))
        for v, z in ((2, "Z0"), (3, "R0")):
            data, deltas = blind_sets(v)
            clean = data[(z, 0.0)]
            s = max(mad([p for _, p, _, _ in clean]), 1e-6)
            cells = []
            for d in deltas[1:]:
                ld = math.log10(1 + d)
                rows = [(False, p, r, ld) for _, p, r, _ in clean] + data[(z, d)]
                cells.append(f"δ={d:.0%} 实际 {measure(rows, ld, judge, s)['miss']:.0%} 第五版 {derive5(rows, judge, s):.0%}")
            print(f"（诊断，已看过）盲测{'二' if v == 2 else '三'} {z}：" + "  ".join(cells))


def main4():
    torch.manual_seed(13)
    sets = (("无压力", {}), ("约束×100", {"cmult": 100.0}), ("抱团", {"same": True}), ("厚尾df=1", {"df": 1}),
            ("N=31", {"n": 31}), ("δ分散1.5", {"dvar": 1.5}), ("ζ=1", {"zeta": 1.0}))
    for judge in ("软", "硬2"):
        print(f"\n===== {judge}（新种子 analytic5，每集 120 个配置） =====")
        print("集合        第四版 MAE / 偏差    ±0.05 内   完美预测者上限    第三版第二稿 MAE")
        for tag, kw in sets:
            e4, e3, cov, ceil = [], [], [], []
            for k in range(120):
                rows, delta, sc = stress_toy(k, judge, tag="analytic5", raw=True, **kw)
                m = measure(rows, delta, judge, sc)
                e4.append(m["miss"] - derive4(rows, judge, sc))
                e3.append(m["miss"] - derive3(rows, judge, sc, True)[1])
                n, q = sum(1 for x in rows if x[0]), m["miss"]
                se_ = math.sqrt(max(q * (1 - q), 1e-6) / n)
                cov.append(abs(e4[-1]) <= 0.05)
                ceil.append(2 * Phi(0.05 / se_) - 1)
            print(f"{tag:10s}  {st.mean(abs(x) for x in e4):.3f} / {st.mean(e4):+.2f}      {st.mean(cov):5.0%}      {st.mean(ceil):5.0%}          {st.mean(abs(x) for x in e3):.3f}")
        g = torch.Generator().manual_seed(2026)
        cells = []
        for dd in (0.02, 0.05, 0.1):
            ld = math.log10(1 + dd)
            rows = []
            for _ in range(3000):
                f = torch.rand(1, generator=g).item() < 0.5
                p = 0.0041 + 0.0201 * torch.randn(1, generator=g).item()
                r = 0.0062 * torch.randn(7, generator=g) + (torch.tensor([0, 0, 1, 1, 1, 1, 1.]) * ld if f else 0)
                rows.append((f, p, r, ld))
            m = measure(rows, ld, judge, 0.0201)
            cells.append(f"δ={dd:.0%} 实际 {m['miss']:.0%} 第四版 {derive4(rows, judge, 0.0201):.0%} 第三版第二稿 {derive3(rows, judge, 0.0201, True)[1]:.0%}")
        print("盲测三复刻（新生成器种子）：" + "   ".join(cells))
        for v, z in ((2, "Z0"), (3, "R0")):
            data, deltas = blind_sets(v)
            clean = data[(z, 0.0)]
            s = max(mad([p for _, p, _, _ in clean]), 1e-6)
            cells = []
            for d in deltas[1:]:
                ld = math.log10(1 + d)
                rows = [(False, p, r, ld) for _, p, r, _ in clean] + data[(z, d)]
                cells.append(f"δ={d:.0%} 实际 {measure(rows, ld, judge, s)['miss']:.0%} 第四版 {derive4(rows, judge, s):.0%}")
            print(f"（诊断，已看过）盲测{'二' if v == 2 else '三'} {z}：" + "  ".join(cells))


def main3(v31=False):
    torch.manual_seed(12 if v31 else 11)
    tag3 = "analytic4" if v31 else "analytic3"
    sets = (("无压力", {}), ("约束×100", {"cmult": 100.0}), ("抱团", {"same": True}), ("厚尾df=1", {"df": 1}),
            ("N=31", {"n": 31}), ("δ分散1.5", {"dvar": 1.5}), ("ζ=1", {"zeta": 1.0}))
    for judge in ("软", "硬2"):
        print(f"\n===== {judge}（新种子 {tag3}，每集 120 个配置） =====")
        print("集合        覆盖率(±0.05)   区间宽度中位   落到上限之外   点预测 MAE / 偏差")
        for tag, kw in sets:
            cov, wid, above, err = [], [], [], []
            for k in range(120):
                rows, delta, sc = stress_toy(k, judge, tag=tag3, raw=True, **kw)
                m = measure(rows, delta, judge, sc)
                lo, mid, hi = derive3(rows, judge, sc, v31)
                cov.append(lo - 0.05 <= m["miss"] <= hi + 0.05)
                wid.append(hi - lo)
                above.append(m["miss"] > hi + 0.05)
                err.append(m["miss"] - mid)
            print(f"{tag:10s}  {st.mean(cov):10.0%}   {st.median(wid):10.2f}   {st.mean(above):10.0%}      {st.mean(abs(x) for x in err):.3f} / {st.mean(err):+.2f}")
        for v, zs in ((2, ("Z0",)), (3, ("R0",))):
            data, deltas = blind_sets(v)
            for z in zs:
                clean = data[(z, 0.0)]
                s = max(mad([p for _, p, _, _ in clean]), 1e-6)
                cells = []
                for d in deltas[1:]:
                    ld = math.log10(1 + d)
                    rows = [(False, p, r, ld) for _, p, r, _ in clean] + data[(z, d)]
                    m = measure(rows, ld, judge, s)
                    lo, mid, hi = derive3(rows, judge, s, v31)
                    cells.append(f"δ={d:.0%} 实际 {m['miss']:.0%} 点 {mid:.0%} [{lo:.0%}, {hi:.0%}]")
                print(f"（诊断）盲测{'二' if v == 2 else '三'} {z}：" + "  ".join(cells))


def main2():
    torch.manual_seed(10)
    sets = (("无压力", {}), ("约束×100", {"cmult": 100.0}), ("抱团", {"same": True}), ("厚尾df=1", {"df": 1}),
            ("N=31", {"n": 31}), ("δ分散1.5", {"dvar": 1.5}), ("ζ=1", {"zeta": 1.0}))
    for judge in ("软", "硬2"):
        w = curve(judge, "H", src="het2")
        print(f"\n===== {judge}（新种子 analytic2，每集 120 个配置）：MAE / 平均误差（实际 − 预测） =====")
        print("集合          第一版             第二版             拟合版 H")
        for tag, kw in sets:
            E = {"v1": [], "v2": [], "H": []}
            for k in range(120):
                rows, delta, sc = stress_toy(k, judge, tag="analytic2", raw=True, **kw)
                m = measure(rows, delta, judge, sc)
                E["v1"].append(m["miss"] - derive(rows, judge, sc)[0])
                E["v2"].append(m["miss"] - derive2(rows, judge, sc))
                E["H"].append(m["miss"] - predict(w, [m], "H").item())
            print(f"{tag:10s}  " + "   ".join(f"{st.mean(abs(x) for x in v):.3f} / {st.mean(v):+.2f}" for v in E.values()))
        for v, zs in ((2, ("Z0",)), (3, ("R0",))):
            data, deltas = blind_sets(v)
            for z in zs:
                clean = data[(z, 0.0)]
                s = max(mad([p for _, p, _, _ in clean]), 1e-6)
                cells = []
                for d in deltas[1:]:
                    ld = math.log10(1 + d)
                    rows = [(False, p, r, ld) for _, p, r, _ in clean] + data[(z, d)]
                    m = measure(rows, ld, judge, s)
                    cells.append(f"δ={d:.0%} 实际 {m['miss']:.0%} 第二版 {derive2(rows, judge, s):.0%}")
                print(f"（诊断，已看过）盲测{'二' if v == 2 else '三'} {z}：" + "  ".join(cells))


def blind_sets(v):
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    if v == 2:
        os.environ["BLIND"] = "2"
        import blind as mod
    else:
        import blind3 as mod
    data = mod.setup()
    os.chdir(cwd)
    return data, mod.DELTAS


def main():
    torch.manual_seed(9)
    for judge in ("软", "硬2"):
        w = curve(judge, "H", src="het2")
        print(f"\n===== {judge} =====")
        for tag, kw in (("无压力基线", {}), ("约束 ×100", {"cmult": 100.0}), ("抱团（同样精确）", {"same": True})):
            E_a, E_h, U = [], [], []
            for k in range(120):
                rows, delta, sc = stress_toy(k, judge, tag="analytic", raw=True, **kw)
                m = measure(rows, delta, judge, sc)
                pa, u = derive(rows, judge, sc)
                E_a.append(abs(pa - m["miss"]))
                E_h.append(abs(predict(w, [m], "H").item() - m["miss"]))
                U.append(u)
            print(f"{tag:12s} MAE  推导版 {st.mean(E_a):.3f}   拟合版 H {st.mean(E_h):.3f}   预测不了的故障题 {st.mean(U):.0%}")
        for v, zs in ((2, ("Z0", "Z1")), (3, ("R0", "R1"))):
            data, deltas = blind_sets(v)
            for z in zs:
                clean = data[(z, 0.0)]
                c = [p for _, p, _, _ in clean]
                s = max(mad(c), 1e-6)
                cells, ea, eh = [], [], []
                for d in deltas[1:]:
                    ld = math.log10(1 + d)
                    rows = [(False, p, r, ld) for _, p, r, _ in clean] + data[(z, d)]
                    m = measure(rows, ld, judge, s)
                    pa, u = derive(rows, judge, s)
                    ph = predict(w, [m], "H").item()
                    ea.append(abs(pa - m["miss"]))
                    eh.append(abs(ph - m["miss"]))
                    cells.append(f"δ={d:.0%} 实际 {m['miss']:.0%} 推导 {pa:.0%} H {ph:.0%} 不确定 {u:.0%}")
                print(f"盲测{'二' if v == 2 else '三'} {z}：平均 |误差| 推导版 {st.mean(ea):.3f}  H {st.mean(eh):.3f}\n    " + "\n    ".join(cells))


if __name__ == "__main__":
    {"v2": main2, "v3": main3, "v31": lambda: main3(True), "v4": main4, "v5": main5}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
