"""时间版：轮次 k = 1, 2, 3 … 在线判、在线学、在线预测（2026-11-01）。

推导版之前是"事后一次性算"：全部题目放在一起汇总谁诚实，再回头判每道题（用了未来的信息）。
真实系统按轮来：第 k 轮只能用第 1 … k−1 轮。另外裁判自己从不学，预测公式却在学。
设置：
  校准期：毒出现之前一段已知干净的轮（真实数据 = δ = 0 那一轮的 60 道题；玩具 = 前 50 轮），估 b、σ_c、各节点噪声
  投毒期：第 k 轮——
    静态裁判：现在的软裁判，每轮从零判（宽度 s 用可观测估计，不偷看真值）
    学习裁判：软裁判的权重再乘"到目前为止这个节点像不像诚实"（只用第 1 … k−1 轮 + 校准期，不需要知道哪轮有毒）
    在线预测：同样只用之前的轮，按推导版第五版的式子预测这一轮裁判被骗的概率（式子不管裁判是谁，给它裁判的输出就行）
  被骗门槛记作 h = δ/2（之前叫 t，和时间撞名）。
  真实数据：同一组故障题打乱成 50 种顺序再按轮次平均（压抽样噪声）；玩具：每个配置 120 道故障题按生成顺序。

预注册（11-01，运行前；真实数据看过，但学习裁判和在线预测的结果没算过 → 半盲；玩具用新种子 online）：
  1. 在线预测跟得上静态裁判：真实盲测三 R0、盲测四 R0 的 δ = 5% / 10%，以及玩具无压力集，按轮次分段（1–5、6–10、11–20、21–40、41–60），
     每段 |实际 − 预测| ≤ 0.10
  2. 学习裁判会随轮次变好：盲测三 R0 δ = 5%（静态裁判 67% 被骗）最后一段 ≤ 第一段 − 0.20，且最后一段 ≤ 0.20
  3. 在线预测也跟得上学习裁判：同上各段 |实际 − 预测| ≤ 0.10
  4. 约束越强，学习裁判学得越快：盲测二 Z0（约束 σ_c ≈ 0.007）达到"最后一段 + 0.05 以内"所需轮数 ≤ 盲测三 R0（σ_c ≈ 0.02）

用法：~/gut-external/torch215-venv/bin/python online.py
"""
import math
import os
import random
import statistics as st
import sys

import torch

from analytic import Phi, sigma_c
from failmap import stress_toy
from hetero import judge_out
from robust import wmedian

HERE = os.path.dirname(os.path.abspath(__file__))
SEGS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 60)]


def calib(clean):
    b, sc = sigma_c(clean)
    D = torch.stack([r - r.median() for _, _, r, _ in clean])
    spread = max(st.median(1.4826 * x.abs().median().item() for x in D), 1e-9)
    prec = 1 / (1.4826 * (D - D.median(0).values).abs().median(0).values).clamp(min=0.5 * spread) ** 2
    return b, sc, prec


def honesty(hist, b):
    """到目前为止各节点像不像诚实：hist = 已经看过的 (p, r)（校准期 + 之前的轮，不分有没有毒）。"""
    X = torch.stack([r - (p - b) for p, r in hist])
    se = X.std(0) / math.sqrt(len(hist))
    return torch.exp(-0.5 * (X.mean(0) / se.clamp(min=1e-12)) ** 2)


def predict_one(p, r, h, o, w, b, sc, prec, M=64, g=None):
    """推导版第五版的单轮式子：按诚实权重抽诚实子集，诚实节点相乘 × 约束，求 |o − 真值| > h 的概率。"""
    S = (torch.rand(M, len(r), generator=g) < w).float()
    S[S.sum(1) == 0, int(torch.argmax(w))] = 1.0
    P = S @ prec + 1 / sc ** 2
    m = (S @ (prec * r) + (p - b) / sc ** 2) / P
    sd = 1 / torch.sqrt(P)
    return st.mean(1 - (Phi((h - (o - mi)) / si) - Phi((-h - (o - mi)) / si)) for mi, si in zip(m.tolist(), sd.tolist()))


def run(clean, faults, order_seed=0, guard=False):
    """按轮跑一遍：返回每轮 (静态被骗, 静态预测, 学习被骗, 学习预测)。faults = [(p, r, h)]（真值 = 0）。"""
    b, sc, prec = calib(clean)
    s = max(sc, 1e-6)
    hist = [(p, r) for _, p, r, _ in clean]
    g = torch.Generator().manual_seed(order_seed)
    out = []
    for p, r, h in faults:
        w = honesty(hist, b)
        o_s = judge_out(r, torch.tensor(p), s, "软")
        kern = torch.exp(-((r - p) ** 2) / (2 * s ** 2))
        o_l = wmedian(r, w * kern + 1e-9).item()
        out.append((abs(o_s) > h, predict_one(p, r, h, o_s, w, b, sc, prec, g=g),
                    abs(o_l) > h, predict_one(p, r, h, o_l, w, b, sc, prec, g=g)))
        if guard:                                      # 秤坏保护：所有节点同向偏离秤的轮不进诚实度（标准化到节点噪声 + 秤噪声）
            from purge import scale_broken
            zz = ((r - (p - b)) * torch.sqrt(prec / (1 + prec * sc ** 2))).tolist()
            if scale_broken(zz, 3.3):
                continue
        hist.append((p, r))
    return out


def segs(curves):
    """curves = 多个顺序的 run 结果 → 每段 (静态实际, 静态预测, 学习实际, 学习预测) 的平均。"""
    res = []
    for lo, hi in SEGS:
        vals = [x for c in curves for x in c[lo - 1:hi]]
        if vals:
            res.append(((lo, hi), *[st.mean(float(v[j]) for v in vals) for j in range(4)]))
    return res


def real(v, z, d, orders=50):
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    if v == 2:
        os.environ["BLIND"] = "2"
        import blind as mod
    elif v == 3:
        import blind3 as mod
    else:
        import blind4 as mod
    data = mod.setup()
    os.chdir(cwd)
    clean, bad = data[(z, 0.0)], data[(z, d)]
    curves = []
    for o in range(orders):
        f = [(p, r, x / 2) for _, p, r, x in bad]
        random.Random(o).shuffle(f)
        curves.append(run(clean, f, o))
    return curves


def report(name, curves):
    print(f"\n{name}")
    print("  轮次        静态：实际 / 预测      学习裁判：实际 / 预测")
    for (lo, hi), a, pa, l, pl in segs(curves):
        print(f"  {lo:2d}–{hi:<2d}        {a:5.0%} / {pa:4.0%}            {l:5.0%} / {pl:4.0%}")


def main():
    for v, z, d in ((3, "R0", 0.05), (3, "R0", 0.1), (4, "R0", 0.05), (4, "R0", 0.1), (2, "Z0", 0.05), (2, "Z0", 0.1)):
        report(f"盲测{'二三四'[v - 2]} {z} δ = {d:.0%}（50 种顺序平均）", real(v, z, d))
    torch.manual_seed(19)
    curves = []
    for k in range(60):
        rows, delta, sc = stress_toy(k, "软", tag="online", raw=True)
        rows = rows[:50] + [x for x in rows[50:] if x[0]][:60] + [x for x in rows[50:] if not x[0]]
        clean = [x for x in rows[:50] if not x[0]]
        faults = [(p, r, dk / 2) for f, p, r, dk in rows[50:] if f][:60]
        if len(clean) >= 10 and len(faults) >= 40:
            curves.append(run(clean, faults, k))
    report(f"玩具无压力（新种子 online，{len(curves)} 个配置；校准期 = 前 50 轮里的干净轮）", curves)


if __name__ == "__main__":
    main()
