"""规律的推导版（零拟合参数）：漏检 = 裁判选中的那一团是故障团的概率。

每道故障题（全部只用观测量；真值只用来计分）：
  1. 分团：节点报告排序，在最大缺口处分成两团；缺口 ≤ 3 × 节点间分散（无故障题上各题报告的 MAD 的中位数）算没分开。
  2. 裁判输出由报告和约束 p 直接算出（软 / 硬2，与 hetero.judge_out 相同），看它落在哪一团。
  3. 哪团诚实：约束误差 e = p − 真值 ~ N(b, σ_c²)；b、σ_c 用无故障题上 (p − 报告中位数) 的中位数和 MAD 估计（不用真值）。
     团 X 诚实的似然 = N(p − X 的中位数; b, σ_c² + X 内分散² / |X|)；先验 50 / 50（谁占多数从内部看不见）。
  4. 预测漏检 = P(选中的团是故障团) · 1[两团距离 > t] + P(选中的团诚实) · P(|N(0, X 内分散² / |X|)| > t)，t = δ/2。
     没分开的题：两团看不出 → 输出偏多少看不见 → 预测 = P(|N(0, σ_out²)| > t)，σ_out = 无故障题上 (输出 − 报告中位数) 的 MAD。
  一道题的后验落在 [0.2, 0.8] 记为"预测不了"（约束分不清哪团诚实，结果取决于看不见的投毒比例）。

和拟合曲线比较：H（het2 拟合）在同一批配置上的 MAE。RAIM 的单颗故障检验是这里 |故障团| = 1、硬裁判的特例。

预注册（11-01，运行前）：
  1. 两次盲测（盲测二 Z0、盲测三 R0），推导版的平均 |误差| ≤ 拟合版 H（软、硬2 各自比）；盲测三 δ = 5% 软裁判 |误差| ≤ 0.15
  2. 压力玩具无压力基线：推导版 MAE ≤ H 的 MAE + 0.03（软、硬2）
  3. ζ = 1（盲测 Z1、R1）：推导版照样乐观，实际 − 预测 ≥ +0.30
  4. 约束 ×100（ρ 中位约 0.1）时"预测不了"的故障题占比，高于无压力基线的 2 倍以上

用法：~/gut-external/torch215-venv/bin/python analytic.py
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
    main()
