"""补"秤每道题可靠程度不一样"（2026-11-01）。

盲测四（核查员整批漏 ×60）、盲测五（阿基米德法误差被 1 / (密度 − 1) 放大）两次撞上"约束噪声每题一样"这条假设。
补两样：
  1. 推导版第八版（analytic.derive8）：约束误差用 t 分布（ν 从校准期峰度估）；每道题先用诚实子集估真值，
     秤离它越远、这道题秤的权重越低：(ν + 1) / (ν + 偏差²)。诚实度改用中位数偏移。
  2. 通用秤坏保护（purge.scale_broken）：所有节点同向越线 = 怪秤；EWMA 线、删换、学习裁判的诚实度学习都用它。
新压力（玩具新种子 robc，每集 120 个配置）：约束误差 t 分布 ν = 3；20% 的题秤严重偏（± (20σ_c + 10σ_节点)）；每 150 题里一整段 50 题坏。

预注册（运行前）：
  1. 三个新压力集上，第八版 MAE ≤ 第五版 MAE − 0.02（软、硬2）
  2. 无压力集：第八版 MAE ≤ 第五版 + 0.01
  3. （诊断，数据看过）盲测五按编号跑 EWMA 线加秤坏保护：δ = 0 那一轮诚实节点越线次数 = 0
  4. （诊断，数据看过）盲测四学习裁判加秤坏保护：最后一段（41–60 轮）被骗率不高于静态裁判

第一稿结果（1a2a1be，4 条对 1 条）：秤坏保护让盲测四学习裁判从 82 / 91% 降到 39 / 33%（好过静态 69 / 56%）✅；
  第八版（t 约束）在所有压力集都不比第五版好、无压力时差得多（0.077 vs 0.022）❌——太复杂：拿"抽样的诚实子集"比秤，
  子集里混了投毒节点时秤显得坏；玩具里 ν 被嘈杂节点带偏，秤被丢得太多。
第二稿：第九版（analytic.derive9）= 第五版 + 逐题直接复用秤坏保护的判据：这道题所有节点同向越线 → 这道题不用秤，否则照常。
第二稿预注册（运行前；新种子 robc2）：
  1. "20% 题秤坏""成批秤坏"两个集：第九版 MAE ≤ 第五版 − 0.02（软、硬2）
  2. 无压力、约束 t3 两个集：第九版 MAE ≤ 第五版 + 0.01

用法：~/gut-external/torch215-venv/bin/python robustc.py        （第一稿）
      ~/gut-external/torch215-venv/bin/python robustc.py v9     （第二稿）
"""
import math
import os
import random
import statistics as st
import sys

import torch

from analytic import derive5, derive8, sigma_c
from failmap import stress_toy
from hetero import measure

HERE = os.path.dirname(os.path.abspath(__file__))


def toys():
    torch.manual_seed(24)
    sets = (("无压力", {}), ("约束t3", {"ct": 3.0}), ("20%题秤坏", {"cfail": 0.2}), ("成批秤坏", {"cbatch": True}))
    for judge in ("软", "硬2"):
        print(f"\n== {judge}（新种子 robc，120 个配置）：MAE / 偏差 ==")
        for tag, kw in sets:
            e5, e8 = [], []
            for k in range(120):
                rows, delta, sc = stress_toy(k, judge, tag="robc", raw=True, **kw)
                s = max(sigma_c([x for x in rows if not x[0]])[1], 1e-9)
                m = measure(rows, delta, judge, s)["miss"]
                e5.append(m - derive5(rows, judge, s))
                e8.append(m - derive8(rows, judge, s))
            print(f"  {tag:10s} 第五版 {st.mean(abs(x) for x in e5):.3f} / {st.mean(e5):+.2f}   第八版 {st.mean(abs(x) for x in e8):.3f} / {st.mean(e8):+.2f}")


def real():
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    import blind4
    import blind5
    from online import run, segs
    from purge import C, SD_S, W, noise, scale_broken
    d5 = blind5.setup()
    d4 = blind4.setup()
    os.chdir(cwd)
    clean = d5[("R0", 0.0)]
    b, sc = sigma_c(clean)
    sig = noise(clean)
    nodes = sorted(blind5.NODES)
    S = torch.zeros(len(nodes))
    crossed = {}
    for k, (_, p, r, _) in enumerate(clean, 1):
        new = (1 - W) * S + W * (r - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)
        if scale_broken(new.tolist()):
            continue
        S = new
        for j in torch.nonzero(S.abs() > C * SD_S).flatten().tolist():
            crossed.setdefault(nodes[j], k)
    print(f"\n（诊断）盲测五 δ = 0 一轮按编号跑 EWMA + 秤坏保护：越线的诚实节点 {crossed or '无'}")
    for name, data, deltas in (("盲测四", d4, (0.05, 0.1)), ("盲测五", d5, (0.02, 0.05))):
        clean = data[("R0", 0.0)]
        for d in deltas:
            res = {}
            for guard in (False, True):
                curves = []
                for o in range(50):
                    f = [(p, r, x / 2) for _, p, r, x in data[("R0", d)]]
                    random.Random(o).shuffle(f)
                    curves.append(run(clean, f, o, guard))
                res[guard] = segs(curves)
            print(f"（诊断）{name} δ = {d:.0%} 学习裁判 41–60 轮：无保护 {res[False][-1][3]:.0%}  有保护 {res[True][-1][3]:.0%}  静态 {res[True][-1][1]:.0%}"
                  f"；1–5 轮：{res[True][0][3]:.0%}")
        for d in deltas:
            rows = [(False, p, r, x) for _, p, r, x in clean] + data[("R0", d)]
            s = max(sigma_c(clean)[1], 1e-6)
            act = measure(rows, math.log10(1 + d), "软", s)["miss"]
            print(f"（诊断）{name} δ = {d:.0%} 软：实际 {act:.0%}  第五版 {derive5(rows, '软', s):.0%}  第八版 {derive8(rows, '软', s):.0%}")


def toys9():
    from analytic import derive9
    torch.manual_seed(25)
    sets = (("无压力", {}), ("约束t3", {"ct": 3.0}), ("20%题秤坏", {"cfail": 0.2}), ("成批秤坏", {"cbatch": True}))
    for judge in ("软", "硬2"):
        print(f"\n== {judge}（新种子 robc2，120 个配置）：MAE / 偏差 ==")
        for tag, kw in sets:
            e5, e9 = [], []
            for k in range(120):
                rows, delta, sc = stress_toy(k, judge, tag="robc2", raw=True, **kw)
                s = max(sigma_c([x for x in rows if not x[0]])[1], 1e-9)
                m = measure(rows, delta, judge, s)["miss"]
                e5.append(m - derive5(rows, judge, s))
                e9.append(m - derive9(rows, judge, s))
            print(f"  {tag:10s} 第五版 {st.mean(abs(x) for x in e5):.3f} / {st.mean(e5):+.2f}   第九版 {st.mean(abs(x) for x in e9):.3f} / {st.mean(e9):+.2f}")
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    import blind4
    import blind5
    for name, mod, deltas in (("盲测四", blind4, (0.02, 0.05, 0.1, 0.2)), ("盲测五", blind5, (0.02, 0.05, 0.1, 0.2))):
        data = mod.setup()
        os.chdir(cwd)
        clean = data[("R0", 0.0)]
        s = max(sigma_c(clean)[1], 1e-6)
        cells = []
        for d in deltas:
            rows = [(False, p, r, x) for _, p, r, x in clean] + data[("R0", d)]
            cells.append(f"δ={d:.0%} 实际 {measure(rows, math.log10(1 + d), '软', s)['miss']:.0%} 五 {derive5(rows, '软', s):.0%} 九 {derive9(rows, '软', s):.0%}")
        print(f"（诊断，已看过）{name} 软：" + "  ".join(cells))


if __name__ == "__main__":
    toys9() if sys.argv[1:] == ["v9"] else (toys(), real())
