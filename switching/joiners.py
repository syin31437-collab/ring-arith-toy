"""动态加入：新节点随时间加入，系统能不能把里面的坏节点找出来（2026-11-01，库伯提问）。

用盲测三真实数据（R0 约束），7 个节点：干净 = claude_haiku、codex；投毒 = 5 个 DeepSeek。
场景（打乱 50 种顺序平均，每种 60 轮）：
  A 一个新坏节点：开局 haiku、codex、ds_flash_a、ds_flash_b（2 好 2 坏）；第 20 轮加入 ds_flash_c
  B 一群新坏节点（女巫）：开局同上；第 20 轮一次加入 ds_flash_c、ds_pro_a、ds_pro_b（坏的变成 5 对 2）
诚实权重两种：
  默认信任：现行 exp(−½ z²)，新节点证据少 → z ≈ 0 → 权重 ≈ 1
  试用期：权重再乘 n / (n + 5)，n = 这个节点已经被观察的轮数（新人先半信半疑，5 轮后过半）——这是设计选择，常数 5 是拍的
指标：新坏节点权重第一次 < 0.5 是加入后第几轮；学习裁判在第 20–25 轮和第 41–60 轮的被骗率。

预注册（运行前）：
  1. 场景 A、δ = 5%：新坏节点在加入后 ≤ 8 轮内权重 < 0.5
  2. 场景 B、默认信任：第 20–25 轮被骗率比第 15–19 轮高 0.3 以上（女巫波冲进来，在被认出来之前占了多数）
  3. 场景 B、试用期：第 20–25 轮被骗率比默认信任低 0.2 以上
  4. 两种权重在第 41–60 轮都回到 ≤ 0.2

用法：~/gut-external/torch215-venv/bin/python joiners.py
"""
import math
import os
import random
import statistics as st
import sys

import torch

from analytic import sigma_c
from robust import wmedian

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = ["claude_haiku", "codex", "ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "ds_pro_b"]


def load(d):
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    import blind3
    data = blind3.setup()
    os.chdir(cwd)
    return data[("R0", 0.0)], data[("R0", d)]


def run(clean, faults, start, joins, probation):
    b, sc = sigma_c(clean)
    s = max(sc, 1e-6)
    idx = {n: i for i, n in enumerate(NODES)}
    since = {n: 0 for n in start}
    hist = {n: [] for n in NODES}
    out, first_low = [], {}
    for k, (p, r, h) in enumerate(faults, 1):
        for n in joins.get(k, []):
            since[n] = k
        act = sorted(since, key=lambda n: idx[n])
        w = []
        for n in act:
            x = hist[n]
            if len(x) < 2:
                z = 0.0
            else:
                z = abs(st.mean(x)) / (st.stdev(x) / math.sqrt(len(x)) + 1e-12)
            wi = math.exp(-0.5 * z * z)
            if probation:
                wi *= len(x) / (len(x) + 5)
            w.append(wi)
            if since[n] > 1 and wi < 0.5 and n not in first_low:
                first_low[n] = k - since[n]
        rr = torch.stack([r[idx[n]] for n in act])
        kern = torch.exp(-((rr - p) ** 2) / (2 * s ** 2))
        o = wmedian(rr, torch.tensor(w) * kern + 1e-9).item()
        out.append(abs(o) > h)
        for n in act:
            hist[n].append(r[idx[n]].item() - (p - b))
    return out, first_low


def main():
    for d in (0.05, 0.1):
        clean, bad = load(d)
        for name, joins in (("A 一个新坏节点", {20: ["ds_flash_c"]}), ("B 女巫波（3 个新坏节点）", {20: ["ds_flash_c", "ds_pro_a", "ds_pro_b"]})):
            print(f"\n== δ = {d:.0%}  {name} ==")
            for probation in (False, True):
                curves, lows = [], []
                for o in range(50):
                    f = [(p, r, x / 2) for _, p, r, x in bad]
                    random.Random(o).shuffle(f)
                    c, fl = run(clean, f, ["claude_haiku", "codex", "ds_flash_a", "ds_flash_b"], joins, probation)
                    curves.append(c)
                    lows += [v for n, v in fl.items() if n in joins[20]]
                seg = lambda lo, hi: st.mean(float(x) for c in curves for x in c[lo - 1:hi])
                print(f"  {'试用期' if probation else '默认信任'}：被骗率 第 6–19 轮 {seg(6, 19):.0%}  第 20–25 轮 {seg(20, 25):.0%}  第 26–40 轮 {seg(26, 40):.0%}  第 41–60 轮 {seg(41, 60):.0%}"
                      f"   新坏节点权重 < 0.5 用了 {st.median(lows) if lows else float('nan'):.0f} 轮（中位）")


if __name__ == "__main__":
    main()
