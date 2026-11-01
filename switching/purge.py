"""规律之上的一层用法：暴露就划掉、换新（2026-11-01，库伯）。规律不变——还是靠约束找出出错的节点；
平庸的评委偶尔犯错是合理的，不追究；节点一旦"暴露"（偏差超出它自己的噪声 + 秤的不确定度能解释的范围）就划掉、换新，
目标是整个系统稳定。

数据：盲测三真实数据（R0 = 独立 GPS 记录约束），每轮 = 一道题，50 种顺序平均。
  节点"数据流"：诚实流 = 某个模型在 δ = 0 那一轮的回答；投毒流 = 同一个模型在 δ 那一轮的回答。
  替换进来的新节点 = 一个 DeepSeek 模型的诚实流（没看到毒的那份），轮流取。
每个节点一条会老化的线（EWMA，权重 0.3）：S ← 0.7·S + 0.3·z，z = (报告 − 约束中心) / √(节点噪声² + 秤噪声²)。
  不下毒时 S 的标准差 = √(0.3 / 1.7)；|S| > 3.3 个标准差（每个节点每轮误报约 1/1000）= 暴露 → 划掉、换新，新人的线从 0 开始。
  权重 0.3 取自 EWMA 控制图的经验范围（偏移 1–2σ 时 0.2–0.4）；这里 δ = 5% 约等于 1σ。
  秤坏保护：同一轮所有在场节点都越过门槛、而且方向相同 → 是秤坏了，这一轮不删人、不更新线（要求至少有一个诚实节点，即第二条前提）。
裁判：普通软裁判（不学习），宽度 = 可观测的秤噪声估计；只在在场节点上判。
场景：
  S1 卧底：7 个节点都诚实，其中 ds_flash_a 第 31 轮起改用投毒流
  S2 女巫：前 19 轮 4 个诚实（haiku、codex、ds_flash_a、ds_flash_b 的诚实流）；第 20 轮一次加入 5 个投毒流（5 对 4，过半）
  S3 间歇：7 个节点，ds_pro_b 每轮以 30% 概率用投毒流
  S4 秤坏一批：7 个节点都诚实，第 40–45 轮约束偏 +1.78（≈ 60 倍，仿盲测四的单位错误）；有 / 没有秤坏保护各跑一遍

预注册（运行前）：
  1. S1：卧底从开始下毒到被划掉的轮数中位，δ = 5% ≤ 6，δ = 10% ≤ 3
  2. S1–S3：诚实节点被误删的个数平均每次运行 ≤ 0.1
  3. S2：第 41–60 轮系统被骗率 ≤ 0.10；不删人的对照 ≥ 0.50
  4. S4：有保护时 ≥ 90% 的运行在第 40–45 轮删 0 个人；没保护时平均删掉 ≥ 一半的节点
  5. S3、δ = 10%：间歇下毒者被划掉的轮数中位 ≤ 20

用法：~/gut-external/torch215-venv/bin/python purge.py
"""
import math
import os
import random
import statistics as st
import sys

import torch

from analytic import sigma_c
from hetero import judge_out

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = ["claude_haiku", "codex", "ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "ds_pro_b"]
DS = NODES[2:]
W, C = 0.3, 3.3
SD_S = math.sqrt(W / (2 - W))


def scale_broken(z, c=C * SD_S):
    """通用秤坏保护：这一轮所有节点相对秤的偏移都越过 c、而且方向相同 → 怪秤，不怪节点。
    z = 各节点的（标准化）偏移或线值。要求至少有一个诚实节点（第二条前提）。"""
    z = list(z)
    return len(z) > 0 and all(abs(x) > c for x in z) and len({x > 0 for x in z}) == 1


def load(d, v=3):
    sys.path.insert(0, os.path.join(HERE, "..", "llm"))
    cwd = os.getcwd()
    if v == 2:
        os.environ["BLIND"] = "2"
        import blind as mod
        z = "Z0"
    else:
        import blind3 as mod
        z = "R0"
    data = mod.setup()
    os.chdir(cwd)
    return data[(z, 0.0)], data[(z, d)]


def noise(clean):
    """各模型的噪声（三角帽法不需要，这里节点少、直接用无故障题上相对其余节点中位数的 MAD）。"""
    R = torch.stack([r for _, _, r, _ in clean])
    return (1.4826 * (R - R.median(1, keepdim=True).values).abs().median(0).values).clamp(min=1e-4)


def simulate(clean, bad, order, scenario, purge=True, guard=True, rng=None):
    """返回 dict：每轮是否被骗、删人记录。"""
    b, sc = sigma_c(clean)
    s = max(sc, 1e-6)
    sig = noise(clean)
    col = {n: i for i, n in enumerate(NODES)}
    # 在场节点：(名字, 模型, 流类型)；流类型 = "honest" / "poison" / "sleeper" / "intermittent"
    if scenario == "S2":
        active = [("haiku", "claude_haiku", "honest"), ("codex", "codex", "honest"), ("fa", "ds_flash_a", "honest"), ("fb", "ds_flash_b", "honest")]
    else:
        active = [(n, n, "honest") for n in NODES]
        if scenario == "S1":
            active[2] = ("ds_flash_a", "ds_flash_a", "sleeper")
        if scenario == "S3":
            active[6] = ("ds_pro_b", "ds_pro_b", "intermittent")
    pq = 0.3 if scenario == "S3" else 1.0
    S = {a[0]: 0.0 for a in active}
    spare = 0
    rec = {"miss": [], "removed": [], "false_removed": 0}
    for k, idx in enumerate(order, 1):
        if scenario == "S2" and k == 20:
            for j, m in enumerate(DS):
                active.append((f"sybil{j}", m, "poison"))
                S[f"sybil{j}"] = 0.0
        _, p, rc, _ = clean[idx]
        _, _, rb, ld = bad[idx]
        if scenario == "S4" and 40 <= k <= 45:
            p = p + 1.78
        vals, poisoned = [], []
        for name, model, kind in active:
            use_bad = kind == "poison" or (kind == "sleeper" and k >= 31) or (kind == "intermittent" and rng.random() < 0.3)
            vals.append((rb if use_bad else rc)[col[model]].item())
            poisoned.append(use_bad)
        r = torch.tensor(vals)
        o = judge_out(r, torch.tensor(p), s, "软")
        rec["miss"].append(abs(o) > ld / 2)
        z = [(v - (p - b)) / math.sqrt(sig[col[m]].item() ** 2 + sc ** 2) for v, (_, m, _) in zip(vals, active)]
        news = {a[0]: (1 - W) * S[a[0]] + W * zi for a, zi in zip(active, z)}
        over = [abs(news[a[0]]) > C * SD_S for a in active]
        if guard and scale_broken(news[a[0]] for a in active):
            continue                                   # 秤坏了：不删人、不更新线
        S.update(news)
        if not purge:
            continue
        keep = []
        for a, ov in zip(active, over):
            if ov:
                rec["removed"].append((k, a[0], a[2]))
                if a[2] == "honest":
                    rec["false_removed"] += 1
                m = DS[spare % len(DS)]
                spare += 1
                new = (f"new{spare}", m, "honest")
                keep.append(new)
                S[new[0]] = 0.0
            else:
                keep.append(a)
        active = keep
    return rec


def main():
    for d in (0.05, 0.1):
        clean, bad = load(d)
        print(f"\n======== δ = {d:.0%} ========")
        for sc_name in ("S1", "S2", "S3", "S4"):
            for purge, guard in ((True, True), (False, True)) if sc_name in ("S2",) else ((True, True), (True, False)) if sc_name == "S4" else ((True, True),):
                runs = []
                for o in range(50):
                    order = list(range(len(clean)))
                    random.Random(o).shuffle(order)
                    runs.append(simulate(clean, bad, order, sc_name, purge, guard, random.Random(1000 + o)))
                seg = lambda lo, hi: st.mean(float(x) for r in runs for x in r["miss"][lo - 1:hi])
                fr = st.mean(r["false_removed"] for r in runs)
                tag = f"{sc_name}{'' if purge else ' 不删人对照'}{'' if guard else ' 无秤坏保护'}"
                line = f"{tag:16s} 被骗率 1–19 {seg(1, 19):4.0%}  20–30 {seg(20, 30):4.0%}  31–40 {seg(31, 40):4.0%}  41–60 {seg(41, 60):4.0%}   误删诚实节点 {fr:.2f} 个/次"
                if sc_name == "S1":
                    delays = [next((k - 31 for k, n, kind in r["removed"] if kind == "sleeper"), None) for r in runs]
                    got = [x for x in delays if x is not None]
                    line += f"   卧底被划掉：{len(got)}/50 次，下毒后第 {st.median(got) if got else float('nan'):.0f} 轮（中位）"
                if sc_name == "S2":
                    delays = [k - 20 for r in runs for k, n, kind in r["removed"] if kind == "poison"]
                    line += f"   女巫被划掉：平均每次 {len(delays) / 50:.1f}/5 个，加入后第 {st.median(delays) if delays else float('nan'):.0f} 轮（中位）"
                if sc_name == "S3":
                    delays = [next((k for k, n, kind in r["removed"] if kind == "intermittent"), None) for r in runs]
                    got = [x for x in delays if x is not None]
                    line += f"   间歇下毒者被划掉：{len(got)}/50 次，第 {st.median(got) if got else float('nan'):.0f} 轮（中位）"
                if sc_name == "S4":
                    n4045 = [sum(1 for k, _, _ in r["removed"] if 40 <= k <= 45) for r in runs]
                    line += f"   第 40–45 轮删人：平均 {st.mean(n4045):.1f} 个，删 0 个的运行 {sum(x == 0 for x in n4045)}/50"
                print(line)


if __name__ == "__main__":
    main()
