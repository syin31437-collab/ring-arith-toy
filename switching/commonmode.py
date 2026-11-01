"""检测层的共模问题（2026-11-01）：同一道题上各节点的 z 相关 0.85–1.00——z 大部分是秤在这道题上的误差，所有节点共享。
线把共享部分当成独立噪声累积：秤一偏，所有线一起动 → 诚实节点被推过线（误报）、投毒节点被拉回来（滞后）。
改法：每道题估秤的共享误差 γ̂_k，从 z 里扣掉再累积。γ̂_k = 按诚实权重（只用到上一轮为止的汇总历史）加权的 z 中位数；
  一开始权重没分开时，投毒占多数会把 γ̂ 往毒那边拉——这是取舍，下面两边都量。
诊断（数据看过）：13 个领域，δ = 0 那一轮的真误报（同 bigtest2 的定义）+ 投毒轮里投毒节点的越线轮数，旧线 vs 去共模线。

cm3（11-01，库伯：修检测层在无噪声节点领域的误报）：
  诊断（planet δ = 0，非预注册）：不是共模问题（共模 γ̂ 大多 ≈ 0）；是分母塌缩——秤 sc = 0.00023、3 个节点留一噪声 = 0 → tot ≈ 0.0002，
  0.2% 的偶发算错就是 z ≈ 4、一次大错 z ≈ 900（截到 3），连着两次就越线。之前"MAD = 0 改平均绝对偏差"的兜底回归不过，
  因为 MAD = 0 在普通 LLM 节点上也常见（和中位数报得一模一样）。
  改法：比伤害门槛还小的偏离不该算——tot 加下限 h_min / C，h_min = log10(1 + 最小 δ) / 2（设计里最小的伤害门槛）；误报判定里的 σ 也用同一下限。
  5% 的持续投毒 z ≈ 6.6 照样抓；0.2% 的偶发 z ≈ 0.6。噪声本来就高于下限的领域（LLM 记忆秤、记录秤）一点不变。
cm3 预注册（运行前，17 套数据）：
  1. 真误报合计 ≤ cm2（8）；planet 的 δ = 0 真误报 = 0
  2. 没抓到 ≤ cm2（0）；越线轮数中位 ≤ cm2（1）
  3. 只有 tot 低于下限的领域会变：其余领域每格结果和 cm2 一模一样

用法：~/gut-external/torch215-venv/bin/python commonmode.py
      ~/gut-external/torch215-venv/bin/python commonmode.py v3     （cm2 vs cm3）
"""
import math
import statistics as st

import torch

from analytic import sigma_c
from invert import loo, sets
from purge import C, SD_S, W, baseline
from robust import wmedian


def floor_of(deltas):
    """cm3 的分母下限：设计里最小的伤害门槛 h_min = log10(1 + 最小 δ) / 2，除以越线常数 C。"""
    return math.log10(1 + min(d for d in deltas if d > 0)) / 2 / C


def run_lines(clean, rows, nodes, mode, floor=0.0):
    b, sc = sigma_c(clean)
    sig = loo(torch.stack([r for _, _, r, _ in clean]))
    u = baseline(clean, b)
    tot = torch.sqrt(sig ** 2 + sc ** 2)
    if mode == "cm3":
        tot = tot.clamp(min=floor)
    S = torch.zeros(len(nodes))
    hist = []
    first = {}
    for _, p, r, _, k in rows:
        z = (r - u - (p - b)) / tot
        if z.abs().min() > 3:
            continue
        if mode in ("cm", "cm2", "cm3"):
            if len(hist) >= 2:
                H = torch.stack(hist)
                w = torch.exp(-0.5 * (H.mean(0) / (H.std(0) / math.sqrt(len(H))).clamp(min=1e-9)) ** 2) + 1e-6
            else:
                w = torch.ones(len(nodes))
            g = wmedian(z, w).item()
            if mode in ("cm2", "cm3"):                                     # 只在谁诚实已经认得出来时扣：权重全一样 → 不扣；差得开 → 全扣
                g *= 1 - (w.min() / w.max()).item()
            hist.append(z.clone())
            z = z - g
        S = (1 - W) * S + W * z.clamp(-3, 3)
        for j in torch.nonzero(S.abs() > C * SD_S).flatten().tolist():
            first.setdefault(nodes[j], k)
    return first, sig


def main():
    tot_false = {"old": 0, "cm": 0, "cm2": 0}
    delays = {"old": [], "cm": [], "cm2": []}
    missed = {"old": 0, "cm": 0, "cm2": 0}
    print("领域               δ=0 真误报 旧 / 去共模 / 认得出来才去      投毒节点越线轮数（旧 / 去共模 / 认得出来才去）")
    for name, data, nodes, always, sleeper, deltas in sets():
        clean5 = data[("R0", 0.0)]
        clean = [(f, p, r, x) for f, p, r, x, _ in clean5]
        line = f"{name.split(':')[-1][:16]:16s}"
        fa = {}
        for mode in ("old", "cm", "cm2"):
            first, sig = run_lines(clean, clean5, nodes, mode)
            n_false = 0
            for n, k in first.items():
                j = nodes.index(n)
                prev = [abs(r[j].item()) for _, _, r, _, kk in clean5 if k - 10 <= kk <= k]
                if prev and st.mean(prev) < 2 * sig[j].item():
                    n_false += 1
            fa[mode] = n_false
            tot_false[mode] += n_false
        line += f"     {fa['old']:2d} / {fa['cm']:2d} / {fa['cm2']:2d}     "
        cells = []
        for d in deltas[1:]:
            if d < 0.1:
                continue
            res = {}
            for mode in ("old", "cm", "cm2"):
                first, _ = run_lines(clean, data[("R0", d)], nodes, mode)
                k0 = data[("R0", d)][0][4]
                ds = []
                for n in sorted(always) + ([sleeper] if sleeper else []):
                    s0 = k0 if n in always else 30
                    k = first.get(n)
                    if k is None or k < s0:
                        missed[mode] += 1
                        ds.append(None)
                    else:
                        ds.append(k - s0)
                        delays[mode].append(k - s0)
                res[mode] = ds
            cells.append(f"δ{d:.0%}: {res['old']} / {res['cm']} / {res['cm2']}")
        print(line + "   ".join(cells))
    for m, nm in (("old", "旧线"), ("cm", "去共模"), ("cm2", "认得出来才去共模")):
        print(f"{nm:10s}：真误报 {tot_false[m]:2d}   投毒节点越线轮数 中位 {st.median(delays[m])} 平均 {st.mean(delays[m]):.2f}   没抓到 {missed[m]}")


def main3():
    tot_false, delays, missed, same = {"cm2": 0, "cm3": 0}, {"cm2": [], "cm3": []}, {"cm2": 0, "cm3": 0}, []
    print("领域               δ=0 真误报 cm2 / cm3      投毒节点越线轮数（cm2 / cm3）")
    for name, data, nodes, always, sleeper, deltas in sets():
        fl = floor_of(deltas)
        clean5 = data[("R0", 0.0)]
        clean = [(f, p, r, x) for f, p, r, x, _ in clean5]
        fa = {}
        for mode in ("cm2", "cm3"):
            first, sig = run_lines(clean, clean5, nodes, mode, fl)
            sg = sig.clamp(min=fl) if mode == "cm3" else sig
            fa[mode] = sum(1 for n, k in first.items()
                           if (prev := [abs(r[nodes.index(n)].item()) for _, _, r, _, kk in clean5 if k - 10 <= kk <= k]) and st.mean(prev) < 2 * sg[nodes.index(n)].item())
            tot_false[mode] += fa[mode]
        cells, res_all = [], {}
        for d in deltas[1:]:
            if d < 0.1:
                continue
            res = {}
            for mode in ("cm2", "cm3"):
                first, _ = run_lines(clean, data[("R0", d)], nodes, mode, fl)
                k0 = data[("R0", d)][0][4]
                ds = []
                for n in sorted(always) + ([sleeper] if sleeper else []):
                    s0 = k0 if n in always else 30
                    k = first.get(n)
                    if k is None or k < s0:
                        missed[mode] += 1
                        ds.append(None)
                    else:
                        ds.append(k - s0)
                        delays[mode].append(k - s0)
                res[mode] = ds
            res_all[d] = res
            cells.append(f"δ{d:.0%}: {res['cm2']} / {res['cm3']}")
        unchanged = fa["cm2"] == fa["cm3"] and all(r["cm2"] == r["cm3"] for r in res_all.values())
        same.append((name, unchanged))
        print(f"{name.split(':')[-1][:16]:16s}     {fa['cm2']:2d} / {fa['cm3']:2d}   {'（不变）' if unchanged else '（变了）'}   " + "   ".join(cells))
    for m in ("cm2", "cm3"):
        print(f"{m}：真误报 {tot_false[m]:2d}   越线轮数 中位 {st.median(delays[m])} 平均 {st.mean(delays[m]):.2f}   没抓到 {missed[m]}")
    print("变了的领域：", [n for n, u in same if not u])


if __name__ == "__main__":
    import sys
    main3() if sys.argv[1:] == ["v3"] else main()
