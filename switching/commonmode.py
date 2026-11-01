"""检测层的共模问题（2026-11-01）：同一道题上各节点的 z 相关 0.85–1.00——z 大部分是秤在这道题上的误差，所有节点共享。
线把共享部分当成独立噪声累积：秤一偏，所有线一起动 → 诚实节点被推过线（误报）、投毒节点被拉回来（滞后）。
改法：每道题估秤的共享误差 γ̂_k，从 z 里扣掉再累积。γ̂_k = 按诚实权重（只用到上一轮为止的汇总历史）加权的 z 中位数；
  一开始权重没分开时，投毒占多数会把 γ̂ 往毒那边拉——这是取舍，下面两边都量。
诊断（数据看过）：13 个领域，δ = 0 那一轮的真误报（同 bigtest2 的定义）+ 投毒轮里投毒节点的越线轮数，旧线 vs 去共模线。

用法：~/gut-external/torch215-venv/bin/python commonmode.py
"""
import math
import statistics as st

import torch

from analytic import sigma_c
from invert import loo, sets
from purge import C, SD_S, W, baseline
from robust import wmedian


def run_lines(clean, rows, nodes, mode):
    b, sc = sigma_c(clean)
    sig = loo(torch.stack([r for _, _, r, _ in clean]))
    u = baseline(clean, b)
    tot = torch.sqrt(sig ** 2 + sc ** 2)
    S = torch.zeros(len(nodes))
    hist = []
    first = {}
    for _, p, r, _, k in rows:
        z = (r - u - (p - b)) / tot
        if z.abs().min() > 3:
            continue
        if mode in ("cm", "cm2"):
            if len(hist) >= 2:
                H = torch.stack(hist)
                w = torch.exp(-0.5 * (H.mean(0) / (H.std(0) / math.sqrt(len(H))).clamp(min=1e-9)) ** 2) + 1e-6
            else:
                w = torch.ones(len(nodes))
            g = wmedian(z, w).item()
            if mode == "cm2":                                     # 只在谁诚实已经认得出来时扣：权重全一样 → 不扣；差得开 → 全扣
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


if __name__ == "__main__":
    main()
