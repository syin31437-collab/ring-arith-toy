"""验证第一性原理：错误能被挡掉多少，只取决于"错误离开合法区域的距离 ÷ 约束的不确定度"。

环境：隐藏信号 x_t（switch.py 的规律，R1 R2 R3 R4 R1 R2 各 1000 步，过程噪声可调）；
      12 个传感器（噪声 σ_i = U(0.05,0.3) × 噪声倍数）；
      第 1000 步起，30% 的步固定那 8 个传感器一起偏 δ（过半）。
裁判：链条直判——规律预测 pred，报告按离 pred 的远近加权（尺度 = 规律近期的预测误差），取加权中位数。
      规律两种：学出来（小 MLP，从真实答案以外的"加权结果≈普通中位数"的步学）/ 给定（直接用真实规律套在系统估计上）。
每次运行测两个数：
  约束不确定度 σ_eff = 没故障的步上，规律预测 和 全体传感器中位数 之差的标准差（纯实测）
  穿透率 = 故障步上输出的平均偏差 ÷ δ（扣掉没故障步的平均偏差）。0 = 完全挡住，1 = 全漏
实验一：扫 δ、噪声倍数、过程噪声，看穿透率是否只由 ρ = δ/σ_eff 决定（和只按 δ 比，谁更能收成一条线）。
实验二：同一设置下 学出来 vs 给定，每 500 步一段，看 σ_eff 和穿透率的差距，以及这些点是否落在实验一的曲线上。

用法：python geometry.py exp1 PROC_NOISE     →  geo1_PROC_NOISE.json
      python geometry.py exp2                →  geo2.json
      python geometry.py report
"""
import json
import math
import statistics as st
import sys

import torch
import torch.nn as nn

from robust import wmedian
from switch import LAGS, rules

N, BAD = 12, 8
PLAN = ["R1", "R2", "R3", "R4", "R1", "R2"]
DELTAS = [0.05, 0.08, 0.12, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.2, 3.0]
SCALES = [0.3, 1.0, 3.0]
PROCS = [0.05, 0.1, 0.2]


def make_series(seed, proc):
    g = torch.Generator().manual_seed(seed)
    r = rules()
    hist, xs, names = [0.0] * LAGS, [], []
    for name in PLAN:
        for _ in range(1000):
            y = max(-3.0, min(3.0, r[name](hist) + proc * torch.randn(1, generator=g).item()))
            xs.append(y)
            names.append(name)
            hist = [y] + hist[:-1]
    return xs, names


def run(seed, proc, scale, delta, given=False, judge="软", guard=True):
    xs, names = make_series(seed, proc)
    g = torch.Generator().manual_seed(1000 + seed)
    sigma = (torch.rand(N, generator=g) * 0.25 + 0.05) * scale
    torch.manual_seed(seed)
    f = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(f.parameters(), lr=3e-3)
    law = rules()
    H, s = [0.0] * 6, 0.3
    log = []
    for t, x in enumerate(xs):
        fault = t >= 1000 and torch.rand(1, generator=g).item() < 0.3
        r = x + sigma * torch.randn(N, generator=g)
        if fault:
            r[:BAD] += delta
        if given:
            pred = torch.tensor(law[names[t]](H))
        else:
            pred = f(torch.tensor(H)).squeeze()
        pd = pred.detach()
        if judge == "软":
            out = wmedian(r, torch.exp(-((r - pd) ** 2) / (2 * s ** 2)) + 1e-6)
        else:                                              # 硬：离预测超过 2.5 倍尺度就剔除
            keep = (r - pd).abs() <= 2.5 * s
            out = r[keep].median() if keep.any() else r.median()   # 全被剔除时退回中位数：约束不能完全取代测量
        plain = r.median()
        if not guard or abs(out.item() - plain.item()) < 0.1:   # 和普通中位数一致时，拿输出当规律的学习目标（guard=False：永远拿自己的输出学，failmap.py 复现"自己当答案"）
            tgt = out.item()
            s = max(0.02, (0.98 * s ** 2 + 0.02 * (pd.item() - tgt) ** 2) ** 0.5)
            if not given:
                loss = (pred - tgt) ** 2
                opt.zero_grad()
                loss.backward()
                opt.step()
        rms = ((r - plain) ** 2).mean().sqrt().item()     # 这一步各节点偏离中位数的典型幅度
        log.append((out.item(), x, fault, pd.item(), plain.item(), rms, (r - pd).tolist()))
        H = H[1:] + [out.item()]
    return log


def measure(log, lo, hi, delta):
    clean = [(o - x, p - m) for o, x, f, p, m, *_ in log[lo:hi] if not f]
    bad = [o - x for o, x, f, p, m, *_ in log[lo:hi] if f]
    sigma_eff = st.pstdev([d for _, d in clean])
    bias0 = st.mean(e for e, _ in clean)
    passthru = (st.mean(bad) - bias0) / delta if bad else float("nan")
    return sigma_eff, passthru


def measure_full(log, lo, hi, delta):
    """measure 的完整版：再加 漏检率（故障步上输出偏移超过 δ/2 的比例）和 实测节点噪声 σ_node。"""
    sigma_eff, passthru = measure(log, lo, hi, delta)
    seg = log[lo:hi]
    bias0 = st.mean(o - x for o, x, f, *_ in seg if not f)
    bad = [abs(o - x - bias0) > delta / 2 for o, x, f, *_ in seg if f]
    sigma_node = math.sqrt(st.mean(e[5] ** 2 for e in seg if not e[2]))
    out = {"sigma_eff": sigma_eff, "pass": passthru, "miss": sum(bad) / len(bad), "sigma_node": sigma_node}
    if len(seg[0]) > 6:                                   # 带各节点偏离约束预测的记录时，再算 σ_min 和 δ_obs（hetero.py 用）
        dev = torch.tensor([e[6] for e in seg if not e[2]])
        out["sigma_min"] = (dev ** 2).mean(0).sqrt().min().item()
        rms = (dev ** 2).mean(0).sqrt()
        out["k_good"] = int((rms <= 2 * rms.min()).sum())
        mad = 1.4826 * (dev - dev.median(0).values).abs().median(0).values.clamp(min=1e-12)
        out["sigma_min_r"], out["k_good_r"] = mad.min().item(), int((mad <= 2 * mad.min()).sum())
        from hetero import honest_count
        out["k_honest"] = honest_count([torch.tensor(e[6]) for e in seg if e[2]][::10], out["sigma_min_r"])   # 每 10 个故障步取 1 个，省时间
        lg = [math.log10(max(abs(m - p), 1e-12)) for _, _, f, p, m, *_ in seg if f]
        out["q"] = st.pstdev(lg) if len(lg) > 1 else 0.0
        out["d_obs"] = st.mean(m - p for _, _, f, p, m, *_ in seg if f) - st.mean(m - p for _, _, f, p, m, *_ in seg if not f)
        oc = [o - p for o, _, f, p, *_ in seg if not f]
        oc_med = st.median(oc)
        out["sigma_out"] = max(1.4826 * st.median(abs(x - oc_med) for x in oc), 1e-9)
        out["sigma_node_r"] = max(st.median(1.4826 * st.median(abs(v - st.median(e[6])) for v in e[6]) for e in seg if not e[2]), 1e-9)
    return out


def exp1(proc):
    torch.set_num_threads(1)
    rows = []
    for scale in SCALES:
        for delta in DELTAS:
            for seed in range(3):
                log = run(seed, proc, scale, delta)
                se, pt = measure(log, 1000, 6000, delta)
                rows.append({"proc": proc, "scale": scale, "delta": delta, "seed": seed, "sigma_eff": se, "pass": pt})
            print(proc, scale, delta, flush=True)
    json.dump(rows, open(f"geo1_{proc}.json", "w"))


def exp2():
    torch.set_num_threads(1)
    rows = []
    for given in (False, True):
        for seed in range(5):
            for delta in (0.3, 0.7):
                log = run(seed, 0.1, 1.0, delta, given)
                for lo in range(1000, 6000, 500):
                    se, pt = measure(log, lo, lo + 500, delta)
                    rows.append({"given": given, "seed": seed, "delta": delta, "bin": lo, "sigma_eff": se, "pass": pt})
        print("given" if given else "learned", flush=True)
    json.dump(rows, open("geo2.json", "w"))


def r2(xs, ys, nbins=12):
    """按 x 分箱，用箱内平均当预测，看能解释多少 y 的方差。"""
    lo, hi = min(xs), max(xs)
    idx = [min(nbins - 1, int((x - lo) / (hi - lo + 1e-9) * nbins)) for x in xs]
    mean = {b: st.mean(y for y, i in zip(ys, idx) if i == b) for b in set(idx)}
    sse = sum((y - mean[i]) ** 2 for y, i in zip(ys, idx))
    sst = sum((y - st.mean(ys)) ** 2 for y in ys)
    return 1 - sse / sst


def curve_value(rows, rho):
    """实验一的经验曲线：在 log ρ 附近（±0.1）取穿透率中位数。"""
    near = [r["pass"] for r in rows if abs(math.log10(r["delta"] / r["sigma_eff"]) - math.log10(rho)) < 0.1]
    return st.median(near) if near else float("nan")


def report():
    rows = [r for p in PROCS for r in json.load(open(f"geo1_{p}.json"))]
    rows = [r for r in rows if r["sigma_eff"] > 0 and not math.isnan(r["pass"])]
    rho = [math.log10(r["delta"] / r["sigma_eff"]) for r in rows]
    dl = [math.log10(r["delta"]) for r in rows]
    ps = [r["pass"] for r in rows]
    print(f"== 实验一：{len(rows)} 次运行（3 种过程噪声 × 3 种传感器噪声 × 11 种偏移 × 3 种子） ==")
    print(f"按 ρ = δ/σ_eff 分箱，能解释穿透率方差的比例 R² = {r2(rho, ps):.3f}")
    print(f"只按 δ 分箱的 R² = {r2(dl, ps):.3f}")
    print("\nσ_eff 的范围：" + f"{min(r['sigma_eff'] for r in rows):.3f} – {max(r['sigma_eff'] for r in rows):.3f}")
    print("\n穿透率 vs ρ（各配置分开列，看是否落在同一条线上；— = 该配置没有落在这一档的点）")
    edges = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    hdr = "ρ 档位          " + "".join(f"{10**a:.2f}-{10**b:<6.2f}" for a, b in zip(edges, edges[1:]))
    print(hdr)
    for p in PROCS:
        for sc in SCALES:
            cells = []
            for a, b in zip(edges, edges[1:]):
                v = [r["pass"] for r, x in zip(rows, rho) if a <= x < b and r["proc"] == p and r["scale"] == sc]
                cells.append(f"{st.median(v):11.2f}" if v else "          —")
            print(f"过程{p:<4} 噪声×{sc:<4}" + "".join(cells))

    r2rows = json.load(open("geo2.json"))
    print("\n== 实验二：规律学出来 vs 直接给定（过程噪声 0.1，传感器噪声 ×1，5 种子中位数） ==")
    print("段        学出 σ_eff  给定 σ_eff   学出穿透(δ=0.3)  给定穿透(δ=0.3)   学出穿透(δ=0.7)  给定穿透(δ=0.7)  规律")
    for lo in range(1000, 6000, 500):
        cell = {}
        for given in (False, True):
            for d in (0.3, 0.7):
                v = [r for r in r2rows if r["given"] == given and r["bin"] == lo and r["delta"] == d]
                cell[(given, d)] = (st.median(x["sigma_eff"] for x in v), st.median(x["pass"] for x in v))
        print(f"{lo:5d}  {cell[(False, 0.3)][0]:10.3f} {cell[(True, 0.3)][0]:10.3f}"
              f"   {cell[(False, 0.3)][1]:14.2f} {cell[(True, 0.3)][1]:15.2f}"
              f"   {cell[(False, 0.7)][1]:14.2f} {cell[(True, 0.7)][1]:15.2f}   {PLAN[lo // 1000]}")
    pts = [(math.log10(r["delta"] / r["sigma_eff"]), r["pass"]) for r in r2rows if r["sigma_eff"] > 0]
    pred = [curve_value(rows, 10 ** x) for x, _ in pts]
    ok = [(p, y) for p, (_, y) in zip(pred, pts) if not math.isnan(p)]
    res = [y - p for p, y in ok]
    print(f"\n实验二的 {len(ok)} 个数据点放到实验一的曲线上：残差中位数 {st.median(res):+.3f}，"
          f"残差绝对值中位数 {st.median(abs(x) for x in res):.3f}，"
          f"用实验一曲线解释实验二穿透率方差 R² = {1 - sum(x*x for x in res) / sum((y - st.mean(y for _, y in ok)) ** 2 for _, y in ok):.3f}")


if __name__ == "__main__":
    {"exp1": lambda: exp1(float(sys.argv[2])), "exp2": exp2, "report": report}[sys.argv[1]]()
