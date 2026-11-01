"""零重拟合迁移：传感器容错的曲线，能不能直接预测信息传播网络里事件的漏检率。

κ 统一改成实测定义：κ = δ / σ_node，σ_node = 没故障的步上，各节点报告偏离全体中位数的典型幅度（RMS）。
ρ 不变：ρ = δ / σ_eff，σ_eff = 没故障的步上，约束预测与报告中位数之差的标准差。
指标：穿透率（故障偏移传到输出的比例）、漏检率（故障步上输出偏移超过 δ/2 的比例）。

玩具（拟合用，随机设计各 300 次）：
  时间环境 geometry.run：过程噪声 ∈ [0.03,0.3]、传感器噪声倍数 ∈ [0.2,5]、δ ∈ [0.1,3]
  代数环境 algebra.run：ab 噪声 ∈ [0.2,12]、c 噪声 ∈ [0.15,5]、δ ∈ [0.1,3]
信息传播网络（只用来检验，300 次）：
  30 个节点连成小世界网络（环上各连 2 个邻居，按概率 p ∈ [0,0.5] 随机重连）。
  每个节点每一步：报告 = (1−α)·自己对 x_t 的带噪观察 + α·邻居上一步报告的平均（α ∈ [0,0.6]）。
  事件：30% 的步上，一个"阵营"（环上连续的一片，占 55%–75%，过半）同时把报告偏 δ；
        偏差还会顺着连接传给阵营外的节点，下一步才消退。
  观察者看到全部 30 个报告，用和玩具同样的裁判（学出来的时间规律 + 软/硬）。
零重拟合：只在玩具上拟合 S 形曲线（输入 log ρ, log κ），原样预测网络；对照 = 网络内部交叉拟合（上限）。

用法：python transfer.py run ENV JUDGE PART     ENV = geo / alg / net，分 4 片
      python transfer.py score
"""
import glob
import json
import math
import random
import sys

import torch
import torch.nn as nn

import algebra
import geometry
from decouple import fit_sigmoid, pred_sigmoid, r2
from geometry import make_series, measure_full
from robust import wmedian

NN = 30


def lu(rng, lo, hi):
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def run_net(seed, proc, scale, delta, alpha, rewire, frac, judge):
    xs, _ = make_series(seed, proc)
    g = torch.Generator().manual_seed(1000 + seed)
    rng = random.Random(seed)
    nbr = [[(i - 2) % NN, (i - 1) % NN, (i + 1) % NN, (i + 2) % NN] for i in range(NN)]
    for i in range(NN):
        nbr[i] = [rng.randrange(NN) if rng.random() < rewire else j for j in nbr[i]]
    camp = torch.zeros(NN)
    camp[:round(frac * NN)] = 1.0
    sigma = (torch.rand(NN, generator=g) * 0.25 + 0.05) * scale
    idx = torch.tensor(nbr)
    torch.manual_seed(seed)
    f = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(f.parameters(), lr=3e-3)
    H, s, prev, log = [0.0] * 6, 0.3, torch.full((NN,), xs[0]), []
    for t, x in enumerate(xs):
        fault = t >= 1000 and torch.rand(1, generator=g).item() < 0.3
        obs = x + sigma * torch.randn(NN, generator=g)
        r = (1 - alpha) * obs + alpha * prev[idx].mean(1) + (delta * camp if fault else 0.0)
        prev = r
        pred = f(torch.tensor(H)).squeeze()
        pd = pred.detach()
        if judge == "软":
            out = wmedian(r, torch.exp(-((r - pd) ** 2) / (2 * s ** 2)) + 1e-6)
        else:
            keep = (r - pd).abs() <= 2.5 * s
            out = r[keep].median() if keep.any() else r.median()
        plain = r.median()
        if abs(out.item() - plain.item()) < 0.1:
            tgt = out.item()
            s = max(0.02, (0.98 * s ** 2 + 0.02 * (pd.item() - tgt) ** 2) ** 0.5)
            loss = (pred - tgt) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        log.append((out.item(), x, fault, pd.item(), plain.item(), ((r - plain) ** 2).mean().sqrt().item(), (r - pd).tolist()))
        H = H[1:] + [out.item()]
    return log


def configs(env):
    out = []
    for k in range(300):
        rng = random.Random(f"{env}-{k}")
        d = lu(rng, 0.1, 3)
        if env == "geo":
            out.append({"proc": lu(rng, 0.03, 0.3), "scale": lu(rng, 0.2, 5), "delta": d, "seed": k})
        elif env == "alg":
            out.append({"ab": lu(rng, 0.2, 12), "scale": lu(rng, 0.15, 5), "delta": d, "seed": k})
        else:
            out.append({"proc": lu(rng, 0.03, 0.3), "scale": lu(rng, 0.2, 5), "delta": d, "seed": k,
                        "alpha": rng.uniform(0, 0.6), "rewire": rng.uniform(0, 0.5), "frac": rng.uniform(0.55, 0.75)})
    return out


def gen(env, judge, part, parts=4):
    torch.set_num_threads(1)
    rows = []
    for k, c in enumerate(configs(env)):
        if k % parts != part:
            continue
        if env == "geo":
            log, lo, hi = geometry.run(c["seed"], c["proc"], c["scale"], c["delta"], judge=judge), 1000, 6000
        elif env == "alg":
            log, lo, hi = algebra.run(c["seed"], c["ab"], c["scale"], c["delta"], judge), 1000, 5000
        else:
            log, lo, hi = run_net(c["seed"], c["proc"], c["scale"], c["delta"], c["alpha"], c["rewire"], c["frac"], judge), 1000, 6000
        rows.append({**c, **measure_full(log, lo, hi, c["delta"])})
    json.dump(rows, open(f"tr_{env}_{judge}_{part}.json", "w"))


def load(env, judge):
    rows = [r for f in sorted(glob.glob(f"tr_{env}_{judge}_*.json")) for r in json.load(open(f))]
    return [r for r in rows if r["sigma_eff"] > 0 and r["sigma_node"] > 0 and not math.isnan(r["pass"])]


def as_feats(rows, target):
    """decouple.py 的拟合函数按 delta/sigma_eff 和 delta/(0.175·scale) 算特征；这里把实测 σ_node 折算进 scale，
    使 κ = δ / σ_node，并把要预测的指标放进 pass 字段。"""
    return [{"delta": r["delta"], "sigma_eff": r["sigma_eff"], "scale": r["sigma_node"] / 0.175, "pass": r[target]} for r in rows]


def score():
    for judge in ("软", "硬"):
        geo, alg, net = load("geo", judge), load("alg", judge), load("net", judge)
        print(f"\n== {judge}裁判：时间玩具 {len(geo)} / 代数玩具 {len(alg)} / 信息网络 {len(net)} ==")
        print("预测目标   拟合来源（零重拟合）         → 网络 R²")
        for target, name in (("pass", "穿透率"), ("miss", "漏检率")):
            for src, rows in (("时间玩具", geo), ("代数玩具", alg), ("两个玩具合并", geo + alg)):
                w = fit_sigmoid(as_feats(rows, target), "ρ+κ")
                print(f"{name:6s}  {src:22s} {r2(as_feats(net, target), pred_sigmoid(w, as_feats(net, target), 'ρ+κ')):8.3f}")
            half = len(net) // 2
            a, b = as_feats(net[:half], target), as_feats(net[half:], target)
            cv = (r2(b, pred_sigmoid(fit_sigmoid(a, "ρ+κ"), b, "ρ+κ")) + r2(a, pred_sigmoid(fit_sigmoid(b, "ρ+κ"), a, "ρ+κ"))) / 2
            only = [{**x, "scale": 1.0} for x in as_feats(net, target)]
            w = fit_sigmoid(as_feats(geo + alg, target), "ρ")
            print(f"{name:6s}  对照：网络内部对半交叉拟合 ρ+κ  {cv:8.3f}")
            print(f"{name:6s}  对照：两个玩具合并、只用 ρ      {r2(only, pred_sigmoid(w, only, 'ρ')):8.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        gen(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        score()
