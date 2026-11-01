"""真实值看不到时，用"答案之间的链条"挡错误。

环境：
  隐藏信号 x_t：switch.py 的规律切换序列（R1–R4 两遍 + 新规律 R5），系统看不到。
  12 个平庸的传感器：每一步各自报告"现在 x 是多少"，噪声各不相同（σ 0.05–0.3）。
  故障：故障时间表沿用 loop.py（一种型号坏 12%、两种一起坏 12%、零星 3%、第 4000 步换型号、第 6000 步出现新型号 D）。
        坏法三种（分别跑）：
          偏+3     报告 = x + 3
          偏+0.5   报告 = x + 0.5
          过时     报告 = 3 步之前的 x（看起来像真的，只是旧）
  真实答案只揭晓 1% / 0.1% / 0%。

裁判（报告分成两团时信哪团；不分团就取中位数）：
  V       中位数
  P       按"偏高团是哪几个传感器"记信任表，只用真实答案学
  链条    学一条规律：用过去 6 步的系统估计值预测这一步。分团时信离预测近的那团。
          规律从两种数据学：真实答案；以及"12 个传感器意见一致"那些步的中位数
          （故障一定会让报告分团，所以一致的步可以当作干净的）
  链条+P  链条两团差距明显（离预测的距离差 > 0.5）时听链条，否则听 P；
          链条有把握时的判断也拿来更新 P 的信任表
  链条直判 不等分团：每一步按"离规律预测有多远"给每个报告加权（尺度 = 规律近期的预测误差），取加权中位数。
          规律只在"加权结果和普通中位数一致"的步上拿输出学（外加真实答案），免得把自己的判断学进去
  交叉链   每个传感器同时挂在三条链上，各自判断"这一步断没断"（偏离超过该传感器平时波动的 2.5 倍）：
            时间链  报告 vs 规律预测
            自身链  这一步的变化量 vs 预测的变化量（自己有没有突然跳一下）
            关系链  和全体中位数的差 vs 平时的差
          关系链只负责分组（哪些传感器一起偏离了平时关系），哪一组在说谎由时间链+自身链的断裂程度决定，
          说谎的那组权重≈0；关系没变时，时间链和自身链都断权重≈0、断一条 0.5。
          各链的"平时波动"只在完全没断链时更新（否则会把连续故障学成正常）
漏过：过半传感器坏的步上，输出离真值比离"坏传感器的中位数"更近于后者（偏离超过故障幅度的一半）。

用法：python latent.py run FAULT [N_SEEDS]  →  latent_FAULT.json
      python latent.py report
"""
import json
import statistics as st
import sys

import torch
import torch.nn as nn

from loop import N, fault_plan, split
from robust import wmedian
from switch import make_series

JUDGES = ["V", "P", "链条", "链条+P", "链条直判", "交叉链"]
FAULTS = {"偏+3": 3.0, "偏+0.5": 0.5, "过时": "stale"}
RATES = [0.01, 0.001, 0.0]


def run(judge, seed, fault, rate, plan):
    data, _ = make_series(seed)
    xs = [y for _, y in data]
    g = torch.Generator().manual_seed(seed)
    sigma = torch.rand(N, generator=g) * 0.25 + 0.05
    lab = (torch.rand(len(xs), generator=g) < rate).tolist()
    torch.manual_seed(seed)
    f = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(f.parameters(), lr=3e-3)
    trust_err, H, log = {}, [0.0] * 6, []
    scale = 0.3                                          # 链条直判：规律近期预测误差的尺度
    s_t, s_o, s_r = torch.full((N,), 0.3), torch.full((N,), 0.3), torch.full((N,), 0.3)   # 交叉链：三条链各自的平时波动
    base_r = torch.zeros(N)                              # 交叉链：和全体中位数的平时差
    prev_r, prev_out = None, 0.0
    for t, x in enumerate(xs):
        m = plan[t][0]
        r = x + sigma * torch.randn(N, generator=g)
        if fault == "stale":
            r = torch.where(m, xs[max(0, t - 3)] + sigma * torch.randn(N, generator=g), r)
        else:
            r = r + fault * m.float()

        h = torch.tensor(H)
        pred = f(h)
        high = split(r)
        confident = False
        if judge == "链条直判":
            w = torch.exp(-((r - pred.detach()) ** 2) / (2 * scale ** 2)) + 1e-6
            out = wmedian(r, w)
            plain = r.median()
        elif judge == "交叉链":
            pd = pred.detach()
            z_t = (r - pd).abs() / s_t
            res_o = (r - prev_r) - (pd - prev_out) if prev_r is not None else torch.zeros(N)
            z_o = res_o.abs() / s_o
            res_r = (r - r.median()) - base_r
            moved = (res_r / s_r).abs() > 2.5          # 关系链：哪些传感器一起偏离了平时的关系
            if moved.any() and not moved.all():
                # 关系链只分组；哪一组在说谎，由锚在历史上的时间链+自身链决定
                bad_side = moved if (z_t + z_o)[moved].mean() > (z_t + z_o)[~moved].mean() else ~moved
                w = torch.where(bad_side, 1e-6, 1.0)
            else:
                nb = (z_t > 2.5).int() + (z_o > 2.5).int()
                w = torch.where(nb == 0, 1.0, torch.where(nb == 1, 0.5, 1e-6))
            out = wmedian(r, w)
            plain = r.median()
            clean = w == 1.0                             # 平时波动只在完全没断链时更新，免得把故障学成正常
            s_t = torch.where(clean, (0.98 * s_t ** 2 + 0.02 * (r - pd) ** 2).sqrt().clamp(0.05, 1.0), s_t)
            s_o = torch.where(clean, (0.98 * s_o ** 2 + 0.02 * res_o ** 2).sqrt().clamp(0.05, 1.0), s_o)
            s_r = torch.where(clean, (0.98 * s_r ** 2 + 0.02 * res_r ** 2).sqrt().clamp(0.05, 1.0), s_r)
            base_r = torch.where(clean, 0.98 * base_r + 0.02 * (r - r.median()), base_r)
            prev_r = r
        elif high is None or judge == "V":
            out = r.median()
        else:
            hi = torch.tensor(sorted(high))
            lo = torch.tensor([i for i in range(N) if i not in high])
            med_hi, med_lo = r[hi].median(), r[lo].median()
            d_hi, d_lo = (med_hi - pred).abs().item(), (med_lo - pred).abs().item()
            chain_pick = med_hi if d_hi < d_lo else med_lo
            confident = abs(d_hi - d_lo) > 0.5
            key = tuple(sorted(high))
            e = trust_err.setdefault(key, torch.full((N,), 0.05))
            p_pick = wmedian(r, torch.exp(-e / 0.1))
            out = chain_pick if judge == "链条" or (judge == "链条+P" and confident) else p_pick
        out = out.detach()

        # 学规律：真实答案，或者大家一致时的中位数
        tgt = x if lab[t] else (out.item() if high is None else None)
        if judge in ("链条直判", "交叉链"):
            tgt = x if lab[t] else (out.item() if abs(out.item() - plain.item()) < 0.1 else None)
            if tgt is not None:
                scale = max(0.05, (0.98 * scale ** 2 + 0.02 * (pred.item() - tgt) ** 2) ** 0.5)
        if tgt is not None and judge in ("链条", "链条+P", "链条直判", "交叉链"):
            loss = (pred - tgt) ** 2
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        # 学信任表：真实答案；链条+P 还用链条有把握时的判断
        if high is not None and judge in ("P", "链条+P"):
            truth = x if lab[t] else (out.item() if judge == "链条+P" and confident else None)
            if truth is not None:
                trust_err[key] = 0.9 * trust_err[key] + 0.1 * ((r - truth) ** 2).clamp(max=1.0)

        H = H[1:] + [out.item()]
        prev_out = out.item()
        bad = r[m].median().item() if m.any() else None
        log.append((out.item(), x, int(m.sum()), bad))
    return log


def leak_rate(log, lo, hi):
    hits = []
    for out, x, nbad, bad in log[lo:hi]:
        if nbad * 2 > N and bad is not None and abs(bad - x) > 0.2:
            hits.append(abs(out - x) > 0.5 * abs(bad - x))
    return sum(hits) / len(hits) if hits else float("nan")


def do_run(fault_name, n_seeds):
    torch.set_num_threads(1)
    fault = FAULTS[fault_name]
    out = {}
    for seed in range(n_seeds):
        plan = fault_plan(seed)
        for rate in RATES:
            for judge in JUDGES:
                if judge == "P" and rate == 0.0:
                    continue
                log = run(judge, seed, fault, rate, plan)
                clean_err = st.mean((o - x) ** 2 for o, x, nb, _ in log[2000:] if nb == 0)
                out.setdefault(f"{rate}|{judge}", []).append(
                    {"all": leak_rate(log, 2000, 9500), "late": leak_rate(log, 6500, 9500), "clean_err": clean_err})
        print(fault_name, seed, flush=True)
    with open(f"latent_{fault_name}.json", "w") as f:
        json.dump(out, f)


def report():
    med = st.median
    for fault_name in FAULTS:
        data = json.load(open(f"latent_{fault_name}.json"))
        print(f"\n== 坏法：{fault_name}　过半传感器坏时的漏过率（全程 / 最后 3000 步），括号 = 无故障步的误差 ==")
        print("答案比例 " + "".join(f"{j:>22s}" for j in JUDGES))
        for rate in RATES:
            cells = []
            for j in JUDGES:
                r = data.get(f"{rate}|{j}")
                cells.append("—" if r is None else
                             f"{med([x['all'] for x in r]):.0%}/{med([x['late'] for x in r]):.0%} ({med([x['clean_err'] for x in r]):.3f})")
            print(f"{rate:7.1%} " + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        report()
