"""对比实验：平庸节点 + 学出来的结构 vs 冗余 + 投票，在相关故障下。

环境：switch.py 的规律切换序列（R1–R4 两遍 + 新规律 R5），12 个节点，节点 i 看 lag {i, i+1, i+2} mod 6。
节点：两段式小 MLP。第一段只看自己的视野得到 base；第二段（R、L 组）再加一个"信息源 base 的中位数"作输入。
故障（加在对外显示的值上，+3）：
  第 2000–4000 步  单点：每 100 步随机一个节点坏 50 步
  第 4000–6500 步  共模：|x_t| > 0.35 时，A 型 4 个节点一起坏
  第 6500–9500 步  多数共模：|x_t| > 0.35 时，A 型 + B 型 8 个节点一起坏（超过一半）
  型号：节点 0,3,6,9 = A，1,4,7,10 = B，2,5,8,11 = C（和视野交错，同型号不共享视野）

四组：
  V    不连，输出 = 12 个最终预测的中位数
  V+W  不连，输出 = 按"当前状态格子里各节点最近误差"加权的中位数
  R    环：信息源 = 左右邻居 + 自己，取中位数；输出同 V
  L    学出来：每 500 步按 互补增益 × 可靠性 重选 3 个信息源（每个节点最多被 3 人依赖），
       信息源和输出都用按状态加权的中位数
状态格子：按最新值 x_t 分 6 格（边界 ±0.25、±0.5、0），故意和触发阈值 0.35 错开。

用法：python robust.py run SYSTEM [N_SEEDS]  →  robust_SYSTEM.json
      python robust.py report
"""
import json
import math
import statistics as st
import sys

import torch

from switch import make_series

N, LAGS, H, K = 12, 6, 32, 3
LR, BIAS, THETA = 3e-3, 3.0, 0.35
TYPE = torch.tensor([i % 3 for i in range(N)])            # 0=A 1=B 2=C
VIEW = torch.tensor([[(i + k) % LAGS for k in range(3)] for i in range(N)])
EDGES = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5])        # 6 个状态格子
PHASES = {"单点": (2000, 4000), "共模A": (4000, 6500), "多数共模AB": (6500, 9500)}
SYSTEMS = ["V", "V+W", "R", "L"]


class MLP:
    def __init__(self, d_in, g):
        s1, s2 = 1 / math.sqrt(d_in), 1 / math.sqrt(H)
        self.p = [((torch.rand(N, d_in, H, generator=g) * 2 - 1) * s1).requires_grad_(),
                  torch.zeros(N, H, requires_grad=True),
                  ((torch.rand(N, H, generator=g) * 2 - 1) * s2).requires_grad_(),
                  torch.zeros(N, requires_grad=True)]

    def __call__(self, x):
        w1, b1, w2, b2 = self.p
        return (torch.relu(torch.einsum("nd,ndh->nh", x, w1) + b1) * w2).sum(1) + b2


def wmedian(vals, w):
    """vals (M,), w (M,) 非负 → 加权中位数。"""
    order = torch.argsort(vals)
    cw = torch.cumsum(w[order], 0)
    return vals[order][torch.searchsorted(cw, cw[-1] / 2).clamp(max=len(vals) - 1)]


def faults(t, x_now, g):
    m = torch.zeros(N, dtype=torch.bool)
    if 2000 <= t < 4000 and (t % 100) < 50:
        m[int(torch.randint(0, N, (1,), generator=torch.Generator().manual_seed(int(g) * 100000 + t // 100)))] = True
    elif 4000 <= t < 6500 and abs(x_now) > THETA:
        m |= TYPE == 0
    elif t >= 6500 and abs(x_now) > THETA:
        m |= (TYPE == 0) | (TYPE == 1)
    return m


def run(system, seed, corrupt):
    data, _ = make_series(seed)
    g = torch.Generator().manual_seed(seed)
    base, fin = MLP(3, g), MLP(4, g)
    params = base.p + (fin.p if system in ("R", "L") else [])
    opt = torch.optim.Adam(params, lr=LR)

    ctx_err = torch.full((6, N), 0.05)                     # 每个状态格子里各节点 base 的近期误差
    src = torch.stack([(torch.arange(N) - 1) % N, torch.arange(N), (torch.arange(N) + 1) % N], 1)  # R 的信息源
    # L 的互补增益：节点 j 的残差和节点 i 的 base 的相关（指数滑动统计）
    ema = {"s": torch.zeros(N), "r": torch.zeros(N), "ss": torch.ones(N), "rr": torch.ones(N), "sr": torch.zeros(N, N)}
    log = []
    for t, (hist, y) in enumerate(data):
        x = torch.tensor(hist)[VIEW]
        bin_ = int(torch.bucketize(torch.tensor(hist[0]), EDGES))
        bad = faults(t, hist[0], seed) if corrupt else torch.zeros(N, dtype=torch.bool)
        bias = BIAS * bad.float()

        b = base(x)
        b_shown = b.detach() + bias
        trust = torch.exp(-ctx_err[bin_] / 0.1)

        if system in ("R", "L"):
            vals = b_shown[src]                            # (N, K)
            if system == "R":
                msg = vals.median(1).values
            else:
                msg = torch.stack([wmedian(vals[j], trust[src[j]]) for j in range(N)])
            p = fin(torch.cat([x, msg.unsqueeze(1)], 1))
        else:
            p = b
        p_shown = p.detach() + bias

        out = wmedian(p_shown, trust) if system in ("V+W", "L") else p_shown.median()
        log.append((p_shown.tolist(), out.item(), y, bad.tolist()))

        loss = ((b - y) ** 2).sum() + (((p - y) ** 2).sum() if system in ("R", "L") else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            e = ((b_shown - y) ** 2).clamp(max=1.0)
            ctx_err[bin_] = 0.9 * ctx_err[bin_] + 0.1 * e
            if system == "L":
                r = y - b_shown
                d = 0.995
                ema["s"] = d * ema["s"] + (1 - d) * b_shown
                ema["r"] = d * ema["r"] + (1 - d) * r
                ema["ss"] = d * ema["ss"] + (1 - d) * b_shown ** 2
                ema["rr"] = d * ema["rr"] + (1 - d) * r ** 2
                ema["sr"] = d * ema["sr"] + (1 - d) * torch.outer(b_shown, r)
                if t % 500 == 499:
                    src = choose_sources(ema, ctx_err)
    return log


def choose_sources(ema, ctx_err):
    """score[i][j] = i 的 base 能解释 j 残差的比例(R²) × i 的整体可靠性；每个 i 最多被 K 个节点选。"""
    vs = (ema["ss"] - ema["s"] ** 2).clamp(min=1e-8)
    vr = (ema["rr"] - ema["r"] ** 2).clamp(min=1e-8)
    cov = ema["sr"] - torch.outer(ema["s"], ema["r"])
    r2 = cov ** 2 / torch.outer(vs, vr)
    rel = torch.exp(-ctx_err.max(0).values / 0.1)          # 在最差的那个状态格子里有多不靠谱
    score = r2 * rel.unsqueeze(1)
    score.fill_diagonal_(-1)
    load = torch.zeros(N, dtype=torch.long)
    chosen = [[j] for j in range(N)]                       # 自己的 base 总是一个信息源
    for flat in torch.argsort(score.flatten(), descending=True).tolist():
        i, j = divmod(flat, N)
        if score[i, j] < 0 or len(chosen[j]) >= K or load[i] >= K:
            continue
        chosen[j].append(i)
        load[i] += 1
    for j in range(N):                                     # 没选满的用环上邻居补
        for i in ((j - 1) % N, (j + 1) % N, (j + 2) % N):
            if len(chosen[j]) < K and i not in chosen[j]:
                chosen[j].append(i)
    return torch.tensor(chosen)


def do_run(system, n_seeds):
    torch.set_num_threads(1)
    out = []
    for seed in range(n_seeds):
        out.append({"dirty": run(system, seed, True), "clean": run(system, seed, False)})
        print(system, seed, flush=True)
    with open(f"robust_{system}.json", "w") as f:
        json.dump(out, f)


def summarize(rec, lo, hi, only_active):
    """返回：输出误差/对照、输出明显被带偏(误差>1)的步数占比、健康节点误差/对照、受影响(>2x)的健康节点占比。"""
    o_d, o_c, h_d, h_c, hit, n = [], [], [], [], 0, 0
    affected = []
    for t in range(lo, hi):
        p, out, y, bad = rec["dirty"][t]
        pc, outc, _, _ = rec["clean"][t]
        if only_active and not any(bad):
            continue
        n += 1
        o_d.append((out - y) ** 2)
        o_c.append((outc - y) ** 2)
        hit += (out - y) ** 2 > 1.0
        hd = [(p[j] - y) ** 2 for j in range(N) if not bad[j]]
        hc = [(pc[j] - y) ** 2 for j in range(N) if not bad[j]]
        h_d.append(st.mean(hd))
        h_c.append(st.mean(hc))
        affected.append(st.mean([a > 2 * c + 1e-3 for a, c in zip(hd, hc)]))
    return st.mean(o_d) / st.mean(o_c), hit / n, st.mean(h_d) / st.mean(h_c), st.mean(affected)


def report():
    data = {s: json.load(open(f"robust_{s}.json")) for s in SYSTEMS}
    med = st.median
    print("（只统计有故障发生的步；中位数跨种子）")
    for phase, (lo, hi) in PHASES.items():
        print(f"\n== {phase} ==")
        print("系统   输出误差/对照   输出被带偏(>1)的步占比   健康节点误差/对照   受影响(>2x)的健康节点占比")
        for s in SYSTEMS:
            rows = [summarize(r, lo, hi, True) for r in data[s]]
            print(f"{s:5s} {med([x[0] for x in rows]):12.2f}x {med([x[1] for x in rows]):18.1%} "
                  f"{med([x[2] for x in rows]):20.2f}x {med([x[3] for x in rows]):22.1%}")
        lo2 = lo + (hi - lo) // 2
        print("  后半段（学过之后）：" + "  ".join(
            f"{s} 输出带偏 {med([summarize(r, lo2, hi, True)[1] for r in data[s]]):.1%}" for s in SYSTEMS))
    print("\n== 没有故障时的代价（干净运行全程输出误差，中位数） ==")
    for s in SYSTEMS:
        print(f"{s:5s} {med([st.mean((o - y) ** 2 for _, o, y, _ in r['clean']) for r in data[s]]):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        report()
