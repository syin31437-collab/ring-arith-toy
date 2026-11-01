"""观测系统：测"节点 i 的数值偏一点，节点 j 跟着偏多少"，不针对任何错误类型。

做法：
  - 同一个种子跑两遍：一遍干净，一遍给所有节点对外显示的值加随机扰动。
  - 扰动 = ε × 随机正负号，各节点互相独立；正负号每 B 步换一次（B 决定错误的时间尺度）。
  - 节点 j 的响应 = 它自己的预测（不含扰动）在两遍之间的差 d_j。
  - 权重 W[i][j] = E[d_j × s_i] / ε：i 偏 +1 单位时 j 平均偏多少。
    因为各节点的正负号互相独立，每一对可以单独估计，不用联立回归。
  - 扫幅度 ε 和时间尺度 B：权重随幅度/时长变化，说明有非线性（比如中位数把大偏差剔掉）。

检验：A 组节点互不相连，W 必须正好为 0。

用法：python observer.py run SYSTEM [N_SEEDS]  →  obs_SYSTEM.json
      python observer.py report
"""
import json
import sys

import torch

from switch import LAM, N, Nodes, make_series

EPSILONS = [0.01, 0.1, 0.5, 2.0]
BLOCKS = [1, 20, 200]
SYSTEMS = ["A", "C", "C-trim"]


def run(system, seed, eps, block, q=1.0):
    """返回 (每步各节点自己的预测, 每步各节点的扰动正负号)。eps=0 即干净的一遍。
    q：每块里每个节点被扰动的概率。q=1 所有节点同时扰动；q 小时多数块只有一个节点在出错。"""
    data, _ = make_series(seed)
    nodes = Nodes(seed)
    g = torch.Generator().manual_seed(7000 + seed)
    preds, signs = [], []
    s = torch.zeros(N)
    for t, (hist, y) in enumerate(data):
        if t % block == 0:
            s = torch.randint(0, 2, (N,), generator=g).float() * 2 - 1
            s = s * (torch.rand(N, generator=g) < q).float()
        p = nodes(hist)
        preds.append(p.tolist())
        signs.append(s.tolist())
        shown = p.detach() + eps * s
        loss_each = (p - y) ** 2
        if system in ("C", "C-trim"):
            left, right = shown.roll(1), shown.roll(-1)
            if system == "C":
                target = (left + right) / 2
            else:
                target = torch.stack([left, shown, right]).median(0).values
            loss_each = loss_each + LAM * (p - target) ** 2
        nodes.opt.zero_grad()
        loss_each.sum().backward()
        nodes.opt.step()
    return torch.tensor(preds), torch.tensor(signs)


def weights(system, seed, eps, block, q=1.0, single=False):
    """single=True：只用"这一块里恰好只有一个节点在出错"的块来估计。"""
    clean, _ = run(system, seed, 0.0, block)
    dirty, signs = run(system, seed, eps, block, q)
    d = dirty - clean                                  # (T, N)
    # 第 t 步的扰动最早影响第 t+1 步的预测；按块取后半段，让慢的传播也来得及显现
    T = d.shape[0]
    rows_s, rows_d = [], []
    for start in range(0, T - block - 1, block):
        lo = start + block // 2
        rows_s.append(signs[start])
        rows_d.append(d[lo + 1:start + block + 1].mean(0))
    S, D = torch.stack(rows_s), torch.stack(rows_d)   # (块数, N)
    if single:
        keep = (S != 0).sum(1) == 1
        S, D = S[keep], D[keep]
    W = (S.T @ D) / (S ** 2).sum(0).clamp(min=1).unsqueeze(1) / eps   # W[i][j]
    return W.tolist()


def do_run(system, n_seeds):
    torch.set_num_threads(1)
    out = {}
    for eps in EPSILONS:
        for block in BLOCKS:
            key = f"{eps}|{block}"
            out[key] = [weights(system, seed, eps, block) for seed in range(n_seeds)]
            print(system, key, flush=True)
    with open(f"obs_{system}.json", "w") as f:
        json.dump(out, f)


def ring_dist(i, j):
    k = abs(i - j) % N
    return min(k, N - k)


def report():
    for system in SYSTEMS:
        data = json.load(open(f"obs_{system}.json"))
        print(f"\n===== {system}：按环上距离平均的权重 W（i 偏 +1，j 平均偏多少；10 种子平均） =====")
        print("幅度ε  时长B    自己(0)    邻居(1)    隔一个(2)   对面(3)    |W|最大")
        for key, mats in data.items():
            eps, block = key.split("|")
            W = torch.tensor(mats).mean(0)
            by = {k: [] for k in range(4)}
            for i in range(N):
                for j in range(N):
                    by[ring_dist(i, j)].append(W[i][j].item())
            cells = "  ".join(f"{sum(v)/len(v):+.5f}" for v in by.values())
            print(f"{float(eps):5.2f}  {int(block):4d}   {cells}   {W.abs().max().item():.5f}")
        W = torch.tensor(data["0.1|200"]).mean(0)
        print(f"\n  ε=0.1, B=200 的完整矩阵（行 i = 出错的节点，列 j = 受影响的节点）")
        for i in range(N):
            print("   " + " ".join(f"{W[i][j].item():+.4f}" for j in range(N)))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    else:
        report()
