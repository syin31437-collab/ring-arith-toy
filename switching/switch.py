"""会犯错的预测器群体 · 规律切换的序列。

环境：x_{t+1} = rule(最近 6 个值) + 噪声(σ=0.1)，隐藏规律每 1000 步切换一次。
  前 8 段：4 条规律各出现两遍（R1 R2 R3 R4 R1 R2 R3 R4）
  第 9 段：从没出现过的新规律 R5（1500 步）
节点：6 个小 MLP 在线学习（每步一次梯度），节点 i 只看到 lag {i, i+1, i+2} mod 6。

系统：
  A  独立：各自只对现实学习；系统误差 = 各节点单独预测误差的平均
  B  集成：同一批独立节点，系统输出 = 6 个预测的固定平均
  C  错误溶解：环上局部耦合，每个节点的损失 = 现实误差 + λ·(和左右邻居平均值的分歧)
  C-trim（扩展）：耦合目标换成 (左, 自己, 右) 三者的中位数，离谱的那个被自动剔除

坏节点：第 4500–4900 步，节点 0 停止学习，输出被加上 +3 的固定偏差。
每个种子同时跑一遍不加坏节点的对照，用来算污染。

用法：python switch.py [N_SEEDS]   结果写到 results.json
"""
import json
import math
import sys

import torch

N, LAGS, H = 6, 6, 32
SEG, NEW_SEG, NOISE = 1000, 1500, 0.1
LR, LAM = 3e-3, 1.0
BAD_NODE, BAD_FROM, BAD_TO, BAD_BIAS = 0, 4500, 4900, 3.0
VIEWS = torch.tensor([[(i + k) % LAGS for k in range(3)] for i in range(N)])


def rules():
    # x[0] 是最新的值，x[k] 是 k 步之前
    return {
        "R1": lambda x: 0.9 * x[0] - 0.5 * x[1],
        "R2": lambda x: -0.6 * x[2] + 0.3 * x[3],      # 特征根模 0.97，稳定
        "R3": lambda x: 0.8 * math.sin(3 * x[1]),
        "R4": lambda x: math.tanh(2.0 * x[0] * x[4]) - 0.3 * x[5],
        "R5": lambda x: 1.2 * abs(x[3]) - 0.6 * x[0] - 0.5,      # 只在最后出现
    }


def make_series(seed):
    g = torch.Generator().manual_seed(seed)
    r = rules()
    plan = ["R1", "R2", "R3", "R4"] * 2
    seq = [(name, SEG) for name in plan] + [("R5", NEW_SEG)]
    hist = [0.0] * LAGS
    xs, labels = [], []
    for name, length in seq:
        for _ in range(length):
            y = r[name](hist) + NOISE * torch.randn(1, generator=g).item()
            y = max(-3.0, min(3.0, y))
            xs.append((list(hist), y))
            labels.append(name)
            hist = [y] + hist[:-1]
    return xs, labels


class Nodes:
    """6 个独立参数的小 MLP，批量前向。"""

    def __init__(self, seed):
        g = torch.Generator().manual_seed(seed)
        s1, s2 = 1 / math.sqrt(3), 1 / math.sqrt(H)
        self.p = [
            ((torch.rand(N, 3, H, generator=g) * 2 - 1) * s1).requires_grad_(),
            torch.zeros(N, H, requires_grad=True),
            ((torch.rand(N, H, generator=g) * 2 - 1) * s2).requires_grad_(),
            torch.zeros(N, requires_grad=True),
        ]
        self.opt = torch.optim.Adam(self.p, lr=LR)

    def __call__(self, hist):
        x = torch.tensor(hist)[VIEWS]                         # (N, 3)
        w1, b1, w2, b2 = self.p
        h = torch.relu(torch.einsum("nd,ndh->nh", x, w1) + b1)
        return (h * w2).sum(1) + b2                           # (N,)


def run(system, seed, corrupt, bad=(BAD_NODE,), preds=None):
    """bad：坏掉的节点编号；preds：传入一个列表时，逐步记录 (各节点对外显示的预测, 真值)。"""
    data, _ = make_series(seed)
    nodes = Nodes(seed)
    mask = torch.zeros(N)
    mask[list(bad)] = 1.0
    sys_err, node_err = [], []
    for t, (hist, y) in enumerate(data):
        p = nodes(hist)
        broken = corrupt and BAD_FROM <= t < BAD_TO
        shown = p.clone()
        if broken:
            shown = shown + BAD_BIAS * mask
        if preds is not None:
            preds.append((shown.tolist(), y))

        with torch.no_grad():
            e = (shown - y) ** 2
            node_err.append(e.tolist())
            sys_err.append(e.mean().item() if system == "A" else ((shown.mean() - y) ** 2).item())

        loss_each = (p - y) ** 2
        if system in ("C", "C-trim"):
            s = shown.detach()
            left, right = s.roll(1), s.roll(-1)
            if system == "C":
                target = (left + right) / 2
            else:
                target = torch.stack([left, s, right]).median(0).values
            loss_each = loss_each + LAM * (p - target) ** 2
        if broken:
            loss_each = loss_each * (1 - mask)
        nodes.opt.zero_grad()
        loss_each.sum().backward()
        nodes.opt.step()
    return sys_err, node_err


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    torch.set_num_threads(1)
    out = {}
    for system in ("A", "B", "C", "C-trim"):
        out[system] = []
        for seed in range(n_seeds):
            # B 和 A 用的是同一种独立学习，只是系统输出的读法不同，所以各自完整跑一遍
            se, ne = run(system, seed, corrupt=True)
            se0, ne0 = run(system, seed, corrupt=False)
            out[system].append({"seed": seed, "sys": se, "node": ne, "sys_clean": se0, "node_clean": ne0})
            print(system, seed, f"mean_sys={sum(se)/len(se):.4f}", flush=True)
    with open("results.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
