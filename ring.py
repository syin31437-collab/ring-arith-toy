"""约束环玩具：乘法世界 c = a*b，三个平庸的小 MLP 互相约束。

节点：
  N_c: (a, b) -> c
  N_a: (c, b) -> a      # 反向运算 c/b
  N_b: (c, a) -> b      # 反向运算 c/a

两种误差（都靠梯度分摊，不追究是谁错）：
  现实误差：有标签的样本上，每个节点对真值的 MSE
  内部分歧：只有 (a,b) 没有标签的样本上，c_hat = N_c(a,b)，
            要求 N_a(c_hat,b) ≈ a、N_b(c_hat,a) ≈ b

训练区 a,b ∈ [2,10]；泛化区 a,b ∈ [10,30]（没见过）。
"""
import json
import sys

import torch
import torch.nn as nn

S_IN, S_OUT = 30.0, 900.0          # 固定缩放，输入 /30，乘积 /900
ID_RANGE, OOD_RANGE = (2, 10), (10, 30)
STEPS, N_SEEDS = 4000, 5
N_SPARSE = 20                      # 泛化区里少量有标签的锚点


def mlp():
    return nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


def sample(n, lo, hi, g):
    a = torch.rand(n, 1, generator=g) * (hi - lo) + lo
    b = torch.rand(n, 1, generator=g) * (hi - lo) + lo
    return a, b


def norm(a, b, c):
    return a / S_IN, b / S_IN, c / S_OUT


class Ring(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc, self.na, self.nb = mlp(), mlp(), mlp()

    def c(self, a, b):
        return self.nc(torch.cat([a, b], 1))

    def reality_loss(self, a, b, c):
        return (((self.c(a, b) - c) ** 2).mean()
                + ((self.na(torch.cat([c, b], 1)) - a) ** 2).mean()
                + ((self.nb(torch.cat([c, a], 1)) - b) ** 2).mean())

    def consistency_loss(self, a, b):
        ch = self.c(a, b)
        return (((self.na(torch.cat([ch, b], 1)) - a) ** 2).mean()
                + ((self.nb(torch.cat([ch, a], 1)) - b) ** 2).mean())


def train(arm, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    a_id, b_id = sample(2000, *ID_RANGE, g)
    a_sp, b_sp = sample(N_SPARSE, *OOD_RANGE, g)
    a_un, b_un = sample(2000, *OOD_RANGE, g)
    a_id, b_id, c_id = norm(a_id, b_id, a_id * b_id)
    a_sp, b_sp, c_sp = norm(a_sp, b_sp, a_sp * b_sp)
    a_un, b_un, _ = norm(a_un, b_un, a_un * b_un)          # 不使用乘积标签

    labeled = arm in ("A", "B", "D", "A+sparse", "D+sparse")
    sparse = arm.endswith("+sparse")
    consist = arm.startswith("C") or arm.startswith("D")
    n_models = 3 if arm == "B" else 1

    if arm in ("A", "B", "A+sparse"):
        models = [mlp() for _ in range(n_models)]
        params = [p for m in models for p in m.parameters()]
    else:
        ring = Ring()
        params = list(ring.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    if sparse:
        a_lab, b_lab, c_lab = (torch.cat([a_id, a_sp]), torch.cat([b_id, b_sp]), torch.cat([c_id, c_sp]))
    else:
        a_lab, b_lab, c_lab = a_id, b_id, c_id

    for _ in range(STEPS):
        loss = 0.0
        if arm in ("A", "B", "A+sparse"):
            for m in models:
                loss = loss + ((m(torch.cat([a_lab, b_lab], 1)) - c_lab) ** 2).mean()
        else:
            if labeled:
                loss = loss + ring.reality_loss(a_lab, b_lab, c_lab)
            if consist:
                # C 组没有标签：分歧就在训练区和泛化区上一起消解
                a_c = a_un if labeled else torch.cat([a_id, a_un])
                b_c = b_un if labeled else torch.cat([b_id, b_un])
                loss = loss + ring.consistency_loss(a_c, b_c)
        opt.zero_grad()
        loss.backward()
        opt.step()

    if arm in ("A", "B", "A+sparse"):
        return lambda a, b: torch.stack([m(torch.cat([a, b], 1)) for m in models]).mean(0), None
    return ring.c, ring


@torch.no_grad()
def evaluate(predict, ring, seed):
    g = torch.Generator().manual_seed(99999 + seed)
    out = {}
    for name, rng in (("id", ID_RANGE), ("ood", OOD_RANGE)):
        a, b = sample(5000, *rng, g)
        an, bn, cn = norm(a, b, a * b)
        rel = ((predict(an, bn) - cn).abs() / cn).median().item()
        out[f"{name}_relerr"] = rel
        if ring is not None:
            out[f"{name}_consistency"] = ring.consistency_loss(an, bn).sqrt().item()
    return out


def main():
    arms = sys.argv[1:] or ["A", "B", "C", "D", "A+sparse", "D+sparse"]
    results = {}
    for arm in arms:
        runs = []
        for seed in range(N_SEEDS):
            predict, ring = train(arm, seed)
            runs.append(evaluate(predict, ring, seed))
        results[arm] = runs
        keys = runs[0].keys()
        summary = {k: sorted(r[k] for r in runs)[N_SEEDS // 2] for k in keys}
        print(arm.ljust(10), "  ".join(f"{k}={v:.3f}" for k, v in summary.items()), flush=True)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
