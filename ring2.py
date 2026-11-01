"""第二轮：30 个种子 + 训练轨迹 + 门控臂。

在 ring.py 的基础上：
  - 每 100 步记录一次：泛化区误差、训练区误差、泛化区分歧、按离训练区远近分段的误差
  - 新臂 D+gate：锚点对得上才信分歧。20 个锚点里 10 个拿来训练，10 个留出来当门，
    门上的相对误差越大，分歧消解的权重越小（w = exp(-err/0.05)，不回传梯度）

用法：python ring2.py ARM [N_SEEDS]   结果写到 traj_ARM.json
"""
import json
import sys

import torch

from ring import ID_RANGE, OOD_RANGE, STEPS, Ring, mlp, norm, sample

LOG_EVERY = 100
GATE_TAU = 0.05
BINS = [(10, 15), (15, 20), (20, 25), (25, 30)]   # 按 max(a,b) 分段：离训练区越来越远


def rel(pred, c):
    return ((pred - c).abs() / c).median().item()


def make_eval_set(seed):
    g = torch.Generator().manual_seed(99999 + seed)
    a_id, b_id = sample(2000, *ID_RANGE, g)
    a_ood, b_ood = sample(4000, *OOD_RANGE, g)
    far = torch.max(a_ood, b_ood).squeeze(1)
    return norm(a_id, b_id, a_id * b_id), norm(a_ood, b_ood, a_ood * b_ood), far


@torch.no_grad()
def snapshot(predict, ring, ev):
    (ai, bi, ci), (ao, bo, co), far = ev
    po = predict(ao, bo)
    row = {"id": rel(predict(ai, bi), ci), "ood": rel(po, co)}
    for lo, hi in BINS:
        m = (far >= lo) & (far < hi)
        row[f"ood_{lo}_{hi}"] = rel(po[m], co[m])
    if ring is not None:
        row["ood_consistency"] = ring.consistency_loss(ao, bo).sqrt().item()
    return row


def run(arm, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    a_id, b_id = sample(2000, *ID_RANGE, g)
    a_sp, b_sp = sample(20, *OOD_RANGE, g)
    a_un, b_un = sample(2000, *OOD_RANGE, g)
    a_id, b_id, c_id = norm(a_id, b_id, a_id * b_id)
    a_sp, b_sp, c_sp = norm(a_sp, b_sp, a_sp * b_sp)
    a_un, b_un, _ = norm(a_un, b_un, a_un * b_un)

    n_fit = {"A": 0, "D": 0, "A+sparse": 20, "D+sparse": 20, "D+gate": 10}[arm]
    a_lab = torch.cat([a_id, a_sp[:n_fit]])
    b_lab = torch.cat([b_id, b_sp[:n_fit]])
    c_lab = torch.cat([c_id, c_sp[:n_fit]])
    a_gate, b_gate, c_gate = a_sp[10:], b_sp[10:], c_sp[10:]

    single = arm.startswith("A")
    if single:
        model, ring = mlp(), None
        predict = lambda a, b: model(torch.cat([a, b], 1))
        params = model.parameters()
    else:
        ring = Ring()
        predict = ring.c
        params = ring.parameters()
    opt = torch.optim.Adam(params, lr=1e-3)
    ev = make_eval_set(seed)

    traj, weights = [], []
    for step in range(STEPS + 1):
        if step % LOG_EVERY == 0:
            row = snapshot(predict, ring, ev)
            row["step"] = step
            traj.append(row)
        if step == STEPS:
            break
        if single:
            loss = ((predict(a_lab, b_lab) - c_lab) ** 2).mean()
        else:
            w = 1.0
            if arm == "D+gate":
                with torch.no_grad():
                    w = torch.exp(-torch.tensor(rel(predict(a_gate, b_gate), c_gate)) / GATE_TAU).item()
                if step % LOG_EVERY == 0:
                    weights.append(w)
            loss = ring.reality_loss(a_lab, b_lab, c_lab) + w * ring.consistency_loss(a_un, b_un)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"seed": seed, "traj": traj, "gate_w": weights}


def main():
    torch.set_num_threads(2)
    arm = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out = []
    for seed in range(n):
        out.append(run(arm, seed))
        print(arm, seed, f"ood={out[-1]['traj'][-1]['ood']:.3f}", flush=True)
    with open(f"traj_{arm}.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
