"""最小成立条件在真实数据上验证：B（至少一个诚实节点）、C（已知干净的校准期）、D（节点身份跨题稳定）（2026-11-01，库伯）。

数据：17 套真实数据（四轮大规模 + 盲测五，invert.sets），每套取 δ = 10%、20% 两档，R0 约束，软裁判。
  校准 = δ = 0 那一轮（已知干净）；评估 = 投毒那一轮的全部题（每题都有毒）。
  预测：推导版第五版；识别：z = 平均偏移 / √(节点标准误² + b 的标准误²)，w < 0.5 判投毒；平衡准确率只在"原理上认得出"
    （真实传出的毒：同一题 投毒轮 − 干净轮 的平均 / 标准误 > 3）的投毒节点上算命中率；卧底按投毒算。
拿掉的方式：
  B  诚实节点也被带偏：投毒那一轮，诚实节点的报告加上同一题投毒节点的平均偏移（半合成：噪声是真的，只是让所有人一起偏）
  C  校准期只用前 n 题（n = 60 / 40 / 20 / 10 / 5）；"不知道哪段干净" = 只用投毒那一轮自己的题估秤和噪声
  D  投毒那一轮每题随机打乱节点顺序

预注册（11-01，运行前；玩具结论搬到真实数据上）：
  0. 基线：识别平衡准确率 ≥ 0.85；预测 MAE ≤ 0.05
  B：预测 MAE ≤ 0.07（推导版能如实预测"会被骗"），识别 < 0.70
  C：每档预测 MAE ≤ 0.07；识别 20 题 ≥ 0.75、5 题 < 0.75、"不知道哪段干净" < 0.70
  D：预测 MAE ≤ 0.07，识别 < 0.70

用法：~/gut-external/torch215-venv/bin/python realcond.py
"""
import math
import random
import statistics as st

import torch

from analytic import derive5, sigma_c
from hetero import judge_out
from invert import sets


def identify(est, ev, nodes, poisoned, cl_by):
    e = [p - r.median().item() for _, p, r, _ in est]
    b = st.median(e)
    se_b = 1.2533 * 1.4826 * st.median(abs(x - b) for x in e) / math.sqrt(len(e))
    X = torch.stack([r - (p - b) for _, p, r, _ in est + ev])
    se = X.std(0) / math.sqrt(len(X))
    z = X.mean(0) / torch.sqrt(se ** 2 + se_b ** 2)
    flag = (torch.exp(-0.5 * z ** 2) < 0.5).tolist()
    hon = [i for i, n in enumerate(nodes) if n not in poisoned]
    tnr = st.mean(not flag[i] for i in hon)
    hits = []
    for i, n in enumerate(nodes):
        if n in poisoned and cl_by is not None:
            sh = [(r[i] - cl_by[k][i]).item() for k, r in cl_by["_ev"]]
            if abs(st.mean(sh)) / max(se[i].item(), 1e-12) > 3:
                hits.append(flag[i])
    return (st.mean(hits) + tnr) / 2 if hits else tnr


def run(est, ev_rows, nodes, poisoned, cl_by):
    est4 = [(False, p, r, x) for _, p, r, x in est]
    ev4 = [(True, p, r, x) for _, p, r, x in ev_rows]
    s = max(sigma_c(est4)[1], 1e-9)
    act = st.mean(abs(judge_out(r, torch.tensor(p), s, "软")) > x / 2 for _, p, r, x in ev4)
    pred = derive5(est4 + ev4, "软", s)
    return abs(act - pred), identify(est4, ev4, nodes, poisoned, cl_by)


def main():
    rng = random.Random(0)
    res = {}
    for name, data, nodes, always, sleeper, deltas in sets():
        poisoned = set(always) | ({sleeper} if sleeper else set())
        clean5 = data[("R0", 0.0)]
        clean = [(f, p, r, x) for f, p, r, x, _ in clean5]
        cl = {k: r for _, _, r, _, k in clean5}
        for d in (0.1, 0.2):
            if d not in deltas:
                continue
            rows5 = data[("R0", d)]
            ev = [(f, p, r, x) for f, p, r, x, k in rows5 if k in cl]
            cl_by = dict(cl)
            cl_by["_ev"] = [(k, r) for _, _, r, _, k in rows5 if k in cl]
            pidx = [i for i, n in enumerate(nodes) if n in poisoned]
            hidx = [i for i, n in enumerate(nodes) if n not in poisoned]
            variants = {"基线": (clean, ev, cl_by)}
            # B：诚实节点也加上同题投毒节点的平均偏移
            evB = []
            for (f, p, r, x), (k, _) in zip(ev, cl_by["_ev"]):
                sh = torch.stack([r[i] - cl[k][i] for i in pidx]).mean()
                r2 = r.clone()
                r2[hidx] = r2[hidx] + sh
                evB.append((f, p, r2, x))
            variants["B 诚实节点也被带偏"] = (clean, evB, cl_by)
            for n_cal in (40, 20, 10, 5):
                variants[f"C 校准 {n_cal} 题"] = (clean[:n_cal], ev, cl_by)
            variants["C 不知道哪段干净"] = (ev, ev, cl_by)
            evD = [(f, p, r[torch.randperm(len(r), generator=torch.Generator().manual_seed(rng.randrange(10 ** 9)))], x) for f, p, r, x in ev]
            variants["D 身份每题打乱"] = (clean, evD, cl_by)
            for tag, (est, evv, cb) in variants.items():
                m, a = run(est, evv, nodes, poisoned, cb)
                res.setdefault(tag, ([], []))
                res[tag][0].append(m)
                res[tag][1].append(a)
    print("条件                    预测 MAE（34 格）   识别平衡准确率")
    for tag, (m, a) in res.items():
        print(f"{tag:20s}   {st.mean(m):.3f}              {st.mean(a):.2f}")


if __name__ == "__main__":
    main()
