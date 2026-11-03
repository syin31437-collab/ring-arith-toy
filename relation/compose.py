"""事后探索（未预注册）：投放的力量能不能相加？
F3 把"集合 + 理由"当成独立形式，低估 set2t_r。假设：s(y) = w·[y = x] + (g_set·[y ∈ 集合] + g_L3·[y = 理由指向的选项])·(1 − v·假)。
只用没有理由的投放拟合（set_r 完全留出），预测 set2t_r、set2f_r；不多一个参数。对照：F3 单独给 g_set_r（自己见过数据）。
"""
import json

import torch

import analyze as AN
import formula as FM
import stage2 as S2

ROWS, items, NODES = FM.ROWS, FM.items, FM.NODES


def tens(rows):
    d = FM.tensors(rows)
    n = len(rows)
    R = torch.zeros(n, 4)
    is_r = torch.zeros(n)
    for i, r in enumerate(rows):
        if r["p"] in ("set2t_r", "set2f_r"):
            it = items[r["id"]]
            R[i, it["gold"] if r["p"] == "set2t_r" else S2.wrong(it)[0]] = 1
            is_r[i] = 1
    d["R"], d["is_r"] = R, is_r
    return d


def add(p, d):
    """p = w0 w1 g_L2 g_L3 g_L4 g_set v；set_r = g_set·集合 + g_L3·理由目标。"""
    w = p[0] + p[1] * d["c"]
    plain = (d["is_r"] == 0).float()
    idx = d["F"].clamp(max=3)                 # 0 L2  1 L3  2 L4  3 set
    g = p[2:6][idx] * d["has"]
    fac = 1 - p[6] * d["fl"]
    s = w[:, None] * d["X"] + (g * fac)[:, None] * d["T"] + (p[3] * fac * d["is_r"])[:, None] * d["R"]
    return torch.softmax(s, 1)


print(f"{'节点':14s}{'log-loss 可加（留出）':>22s}{'F3（见过）':>12s}{'混合 F2（见过）':>14s} ｜ B(set2t_r) 观测/可加/F3 ｜ K(set2f_r) 观测/可加/F3")
win = 0
errA, errF = [], []
for a in NODES:
    rs = [r for r in ROWS if r["a"] == a]
    tr = [r for r in rs if r["p"] not in ("set2t_r", "set2f_r")]
    te = [r for r in rs if r["p"] in ("set2t_r", "set2f_r")]
    dtr, dte = tens(tr), tens(te)
    pa = torch.zeros(7, requires_grad=True)
    with torch.no_grad():
        pa[0] = 3.0
    opt = torch.optim.LBFGS([pa], max_iter=300, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.nll_loss(add(pa, dtr).clamp(min=1e-6).log(), dtr["y"]) + 1e-3 * (pa ** 2).mean()
        l.backward()
        return l
    opt.step(cl)
    pa = pa.detach()
    la = torch.nn.functional.nll_loss(add(pa, dte).clamp(min=1e-6).log(), dte["y"]).item()
    # 对照用留一半题的 F3 / F2（它们见过 set_r）
    A, B = [r for r in rs if FM.half(r["id"]) == 0], [r for r in rs if FM.half(r["id"]) == 1]
    def held(fn, n):
        out = []
        for u, v in ((A, B), (B, A)):
            p = FM.fit(fn, n, FM.tensors(u))
            vv = [r for r in v if r["p"] in ("set2t_r", "set2f_r")]
            out.append(FM.nll(fn, p, FM.tensors(vv)))
        return sum(out) / 2
    lf, l2 = held(FM.f3, 8), held(FM.f2, 6)
    qa = add(pa, dte)
    pf3 = FM.fit(FM.f3, 8, FM.tensors(rs))
    qf = FM.f3(pf3, FM.tensors(te))
    def cell(q, p_, st_):
        idx = [i for i, r in enumerate(te) if r["p"] == p_ and r["right"] == st_]
        return (sum(q[i, te[i]["g"]].item() for i in idx) / max(len(idx), 1), sum(te[i]["ok"] for i in idx) / max(len(idx), 1), len(idx))
    bA, bO, nb = cell(qa, "set2t_r", False)
    kA, kO, nk = cell(qa, "set2f_r", True)
    bF, _, _ = cell(qf, "set2t_r", False)
    kF, _, _ = cell(qf, "set2f_r", True)
    win += la <= min(lf, l2)
    errA += [abs(bA - bO), abs(kA - kO)]
    errF += [abs(bF - bO), abs(kF - kO)]
    print(f"{a:14s}{la:22.3f}{lf:12.3f}{l2:14.3f} ｜ {bO:4.0%} {bA:4.0%} {bF:4.0%}（n={nb}）｜ {kO:4.0%} {kA:4.0%} {kF:4.0%}（n={nk}）")
print(f"\n可加版（完全没见过 set_r）log-loss 不差于见过的 F3 和 F2：{win}/{len(NODES)} 节点")
print(f"K、B 平均绝对误差：可加版（留出）{sum(errA) / len(errA):.3f}；F3（全量拟合）{sum(errF) / len(errF):.3f}")
