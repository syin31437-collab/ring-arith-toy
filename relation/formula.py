"""第四阶段的公式比较（规则在 stage4.py 预注册，代码在数据之后写；规则不改）。

Q7 F3（软投影）留出 log-loss 在 ≥ 2/3 的节点上不差于 F0、F1、F2。Q8 v、g_L3 / (g_L3 + g_set) 和第一次准确率的 Spearman ≥ 0.5。
只记录：F3 去掉 v 的版本（看辨真项值不值）；预测的 K / B 和观测的差；数值节点上近端形式的闭式核对。
用法：~/gut-external/torch215-venv/bin/python formula.py
"""
import hashlib
import json
import os
import random

import torch

import analyze as AN
import stage2 as S2
import stage4 as S4

items = {x["id"]: x for x in json.load(open(S2.ITEMS2))}
R4 = json.load(open("rows_s4.json"))
RS = [r for r in json.load(open("rows_s2.json")) if r["a"] in S4.TOP2]
INV = {"tiny_llama", "tiny_smol"}          # 预检：无效输出 > 20%，不计入判据
ROWS = R4 + RS
NODES = [n for n in S4.NODES + S4.TOP2 if n not in INV]
FORM = {"L2t": 0, "L2f": 0, "L3t": 1, "L3f": 1, "L4t": 2, "L4f": 2, "set2t": 3, "set2f": 3, "set2t_r": 4, "set2f_r": 4}
FALSE = {"L2f", "L3f", "L4f", "set2f", "set2f_r"}


def target(it, p):
    w, g = S2.wrong(it), it["gold"]
    return {"L2t": {g}, "L3t": {g}, "L4t": {g}, "set2t": {g, w[0]}, "set2t_r": {g, w[0]}, "L2f": {w[0]}, "L3f": {w[0]}, "L4f": {w[0]}, "set2f": set(w[:2]), "set2f_r": set(w[:2])}.get(p, set())


def tensors(rows):
    n = len(rows)
    X = torch.zeros(n, 4)
    T = torch.zeros(n, 4)
    F = torch.zeros(n, dtype=torch.long)
    fl = torch.zeros(n)
    has = torch.zeros(n)
    c = torch.zeros(n)
    y = torch.zeros(n, dtype=torch.long)
    for i, r in enumerate(rows):
        X[i, r["x"]] = 1
        for t in target(items[r["id"]], r["p"]):
            T[i, t] = 1
        has[i] = r["p"] in FORM
        F[i] = FORM.get(r["p"], 0)
        fl[i] = r["p"] in FALSE
        c[i] = (r["cf"] - 75) / 25
        y[i] = r["y"]
    return dict(X=X, T=T, F=F, fl=fl, has=has, c=c, y=y)


def uni(T):
    return T / T.sum(1, keepdim=True).clamp(min=1)


def f0(p, d):
    e = torch.sigmoid(p[0])
    return (1 - e) * d["X"] + e / 4


def f1(p, d):
    e = torch.sigmoid(p[0])
    isset = (d["has"] > 0) & (d["F"] >= 3)
    inside = (d["X"] * d["T"]).sum(1) > 0
    dist = d["X"].clone()
    mv = isset & ~inside
    dist[mv] = uni(d["T"])[mv]
    return (1 - e) * dist + e / 4


def f2(p, d):
    e = torch.sigmoid(p[0])
    a = torch.sigmoid(p[1:6])[d["F"]] * d["has"]
    dist = a[:, None] * uni(d["T"]) + (1 - a)[:, None] * d["X"]
    return (1 - e) * dist + e / 4


def f3(p, d, use_v=True):
    w = p[0] + p[1] * d["c"]
    g = p[2:7][d["F"]] * d["has"]
    if use_v:
        g = g * (1 - p[7] * d["fl"])
    s = w[:, None] * d["X"] + g[:, None] * d["T"]
    return torch.softmax(s, 1)


MODELS = {"F0": (f0, 1), "F1": (f1, 1), "F2": (f2, 6), "F3": (f3, 8), "F3无v": (lambda p, d: f3(p, d, False), 7)}


def fit(fn, k, d):
    p = torch.zeros(k, requires_grad=True)
    with torch.no_grad():
        if fn is f3 or k == 7:
            p[0] = 3.0
    opt = torch.optim.LBFGS([p], max_iter=300, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        q = fn(p, d).clamp(min=1e-6)
        l = torch.nn.functional.nll_loss(q.log(), d["y"]) + 1e-3 * (p ** 2).mean()
        l.backward()
        return l
    opt.step(cl)
    return p.detach()


def nll(fn, p, d):
    return torch.nn.functional.nll_loss(fn(p, d).clamp(min=1e-6).log(), d["y"]).item()


def half(i):
    return int(hashlib.md5(i.encode()).hexdigest(), 16) % 2


def main():
    print("Q7 留出 log-loss（对半交叉，两折平均）")
    print(f"{'节点':14s}{'准确率':>7s}" + "".join(f"{k:>8s}" for k in MODELS) + "   最优")
    res, acc = {}, {}
    wins = 0
    for a in NODES:
        rs = [r for r in ROWS if r["a"] == a]
        acc[a] = AN.mean(r["right"] for r in rs if not r["hard"] and r["p"] == "blank")
        A, B = tensors([r for r in rs if half(r["id"]) == 0]), tensors([r for r in rs if half(r["id"]) == 1])
        res[a] = {}
        for k, (fn, n) in MODELS.items():
            res[a][k] = (nll(fn, fit(fn, n, A), B) + nll(fn, fit(fn, n, B), A)) / 2
        best = min(res[a], key=res[a].get)
        w = res[a]["F3"] <= min(res[a][k] for k in ("F0", "F1", "F2"))
        wins += w
        print(f"{a:14s}{acc[a]:7.0%}" + "".join(f"{res[a][k]:8.3f}" for k in MODELS) + f"   {best}{'' if w else '  (F3 输)'}")
    print(f"Q7：F3 不差于 F0 F1 F2 的节点 {wins}/{len(NODES)}（≥ 2/3）{S2.mark(wins * 3 >= 2 * len(NODES))}")
    v_better = sum(res[a]["F3"] < res[a]["F3无v"] for a in NODES)
    print(f"只记录：加辨真项 v 后留出变好的节点 {v_better}/{len(NODES)}")
    print("\nQ8 全量拟合 F3 的参数（节点按准确率排序）")
    print(f"{'节点':14s}{'准确率':>7s}{'w':>7s}{'g_L2':>7s}{'g_L3':>7s}{'g_L4':>7s}{'g_set':>7s}{'g_set_r':>8s}{'v':>7s}   内容占比")
    P = {}
    for a in sorted(NODES, key=lambda z: -acc[z]):
        d = tensors([r for r in ROWS if r["a"] == a])
        p = fit(f3, 8, d)
        P[a] = p
        sh = max(p[3].item(), 0) / max(max(p[3].item(), 0) + max(p[5].item(), 0), 1e-6)
        print(f"{a:14s}{acc[a]:7.0%}{p[0].item():7.2f}{p[2].item():7.2f}{p[3].item():7.2f}{p[4].item():7.2f}{p[5].item():7.2f}{p[6].item():8.2f}{p[7].item():7.2f}   {sh:.2f}")
    ac = [acc[a] for a in NODES]
    sv = S2.spearman(ac, [P[a][7].item() for a in NODES])
    sh = [max(P[a][3].item(), 0) / max(max(P[a][3].item(), 0) + max(P[a][5].item(), 0), 1e-6) for a in NODES]
    ss = S2.spearman(ac, sh)
    print(f"Q8：v × 准确率 Spearman {sv:.2f}（≥ 0.5）{S2.mark(sv >= .5)}；内容占比 × 准确率 {ss:.2f}（≥ 0.5）{S2.mark(ss >= .5)}")
    print("\n只记录：F3 预测 vs 观测（K 对节点，B 错节点；全量拟合）")
    errs = []
    for a in NODES:
        rs = [r for r in ROWS if r["a"] == a]
        d = tensors(rs)
        q = f3(P[a], d)
        for p_ in ("set2t", "set2f", "set2t_r", "set2f_r", "L3t", "L3f", "L4t", "L4f"):
            for st_ in (True, False):
                idx = [i for i, r in enumerate(rs) if r["p"] == p_ and r["right"] == st_]
                if len(idx) >= 10:
                    pred = sum(q[i, rs[i]["g"]].item() for i in idx) / len(idx)
                    obs = sum(rs[i]["ok"] for i in idx) / len(idx)
                    errs.append((abs(pred - obs), a, p_, st_, pred, obs, len(idx)))
    print(f"  {len(errs)} 格，平均绝对误差 {sum(e[0] for e in errs) / len(errs):.3f}，最大 {max(e[0] for e in errs):.3f}")
    for e in sorted(errs, reverse=True)[:8]:
        print(f"    误差 {e[0]:.2f}：{e[1]} {e[2]} {'对节点' if e[3] else '错节点'} 预测 {e[4]:.0%} 观测 {e[5]:.0%}（n={e[6]}）")
    print("\n只记录：数值节点上的近端形式  x' = argmin_y (w/2)(y − x)² + (g/2)d(y, C)² = x + κ(P_C(x) − x)，κ = g / (g + w)；C = [e − r, e + r]，真值 0")
    print("  第一版闭式 |e| ≤ r − h + 2h/κ 是错的（核对出 30612 / 100000 不一致）：离集合最近的边界节点只要被拉一点就出界。")
    print("  改正：|x| ≤ h − m 的节点全部保住 ⇔ |e| ≤ r + h + m(1 − κ)/κ（m = 裕量；m = 0 或 κ = 1 退回 h + r）")
    g = random.Random(0)
    bad = badp = 0
    N = 100000
    for _ in range(N):
        h, r, e, w_, g_ = g.uniform(.1, 3), g.uniform(0, 3), g.uniform(-8, 8), g.uniform(.05, 5), g.uniform(.05, 5)
        k = g_ / (g_ + w_)
        m_ = g.uniform(0, h)
        lim = h - m_
        pred = abs(e) <= r + h + m_ * (1 - k) / k
        act = True
        for j in range(201):                                                # 逐点：好节点区间上每一点
            x = -lim + 2 * lim * j / 200
            pc = max(min(x, e + r), e - r)
            act = act and abs(x + k * (pc - x)) <= h + 1e-9
        bad += pred != act
        if g.random() < .02:                                                # 近端形式 = 数值最小化
            x = g.uniform(-8, 8)
            pc = max(min(x, e + r), e - r)
            grid = min((x + (t / 2000) * 24 - 12 for t in range(-2000, 2001)), key=lambda y: w_ / 2 * (y - x) ** 2 + g_ / 2 * max(abs(y - e) - r, 0) ** 2)
            badp += abs(grid - (x + k * (pc - x))) > .02
    print(f"  {N} 个随机配置：闭式和逐点模拟不一致 {bad} 个；近端形式和数值最小化不一致 {badp} 个")


if __name__ == "__main__":
    main()
