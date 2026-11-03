"""通用性检验（2026-11-03，库伯：每个节点 8 个自由参数的多项 logit 几乎能拟合任何分类器，拟合好不代表通用规律；
要撑住主张需要两个对照：参数数量相同的通用替代模型，以及把节点或题目打乱后的零假设；然后做预测新节点）。

运行前提交。这些结果决定主张是否成立，失败就报失败，不改公式补救。

一、对照（现有数据，15 个节点 = 第四阶段 13 个有效节点 + 第二阶段没进拟合的 ds_flash_b、codex；对半交叉留出 log-loss）
  C1 同参数随机形式：从 41 个特征池（是否原答案、是否目标、× 形式、× 真假、× 任务、× 置信度、选项偏置等）固定保留"是否原答案"，再随机抽 7 个，共 8 个参数；
     同一形式用于所有节点，抽 200 种。F3 的平均留出 log-loss 要好于其中 ≥ 95%
  C2 同参数数据驱动：每个节点在训练折上用梯度前向选择 8 个特征（对手自己挑形式）。F3 平均留出 log-loss 不差于它超过 0.01
  只记录：几乎饱和的分层频率表（投放 × 第一次答案和目标的关系），作为可解释部分的上限
  C3 打乱题目：节点 × 投放内打乱"投放指向哪个选项"，保留答案和形式的边际分布，重拟合 F3，50 次。真实留出 log-loss 要低于全部 50 次
  C4 打乱节点：同题同投放内打乱节点标签 30 次。"逐节点拟合比全体共用一组参数好多少"的真实值要高于全部 30 次
  C5 辨真的零假设：每个节点用 v = 0 的 F3 模拟答案 100 次再拟合。
     a. 观测 v × 准确率 Spearman（0.93）要高于模拟分布的 95%；b. 只看 LLM 节点（去掉 lex lex_proj emb rand_proj）同样要求；
     c. 纯规则节点 rand_proj 真实辨真力就是 0，它的观测 v 要落在自己模拟分布的 2.5–97.5% 内（估计量校准；不过就说明 v 有几何偏差，v 的数值不可解读）

二、留一节点预测（现有数据，不花钱；一次留出一个节点）
  新节点只有 5 个自由参数；它的题分两半：校准 = 每类 10 题共 40 题，只用 blank L3t L3f set2t set2f；预测 = 其余题上它从没见过的投放 L0t L2t L2f L4t L4f set2t_r set2f_r。
  F4（要检验的规律）：s(y) = w·[y = x] + g_内容·a_层级·(1 − v·假)·[y ∈ T]（L2、L3、L4）；集合 = g_集合·(1 − v·假)·[y ∈ T]；集合 + 理由 = (g_集合·[y ∈ 集合] + g_内容·[y = 理由目标])·(1 − v_r·假)。
     节点参数 w0 w1 g_内容 g_集合 v；a_L2、a_L4、v_r = c0 + c1·v 三者的 4 个常数全部只由其他节点定。新节点用 N(μ, Σ) 先验做 MAP（μ、Σ 只由其他节点定）。
  对手：
    PCA5 = 同样 5 个自由参数的通用模型：其他节点各拟合 14 维 logit（每种投放一个系数），新节点系数 = 均值 + 前 5 主成分 × z，同样 MAP
    共用 = 其他节点的 μ，不用新节点任何数据；最近邻 = 校准 log-loss 最好的那个其他节点的完整 F3
    错校准 = F4 但用另一个节点的校准数据（节点信息的零假设；随机 5 次取平均）
  M1 F4 的预测 log-loss 低于 PCA5、共用、最近邻、错校准，每一项都要 ≥ 2/3 的节点赢，且全节点平均差的区间（按题 bootstrap）不含 0
  M2 四个关键格 K(L4f) B(L4t) K(set2f_r) B(set2t_r)（n ≥ 10）平均绝对误差 ≤ 0.12，且低于共用和最近邻
  M3 假理由暴露：K(set2f_r) 观测 − 可加预测（v_r = v）在节点间和校准 v 的 Spearman ≥ 0.5（辨真越强的节点，假理由暴露得越厉害）

三、真正的新节点（C1 或 C2、M1、M2 都过才做；否则不花钱，直接报告）
  节点：Sonnet、Kimi、Grok、本地 qwen2.5:14b、qwen3.5:9b、遮 65% 题面的 Flash，以及没见过的规则节点 kw_rule（选含投放关键词最多的选项；没有就不动）。全部禁用联网。
  题 = 第二阶段筛选后剩下的 + 没用过的（没进过任何拟合）；同样 40 道校准，其余预测。判据同 M1、M2、M3，运行前再提交一次题和节点名单，判据不改。

用法：~/gut-external/torch215-venv/bin/python generality.py c1 | c2 | c3 | c4 | c5 | lono
"""
import hashlib
import json
import random
import sys

import torch

import analyze as AN
import formula as FM
import stage2 as S2

items = FM.items
PAY12 = {"blank", "L0t", "L2t", "L2f", "L3t", "L3f", "L4t", "L4f", "set2t", "set2f", "set2t_r", "set2f_r"}
RAW2 = [r for r in json.load(open("rows_s2.json")) if r["a"] in ("ds_flash_b", "codex")]
ROWS = [r for r in FM.ROWS + RAW2 if r["p"] in PAY12]
NODES = FM.NODES + ["ds_flash_b", "codex"]
RULES = {"lex", "lex_proj", "emb", "rand_proj"}
PL = ["L2t", "L2f", "L3t", "L3f", "L4t", "L4f", "set2t", "set2f", "set2t_r", "set2f_r"]
TASKS = ("recall", "reason", "evidence", "case")
BY = {a: [r for r in ROWS if r["a"] == a] for a in NODES}
half = FM.half


def tens(rows):
    d = FM.tensors(rows)
    n = len(rows)
    R = torch.zeros(n, 4)
    is_r = torch.zeros(n)
    task = torch.zeros(n, dtype=torch.long)
    pid = torch.zeros(n, dtype=torch.long)
    for i, r in enumerate(rows):
        it = items[r["id"]]
        task[i] = TASKS.index(r["task"])
        pid[i] = PL.index(r["p"]) if r["p"] in PL else -1
        if r["p"] in ("set2t_r", "set2f_r"):
            R[i, it["gold"] if r["p"] == "set2t_r" else S2.wrong(it)[0]] = 1
            is_r[i] = 1
    d.update(R=R, is_r=is_r, task=task, pid=pid)
    return d


def pool(d):
    """特征池 Φ[n, 4, 41]。"""
    X, T, F, fl, has, c = d["X"], d["T"], d["F"], d["fl"], d["has"], d["c"]
    n = len(F)
    fs = [X, X * c[:, None]]                                               # 0 是否原答案；1 × 置信度
    for i in range(5):
        on = ((F == i).float() * has)[:, None]
        fs += [T * on, T * on * fl[:, None], X * on, X * on * fl[:, None]]  # 2–21
    fs += [T, T * fl[:, None], X * fl[:, None], X * has[:, None], T * c[:, None]]  # 22–26
    inside = (X * T).sum(1, keepdim=True)
    fs += [X * inside, T * inside]                                          # 27–28
    for k in range(4):
        e = torch.zeros(n, 4)
        e[:, k] = 1
        fs.append(e)                                                        # 29–32
    for k in range(4):
        on = (d["task"] == k).float()[:, None]
        fs += [X * on, T * on]                                              # 33–40
    return torch.stack(fs, 2)


def fit_lin(P, y, l2=1e-3):
    b = torch.zeros(P.shape[2], requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=100, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(P @ b, y) + l2 * (b ** 2).mean()
        l.backward()
        return l
    opt.step(cl)
    return b.detach()


def nll_lin(P, y, b):
    return torch.nn.functional.cross_entropy(P @ b, y).item()


def folds(a, rows=None):
    rs = rows if rows is not None else BY[a]
    return tens([r for r in rs if half(r["id"]) == 0]), tens([r for r in rs if half(r["id"]) == 1])


def f3_score():
    s = {}
    for a in NODES:
        A, B = folds(a)
        s[a] = (FM.nll(FM.f3, FM.fit(FM.f3, 8, A), B) + FM.nll(FM.f3, FM.fit(FM.f3, 8, B), A)) / 2
    return s


def c1():
    torch.set_num_threads(4)
    base = f3_score()
    F = {a: folds(a) for a in NODES}
    PF = {a: (pool(F[a][0]), pool(F[a][1])) for a in NODES}
    g = random.Random(11)
    scores, top = [], []
    for k in range(200):
        idx = [0] + g.sample(range(1, 41), 7)
        tot = 0
        for a in NODES:
            PA, PB = PF[a][0][:, :, idx], PF[a][1][:, :, idx]
            tot += (nll_lin(PB, F[a][1]["y"], fit_lin(PA, F[a][0]["y"])) + nll_lin(PA, F[a][0]["y"], fit_lin(PB, F[a][1]["y"]))) / 2
        scores.append(tot / len(NODES))
        top.append(idx)
        if k % 40 == 39:
            print(k + 1, f"目前最好 {min(scores):.4f}", flush=True)
    obs = sum(base.values()) / len(NODES)
    better = sum(obs < s for s in scores) / len(scores)
    print(f"C1：F3 平均留出 {obs:.4f}；200 种同参数随机形式 最好 {min(scores):.4f} 中位 {sorted(scores)[100]:.4f}；F3 好于 {better:.1%}（≥ 95%）{S2.mark(better >= .95)}")
    print("  最好的随机形式特征", top[scores.index(min(scores))])


def forward(P, y, k=8):
    """梯度前向选择：每步加梯度最大的特征后重拟合。"""
    sel = [0]
    while len(sel) < k:
        b = fit_lin(P[:, :, sel], y)
        full = torch.zeros(P.shape[2], requires_grad=True)
        with torch.no_grad():
            full[sel] = b
        l = torch.nn.functional.cross_entropy(P @ full, y)
        l.backward()
        gr = full.grad.abs()
        gr[sel] = -1
        sel.append(int(gr.argmax()))
    return sel


def table_nll(tr, te):
    """几乎饱和的分层频率表（投放 × 第一次答案是否在目标内，选项相对分类：原答案 / 目标 / 其他），上限参照。"""
    def key(r, y):
        T = FM.target(items[r["id"]], r["p"])
        return r["p"], r["x"] in T, ("x" if y == r["x"] else "") + ("t" if y in T else "")
    cnt, tot = {}, {}
    for r in tr:
        k = key(r, r["y"])
        cnt[k] = cnt.get(k, 0) + 1
    import math
    out = []
    for r in te:
        T = FM.target(items[r["id"]], r["p"])
        ws = [cnt.get(key(r, y), 0) + .25 for y in range(4)]
        size = {}
        for y in range(4):
            size[key(r, y)] = size.get(key(r, y), 0) + 1
        ws = [w / size[key(r, y)] for y, w in enumerate(ws)]
        out.append(-math.log(ws[r["y"]] / sum(ws)))
    return sum(out) / len(out)


def c2():
    torch.set_num_threads(4)
    base = f3_score()
    print(f"{'节点':14s}{'F3':>8s}{'前向选 8':>10s}{'饱和表':>9s}{'F2':>8s}")
    tot, tt = 0, 0
    wins = 0
    for a in NODES:
        A, B = folds(a)
        PA, PB = pool(A), pool(B)
        sa, sb = forward(PA, A["y"]), forward(PB, B["y"])
        g = (nll_lin(PB[:, :, sa], B["y"], fit_lin(PA[:, :, sa], A["y"])) + nll_lin(PA[:, :, sb], A["y"], fit_lin(PB[:, :, sb], B["y"]))) / 2
        ra, rb = [r for r in BY[a] if half(r["id"]) == 0], [r for r in BY[a] if half(r["id"]) == 1]
        t = (table_nll(ra, rb) + table_nll(rb, ra)) / 2
        f2 = (FM.nll(FM.f2, FM.fit(FM.f2, 6, A), B) + FM.nll(FM.f2, FM.fit(FM.f2, 6, B), A)) / 2
        tot += g
        tt += t
        wins += base[a] <= g
        print(f"{a:14s}{base[a]:8.3f}{g:10.3f}{t:9.3f}{f2:8.3f}", flush=True)
    obs = sum(base.values()) / len(NODES)
    print(f"C2：F3 {obs:.4f} vs 前向选择 {tot / len(NODES):.4f}（差 {obs - tot / len(NODES):+.4f}，F3 不差于 0.01 以内）{S2.mark(obs - tot / len(NODES) <= .01)}；赢 {wins}/{len(NODES)} 节点；饱和表 {tt / len(NODES):.4f}")


def shuffle_targets(d, g):
    e = dict(d)
    T = d["T"].clone()
    pid = d["pid"]
    for k in range(len(PL)):
        idx = (pid == k).nonzero().flatten().tolist()
        p = idx[:]
        g.shuffle(p)
        T[idx] = d["T"][p]
    e["T"] = T
    return e


def c3():
    torch.set_num_threads(4)
    base = f3_score()
    obs = sum(base.values()) / len(NODES)
    F = {a: folds(a) for a in NODES}
    g = random.Random(13)
    null = []
    for k in range(50):
        tot = 0
        for a in NODES:
            A, B = shuffle_targets(F[a][0], g), shuffle_targets(F[a][1], g)
            tot += (FM.nll(FM.f3, FM.fit(FM.f3, 8, A), B) + FM.nll(FM.f3, FM.fit(FM.f3, 8, B), A)) / 2
        null.append(tot / len(NODES))
        if k % 10 == 9:
            print(k + 1, flush=True)
    print(f"C3：真实 {obs:.4f}；打乱题目 50 次 最小 {min(null):.4f} 均值 {sum(null) / 50:.4f}；{S2.mark(obs < min(null))}")


def node_gain(rows):
    by = {a: [r for r in rows if r["a"] == a] for a in NODES}
    per, pooled = 0, 0
    A_all, B_all = tens([r for r in rows if half(r["id"]) == 0]), tens([r for r in rows if half(r["id"]) == 1])
    pa, pb = FM.fit(FM.f3, 8, A_all), FM.fit(FM.f3, 8, B_all)
    for a in NODES:
        A, B = folds(a, by[a])
        per += (FM.nll(FM.f3, FM.fit(FM.f3, 8, A), B) + FM.nll(FM.f3, FM.fit(FM.f3, 8, B), A)) / 2
        pooled += (FM.nll(FM.f3, pb, A) + FM.nll(FM.f3, pa, B)) / 2
    return (pooled - per) / len(NODES)


def c4():
    torch.set_num_threads(4)
    obs = node_gain(ROWS)
    g = random.Random(17)
    groups = {}
    for r in ROWS:
        groups.setdefault((r["id"], r["p"]), []).append(r)
    null = []
    for k in range(30):
        rows = []
        for v in groups.values():
            lab = [r["a"] for r in v]
            g.shuffle(lab)
            rows += [dict(r, a=l) for r, l in zip(v, lab)]
        null.append(node_gain(rows))
        print(k + 1, f"{null[-1]:.4f}", flush=True)
    print(f"C4：逐节点比共用好 {obs:.4f}；打乱节点 30 次 最大 {max(null):.4f} 均值 {sum(null) / 30:.4f}；{S2.mark(obs > max(null))}")


def c5():
    torch.set_num_threads(4)
    acc = {a: AN.mean(r["right"] for r in BY[a] if not r["hard"] and r["p"] == "blank") for a in NODES}
    obs, sim = {}, {a: [] for a in NODES}
    f3nv = lambda p, d: FM.f3(p, d, False)
    for a in NODES:
        d = tens(BY[a])
        obs[a] = FM.fit(FM.f3, 8, d)[7].item()
        p0 = FM.fit(f3nv, 7, d)
        q = f3nv(p0, d)
        torch.manual_seed(hash(a) % 10000 if False else int(hashlib.md5(a.encode()).hexdigest(), 16) % 10000)
        for k in range(100):
            e = dict(d)
            e["y"] = torch.multinomial(q, 1).flatten()
            sim[a].append(FM.fit(FM.f3, 8, e)[7].item())
        s = sorted(sim[a])
        print(f"{a:14s}{acc[a]:6.0%} 观测 v {obs[a]:+.2f}；v=0 模拟 [{s[2]:+.2f},{s[97]:+.2f}] 中位 {s[50]:+.2f}", flush=True)
    for lab, ns in (("全部节点", NODES), ("只看 LLM", [a for a in NODES if a not in RULES])):
        o = S2.spearman([acc[a] for a in ns], [obs[a] for a in ns])
        null = sorted(S2.spearman([acc[a] for a in ns], [sim[a][k] for a in ns]) for k in range(100))
        print(f"C5 {lab}：Spearman {o:.2f}；v = 0 模拟 中位 {null[50]:.2f}，95% 分位 {null[94]:.2f} {S2.mark(o > null[94])}")
    s = sorted(sim["rand_proj"])
    print(f"C5c rand_proj 观测 v {obs['rand_proj']:+.2f} 在 [{s[2]:+.2f},{s[97]:+.2f}] 内 {S2.mark(s[2] <= obs['rand_proj'] <= s[97])}")


# —— 留一节点 ——
CAL_P = {"blank", "L3t", "L3f", "set2t", "set2f"}
TEST_P = {"L0t", "L2t", "L2f", "L4t", "L4f", "set2t_r", "set2f_r"}


def cal_items():
    out = set()
    for t in TASKS:
        ids = sorted({r["id"] for r in ROWS if r["task"] == t}, key=lambda i: hashlib.md5(("cal" + i).encode()).hexdigest())
        out |= set(ids[:10])
    return out


CAL = cal_items()


def f4(t, phi, d, additive=False):
    w = t[0] + t[1] * d["c"]
    gc, gs, v = t[2], t[3], t[4]
    a2, a4, c0, c1 = phi
    level = torch.stack([a2, torch.tensor(1.), a4])
    F, fl, has = d["F"], d["fl"], d["has"]
    fac = 1 - v * fl
    vr = v if additive else c0 + c1 * v
    fac_r = 1 - vr * fl
    content = (F <= 2).float() * has * gc * level[F.clamp(max=2)] * fac
    setf = (F == 3).float() * gs * fac + (F == 4).float() * gs * fac_r
    reason = d["is_r"] * gc * fac_r
    s = w[:, None] * d["X"] + (content + setf)[:, None] * d["T"] + reason[:, None] * d["R"]
    return torch.softmax(s, 1)


def fit_pop(train):
    D = [tens(BY[a]) for a in train]
    Th = torch.zeros(len(train), 5)
    Th[:, 0], Th[:, 2], Th[:, 3] = 3., 2., 4.
    Th.requires_grad_(True)
    phi = torch.tensor([.3, 1.5, 0., 1.], requires_grad=True)
    opt = torch.optim.LBFGS([Th, phi], max_iter=400, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = sum(torch.nn.functional.nll_loss(f4(Th[i], phi, d).clamp(min=1e-6).log(), d["y"]) for i, d in enumerate(D)) / len(D) + 1e-3 * (Th ** 2).mean()
        l.backward()
        return l
    opt.step(cl)
    T = Th.detach()
    mu = T.mean(0)
    Sig = torch.cov(T.T) + .1 * torch.eye(5)
    return mu, Sig, phi.detach()


def map_fit(d, mu, Sig, fn, k):
    Inv = torch.linalg.inv(Sig)
    z = mu.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([z], max_iter=200, line_search_fn="strong_wolfe")
    n = len(d["y"])

    def cl():
        opt.zero_grad()
        dd = z - mu
        l = torch.nn.functional.nll_loss(fn(z, d).clamp(min=1e-6).log(), d["y"]) + .5 * dd @ Inv @ dd / n
        l.backward()
        return l
    opt.step(cl)
    return z.detach()


def basis(d):
    """14 维：是否原答案、× 置信度、每种投放的目标系数（10）、理由目标（2）。"""
    X, T, c = d["X"], d["T"], d["c"]
    fs = [X, X * c[:, None]]
    for k in range(10):
        fs.append(T * (d["pid"] == k).float()[:, None])
    fs.append(d["R"] * (d["pid"] == 8).float()[:, None])
    fs.append(d["R"] * (d["pid"] == 9).float()[:, None])
    return torch.stack(fs, 2)


def split(a, rows=None):
    rs = rows if rows is not None else BY[a]
    return [r for r in rs if r["id"] in CAL and r["p"] in CAL_P], [r for r in rs if r["id"] not in CAL and r["p"] in TEST_P]


CELLS = (("L4f", True), ("L4t", False), ("set2f_r", True), ("set2t_r", False))


def per_item(q, d, rows):
    nl = -q.clamp(min=1e-6).log()[torch.arange(len(rows)), d["y"]]
    return {i: [] for i in set()}, nl


def cell_err(q, rows):
    e = []
    for p, st_ in CELLS:
        idx = [i for i, r in enumerate(rows) if r["p"] == p and r["right"] == st_]
        if len(idx) >= 10:
            e.append(abs(sum(q[i, rows[i]["g"]].item() for i in idx) / len(idx) - sum(rows[i]["ok"] for i in idx) / len(idx)))
    return sum(e) / max(len(e), 1)


def lono():
    torch.set_num_threads(4)
    full3 = {a: FM.fit(FM.f3, 8, tens(BY[a])) for a in NODES}
    res, cells, detail = {}, {}, {}
    expose = {}
    per = {m: {} for m in ("F4", "PCA5", "共用", "最近邻", "错校准")}
    print(f"{'留出节点':14s}{'F4':>8s}{'PCA5':>8s}{'共用':>8s}{'最近邻':>8s}{'错校准':>8s} ｜ 关键格误差 F4 / 共用 / 最近邻")
    for a in NODES:
        train = [n for n in NODES if n != a]
        mu, Sig, phi = fit_pop(train)
        cal, test = split(a)
        dc, dt = tens(cal), tens(test)
        fn = lambda t, d: f4(t, phi, d)
        th = map_fit(dc, mu, Sig, fn, 5)
        qs = {"F4": fn(th, dt), "共用": fn(mu, dt)}
        # PCA5
        B = torch.stack([fit_lin(basis(tens(BY[n])), tens(BY[n])["y"]) for n in train])
        m0 = B.mean(0)
        U, S, Vh = torch.linalg.svd(B - m0, full_matrices=False)
        Vk, sk = Vh[:5], (S[:5] ** 2 / (len(train) - 1) + .1)
        Pc, Pt = basis(dc), basis(dt)
        z = torch.zeros(5, requires_grad=True)
        opt = torch.optim.LBFGS([z], max_iter=200, line_search_fn="strong_wolfe")

        def cl():
            opt.zero_grad()
            l = torch.nn.functional.cross_entropy(Pc @ (m0 + z @ Vk), dc["y"]) + .5 * (z ** 2 / sk).sum() / len(dc["y"])
            l.backward()
            return l
        opt.step(cl)
        qs["PCA5"] = torch.softmax(Pt @ (m0 + z.detach() @ Vk), 1)
        # 最近邻
        best = min(train, key=lambda n: FM.nll(FM.f3, full3[n], dc))
        qs["最近邻"] = FM.f3(full3[best], dt)
        # 错校准
        g = random.Random(a)
        wrong_q = []
        for other in g.sample(train, 5):
            co, _ = split(other)
            wrong_q.append(fn(map_fit(tens(co), mu, Sig, fn, 5), dt))
        qs["错校准"] = sum(wrong_q) / 5
        ids = [r["id"] for r in test]
        for m, q in qs.items():
            per[m][a] = (ids, (-q.clamp(min=1e-6).log()[torch.arange(len(test)), dt["y"]]).detach())
        res[a] = {m: per[m][a][1].mean().item() for m in per}
        cells[a] = {m: cell_err(qs[m], test) for m in ("F4", "共用", "最近邻")}
        qa = f4(th, phi, dt, additive=True)
        idx = [i for i, r in enumerate(test) if r["p"] == "set2f_r" and r["right"]]
        if len(idx) >= 10:
            expose[a] = (sum(test[i]["ok"] for i in idx) / len(idx) - sum(qa[i, test[i]["g"]].item() for i in idx) / len(idx), th[4].item())
        detail[a] = dict(th=th.tolist(), phi=phi.tolist(), near=best)
        print(f"{a:14s}" + "".join(f"{res[a][m]:8.3f}" for m in per) + f" ｜ {cells[a]['F4']:.3f} / {cells[a]['共用']:.3f} / {cells[a]['最近邻']:.3f}  θ {[round(x, 2) for x in th.tolist()]} 近邻 {best}", flush=True)
    print(f"\n常数（最后一折）a_L2 {phi[0]:.2f} a_L4 {phi[1]:.2f} v_r = {phi[2]:.2f} + {phi[3]:.2f}·v")
    n = len(NODES)
    print("M1：")
    allm = True
    for m in ("PCA5", "共用", "最近邻", "错校准"):
        w = sum(res[a]["F4"] < res[a][m] for a in NODES)
        ids = sorted({i for a in NODES for i in per["F4"][a][0]})
        by = {i: [] for i in ids}
        for a in NODES:
            I, l4 = per["F4"][a]
            lm = per[m][a][1]
            for k, i in enumerate(I):
                by[i].append((l4[k] - lm[k]).item())
        g = random.Random(3)
        boots = []
        for _ in range(1000):
            v = [x for i in g.choices(ids, k=len(ids)) for x in by[i]]
            boots.append(sum(v) / len(v))
        boots.sort()
        mean = sum(x for v in by.values() for x in v) / sum(len(v) for v in by.values())
        ok = w * 3 >= 2 * n and boots[974] < 0
        allm = allm and ok
        print(f"  F4 vs {m}：赢 {w}/{n}；平均差 {mean:+.4f} [{boots[25]:+.4f},{boots[974]:+.4f}] {S2.mark(ok)}")
    print(f"  M1 总体 {S2.mark(allm)}")
    ce = {m: sum(cells[a][m] for a in NODES) / n for m in ("F4", "共用", "最近邻")}
    print(f"M2：关键格误差 F4 {ce['F4']:.3f}（≤ 0.12）共用 {ce['共用']:.3f} 最近邻 {ce['最近邻']:.3f} {S2.mark(ce['F4'] <= .12 and ce['F4'] < ce['共用'] and ce['F4'] < ce['最近邻'])}")
    ks = list(expose)
    sp = S2.spearman([expose[a][1] for a in ks], [expose[a][0] for a in ks])
    print(f"M3：暴露量 × 校准 v，Spearman {sp:.2f}（≥ 0.5）{S2.mark(sp >= .5)}；" + " ".join(f"{a} {expose[a][0]:+.2f}/{expose[a][1]:+.2f}" for a in ks))
    json.dump(dict(res=res, cells=cells, detail=detail), open("lono.json", "w"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "lono": lono}[sys.argv[1]]()
