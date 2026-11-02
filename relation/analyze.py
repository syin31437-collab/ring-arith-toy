"""第一阶段观测的统计（M1–M4、M6）。探索用，不是检验。用法：python analyze.py [目录]"""
import glob
import json
import math
import os
import random
import statistics as st
import sys

import torch

import run as RUN

LET, A, PAYLOADS = RUN.LET, RUN.AGENTS, RUN.PAYLOADS
TASKS = ("recall", "reason", "evidence")
NS = ("set2t", "set3t", "set2f", "set3f")
TRUE_P, FALSE_P = ("set2t", "set3t", "fact_t"), ("set2f", "set3f", "fact_f")


def sets(it):
    rng = random.Random(f"set-{it['id']}")
    w = [i for i in range(4) if i != it["gold"]]
    rng.shuffle(w)
    return {"set2t": {it["gold"], w[0]}, "set3t": {it["gold"], w[0], w[1]}, "set2f": set(w[:2]), "set3f": set(w)}


def letter(v):
    s = str(v.get("answer", "")).strip().upper()[:1]
    return LET.index(s) if s and s in LET else None


def load(out):
    items = {x["id"]: x for x in json.load(open(RUN.ITEMS))}
    pay = json.load(open(RUN.PAY))
    R, miss, tot = [], 0, 0
    first = {}
    for fn in glob.glob(os.path.join(out, "s1_*.json")):
        a = next(x for x in sorted(A, key=len, reverse=True) if fn.endswith("_" + x + ".json"))
        J = json.load(open(fn))
        for q, v in J["answers"].items():
            first[q, a] = (letter(v), v.get("confidence"), J.get("searches"))
    for fn in glob.glob(os.path.join(out, "*.json")):
        b = os.path.basename(fn)[:-5]
        if b.startswith("s1_"):
            continue
        a = next(x for x in sorted(A, key=len, reverse=True) if b.endswith("_" + x))
        p = next(x for x in sorted(PAYLOADS, key=len, reverse=True) if b.startswith(x + "_"))
        J = json.load(open(fn))
        for q, v in J["answers"].items():
            if q not in items:
                continue
            tot += 1
            it = items[q]
            x, cf, s0 = first.get((q, a), (None, None, None))
            y = letter(v)
            if x is None or y is None:
                miss += 1
                continue
            try:
                cf = float(cf)
            except (TypeError, ValueError):
                cf = 50.0
            S = sets(it)
            if p in S:
                target, conflict = S[p], x not in S[p]
            elif p == "fact_t":
                target, conflict = {it["gold"]}, x != it["gold"]
            elif p == "fact_f":
                target, conflict = {LET.index(pay[q]["false_target"])}, x != LET.index(pay[q]["false_target"])
            else:
                target, conflict = None, None
            R.append(dict(id=q, task=it["task"], source=it["source"], a=a, p=p, x=x, y=y, g=it["gold"], cf=cf, right=x == it["gold"], ok=y == it["gold"], changed=y != x, conflict=conflict, adopt=(y in target) if target else None, searches=J.get("searches") or 0))
    return R, miss, tot, items


def mean(v):
    v = list(v)
    return st.mean(v) if v else float("nan")


def boot(rs, fn, n=300):
    ids = sorted({r["id"] for r in rs})
    by = {i: [r for r in rs if r["id"] == i] for i in ids}
    g = random.Random(0)
    out = []
    for _ in range(n):
        v = fn([r for i in g.choices(ids, k=len(ids)) for r in by[i]])
        if v == v:
            out.append(v)
    out.sort()
    return (out[int(.025 * len(out))], out[int(.975 * len(out))]) if out else (float("nan"), float("nan"))


def z(p, n):
    p = (p * n + .5) / (n + 1)
    return torch.distributions.Normal(0, 1).icdf(torch.tensor(p)).item()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else RUN.OUT
    R, miss, tot, items = load(out)
    print(f"记录 {len(R)}，缺失 {miss}（{miss / max(tot, 1):.1%}）")
    first = {(r["id"], r["a"]): r for r in R if r["p"] == "blank"}
    print("\n一、第一次准确率（任务 × 接收者）")
    print("          " + " ".join(f"{a[:10]:>10s}" for a in A) + "   错误相关")
    for t in TASKS:
        s = [f"{mean(r['right'] for r in first.values() if r['task'] == t and r['a'] == a):10.0%}" for a in A]
        ids = sorted({r["id"] for r in first.values() if r["task"] == t})
        M = torch.tensor([[float(not first[i, a]["right"]) if (i, a) in first else 0. for a in A] for i in ids])
        C = torch.corrcoef(M.T)
        C = C[~torch.isnan(C).any(1)][:, ~torch.isnan(C).any(0)]
        print(f"{t:10s}" + " ".join(s) + f"   {(C.sum() - len(C)) / max(len(C) * (len(C) - 1), 1):.2f}")
    print("\n二、好节点保住 K ／ 错节点回来 B ／ 净变化 net（全部接收者；net 已减 blank）")
    print(f"{'投放':10s}" + "".join(f"{t + ' K':>11s}{'B':>6s}{'net':>7s}" for t in TASKS) + f"{'全部 K':>10s}{'B':>6s}{'net':>7s}")
    base = {t: mean(r["ok"] for r in R if r["p"] == "blank" and r["task"] == t) for t in TASKS}
    base["all"] = mean(r["ok"] for r in R if r["p"] == "blank")
    for p in PAYLOADS:
        s = f"{p:10s}"
        for t in TASKS + ("all",):
            X = [r for r in R if r["p"] == p and (t == "all" or r["task"] == t)]
            K, B = mean(r["ok"] for r in X if r["right"]), mean(r["ok"] for r in X if not r["right"])
            n = mean(r["ok"] for r in X) - base[t]
            s += f"{K:11.0%}{B:6.0%}{n:+7.2f}" if t != "all" else f"{K:10.0%}{B:6.0%}{n:+7.2f}"
        print(s)
    print("\n三、集合投放 vs 理想投影（参照线）")
    for p in NS:
        X = [r for r in R if r["p"] == p]
        size = 2 if "2" in p else 3
        tr = p.endswith("t")
        kr = 1.0 if tr else 0.0
        out_w = [r for r in X if not r["right"] and r["conflict"]]
        in_w = [r for r in X if not r["right"] and not r["conflict"]]
        print(f"  {p}: K 观测 {mean(r['ok'] for r in X if r['right']):.0%}（投影 {kr:.0%}）；错节点在集合外→对 {mean(r['ok'] for r in out_w):.0%}（投影 {(1 / size if tr else 0):.0%}，n={len(out_w)}）；错节点在集合内→对 {mean(r['ok'] for r in in_w):.0%}（投影 0%，n={len(in_w)}）；集合外→进集合 {mean(r['adopt'] for r in X if r['conflict']):.0%}")
    print("\n四、分辨力：冲突时的采纳（只比错节点，状态相同）")
    print("  任务 × 形式：a_真 ／ a_假（错节点）｜ D ［区间］｜ d′ ｜ 对节点被假投放带走")
    for form, pt, pf in (("集合2", "set2t", "set2f"), ("集合3", "set3t", "set3f"), ("事实", "fact_t", "fact_f")):
        for t in TASKS + ("all",):
            X = [r for r in R if (t == "all" or r["task"] == t) and r["p"] in (pt, pf) and r["conflict"]]
            W = [r for r in X if not r["right"]]
            T, F = [r for r in W if r["p"] == pt], [r for r in W if r["p"] == pf]
            at, af = mean(r["adopt"] for r in T), mean(r["adopt"] for r in F)
            D = lambda y: mean(r["adopt"] for r in y if r["p"] == pt) - mean(r["adopt"] for r in y if r["p"] == pf)
            lo, hi = boot(W, D)
            dp = z(at, len(T)) - z(af, len(F)) if T and F else float("nan")
            rr = mean(r["adopt"] for r in X if r["right"] and r["p"] == pf)
            print(f"  {form} {t:9s} {at:4.0%} ／ {af:4.0%}（n={len(T)}/{len(F)}）｜ {at - af:+.2f} [{lo:+.2f},{hi:+.2f}] ｜ {dp:+.2f} ｜ {rr:4.0%}")
    print("\n五、接收者：好节点被假集合 set2f 带走、错节点被 set2t 救回、假事实带走、真事实救回")
    for a in A:
        X = [r for r in R if r["a"] == a]
        f = lambda p, st_: mean(r["ok"] for r in X if r["p"] == p and r["right"] == st_)
        print(f"  {a:13s} set2f K {f('set2f', True):4.0%} ｜ set2t B {f('set2t', False):4.0%} ｜ fact_f K {f('fact_f', True):4.0%} ｜ fact_t B {f('fact_t', False):4.0%} ｜ blank K {f('blank', True):4.0%} B {f('blank', False):4.0%} ｜ recheck K {f('recheck', True):4.0%} B {f('recheck', False):4.0%} ｜ method_m B {f('method_m', False):4.0%} ｜ 平均搜索 {mean(r['searches'] for r in X):.2f}")
    print("\n六、置信度：第一次置信度分箱的改答率（错节点 / 对节点）")
    for p in ("set2t", "set2f", "fact_t", "fact_f", "recheck", "blank"):
        s = f"  {p:8s}"
        for lo, hi in ((0, 70), (70, 90), (90, 101)):
            X = [r for r in R if r["p"] == p and lo <= r["cf"] < hi]
            s += f" [{lo}-{hi}) 错 {mean(r['changed'] for r in X if not r['right']):4.0%}（{sum(not r['right'] for r in X)}）对 {mean(r['changed'] for r in X if r['right']):4.0%}（{sum(r['right'] for r in X)}）"
        print(s)
    print("\n七、Codex 联网对比（记忆题最相关）")
    for t in TASKS:
        for a in ("codex", "codex_ns"):
            X = [r for r in R if r["a"] == a and r["task"] == t]
            f = lambda p, st_: mean(r["ok"] for r in X if r["p"] == p and r["right"] == st_)
            print(f"  {t:9s}{a:9s} 第一次 {mean(r['right'] for r in X if r['p'] == 'blank'):4.0%} ｜ set2f K {f('set2f', True):4.0%} fact_f K {f('fact_f', True):4.0%} ｜ set2t B {f('set2t', False):4.0%} fact_t B {f('fact_t', False):4.0%} ｜ method_m B {f('method_m', False):4.0%} method_x B {f('method_x', False):4.0%} ｜ 平均搜索 {mean(r['searches'] for r in X):.2f}")
    print("\n八、方法和事实是否相关（对口 vs 不相关）")
    for p, q in (("method_m", "method_x"), ("fact_t", "fact_x")):
        for t in TASKS:
            X = [r for r in R if r["task"] == t]
            f = lambda pp, st_: mean(r["ok"] for r in X if r["p"] == pp and r["right"] == st_)
            print(f"  {t:9s} {p} K {f(p, True):4.0%} B {f(p, False):4.0%} ｜ {q} K {f(q, True):4.0%} B {f(q, False):4.0%} ｜ blank K {f('blank', True):4.0%} B {f('blank', False):4.0%}")
    json.dump(R, open(os.path.join(RUN.HERE, "rows_s1.json"), "w"))
    predict(R)


def feats(r, level):
    tr = {"set2t": 1, "set3t": 1, "fact_t": 1, "set2f": -1, "set3f": -1, "fact_f": -1}.get(r["p"], 0)
    bits = {"set2t": 1, "set2f": 1, "set3t": .415, "set3f": .415}.get(r["p"], 0)
    form = [r["p"].startswith(k) for k in ("set", "fact_t", "fact_f", "method_m", "method_x", "fact_x", "recheck")]
    right = 2 * r["right"] - 1
    c = (r["cf"] - 75) / 25
    con = 1 if r["conflict"] else 0
    f = [1.0, right, c, right * c]
    if level >= 1:
        f += [float(x) for x in form] + [bits, con]
    if level >= 2:
        f += [tr, tr * right, tr * con, con * right, con * c, bits * con, tr * c, bits * tr]
    return f


def auc(y, s):
    pos = [v for v, q in zip(s, y) if q]
    neg = [v for v, q in zip(s, y) if not q]
    if not pos or not neg:
        return float("nan")
    t = torch.tensor(neg).sort().values
    p = torch.tensor(pos)
    lt = torch.searchsorted(t, p, right=False).float()
    le = torch.searchsorted(t, p, right=True).float()
    return ((lt + le) / 2).mean().item() / len(neg)


def fit(X, y):
    X, y = torch.tensor(X), torch.tensor(y, dtype=torch.float)
    w = torch.zeros(X.shape[1], requires_grad=True)
    o = torch.optim.LBFGS([w], max_iter=200)

    def cl():
        o.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y) + 1e-3 * (w ** 2).sum()
        l.backward()
        return l
    o.step(cl)
    return w.detach()


def predict(R):
    print("\n九、留一预测（只用 投放 × 接收者状态，不用任务 / 模型名）：第②步答对的 AUC")
    print("  层 0 = 第一次对错 + 置信度；层 1 = 加投放形式、信息量、冲突；层 2 = 加 真假 × 状态 / 冲突 / 置信度 的交互")
    for key, vals in (("task", TASKS), ("a", A)):
        for v in vals:
            s = f"  留出 {v:13s}"
            for lvl in (0, 1, 2):
                tr = [r for r in R if r[key] != v]
                te = [r for r in R if r[key] == v]
                w = fit([feats(r, lvl) for r in tr], [r["ok"] for r in tr])
                sc = (torch.tensor([feats(r, lvl) for r in te]) @ w).tolist()
                s += f"  层{lvl} {auc([r['ok'] for r in te], sc):.3f}"
            print(s)


if __name__ == "__main__":
    main()
