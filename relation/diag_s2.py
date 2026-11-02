"""第二阶段事后诊断（未预注册，只用来解释和提出下一轮假说）。"""
import json
import random

import analyze as AN
import stage2 as S

R = json.load(open("rows_s2.json"))
items = {x["id"]: x for x in json.load(open(S.ITEMS2))}
m = AN.mean
TG = lambda it, p: (lambda w, g: {"set2t": {g, w[0]}, "set2t_r": {g, w[0]}, "set2f": set(w[:2]), "set2f_r": set(w[:2]), "L3t": {g}, "L4t": {g}, "L3f": {w[0]}, "L4f": {w[0]}}[p])(S.wrong(it), it["gold"])
blank = {(r["id"], r["a"]): r for r in R if r["p"] == "blank"}


def dadj(rs, pt, pf):
    """错节点、冲突：采纳率减去同一批节点在 blank 下本来就落进该目标的比例（回归均值 / 本来就会改）。"""
    v = []
    for p in (pt, pf):
        X = [r for r in rs if r["p"] == p and r["conflict"] and not r["right"] and (r["id"], r["a"]) in blank]
        v.append(m(r["adopt"] for r in X) - m(blank[r["id"], r["a"]]["y"] in TG(items[r["id"]], p) for r in X))
    return v[0] - v[1], v


print("一、分辨力减去 blank 基线（区间按题 bootstrap）")
for nm, pt, pf in (("集合", "set2t", "set2f"), ("集合+理由", "set2t_r", "set2f_r"), ("L3 前提", "L3t", "L3f"), ("L4 结论", "L4t", "L4f")):
    for lab, X in (("全部", R), ("自然", [r for r in R if not r["hard"]]), ("困难", [r for r in R if r["hard"]])):
        W = [r for r in X if r["p"] in (pt, pf) and r["conflict"] and not r["right"]]
        raw = S.D(W, pt, pf)
        d, (ut, uf) = dadj(X, pt, pf)
        lo, hi = AN.boot(W, lambda y: dadj(y, pt, pf)[0], 300)
        print(f"  {nm:8s}{lab}: 原始 D {raw:+.2f} ｜ 减基线后 真 {ut:+.2f} 假 {uf:+.2f} D {d:+.2f} [{lo:+.2f},{hi:+.2f}] (n={len(W)})")

print("\n二、盈亏点 v*：各形式（net_f < 0 且 net_t > 0 才有定义）")


def vs(y, pt, pf):
    b = m(r["ok"] for r in y if r["p"] == "blank")
    t, f = m(r["ok"] for r in y if r["p"] == pt) - b, m(r["ok"] for r in y if r["p"] == pf) - b
    return 1.0 if t <= 0 else max(min(-f / (t - f), 1.0), 0.0), t, f


for lab, X in (("自然 ε≈9%", [r for r in R if not r["hard"]]), ("困难 ε≈54%", [r for r in R if r["hard"]])):
    for nm, pt, pf in (("集合", "set2t", "set2f"), ("集合+理由", "set2t_r", "set2f_r"), ("L3 前提", "L3t", "L3f"), ("L4 结论", "L4t", "L4f")):
        v, t, f = vs(X, pt, pf)
        lo, hi = AN.boot([r for r in X if r["p"] in ("blank", pt, pf)], lambda y: vs(y, pt, pf)[0], 300)
        print(f"  {lab} {nm:8s} net真 {t:+.3f} net假 {f:+.3f} v* {v:.2f} [{lo:.2f},{hi:.2f}]")

print("\n三、L3（前提）vs L4（结论）vs 集合：错节点救回 B，对节点保住 K（自然 + 困难，按接收者）")
print(f"{'':14s}{'n错':>5s}" + "".join(f"{p:>9s}" for p in ("blank", "L3t", "L4t", "set2t", "set2t_r")) + " ｜" + "".join(f"{p:>9s}" for p in ("L3f", "L4f", "set2f", "set2f_r")))
for a in S.AGENTS:
    X = [r for r in R if r["a"] == a]
    print(f"{a:14s}{sum(not r['right'] for r in X if r['p'] == 'blank'):5d}" + "".join(f"{S.B(X, p):9.0%}" for p in ("blank", "L3t", "L4t", "set2t", "set2t_r")) + " ｜" + "".join(f"{S.K(X, p):9.0%}" for p in ("L3f", "L4f", "set2f", "set2f_r")))
for t in S.TASKS:
    X = [r for r in R if r["task"] == t]
    print(f"{t:14s}{sum(not r['right'] for r in X if r['p'] == 'blank'):5d}" + "".join(f"{S.B(X, p):9.0%}" for p in ("blank", "L3t", "L4t", "set2t", "set2t_r")) + " ｜" + "".join(f"{S.K(X, p):9.0%}" for p in ("L3f", "L4f", "set2f", "set2f_r")))

print("\n四、可核对性评分是否在同一层内有用（Haiku 不知真值）")
rt = {(r["id"], r["p"]): r for r in json.load(open(S.RATE)) if r["a"] == "claude_haiku"}
for p, st_, lab in (("L3t", False, "B"), ("L4t", False, "B"), ("L3f", True, "K"), ("L4f", True, "K")):
    s = f"  {p} {lab}:"
    for v in range(3):
        X = [r for r in R if r["p"] == p and r["right"] == st_ and rt.get((r["id"], p), {}).get("checkable") == v]
        s += f"  核对{v} {m(r['ok'] for r in X):4.0%}（{len(X)}）"
    print(s)

print("\n五、错节点第一次的把握：高把握错节点能不能被真前提救回（L3t / L4t / set2t_r）")
for p in ("L3t", "L4t", "set2t_r"):
    s = f"  {p:8s}"
    for lo, hi in ((0, 70), (70, 90), (90, 101)):
        X = [r for r in R if r["p"] == p and not r["right"] and lo <= r["cf"] < hi]
        s += f"  [{lo},{hi}) {m(r['ok'] for r in X):4.0%}（{len(X)}）"
    print(s)
