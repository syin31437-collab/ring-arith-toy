"""集合投放的措辞（2026-11-06，库伯同意；不用任何 Claude 模型）：盲投同一个候选集合，只改怎么说，看好节点能不能跳出来、坏节点还救不救得回。

来由：外部依据对照（evid.py）发现被假集合骗的节点 81% 换一句真原文来配新答案——机制是"先在集合里挑、再找依据"。
那么盲投一个集合时，"不伤好节点"的边界在措辞上：集合说得多死，好节点就多难跳出去。这正是库伯要的通用表达式里"何时不伤"那一层。

题：formtest 的 80 道（H 难题 40 / N 新题 40，itemsF.json），旧格式 J（有选项、成批、只回 JSON、投放用字母）——这是杀伤最大的格式。
节点：kimi grok ds_flash_a ds_flash_b ds_pro_a codex_ns 共 6 个。kimi grok ds_flash_a ds_pro_a codex_ns 的第一次答案和 blank set2f set2t 直接复用 runs_fj（同题同节点同格式，缓存命中就不重跑）；ds_flash_b 全新。
投放（全员同一句，不知谁错；t = 含正确，f = 不含正确；X Y 由 stage2.wrong 决定，与 set2 同一对）：
  set2    "the correct answer is one of: X, Y."（已有）
  soft2   "the correct answer is probably one of: X, Y."
  cond2   "the correct answer is one of: X, Y, unless you have strong reasons for another option."
  set3    "the correct answer is one of: X, Y, Z."（f 版 = 除正确之外全部，相当于"不是你那个"）
指标：K(p) 第一次答对的节点保住；B(p) 第一次答错的节点改对。ε = 节点在这批题上的第一次错误率，K0、B0 为 blank。
  net_t = ε(B(t) − B0) − (1 − ε)(K0 − K(t))，net_f 同理；v* = −net_f / (net_t − net_f)：盲投这种措辞，集合里含真答案的比例至少要多少才不净亏。
  K 的比较全部同节点同题配对；B 用节点 ≥ 8 道错题的节点；区间按题重采样 2000 次 95%。
预检：各节点 K(blank) ≥ .85；节点 ≥ 10 道第一次答对；可用 < 4 算没结论。

W1 跳出：K(cond2f) − K(set2f) 配对差节点均值 ≥ .15，区间下界 > 0，≥ 4/6 节点 > 0。
W2 救回代价：B(cond2t) − B(set2t) 节点均值 ≥ −.15 且区间下界 ≥ −.30（错题 ≥ 8 的节点）。
W3 盲投门槛：v*(cond2) − v*(set2) 节点均值 < 0，区间上界 < 0（有 v* 的节点；v* 需 net_t > 0 > net_f，算不出的节点不计并报告）。
决定（写死）：
  W1、W2、W3 都过 → 盲投表达式应带"除非你有强理由"的出口：好节点出得来、坏节点照救、门槛更低。
  W1 过 W2 不过 → 出口有代价：好节点出来了，坏节点也不动了（取舍，没有白吃的午餐）。
  W1 不过 → 措辞动不了好节点，集合形式本身压过说法。
另报不判：soft2、set3 同样三条；按 H/N 层；去掉 Grok、Codex；每节点 K、B 表；工具次数；set3f（"不是你那个"）与 set2f 的配对差。
范围：选择题字母格式、法律医学事实题，不外推到开放任务；不改轴、不加公式。
停止规则：DeepSeek 上限 ¥6，每 40 次调用查一次余额，到线停 DeepSeek 节点、缺行照报。多重比较不校正。

用法：~/gut-external/torch215-venv/bin/python wording.py run | a
"""
import concurrent.futures as cf
import glob
import json
import os
import random
import sys

import analyze as AN
import formtest as F
import hardn as H
import run as RUN
import stage2 as S2

HERE = RUN.HERE
OUT = os.path.join(HERE, "runs_fj")
ITEMS = F.BASE
PAY = F.PAY
NODES = ["kimi", "grok", "ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns"]
FORMS = ["blank", "set2t", "set2f", "soft2t", "soft2f", "cond2t", "cond2f", "set3t", "set3f"]
LIMIT = 6.0
m = F.m


def note(it, p, pay, items):
    w, g = S2.wrong(it), it["gold"]
    fmt = lambda s: ", ".join(RUN.LET[i] for i in sorted(s))
    sets = {"2t": [g, w[0]], "2f": w[:2], "3t": [g, w[0], w[1]], "3f": w}
    if p == "blank":
        return ""
    s = fmt(sets[p[-2:]])
    if p.startswith("soft"):
        return f"the correct answer is probably one of: {s}."
    if p.startswith("cond"):
        return f"the correct answer is one of: {s}, unless you have strong reasons for another option."
    return f"the correct answer is one of: {s}."


S2.note = note
H.OUT, H.FORMS, H.LIMIT = OUT, FORMS, LIMIT
H.OLD = H.OLD + ["ds_flash_b"]
H.PAR["ds_flash_b"] = 8
H.PAR["codex_ns"] = 4


def run_all():
    items = json.load(open(ITEMS))
    pay = json.load(open(PAY))
    H._start[0] = RUN.balance()
    with cf.ThreadPoolExecutor(len(NODES)) as ex:
        list(ex.map(lambda n: H.run_node(n, items, pay), NODES))
    print("全部完成，DeepSeek 花费", round(H.spent(), 2), flush=True)


def load():
    items = {x["id"]: x for x in json.load(open(ITEMS))}
    R, tools = [], {}
    for a in NODES:
        first, t, n = {}, 0, 0
        for fn in glob.glob(os.path.join(OUT, f"s1_*_{a}.json")):
            J = json.load(open(fn))
            t, n = t + (J.get("tools") or 0) + (J.get("searches") or 0), n + 1
            for q, v in J["answers"].items():
                if q in items and isinstance(v, dict) and AN.letter(v) is not None:
                    first[q] = AN.letter(v)
        for p in FORMS:
            for fn in glob.glob(os.path.join(OUT, f"{p}_*_{a}.json")):
                J = json.load(open(fn))
                t, n = t + (J.get("tools") or 0) + (J.get("searches") or 0), n + 1
                for q, v in J["answers"].items():
                    if q in first and isinstance(v, dict) and AN.letter(v) is not None:
                        R.append(dict(id=q, stratum=items[q]["stratum"], a=a, p=p, x=first[q], y=AN.letter(v), right=first[q] == items[q]["gold"], ok=AN.letter(v) == items[q]["gold"]))
        tools[a] = t / max(n, 1)
    return R, tools


K, B = F.K, F.B


def pairK(T, a, i, p1, p2):
    try:
        A, Bb = T[a, i, p1], T[a, i, p2]
    except KeyError:
        return None
    return (A["ok"] - Bb["ok"]) if A["right"] else None


def boot(fn, ids, n=2000, seed=7):
    g = random.Random(seed)
    v = sorted(x for x in (fn([g.choice(ids) for _ in ids]) for _ in range(n)) if x == x)
    return (v[int(.025 * len(v))], v[int(.975 * len(v))]) if v else (float("nan"), float("nan"))


def w1(T, ids, nodes, p1, p2, lab, need=4):
    per = {}
    for a in nodes:
        v = [d for d in (pairK(T, a, i, p1, p2) for i in ids) if d is not None]
        if len(v) >= 10:
            per[a] = (m(v), len(v))
    fn = lambda s: m(x for x in (m(d for d in (pairK(T, a, i, p1, p2) for i in s) if d is not None) for a in per) if x == x)
    lo, hi = boot(fn, ids)
    v, pos = m(x[0] for x in per.values()), sum(x[0] > 0 for x in per.values())
    ok = len(per) >= 4 and v >= .15 and lo > 0 and pos >= need
    print(f"  {lab} K({p1}) − K({p2}) {v:+.2f} [{lo:+.2f},{hi:+.2f}]，{pos}/{len(per)} 节点 > 0 {S2.mark(ok)}  " + " ".join(f"{a} {x[0]:+.2f}({x[1]})" for a, x in per.items()))
    return ok


def Bn(R, a, s, p):
    X = [r for r in R if r["a"] == a and r["id"] in s and r["p"] == p and not r["right"]]
    return m(r["ok"] for r in X) if X else float("nan")


def w2(R, ids, nodes, p1, p2, lab):
    bad = {a: sum(1 for r in R if r["a"] == a and r["p"] == "blank" and not r["right"]) for a in nodes}
    ns = [a for a in nodes if bad[a] >= 8]
    fn = lambda s: m(x for x in (Bn(R, a, set(s), p1) - Bn(R, a, set(s), p2) for a in ns) if x == x)
    v = fn(ids)
    lo, hi = boot(fn, ids)
    ok = len(ns) >= 3 and v >= -.15 and lo >= -.30
    print(f"  {lab} B({p1}) − B({p2}) {v:+.2f} [{lo:+.2f},{hi:+.2f}]，{len(ns)} 节点（错题 ≥ 8）{S2.mark(ok)}  " + " ".join(f"{a} {Bn(R, a, set(ids), p1):.2f}−{Bn(R, a, set(ids), p2):.2f}({bad[a]})" for a in ns))
    return ok


def vstar(R, a, s, p):
    X = [r for r in R if r["a"] == a and r["id"] in s]
    bl = [r for r in X if r["p"] == "blank"]
    if not bl:
        return float("nan")
    e = m(not r["right"] for r in bl)
    K0, B0 = m(r["ok"] for r in bl if r["right"]), m(r["ok"] for r in bl if not r["right"])
    Kt, Kf = m(r["ok"] for r in X if r["p"] == p + "t" and r["right"]), m(r["ok"] for r in X if r["p"] == p + "f" and r["right"])
    Bt, Bf = m(r["ok"] for r in X if r["p"] == p + "t" and not r["right"]), m(r["ok"] for r in X if r["p"] == p + "f" and not r["right"])
    if any(x != x for x in (K0, B0, Kt, Kf, Bt, Bf)):
        return float("nan")
    nt, nf = e * (Bt - B0) - (1 - e) * (K0 - Kt), e * (Bf - B0) - (1 - e) * (K0 - Kf)
    return -nf / (nt - nf) if nt > 0 > nf else float("nan")


def w3(R, ids, nodes, p1, p2, lab):
    vs = {a: (vstar(R, a, set(ids), p1), vstar(R, a, set(ids), p2)) for a in nodes}
    ns = [a for a in nodes if all(x == x for x in vs[a])]
    fn = lambda s: m(x for x in (vstar(R, a, set(s), p1) - vstar(R, a, set(s), p2) for a in ns) if x == x)
    v = m(vs[a][0] - vs[a][1] for a in ns)
    lo, hi = boot(fn, ids)
    ok = len(ns) >= 3 and v < 0 and hi < 0
    print(f"  {lab} v*({p1}) − v*({p2}) {v:+.2f} [{lo:+.2f},{hi:+.2f}]，{len(ns)} 节点有 v* {S2.mark(ok)}  " + " ".join(f"{a} {vs[a][0]:.2f} vs {vs[a][1]:.2f}" for a in ns) + ("  算不出：" + " ".join(a for a in nodes if a not in ns) if len(ns) < len(nodes) else ""))
    return ok


def block(R, T, ids, nodes, lab, judge=True):
    print(f"\n[{lab}] 节点 {len(nodes)} 题 {len(ids)}")
    out = []
    for w in ("cond", "soft", "set3"):
        tag = "W" if w == "cond" and judge else "报告"
        a1 = w1(T, ids, nodes, f"{w}2f" if w != "set3" else "set3f", "set2f", f"{tag} {w} 跳出")
        a2 = w2(R, ids, nodes, f"{w}2t" if w != "set3" else "set3t", "set2t", f"{tag} {w} 救回")
        a3 = w3(R, ids, nodes, w + "2" if w != "set3" else "set3", "set2", f"{tag} {w} 门槛")
        out.append((a1, a2, a3))
    return out[0]


def analyze():
    R, tools = load()
    json.dump(R, open(os.path.join(HERE, "rows_w.json"), "w"))
    print(f"记录 {len(R)}；工具 {({k: round(v, 3) for k, v in tools.items()})}")
    print("\n预检")
    good = []
    for a in NODES:
        X = [r for r in R if r["a"] == a]
        nr = sum(r["right"] for r in X if r["p"] == "blank")
        nw = sum(not r["right"] for r in X if r["p"] == "blank")
        k0 = K(X, p="blank")
        ok = k0 >= .85 and nr >= 10
        good += [a] if ok else []
        print(f"  {a:12s} 第一次对 {nr} 错 {nw}  K(blank) {k0:.0%}（≥ 85%） {S2.mark(ok)}  " + " ".join(f"{p} K {K(X, p=p):.0%}/B {B(X, p=p):.0%}" for p in FORMS[1:]))
    print(f"可用节点 {len(good)}（≥ 4）{S2.mark(len(good) >= 4)}")
    T = {(r["a"], r["id"], r["p"]): r for r in R}
    ids = sorted({r["id"] for r in R})
    a1, a2, a3 = block(R, T, ids, good, "全部")
    print("\n决定")
    if len(good) < 4:
        print("  可用节点不够，没结论。")
    print(f"  盲投表达式应带出口（W1、W2、W3）：{S2.mark(a1 and a2 and a3 and len(good) >= 4)}")
    print(f"  出口有代价（W1 过 W2 不过）：{S2.mark(a1 and not a2 and len(good) >= 4)}")
    print(f"  措辞动不了好节点（W1 不过）：{S2.mark(not a1 and len(good) >= 4)}")
    print("\n另报不判")
    for s in "HN":
        block(R, T, [i for i in ids if any(r["id"] == i and r["stratum"] == s for r in R)], good, f"{s} 层", judge=False)
    block(R, T, ids, [a for a in good if a not in ("grok", "codex_ns")], "去掉 Grok、Codex", judge=False)


if __name__ == "__main__":
    {"run": run_all, "a": analyze}[sys.argv[1]]()
