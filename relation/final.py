"""最终表达式对照（预注册，跑之前提交）

问题：盲投（找不到故障在哪、广播给所有节点）用哪句话最划算？到现在"前提 L3"和"带出口的集合 cond2"的 v* 是两批数据拼的，没在同一批题同一批节点上比过；出口只在 J 格式（选择题成批）测过，开放题没测。

复用：题型对照的 80 题（H 40 / N 40）、6 个非 Claude 节点（kimi grok ds_flash_a ds_flash_b ds_pro_a codex_ns）、第一次答案和已跑的形式。不用 Claude 做节点也不做裁判。

臂 J（runs_fj，选择题成批，答案是字母，不用裁判）：
  已有 blank set2t set2f cond2t cond2f L3f；新跑 L3t、L3ct、L3cf。
  L3c = 前提原句 + " Rely on this unless you have strong reasons to think otherwise."（和 cond2 的出口平行）
  三个候选表达式的盈亏平衡真值率 v*（和措辞对照同一个 vstar：净收益 = ε(B−B0) − (1−ε)(K0−K)，v* = −net_f/(net_t−net_f)）：
    cond2 = 集合+出口；L3 = 前提；L3c = 前提+出口。
  F1  v*(L3c) − v*(cond2)：按题 bootstrap 2000 次，95% CI 上界 < 0 才算 L3c 赢。
  F2  v*(L3) − v*(cond2)：同上。
  F3  v*(L3c) − v*(L3)：同上（出口对前提有没有用）。
  组件另报：K(L3cf) − K(L3f)、B(L3ct) − B(L3t)、K(L3f) − K(cond2f)、B(L3t) − B(cond2t)（配对、bootstrap）。
  决定（J）：三个候选里均值 v* 最低的那个，如果对另外两个的配对差 CI 上界都 < 0 → 推荐它；否则"分不出"，只报组件。
  只用 v* 算得出的节点（net_t > 0 > net_f）；算不出的列出来。

臂 O（runs_f，开放作答、裁判判结论）：
  已有 5 节点的 s1 blank set2f set2t（ds_flash_b 没有，全新跑）；新跑 cond2t cond2f（集合文本同 set2，末尾加 ", unless you have strong reasons for another option."）。
  裁判换成 ds_pro_a（不知道节点和投放），所有 O 行（含旧行）重判一遍，存 gradesFinal.json，和旧 Sonnet 评分只做一致率报告。
  裁判预检：随机 100 行另由 codex_ns 判，kappa ≥ .7；不到就臂 O 只报不判。
  F4  K(cond2f) − K(set2f) 配对均值 ≥ .15，CI 下界 > 0，≥ 4/6 节点 > 0（节点第一次对 ≥ 10 题）。
  F5  B(cond2t) − B(set2t) ≥ −.15 且 CI 下界 ≥ −.30（节点第一次错 ≥ 8 题，≥ 3 个节点）。
  F6  v*(cond2) − v*(set2) < 0 且 CI 上界 < 0。
  决定（O）：F4 F5 F6 全过 → 出口在开放题上也成立；F4 过 F5 不过 → 出口在开放题上有代价；F4 不过 → 出口在开放题上不成立。

预检：每臂每节点 K(blank) ≥ 85% 且第一次对 ≥ 10 题；可用节点 ≥ 4。
另报不判：去掉 Grok、Codex（只剩 4 个纯 API 节点）重算 F1–F6；每节点 K、B 全表。
预告的局限：错题每节点只有 11–21 道，B 和 v* 噪声大；Grok 留着 read_file、Codex 有只读沙盒（记录次数）；裁判判不了理由；这批题全是选择题库改的开放题，不外推到代码、排班。
停止规则：DeepSeek 上限 ¥8（每 30 次调用查一次余额），到线就停，缺的行照实报。跑完不管结果都停，不再加形式、不加轴、不改公式。多重比较不校正，全部报告。

用法：~/gut-external/torch215-venv/bin/python final.py run | grade | a
"""
import concurrent.futures as cf
import json
import os
import random
import sys

import formtest as F
import hardn as H
import run as RUN
import stage2 as S2
import stage3 as S3
import wording as W

HERE = RUN.HERE
NODES = W.NODES
JFORMS = ["blank", "set2t", "set2f", "cond2t", "cond2f", "L3f", "L3t", "L3ct", "L3cf"]
OFORMS = ["blank", "set2f", "set2t", "cond2t", "cond2f"]
ITEMS_O = os.path.join(HERE, "itemsF_o.json")
GRADE = os.path.join(HERE, "gradesFinal.json")
KAPPA = os.path.join(HERE, "kappaFinal.json")
EXIT_SET = ", unless you have strong reasons for another option."
EXIT_PREM = " Rely on this unless you have strong reasons to think otherwise."
LIMIT = 8.0
JUDGE = "ds_pro_a"
m = W.m


def note_j(it, p, pay, items):
    if p.startswith("L3"):
        base = pay.get(it["id"], {}).get("L3" + p[-1], "")
        return (base + EXIT_PREM) if base and p.endswith(("ct", "cf")) else base
    return W.note(it, p, pay, items)


_note_o = S3.note  # formtest 已把 id 换成 base 再调 stage3.note


def note_o(it, p, pay, items):
    if p.startswith("cond2"):
        t = _note_o(it, "set2" + p[5], pay, items)
        return t[:-1] + EXIT_SET if t.endswith(".") else t + EXIT_SET
    return _note_o(it, p, pay, items)


S2.note = note_j
S3.note = note_o
S3.AGENTS, S3.PAYLOADS, S3.ITEMS, S3.GRADE = NODES, OFORMS, ITEMS_O, GRADE
H.FORMS, H.LIMIT, F.LIMIT = JFORMS, LIMIT, LIMIT
W.FORMS = JFORMS
H.PAR["ds_pro_a"] = 8


def items_o():
    om = [x for x in json.load(open(F.ITEMS)) if x["cond"] == "O"]
    json.dump(om, open(ITEMS_O, "w"), ensure_ascii=False, indent=0)
    return om


def run_o(node, om, pay):
    par = F.PAR.get(node, 4)
    stopped = lambda: H._stop[0] and node.startswith("ds_")

    def first(it):
        if not stopped():
            S3.call(node, S3.statement(it), S3.first_path(it["id"], node))
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(first, om))

    def second(j):
        it, p = j
        if os.path.exists(S3.first_path(it["id"], node)) and not stopped():
            S3.call(node, S3.second_text(it, node, p, pay, om), S3.second_path(it, node, p))
    jobs = [(it, p) for it in om for p in OFORMS]
    random.Random(node + "final").shuffle(jobs)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(second, jobs))


def run_all():
    base = json.load(open(F.BASE))
    om = items_o()
    pay = json.load(open(F.PAY))
    H._start[0] = RUN.balance()

    def node(a):
        H.run_node(a, base, pay)
        run_o(a, om, pay)
        print(a, "O 完成", flush=True)
    with cf.ThreadPoolExecutor(len(NODES)) as ex:
        list(ex.map(node, NODES))
    print("全部完成，DeepSeek 花费", round(H.spent(), 2), flush=True)


def grade_all():
    om = items_o()
    rows = S3.gather(om, OFORMS)
    G = json.load(open(GRADE)) if os.path.exists(GRADE) else {}
    todo = [r for r in rows if r[0] not in G]
    random.Random("jf").shuffle(todo)
    groups = [todo[i:i + 6] for i in range(0, len(todo), 6)]
    with cf.ThreadPoolExecutor(8) as ex:
        for v in ex.map(lambda g: S3.judge_batch(g, JUDGE), groups):
            G.update(v)
    json.dump(G, open(GRADE, "w"))
    print("评分", len(G), "/", len(rows), flush=True)
    s = [r for r in rows if r[0] in G]
    random.Random("kF").shuffle(s)
    s = s[:100]
    res = {}
    with cf.ThreadPoolExecutor(4) as ex:
        for v in ex.map(lambda i: S3.judge_batch(s[i:i + 6], "codex_ns"), range(0, len(s), 6)):
            res.update(v)
    both = [(G[k], res[k]) for k in res if k in G]
    out = dict(n=len(both), agree=agree(both), kappa=kappa(both))
    old = json.load(open(F.GRADE)) if os.path.exists(F.GRADE) else {}
    ov = [(G[k], old[k]) for k in G if k in old]
    out.update(sonnet_n=len(ov), sonnet_agree=agree(ov), sonnet_kappa=kappa(ov))
    json.dump(out, open(KAPPA, "w"))
    print("裁判预检", out, flush=True)


def agree(both):
    return sum(a == b for a, b in both) / len(both) if both else float("nan")


def kappa(both):
    if not both:
        return float("nan")
    ag = agree(both)
    pa, pb = m(a for a, _ in both), m(b for _, b in both)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (ag - pe) / (1 - pe) if pe < 1 else 1.0


def precheck(R, lab, forms):
    print(f"\n预检 {lab}")
    good = []
    for a in NODES:
        X = [r for r in R if r["a"] == a]
        nr = sum(r["right"] for r in X if r["p"] == "blank")
        nw = sum(not r["right"] for r in X if r["p"] == "blank")
        k0 = W.K(X, p="blank")
        ok = k0 >= .85 and nr >= 10
        good += [a] if ok else []
        print(f"  {a:12s} 第一次对 {nr} 错 {nw}  K(blank) {k0:.0%} {S2.mark(ok)}  " + " ".join(f"{p} K {W.K(X, p=p):.0%}/B {W.B(X, p=p):.0%}" for p in forms[1:]))
    print(f"可用节点 {len(good)}（≥ 4）{S2.mark(len(good) >= 4)}")
    return good


def diff3(R, ids, nodes, p1, p2, lab):
    """v*(p1) − v*(p2)，CI 上界 < 0 才算 p1 赢。"""
    return W.w3(R, ids, nodes, p1, p2, lab)


def arm_j(R, ids, nodes, lab, judge=True):
    print(f"\n[J {lab}] 节点 {len(nodes)} 题 {len(ids)}")
    T = {(r["a"], r["id"], r["p"]): r for r in R}
    cand = ["cond2", "L3", "L3c"]
    vs = {a: {c: W.vstar(R, a, set(ids), c) for c in cand} for a in nodes}
    ns = [a for a in nodes if all(vs[a][c] == vs[a][c] for c in cand)]
    mean = {c: m(vs[a][c] for a in ns) for c in cand}
    print("  v* 均值 " + "  ".join(f"{c} {mean[c]:.2f}" for c in cand) + f"（{len(ns)} 节点）" + ("  算不出：" + " ".join(a for a in nodes if a not in ns) if len(ns) < len(nodes) else ""))
    tag = "F" if judge else "报告"
    f1 = diff3(R, ids, nodes, "L3c", "cond2", f"{tag}1")
    f2 = diff3(R, ids, nodes, "L3", "cond2", f"{tag}2")
    f3 = diff3(R, ids, nodes, "L3c", "L3", f"{tag}3")
    W.w1(T, ids, nodes, "L3cf", "L3f", "  组件 出口对前提 跳出")
    W.w2(R, ids, nodes, "L3ct", "L3t", "  组件 出口对前提 救回")
    W.w1(T, ids, nodes, "L3f", "cond2f", "  组件 前提 vs 集合+出口 跳出")
    W.w2(R, ids, nodes, "L3t", "cond2t", "  组件 前提 vs 集合+出口 救回")
    if not ns:
        return None
    best = min(cand, key=lambda c: mean[c])
    others = [c for c in cand if c != best]
    wins = all(diff3(R, ids, nodes, best, o, f"  {best} vs {o}") for o in others)
    return best if wins else None


def arm_o(R, ids, nodes, lab, judge=True):
    print(f"\n[O {lab}] 节点 {len(nodes)} 题 {len(ids)}")
    T = {(r["a"], r["id"], r["p"]): r for r in R}
    tag = "F" if judge else "报告"
    f4 = W.w1(T, ids, nodes, "cond2f", "set2f", f"{tag}4 出口 跳出")
    f5 = W.w2(R, ids, nodes, "cond2t", "set2t", f"{tag}5 出口 救回")
    f6 = W.w3(R, ids, nodes, "cond2", "set2", f"{tag}6 出口 门槛")
    return f4, f5, f6


def analyze():
    Rj, tools = W.load()
    Rj = [r for r in Rj if r["p"] in JFORMS]
    items_o()
    Ro, miss, tot = S3.load()
    for r in Ro:
        r["id"] = r["id"][1:]  # O 前缀去掉，和 J 同题号
    json.dump(dict(j=Rj, o=Ro), open(os.path.join(HERE, "rows_final.json"), "w"))
    kp = json.load(open(KAPPA)) if os.path.exists(KAPPA) else {}
    print(f"J 记录 {len(Rj)}；O 记录 {len(Ro)} 缺 {miss}/{tot}；工具 {({k: round(v, 3) for k, v in tools.items()})}")
    print(f"裁判 {JUDGE}：和 codex_ns kappa {kp.get('kappa', float('nan')):.2f}（≥ .7）{S2.mark(kp.get('kappa', 0) >= .7)}；和旧 Sonnet 评分一致 {kp.get('sonnet_agree', float('nan')):.0%} kappa {kp.get('sonnet_kappa', float('nan')):.2f}（{kp.get('sonnet_n', 0)} 行）")
    gj = precheck(Rj, "J", JFORMS)
    go = precheck(Ro, "O", OFORMS)
    ids = sorted({r["id"] for r in Rj})
    best = arm_j(Rj, ids, gj, "全部")
    ids_o = sorted({r["id"] for r in Ro})
    f4, f5, f6 = arm_o(Ro, ids_o, go, "全部")
    print("\n决定")
    print(f"  J 推荐表达式：{best or '分不出'}" + ("" if len(gj) >= 4 else "（可用节点不够）"))
    jo = kp.get("kappa", 0) >= .7 and len(go) >= 4
    print(f"  O 裁判与节点够用：{S2.mark(jo)}")
    print(f"  出口在开放题上成立（F4 F5 F6）：{S2.mark(jo and f4 and f5 and f6)}")
    print(f"  出口在开放题上有代价（F4 过 F5 不过）：{S2.mark(jo and f4 and not f5)}")
    print(f"  出口在开放题上不成立（F4 不过）：{S2.mark(jo and not f4)}")
    print("\n另报不判")
    sub = [a for a in gj if a not in ("grok", "codex_ns")]
    arm_j(Rj, ids, sub, "去掉 Grok、Codex", judge=False)
    arm_o(Ro, ids_o, [a for a in go if a not in ("grok", "codex_ns")], "去掉 Grok、Codex", judge=False)
    for s in "HN":
        sid = [i for i in ids if any(r["id"] == i and r["stratum"] == s for r in Rj)]
        arm_j(Rj, sid, gj, f"{s} 层", judge=False)


if __name__ == "__main__":
    {"run": run_all, "grade": grade_all, "a": analyze}[sys.argv[1]]()
