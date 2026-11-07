"""补充小实验（2026-11-07，库伯同意；不用 Claude 模型）：回应审稿的两条局限——形式与强度混淆、第二次提示回显自答造成的自锚定。

来由：Findings of ACL 初稿审稿指出 (a) 假集合"硬排除两项"和假前提"软句子"强度没对齐，缺单选项断言基线；
(b) 第二次提示把节点自己的第一次答案、置信、理由回显给它，blank 的 K ≈ 100% 可能是协议造的，所有投放效应都是对着自锚定先验量的。
第三条局限（前提没人工审）由库伯人工审 40 条，本脚本只出审核表，不判。

题、节点、第一次答案：全部复用扩题轮（itemsS.json 297 题、payloadsS.json、runs_s 的 s1_* 第一次答案，拷进 runs_sm），
节点 6 个：ds_flash_a ds_flash_b ds_pro_a codex_ns qwen14 llama8（Kimi、Grok 配额已尽，不补）。
新形式 9 种（措辞与扩题轮一字不差，只加下面两类）：
  臂 A 强度基线（带回显）：one1t = "the correct answer is G."（G 金标字母）；one1f = "the correct answer is W."（W = S2.wrong(it)[0]，和 L3f/set2f 的目标一致）。
  臂 B 无回显：blank_ne set2t_ne set2f_ne cond2t_ne cond2f_ne L3t_ne L3f_ne —— 第二次提示去掉
    "Your earlier answer: … Your earlier reason: …" 那一行，其余（"Here are questions you answered earlier. Give your answer again."、投放句、尾句）不变。
  旧形式 blank set2t/f cond2t/f L3t/f 的行直接从 rows_s.json 读，不重跑。
指标同前：K（第一次对的保住率）、B（第一次错的救回率）、v*（截断定义为主，原定义另报）；"第一次对/错"按 runs_s 的第一次答案。
  无回显臂的 v* 以 blank_ne 为基线（把 _ne 后缀去掉后当一套独立形式算）。配对差按同节点同题；区间按题重采样 2000 次 95%；节点均值。

预检：每节点新形式覆盖 ≥ 90%；K(blank) ≥ .85、第一次对 ≥ 30、错 ≥ 15（同扩题轮）；可用节点 < 5 算没结论。K(blank_ne) 不设门槛，只报。
判据：
  E1a（只报）单选项断言 vs 集合：K(one1f) − K(set2f)，B(one1t) − B(set2t)。
  E1b 前提比单选项断言安全：K(L3f) − K(one1f) ≥ .15，区间下界 > 0，≥ 4/6 节点 > 0。
  E1c 前提比单选项断言便宜：v*(L3) − v*(one1) 区间上界 < 0（截断定义；原定义另报）。
  E2a（只报）自锚定大小：每节点 K(blank_ne)，和 K(blank) 的差。
  E2b 出口在无回显下成立：K(cond2f_ne) − K(set2f_ne) ≥ .15，下界 > 0，≥ 4/6 节点 > 0。
  E2c 前提优于集合在无回显下成立：K(L3f_ne) − K(set2f_ne) ≥ .15，下界 > 0，≥ 4/6 节点 > 0。
  E2d（只报）无回显下 v* 顺序：v*(cond2_ne) − v*(set2_ne)，v*(L3_ne) − v*(cond2_ne)；回显保护量 K(set2f) − K(set2f_ne)、K(L3f) − K(L3f_ne)、B(set2t) − B(set2t_ne)、B(L3t) − B(L3t_ne)。
决定（写死）：
  E1b 且 E1c 过 → "前提比最强的点名形式还便宜，强度混淆解释不了前提优势"；E1b 过 E1c 不过 → "前提只是假时更安全，不更便宜"；E1b 不过 → 强度混淆成立，论文局限保留并加数字。
  E2b 且 E2c 过 → "出口和前提结论不是自锚定的产物"；任一不过 → 对应结论加"依赖回显"的限定。
停止规则：DeepSeek 上限 ¥15；本地模型 48 小时没跑完按已完成分析。多重比较不校正，全部报告。
人工审核表：audit_premises.md，随机 40 题（种子 "audit"），每题给题干、选项、金标、目标错项、L3t、L3f，库伯标 真前提是否真且不泄答案 / 假前提是否假且像真的。

用法：~/gut-external/torch215-venv/bin/python small.py prep | run | a | audit
"""
import concurrent.futures as cf
import glob
import json
import os
import random
import shutil
import sys

import hardn as H
import run as RUN
import final as FN
import scale as X
import stage2 as S2
import wording as W

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_sm")
ITEMS, PAY = X.ITEMS, X.PAY
NODES = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns", "qwen14", "llama8"]
FORMS = ["one1t", "one1f", "blank_ne", "set2t_ne", "set2f_ne", "cond2t_ne", "cond2f_ne", "L3t_ne", "L3f_ne"]
OLD = ["blank", "set2t", "set2f", "cond2t", "cond2f", "L3t", "L3f"]
LIMIT = 15.0
m = W.m


def note(it, p, pay, items):
    q = p[:-3] if p.endswith("_ne") else p
    if q == "blank":
        return ""
    if q == "one1t":
        return f"the correct answer is {RUN.LET[it['gold']]}."
    if q == "one1f":
        return f"the correct answer is {RUN.LET[S2.wrong(it)[0]]}."
    return FN.note_j(it, q, pay, items)


def run_node(node, items, pay):
    ids = {x["id"] for x in items}
    first = {}
    for fn in glob.glob(os.path.join(OUT, f"s1_*_{node}.json")):
        first.update(json.load(open(fn))["answers"])
    par = H.PAR.get(node, 4)
    bs = RUN.batches(items)

    def do_second(j):
        bid, g, p = j
        parts = []
        for it in g:
            e = first.get(it["id"], {}) if isinstance(first.get(it["id"], {}), dict) else {}
            n = note(it, p, pay, items)
            echo = "" if p.endswith("_ne") else f"\nYour earlier answer: {e.get('answer', '?')} (confidence {e.get('confidence', '?')}). Your earlier reason: {e.get('reason', '')}"
            parts.append(RUN.item_text(it) + echo + (f"\nA note shared with the whole team: {n}" if n else ""))
        return H.call(node, "Here are questions you answered earlier. Give your answer again.\n\n" + "\n\n".join(parts) + "\n\n" + RUN.TAIL, os.path.join(OUT, f"{p}_{bid}_{node}.json"), ids)
    jobs = [(b, g, p) for b, g in bs for p in FORMS]
    random.Random(node).shuffle(jobs)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(do_second, jobs))
    print(node, "完成", flush=True)


def prep():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for a in NODES:
        for fn in glob.glob(os.path.join(X.OUT, f"s1_*_{a}.json")):
            shutil.copy(fn, os.path.join(OUT, os.path.basename(fn)))
            n += 1
    print("拷贝第一次答案", n)
    items = json.load(open(ITEMS))
    pay = json.load(open(PAY))
    it = next(x for x in items if x["id"] in pay)
    for p in FORMS:
        print(p, "|", note(it, p, pay, items)[:90])


def run_all():
    H.OUT, H.FORMS, H.LIMIT = OUT, FORMS, LIMIT
    pay = json.load(open(PAY))
    items = [x for x in json.load(open(ITEMS)) if x["id"] in pay]
    print("题", len(items), flush=True)
    H._start[0] = RUN.balance()
    api = [a for a in NODES if a not in X.LOCAL]
    with cf.ThreadPoolExecutor(len(api) + 1) as ex:
        futs = [ex.submit(run_node, a, items, pay) for a in api]
        futs.append(ex.submit(lambda: [run_node(a, items, pay) for a in NODES if a in X.LOCAL]))
        for f in futs:
            f.result()
    print("全部完成，DeepSeek 花费", round(H.spent(), 2), "停止标志", H._stop[0], flush=True)


def analyze():
    W.OUT, W.FORMS, W.NODES = OUT, FORMS, NODES
    Rn, tools = W.load()
    Ro = [r for r in json.load(open(os.path.join(HERE, "rows_s.json"))) if r["a"] in NODES and r["p"] in OLD]
    R = Ro + Rn
    json.dump(R, open(os.path.join(HERE, "rows_sm.json"), "w"))
    items = {x["id"]: x for x in json.load(open(ITEMS))}
    ids = sorted({r["id"] for r in Rn})
    print(f"新记录 {len(Rn)}；旧记录 {len(Ro)}；题 {len(ids)}；工具 {({k: round(v, 3) for k, v in tools.items()})}")
    print("\n预检")
    good = []
    for a in NODES:
        X_ = [r for r in R if r["a"] == a]
        nr = sum(r["right"] for r in X_ if r["p"] == "blank")
        nw = sum(not r["right"] for r in X_ if r["p"] == "blank")
        k0 = W.K(X_, p="blank")
        cover = sum(r["p"] in FORMS for r in X_) / max(len(ids) * len(FORMS), 1)
        ok = k0 >= .85 and nr >= 30 and nw >= 15 and cover >= .90
        good += [a] if ok else []
        print(f"  {a:12s} 新形式覆盖 {cover:.0%} 第一次对 {nr} 错 {nw}  K(blank) {k0:.0%}  K(blank_ne) {W.K(X_, p='blank_ne'):.0%} {S2.mark(ok)}  "
              + " ".join(f"{p} K {W.K(X_, p=p):.0%}/B {W.B(X_, p=p):.0%}" for p in FORMS if p != "blank_ne"))
    enough = len(good) >= 5
    print(f"可用节点 {len(good)}（≥ 5）{S2.mark(enough)}")
    T = {(r["a"], r["id"], r["p"]): r for r in R}
    need = -(-2 * len(good) // 3)

    print("\n臂 A 强度基线（带回显）")
    W.w1(T, ids, good, "one1f", "set2f", "E1a 单选项断言 vs 集合 跳出", need=need)
    X.w2(R, ids, good, "one1t", "set2t", "E1a 单选项断言 vs 集合 救回")
    e1b = W.w1(T, ids, good, "L3f", "one1f", "E1b 前提 vs 单选项断言 跳出", need=need)
    X.w2(R, ids, good, "L3t", "one1t", "E1b 组件 救回")
    _, _, hi = X.dv(R, ids, good, "L3", "one1", "E1c 截断")
    X.dv(R, ids, good, "L3", "one1", "E1c 原定义", vf=W.vstar)
    X.dv(R, ids, good, "cond2", "one1", "另报 集合+出口 vs 单选项断言 截断")
    X.dv(R, ids, good, "set2", "one1", "另报 集合 vs 单选项断言 截断")
    for p in ("set2", "cond2", "L3", "one1"):
        vs = [X.v0(R, a, set(ids), p) for a in good]
        print(f"  v*({p}) 截断均值 {m(vs):.2f}  " + " ".join(f"{a} {v:.2f}" for a, v in zip(good, vs)))
    e1c = hi < 0

    print("\n臂 B 无回显")
    Rne = [dict(r, p=r["p"][:-3]) for r in R if r["p"].endswith("_ne")]
    Tne = {(r["a"], r["id"], r["p"]): r for r in Rne}
    for a in good:
        X_ = [r for r in R if r["a"] == a]
        print(f"  E2a {a:12s} K(blank) {W.K(X_, p='blank'):.0%}  K(blank_ne) {W.K(X_, p='blank_ne'):.0%}  B(blank_ne) {W.B(X_, p='blank_ne'):.0%}")
    e2b = W.w1(Tne, ids, good, "cond2f", "set2f", "E2b 无回显 出口 跳出", need=need)
    e2c = W.w1(Tne, ids, good, "L3f", "set2f", "E2c 无回显 前提 vs 集合 跳出", need=need)
    X.w2(Rne, ids, good, "cond2t", "set2t", "另报 无回显 出口 救回")
    X.w2(Rne, ids, good, "L3t", "set2t", "另报 无回显 前提 vs 集合 救回")
    X.dv(Rne, ids, good, "cond2", "set2", "E2d 无回显 截断")
    X.dv(Rne, ids, good, "L3", "cond2", "E2d 无回显 截断")
    X.dv(Rne, ids, good, "L3", "cond2", "E2d 无回显 原定义", vf=W.vstar)
    for p in ("set2", "cond2", "L3"):
        vs = [X.v0(Rne, a, set(ids), p) for a in good]
        print(f"  v*({p}_ne) 截断均值 {m(vs):.2f}  " + " ".join(f"{a} {v:.2f}" for a, v in zip(good, vs)))
    for p in ("set2f", "L3f", "cond2f"):
        W.w1(T, ids, good, p, p + "_ne", f"E2d 回显保护 K({p}) − K({p}_ne)", need=need)
    for p in ("set2t", "L3t", "cond2t"):
        X.w2(R, ids, good, p, p + "_ne", f"E2d 回显对救回 B({p}) − B({p}_ne)")

    print("\n决定")
    print(f"  E1b 前提比单选项断言安全：{S2.mark(enough and bool(e1b))}")
    print(f"  E1c 前提比单选项断言便宜：{S2.mark(enough and e1c)}")
    print(f"  强度混淆解释不了前提优势（E1b 且 E1c）：{S2.mark(enough and bool(e1b) and e1c)}")
    print(f"  出口和前提结论不是自锚定产物（E2b 且 E2c）：{S2.mark(enough and bool(e2b) and bool(e2c))}")


def audit():
    items = [x for x in json.load(open(ITEMS))]
    pay = json.load(open(PAY))
    items = [x for x in items if x["id"] in pay]
    rng = random.Random("audit")
    rng.shuffle(items)
    lines = ["# 前提人工审核表（40 题，种子 audit）", "", "每题两问：真前提（L3t）是否**真**且**不泄答案**？假前提（L3f）是否**假**且**像真的**？在 `[ ]` 里填 Y / N，可加一句备注。", ""]
    for it in items[:40]:
        w0 = S2.wrong(it)[0]
        lines += [f"## {it['id']}（{it['task']} / {it.get('source', '')}）", "", it["q"], ""]
        lines += [f"- {l}. {o}" + ("  ← 金标" if i == it["gold"] else "  ← 假前提目标" if i == w0 else "") for i, (l, o) in enumerate(zip(RUN.LET, it["opts"]))]
        lines += ["", f"- L3t：{pay[it['id']]['L3t']}", "  - 真且不泄答案？[ ]", f"- L3f：{pay[it['id']]['L3f']}", "  - 假且像真的？[ ]", ""]
    open(os.path.join(HERE, "audit_premises.md"), "w").write("\n".join(lines))
    print("写出 audit_premises.md")


if __name__ == "__main__":
    {"prep": prep, "run": run_all, "a": analyze, "audit": audit}[sys.argv[1]]()
