"""投稿前三件补强（2026-11-08，库伯同意；不用 Claude 模型）：回应三条最重的审稿风险——
(1) 前提与集合的说服强度不匹配；(2) 297 题答案键与前提质量没审；(3) 统计与模型覆盖薄（六节点三个是 DeepSeek）。

一、强度匹配的前提（预注册，本脚本 run）
  新形式 L3n（"前提 + 点名"）：L3nt = L3t + " So the correct answer is G."（G 金标字母）；L3nf = L3f + " So the correct answer is W."（W = S2.wrong(it)[0]，
  与 one1f / L3f / set2f 的目标一致）。带回显，提示其余部分与扩题轮一字不差。它的断言强度和 one1 一样，只多了一句理由。
  题、第一次答案复用扩题轮（itemsS.json 297 题、runs_s 的 s1_*）。节点：云端 ds_flash_a ds_flash_b ds_pro_a codex_ns 先跑；
  本地 qwen14 llama8 等 GUT 的本地实验结束后补跑（H.call 跳过已有文件，补跑不重跑）；最终判定在 ≥ 5 节点可用时下，云端四节点只报中间结果。
  旧形式 one1t/f、L3t/f、set2t/f、cond2t/f 的行从 rows_sm.json 读，不重跑。
  预检同 small.py：每节点新形式覆盖 ≥ 90%，K(blank) ≥ .85，第一次对 ≥ 30、错 ≥ 15。
  判据（配对差按同节点同题，区间按题重采样 2000 次 95%，节点均值；need = ceil(2n/3)）：
    F1 理由本身在挡：K(L3nf) − K(one1f) ≥ .15，下界 > 0，≥ need 节点 > 0。
    F2 不点名本身在挡：K(L3f) − K(L3nf) ≥ .15，下界 > 0，≥ need 节点 > 0。
    F3（只报）：B(L3nt) − B(one1t)，B(L3nt) − B(L3t)，v*(L3n) 截断与原定义，v*(L3n) − v*(one1)，v*(L3n) − v*(L3)。
  决定（写死）：
    F1 过且 F2 不过 → "理由在挡，点不点名无所谓，强度混淆排除"；
    F1 不过 → "安全来自不点名，前提本身没功劳，强度混淆成立"；
    F1 过且 F2 过 → "理由和不点名各挡一部分"，论文按两段分别报数。
  停止规则：DeepSeek 上限 ¥15；多重比较不校正，全部报告。

二、答案键与前提审核 + 敏感性分析（本脚本 flag / sens）
  筛题只用无投放的第一次答案（看不到任何投放效应）：codex_ns、ds_pro_a、ds_flash_a 三个节点里 ≥ 2 个答错且答的是同一个错项 → 进审核表；
  再加已知有问题的 S0239 S0971 S0881 S0146。库伯对每题标：答案键对不对（不对给正确字母）、L3t 是否真且不泄、L3f 是否假、可信、且指向目标错项。
  敏感性分析：把"答案键错或有争议"和"任一前提写坏"的题剔掉，用剩余题重算扩题轮 S1–S5、small 的 E1b/E1c/E2b/E2c/E2d、本脚本 F1/F2，
  每条给剔除前后两列。判据和阈值不变；结论的写法：哪条方向翻了就在论文里按翻后的写。

三、新节点复现（预注册，等本地空闲后跑）
  目标：两个与现有家族不同的开源模型（候选 gemma3:12b、mistral:7b，以本机能跑为准；型号在看到它们任何数据前写进 collect.py 并提交），
  跑扩题轮九条件 + one1t/f + L3nt/f，第一次答案同协议新采。判据为复现：R1 K(cond2f) − K(set2f) > 0；R2 K(L3f) − K(set2f) > 0；
  R3 v*(L3) < v*(cond2)（截断与原定义都报）；按节点逐个报方向，不设节点均值门槛（n=2）。Flash-B（唯一采样节点）另补跑两次 blank/set2/cond2/L3 给节点内方差，只报。

用法：~/gut-external/torch215-venv/bin/python followup.py flag | prep | run [node ...] | a | sens
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
import small as SM
import stage2 as S2
import wording as W

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_fu")
ITEMS, PAY = X.ITEMS, X.PAY
CLOUD = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns"]
LOCAL = ["qwen14", "llama8"]
NODES = CLOUD + LOCAL
FORMS = ["L3nt", "L3nf"]
OLD = ["blank", "one1t", "one1f", "L3t", "L3f", "set2t", "set2f", "cond2t", "cond2f", "L3ct", "L3cf"]
KNOWN_BAD = ["S0239", "S0971", "S0881", "S0146"]
FLAG_NODES = ["codex_ns", "ds_pro_a", "ds_flash_a"]
LIMIT = 15.0
m = W.m


def note(it, p, pay, items):
    if p == "L3nt":
        return pay[it["id"]]["L3t"] + f" So the correct answer is {RUN.LET[it['gold']]}."
    if p == "L3nf":
        return pay[it["id"]]["L3f"] + f" So the correct answer is {RUN.LET[S2.wrong(it)[0]]}."
    return SM.note(it, p, pay, items)


def first_answers(node, src=X.OUT):
    first = {}
    for fn in glob.glob(os.path.join(src, f"s1_*_{node}.json")):
        first.update(json.load(open(fn))["answers"])
    return first


def run_node(node, items, pay):
    ids = {x["id"] for x in items}
    first = first_answers(node, OUT)
    par = H.PAR.get(node, 4)
    bs = RUN.batches(items)

    def do_second(j):
        bid, g, p = j
        parts = []
        for it in g:
            e = first.get(it["id"], {}) if isinstance(first.get(it["id"], {}), dict) else {}
            n = note(it, p, pay, items)
            echo = f"\nYour earlier answer: {e.get('answer', '?')} (confidence {e.get('confidence', '?')}). Your earlier reason: {e.get('reason', '')}"
            parts.append(RUN.item_text(it) + echo + f"\nA note shared with the whole team: {n}")
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
        print(p, "|", note(it, p, pay, items)[:160])


def run_all():
    H.OUT, H.FORMS, H.LIMIT = OUT, FORMS, LIMIT
    pay = json.load(open(PAY))
    items = [x for x in json.load(open(ITEMS)) if x["id"] in pay]
    nodes = sys.argv[2:] or CLOUD
    print("题", len(items), "节点", nodes, flush=True)
    H._start[0] = RUN.balance()
    api = [a for a in nodes if a not in X.LOCAL]
    with cf.ThreadPoolExecutor(len(api) + 1) as ex:
        futs = [ex.submit(run_node, a, items, pay) for a in api]
        futs.append(ex.submit(lambda: [run_node(a, items, pay) for a in nodes if a in X.LOCAL]))
        for f in futs:
            f.result()
    print("全部完成，DeepSeek 花费", round(H.spent(), 2), "停止标志", H._stop[0], flush=True)


def flag():
    items = {x["id"]: x for x in json.load(open(ITEMS))}
    pay = json.load(open(PAY))
    first = {a: first_answers(a) for a in FLAG_NODES}
    ans = lambda a, i: (first[a].get(i) or {}).get("answer") if isinstance(first[a].get(i), dict) else None
    picked = []
    for i in sorted(items):
        if i not in pay:
            continue
        gold = RUN.LET[items[i]["gold"]]
        w = [ans(a, i) for a in FLAG_NODES if ans(a, i) and ans(a, i) != gold]
        if (len(w) >= 2 and len(set(w)) == 1) or i in KNOWN_BAD:
            picked.append(i)
    lines = ["# 答案键与前提审核表（筛选：Codex / Pro / Flash-A 三者中 ≥ 2 个答同一个错项，加已知 4 题）", "",
             f"共 {len(picked)} 题。每题三问，在 `[ ]` 里填 Y / N；答案键错的在后面写正确字母。", ""]
    for i in picked:
        it = items[i]
        w0 = S2.wrong(it)[0]
        lines += [f"## {i}（{it['task']} / {it.get('source', '')}）", "", it["q"], ""]
        lines += [f"- {l}. {o}" + ("  ← 答案键" if k == it["gold"] else "  ← 假前提目标" if k == w0 else "") for k, (l, o) in enumerate(zip(RUN.LET, it["opts"]))]
        lines += ["", "- 模型第一次答案：" + "，".join(f"{a} {ans(a, i) or '?'}" for a in FLAG_NODES), "",
                  "  - 答案键对吗？[ ]（不对写正确字母：  ）",
                  f"- L3t：{pay[i]['L3t']}", "  - 真且不泄答案？[ ]",
                  f"- L3f：{pay[i]['L3f']}", "  - 假、可信、且指向目标错项？[ ]", ""]
    open(os.path.join(HERE, "audit_keys.md"), "w").write("\n".join(lines))
    json.dump(picked, open(os.path.join(HERE, "audit_keys_ids.json"), "w"))
    print("写出 audit_keys.md", len(picked), "题")


def precheck(R, ids, nodes, forms):
    good = []
    for a in nodes:
        X_ = [r for r in R if r["a"] == a]
        nr = sum(r["right"] for r in X_ if r["p"] == "blank")
        nw = sum(not r["right"] for r in X_ if r["p"] == "blank")
        k0 = W.K(X_, p="blank")
        cover = sum(r["p"] in forms for r in X_) / max(len(ids) * len(forms), 1)
        ok = k0 >= .85 and nr >= 30 and nw >= 15 and cover >= .90
        good += [a] if ok else []
        print(f"  {a:12s} 新形式覆盖 {cover:.0%} 第一次对 {nr} 错 {nw}  K(blank) {k0:.0%} {S2.mark(ok)}  "
              + " ".join(f"{p} K {W.K(X_, p=p):.0%}/B {W.B(X_, p=p):.0%}" for p in forms))
    return good


def tests(R, ids, good, lab=""):
    T = {(r["a"], r["id"], r["p"]): r for r in R}
    need = -(-2 * len(good) // 3)
    print(f"\n[{lab}] 节点 {len(good)} 题 {len(ids)}")
    f1 = W.w1(T, ids, good, "L3nf", "one1f", "F1 理由在挡 跳出", need=need)
    f2 = W.w1(T, ids, good, "L3f", "L3nf", "F2 不点名在挡 跳出", need=need)
    X.w2(R, ids, good, "L3nt", "one1t", "F3 救回 前提+点名 vs 点名")
    X.w2(R, ids, good, "L3nt", "L3t", "F3 救回 前提+点名 vs 前提")
    X.dv(R, ids, good, "L3n", "one1", "F3 截断")
    X.dv(R, ids, good, "L3n", "L3", "F3 截断")
    X.dv(R, ids, good, "L3n", "one1", "F3 原定义", vf=W.vstar)
    for p in ("one1", "L3n", "L3", "set2", "cond2"):
        vs = [X.v0(R, a, set(ids), p) for a in good]
        print(f"  v*({p}) 截断均值 {m(vs):.2f}  " + " ".join(f"{a} {v:.2f}" for a, v in zip(good, vs)))
    return bool(f1), bool(f2)


def load_rows():
    W.OUT, W.FORMS, W.NODES = OUT, FORMS, NODES
    Rn, tools = W.load()
    Ro = [r for r in json.load(open(os.path.join(HERE, "rows_sm.json"))) if r["a"] in NODES and r["p"] in OLD]
    Rc = [r for r in json.load(open(os.path.join(HERE, "rows_s.json"))) if r["a"] in NODES and r["p"] in ("L3ct", "L3cf")]
    return Ro + Rc + Rn, Rn


def analyze():
    R, Rn = load_rows()
    json.dump(R, open(os.path.join(HERE, "rows_fu.json"), "w"))
    ids = sorted({r["id"] for r in Rn})
    print(f"新记录 {len(Rn)}；题 {len(ids)}")
    print("\n预检")
    good = precheck(R, ids, NODES, FORMS)
    enough = len(good) >= 5
    print(f"可用节点 {len(good)}（≥ 5）{S2.mark(enough)}" + ("" if enough else "  —— 中间结果，最终判定等本地节点"))
    f1, f2 = tests(R, ids, good, "F 强度匹配的前提")
    print("\n决定")
    print(f"  F1 理由在挡：{S2.mark(enough and f1)}")
    print(f"  F2 不点名在挡：{S2.mark(enough and f2)}")
    if enough:
        print("  → " + ("理由在挡，点不点名无所谓，强度混淆排除" if f1 and not f2 else "安全来自不点名，前提本身没功劳，强度混淆成立" if not f1 else "理由和不点名各挡一部分"))


def sens():
    """剔除答案键错/有争议或前提写坏的题后重算（读 audit_keys_result.json：{"exclude": [ids]}）。"""
    ex = set(json.load(open(os.path.join(HERE, "audit_keys_result.json")))["exclude"])
    Rs = json.load(open(os.path.join(HERE, "rows_s.json")))
    Rm = json.load(open(os.path.join(HERE, "rows_sm.json")))
    R, _ = load_rows()
    allids = sorted({r["id"] for r in Rs})
    keep = [i for i in allids if i not in ex]
    print(f"剔除 {len(ex)} 题，剩 {len(keep)} / {len(allids)}")
    nodes6 = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns", "qwen14", "llama8"]
    for lab, ids in (("剔除前", allids), ("剔除后", keep)):
        print(f"\n===== {lab} =====")
        T = {(r["a"], r["id"], r["p"]): r for r in Rs}
        need = 4
        X.block(Rs, ids, nodes6, f"扩题轮 S1–S5 {lab}")
        W.w1(T, ids, nodes6, "cond2f", "set2f", "S3 出口 跳出", need=need)
        X.w2(Rs, ids, nodes6, "cond2t", "set2t", "S4 出口 救回")
        W.w1(T, ids, nodes6, "L3cf", "L3f", "S5 前提出口 跳出", need=need)
        W.w1(T, ids, nodes6, "L3f", "set2f", "前提 vs 集合 跳出", need=need)
        Tm = {(r["a"], r["id"], r["p"]): r for r in Rm}
        W.w1(Tm, ids, nodes6, "L3f", "one1f", "E1b", need=need)
        X.dv(Rm, ids, nodes6, "L3", "one1", "E1c 截断")
        X.dv(Rm, ids, nodes6, "L3", "one1", "E1c 原定义", vf=W.vstar)
        Rne = [dict(r, p=r["p"][:-3]) for r in Rm if r["p"].endswith("_ne")]
        Tne = {(r["a"], r["id"], r["p"]): r for r in Rne}
        W.w1(Tne, ids, nodes6, "cond2f", "set2f", "E2b 无回显 出口", need=need)
        W.w1(Tne, ids, nodes6, "L3f", "set2f", "E2c 无回显 前提 vs 集合", need=need)
        X.dv(Rne, ids, nodes6, "L3", "cond2", "E2d 无回显 截断")
        X.dv(Rne, ids, nodes6, "L3", "cond2", "E2d 无回显 原定义", vf=W.vstar)
        have = [a for a in nodes6 if any(r["a"] == a and r["p"] == "L3nf" for r in R)]
        if have:
            tests(R, ids, have, f"F {lab}")


if __name__ == "__main__":
    {"flag": flag, "prep": prep, "run": run_all, "a": analyze, "sens": sens}[sys.argv[1]]()
