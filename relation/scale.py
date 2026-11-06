"""扩题轮（预注册，跑之前提交）——为 Findings of ACL 投稿补规模，不加新假设

目的：最终表达式对照只有 80 题、6 个商业 API 节点，每节点错题 11–21 道，B 和 v* 区间宽，前提对集合的直接比较按规则"分不出"。
本轮把同一个对照扩到约 300 题、8 个节点（加两个本地开源家族 Qwen、Llama），判据不变，只为收窄区间和回答"前提是否优于集合"。

题：从 raw/raw2 里没用过的选择题（排除 race 和两门历史，它们带原文或已用尽）抽 1403 道做池，先用 5 个便宜节点
（ds_flash_a ds_flash_b ds_pro_a qwen14 llama8）各答一遍做难度筛选；选"至少 1 个筛选节点答错"的题，按任务类
（recall / reason / case）尽量均衡，取 320 道；投放物不合格的剔除后剩多少用多少（预计 ≥ 300）。
筛选只用第一次作答，和后面的第一次作答分开跑；选题偏向筛选节点的错题是有意的，B0（空白下的救回率）会吸收回归均值。

投放物：L3t / L3f 由 ds_pro_a 写（不用 Claude），合格条件：≥ 20 字、不出现选项字母、L3t 不抄正确选项原文、L3f 不抄目标选项原文；两次不合格剔除该题。

节点 8 个：kimi grok ds_flash_a ds_flash_b ds_pro_a codex_ns（同上轮）+ qwen14（qwen2.5:14b）+ llama8（llama3.1:8b），后两个走本地 Ollama，温度 0。
形式 9 种：blank set2t set2f cond2t cond2f L3t L3f L3ct L3cf，措辞和上轮完全一样（soft2 不跑，上轮已知和 cond2 等效）。

指标：K、B 同前；v* 用修正定义（假版零杀伤记 0，真版零收益记 1，其余 −net_f/(net_t−net_f)），原定义另报。

预检：每节点 K(blank) ≥ 85%、第一次对 ≥ 30 题、第一次错 ≥ 15 题；可用节点 ≥ 5。不过的节点剔除，另报。
判据（按题 bootstrap 2000 次，95% 区间；节点均值）：
  S1  v*(L3) − v*(cond2)：上界 < 0 → 前提优于集合+出口。
  S2  v*(L3c) − v*(L3)：上界 < 0 → 出口对前提有用；下界 > 0 → 出口对前提有害；否则无增益。
  S3  K(cond2f) − K(set2f) 配对均值 ≥ .15、下界 > 0、≥ 2/3 可用节点 > 0。
  S4  B(cond2t) − B(set2t) ≥ −.15 且下界 ≥ −.30（错题 ≥ 15 的节点）。
  S5  v*(cond2) − v*(set2) 上界 < 0。
  决定：四个候选 {set2, cond2, L3, L3c} 里均值 v* 最低者，若对其余三个的配对差上界都 < 0 → 推荐它；否则"分不出"，报排序。
  S3 S4 S5 全过 → 出口结论在 300 题上复现；S3 过 S4 不过 → 出口有代价；S3 不过 → 出口结论不复现。
另报不判：只看 API 节点 / 只看本地开源节点；按任务类；原定义 v*；每节点 K、B 全表；Grok 工具轮、Codex 命令数。
停止规则：DeepSeek 上限 ¥30，每 40 次调用查一次余额，到线停 DeepSeek；Kimi / Grok 配额用尽就等窗口，等不到的行照实报缺；
  本地模型超过 48 小时未跑完就按已完成的行分析并报覆盖率。跑完不管结果都停，不加形式、不加轴、不改公式。多重比较不校正。
局限预告：筛选节点和测试节点重合（5/8），筛选偏向它们的错题；本地 8B / 14B 模型 JSON 格式错误率可能高，照实报覆盖率；
  题全是 MMLU 类选择题，不外推到开放题和代码。

用法：~/gut-external/torch215-venv/bin/python scale.py pool | screen | gen | run | a
"""
import collections
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import subprocess
import sys
import urllib.request

import analyze as AN
import final as X
import formtest as F
import hardn as H
import newnodes as N
import run as RUN
import stage2 as S2
import wording as W

HERE = RUN.HERE
OUT = os.path.join(HERE, "runs_s")
SCRN = os.path.join(HERE, "screen_s")
POOL = os.path.join(HERE, "poolS.json")
ITEMS = os.path.join(HERE, "itemsS.json")
PAY = os.path.join(HERE, "payloadsS.json")
NODES = ["kimi", "grok", "ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns", "qwen14", "llama8"]
SCREEN = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "qwen14", "llama8"]
LOCAL = {"qwen14": "qwen2.5:14b", "llama8": "llama3.1:8b"}
FORMS = X.JFORMS
TARGET = 320
LIMIT = 30.0
SKIP = {"race", "high_school_us_history", "high_school_world_history"}
m = W.m

# 本地模型：newnodes.ask_cli 只认 qwen14 / qwen9，这里补 llama8，其余原样
_ask = N.ask_cli


def ask_cli(node, text):
    if node in LOCAL:
        body = json.dumps({"model": LOCAL[node], "stream": False, "think": False, "format": "json", "messages": [{"role": "user", "content": text}], "options": {"temperature": 0, "num_predict": 1500, "num_ctx": 8192}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=900))["message"]["content"], 0
    return _ask(node, text)


N.ask_cli = ask_cli
H.OUT, H.FORMS, H.LIMIT = OUT, FORMS, LIMIT
H.PAR.update({"qwen14": 1, "llama8": 1, "ds_flash_a": 8, "ds_flash_b": 8, "ds_pro_a": 8, "codex_ns": 4})
S2.note = X.note_j
W.OUT, W.ITEMS, W.FORMS, W.NODES = OUT, ITEMS, FORMS, NODES


def task_of(source):
    for t, ns in RUN.POOLS.items():
        if source in ns:
            return t
    for t, ns in H.EXTRA.items():
        if source in ns:
            return t
    return None


def pool():
    seen = F.used() | {x["q"] for x in json.load(open(F.BASE))}
    out = []
    for d in ("raw", "raw2"):
        for p in sorted(glob.glob(os.path.join(HERE, d, "*.json"))):
            name = os.path.basename(p)[:-5]
            if name in SKIP or not task_of(name):
                continue
            for r in json.load(open(p)):
                opts = r.get("choices") or r.get("options")
                if opts and r["question"] not in seen and F.ok_item(r["question"], opts, r.get("article", "")):
                    seen.add(r["question"])
                    out.append(dict(task=task_of(name), source=name, q=r["question"], opts=opts, gold=r["answer"] if isinstance(r["answer"], int) else RUN.LET.index(r["answer"]), article=""))
    random.Random("scale-pool").shuffle(out)
    for i, k in enumerate(out):
        k.update(id=f"S{i:04d}", stratum="S", hard=False, cal=False, flagged=False)
    json.dump(out, open(POOL, "w"), ensure_ascii=False, indent=0)
    print("题池", len(out), collections.Counter(x["task"] for x in out))


def screen():
    os.makedirs(SCRN, exist_ok=True)
    os.makedirs(N.SCR, exist_ok=True)
    P = json.load(open(POOL))
    ids = {x["id"] for x in P}
    gold = {x["id"]: x["gold"] for x in P}
    H._start[0] = RUN.balance()

    def one(j):
        node, bid, g = j
        rec = H.call(node, "Answer each question.\n\n" + "\n\n".join(RUN.item_text(it) for it in g) + "\n\n" + RUN.TAIL, os.path.join(SCRN, f"{bid}_{node}.json"), ids)
        return node, rec.get("answers", {})
    wrong = collections.defaultdict(set)
    answered = collections.Counter()
    for node in SCREEN:
        jobs = [(node, b, g) for b, g in RUN.batches(P)]
        with cf.ThreadPoolExecutor(H.PAR.get(node, 4)) as ex:
            for n, a in ex.map(one, jobs):
                for q, v in a.items():
                    if q in gold and isinstance(v, dict) and AN.letter(v) is not None:
                        answered[n] += 1
                        if AN.letter(v) != gold[q]:
                            wrong[q].add(n)
        print(node, "筛选完成，答出", answered[node], "/", len(P), "错", sum(node in s for s in wrong.values()), flush=True)
    cand = [x for x in P if wrong.get(x["id"])]
    by = collections.defaultdict(list)
    for x in cand:
        by[x["task"]].append(x)
    g = random.Random("scale-pick")
    for v in by.values():
        g.shuffle(v)
    pick, i = [], 0
    while len(pick) < TARGET and any(by.values()):
        t = sorted(by)[i % len(by)]
        if by[t]:
            pick.append(by[t].pop())
        i += 1
    for x in pick:
        x["n_wrong"] = len(wrong[x["id"]])
    json.dump(pick, open(ITEMS, "w"), ensure_ascii=False, indent=0)
    print("候选", len(cand), "选出", len(pick), collections.Counter(x["task"] for x in pick), "错节点数分布", collections.Counter(x["n_wrong"] for x in pick), "DeepSeek 花费", round(H.spent(), 2))


def gen():
    items = json.load(open(ITEMS))
    have = json.load(open(PAY)) if os.path.exists(PAY) else {}
    os.makedirs(os.path.join(HERE, "paygenS"), exist_ok=True)
    opts = lambda it: "\n".join(f"{l}. {o}" for l, o in zip(RUN.LET, it["opts"]))

    def ok(it, k, s):
        avoid = it["opts"][it["gold"] if k == "L3t" else S2.wrong(it)[0]].lower()
        return s and len(s) >= 20 and not re.search(r"\b(option|choice|answer)\s*[A-D]\b|\b[A-D]\.", s) and avoid not in s.lower()

    def one(it):
        for k in range(2):
            p = os.path.join(HERE, "paygenS", f"{it['id']}_{k}.json")
            txt = ("We are studying how readers evaluate notes attached to exam questions, so we need matched true and misleading notes. Below is a question with options, the correct option and a wrong target option. Write one short sentence for each field. Never use an option letter.\n"
                   '"L3t": the true key premise or step from which a careful reader could work out the answer. Do not state the answer or repeat the correct option text.\n'
                   '"L3f": a false key premise that would lead a careful reader toward the wrong target. Do not repeat the target option text.\n'
                   'Reply with ONLY a JSON object mapping the question ID to {"L3t": "...", "L3f": "..."}.\n\n'
                   f"{it['id']}: {it['q']}\n{opts(it)}\nCorrect: {RUN.LET[it['gold']]}\nWrong target: {RUN.LET[S2.wrong(it)[0]]}" + ("\n\nYour previous attempt was rejected; write different sentences." if k else ""))
            r = RUN.call("ds_pro_a", txt, p)["answers"].get(it["id"], {})
            if isinstance(r, dict) and ok(it, "L3t", r.get("L3t", "")) and ok(it, "L3f", r.get("L3f", "")):
                return it["id"], {"L3t": r["L3t"], "L3f": r["L3f"]}
        return it["id"], None
    with cf.ThreadPoolExecutor(6) as ex:
        for i, d in ex.map(one, [it for it in items if it["id"] not in have]):
            if d:
                have[i] = d
    json.dump(have, open(PAY, "w"), ensure_ascii=False, indent=0)
    bad = [it["id"] for it in items if it["id"] not in have]
    print("投放物", len(have), "/", len(items), "不合格剔除", len(bad), bad[:20])


def run_all():
    os.makedirs(OUT, exist_ok=True)
    pay = json.load(open(PAY))
    items = [x for x in json.load(open(ITEMS)) if x["id"] in pay]
    print("题", len(items), flush=True)
    H._start[0] = RUN.balance()

    def node(a):
        H.run_node(a, items, pay)
    api = [a for a in NODES if a not in LOCAL]
    with cf.ThreadPoolExecutor(len(api) + 1) as ex:
        futs = [ex.submit(node, a) for a in api]
        futs.append(ex.submit(lambda: [node(a) for a in LOCAL]))  # 本地两个串行，Ollama 一次一个
        for f in futs:
            f.result()
    print("全部完成，DeepSeek 花费", round(H.spent(), 2), "停止标志", H._stop[0], flush=True)


def v0(R, a, s, p):
    X_ = [r for r in R if r["a"] == a and r["id"] in s]
    bl = [r for r in X_ if r["p"] == "blank"]
    if not bl:
        return float("nan")
    e = m(not r["right"] for r in bl)
    K0, B0 = m(r["ok"] for r in bl if r["right"]), m(r["ok"] for r in bl if not r["right"])
    g = lambda q, right: m(r["ok"] for r in X_ if r["p"] == q and r["right"] == right)
    nt = e * (g(p + "t", False) - B0) - (1 - e) * (K0 - g(p + "t", True))
    nf = e * (g(p + "f", False) - B0) - (1 - e) * (K0 - g(p + "f", True))
    if any(x != x for x in (nt, nf)):
        return float("nan")
    return 0.0 if nf >= 0 else 1.0 if nt <= 0 else -nf / (nt - nf)


def dv(R, ids, nodes, p1, p2, lab, vf=v0):
    vs = {a: (vf(R, a, set(ids), p1), vf(R, a, set(ids), p2)) for a in nodes}
    ns = [a for a in nodes if all(x == x for x in vs[a])]
    fn = lambda s: m(x for x in (vf(R, a, set(s), p1) - vf(R, a, set(s), p2) for a in ns) if x == x)
    v = m(vs[a][0] - vs[a][1] for a in ns)
    lo, hi = W.boot(fn, ids)
    print(f"  {lab} v*({p1}) − v*({p2}) {v:+.2f} [{lo:+.2f},{hi:+.2f}]，{len(ns)} 节点  " + " ".join(f"{a} {vs[a][0]:.2f} vs {vs[a][1]:.2f}" for a in ns) + ("  算不出：" + " ".join(a for a in nodes if a not in ns) if len(ns) < len(nodes) else ""))
    return v, lo, hi, len(ns)


def w2(R, ids, nodes, p1, p2, lab, need=15):
    bad = {a: sum(1 for r in R if r["a"] == a and r["p"] == "blank" and not r["right"]) for a in nodes}
    ns = [a for a in nodes if bad[a] >= need]
    fn = lambda s: m(x for x in (W.Bn(R, a, set(s), p1) - W.Bn(R, a, set(s), p2) for a in ns) if x == x)
    v = fn(ids)
    lo, hi = W.boot(fn, ids)
    ok = len(ns) >= 3 and v >= -.15 and lo >= -.30
    print(f"  {lab} B({p1}) − B({p2}) {v:+.2f} [{lo:+.2f},{hi:+.2f}]，{len(ns)} 节点（错题 ≥ {need}）{S2.mark(ok)}  " + " ".join(f"{a} {W.Bn(R, a, set(ids), p1):.2f}−{W.Bn(R, a, set(ids), p2):.2f}({bad[a]})" for a in ns))
    return ok


def block(R, ids, nodes, lab, judge=True):
    print(f"\n[{lab}] 节点 {len(nodes)} 题 {len(ids)}")
    if len(nodes) < 2 or not ids:
        print("  节点或题不够")
        nan = (float("nan"),) * 3 + (0,)
        return None, (False, False, False), nan, nan
    T = {(r["a"], r["id"], r["p"]): r for r in R}
    cand = ["set2", "cond2", "L3", "L3c"]
    vs = {c: m(x for x in (v0(R, a, set(ids), c) for a in nodes) if x == x) for c in cand}
    print("  v* 均值（修正定义）" + "  ".join(f"{c} {vs[c]:.2f}" for c in cand))
    tag = "S" if judge else "报告"
    s1 = dv(R, ids, nodes, "L3", "cond2", f"{tag}1")
    s2 = dv(R, ids, nodes, "L3c", "L3", f"{tag}2")
    need = max(2, -(-2 * len(nodes) // 3))
    s3 = W.w1(T, ids, nodes, "cond2f", "set2f", f"{tag}3 出口 跳出", need=need)
    s4 = w2(R, ids, nodes, "cond2t", "set2t", f"{tag}4 出口 救回")
    s5 = dv(R, ids, nodes, "cond2", "set2", f"{tag}5")
    W.w1(T, ids, nodes, "L3f", "cond2f", "  组件 前提 vs 集合+出口 跳出", need=need)
    w2(R, ids, nodes, "L3t", "cond2t", "  组件 前提 vs 集合+出口 救回")
    W.w1(T, ids, nodes, "L3cf", "L3f", "  组件 出口对前提 跳出", need=need)
    w2(R, ids, nodes, "L3ct", "L3t", "  组件 出口对前提 救回")
    best = min(cand, key=lambda c: vs[c] if vs[c] == vs[c] else 9)
    wins = all(dv(R, ids, nodes, best, o, f"  {best} vs {o}")[2] < 0 for o in cand if o != best)
    print(f"  原定义 v*：" + "  ".join(f"{c} {m(x for x in (W.vstar(R, a, set(ids), c) for a in nodes) if x == x):.2f}" for c in cand))
    return (best if wins else None), (s3, s4, s5[2] < 0), s1, s2


def analyze():
    R, tools = W.load()
    R = [r for r in R if r["p"] in FORMS]
    items = {x["id"]: x for x in json.load(open(ITEMS))}
    json.dump(R, open(os.path.join(HERE, "rows_s.json"), "w"))
    n_items = len({r["id"] for r in R})
    print(f"记录 {len(R)}；题 {n_items}；工具 {({k: round(v, 3) for k, v in tools.items()})}")
    print("\n预检")
    good = []
    for a in NODES:
        X_ = [r for r in R if r["a"] == a]
        nr = sum(r["right"] for r in X_ if r["p"] == "blank")
        nw = sum(not r["right"] for r in X_ if r["p"] == "blank")
        k0 = W.K(X_, p="blank")
        cover = len(X_) / max(n_items * len(FORMS), 1)
        ok = k0 >= .85 and nr >= 30 and nw >= 15
        good += [a] if ok else []
        print(f"  {a:12s} 覆盖 {cover:.0%} 第一次对 {nr} 错 {nw}  K(blank) {k0:.0%} {S2.mark(ok)}  " + " ".join(f"{p} K {W.K(X_, p=p):.0%}/B {W.B(X_, p=p):.0%}" for p in FORMS[1:]))
    print(f"可用节点 {len(good)}（≥ 5）{S2.mark(len(good) >= 5)}")
    ids = sorted({r["id"] for r in R})
    best, (s3, s4, s5), s1, s2 = block(R, ids, good, "全部")
    print("\n决定")
    enough = len(good) >= 5
    print(f"  推荐表达式：{best or '分不出'}" + ("" if enough else "（可用节点不够）"))
    print(f"  S1 前提优于集合+出口：{S2.mark(enough and s1[2] < 0)}")
    print(f"  S2 出口对前提：{'有用' if s2[2] < 0 else '有害' if s2[1] > 0 else '无增益'}")
    print(f"  出口结论在扩题上复现（S3 S4 S5）：{S2.mark(enough and s3 and s4 and s5)}")
    print(f"  出口有代价（S3 过 S4 不过）：{S2.mark(enough and s3 and not s4)}")
    print(f"  出口结论不复现（S3 不过）：{S2.mark(enough and not s3)}")
    print("\n另报不判")
    block(R, ids, [a for a in good if a not in LOCAL], "只看 API 节点", judge=False)
    block(R, ids, [a for a in good if a in LOCAL], "只看本地开源节点", judge=False)
    for t in sorted({x["task"] for x in items.values()}):
        block(R, [i for i in ids if items[i]["task"] == t], good, f"任务 {t}", judge=False)


if __name__ == "__main__":
    {"pool": pool, "screen": screen, "gen": gen, "run": run_all, "a": analyze}[sys.argv[1]]()
