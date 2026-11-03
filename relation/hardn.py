"""纠错预测（2026-11-03，库伯同意）：专门测"纠正坏节点"，不加新公式。

上一轮（newnodes.py，eb8937de）留下的缺口：大模型错题太少（Kimi、Grok 各 4 道），错行预测很差，两条轴是看过结果才命名的。
这一轮先把轴和坐标冻结进 axes.json 并提交，再碰难题。

冻结的东西：
  PCA2 = 15 个旧节点的 14 维 logit 的前 2 主成分（均值、载荷、尺度）。
  第一条轴 z1 = "吃形式"：不分真假，采纳集合约束。第二条轴 z2 = "分真假"：真投放采纳、假投放不动。
  每个节点的坐标只来自已有校准数据：7 个新节点用 newnodes 的校准题，4 个旧 LLM（ds_flash_a ds_pro_a claude_haiku codex_ns）用 generality 的校准题。旧节点进过 PCA，所以另报"只看 7 个新节点"。
  qwen14 上一轮因题号缺 Q 被剔除；这一轮解析时统一补全题号，算一个声明过的修复。

题：raw 里从没进过任何投放实验的题（同四类任务；部分旧 LLM 在早期筛选时答过其中一些，但没见过投放）。Sonnet、Kimi、Grok 各答一次（筛选），留下 ≥ 1 个答错的题。
  若有大模型筛选错题 < 50，就加同任务类型的 MMLU 科目（EXTRA，放进 raw2/）再筛一次；筛完不再加。
  答案键错误会把"抗拒错误投放"算成"救不回来"，所以每道候选题由 DeepSeek Pro 和 Codex（可联网）各审一次答案键：
    两个都说键错 → 剔除；只有一个说错 → 保留并标记，判据另报一遍"去掉标记题"。
  轮流按大模型抽题（各自错题优先），最多 150 题。所有节点重新答第一次，错节点按这次算（避免筛选带来的回归均值）。
  投放由 Sonnet 用第二阶段同一提示词写。形式 FORMS = blank L3t L3f L4t L4f set2t set2f set2t_r set2f_r。
节点：7 个新节点 + 4 个旧 LLM，共 11 个。第一次答案覆盖 < 80% 剔除；可用节点 < 8 没结论。

判据（在难题上，坐标全部用冻结值）：
  M0 难度：Sonnet、Kimi、Grok 第一次错误率各 ≥ 20% 且错题 ≥ 25；少于 2 个达标就算"大模型部分没结论"。
  H1 分真假轴预测纠错选择性：D_w = 错节点上（L3、L4 平均）采纳真目标 − 采纳假目标（只算冲突行）；错题 ≥ 15 的节点上 Spearman(z2, D_w) ≥ 0.6，且比 |Spearman(z1, D_w)| 高 ≥ 0.2。
  H2 吃形式轴预测形式服从：A_set = 四种集合投放在冲突行的平均采纳率；Spearman(z1, A_set) ≥ 0.5，且比 |Spearman(z2, A_set)| 高 ≥ 0.2。
  H3 关键格：B(L3t) B(L4t) B(set2t_r) 和 K(L3f) K(L4f) K(set2f_r)（n ≥ 15）；PCA2 的 B 与 K 平均绝对误差各 ≤ 0.15，且各低于共用（z = 0）。
  H4 错行预测（上一轮最弱处）：第一次答错、带投放的行上，PCA2 的 log-loss 比 共用、错校准（别的节点坐标的平均预测）在 ≥ 2/3 节点上好且全节点差区间 < 0；并且比均匀猜（ln 4）差区间 < 0。
决定：H1 和 H2 都过，才说这两条轴是统一方向的候选；另外要求"只看 LLM（去掉 flash_m65、kw_rule）"两个相关同号。H3、H4 都过，才说它能预测纠错。
  H1 不过 → 撤回"分真假轴能预测纠错"。任何情况都不补公式。
另报不判：去掉 Sonnet（它写投放又当接收者）、只看 7 个新节点、去掉标记题；每个节点逐形式 K、B；留一节点的相关范围；同一批题上 set2f 与 set2f_r 的配对差（上一轮两者不同题）。
停止规则：DeepSeek 上限 ¥10，到了就停 DeepSeek 节点。多重比较不校正，全部报告。

用法：~/gut-external/torch215-venv/bin/python hardn.py freeze | screen | referee | select | gen | run | analyze
"""
import concurrent.futures as cf
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import urllib.request

import torch

import analyze as AN
import formula as FM
import generality as G
import newnodes as N
import run as RUN
import stage2 as S2

HERE = RUN.HERE
AXES = os.path.join(HERE, "axes.json")
OUT = os.path.join(HERE, "runs_h")
SCRN = os.path.join(HERE, "screen_h")
RAW2 = os.path.join(HERE, "raw2")
POOL = os.path.join(HERE, "poolH.json")
REF = os.path.join(HERE, "refereeH.json")
ITEMS = os.path.join(HERE, "itemsH.json")
PAY = os.path.join(HERE, "payloadsH.json")
BIG = ["claude_sonnet", "kimi", "grok"]
OLD = ["ds_flash_a", "ds_pro_a", "claude_haiku", "codex_ns"]
LLMS = BIG + ["qwen14", "qwen9"] + OLD
NODES = LLMS + ["flash_m65", "kw_rule"]
NEW7 = BIG + ["qwen14", "qwen9", "flash_m65", "kw_rule"]
FORMS = ["blank", "L3t", "L3f", "L4t", "L4f", "set2t", "set2f", "set2t_r", "set2f_r"]
EXTRA = {"recall": ["prehistory", "high_school_us_history", "high_school_world_history", "astronomy"], "reason": ["abstract_algebra", "college_mathematics", "econometrics", "high_school_statistics"], "case": ["jurisprudence", "international_law", "professional_medicine", "college_medicine"]}
CAP = 150
LIMIT = 10.0
PAR = {"qwen14": 1, "qwen9": 1, "ds_flash_a": 8, "ds_pro_a": 8}
_start = [None]
_lock = threading.Lock()
_n = [0]
_stop = [False]


def fixkeys(rec, ids):
    return {("Q" + k if k not in ids and "Q" + k in ids else k): v for k, v in rec.items()}


def pca(z, d, A):
    return torch.softmax(G.basis(d) @ (A["m0"] + z @ A["V"]), 1)


def map_z(dc, m0, V, s):
    Pc = G.basis(dc)
    z = torch.zeros(len(V), requires_grad=True)
    opt = torch.optim.LBFGS([z], max_iter=200, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(Pc @ (m0 + z @ V), dc["y"]) + .5 * (z ** 2 / s).sum() / len(dc["y"])
        l.backward()
        return l
    opt.step(cl)
    return z.detach()


def freeze():
    torch.set_num_threads(4)
    fx = os.path.join(N.SCR, "fixed_h")
    shutil.rmtree(fx, ignore_errors=True)
    os.makedirs(fx)
    ids = {x["id"] for x in json.load(open(N.ITEMS))}
    for fn in os.listdir(N.OUT):
        src = os.path.join(N.OUT, fn)
        if fn.endswith("_qwen14.json"):
            J = json.load(open(src))
            J["answers"] = fixkeys(J["answers"], ids)
            json.dump(J, open(os.path.join(fx, fn), "w"))
        else:
            os.symlink(src, os.path.join(fx, fn))
    N.OUT = fx
    R, cover, _, items = N.load()
    G.items.update(items)
    FM.items.update(items)
    B = torch.stack([G.fit_lin(G.basis(G.tens(G.BY[n])), G.tens(G.BY[n])["y"]) for n in G.NODES])
    m0 = B.mean(0)
    _, Sv, Vh = torch.linalg.svd(B - m0, full_matrices=False)
    V, s = Vh[:2], Sv[:2] ** 2 / (len(G.NODES) - 1) + .1
    z = {}
    for a in NEW7:
        z[a] = map_z(G.tens([r for r in R if r["a"] == a and r["cal"] and r["p"] in G.CAL_P]), m0, V, s).tolist()
    for a in OLD:
        z[a] = map_z(G.tens(G.split(a)[0]), m0, V, s).tolist()
    json.dump(dict(m0=m0.tolist(), V=V.tolist(), z=z, cover={a: round(cover[a], 3) for a in NEW7}), open(AXES, "w"), indent=0)
    for a, v in sorted(z.items(), key=lambda kv: kv[1][1]):
        print(f"{a:14s} z1 {v[0]:+6.2f} z2 {v[1]:+6.2f}")


def pool_rows(raws):
    used = set()
    for f in ("items.json", "items2.json", "itemsN.json", "pool3.json", "items3.json"):
        if os.path.exists(os.path.join(HERE, f)):
            used |= {x["q"] for x in json.load(open(os.path.join(HERE, f))) if isinstance(x, dict) and "q" in x}
    out = []
    for task, name, path in raws:
        for r in json.load(open(path)):
            opts = r.get("choices") or r.get("options")
            if not opts or len(opts) != 4 or any(RUN.BAD_OPT.search(o) for o in opts) or len(r["question"]) > 1200 or len(r.get("article", "")) > 2500 or r["question"] in used:
                continue
            used.add(r["question"])
            out.append(dict(task=task, source=name, q=r["question"], opts=opts, gold=r["answer"] if isinstance(r["answer"], int) else RUN.LET.index(r["answer"]), article=r.get("article", "")))
    return out


def screen_round(pool):
    os.makedirs(SCRN, exist_ok=True)
    os.makedirs(N.SCR, exist_ok=True)
    open(N.AGENT, "w").write("---\nname: plain\ndescription: answer directly with no tools\ntools: []\n---\nYou answer questions directly from what you know. You have no tools.\n")
    ids = {k["id"] for k in pool}

    def one(j):
        node, bid, g = j
        rec = N.call(node, "Answer each question.\n\n" + "\n\n".join(RUN.item_text(it) for it in g) + "\n\n" + RUN.TAIL, os.path.join(SCRN, f"{bid}_{node}.json"))
        return node, fixkeys(rec.get("answers", {}), ids)
    jobs = [(node, b, g) for node in BIG for b, g in RUN.batches(pool)]
    out = {n: {} for n in BIG}
    with cf.ThreadPoolExecutor(12) as ex:
        for node, a in ex.map(one, jobs):
            out[node].update(a)
    return out


def errs(pool, ans):
    gold = {k["id"]: k["gold"] for k in pool}
    return {n: {q for q, v in a.items() if q in gold and isinstance(v, dict) and AN.letter(v) is not None and AN.letter(v) != gold[q]} for n, a in ans.items()}


def screen():
    raws = [(t, n, os.path.join(HERE, "raw", n + ".json")) for t, ns in RUN.POOLS.items() for n in ns]
    pool = pool_rows(raws)
    random.Random("rel-hardn").shuffle(pool)
    for i, k in enumerate(pool):
        k["id"] = f"Q{5000 + i}"
    ans = screen_round(pool)
    e = errs(pool, ans)
    print("第一轮池", len(pool), {n: len(v) for n, v in e.items()}, {n: len(a) for n, a in ans.items()}, flush=True)
    if min(len(v) for v in e.values()) < 50:
        os.makedirs(RAW2, exist_ok=True)
        extra = []
        for t, names in EXTRA.items():
            for name in names:
                fn = os.path.join(RAW2, name + ".json")
                if not os.path.exists(fn):
                    rows = []
                    for off in range(0, 400, 100):
                        u = f"https://datasets-server.huggingface.co/rows?dataset=cais/mmlu&config={name}&split=test&offset={off}&length=100"
                        d = json.load(urllib.request.urlopen(u, timeout=60))
                        got = [x["row"] for x in d["rows"]]
                        rows += got
                        if len(got) < 100:
                            break
                    json.dump(rows, open(fn, "w"))
                extra.append((t, name, fn))
        more = pool_rows(raws + extra)
        known = {k["q"] for k in pool}
        more = [k for k in more if k["q"] not in known]
        random.Random("rel-hardn2").shuffle(more)
        for i, k in enumerate(more):
            k["id"] = f"Q{7000 + i}"
        a2 = screen_round(more)
        for n in BIG:
            ans[n].update(a2[n])
        pool += more
        e = errs(pool, ans)
        print("加科目后池", len(pool), {n: len(v) for n, v in e.items()}, flush=True)
    cand = set().union(*e.values())
    json.dump(dict(pool=pool, errs={n: sorted(v) for n, v in e.items()}, cand=sorted(cand)), open(POOL, "w"), ensure_ascii=False)
    print("候选", len(cand))


def referee():
    P = json.load(open(POOL))
    by = {k["id"]: k for k in P["pool"]}
    have = json.load(open(REF)) if os.path.exists(REF) else {}
    q = sorted(P["cand"])
    groups = [q[i:i + 3] for i in range(0, len(q), 3)]

    def one(j):
        node, g = j
        fn = os.path.join(SCRN, f"ref_{node}_{g[0]}.json")
        if os.path.exists(fn):
            return node, json.load(open(fn))
        txt = ("You are auditing the answer key of an exam. For each question below, decide whether the official key is correct. Check carefully and independently. "
               'Reply with ONLY a JSON object mapping each question ID to {"verdict": "correct" | "wrong" | "unsure"}.\n\n' +
               "\n\n".join(RUN.item_text(by[i]) + f"\nOfficial key: {RUN.LET[by[i]['gold']]}" for i in g))
        for _ in range(3):
            try:
                raw = RUN.ask_codex("codex", txt)[0] if node == "codex" else RUN.C.ask("ds_pro_a", txt, 6000, False)
                d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                json.dump(d, open(fn, "w"))
                return node, d
            except Exception:
                continue
        return node, {}
    with cf.ThreadPoolExecutor(8) as ex:
        for node, d in ex.map(one, [(n, g) for n in ("codex", "ds_pro") for g in groups]):
            for k, v in d.items():
                have.setdefault(k, {})[node] = v.get("verdict") if isinstance(v, dict) else v
    json.dump(have, open(REF, "w"), indent=0)
    from collections import Counter
    print(Counter(tuple(sorted(v.items())) for v in have.values()).most_common(12))


def select():
    P = json.load(open(POOL))
    by = {k["id"]: k for k in P["pool"]}
    ref = json.load(open(REF))
    both = {q for q in P["cand"] if ref.get(q, {}).get("codex") == "wrong" and ref.get(q, {}).get("ds_pro") == "wrong"}
    one = {q for q in P["cand"] if q not in both and "wrong" in ref.get(q, {}).values()}
    g = random.Random("rel-hsel")
    rest = {n: [q for q in sorted(P["errs"][n]) if q not in both] for n in BIG}
    for v in rest.values():
        g.shuffle(v)
    chosen = []
    while len(chosen) < CAP and any(rest.values()):
        for n in BIG:
            while rest[n] and rest[n][0] in chosen:
                rest[n].pop(0)
            if rest[n] and len(chosen) < CAP:
                chosen.append(rest[n].pop(0))
    items = [dict(by[q], hard=True, cal=False, flagged=q in one) for q in chosen]
    json.dump(items, open(ITEMS, "w"), ensure_ascii=False, indent=0)
    from collections import Counter
    print(f"候选 {len(P['cand'])} 两审都说键错剔除 {len(both)} 标记 {len(one)} 入选 {len(items)}（标记 {sum(i['flagged'] for i in items)}）")
    print(Counter(i["task"] for i in items), Counter(i["source"] for i in items))
    print({n: sum(q in chosen for q in P["errs"][n]) for n in BIG})


def gen():
    S2.ITEMS2, S2.PAY2 = ITEMS, PAY
    S2.gen()


def spent():
    return _start[0] - RUN.balance()


def call(node, text, path, ids):
    if os.path.exists(path):
        J = json.load(open(path))
        return J
    if node.startswith("ds_"):
        with _lock:
            _n[0] += 1
            check = _n[0] % 40 == 0
        if check and spent() > LIMIT:
            _stop[0] = True
        if _stop[0]:
            return {"answers": {}}
    r = RUN.call(node, text, path) if node in OLD else N.call(node, text, path)
    fx = fixkeys(r.get("answers", {}), ids)
    if fx != r.get("answers") and os.path.exists(path):
        r["answers"] = fx
        json.dump(r, open(path, "w"), ensure_ascii=False)
    r["answers"] = fx
    return r


def run_node(node, items, pay):
    ids = {x["id"] for x in items}
    first = {}
    par = PAR.get(node, 4)
    bs = RUN.batches(items)

    def do_first(j):
        bid, g = j
        return call(node, "Answer each question.\n\n" + "\n\n".join(RUN.item_text(it) for it in g) + "\n\n" + RUN.TAIL, os.path.join(OUT, f"s1_{bid}_{node}.json"), ids)
    with cf.ThreadPoolExecutor(par) as ex:
        for r in ex.map(do_first, bs):
            first.update(r["answers"])

    def do_second(j):
        bid, g, p = j
        parts = []
        for it in g:
            e = first.get(it["id"], {}) if isinstance(first.get(it["id"], {}), dict) else {}
            n = S2.note(it, p, pay, items)
            parts.append(RUN.item_text(it) + f"\nYour earlier answer: {e.get('answer', '?')} (confidence {e.get('confidence', '?')}). Your earlier reason: {e.get('reason', '')}" + (f"\nA note shared with the whole team: {n}" if n else ""))
        return call(node, "Here are questions you answered earlier. Give your answer again.\n\n" + "\n\n".join(parts) + "\n\n" + RUN.TAIL, os.path.join(OUT, f"{p}_{bid}_{node}.json"), ids)
    jobs = [(b, g, p) for b, g in bs for p in FORMS]
    random.Random(node).shuffle(jobs)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(do_second, jobs))
    print(node, "完成", flush=True)


def run_all():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(N.SCR, exist_ok=True)
    N.OUT = OUT
    N.CAL_P, N.TEST_P = FORMS, FORMS
    items = json.load(open(ITEMS))
    pay = json.load(open(PAY))
    _start[0] = RUN.balance()
    N.run_kw(items, pay)
    N.run_m65(items, pay)
    with cf.ThreadPoolExecutor(len(LLMS)) as ex:
        list(ex.map(lambda n: run_node(n, items, pay), LLMS))
    print(f"DeepSeek ¥{spent():.2f}", flush=True)
    print("全部完成", flush=True)


def analyze():
    torch.set_num_threads(4)
    A = json.load(open(AXES))
    A = dict(m0=torch.tensor(A["m0"]), V=torch.tensor(A["V"]), z={k: torch.tensor(v) for k, v in A["z"].items()})
    N.OUT, N.ITEMS, N.LLMS = OUT, ITEMS, LLMS
    R, cover, tools, items = N.load()
    G.items.update(items)
    FM.items.update(items)
    json.dump(R, open(os.path.join(HERE, "rows_h.json"), "w"))
    nodes = [a for a in NODES if cover.get(a, 0) >= .8]
    print("覆盖", {k: round(v, 2) for k, v in cover.items()}, "工具", {k: round(v, 3) for k, v in tools.items()})
    print(f"可用节点 {len(nodes)}（≥ 8）；题 {len(items)}，标记 {sum(i.get('flagged', False) for i in items.values())}")
    m = lambda xs: (lambda v: sum(v) / len(v) if v else float("nan"))(list(xs))
    print("\nM0 第一次错误率")
    okbig = 0
    for a in nodes:
        rs = [r for r in R if r["a"] == a and r["p"] == "blank"]
        w = sum(not r["right"] for r in rs)
        if a in BIG:
            okbig += (w / len(rs) >= .2 and w >= 25)
        print(f"  {a:14s} 错 {w:3d}/{len(rs)} = {w / max(len(rs), 1):.0%}  blank K {S2.K(rs, 'blank'):.0%} B {S2.B(rs, 'blank'):.0%}")
    print(f"M0 大模型达标 {okbig}/3 {S2.mark(okbig >= 2)}")

    def stats(rs, a):
        X = [r for r in rs if r["a"] == a]
        cw = lambda p: m(r["adopt"] for r in X if r["p"] == p and not r["right"] and r["conflict"])
        dw = (cw("L3t") - cw("L3f") + cw("L4t") - cw("L4f")) / 2
        aset = m(m(r["adopt"] for r in X if r["p"] == p and r["conflict"]) for p in ("set2t", "set2f", "set2t_r", "set2f_r"))
        nw = sum(not r["right"] for r in X if r["p"] == "blank")
        return dw, aset, nw

    def test(rs, ns, lab, loo=False):
        tb = {a: stats(rs, a) for a in ns}
        h1n = [a for a in ns if tb[a][2] >= 15 and not math.isnan(tb[a][0])]
        z = lambda a, i: A["z"][a][i].item()
        r21 = S2.spearman([z(a, 1) for a in h1n], [tb[a][0] for a in h1n])
        r11 = S2.spearman([z(a, 0) for a in h1n], [tb[a][0] for a in h1n])
        r12 = S2.spearman([z(a, 0) for a in ns], [tb[a][1] for a in ns])
        r22 = S2.spearman([z(a, 1) for a in ns], [tb[a][1] for a in ns])
        h1, h2 = r21 >= .6 and r21 - abs(r11) >= .2, r12 >= .5 and r12 - abs(r22) >= .2
        print(f"[{lab}] H1 ρ(z2,D_w) {r21:+.2f} ρ(z1,D_w) {r11:+.2f}（{len(h1n)} 节点）{S2.mark(h1)}；H2 ρ(z1,A_set) {r12:+.2f} ρ(z2,A_set) {r22:+.2f}（{len(ns)} 节点）{S2.mark(h2)}")
        if loo:
            a1 = [S2.spearman([z(a, 1) for a in h1n if a != o], [tb[a][0] for a in h1n if a != o]) for o in h1n]
            a2 = [S2.spearman([z(a, 0) for a in ns if a != o], [tb[a][1] for a in ns if a != o]) for o in ns]
            print(f"  留一节点范围 H1 {min(a1):+.2f}…{max(a1):+.2f}；H2 {min(a2):+.2f}…{max(a2):+.2f}")
        return h1, h2, r21, r12

    print()
    h1, h2, _, _ = test(R, nodes, "全部", True)
    llm = [a for a in nodes if a in LLMS]
    _, _, r21l, r12l = test(R, llm, "只看 LLM")
    test(R, [a for a in nodes if a != "claude_sonnet"], "去掉 Sonnet")
    test(R, [a for a in nodes if a in NEW7], "只看 7 个新节点")
    test([r for r in R if not items[r["id"]].get("flagged")], nodes, "去掉标记题")

    def pred(a, rs):
        d = G.tens(rs)
        return pca(A["z"][a], d, A), pca(torch.zeros(2), d, A), sum(pca(A["z"][o], d, A) for o in nodes if o != a) / (len(nodes) - 1), d
    print("\nH3 关键格（观测/PCA2/共用）")
    errB, errK, errB0, errK0 = [], [], [], []
    for a in nodes:
        s = []
        for p, st_ in (("L3t", False), ("L4t", False), ("set2t_r", False), ("L3f", True), ("L4f", True), ("set2f_r", True)):
            rs = [r for r in R if r["a"] == a and r["p"] == p and r["right"] == st_]
            if len(rs) < 15:
                continue
            q, q0, _, _ = pred(a, rs)
            ob = m(r["ok"] for r in rs)
            pr = m(q[i, r["g"]].item() for i, r in enumerate(rs))
            pr0 = m(q0[i, r["g"]].item() for i, r in enumerate(rs))
            (errK if st_ else errB).append(abs(ob - pr))
            (errK0 if st_ else errB0).append(abs(ob - pr0))
            s.append(f"{'K' if st_ else 'B'}({p}) {ob:.2f}/{pr:.2f}/{pr0:.2f}")
        print(f"  {a:14s} " + " ".join(s))
    mb, mk, mb0, mk0 = m(errB), m(errK), m(errB0), m(errK0)
    h3 = mb <= .15 and mk <= .15 and mb < mb0 and mk < mk0
    print(f"H3 B 误差 {mb:.3f}（共用 {mb0:.3f}）K 误差 {mk:.3f}（共用 {mk0:.3f}）{S2.mark(h3)}")

    print("\nH4 错行 log-loss（第一次答错、带投放）")
    per = {}
    for a in nodes:
        rs = [r for r in R if r["a"] == a and r["p"] != "blank" and not r["right"]]
        if len(rs) < 15:
            continue
        q, q0, qo, d = pred(a, rs)
        ix = torch.arange(len(rs))
        nl = lambda x: (-x.clamp(min=1e-6).log()[ix, d["y"]]).detach()
        per[a] = (rs, nl(q), nl(q0), nl(qo))
        print(f"  {a:14s} n={len(rs):4d} PCA2 {per[a][1].mean():.3f} 共用 {per[a][2].mean():.3f} 错校准 {per[a][3].mean():.3f}")

    def cmp(j):
        w = sum(per[a][1].mean() < per[a][j].mean() for a in per) if j else None
        by = {}
        for a, t in per.items():
            for k, r in enumerate(t[0]):
                by.setdefault(r["id"], []).append((t[1][k] - (t[j][k] if j else math.log(4))).item())
        I = sorted(by)
        g = random.Random(3)
        bs = []
        for _ in range(1000):
            v = [x for i in g.choices(I, k=len(I)) for x in by[i]]
            bs.append(sum(v) / len(v))
        bs.sort()
        return w, sum(x for i in I for x in by[i]) / sum(len(by[i]) for i in I), bs[25], bs[974]
    w2, d2, l2, h2_ = cmp(2)
    w3, d3, l3, h3_ = cmp(3)
    _, d0, l0, h0 = cmp(0)
    n = len(per)
    h4 = w2 * 3 >= 2 * n and h2_ < 0 and w3 * 3 >= 2 * n and h3_ < 0 and h0 < 0
    print(f"H4 对共用 赢 {w2}/{n} {d2:+.3f} [{l2:+.3f},{h2_:+.3f}]；对错校准 赢 {w3}/{n} {d3:+.3f} [{l3:+.3f},{h3_:+.3f}]；对均匀 {d0:+.3f} [{l0:+.3f},{h0:+.3f}] {S2.mark(h4)}")

    print("\n决定")
    print(f"  统一方向候选（H1、H2 过，LLM 同号）：{S2.mark(h1 and h2 and r21l > 0 and r12l > 0)}")
    print(f"  能预测纠错（H3、H4 过）：{S2.mark(h3 and h4)}")
    print("\n各节点 K/B（括号内为第一次对/错的行数）")
    for a in nodes:
        X = [r for r in R if r["a"] == a]
        print(f"  {a:14s} K " + " ".join(f"{p} {S2.K(X, p):4.0%}" for p in ("blank", "L3f", "L4f", "set2f", "set2f_r")) + f" ｜ B " + " ".join(f"{p} {S2.B(X, p):4.0%}" for p in ("blank", "L3t", "L4t", "set2t", "set2t_r")) + f" （{sum(r['right'] for r in X if r['p'] == 'blank')}/{sum(not r['right'] for r in X if r['p'] == 'blank')}）")
    print("\n同题配对：K(set2f_r) − K(set2f)")
    for a in nodes:
        X = {(r["id"], r["p"]): r for r in R if r["a"] == a and r["right"]}
        d = [X[(i, "set2f_r")]["ok"] - X[(i, "set2f")]["ok"] for i, p in X if p == "set2f" and (i, "set2f_r") in X]
        print(f"  {a:14s} {m(d):+.2f} n={len(d)}")


if __name__ == "__main__":
    {"freeze": freeze, "screen": screen, "referee": referee, "select": select, "gen": gen, "run": run_all, "analyze": analyze}[sys.argv[1]]()
