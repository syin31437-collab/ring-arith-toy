"""题型对照（2026-11-04，库伯同意）：同一批题、同一批节点、同样的假集合，只改题型，看杀伤力差在哪。

来由：cross.py 的探索发现 K(set2f) 难选择题 → 开放题 Sonnet 16→94、Kimi 0→62，但两边题不同。核对提示词后发现旧选择题和开放题还差四样：有没有选项、批量还是单题、只回 JSON 还是三句话、投放用字母还是文字。所以拆成三个条件，每次只动一组。

条件（同一题在三个条件下都答）：
  J 旧格式：有选项，按旧流程成批，只回 JSON，投放用字母（"the correct answer is one of: A, C."），即 hardn.py 的提示词。
  M 选项 + 文字：有选项，单题，三句话，投放用选项原文（"the correct answer is one of: 36%; 81%."）。
  O 开放：没有选项，其余和 M 一字不差（同尾句、同投放文字、同顺序）。
  O − M 只差选项；M − J 是回答格式、成批、字母三样一起（这一轮不再细拆）。
题：只留不看选项也能答的题（没有 following / above / statement / option 等词，没有长文章）。
  H 层：itemsH.json 里符合的 63 题随机 40（发现这个现象的那批，选的是大模型答错过的题）。
  N 层：同来源、同过滤、从没进过任何实验和筛选池的题 40（不挑难度；来源不够时用同任务类的其他来源）。
节点：Sonnet Kimi Grok ds_flash_a ds_pro_a claude_haiku codex_ns 共 7 个。去掉两个本地 qwen（太慢）和遮题面的 Flash（只对 2 题，上一轮进不了相关）。
投放：blank set2f set2t L3f（L3f 是对照：同样是假信息，但不是集合形式；同一句话进 J M O）。第一次答案在每个条件下分别答。
判分：J 按字母；O、M 同一裁判（Sonnet）、不知道投放和接收者、看得到选项（M）或看不到（O）。

指标：杀伤 d = K(blank) − K(p)，同节点同题。上一轮案例题 blank K 只有 71%，所以不看 K 绝对值，减掉 blank。
  配对差：节点 a 在条件 c1、c2 下都第一次答对的题，算 d(c1) − d(c2)；节点要 ≥ 5 题；7 个节点里可用 < 5 个算没结论。总量 = 节点均值；区间按题重采样 2000 次，95%。

映射预检（看不到投放效果）：裁判 Sonnet vs Codex 在 O、M 各复评 100 条 kappa ≥ 0.7；每个条件 × 层第一次错误率 5–85%；每个条件 blank K ≥ 0.70（放宽理由见上）。不过的条件涉及的判据记为没结论。

判据（全部在 H 层，除非写明）：
  P0 复现：J × H，Sonnet Kimi Grok Haiku Codex 平均 K(set2f) ≤ 0.50；且 J × N 比 J × H 高 ≥ 0.20。不过就说旧现象在这批没复现，后面的判据只报告。
  P1 选项：d(M, set2f) − d(O, set2f) ≥ 0.15，区间下界 > 0，≥ 5/7 节点 > 0。
  P2 回答格式 + 成批 + 字母：d(J, set2f) − d(M, set2f) ≥ 0.15，区间下界 > 0，≥ 5/7 节点 > 0。
  P3 难度：O 条件下 d(H, set2f) − d(N, set2f) ≥ 0.15，区间下界 > 0（题不同，按层分别重采样）。
  P4 集合特异：[d(J, set2f) − d(O, set2f)] − [d(J, L3f) − d(O, L3f)] ≥ 0.10，区间下界 > 0。
  P5 不是单纯更难被改动：分辨力 S = B(set2t) + K(set2f) − 1，节点平均；S(O) − S(J) ≥ 0.10，区间下界 > 0（按题重采样）。注意 J、M 上 B(set2t) 有二选一的便宜，这个判据对 O 不利。
决定（写死）：
  P0 过且 P1 过 → 选项本身是原因之一；P2 过 → 回答格式 / 成批 / 字母是原因之一（要再拆才知道是哪样）；P3 过 → 难度在开放题上也起作用；
  P1、P2 都不过且 P3 过 → 上一轮的差别主要来自题不同，"假集合杀伤是选择题现象"收回。
  P5 不过 → 开放题只是更难被改动，不能说分辨力更好。
范围：题全部来自选择题库（法律、医学、事实等），结论不外推到代码、排班。不改轴、不加公式、不据此写任何投放表达式。
另报不判：N 层同样的比较；每个节点的 K、B 表；去掉标记题；去掉带工具的回复；去掉 Sonnet；旧 runs_h 的 K(set2f) 和新 J × H 的对比。
预告的局限：Grok 的 CLI 留着 read_file，Codex 有只读沙盒（都禁联网），记录次数；N 层没有审答案键；裁判只能判结论，判不了理由质量。
停止规则：DeepSeek 上限 ¥10，每 30 次调用查一次余额；到线就停 DeepSeek，缺的行照实报告。多重比较不校正，全部报告。

用法：~/gut-external/torch215-venv/bin/python formtest.py build | gen | run | a
"""
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading

import cross as X
import generality as G
import hardn as H
import formula as FM
import newnodes as N
import run as RUN
import stage2 as S2
import stage3 as S3

HERE = RUN.HERE
OUT = os.path.join(HERE, "runs_f")
OUTJ = os.path.join(HERE, "runs_fj")
BASE = os.path.join(HERE, "itemsF.json")
ITEMS = os.path.join(HERE, "itemsF_om.json")
PAY = os.path.join(HERE, "payloadsF.json")
GRADE = os.path.join(HERE, "gradesF.json")
KAPPA = os.path.join(HERE, "kappaF.json")
NODES = ["claude_sonnet", "kimi", "grok", "ds_flash_a", "ds_pro_a", "claude_haiku", "codex_ns"]
FRONT = ["claude_sonnet", "kimi", "grok", "claude_haiku", "codex_ns"]
FORMS = ["blank", "set2f", "set2t", "L3f"]
PER = 40
LIMIT = 10.0
PAR = {"ds_flash_a": 8, "ds_pro_a": 8, "codex_ns": 4}
OPENABLE = re.compile(r"\b(following|above|below|which of these|statements?|options?|choices?)\b", re.I)

S3.OUT, S3.AGENTS, S3.PAYLOADS, S3.ITEMS, S3.PAY, S3.GRADE = OUT, NODES, FORMS, ITEMS, PAY, GRADE
H.OUT, H.FORMS, H.LIMIT, H.ITEMS, H.PAY = OUTJ, FORMS, LIMIT, BASE, PAY
_note = S3.note
S3.note = lambda it, p, pay, items: _note(dict(it, id=it["base"]), p, pay, items)
_plain = S3.plain
_n = [0]
_lock = threading.Lock()


def plain(node, text):
    """S3 路径的 DeepSeek 也走同一条上限。"""
    if node.startswith("ds_"):
        with _lock:
            _n[0] += 1
            check = _n[0] % 30 == 0
        try:
            if check and H.spent() > LIMIT:
                H._stop[0] = True
        except OSError:
            pass
    return _plain(node, text)


S3.plain = plain


def ok_item(q, opts, article=""):
    return not OPENABLE.search(q) and not article and len(opts) == 4 and len(q) <= 1200 and not any(RUN.BAD_OPT.search(o) for o in opts)


def path_of(name):
    for d in ("raw", "raw2"):
        p = os.path.join(HERE, d, name + ".json")
        if os.path.exists(p):
            return p


def used():
    out = set(X.used_q())
    for f in ("poolH.json", "itemsX.json"):
        out |= {x["q"] for x in json.load(open(os.path.join(HERE, f))) if isinstance(x, dict) and "q" in x}
    out |= {x["q"] for x in json.load(open(os.path.join(HERE, "poolX.json")))["pool"]}
    return out


def build():
    g = random.Random("form-test")
    H0 = [x for x in json.load(open(os.path.join(HERE, "itemsH.json"))) if ok_item(x["q"], x["opts"], x["article"])]
    print("H 层可开放的题", len(H0))
    g.shuffle(H0)
    hard = [dict(x, stratum="H") for x in H0[:PER]]
    seen = used()
    fresh = {}
    for s in sorted({x["source"] for x in hard}):
        rows = json.load(open(path_of(s)))
        g.shuffle(rows)
        for r in rows:
            opts = r.get("choices") or r.get("options")
            if opts and r["question"] not in seen and ok_item(r["question"], opts, r.get("article", "")):
                fresh.setdefault(s, []).append(dict(q=r["question"], opts=opts, gold=r["answer"] if isinstance(r["answer"], int) else RUN.LET.index(r["answer"]), source=s, article=""))
    task = {x["source"]: x["task"] for x in hard}
    new = []
    for x in hard:
        cand = fresh.get(x["source"]) or next(v for s, v in sorted(fresh.items()) if task[s] == x["task"] and v)
        r = cand.pop()
        new.append(dict(r, task=task[r["source"]], stratum="N", id=f"Q{9000 + len(new)}", hard=False, cal=False, flagged=False))
    items = hard + new
    json.dump(items, open(BASE, "w"), ensure_ascii=False, indent=0)
    from collections import Counter
    print(len(items), Counter((x["stratum"], x["source"]) for x in items).most_common())


def gen():
    S2.ITEMS2, S2.PAY2 = BASE, PAY
    items = json.load(open(BASE))
    if not os.path.exists(PAY):
        old = json.load(open(os.path.join(HERE, "payloadsH.json")))
        json.dump({x["id"]: old[x["id"]] for x in items if x["id"] in old}, open(PAY, "w"), ensure_ascii=False, indent=0)
    S2.gen()
    P = json.load(open(PAY))
    om = []
    for x in items:
        w = S2.wrong(x)
        P[x["id"]].update(cand_true=x["opts"][x["gold"]], cand_wrong1=x["opts"][w[0]], cand_wrong2=x["opts"][w[1]])
        shown = x["q"] + "\n" + "\n".join(f"  {l}. {o}" for l, o in zip(RUN.LET, x["opts"]))
        base = dict(base=x["id"], family="case", source=x["source"], stratum=x["stratum"], hard=x["stratum"] == "H", flagged=x.get("flagged", False), gold=x["opts"][x["gold"]])
        om.append(dict(base, id="O" + x["id"], cond="O", q=x["q"]))
        om.append(dict(base, id="M" + x["id"], cond="M", q=shown))
    json.dump(P, open(PAY, "w"), ensure_ascii=False, indent=0)
    json.dump(om, open(ITEMS, "w"), ensure_ascii=False, indent=0)
    bad = [k for k in (x["id"] for x in items) if not P[k].get("L3f") or re.search(r"\b(option|choice) [A-D]\b", P[k]["L3f"])]
    print("投放物", len(P), "O+M", len(om), "L3f 有问题", bad)


def run_om(node, om, pay):
    par = PAR.get(node, 4)
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
    jobs = [(it, p) for it in om for p in FORMS]
    random.Random(node).shuffle(jobs)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(second, jobs))


def run_all():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUTJ, exist_ok=True)
    base = json.load(open(BASE))
    om = json.load(open(ITEMS))
    pay = json.load(open(PAY))
    H._start[0] = RUN.balance()

    def node(a):
        H.run_node(a, base, pay)
        run_om(a, om, pay)
        print(a, "O M 完成", flush=True)
    with cf.ThreadPoolExecutor(len(NODES)) as ex:
        list(ex.map(node, NODES))
    grade_all()


def grade_all():
    om = json.load(open(ITEMS))
    rows = S3.gather(om, FORMS)
    Gd = S3.grade(rows)
    print("评分", len(Gd), "/", len(rows), flush=True)
    by = {x["id"]: x for x in om}
    out = {}
    for c in "OM":
        s = [r for r in rows if by[r[0].split("|")[0]]["cond"] == c]
        random.Random("k" + c).shuffle(s)
        s = s[:100]
        res = {}
        with cf.ThreadPoolExecutor(4) as ex:
            for v in ex.map(lambda i: S3.judge_batch(s[i:i + 6], "codex_ns"), range(0, len(s), 6)):
                res.update(v)
        both = [(Gd[k], res[k]) for k in res if k in Gd]
        ag = sum(a == b for a, b in both) / len(both)
        pa, pb = sum(a for a, _ in both) / len(both), sum(b for _, b in both) / len(both)
        pe = pa * pb + (1 - pa) * (1 - pb)
        out[c] = dict(n=len(both), agree=ag, kappa=(ag - pe) / (1 - pe) if pe < 1 else 1.0)
    json.dump(out, open(KAPPA, "w"))
    print("裁判复评", out, flush=True)


def load_rows():
    base = {x["id"]: x for x in json.load(open(BASE))}
    om = {x["id"]: x for x in json.load(open(ITEMS))}
    R = []
    Ro, miss, tot = S3.load()
    for r in Ro:
        it = om[r["id"]]
        R.append(dict(base=it["base"], cond=it["cond"], stratum=it["stratum"], flagged=it["flagged"], a=r["a"], p=r["p"], right=r["right"], ok=r["ok"]))
    N.OUT, N.ITEMS, N.LLMS = OUTJ, BASE, NODES
    json.dump({}, open(os.path.join(OUTJ, "kw_rule.json"), "w"))  # N.load 要这个文件；本轮没有 kw_rule 节点
    Rj, cover, tools, _ = N.load()
    for r in Rj:
        if r["a"] in NODES and r["p"] in FORMS:
            it = base[r["id"]]
            R.append(dict(base=r["id"], cond="J", stratum=it["stratum"], flagged=it.get("flagged", False), a=r["a"], p=r["p"], right=r["right"], ok=r["ok"]))
    return R, miss, tot, cover, tools


m = lambda xs: (lambda v: sum(v) / len(v) if v else float("nan"))(list(xs))


def index(R):
    T = {}
    for r in R:
        T[r["cond"], r["a"], r["base"], r["p"]] = r
    return T


def item_diff(T, a, i, c1, c2, p):
    """同题配对差 d(c1) − d(c2)；缺行或不是两边都第一次答对就返回 None。"""
    try:
        A = [T[c, a, i, q] for c in (c1, c2) for q in ("blank", p)]
    except KeyError:
        return None
    if not all(x["right"] for x in A):
        return None
    return (A[0]["ok"] - A[1]["ok"]) - (A[2]["ok"] - A[3]["ok"])


def paired(T, ids, c1, c2, p, nodes=NODES, mn=5):
    per = {}
    for a in nodes:
        v = [d for d in (item_diff(T, a, i, c1, c2, p) for i in ids) if d is not None]
        if len(v) >= mn:
            per[a] = (m(v), len(v))
    return per


def boot(fn, ids, n=2000, seed=7):
    """fn(重采样的题) → 数；按题重采样。"""
    g = random.Random(seed)
    v = sorted(x for x in (fn([g.choice(ids) for _ in ids]) for _ in range(n)) if x == x)
    return v[int(.025 * len(v))], v[int(.975 * len(v))]


def summary(T, ids, c1, c2, p, nodes=NODES):
    per = paired(T, ids, c1, c2, p, nodes)
    if len(per) < 5:
        return dict(v=float("nan"), lo=float("nan"), hi=float("nan"), pos=0, n=len(per), per=per)
    fn = lambda s: m(q for q in (mean_per(T, s, a, c1, c2, p) for a in per) if q == q)
    lo, hi = boot(fn, ids)
    return dict(v=m(x[0] for x in per.values()), lo=lo, hi=hi, pos=sum(x[0] > 0 for x in per.values()), n=len(per), per=per)


def mean_per(T, s, a, c1, c2, p):
    return m(d for d in (item_diff(T, a, i, c1, c2, p) for i in s) if d is not None)


def damage(T, a, ids, c, p):
    return m(T[c, a, i, "blank"]["ok"] - T[c, a, i, p]["ok"] for i in ids if (c, a, i, "blank") in T and (c, a, i, p) in T and T[c, a, i, "blank"]["right"])


def K(R, **f):
    return m(r["ok"] for r in R if r["right"] and all(r[k] == v for k, v in f.items()))


def B(R, **f):
    return m(r["ok"] for r in R if not r["right"] and all(r[k] == v for k, v in f.items()))


def sharp(T, ids, c, nodes=NODES):
    """分辨力 S = B(set2t) + K(set2f) − 1。"""
    out = []
    for a in nodes:
        t = [T[c, a, i, "set2t"]["ok"] for i in ids if (c, a, i, "set2t") in T and not T[c, a, i, "set2t"]["right"]]
        f = [T[c, a, i, "set2f"]["ok"] for i in ids if (c, a, i, "set2f") in T and T[c, a, i, "set2f"]["right"]]
        if t and f:
            out.append(m(t) + m(f) - 1)
    return m(out)


def verdicts(R, lab, nodes=NODES):
    """判据只读 H 层。"""
    T = index(R)
    ids = sorted({r["base"] for r in R if r["stratum"] == "H"})
    idn = sorted({r["base"] for r in R if r["stratum"] == "N"})
    print(f"\n[{lab}] 节点 {len(nodes)} H 题 {len(ids)} N 题 {len(idn)}")
    fr = [a for a in FRONT if a in nodes]
    kh = m(K(R, cond="J", stratum="H", p="set2f", a=a) for a in fr)
    kn = m(K(R, cond="J", stratum="N", p="set2f", a=a) for a in fr)
    p0 = kh <= .5 and kn - kh >= .2
    print(f"  P0 复现 J×H K(set2f) {kh:.2f}（≤ .50），J×N {kn:.2f}，差 {kn - kh:+.2f}（≥ .20）{S2.mark(p0)}")
    need = max(5, round(5 * len(nodes) / 7))

    def show(name, c1, c2, th=.15):
        s = summary(T, ids, c1, c2, "set2f", nodes)
        ok = s["n"] >= 5 and s["v"] >= th and s["lo"] > 0 and s["pos"] >= need
        print(f"  {name} d({c1}) − d({c2}) {s['v']:+.2f} [{s['lo']:+.2f},{s['hi']:+.2f}]，{s['pos']}/{s['n']} 节点 > 0 {S2.mark(ok)}  " + " ".join(f"{a} {v[0]:+.2f}({v[1]})" for a, v in s["per"].items()))
        return ok
    p1 = show("P1 选项", "M", "O")
    p2 = show("P2 格式", "J", "M")
    dh = m(damage(T, a, ids, "O", "set2f") for a in nodes)
    dn = m(damage(T, a, idn, "O", "set2f") for a in nodes)
    g = random.Random(11)
    v = []
    for _ in range(2000):
        a_, b_ = [g.choice(ids) for _ in ids], [g.choice(idn) for _ in idn]
        x = m(damage(T, a, a_, "O", "set2f") for a in nodes) - m(damage(T, a, b_, "O", "set2f") for a in nodes)
        if x == x:
            v.append(x)
    v.sort()
    lo3, hi3 = v[int(.025 * len(v))], v[int(.975 * len(v))]
    p3 = dh - dn >= .15 and lo3 > 0
    print(f"  P3 难度 O 下 d(H) {dh:.2f} − d(N) {dn:.2f} = {dh - dn:+.2f} [{lo3:+.2f},{hi3:+.2f}] {S2.mark(p3)}")

    def inter(s):
        A, Bb = [], []
        for a in nodes:
            A.append(mean_per(T, s, a, "J", "O", "set2f"))
            Bb.append(mean_per(T, s, a, "J", "O", "L3f"))
        return m(x for x in A if x == x) - m(x for x in Bb if x == x)
    v4 = inter(ids)
    lo4, hi4 = boot(inter, ids)
    p4 = v4 >= .10 and lo4 > 0
    print(f"  P4 集合特异 {v4:+.2f} [{lo4:+.2f},{hi4:+.2f}] {S2.mark(p4)}（set2f {m(x for x in (mean_per(T, ids, a, 'J', 'O', 'set2f') for a in nodes) if x == x):+.2f}，L3f {m(x for x in (mean_per(T, ids, a, 'J', 'O', 'L3f') for a in nodes) if x == x):+.2f}）")
    sd = lambda s: sharp(T, s, "O", nodes) - sharp(T, s, "J", nodes)
    v5 = sd(ids)
    lo5, hi5 = boot(sd, ids)
    p5 = v5 >= .10 and lo5 > 0
    print(f"  P5 分辨力 S(O) {sharp(T, ids, 'O', nodes):+.2f} S(M) {sharp(T, ids, 'M', nodes):+.2f} S(J) {sharp(T, ids, 'J', nodes):+.2f}；O − J {v5:+.2f} [{lo5:+.2f},{hi5:+.2f}] {S2.mark(p5)}")
    return p0, p1, p2, p3, p4, p5


def analyze():
    R, miss, tot, cover, tools = load_rows()
    json.dump(R, open(os.path.join(HERE, "rows_f.json"), "w"))
    kap = json.load(open(KAPPA)) if os.path.exists(KAPPA) else {}
    print(f"O+M 记录 {len(R)}，O+M 缺失 {miss}（{miss / max(tot, 1):.1%}）；J 覆盖 {({k: round(v, 2) for k, v in cover.items() if k in NODES})}；J 工具 {({k: round(v, 3) for k, v in tools.items() if k in NODES})}")
    print("裁判", kap)
    good = {}
    for c in "JMO":
        ok = c == "J" or kap.get(c, {}).get("kappa", 0) >= .7
        for s in "HN":
            X_ = [r for r in R if r["cond"] == c and r["stratum"] == s and r["p"] == "blank"]
            e = m(not r["right"] for r in X_)
            k0 = K(X_)
            okc = .05 <= e <= .85
            print(f"预检 {c}×{s} 第一次错 {e:.0%} blank K {k0:.0%} B {B(X_):.0%} {S2.mark(okc)}")
            ok = ok and okc
        k_all = K([r for r in R if r["cond"] == c and r["p"] == "blank"])
        ok = ok and k_all >= .70
        print(f"  {c} blank K {k_all:.0%}（≥ 70%）条件 {S2.mark(ok)}")
        good[c] = ok
    p0, p1, p2, p3, p4, p5 = verdicts(R, "全部")
    print("\n决定")
    print(f"  映射预检 {good}")
    if not p0:
        print("  P0 不过：旧现象在这批没复现，下面只是报告。")
    print(f"  选项是原因之一：{S2.mark(p0 and p1 and good['M'] and good['O'])}")
    print(f"  回答格式 / 成批 / 字母是原因之一：{S2.mark(p0 and p2 and good['J'] and good['M'])}")
    print(f"  难度在开放题上也起作用：{S2.mark(p3 and good['O'])}")
    print(f"  上一轮差别主要来自题不同（P1、P2 都不过且 P3 过）：{S2.mark(p0 and not p1 and not p2 and p3)}")
    print(f"  开放题分辨力确实更好（P5）：{S2.mark(p5)}")

    print("\n另报不判")
    T = index(R)
    idn = sorted({r["base"] for r in R if r["stratum"] == "N"})
    for lab, c1, c2 in (("选项", "M", "O"), ("格式", "J", "M")):
        s = summary(T, idn, c1, c2, "set2f")
        print(f"  N 层 {lab} d({c1}) − d({c2}) {s['v']:+.2f} [{s['lo']:+.2f},{s['hi']:+.2f}] {s['pos']}/{s['n']}")
    verdicts([r for r in R if not r["flagged"]], "去掉标记题")
    verdicts(R, "去掉 Sonnet", [a for a in NODES if a != "claude_sonnet"])
    print("\n每个节点 K、B（H 层 / N 层）")
    for s in "HN":
        for a in NODES:
            print(f"  {s} {a:14s} " + " ｜ ".join(c + " " + " ".join(f"K{p} {K(R, cond=c, stratum=s, a=a, p=p):.0%}" for p in ("blank", "set2f", "L3f")) + f" B(set2t) {B(R, cond=c, stratum=s, a=a, p='set2t'):.0%}" for c in "JMO"))
    old = json.load(open(os.path.join(HERE, "rows_h.json")))
    ids = {r["base"] for r in R if r["stratum"] == "H"}
    print("\n复现：旧 runs_h（同这 40 题）→ 新 J×H，K(set2f)")
    for a in NODES:
        o = [r for r in old if r["a"] == a and r["id"] in ids and r["p"] == "set2f"]
        print(f"  {a:14s} {K(o):.0%} → {K(R, cond='J', stratum='H', a=a, p='set2f'):.0%}")


if __name__ == "__main__":
    {"build": build, "gen": gen, "run": run_all, "grade": grade_all, "a": analyze}[sys.argv[1]]()
