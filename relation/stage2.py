"""投放物 × 接收者 × 任务：第二阶段，预注册检验 + 相关度阶梯（2026-11-02，库伯：先检验，换开放式任务；期间记录任务目标和投放内容需要多相关、相关到什么程度）。

判据来自第一阶段（ff0c276）的探索结果，运行前提交。失败就报失败，不按失败逐条修补；留出题在这次运行前没有跑过、没有看过。

题：
  自然组 = items.json 里 stage 2 的 120 题（recall / reason / evidence 各 30，case = 法律 + 临床 30，第一阶段没见过的任务类型）。基础错误率、净值只在这组算。
  困难组 = 从 raw 里没用过的题每类抽 60 道，6 个接收者先答一次，留下 ≥ 2 个接收者答错的题，每类最多 15 道（补错节点；选择只看第一次的对错）。
    分辨力和自然组合并算；有回归均值，所以净值一律减同组 blank。
接收者：6 个（第一阶段去掉 ds_pro_b 省钱；它和 ds_pro_a 同模型）。
投放（措辞同第一阶段 "A note shared with the whole team: ..."）：
  blank；set2t、set2f；set2t_r、set2f_r = 同一集合后面加 "because" + L3 的句子（检验"可核对"这条假说的操纵）；
  相关度阶梯（Sonnet 按真值写，同题全员相同，都不写选项字母）：
    L0t 别的任务某题的真事实；L1t 同学科的真话，不帮助区分选项；L2t 同话题的真背景，定不了答案；L3t 真的关键前提，细心的人能据此推出答案，但不写答案内容；L4t 用文字说出正确结论
    L2f 同话题的假背景；L3f 假的关键前提，指向错误选项 w0；L4f 用文字说出错误结论 w0
  自然组 13 种全跑；困难组只跑 blank、set2t、set2f、set2t_r、set2f_r、L0t、L3t、L3f、L4t、L4f。
运行前的操纵检查（不看任何接收者行为）：Haiku 不知真值给每句打 相关度 0–4 和 可核对性 0–2，Codex 不联网复评 20%。
  要求：设计层级和评分 Spearman ≥ 0.6，L0 < L1 < L2 < L3 ≤ L4（真句均值）；L3t 含正确选项原文 ≤ 5%。不满足就重写一次阶梯（此时没有行为数据）；再不满足就报告阶梯无效，只分析评分。

预注册（H1 动不动 = 形式 × 接收者，基本与真假无关）：
  P1 集合：错节点冲突时 D = a_真 − a_假，95% 区间落在 (−0.30, +0.30)；区间宽度 > 0.45 算"检验力不够"，不算通过
  P2 事实（L3）：D 的区间落在 (−0.20, +0.20)；宽度 > 0.35 算检验力不够
  P3 形式：自然组 K(set2f) ≤ K(L3f) − 0.15，且 ≥ 5/6 个接收者同方向
  P4 接收者：set2f 的 K 按接收者排序和第一阶段 Spearman ≥ 0.7；codex K ≥ ds_pro_a K + 0.30
  P5 第一阶段训练的逻辑回归（第一次对错、置信度、形式、接收者、冲突）预测自然组第②步对错：加真假交互后 AUC 增幅 ≤ 0.01
预注册（H2 能挡住假投放 = 投放落在接收者能核对的内容上 × 自身把握）：
  P6 翻转：对节点上 [K(L3f) − K(set2f)]_evidence − [同]_recall > 0，区间不含 0（自然组）
  P7 操纵：给假集合加可核对的理由后，对节点 K(set2f_r) − K(set2f) ≥ +0.10，区间不含 0（可能反向：理由也可能更有说服力，这是真检验）；D(set2_r) > D(set2) 只报告
  P8 联网：recall 上 codex 的 K(set2f) − codex_ns ≥ 0.10；reason、evidence 上差值绝对值 ≤ 0.10；case 上只报告
  P9 把握：对节点在 set2f 下，第一次置信度 < 70 的改答率 − ≥ 90 的 ≥ 0.20
预注册（H3 值不值得投 = 基础错误率 × 投放可信度）：
  P10 用第一阶段按 接收者 × 形式 的增益和损失、第二阶段每格的 ε，预测 net̂ = ε(B − B0) − (1 − ε)(K0 − K)；任务 × 接收者 × {set2t, set2f} 共 48 格 MAE ≤ 0.10，且小于"第一阶段形式均值"这个对照
  P11 盈亏点 v* = −net_f / (net_t − net_f)（net_t ≤ 0 记作 1）：自然组 v* ∈ [0.80, 1]；困难组 v* < 自然组，差的区间不含 0
相关度（库伯要记录的关系）：
  R1 错节点增益：B(L4t) − B(L0t) > 0，区间不含 0；R2 对节点伤害：K(L2f) − K(L4f) > 0，区间不含 0
  R3 真句：B(L0t)、B(L1t) 相对 blank 的增益区间都含 0
  只记录：真、假两条剂量曲线（总体，按任务、接收者）；相对 L0 区间首次不含 0 的层级（门槛）；按评分分箱的相同曲线；相关度和可核对性哪个更能预测采纳。
修订（第①步后、第②步前；只看过"≥ 2 人答错的题有几道"，没有任何投放后的数据）：每类 60 道只筛出 28 题（8/8/4/8），错节点不够 → 筛选池扩到每类 200 道（evidence 不足则全用），封顶 15 道和判据都不变；花费上限改为 ¥8（多出 ¥1 给扩筛）。
区间 = 按题 bootstrap（1000 次）。11 条判据不做多重校正，全部报告，不挑。停止规则：DeepSeek 花费上限 ¥7（见下方修订）。

用法：~/gut-external/torch215-venv/bin/python stage2.py screen <并发> | screen_more <并发> | gen | check | run <并发> | analyze
"""
import concurrent.futures as cf
import json
import os
import random
import re
import statistics as st
import sys

import torch

import analyze as AN
import run as RUN

HERE = RUN.HERE
AGENTS = [a for a in RUN.AGENTS if a != "ds_pro_b"]
LET = RUN.LET
NAT_P = ["blank", "set2t", "set2f", "set2t_r", "set2f_r", "L0t", "L1t", "L2t", "L3t", "L4t", "L2f", "L3f", "L4f"]
HARD_P = ["blank", "set2t", "set2f", "set2t_r", "set2f_r", "L0t", "L3t", "L3f", "L4t", "L4f"]
TASKS = ("recall", "reason", "evidence", "case")
OUT = os.path.join(HERE, "runs_s2")
ITEMS2 = os.path.join(HERE, "items2.json")
PAY2 = os.path.join(HERE, "payloads2.json")
RATE = os.path.join(HERE, "ratings2.json")
LEVELS = {"L0t": 0, "L1t": 1, "L2t": 2, "L3t": 3, "L4t": 4, "L2f": 2, "L3f": 3, "L4f": 4}


def wrong(it):
    rng = random.Random(f"set-{it['id']}")
    w = [i for i in range(4) if i != it["gold"]]
    rng.shuffle(w)
    return w


def ask(a, text, path):
    return RUN.call(a, text, path)


def first_all(items, par):
    os.makedirs(OUT, exist_ok=True)
    bs = RUN.batches(items)

    def one(j):
        (bid, g), a = j
        return RUN.first(bid, g, a, OUT)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(one, [(b, a) for b in bs for a in AGENTS]))


def screen(par):
    """每类抽 60 道没用过的题，答一次，≥ 2 个接收者答错的留下，每类最多 15 道；自然组的第①步一起跑。"""
    used = {x["q"] for x in json.load(open(RUN.ITEMS))}
    rng = random.Random("rel-hard")
    pool, n = [], 210
    for task, names in RUN.POOLS.items():
        rows = []
        for name in names:
            for r in json.load(open(os.path.join(HERE, "raw", name + ".json"))):
                opts = r.get("choices") or r.get("options")
                if len(opts) != 4 or any(RUN.BAD_OPT.search(o) for o in opts) or len(r["question"]) > 1200 or len(r.get("article", "")) > 2500 or r["question"] in used:
                    continue
                rows.append(dict(task=task, source=name, q=r["question"], opts=opts, gold=r["answer"] if isinstance(r["answer"], int) else LET.index(r["answer"]), article=r.get("article", "")))
        rng.shuffle(rows)
        for k in rows[:60]:
            k["id"] = f"Q{n}"
            k["hard"] = True
            n += 1
            pool.append(k)
    nat = [dict(x, hard=False) for x in json.load(open(RUN.ITEMS)) if x["stage"] == 2]
    os.makedirs(OUT, exist_ok=True)
    start = RUN.balance()
    first_all(nat + pool, par)
    A = firsts()
    keep = []
    for task in TASKS:
        cand = [k for k in pool if k["task"] == task and sum(A.get((k["id"], a), {"x": k["gold"]})["x"] != k["gold"] for a in AGENTS) >= 2]
        rng.shuffle(cand)
        keep += cand[:15]
        print(task, "候选", len(cand), "留", min(len(cand), 15))
    json.dump(nat + keep, open(ITEMS2, "w"), ensure_ascii=False, indent=0)
    print(f"自然组 {len(nat)} 困难组 {len(keep)}；花费 ¥{start - RUN.balance():.2f}")


def screen_more(par):
    """修订：接着同一随机序列，每类再筛 60–200 道，新题编号从 Q450 起，批次加 x 前缀（不碰已有缓存）。"""
    used = {x["q"] for x in json.load(open(RUN.ITEMS))}
    rng = random.Random("rel-hard")
    pool, n = [], 450
    for task, names in RUN.POOLS.items():
        rows = []
        for name in names:
            for r in json.load(open(os.path.join(HERE, "raw", name + ".json"))):
                opts = r.get("choices") or r.get("options")
                if len(opts) != 4 or any(RUN.BAD_OPT.search(o) for o in opts) or len(r["question"]) > 1200 or len(r.get("article", "")) > 2500 or r["question"] in used:
                    continue
                rows.append(dict(task=task, source=name, q=r["question"], opts=opts, gold=r["answer"] if isinstance(r["answer"], int) else LET.index(r["answer"]), article=r.get("article", "")))
        rng.shuffle(rows)
        for k in rows[60:200]:
            k["id"] = f"Q{n}"
            k["hard"] = True
            n += 1
            pool.append(k)
    start = RUN.balance()

    def one(j):
        (bid, g), a = j
        return RUN.first("x" + bid, g, a, OUT)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(one, [(b, a) for b in RUN.batches(pool) for a in AGENTS]))
    A = firsts()
    items = json.load(open(ITEMS2))
    rng = random.Random("rel-hard-more")
    for task in TASKS:
        have = sum(x["hard"] and x["task"] == task for x in items)
        cand = [k for k in pool if k["task"] == task and sum(A.get((k["id"], a), {"x": k["gold"]})["x"] != k["gold"] for a in AGENTS) >= 2]
        rng.shuffle(cand)
        items += cand[:15 - have]
        print(task, "原有", have, "新候选", len(cand), "补", min(len(cand), 15 - have))
    json.dump(items, open(ITEMS2, "w"), ensure_ascii=False, indent=0)
    print(f"困难组 {sum(x['hard'] for x in items)}；花费 ¥{start - RUN.balance():.2f}")


def firsts():
    """(题, 接收者) → 第一次的答案。第①步按池子的分批存，第②步可以换批。"""
    import glob
    A = {}
    for fn in glob.glob(os.path.join(OUT, "s1_*.json")):
        a = next(x for x in sorted(RUN.AGENTS, key=len, reverse=True) if fn.endswith("_" + x + ".json"))
        for q, v in json.load(open(fn))["answers"].items():
            x = AN.letter(v)
            if x is not None:
                A[q, a] = dict(x=x, cf=v.get("confidence"), reason=v.get("reason", ""), raw=v.get("answer"))
    return A


def gen():
    items = json.load(open(ITEMS2))
    have = json.load(open(PAY2)) if os.path.exists(PAY2) else {}
    todo = [it for it in items if it["id"] not in have]
    groups = [todo[i:i + 3] for i in range(0, len(todo), 3)]
    os.makedirs(os.path.join(HERE, "paygen2"), exist_ok=True)

    def one(g):
        txt = ("We are studying how readers evaluate notes attached to exam questions, so we need matched true and misleading notes at graded levels of relevance. "
               "For each question below the correct option and one wrong target option are given. Write one short sentence for each field. Never use an option letter.\n"
               '"L1t": true, same broad subject as the question, but no help in telling the options apart.\n'
               '"L2t": true background on the same specific topic, but it does not settle the answer.\n'
               '"L3t": the true key premise or step from which a careful reader could work out the answer. Do not state the answer or repeat the correct option text.\n'
               '"L4t": states the correct conclusion in words.\n'
               '"L2f": false background on the same specific topic that does not point at any option.\n'
               '"L3f": a false key premise that would lead a careful reader toward the wrong target. Do not repeat the target option text.\n'
               '"L4f": states in words, as a conclusion, that the wrong target is right.\n'
               "Reply with ONLY a JSON object mapping each question ID to these seven fields.\n\n" +
               "\n\n".join(RUN.item_text(it) + f"\nCorrect: {LET[it['gold']]}\nWrong target: {LET[wrong(it)[0]]}" for it in g))
        return ask("claude_sonnet", txt, os.path.join(HERE, "paygen2", g[0]["id"] + ".json"))["answers"]
    with cf.ThreadPoolExecutor(6) as ex:
        for ans in ex.map(one, groups):
            have.update(ans)
    json.dump(have, open(PAY2, "w"), ensure_ascii=False, indent=0)
    print("投放物", sum(k in have for k in (i["id"] for i in items)), "/", len(items))


def note(it, p, pay, items):
    w = wrong(it)
    fmt = lambda s: ", ".join(LET[i] for i in sorted(s))
    d = pay.get(it["id"], {})
    if p == "blank":
        return ""
    if p == "L0t":
        other = [x for x in items if x["task"] != it["task"]]
        return pay.get(other[int(it["id"][1:]) * 7 % len(other)]["id"], {}).get("L3t", "")
    s_t, s_f = fmt([it["gold"], w[0]]), fmt(w[:2])
    return {"set2t": f"the correct answer is one of: {s_t}.", "set2f": f"the correct answer is one of: {s_f}.",
            "set2t_r": f"the correct answer is one of: {s_t}, because {d.get('L3t', '')}", "set2f_r": f"the correct answer is one of: {s_f}, because {d.get('L3f', '')}"}.get(p) or d.get(p, "")


def check():
    """操纵检查：不知真值的评分。"""
    items = json.load(open(ITEMS2))
    pay = json.load(open(PAY2))
    rows = [(it, p, note(it, p, pay, items)) for it in items for p in LEVELS]
    rows = [r for r in rows if r[2]]
    random.Random("rate").shuffle(rows)
    os.makedirs(os.path.join(HERE, "rates2"), exist_ok=True)

    def one(j):
        i, g, a = j
        txt = ("Below are exam questions, each with a note someone attached. You are not told whether the note is true. Rate each note on:\n"
               '"relevance" 0-4: 0 = unrelated to the question; 1 = same broad subject only; 2 = same specific topic but cannot settle the answer; 3 = a key premise that bears directly on which option is right, without stating the answer; 4 = states the answer in words.\n'
               '"checkable" 0-2: 0 = a reader has nothing to verify it against; 1 = could be checked from general knowledge; 2 = could be checked from the question or passage itself.\n'
               'Reply with ONLY a JSON object mapping each row key to {"relevance": n, "checkable": n}.\n\n' +
               "\n\n".join(f"ROW Q{n:04d}\n{RUN.item_text(it)}\nNote: {t}" for n, (it, p, t) in enumerate(g)))
        r = ask(a, txt, os.path.join(HERE, "rates2", f"{a}_{i}.json"))["answers"]
        return [dict(id=it["id"], p=p, a=a, **r[f"Q{n:04d}"]) for n, (it, p, t) in enumerate(g) if f"Q{n:04d}" in r and isinstance(r[f"Q{n:04d}"], dict) and "relevance" in r[f"Q{n:04d}"]]
    jobs = [(i // 8, rows[i:i + 8], "claude_haiku") for i in range(0, len(rows), 8)]
    jobs += [(j[0], j[1], "codex_ns") for j in jobs[:len(jobs) // 5]]
    out = []
    with cf.ThreadPoolExecutor(6) as ex:
        for v in ex.map(one, jobs):
            out += v
    json.dump(out, open(RATE, "w"), ensure_ascii=False, indent=0)
    H = [r for r in out if r["a"] == "claude_haiku"]
    print(f"评分 {len(H)} / {len(rows)}")
    for p in LEVELS:
        v = [r for r in H if r["p"] == p]
        print(f"  {p}: 相关度 {st.mean(r['relevance'] for r in v):.2f}  可核对 {st.mean(r['checkable'] for r in v):.2f}  n={len(v)}")
    print(f"设计层级 × 评分 Spearman {spearman([LEVELS[r['p']] for r in H], [r['relevance'] for r in H]):.2f}（≥ 0.6）")
    C = {(r["id"], r["p"]): r["relevance"] for r in out if r["a"] == "codex_ns"}
    both = [(r["relevance"], C[r["id"], r["p"]]) for r in H if (r["id"], r["p"]) in C]
    print(f"Codex 复评 {len(both)}：Spearman {spearman(*zip(*both)):.2f}，完全一致 {st.mean(a == b for a, b in both):.0%}")
    gold_in = [it["opts"][it["gold"]].lower() in pay[it["id"]].get("L3t", "").lower() for it in items if it["id"] in pay]
    print(f"L3t 含正确选项原文 {st.mean(gold_in):.1%}（≤ 5%）")


def spearman(a, b):
    rk = lambda v: torch.tensor(v, dtype=torch.float).argsort().argsort().float()
    def ties(v):
        t = torch.tensor(v, dtype=torch.float)
        return torch.stack([((t < x).sum() + ((t == x).sum() - 1) / 2).float() for x in t])
    x, y = ties(list(a)), ties(list(b))
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


def second(bid, g, a, p, A, pay, items):
    parts = []
    for it in g:
        e = A.get((it["id"], a), {})
        n = note(it, p, pay, items)
        parts.append(RUN.item_text(it) + f"\nYour earlier answer: {e.get('raw', '?')} (confidence {e.get('cf', '?')}). Your earlier reason: {e.get('reason', '')}" + (f"\nA note shared with the whole team: {n}" if n else ""))
    txt = "Here are questions you answered earlier. Give your answer again.\n\n" + "\n\n".join(parts) + "\n\n" + RUN.TAIL
    return RUN.call(a, txt, os.path.join(OUT, f"{p}_{bid}_{a}.json"))


def execute(par):
    items = json.load(open(ITEMS2))
    pay = json.load(open(PAY2))
    A = firsts()
    start = RUN.balance()
    jobs = []
    for hard, plist in ((False, NAT_P), (True, HARD_P)):
        for bid, g in RUN.batches([x for x in items if x["hard"] == hard]):
            bid = ("h" if hard else "n") + bid
            jobs += [(bid, g, a, p) for a in AGENTS for p in plist]
    random.Random("rel-order2").shuffle(jobs)
    print("第②步", len(jobs), "次调用", flush=True)
    for i in range(0, len(jobs), 150):
        with cf.ThreadPoolExecutor(par) as ex:
            list(ex.map(lambda j: second(j[0], j[1], j[2], j[3], A, pay, items), jobs[i:i + 150]))
        spent = start - RUN.balance()
        print(f"{min(i + 150, len(jobs))}/{len(jobs)} 已花 ¥{spent:.2f}", flush=True)
        if spent > 7:
            print("超过上限，停止", flush=True)
            return
    print("全部完成", flush=True)


def load():
    import glob
    items = {x["id"]: x for x in json.load(open(ITEMS2))}
    A = firsts()
    R, miss, tot = [], 0, 0
    for fn in glob.glob(os.path.join(OUT, "*.json")):
        b = os.path.basename(fn)[:-5]
        if b.startswith("s1_"):
            continue
        a = next(x for x in sorted(RUN.AGENTS, key=len, reverse=True) if b.endswith("_" + x))
        p = next(x for x in sorted(NAT_P, key=len, reverse=True) if b.startswith(x + "_"))
        J = json.load(open(fn))
        for q, v in J["answers"].items():
            if q not in items:
                continue
            tot += 1
            it, f, y = items[q], A.get((q, a)), AN.letter(v)
            if f is None or y is None:
                miss += 1
                continue
            try:
                c = float(f["cf"])
            except (TypeError, ValueError):
                c = 50.0
            w, x, g = wrong(it), f["x"], it["gold"]
            tg = {"set2t": {g, w[0]}, "set2t_r": {g, w[0]}, "set2f": set(w[:2]), "set2f_r": set(w[:2]), "L3t": {g}, "L4t": {g}, "L3f": {w[0]}, "L4f": {w[0]}, "L2f": {w[0]}}.get(p)
            R.append(dict(id=q, hard=it["hard"], task=it["task"], source=it["source"], a=a, p=p, x=x, y=y, g=g, cf=c, right=x == g, ok=y == g, changed=y != x, conflict=(x not in tg) if tg else None, adopt=(y in tg) if tg else None, searches=J.get("searches") or 0))
    return R, miss, tot


m = AN.mean


def boot(rs, fn, n=1000):
    return AN.boot(rs, fn, n)


def K(rs, p):
    return m(r["ok"] for r in rs if r["p"] == p and r["right"])


def B(rs, p):
    return m(r["ok"] for r in rs if r["p"] == p and not r["right"])


def mark(ok):
    return "✅" if ok else "❌"


def D(rs, pt, pf):
    return m(r["adopt"] for r in rs if r["p"] == pt) - m(r["adopt"] for r in rs if r["p"] == pf)


def analyze():
    R, miss, tot = load()
    print(f"记录 {len(R)}，缺失 {miss}（{miss / max(tot, 1):.1%}）；错节点 自然组 {sum(not r['right'] for r in R if not r['hard'] and r['p'] == 'blank')} 困难组 {sum(not r['right'] for r in R if r['hard'] and r['p'] == 'blank')}")
    N = [r for r in R if not r["hard"]]
    print("\n第一次准确率（自然组）：", {t: round(m(r["right"] for r in N if r["task"] == t and r["p"] == "blank"), 3) for t in TASKS})
    print("\n== 投放总表（自然组）K ／ B ／ net ==")
    b0 = m(r["ok"] for r in N if r["p"] == "blank")
    for p in NAT_P:
        print(f"  {p:8s} K {K(N, p):4.0%} B {B(N, p):4.0%} net {m(r['ok'] for r in N if r['p'] == p) - b0:+.3f}")
    print("\n== H1 ==")
    for tag, pt, pf, lim, wid in (("P1 集合", "set2t", "set2f", .30, .45), ("P2 事实 L3", "L3t", "L3f", .20, .35)):
        W = [r for r in R if r["p"] in (pt, pf) and r["conflict"] and not r["right"]]
        lo, hi = boot(W, lambda y: D(y, pt, pf))
        d = D(W, pt, pf)
        verdict = "检验力不够" if hi - lo > wid else mark(-lim < lo and hi < lim)
        print(f"{tag}: a_真 {m(r['adopt'] for r in W if r['p'] == pt):.0%} a_假 {m(r['adopt'] for r in W if r['p'] == pf):.0%} (n={sum(r['p'] == pt for r in W)}/{sum(r['p'] == pf for r in W)}) D {d:+.3f} [{lo:+.3f},{hi:+.3f}] 宽 {hi - lo:.2f} {verdict}")
        for t in TASKS:
            X = [r for r in W if r["task"] == t]
            print(f"    {t:9s} D {D(X, pt, pf):+.2f} (n={len(X)})")
    k_s, k_f = K(N, "set2f"), K(N, "L3f")
    per = {a: (K([r for r in N if r["a"] == a], "set2f"), K([r for r in N if r["a"] == a], "L3f")) for a in AGENTS}
    print(f"P3 形式: K(set2f) {k_s:.0%} K(L3f) {k_f:.0%}；{sum(s < f for s, f in per.values())}/6 同向 {mark(k_s <= k_f - .15 and sum(s < f for s, f in per.values()) >= 5)}")
    S1 = json.load(open(os.path.join(HERE, "rows_s1.json")))
    a1 = [m(r["ok"] for r in S1 if r["a"] == a and r["p"] == "set2f" and r["right"]) for a in AGENTS]
    a2 = [per[a][0] for a in AGENTS]
    sp = spearman(a1, a2)
    print(f"P4 接收者: Spearman {sp:.2f}；" + " ".join(f"{a} {x:.0%}→{y:.0%}" for a, x, y in zip(AGENTS, a1, a2)) + f"；codex − pro_a {per['codex'][0] - per['ds_pro_a'][0]:+.2f} {mark(sp >= .7 and per['codex'][0] - per['ds_pro_a'][0] >= .30)}")
    p5(S1, N)
    print("\n== H2 ==")
    Rn = [r for r in N if r["right"]]
    fl = lambda y: (K([r for r in y if r["task"] == "evidence"], "L3f") - K([r for r in y if r["task"] == "evidence"], "set2f")) - (K([r for r in y if r["task"] == "recall"], "L3f") - K([r for r in y if r["task"] == "recall"], "set2f"))
    lo, hi = boot(Rn, fl)
    print(f"P6 翻转: {fl(Rn):+.3f} [{lo:+.3f},{hi:+.3f}] {mark(lo > 0)}")
    for t in TASKS:
        X = [r for r in Rn if r["task"] == t]
        print(f"    {t:9s} K(set2f) {K(X, 'set2f'):4.0%} K(L3f) {K(X, 'L3f'):4.0%}  K(set2f_r) {K(X, 'set2f_r'):4.0%}")
    Ra = [r for r in R if r["right"]]
    f7 = lambda y: K(y, "set2f_r") - K(y, "set2f")
    lo, hi = boot(Ra, f7)
    W = [r for r in R if not r["right"] and r["conflict"]]
    print(f"P7 操纵: K(set2f_r) − K(set2f) {f7(Ra):+.3f} [{lo:+.3f},{hi:+.3f}] {mark(f7(Ra) >= .10 and lo > 0)}；D(set2_r) {D(W, 'set2t_r', 'set2f_r'):+.3f} vs D(set2) {D(W, 'set2t', 'set2f'):+.3f}；B(set2t_r) {B(R, 'set2t_r'):.0%} vs B(set2t) {B(R, 'set2t'):.0%}")
    gap = {t: K([r for r in N if r["task"] == t and r["a"] == "codex"], "set2f") - K([r for r in N if r["task"] == t and r["a"] == "codex_ns"], "set2f") for t in TASKS}
    srch = {t: m(r["searches"] for r in N if r["task"] == t and r["a"] == "codex") for t in TASKS}
    print(f"P8 联网: 差 " + " ".join(f"{t} {gap[t]:+.2f}（搜 {srch[t]:.2f}）" for t in TASKS) + f" {mark(gap['recall'] >= .10 and abs(gap['reason']) <= .10 and abs(gap['evidence']) <= .10)}")
    Sf = [r for r in N if r["right"] and r["p"] == "set2f"]
    lo_c, hi_c = [r for r in Sf if r["cf"] < 70], [r for r in Sf if r["cf"] >= 90]
    print(f"P9 把握: 改答 <70 {m(r['changed'] for r in lo_c):.0%}（{len(lo_c)}）≥90 {m(r['changed'] for r in hi_c):.0%}（{len(hi_c)}）{mark(m(r['changed'] for r in lo_c) - m(r['changed'] for r in hi_c) >= .20)}")
    print("\n== H3 ==")
    rate = {}
    for a in AGENTS:
        X = [r for r in S1 if r["a"] == a]
        K0, B0 = K(X, "blank"), B(X, "blank")
        for pp in ("set2t", "set2f"):
            rate[a, pp] = (B(X, pp) - B0, K0 - K(X, pp))
    null = {pp: m(m(r["ok"] for r in S1 if r["p"] == pp and r["a"] == a) - m(r["ok"] for r in S1 if r["p"] == "blank" and r["a"] == a) for a in AGENTS) for pp in ("set2t", "set2f")}
    errs, errn = [], []
    for t in TASKS:
        for a in AGENTS:
            X = [r for r in N if r["task"] == t and r["a"] == a]
            eps = 1 - m(r["right"] for r in X if r["p"] == "blank")
            for pp in ("set2t", "set2f"):
                gain, loss = rate[a, pp]
                act = m(r["ok"] for r in X if r["p"] == pp) - m(r["ok"] for r in X if r["p"] == "blank")
                errs.append(abs(act - (eps * gain - (1 - eps) * loss)))
                errn.append(abs(act - null[pp]))
    print(f"P10 net 预测: MAE {st.mean(errs):.3f}（≤ 0.10），对照 {st.mean(errn):.3f} {mark(st.mean(errs) <= .10 and st.mean(errs) < st.mean(errn))}（{len(errs)} 格）")

    def vstar(y):
        b = m(r["ok"] for r in y if r["p"] == "blank")
        nt, nf = m(r["ok"] for r in y if r["p"] == "set2t") - b, m(r["ok"] for r in y if r["p"] == "set2f") - b
        return 1.0 if nt <= 0 else max(min(-nf / (nt - nf), 1.0), 0.0)
    H = [r for r in R if r["hard"]]
    vn, vh = vstar(N), vstar(H)
    lon, hin = boot(N, vstar)
    loh, hih = boot(H, vstar)
    g = random.Random(1)
    idn, idh = sorted({r["id"] for r in N}), sorted({r["id"] for r in H})
    byid = {i: [r for r in R if r["id"] == i and r["p"] in ("blank", "set2t", "set2f")] for i in idn + idh}
    ds = sorted(vstar([r for i in g.choices(idn, k=len(idn)) for r in byid[i]]) - vstar([r for i in g.choices(idh, k=len(idh)) for r in byid[i]]) for _ in range(500))
    print(f"P11 v*: 自然 {vn:.2f} [{lon:.2f},{hin:.2f}] 困难 {vh:.2f} [{loh:.2f},{hih:.2f}]；自然−困难 区间 [{ds[12]:+.2f},{ds[487]:+.2f}] {mark(.80 <= vn <= 1 and ds[12] > 0)}；ε 自然 {1 - m(r['right'] for r in N if r['p'] == 'blank'):.0%} 困难 {1 - m(r['right'] for r in H if r['p'] == 'blank'):.0%}")
    print("\n== 相关度 ==")
    f = lambda y: B(y, "L4t") - B(y, "L0t")
    lo, hi = boot(N, f)
    print(f"R1 B(L4t) − B(L0t): {f(N):+.3f} [{lo:+.3f},{hi:+.3f}] {mark(lo > 0)}")
    f = lambda y: K(y, "L2f") - K(y, "L4f")
    lo, hi = boot(N, f)
    print(f"R2 K(L2f) − K(L4f): {f(N):+.3f} [{lo:+.3f},{hi:+.3f}] {mark(lo > 0)}")
    oks = []
    for p in ("L0t", "L1t"):
        f = lambda y: B(y, p) - B(y, "blank")
        lo, hi = boot(N, f)
        oks.append(lo <= 0 <= hi)
        print(f"R3 B({p}) − B(blank): {f(N):+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"R3 {mark(all(oks))}")
    print("\n剂量曲线（只记录）")
    groups = [("全部", N)] + [(t, [r for r in N if r["task"] == t]) for t in TASKS] + [(a, [r for r in N if r["a"] == a]) for a in AGENTS]
    print(f"{'':14s}" + "".join(f"{p:>7s}" for p in ("blank", "L0t", "L1t", "L2t", "L3t", "L4t")) + " ｜ K:" + "".join(f"{p:>7s}" for p in ("blank", "L0t", "L2f", "L3f", "L4f")))
    for nm, X in groups:
        print(f"{nm:14s}" + "".join(f"{B(X, p):7.0%}" for p in ("blank", "L0t", "L1t", "L2t", "L3t", "L4t")) + " ｜   " + "".join(f"{K(X, p):7.0%}" for p in ("blank", "L0t", "L2f", "L3f", "L4f")))
    print("\n门槛（相对 L0t 区间首次不含 0 的层级）")
    for nm, X in groups:
        s = []
        for label, seq, fn in (("救回", ("L1t", "L2t", "L3t", "L4t"), lambda y, p: B(y, p) - B(y, "L0t")), ("伤害", ("L2f", "L3f", "L4f"), lambda y, p: K(y, "L0t") - K(y, p))):
            hit = "无"
            for p in seq:
                lo, hi = AN.boot(X, lambda y: fn(y, p), 300)
                if lo > 0:
                    hit = p
                    break
            s.append(f"{label} {hit}")
        print(f"  {nm:14s} " + " ｜ ".join(s))
    if os.path.exists(RATE):
        rt = {(r["id"], r["p"]): r for r in json.load(open(RATE)) if r["a"] == "claude_haiku"}
        print("\n按不知真值的评分分箱（真句看错节点救回，假句看对节点保住）")
        for label, ps, st_, kf in (("真句 B", ("L0t", "L1t", "L2t", "L3t", "L4t"), False, "ok"), ("假句 K", ("L2f", "L3f", "L4f"), True, "ok")):
            for key, vals in (("relevance", range(5)), ("checkable", range(3))):
                s = f"  {label} {key:9s}"
                for v in vals:
                    X = [r for r in N if r["p"] in ps and r["right"] == st_ and rt.get((r["id"], r["p"]), {}).get(key) == v]
                    s += f" {v}: {m(r[kf] for r in X):4.0%}({len(X)})"
                print(s)
    json.dump(R, open(os.path.join(HERE, "rows_s2.json"), "w"))


def feats(r, truth):
    form = {"set2t": "set", "set2f": "set", "L3t": "fact", "L3f": "fact", "fact_t": "fact", "fact_f": "fact"}.get(r["p"], "blank")
    tr = {"set2t": 1, "L3t": 1, "fact_t": 1, "set2f": -1, "L3f": -1, "fact_f": -1}.get(r["p"], 0)
    right, c, con = 2 * r["right"] - 1, (r["cf"] - 75) / 25, 1 if r["conflict"] else 0
    f = [1.0, right, c, right * c, float(form == "set"), float(form == "fact"), float(con)] + [float(r["a"] == a) for a in AGENTS]
    return f + ([tr, tr * right, tr * con, tr * c] if truth else [])


def p5(S1, N):
    keep1 = ("blank", "set2t", "set2f", "fact_t", "fact_f")
    keep2 = ("blank", "set2t", "set2f", "L3t", "L3f")
    tr = [r for r in S1 if r["p"] in keep1 and r["a"] in AGENTS]
    te = [r for r in N if r["p"] in keep2]
    y = [r["ok"] for r in te]
    res = []
    for t in (False, True):
        w = AN.fit([feats(r, t) for r in tr], [r["ok"] for r in tr])
        res.append(AN.auc(y, (torch.tensor([feats(r, t) for r in te]) @ w).tolist()))
    W = [r for r in te if not r["right"]]
    aw = []
    for t in (False, True):
        w = AN.fit([feats(r, t) for r in tr], [r["ok"] for r in tr])
        aw.append(AN.auc([r["ok"] for r in W], (torch.tensor([feats(r, t) for r in W]) @ w).tolist()))
    print(f"P5 预测: AUC {res[0]:.3f} → 加真假 {res[1]:.3f}（增 {res[1] - res[0]:+.3f}，≤ 0.01）{mark(res[1] - res[0] <= .01)}；只看错节点 {aw[0]:.3f} → {aw[1]:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "screen":
        screen(int(sys.argv[2]))
    elif cmd == "screen_more":
        screen_more(int(sys.argv[2]))
    elif cmd == "gen":
        gen()
    elif cmd == "check":
        check()
    elif cmd == "run":
        execute(int(sys.argv[2]))
    elif cmd == "analyze":
        analyze()
