"""新节点比赛（2026-11-03，库伯同意放宽 generality.py 的停止规则：F4 在 M1、M2 上没过，但让它和通用模型在真正的新节点上比）。

运行前提交。判据不看结果就定；不再补公式。

新节点（从没进过任何拟合）：
  claude_sonnet、kimi（k3.1-pro，默认会联网 → 用空工具 agent 文件禁掉）、grok（grok-4.7，禁联网、只留 read_file、空目录）、
  qwen14 = 本地 qwen2.5:14b、qwen9 = 本地 qwen3.5:9b（都是第一次用）；全部按第二阶段的 JSON 分批格式（答案 + 置信度 + 一句理由）
  flash_m65 = 遮 65% 题面的 Flash、kw_rule = 没见过的规则节点（第一次选和题面重合词最多的选项；有投放时选含投放关键词最多的选项，没有就不动）；按第四阶段的单字母格式
  第一次答案覆盖 < 80% 的节点剔除并报告。可用节点 < 4 个就算没结论。
  注意：投放是 Sonnet 写的，claude_sonnet 又是接收者，所以判据另报一遍"去掉 Sonnet"。
题（也没进过任何拟合）：第二阶段筛选池里没被选中、也不在第三阶段题池的题。每类随机 27 道（校准 7 + 预测 20）+ 筛选时 ≥ 2 个旧节点答错的剩余题（校准 3，其余预测）。投放由 Sonnet 用第二阶段同一提示词写。
  校准：每类 10 题，只用 blank L3t L3f set2t set2f。预测：其余题，只用节点从没见过的 L0t L2t L2f L4t L4f set2t_r set2f_r。
选手（总体只由 15 个旧节点定，新节点只用校准数据）：
  F4（5 个节点参数 + 4 个总体常数）；F4w = F4 + 每种形式一个总体常数改变"坚持原答案"的强度（来自上一轮线索；节点参数不增加）；
  PCA-k（k = 1、2、3、5，14 维 logit 的前 k 主成分，节点参数 = k）；最近邻；共用（不校准）；错校准（F4 用别的新节点的校准数据）
判据（对半只比预测部分的 log-loss，区间按题 bootstrap）：
  N1 "这个形式有特权"：F4（或 F4w）比 PCA5 在 ≥ 2/3 节点上好，且全节点平均差的区间 < 0。不过就不再称它为规律形式
  N2 节点需要几个数：平均 log-loss 不差于 PCA5 超过 0.01 的最小 k（只报告 k）；k ≤ 2 才说"两个数够了"
  N3 低维校准可以迁移：F4、PCA5 各自比 共用、最近邻、错校准 在 ≥ 2/3 节点上好且区间 < 0；关键格 K(L4f) B(L4t) K(set2f_r) B(set2t_r)（n ≥ 10）平均绝对误差 ≤ 0.12 且低于共用和最近邻
  N4 假理由暴露的复核：用 F4 校准出的 v；暴露量 = K(set2f_r) 观测 − 可加预测。节点间 Spearman(暴露量, v) ≥ 0.5；sonnet、kimi、grok 里 ≥ 2 个暴露量 ≥ +0.20
停止规则：DeepSeek 上限 ¥3。多重比较不校正，全部报告。

用法：~/gut-external/torch215-venv/bin/python newnodes.py select | run | analyze
"""
import concurrent.futures as cf
import json
import os
import random
import re
import subprocess
import sys
import urllib.request

import torch

import analyze as AN
import formula as FM
import generality as G
import run as RUN
import stage2 as S2
import stage4 as S4

HERE = RUN.HERE
OUT = os.path.join(HERE, "runs_n")
ITEMS = os.path.join(HERE, "itemsN.json")
PAY = os.path.join(HERE, "payloadsN.json")
SCR = os.path.join(RUN.C.SCRATCH, "nn")
LLMS = ["claude_sonnet", "kimi", "grok", "qwen14", "qwen9"]
ALL = LLMS + ["flash_m65", "kw_rule"]
CAL_P, TEST_P = sorted(G.CAL_P), sorted(G.TEST_P)
AGENT = os.path.join(SCR, "agent.md")


def rebuild_pool():
    """按第二阶段 screen / screen_more 的同一随机序列重建筛选池。"""
    used = {x["q"] for x in json.load(open(RUN.ITEMS))}
    rng = random.Random("rel-hard")
    out = []
    n1, n2 = 210, 450
    for task, names in RUN.POOLS.items():
        rows = []
        for name in names:
            for r in json.load(open(os.path.join(HERE, "raw", name + ".json"))):
                opts = r.get("choices") or r.get("options")
                if len(opts) != 4 or any(RUN.BAD_OPT.search(o) for o in opts) or len(r["question"]) > 1200 or len(r.get("article", "")) > 2500 or r["question"] in used:
                    continue
                rows.append(dict(task=task, source=name, q=r["question"], opts=opts, gold=r["answer"] if isinstance(r["answer"], int) else RUN.LET.index(r["answer"]), article=r.get("article", "")))
        rng.shuffle(rows)
        for k in rows[:60]:
            k["id"] = f"Q{n1}"
            n1 += 1
            out.append(k)
        for k in rows[60:200]:
            k["id"] = f"Q{n2}"
            n2 += 1
            out.append(k)
    return out


def select():
    pool = rebuild_pool()
    chosen = {x["id"] for x in json.load(open(S2.ITEMS2))}
    stage3 = {x["q"] for x in json.load(open(os.path.join(HERE, "pool3.json")))}
    A = S2.firsts()
    g = random.Random("rel-new")
    items = []
    for t in G.TASKS:
        rest = [k for k in pool if k["task"] == t and k["id"] not in chosen and k["q"] not in stage3]
        hard = [k for k in rest if sum(A.get((k["id"], a), {"x": k["gold"]})["x"] != k["gold"] for a in S2.AGENTS) >= 2]
        easy = [k for k in rest if k not in hard]
        g.shuffle(hard)
        g.shuffle(easy)
        nat = easy[:27]
        for i, k in enumerate(nat):
            items.append(dict(k, hard=False, cal=i < 7))
        for i, k in enumerate(hard):
            items.append(dict(k, hard=True, cal=i < 3))
        print(t, "自然", len(nat), "困难", len(hard), "校准", sum(i["cal"] and i["task"] == t for i in items))
    json.dump(items, open(ITEMS, "w"), ensure_ascii=False, indent=0)
    S2.ITEMS2, S2.PAY2 = ITEMS, PAY
    S2.gen()


def ask_cli(node, text):
    if node == "claude_sonnet":
        return subprocess.run(["claude", "-p", "--model", "sonnet", "--tools", ""], input=text, cwd=SCR, capture_output=True, text=True, timeout=600).stdout, 0
    if node == "kimi":
        out = subprocess.run(["kimi", "--agent-file", AGENT, "-p", text, "--output-format", "stream-json"], cwd=SCR, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL).stdout
        msg, tools = "", 0
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("role") == "assistant":
                if ev.get("content"):
                    msg = ev["content"] if isinstance(ev["content"], str) else json.dumps(ev["content"])
                tools += len(ev.get("tool_calls") or [])
        return msg, tools
    if node == "grok":
        out = subprocess.run(["grok", "-p", text, "--disable-web-search", "--tools", "read_file", "--no-subagents", "--verbatim", "--system-prompt-override", "You answer directly from what you know. You have no tools.", "--output-format", "json"], cwd=SCR, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL).stdout
        d = json.loads(out)
        return d["text"], d.get("num_turns", 1) - 1
    model = {"qwen14": "qwen2.5:14b", "qwen9": "qwen3.5:9b"}[node]
    body = json.dumps({"model": model, "stream": False, "think": False, "format": "json", "messages": [{"role": "user", "content": text}], "options": {"temperature": 0, "num_predict": 1500, "num_ctx": 8192}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["message"]["content"], 0


def call(node, text, path):
    if os.path.exists(path):
        return json.load(open(path))
    for _ in range(3):
        try:
            raw, tools = ask_cli(node, text)
            rec = {"answers": RUN.parse(raw), "tools": tools}
            json.dump(rec, open(path, "w"), ensure_ascii=False)
            return rec
        except Exception:
            continue
    return {"answers": {}}


def run_node(node, items, pay):
    bs = lambda xs, pre: [(pre + b, g) for b, g in RUN.batches(xs)]
    cal, test = [x for x in items if x["cal"]], [x for x in items if not x["cal"]]
    par = {"qwen14": 1, "qwen9": 1}.get(node, 4)
    first = {}

    def do_first(j):
        bid, g = j
        return call(node, "Answer each question.\n\n" + "\n\n".join(RUN.item_text(it) for it in g) + "\n\n" + RUN.TAIL, os.path.join(OUT, f"s1_{bid}_{node}.json"))
    with cf.ThreadPoolExecutor(par) as ex:
        for r in ex.map(do_first, bs(cal, "c") + bs(test, "t")):
            first.update(r["answers"])

    def do_second(j):
        bid, g, p = j
        parts = []
        for it in g:
            e = first.get(it["id"], {}) if isinstance(first.get(it["id"], {}), dict) else {}
            n = S2.note(it, p, pay, items)
            parts.append(RUN.item_text(it) + f"\nYour earlier answer: {e.get('answer', '?')} (confidence {e.get('confidence', '?')}). Your earlier reason: {e.get('reason', '')}" + (f"\nA note shared with the whole team: {n}" if n else ""))
        return call(node, "Here are questions you answered earlier. Give your answer again.\n\n" + "\n\n".join(parts) + "\n\n" + RUN.TAIL, os.path.join(OUT, f"{p}_{bid}_{node}.json"))
    jobs = [(b, g, p) for b, g in bs(cal, "c") for p in CAL_P] + [(b, g, p) for b, g in bs(test, "t") for p in TEST_P]
    random.Random(node).shuffle(jobs)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(do_second, jobs))
    print(node, "完成", flush=True)


def run_m65(items, pay):
    s = S4.Store("flash_m65")
    s.fn = os.path.join(OUT, "flash_m65.jsonl")
    S4.LLM["flash_m65"] = ("deepseek-flash", 0.0, 0.65)
    s.d = {}
    if os.path.exists(s.fn):
        for line in open(s.fn):
            k, v, raw = json.loads(line)
            s.d[k] = (v, raw)

    def one(j):
        it, p = j
        k = f"{it['id']}|{p}"
        if k in s.d:
            return
        x = None if p == "s1" else s.d.get(f"{it['id']}|s1", ("?",))[0]
        for _ in range(3):
            try:
                raw = S4.ask_ds("flash_m65", S4.prompt(it, .65, x, "" if p == "s1" else S2.note(it, p, pay, items)))
                s.put(k, S4.letter(raw), raw)
                return
            except Exception:
                continue
    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(one, [(it, "s1") for it in items]))
    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(one, [(it, p) for it in items for p in (CAL_P if it["cal"] else TEST_P)]))
    print("flash_m65 完成", flush=True)


STOP = set("the a an of to in is are was were be been that this it its and or for with as by on at from which who whom whose what when where why how not no but if then than so such their his her they he she we you i note correct answer one".split())


def kw(text):
    return {w for w in S4.TOK.findall(text.lower()) if len(w) >= 3 and w not in STOP}


def run_kw(items, pay):
    fn = os.path.join(OUT, "kw_rule.json")
    d = {}
    for it in items:
        base = kw(it.get("article", "") + " " + it["q"])
        x = max(range(4), key=lambda i: (len(base & kw(it["opts"][i])), -i))
        d[f"{it['id']}|s1"] = x
        for p in CAL_P if it["cal"] else TEST_P:
            n = kw(S2.note(it, p, pay, items))
            sc = [len(n & kw(o)) for o in it["opts"]]
            d[f"{it['id']}|{p}"] = max(range(4), key=lambda i: (sc[i], i == x)) if max(sc) else x
    json.dump(d, open(fn, "w"))


def run_all():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SCR, exist_ok=True)
    open(AGENT, "w").write("---\nname: plain\ndescription: answer directly with no tools\ntools: []\n---\nYou answer questions directly from what you know. You have no tools.\n")
    items = json.load(open(ITEMS))
    pay = json.load(open(PAY))
    start = RUN.balance()
    run_kw(items, pay)
    run_m65(items, pay)
    print(f"DeepSeek ¥{start - RUN.balance():.2f}", flush=True)
    with cf.ThreadPoolExecutor(5) as ex:
        list(ex.map(lambda n: run_node(n, items, pay), LLMS))
    print("全部完成", flush=True)


def load():
    import glob
    items = {x["id"]: x for x in json.load(open(ITEMS))}
    R, cover, tools = [], {}, {}

    def row(i, a, p, x, y, cf_):
        it = items[i]
        w, g = S2.wrong(it), it["gold"]
        tg = FM.target(it, p)
        return dict(id=i, hard=it["hard"], cal=it["cal"], task=it["task"], a=a, p=p, x=x, y=y, g=g, cf=cf_, right=x == g, ok=y == g, changed=y != x, conflict=(x not in tg) if tg else None, adopt=(y in tg) if tg else None)
    for a in LLMS:
        first = {}
        nf = 0
        for fn in glob.glob(os.path.join(OUT, f"s1_*_{a}.json")):
            for q, v in json.load(open(fn))["answers"].items():
                if q in items and isinstance(v, dict) and AN.letter(v) is not None:
                    first[q] = v
        cover[a] = len(first) / len(items)
        t = n = 0
        for fn in glob.glob(os.path.join(OUT, f"*_{a}.json")):
            b = os.path.basename(fn)
            J = json.load(open(fn))
            t += J.get("tools", 0)
            n += 1
            if b.startswith("s1_"):
                continue
            p = next(x for x in sorted(G.PAY12, key=len, reverse=True) if b.startswith(x + "_"))
            for q, v in J["answers"].items():
                if q in first and isinstance(v, dict) and AN.letter(v) is not None:
                    try:
                        c = float(first[q].get("confidence"))
                    except (TypeError, ValueError):
                        c = 50.0
                    R.append(row(q, a, p, AN.letter(first[q]), AN.letter(v), c))
        tools[a] = t / max(n, 1)
    s = {}
    fn = os.path.join(OUT, "flash_m65.jsonl")
    if os.path.exists(fn):
        for line in open(fn):
            k, v, raw = json.loads(line)
            s[k] = v
    F = {k.split("|")[0]: v for k, v in s.items() if k.endswith("|s1") and v}
    cover["flash_m65"] = len(F) / len(items)
    for k, v in s.items():
        i, p = k.split("|")
        if p != "s1" and v and i in F:
            R.append(row(i, "flash_m65", p, RUN.LET.index(F[i]), RUN.LET.index(v), 50.0))
    d = json.load(open(os.path.join(OUT, "kw_rule.json")))
    cover["kw_rule"] = 1.0
    for k, v in d.items():
        i, p = k.split("|")
        if p != "s1":
            R.append(row(i, "kw_rule", p, d[f"{i}|s1"], v, 50.0))
    return R, cover, tools, items


def f4w(t, phi, d, additive=False):
    """F4 + 每种形式一个总体常数改变 w（phi[4:9]）。"""
    q = G.f4(t, phi[:4], d, additive)
    bump = phi[4:9][d["F"]] * d["has"]
    s = q.clamp(min=1e-9).log() + bump[:, None] * d["X"]
    return torch.softmax(s, 1)


def fit_popw(train, phi0):
    D = [G.tens(G.BY[a]) for a in train]
    Th = torch.zeros(len(train), 5)
    Th[:, 0], Th[:, 2], Th[:, 3] = 3., 2., 4.
    Th.requires_grad_(True)
    phi = torch.cat([phi0, torch.zeros(5)]).requires_grad_(True)
    opt = torch.optim.LBFGS([Th, phi], max_iter=400, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = sum(torch.nn.functional.nll_loss(f4w(Th[i], phi, d).clamp(min=1e-6).log(), d["y"]) for i, d in enumerate(D)) / len(D) + 1e-3 * (Th ** 2).mean()
        l.backward()
        return l
    opt.step(cl)
    T = Th.detach()
    return T.mean(0), torch.cov(T.T) + .1 * torch.eye(5), phi.detach()


def analyze():
    torch.set_num_threads(4)
    R, cover, tools, items = load()
    G.items.update(items)
    FM.items.update(items)
    nodes = [a for a in ALL if cover.get(a, 0) >= .8]
    print("覆盖", {k: round(v, 2) for k, v in cover.items()}, "工具轮次（每次调用）", {k: round(v, 3) for k, v in tools.items()})
    print(f"可用节点 {nodes}（≥ 4）")
    train = G.NODES
    mu, Sig, phi = G.fit_pop(train)
    muw, Sigw, phiw = fit_popw(train, phi)
    print("总体常数 a_L2 %.2f a_L4 %.2f v_r = %.2f + %.2f·v；F4w 形式常数 %s" % (phi[0], phi[1], phi[2], phi[3], [round(x, 2) for x in phiw[4:].tolist()]))
    full3 = {n: FM.fit(FM.f3, 8, G.tens(G.BY[n])) for n in train}
    B = torch.stack([G.fit_lin(G.basis(G.tens(G.BY[n])), G.tens(G.BY[n])["y"]) for n in train])
    m0 = B.mean(0)
    U, Sv, Vh = torch.linalg.svd(B - m0, full_matrices=False)
    print("通用模型主成分方差占比", [round(x, 3) for x in (Sv ** 2 / (Sv ** 2).sum()).tolist()[:6]])

    def pca(k, dc, dt):
        Vk, sk = Vh[:k], Sv[:k] ** 2 / (len(train) - 1) + .1
        Pc, Pt = G.basis(dc), G.basis(dt)
        z = torch.zeros(k, requires_grad=True)
        opt = torch.optim.LBFGS([z], max_iter=200, line_search_fn="strong_wolfe")

        def cl():
            opt.zero_grad()
            l = torch.nn.functional.cross_entropy(Pc @ (m0 + z @ Vk), dc["y"]) + .5 * (z ** 2 / sk).sum() / len(dc["y"])
            l.backward()
            return l
        opt.step(cl)
        return torch.softmax(Pt @ (m0 + z.detach() @ Vk), 1), z.detach()
    MS = ["F4", "F4w", "PCA1", "PCA2", "PCA3", "PCA5", "共用", "最近邻", "错校准"]
    cal = {a: G.tens([r for r in R if r["a"] == a and r["cal"] and r["p"] in G.CAL_P]) for a in nodes}
    res, per, cells, expose, wr = {}, {m: {} for m in MS}, {}, {}, {}
    print(f"\n{'节点':14s}{'准确率':>7s}{'n预测':>7s}" + "".join(f"{m:>8s}" for m in MS))
    for a in nodes:
        te = [r for r in R if r["a"] == a and not r["cal"] and r["p"] in G.TEST_P]
        dt, dc = G.tens(te), cal[a]
        fn = lambda t, d: G.f4(t, phi, d)
        th = G.map_fit(dc, mu, Sig, fn, 5)
        fw = lambda t, d: f4w(t, phiw, d)
        thw = G.map_fit(dc, muw, Sigw, fw, 5)
        qs = {"F4": fn(th, dt), "F4w": fw(thw, dt), "共用": fn(mu, dt)}
        zs = {}
        for k in (1, 2, 3, 5):
            qs[f"PCA{k}"], zs[k] = pca(k, dc, dt)
        best = min(train, key=lambda n: FM.nll(FM.f3, full3[n], dc))
        qs["最近邻"] = FM.f3(full3[best], dt)
        others = [o for o in nodes if o != a]
        qs["错校准"] = sum(fn(G.map_fit(cal[o], mu, Sig, fn, 5), dt) for o in others) / max(len(others), 1)
        ids = [r["id"] for r in te]
        idx = torch.arange(len(te))
        for m, q in qs.items():
            per[m][a] = (ids, (-q.clamp(min=1e-6).log()[idx, dt["y"]]).detach())
        res[a] = {m: per[m][a][1].mean().item() for m in MS}
        cells[a] = {m: G.cell_err(qs[m], te) for m in ("F4", "F4w", "PCA5", "PCA2", "共用", "最近邻")}
        qa = G.f4(th, phi, dt, additive=True)
        ri = [i for i, r in enumerate(te) if r["p"] == "set2f_r" and r["right"]]
        if len(ri) >= 10:
            expose[a] = (sum(te[i]["ok"] for i in ri) / len(ri) - sum(qa[i, te[i]["g"]].item() for i in ri) / len(ri), th[4].item(), sum(te[i]["ok"] for i in ri) / len(ri))
        acc = AN.mean(r["right"] for r in R if r["a"] == a and r["p"] in ("blank", "L0t") and not r["hard"])
        wr[a] = sum(not r["right"] for r in te if r["p"] == "L0t")
        print(f"{a:14s}{acc:7.0%}{len(te):7d}" + "".join(f"{res[a][m]:8.3f}" for m in MS) + f"  近邻 {best} θ {[round(x, 2) for x in th.tolist()]} 错节点 {wr[a]}", flush=True)

    def compare(ns, a_m, b_m):
        w = sum(res[a][a_m] < res[a][b_m] for a in ns)
        by = {}
        for a in ns:
            I, la = per[a_m][a]
            lb = per[b_m][a][1]
            for k, i in enumerate(I):
                by.setdefault(i, []).append((la[k] - lb[k]).item())
        ids = sorted(by)
        g = random.Random(3)
        boots = []
        for _ in range(1000):
            v = [x for i in g.choices(ids, k=len(ids)) for x in by[i]]
            boots.append(sum(v) / len(v))
        boots.sort()
        mean = sum(x for v in by.values() for x in v) / sum(len(v) for v in by.values())
        return w, mean, boots[25], boots[974], (w * 3 >= 2 * len(ns) and boots[974] < 0)
    for lab, ns in (("全部新节点", nodes), ("去掉 Sonnet", [a for a in nodes if a != "claude_sonnet"]), ("只看没见过架构的 LLM", [a for a in nodes if a in LLMS])):
        if len(ns) < 2:
            continue
        print(f"\n== {lab}（{len(ns)} 个）==  平均 log-loss " + " ".join(f"{m} {sum(res[a][m] for a in ns) / len(ns):.3f}" for m in MS))
        for fm in ("F4", "F4w"):
            w, me, lo, hi, ok = compare(ns, fm, "PCA5")
            print(f"N1 {fm} 对 PCA5：赢 {w}/{len(ns)}；差 {me:+.4f} [{lo:+.4f},{hi:+.4f}] {S2.mark(ok)}")
        pm = {k: sum(res[a][f'PCA{k}'] for a in ns) / len(ns) for k in (1, 2, 3, 5)}
        kk = min(k for k in pm if pm[k] <= pm[5] + .01)
        print(f"N2 节点需要几个数：k = {kk}（PCA1 {pm[1]:.3f} PCA2 {pm[2]:.3f} PCA3 {pm[3]:.3f} PCA5 {pm[5]:.3f}）")
        for fm in ("F4", "PCA5"):
            s = []
            allok = True
            for b in ("共用", "最近邻", "错校准"):
                if fm == "PCA5" and b == "错校准":
                    continue
                w, me, lo, hi, ok = compare(ns, fm, b)
                allok = allok and ok
                s.append(f"对 {b} 赢 {w}/{len(ns)} {me:+.3f} [{lo:+.3f},{hi:+.3f}] {S2.mark(ok)}")
            print(f"N3 {fm}：" + "；".join(s))
        ce = {m: sum(cells[a][m] for a in ns) / len(ns) for m in ("F4", "F4w", "PCA5", "PCA2", "共用", "最近邻")}
        print("N3 关键格误差 " + " ".join(f"{m} {v:.3f}" for m, v in ce.items()) + f"；F4 {S2.mark(ce['F4'] <= .12 and ce['F4'] < ce['共用'] and ce['F4'] < ce['最近邻'])} PCA5 {S2.mark(ce['PCA5'] <= .12 and ce['PCA5'] < ce['共用'] and ce['PCA5'] < ce['最近邻'])}")
    ks = list(expose)
    sp = S2.spearman([expose[a][1] for a in ks], [expose[a][0] for a in ks]) if len(ks) >= 3 else float("nan")
    big = sum(expose.get(a, (0,))[0] >= .20 for a in ("claude_sonnet", "kimi", "grok"))
    print(f"\nN4 暴露：Spearman {sp:.2f}（≥ 0.5）；大模型 ≥ +0.20 的 {big}/3 {S2.mark(sp >= .5 and big >= 2)}")
    for a in ks:
        print(f"  {a:14s} K(set2f_r) {expose[a][2]:.0%} 暴露量 {expose[a][0]:+.2f} v {expose[a][1]:+.2f}")
    print("\n各节点观测（校准 K/B + 预测 K/B）")
    for a in nodes:
        X = [r for r in R if r["a"] == a]
        print(f"  {a:14s} K: blank {S2.K(X, 'blank'):4.0%} L3f {S2.K(X, 'L3f'):4.0%} set2f {S2.K(X, 'set2f'):4.0%} L4f {S2.K(X, 'L4f'):4.0%} set2f_r {S2.K(X, 'set2f_r'):4.0%} ｜ B: blank {S2.B(X, 'blank'):4.0%} L3t {S2.B(X, 'L3t'):4.0%} set2t {S2.B(X, 'set2t'):4.0%} L4t {S2.B(X, 'L4t'):4.0%} set2t_r {S2.B(X, 'set2t_r'):4.0%}")
    json.dump(R, open(os.path.join(HERE, "rows_n.json"), "w"))


if __name__ == "__main__":
    {"select": select, "run": run_all, "analyze": analyze}[sys.argv[1]]()
