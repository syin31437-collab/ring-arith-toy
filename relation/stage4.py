"""投放物 × 接收者：第四阶段，给接收者降智（2026-11-03，库伯：受试者智力下降、甚至到分布式系统里本身不太算有智力时，这些关系还有效吗；更适合它们的公式是什么；几个公式里有没有统一方向）。

判据来自第二、三阶段（22688163）；运行前提交。失败就报失败，不逐条修补。

题和投放：第二阶段的 180 道选择题（四类，自然 120 + 困难 60）和同一批投放，12 种：blank L0t L2t L3t L4t L2f L3f L4f set2t set2f set2t_r set2f_r。接收者是新的（题对它们没见过）。逐题调用。
接收者阶梯（从有智力到没智力）：
  flash_short = Flash，T = 0，只许回一个字母，没有理由和置信度（去掉推理）
  flash_hot   = 同上，T = 2（决策噪声）
  flash_m50 / flash_m80 = 同 flash_short，但题面 50% / 80% 的词被遮住（两轮同样的遮挡；投放不遮，模拟"看不懂任务但收得到投放"）
  tiny_qwen / tiny_llama / tiny_smol = 本地极小模型 qwen2.5:0.5b、tinyllama、smollm:135m，T = 0，只取字母
  emb = 不理解，只比向量：选项和"题 + 投放"的 all-minilm 余弦最高的那个
  lex = 不理解，只数词：选项和题的 IDF 重合 + 3 × 和投放的重合
  lex_proj = lex + 硬规则：投放写了字母集合就只在集合内选（程序化的投影）
  rand_proj = 第一次随机答；集合内不动，集合外在集合内均匀选；其他投放一律读不懂（纯规则节点）
  对照顶部：第二阶段的 ds_flash_a、ds_pro_a、claude_haiku、codex_ns（同题同投放）。
映射预检（不看投放效果）：无效输出 ≤ 20%，否则该节点不计入判据；第一次准确率要拉开（最高 ≥ 70%，最低 ≤ 40%）。
分组按第一次准确率（自然组）定，不看投放结果：TOP = ≥ 75%；FLOOR = ≤ 45%，且不含 lex_proj、rand_proj（它们是用来核对公式的硬规则）。

预注册：
  Q1 分辨力随智力消失：D（L4，减同节点 blank 基线）TOP − FLOOR > 0，区间不含 0
  Q2 "相关"的门槛上移：B(L3t) − B(L0t) 在 TOP − FLOOR > 0，区间不含 0；FLOOR 内 B(L4t) − B(L0t) > B(L3t) − B(L0t)（要读得懂才用得上前提，直说答案的还能起点作用）
  Q3 理由的保护随智力消失：K(set2f_r) − K(set2f) 在 TOP（含第二阶段 haiku、codex_ns）− FLOOR > 0，区间不含 0
  Q4 规则节点严格等于公式：rand_proj、lex_proj 的 K(set2t) = 100%，K(set2f) = 0%（±2%）；rand_proj 的 B(set2t) 相对投影的算术值（错节点在集合外 → 1/2，集合内 → 0）±5%，net_t 和 net_f 同样 ±5%
  Q5 能读形式就受形式驱动：flash_m80、tiny_* 中，凡 set2f 的 K 能算的节点，K(set2f) ≤ K(L3f) − 0.10 的 ≥ 一半
  Q6 盈亏点仍由 ε 决定：全部节点（自然 + 困难）ε 和 v*(set2) 的 Spearman ≤ −0.5
公式比较（数据跑完再写拟合代码，规则现在定，不改）：按题哈希对半分，一半拟合、另一半算 log-loss，预测 4 个选项的完整分布。每个节点单独拟合。
  F0 不动：P(y = x) = 1 − e，其余均分
  F1 硬投影：集合内不动，集合外均匀进集合，其他投放不动（+ e）
  F2 混合：以 α_形式 的概率跳到目标（真假共用），否则不动（+ e）
  F3 软投影（门控）：s(y) = w·[y = x] + g_形式·(1 − v·假)·[y ∈ 目标]，P(y) ∝ exp(s(y))，w 随置信度线性（有置信度时）；参数 w、g_L2 g_L3 g_L4 g_set g_set_r、v
  Q7 F3 的留出 log-loss 在 ≥ 2/3 的节点上不差于 F0、F1、F2（统一形式能覆盖全部智力层）
  Q8 统一方向：全部节点上 v 和第一次准确率 Spearman ≥ 0.5（辨真需要智力）；g_L3 / (g_L3 + g_set) 和第一次准确率 Spearman ≥ 0.5（越有智力越听内容，越没智力越听形式）
  只记录：F3 在 g → ∞、v = 0、w 有限时退化成投影；数值节点上对应 x' = x + κ(P_C(x) − x)，κ = g / (g + 1/σ²)。这部分数学不新（近端算子 / 卡尔曼增益），只是映射。
区间 = 按题 bootstrap。多重比较不校正，全部报告。停止规则：DeepSeek 花费上限 ¥10。

用法：~/gut-external/torch215-venv/bin/python stage4.py pull | pilot | run | analyze
"""
import concurrent.futures as cf
import json
import math
import os
import random
import re
import statistics as st
import sys
import urllib.request
from collections import Counter

import torch

import analyze as AN
import run as RUN
import stage2 as S2

HERE = RUN.HERE
OUT = os.path.join(HERE, "runs_s4")
LET = RUN.LET
PAYLOADS = ["blank", "L0t", "L2t", "L3t", "L4t", "L2f", "L3f", "L4f", "set2t", "set2f", "set2t_r", "set2f_r"]
LLM = {"flash_short": ("deepseek-flash", 0.0, 0.0), "flash_hot": ("deepseek-flash", 2.0, 0.0), "flash_m50": ("deepseek-flash", 0.0, 0.5), "flash_m80": ("deepseek-flash", 0.0, 0.8)}
TINY = {"tiny_qwen": "qwen2.5:0.5b", "tiny_llama": "tinyllama", "tiny_smol": "smollm:135m"}
RULE = ["emb", "lex", "lex_proj", "rand_proj"]
NODES = list(LLM) + list(TINY) + RULE
TOP2 = ["ds_flash_a", "ds_pro_a", "claude_haiku", "codex_ns"]
TOK = re.compile(r"[a-z0-9]+")


def mask(text, rate, seed):
    if not rate:
        return text
    g = random.Random(seed)
    return re.sub(r"\S+", lambda m: "___" if g.random() < rate else m.group(0), text)


def stem(it, rate):
    art = f"Passage:\n{mask(it['article'], rate, it['id'] + 'a')}\n" if it.get("article") else ""
    return art + mask(it["q"], rate, it["id"] + "q") + "\n" + "\n".join(f"{l}. {mask(o, rate, it['id'] + l)}" for l, o in zip(LET, it["opts"]))


def prompt(it, rate, x, n):
    s = stem(it, rate)
    if x is None:
        return s + "\nReply with one letter only (A, B, C or D).\nAnswer:"
    return s + f"\nYour earlier answer: {x}" + (f"\nA note shared with the whole team: {n}" if n else "") + "\nGive your answer again. Reply with one letter only (A, B, C or D).\nAnswer:"


def letter(t):
    m = re.search(r"\b([ABCD])\b", t or "")
    if m:
        return m.group(1)
    m = re.match(r"\s*([ABCDabcd])", t or "")
    return m.group(1).upper() if m else None


def ask_ds(node, text):
    model, temp, _ = LLM[node]
    e = json.load(open(os.path.expanduser("~/.hermes/auth.json")))["credential_pool"]["deepseek"][0]
    body = json.dumps({"model": model, "temperature": temp, "max_tokens": 4, "thinking": {"type": "disabled"}, "messages": [{"role": "system", "content": "You are a precise data assistant."}, {"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(e["base_url"].rstrip("/") + "/chat/completions", data=body, headers={"Authorization": "Bearer " + e["access_token"], "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]


def ask_tiny(node, text):
    body = json.dumps({"model": TINY[node], "prompt": text, "stream": False, "options": {"temperature": 0, "num_predict": 6, "num_ctx": 2048}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["response"]


def embed(texts):
    body = json.dumps({"model": "all-minilm", "input": texts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embed", data=body, headers={"Content-Type": "application/json"})
    return torch.tensor(json.load(urllib.request.urlopen(req, timeout=300))["embeddings"])


class Store:
    """每个节点一个 jsonl：键 → (字母, 原文)。可续跑。"""

    def __init__(self, node):
        self.fn = os.path.join(OUT, node + ".jsonl")
        self.d = {}
        if os.path.exists(self.fn):
            for line in open(self.fn):
                k, v, raw = json.loads(line)
                self.d[k] = (v, raw)

    def put(self, k, v, raw):
        self.d[k] = (v, raw)
        with open(self.fn, "a") as f:
            f.write(json.dumps([k, v, raw], ensure_ascii=False) + "\n")


def run_llm(node, items, pay, ask, par):
    st_ = Store(node)
    rate = LLM[node][2] if node in LLM else 0.0

    def one(j):
        it, p = j
        k = f"{it['id']}|{p}"
        if k in st_.d:
            return
        x = None if p == "s1" else st_.d.get(f"{it['id']}|s1", (None,))[0]
        if p != "s1" and x is None:
            x = "?"
        n = "" if p == "s1" else S2.note(it, p, pay, items)
        for _ in range(3):
            try:
                raw = ask(node, prompt(it, rate, None if p == "s1" else x, n))
                st_.put(k, letter(raw), raw)
                return
            except Exception:
                continue
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(one, [(it, "s1") for it in items]))
    jobs = [(it, p) for it in items for p in PAYLOADS]
    random.Random(node).shuffle(jobs)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(one, jobs))
    return st_


def idf_table(items):
    df = Counter()
    for it in items:
        df.update(set(TOK.findall((it["q"] + " " + " ".join(it["opts"])).lower())))
    return lambda w: math.log(1 + len(items) / (1 + df[w]))


def run_rules(items, pay):
    idf = idf_table(items)
    lex, lexp, rnd, em = Store("lex"), Store("lex_proj"), Store("rand_proj"), Store("emb")
    def ov(a, b):
        return sum(idf(w) for w in set(TOK.findall(a.lower())) & set(TOK.findall(b.lower())))
    for it in items:
        base = (it.get("article", "") + " " + it["q"])
        score0 = [ov(base, o) for o in it["opts"]]
        x = max(range(4), key=lambda i: (score0[i], -i))
        g = random.Random("rand" + it["id"])
        x_r = g.randrange(4)
        if f"{it['id']}|s1" not in lex.d:
            lex.put(f"{it['id']}|s1", LET[x], "")
            lexp.put(f"{it['id']}|s1", LET[x], "")
            rnd.put(f"{it['id']}|s1", LET[x_r], "")
        for p in PAYLOADS:
            n = S2.note(it, p, pay, items)
            sc = [score0[i] + 3 * ov(n, it["opts"][i]) for i in range(4)]
            y = max(range(4), key=lambda i: (sc[i], i == x, -i))
            m = re.search(r"one of: ([A-D](?:, [A-D])*)", n)
            C = [LET.index(c) for c in m.group(1).split(", ")] if m else None
            yp = max(C, key=lambda i: (sc[i], i == x, -i)) if C else y
            yr = x_r if not C or x_r in C else random.Random(f"{it['id']}{p}").choice(C)
            if f"{it['id']}|{p}" not in lex.d:
                lex.put(f"{it['id']}|{p}", LET[y], "")
                lexp.put(f"{it['id']}|{p}", LET[yp], "")
                rnd.put(f"{it['id']}|{p}", LET[yr], "")
    for it in items:
        if f"{it['id']}|s1" in em.d and all(f"{it['id']}|{p}" in em.d for p in PAYLOADS):
            continue
        base = (it.get("article", "")[:1500] + " " + it["q"])
        opts = embed(it["opts"])
        def pick(text):
            q = embed([text])[0]
            return torch.nn.functional.cosine_similarity(q[None], opts).tolist()
        s0 = pick(base)
        x = max(range(4), key=lambda i: s0[i])
        em.put(f"{it['id']}|s1", LET[x], "")
        for p in PAYLOADS:
            n = S2.note(it, p, pay, items)
            s = pick(base + " " + n) if n else s0
            em.put(f"{it['id']}|{p}", LET[max(range(4), key=lambda i: s[i])], "")


def pull():
    import subprocess
    for m in list(TINY.values()) + ["all-minilm"]:
        print(m, subprocess.run(["ollama", "pull", m], capture_output=True, text=True).returncode, flush=True)


def runall(items):
    os.makedirs(OUT, exist_ok=True)
    pay = json.load(open(S2.PAY2))
    start = RUN.balance()
    run_rules(items, pay)
    print("规则节点完成", flush=True)
    with cf.ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda n: run_llm(n, items, pay, lambda a, t: ask_tiny(a, t), 1), list(TINY)))
    print("本地小模型完成", flush=True)
    for n in LLM:
        run_llm(n, items, pay, ask_ds, 24)
        print(n, f"已花 ¥{start - RUN.balance():.2f}", flush=True)
        if start - RUN.balance() > 10:
            print("超过上限，停止", flush=True)
            return


def load():
    items = {x["id"]: x for x in json.load(open(S2.ITEMS2))}
    R, invalid = [], {}
    for node in NODES:
        if not os.path.exists(os.path.join(OUT, node + ".jsonl")):
            continue
        s = Store(node)
        bad = tot = 0
        for k, (v, _) in s.d.items():
            tot += 1
            bad += v is None
        invalid[node] = bad / max(tot, 1)
        for i, it in items.items():
            f = s.d.get(f"{i}|s1", (None,))[0]
            if f is None:
                continue
            x = LET.index(f)
            for p in PAYLOADS:
                y_ = s.d.get(f"{i}|{p}", (None,))[0]
                if y_ is None:
                    continue
                y = LET.index(y_)
                w, g = S2.wrong(it), it["gold"]
                tg = {"set2t": {g, w[0]}, "set2t_r": {g, w[0]}, "set2f": set(w[:2]), "set2f_r": set(w[:2]), "L3t": {g}, "L4t": {g}, "L3f": {w[0]}, "L4f": {w[0]}, "L2f": {w[0]}}.get(p)
                R.append(dict(id=i, hard=it["hard"], task=it["task"], a=node, p=p, x=x, y=y, g=g, cf=50.0, right=x == g, ok=y == g, changed=y != x, conflict=(x not in tg) if tg else None, adopt=(y in tg) if tg else None))
    return R, invalid, items


m = AN.mean
K, B, mark = S2.K, S2.B, S2.mark


def targets(it, p):
    w, g = S2.wrong(it), it["gold"]
    return {"set2t": {g, w[0]}, "set2t_r": {g, w[0]}, "set2f": set(w[:2]), "set2f_r": set(w[:2]), "L3t": {g}, "L4t": {g}, "L3f": {w[0]}, "L4f": {w[0]}}[p]


def dadj(rs, pt, pf, items):
    blank = {(r["id"], r["a"]): r for r in rs if r["p"] == "blank"}
    v = []
    for p in (pt, pf):
        X = [r for r in rs if r["p"] == p and r["conflict"] and not r["right"] and (r["id"], r["a"]) in blank]
        if not X:
            return float("nan")
        v.append(m(r["adopt"] for r in X) - m(blank[r["id"], r["a"]]["y"] in targets(items[r["id"]], p) for r in X))
    return v[0] - v[1]


def vs(y, pt="set2t", pf="set2f"):
    b = m(r["ok"] for r in y if r["p"] == "blank")
    t, f = m(r["ok"] for r in y if r["p"] == pt) - b, m(r["ok"] for r in y if r["p"] == pf) - b
    return 1.0 if t <= 0 else max(min(-f / (t - f), 1.0), 0.0)


def analyze():
    R4, invalid, items = load()
    S = json.load(open(os.path.join(HERE, "rows_s2.json")))
    S = [r for r in S if r["a"] in TOP2]
    R = R4 + S
    nat = [r for r in R if not r["hard"]]
    nodes = NODES + TOP2
    acc = {a: m(r["right"] for r in nat if r["a"] == a and r["p"] == "blank") for a in nodes}
    print("节点（第一次准确率，自然组）｜ 无效输出 ｜ K / B：blank L3f L4f set2f set2f_r / blank L3t L4t set2t set2t_r")
    for a in sorted(nodes, key=lambda z: -acc.get(z, 0)):
        X = [r for r in R if r["a"] == a]
        print(f"  {a:13s}{acc[a]:5.0%} ｜ {invalid.get(a, 0):4.0%} ｜ " + " ".join(f"{K(X, p):4.0%}" for p in ("blank", "L3f", "L4f", "set2f", "set2f_r")) + " ／ " + " ".join(f"{B(X, p):4.0%}" for p in ("blank", "L3t", "L4t", "set2t", "set2t_r")))
    ok_nodes = [a for a in nodes if invalid.get(a, 0) <= .2]
    TOP = [a for a in ok_nodes if acc[a] >= .75]
    FLOOR = [a for a in NODES if a in ok_nodes and acc[a] <= .45 and a not in ("lex_proj", "rand_proj")]
    print(f"\n预检：准确率最高 {max(acc.values()):.0%}（≥ 70%）最低 {min(acc.values()):.0%}（≤ 40%）；TOP = {TOP}；FLOOR = {FLOOR}；无效过多 = {[a for a in nodes if invalid.get(a, 0) > .2]}")
    T_, F_ = lambda y: [r for r in y if r["a"] in TOP], lambda y: [r for r in y if r["a"] in FLOOR]
    A_ = [r for r in R if r["a"] in TOP + FLOOR]
    print("\n== 判据 ==")
    f = lambda y: dadj(T_(y), "L4t", "L4f", items) - dadj(F_(y), "L4t", "L4f", items)
    lo, hi = AN.boot(A_, f, 500)
    print(f"Q1 分辨力 D(L4)：TOP {dadj(T_(R), 'L4t', 'L4f', items):+.3f} FLOOR {dadj(F_(R), 'L4t', 'L4f', items):+.3f} 差 {f(R):+.3f} [{lo:+.3f},{hi:+.3f}] {mark(lo > 0)}")
    g3 = lambda y: B(y, "L3t") - B(y, "L0t")
    g4 = lambda y: B(y, "L4t") - B(y, "L0t")
    f = lambda y: g3(T_(y)) - g3(F_(y))
    lo, hi = AN.boot(A_, f, 500)
    print(f"Q2 门槛：B(L3t) − B(L0t) TOP {g3(T_(R)):+.3f} FLOOR {g3(F_(R)):+.3f} 差 {f(R):+.3f} [{lo:+.3f},{hi:+.3f}]；FLOOR 内 L4 增益 {g4(F_(R)):+.3f} vs L3 {g3(F_(R)):+.3f} {mark(lo > 0 and g4(F_(R)) > g3(F_(R)))}")
    rr = lambda y: K(y, "set2f_r") - K(y, "set2f")
    f = lambda y: rr(T_(y)) - rr(F_(y))
    lo, hi = AN.boot(A_, f, 500)
    print(f"Q3 理由保护：TOP {rr(T_(R)):+.3f} FLOOR {rr(F_(R)):+.3f} 差 {f(R):+.3f} [{lo:+.3f},{hi:+.3f}] {mark(lo > 0)}")
    Xr, Xl = [r for r in R if r["a"] == "rand_proj"], [r for r in R if r["a"] == "lex_proj"]
    ww = [r for r in Xr if r["p"] == "set2t" and not r["right"]]
    bexp = st.mean(.5 if r["conflict"] else 0 for r in ww)
    b0 = m(r["ok"] for r in Xr if r["p"] == "blank")
    nt, nf = m(r["ok"] for r in Xr if r["p"] == "set2t") - b0, m(r["ok"] for r in Xr if r["p"] == "set2f") - b0
    eps = 1 - m(r["right"] for r in Xr if r["p"] == "blank")
    ok4 = all(abs(v - t) <= .02 for v, t in ((K(Xr, "set2t"), 1), (K(Xr, "set2f"), 0), (K(Xl, "set2t"), 1), (K(Xl, "set2f"), 0))) and abs(B(Xr, "set2t") - bexp) <= .05
    print(f"Q4 规则节点：rand K {K(Xr, 'set2t'):.0%}/{K(Xr, 'set2f'):.0%} lex_proj K {K(Xl, 'set2t'):.0%}/{K(Xl, 'set2f'):.0%}；rand B(set2t) {B(Xr, 'set2t'):.0%} vs 算术 {bexp:.0%}；ε {eps:.0%} net_t {nt:+.3f} net_f {nf:+.3f} {mark(ok4)}")
    cand = [a for a in ("flash_m80", "tiny_qwen", "tiny_llama", "tiny_smol") if a in ok_nodes]
    hits = [K([r for r in R if r["a"] == a], "set2f") <= K([r for r in R if r["a"] == a], "L3f") - .10 for a in cand]
    print(f"Q5 形式：" + " ".join(f"{a} set2f {K([r for r in R if r['a'] == a], 'set2f'):.0%} L3f {K([r for r in R if r['a'] == a], 'L3f'):.0%}" for a in cand) + f" {mark(sum(hits) * 2 >= max(len(cand), 1))}")
    eps_n = [1 - m(r["right"] for r in R if r["a"] == a and r["p"] == "blank") for a in ok_nodes]
    v_n = [vs([r for r in R if r["a"] == a]) for a in ok_nodes]
    print(f"Q6 ε × v*：Spearman {S2.spearman(eps_n, v_n):.2f}（≤ −0.5）{mark(S2.spearman(eps_n, v_n) <= -.5)}；" + " ".join(f"{a} {e:.0%}→{v:.2f}" for a, e, v in zip(ok_nodes, eps_n, v_n)))
    print("\n== 剂量曲线（只记录）== 错节点 B：blank L0t L2t L3t L4t set2t set2t_r ｜ 对节点 K：blank L2f L3f L4f set2f set2f_r ｜ 门槛")
    for a in sorted(nodes, key=lambda z: -acc[z]):
        X = [r for r in R if r["a"] == a]
        print(f"  {a:13s}" + " ".join(f"{B(X, p):4.0%}" for p in ("blank", "L0t", "L2t", "L3t", "L4t", "set2t", "set2t_r")) + " ｜ " + " ".join(f"{K(X, p):4.0%}" for p in ("blank", "L2f", "L3f", "L4f", "set2f", "set2f_r")))
    json.dump(R4, open(os.path.join(HERE, "rows_s4.json"), "w"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    all_items = json.load(open(S2.ITEMS2))
    if cmd == "pull":
        pull()
    elif cmd == "pilot":
        runall(all_items[:12])
    elif cmd == "run":
        runall(all_items)
    elif cmd == "analyze":
        analyze()
