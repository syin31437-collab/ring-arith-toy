"""跨领域检验（2026-11-04，库伯同意）：带着冻结的坐标和两条轴，不改轴、不加公式，去开放任务和非 LLM 节点。

冻结：axes.json（bac2ea1a）的均值、载荷、坐标。ds_flash_b 按同一流程补坐标写进 axes2.json，其余数字不动。
上一轮（hardn.py）绝对值预测不过，所以这里只判排序。探索发现的三条在这里正式预注册：假理由暴露的配对差、强弱节点的最佳投放。

臂 A 开放任务（没有选项，自由输出；判分程序或裁判，同第三阶段）：
  题：code（HumanEval + MBPP）100、qa（SimpleQA 80 + TruthfulQA 40）120、case（法律、医学去选项）100、plan（9 人排班新题）50；全部没进过任何投放实验。
  Sonnet、Kimi、Grok 各答一次筛选。每类 15 道（共 60）：优先轮流取各大模型答错的题，不够用其余题随机补。
  qa、case 的候选由 DeepSeek Pro 和 Codex 审参考答案，两个都说错的剔除；只有一个说错的保留并标记。
  节点：Sonnet Kimi Grok qwen14 qwen9 ds_flash_a ds_flash_b ds_pro_a claude_haiku codex_ns flash_m65（遮 65% 题面）共 11 个，全部禁工具、禁联网。所有节点重新答第一次。
  投放：blank L3t L3f L4t L4f set2t set2f set2t_r set2f_r（第三阶段同一提示词，Sonnet 写）。
  映射预检（看不到投放效果；不过的任务类只报告、不计入判据）：裁判 Sonnet vs Codex 复评 kappa ≥ 0.7；该类全节点第一次错误率 5–85%（这批题是挑难的，上限放宽）；第一次答对的节点 blank 下保住 ≥ 85%。
  替代量（开放任务算不了采纳率，已在 rows_h.json 上验证：S×D_w .89，A×A_set .72）：
    S = (B(L3t)+B(L4t))/2 + (K(L3f)+K(L4f))/2 − 1；A = [B(set2t)+B(set2t_r)+(1−K(set2f))+(1−K(set2f_r))]/4
  节点要有 ≥ 10 个第一次对的题和 ≥ 10 个第一次错的题才进相关。可用 < 8 个算没结论。
  M0：Sonnet、Kimi、Grok 里 ≥ 2 个第一次错 ≥ 20%（≥ 12 题）。
  A1 ρ(z2, S) ≥ 0.6，且比 |ρ(z1, S)| 高 ≥ 0.2。
  A2 ρ(z1, A) ≥ 0.5，且比 |ρ(z2, A)| 高 ≥ 0.2。
  A3 假理由暴露：配对差 K(set2f_r) − K(set2f)（同题）；ρ(z2, 配对差) ≥ 0.5，且 Sonnet Kimi Grok Haiku Codex 里 ≥ 4 个 ≥ +0.10。
  A4 该投什么：z2 ≥ 2 的节点 v*(L3) < v*(set2)；z2 < 2 的节点 v*(L4) ≤ v*(L3)。
  另报不判：每类任务分开的 ρ；去掉 Sonnet；去掉标记题；校准准确率、校准真假差、校准集合采纳这些简单量的 ρ。

臂 B 非 LLM（代码和映射预检见 sensors.py、cross_preb.py）：
  数据：Vehicle（846 行、18 特征、4 类）。546 行训练；其余 300 行里只留全视野 lr、knn 都答对的行（"关键前提确实能推出答案"），前 40 行校准（只用 blank L3t L3f set2t set2f），其余预测（9 种投放）。
  节点 24 个 = 3 学习器 × 2 视野 × 4 规则。坐标用冻结的 m0、V 和校准数据按同一 MAP 算。公开特征 8 个由训练折贪心选定。
  预检记录（内部切分，没碰这 300 行）：第一版公开 4 个共线特征，真前提反而伤人（lr6 K(L3t) .43）→ 改贪心选特征、lr 训练时随机遮特征、只留全视野答对的行；k = 8 时 K(L3t) .88，B(L3t) .35，K(L3f) .66。正式数据上再查一遍：K(L3t) ≥ .85、B(L3t) ≥ .30、K(L3t) − K(L3f) ≥ .10，不过就算臂 B 映射失败、不判。
  B1 ρ(z2, D_w) ≥ 0.6，且比 |ρ(z1, D_w)| 高 ≥ 0.2（错题 ≥ 15 的节点）。
  B2 ρ(z1, A_set) ≥ 0.5，且比 |ρ(z2, A_set)| 高 ≥ 0.2。
  B3 规则落点：chk3 的平均 z2 > proj 的平均 z2；proj 的平均 z1 > chk3 的平均 z1。
  另报不判：简单量的 ρ；按规则、学习器的均值；置换 p（节点不独立，只作参考）。
  局限（写在前面）：规则是我写的，而且知道两条轴是什么，chk 规则本身就带"用自己的判断筛"，所以 B 不能证明自然系统有这两条轴，只检验"LLM 上冻结的尺子能不能度量并预测非 LLM 节点"。

决定：A1、A2 都过 → 排序跨到开放任务；B1、B2 都过 → 尺子跨到非 LLM。两个都过才说"统一方向"得到跨领域支持；否则只说它在选择题 LLM 上成立。
任何情况不改轴、不补公式。停止规则：DeepSeek 上限 ¥15。多重比较不校正，全部报告。

用法：~/gut-external/torch215-venv/bin/python cross.py freeze | b | screen | referee | select | gen | run | a
"""
import concurrent.futures as cf
import csv
import gzip
import json
import math
import os
import random
import re
import subprocess
import sys
import urllib.request

import numpy as np
import torch

import formula as FM
import generality as G
import hardn as H
import newnodes as N
import run as RUN
import sensors as SE
import stage2 as S2
import stage3 as S3
import stage4 as S4

HERE = RUN.HERE
AX2 = os.path.join(HERE, "axes2.json")
OUT = os.path.join(HERE, "runs_x")
SCRN = os.path.join(HERE, "screen_x")
POOL = os.path.join(HERE, "poolX.json")
REF = os.path.join(HERE, "refereeX.json")
ITEMS = os.path.join(HERE, "itemsX.json")
PAY = os.path.join(HERE, "payloadsX.json")
GRADE = os.path.join(HERE, "gradesX.json")
BIG = H.BIG
FRONT = BIG + ["claude_haiku", "codex_ns"]
NODES = BIG + ["qwen14", "qwen9", "ds_flash_a", "ds_flash_b", "ds_pro_a", "claude_haiku", "codex_ns", "flash_m65"]
FORMS = H.FORMS
FAMS = ("code", "qa", "case", "plan")
PER = 15
LIMIT = 15.0
PAR = {"qwen14": 1, "qwen9": 1, "ds_flash_a": 8, "ds_flash_b": 8, "ds_pro_a": 8, "flash_m65": 8, "codex_ns": 4}
S3.OUT, S3.AGENTS, S3.PAYLOADS, S3.ITEMS, S3.PAY, S3.GRADE = OUT, NODES, FORMS, ITEMS, PAY, GRADE
m = lambda xs: (lambda v: sum(v) / len(v) if v else float("nan"))(list(xs))


def axes():
    A = json.load(open(H.AXES))
    if os.path.exists(AX2):
        A["z"].update(json.load(open(AX2)))
    return A


def freeze():
    torch.set_num_threads(4)
    A = json.load(open(H.AXES))
    m0, V = torch.tensor(A["m0"]), torch.tensor(A["V"])
    B = torch.stack([G.fit_lin(G.basis(G.tens(G.BY[n])), G.tens(G.BY[n])["y"]) for n in G.NODES])
    Sv = torch.linalg.svdvals(B - B.mean(0))
    s = Sv[:2] ** 2 / (len(G.NODES) - 1) + .1
    z = H.map_z(G.tens(G.split("ds_flash_b")[0]), m0, V, s).tolist()
    json.dump({"ds_flash_b": z, "scale": s.tolist()}, open(AX2, "w"))
    print("ds_flash_b", z, "尺度", s.tolist())


def arm_b():
    torch.set_num_threads(4)
    A = json.load(open(H.AXES))
    m0, V = torch.tensor(A["m0"]), torch.tensor(A["V"])
    s = torch.tensor(json.load(open(AX2))["scale"])
    X, y = SE.load(os.path.join(HERE, "raw3", "vehicle.arff"))
    order = list(range(len(y)))
    random.Random("rel-vehicle").shuffle(order)
    tr, rest = order[:546], order[546:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    Xt, yt = Z[tr], y[tr]
    top = SE.info_order(Xt, yt, 8)
    full = np.ones(X.shape[1], dtype=bool)
    oracle = [SE.Model(k, Xt, yt) for k in ("lr", "knn")]
    keep = [i for i in rest if all(int(np.argmax(o.p(Z[i], full))) == y[i] for o in oracle)]
    print(f"公开特征 {top}；300 行里全视野答对 {len(keep)}；校准 40，预测 {len(keep) - 40}")
    cal, test = keep[:40], keep[40:]
    nodes = SE.build_nodes(Xt, yt, "formal")
    items = {}
    pays = {}
    for i in keep:
        idt = f"V{i}"
        items[idt] = dict(id=idt, gold=int(y[i]), task="reason", hard=False, cal=i in cal, q="", opts=[""] * 4, article="")
        pays[idt] = SE.payload(idt, Z[i], int(y[i]), Xt, yt, top)
    G.items.update(items)
    FM.items.update(items)

    def rows(name, ix, forms):
        rl, mod, seen = nodes[name]
        out = []
        for i in ix:
            idt = f"V{i}"
            it = items[idt]
            for p in forms:
                x, cf_, a = SE.answer(rl, mod, Z[i], seen, top, pays[idt], p)
                tg = FM.target(it, p)
                out.append(dict(id=idt, a=name, p=p, x=x, y=a, g=it["gold"], cf=float(cf_), right=x == it["gold"], ok=a == it["gold"], task="reason", hard=False, cal=it["cal"], changed=a != x, conflict=(x not in tg) if tg else None, adopt=(a in tg) if tg else None))
        return out
    T = {}
    lines = []
    allrows = []
    for name in nodes:
        rc, rt = rows(name, cal, sorted(G.CAL_P)), rows(name, test, FORMS)
        allrows += rt
        z = H.map_z(G.tens(rc), m0, V, s).tolist()
        cw = lambda q: m(r["adopt"] for r in rt if r["p"] == q and not r["right"] and r["conflict"])
        ca = lambda q: m(r["adopt"] for r in rt if r["p"] == q and r["conflict"])
        cc = lambda q, rs: m(r["adopt"] for r in rs if r["p"] == q and r["conflict"])
        T[name] = dict(z1=z[0], z2=z[1], Dw=(cw("L3t") - cw("L3f") + cw("L4t") - cw("L4f")) / 2, Aset=m(ca(q) for q in ("set2t", "set2f", "set2t_r", "set2f_r")), nw=sum(not r["right"] for r in rt if r["p"] == "blank"), acc=m(r["right"] for r in rc if r["p"] == "blank"), dcal=cc("L3t", rc) - cc("L3f", rc), acal=m(cc(q, rc) for q in ("set2t", "set2f")))
        lines.append(f"  {name:14s} z1 {z[0]:+6.2f} z2 {z[1]:+6.2f} ｜ D_w {T[name]['Dw']:+.2f} A_set {T[name]['Aset']:.2f} 错 {T[name]['nw']:3d} ｜ 校准准确率 {T[name]['acc']:.2f}")
    print("\n".join(lines))
    json.dump(allrows, open(os.path.join(HERE, "rows_xb.json"), "w"))
    K3t, B3t, K3f = m(S2.K([r for r in allrows if r["a"] == a], "L3t") for a in nodes), m(S2.B([r for r in allrows if r["a"] == a], "L3t") for a in nodes), m(S2.K([r for r in allrows if r["a"] == a], "L3f") for a in nodes)
    ok = K3t >= .85 and B3t >= .30 and K3t - K3f >= .10
    print(f"\n映射复查 K(L3t) {K3t:.2f} B(L3t) {B3t:.2f} K(L3f) {K3f:.2f} {S2.mark(ok)}")
    ns = [a for a in nodes if T[a]["nw"] >= 15 and not math.isnan(T[a]["Dw"])]
    r = lambda k, v, q: S2.spearman([T[a][k] for a in q], [T[a][v] for a in q])
    b1 = r("z2", "Dw", ns) >= .6 and r("z2", "Dw", ns) - abs(r("z1", "Dw", ns)) >= .2
    aa = list(nodes)
    b2 = r("z1", "Aset", aa) >= .5 and r("z1", "Aset", aa) - abs(r("z2", "Aset", aa)) >= .2
    by = lambda k, t: m(T[a][k] for a in nodes if a.endswith("_" + t))
    b3 = by("z2", "chk3") > by("z2", "proj") and by("z1", "proj") > by("z1", "chk3")
    print(f"B1 ρ(z2,D_w) {r('z2', 'Dw', ns):+.2f} ρ(z1,D_w) {r('z1', 'Dw', ns):+.2f}（{len(ns)} 节点）{S2.mark(b1)}")
    print(f"B2 ρ(z1,A_set) {r('z1', 'Aset', aa):+.2f} ρ(z2,A_set) {r('z2', 'Aset', aa):+.2f} {S2.mark(b2)}")
    print(f"B3 规则均值 " + " ".join(f"{t}: z1 {by('z1', t):+.2f} z2 {by('z2', t):+.2f} D_w {by('Dw', t):+.2f} A_set {by('Aset', t):.2f}；" for t in SE.RULES) + S2.mark(b3))
    print("另报 简单量：D_w ← 准确率 %+.2f、校准真假差 %+.2f；A_set ← 校准集合采纳 %+.2f" % (r("acc", "Dw", ns), r("dcal", "Dw", ns), r("acal", "Aset", aa)))
    g = random.Random(1)
    perm = lambda k, v, q: sum(S2.spearman([T[a][k] for a in q], g.sample([T[a][v] for a in q], len(q))) >= r(k, v, q) for _ in range(3000)) / 3000
    print(f"置换 p（节点不独立，只作参考）B1 {perm('z2', 'Dw', ns):.3f} B2 {perm('z1', 'Aset', aa):.3f}")
    for l in SE.LEARN:
        print(f"  学习器 {l}: ρ(z2,D_w) {r('z2', 'Dw', [a for a in ns if a.startswith(l)]):+.2f} ρ(z1,A_set) {r('z1', 'Aset', [a for a in aa if a.startswith(l)]):+.2f}")
    print("决定 臂 B：" + S2.mark(ok and b1 and b2))


def ask(node, text):
    if node in ("claude_sonnet", "claude_haiku"):
        return subprocess.run(["claude", "-p", "--model", node.split("_")[1], "--tools", ""], input=text, cwd=N.SCR, capture_output=True, text=True, timeout=900).stdout, 0, 0
    if node in ("kimi", "grok"):
        t, n = N.ask_cli(node, text)
        return t, 0, n
    if node in ("qwen14", "qwen9"):
        model = {"qwen14": "qwen2.5:14b", "qwen9": "qwen3.5:9b"}[node]
        body = json.dumps({"model": model, "stream": False, "think": False, "messages": [{"role": "user", "content": text}], "options": {"temperature": 0, "num_predict": 1200, "num_ctx": 8192}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=900))["message"]["content"], 0, 0
    return PLAIN("ds_flash_a" if node == "flash_m65" else node, text)


PLAIN = S3.plain
S3.plain = ask


def used_q():
    out = set()
    for f in ("items.json", "items2.json", "items3.json", "pool3.json", "itemsN.json", "itemsH.json"):
        out |= {x["q"] for x in json.load(open(os.path.join(HERE, f))) if isinstance(x, dict) and "q" in x}
    return out


def build_pool():
    rng = random.Random("rel-cross")
    used = used_q()
    raw = S3.RAW
    he = [json.loads(l) for l in gzip.open(os.path.join(raw, "HumanEval.jsonl.gz"), "rt")]
    mb = json.load(open(os.path.join(raw, "mbpp.jsonl")))
    rng.shuffle(he)
    rng.shuffle(mb)
    code = [dict(family="code", source="humaneval", q="Complete this Python function.\n" + r["prompt"], gold=r["canonical_solution"], test=r["test"] + f"\ncheck({r['entry_point']})\n") for r in he]
    code += [dict(family="code", source="mbpp", q=r["prompt"] + "\nYour function must work with calls like: " + r["test_list"][0], gold=r["code"], test="\n".join(r.get("test_imports") or []) + "\n" + "\n".join(r["test_list"]) + "\n") for r in mb]
    code = [k for k in code if k["q"] not in used]
    rng.shuffle(code)
    sq = list(csv.DictReader(open(os.path.join(raw, "simple_qa.csv"))))
    tq = list(csv.DictReader(open(os.path.join(raw, "TruthfulQA.csv"))))
    rng.shuffle(sq)
    rng.shuffle(tq)
    qa = [dict(family="qa", source="simpleqa", q=r["problem"], gold=r["answer"], wrong="") for r in sq if r["problem"] not in used][:80]
    qa += [dict(family="qa", source="truthfulqa", q=r["Question"], gold=r["Best Answer"] + " (Also acceptable: " + r["Correct Answers"] + ". Incorrect: " + r["Incorrect Answers"] + ")", wrong=r["Best Incorrect Answer"]) for r in tq if r["Question"] not in used][:40]
    case = []
    for path, name in [(os.path.join(HERE, "raw", n + ".json"), n) for n in ("professional_law", "clinical")] + [(os.path.join(HERE, "raw2", n + ".json"), n) for n in ("jurisprudence", "professional_medicine", "college_medicine")]:
        rows = json.load(open(path))
        rng.shuffle(rows)
        for r in rows:
            opts = r.get("choices") or r.get("options")
            if r["question"] in used or "following" in r["question"].lower() or len(opts) != 4 or len(r["question"]) > 1200 or any(RUN.BAD_OPT.search(o) for o in opts):
                continue
            g = r["answer"] if isinstance(r["answer"], int) else RUN.LET.index(r["answer"])
            case.append(dict(family="case", source=name, q=r["question"], gold=opts[g], wrong=random.Random(r["question"]).choice([o for i, o in enumerate(opts) if i != g])))
    rng.shuffle(case)
    g = random.Random("plan-cross")
    plan = [dict(family="plan", **S3.plan9(g)) for _ in range(50)]
    out = code[:100] + qa + case[:100] + plan
    for i, k in enumerate(out):
        k["id"] = f"X{i:03d}"
        k["hard"] = True
    return out


def screen():
    os.makedirs(SCRN, exist_ok=True)
    os.makedirs(N.SCR, exist_ok=True)
    open(N.AGENT, "w").write("---\nname: plain\ndescription: answer directly with no tools\ntools: []\n---\nYou answer questions directly from what you know. You have no tools.\n")
    pool = build_pool()
    from collections import Counter
    print(Counter(k["family"] for k in pool), Counter(k["source"] for k in pool), flush=True)

    def one(j):
        node, it = j
        return S3.call(node, S3.statement(it), os.path.join(SCRN, f"{it['id']}_{node}.json"))
    jobs = [(n, it) for n in BIG for it in pool]
    random.Random("x").shuffle(jobs)
    with cf.ThreadPoolExecutor(12) as ex:
        list(ex.map(one, jobs))
    S3.GRADE = os.path.join(HERE, "gradesXscreen.json")
    rows = []
    for it in pool:
        for n in BIG:
            f = os.path.join(SCRN, f"{it['id']}_{n}.json")
            if os.path.exists(f):
                rows.append((f"{it['id']}|{n}|s", it, json.load(open(f))["text"]))
    Gd = S3.grade(rows)
    e = {n: sorted(it["id"] for it in pool if Gd.get(f"{it['id']}|{n}|s") == 0) for n in BIG}
    print("评分", len(Gd), "/", len(rows), {n: (len(v), Counter(i["family"] for i in pool if i["id"] in v)) for n, v in e.items()}, flush=True)
    judged = [(k, it, t) for k, it, t in rows if it["family"] in ("qa", "case")]
    random.Random("k").shuffle(judged)
    sample = judged[:150]
    with cf.ThreadPoolExecutor(6) as ex:
        res = {}
        for v in ex.map(lambda i: S3.judge_batch(sample[i:i + 6], "codex_ns"), range(0, len(sample), 6)):
            res.update(v)
    both = [(Gd[k], res[k]) for k in res if k in Gd]
    agree = m(a == b for a, b in both)
    pa, pb = m(a for a, _ in both), m(b for _, b in both)
    pe = pa * pb + (1 - pa) * (1 - pb)
    print(f"裁判复评 {len(both)}：一致 {agree:.0%} kappa {(agree - pe) / (1 - pe):.2f}（≥ 0.7）")
    json.dump(dict(pool=pool, errs=e, kappa=(agree - pe) / (1 - pe)), open(POOL, "w"), ensure_ascii=False)


def referee():
    P = json.load(open(POOL))
    by = {k["id"]: k for k in P["pool"]}
    cand = sorted({q for v in P["errs"].values() for q in v if by[q]["family"] in ("qa", "case")})
    groups = [cand[i:i + 3] for i in range(0, len(cand), 3)]
    have = {}

    def one(j):
        node, g = j
        fn = os.path.join(SCRN, f"ref_{node}_{g[0]}.json")
        if os.path.exists(fn):
            return node, json.load(open(fn))
        txt = ("You are auditing the reference answers of an exam. For each question below, decide whether the reference answer is correct. Check carefully and independently. "
               'Reply with ONLY a JSON object mapping each question ID to {"verdict": "correct" | "wrong" | "unsure"}.\n\n' +
               "\n\n".join(f"{i}: {by[i]['q']}\nReference answer: {by[i]['gold']}" for i in g))
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
    print("候选", len(cand), "缺审", sum(len(have.get(q, {})) < 2 for q in cand), Counter(tuple(sorted(v.items())) for v in have.values()).most_common(10))


def select():
    P = json.load(open(POOL))
    ref = json.load(open(REF))
    both = {q for q, v in ref.items() if v.get("codex") == "wrong" and v.get("ds_pro") == "wrong"}
    one = {q for q, v in ref.items() if q not in both and "wrong" in v.values()}
    g = random.Random("x-sel")
    chosen = []
    for fam in FAMS:
        ids = [k["id"] for k in P["pool"] if k["family"] == fam and k["id"] not in both]
        rest = {n: [q for q in P["errs"][n] if q in ids] for n in BIG}
        for v in rest.values():
            g.shuffle(v)
        pick = []
        while len(pick) < PER and any(rest.values()):
            for n in BIG:
                while rest[n] and rest[n][0] in pick:
                    rest[n].pop(0)
                if rest[n] and len(pick) < PER:
                    pick.append(rest[n].pop(0))
        fill = [q for q in ids if q not in pick]
        g.shuffle(fill)
        pick += fill[:PER - len(pick)]
        chosen += pick
    by = {k["id"]: k for k in P["pool"]}
    items = [dict(by[q], flagged=q in one) for q in chosen]
    json.dump(items, open(ITEMS, "w"), ensure_ascii=False, indent=0)
    from collections import Counter
    print(f"两审都说错剔除 {len(both)} 标记 {len(one)} 入选 {len(items)}（标记 {sum(i['flagged'] for i in items)}）", Counter(i["source"] for i in items))
    print({n: Counter(by[q]["family"] for q in chosen if q in P["errs"][n]) for n in BIG})


def gen():
    S3.gen()


def view(node, it):
    return dict(it, q=S4.mask(it["q"], .65, it["id"] + "q")) if node == "flash_m65" else it


def run_node(node, items, pay, stop):
    par = PAR.get(node, 4)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(lambda it: S3.call(node, S3.statement(view(node, it)), S3.first_path(it["id"], node)), items))

    def second(j):
        it, p = j
        if not os.path.exists(S3.first_path(it["id"], node)) or (stop[0] and node.startswith(("ds_", "flash"))):
            return
        S3.call(node, S3.second_text(view(node, it), node, p, pay, items), S3.second_path(it, node, p))
    jobs = [(it, p) for it in items for p in FORMS]
    random.Random(node).shuffle(jobs)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(second, jobs))
    print(node, "完成", flush=True)


def run_all():
    os.makedirs(OUT, exist_ok=True)
    items = json.load(open(ITEMS))
    pay = json.load(open(PAY))
    start = RUN.balance()
    stop = [False]

    def watch():
        import time
        while not stop[1:]:
            time.sleep(120)
            try:
                if start - RUN.balance() > LIMIT:
                    stop[0] = True
                    print("超过上限，停 DeepSeek 节点", flush=True)
            except OSError:
                pass
    import threading
    threading.Thread(target=watch, daemon=True).start()
    with cf.ThreadPoolExecutor(len(NODES)) as ex:
        list(ex.map(lambda n: run_node(n, items, pay, stop), NODES))
    S3.grade(S3.gather(items, FORMS))
    print("全部完成", flush=True)


def analyze_a():
    A = axes()
    z = A["z"]
    items = {x["id"]: x for x in json.load(open(ITEMS))}
    P = json.load(open(POOL))
    R, miss, tot = S3.load()
    json.dump(R, open(os.path.join(HERE, "rows_xa.json"), "w"))
    print(f"记录 {len(R)}，缺失 {miss}（{miss / max(tot, 1):.1%}）；裁判 kappa {P['kappa']:.2f}")
    fam_ok = {}
    for f in FAMS:
        X = [r for r in R if r["family"] == f and r["p"] == "blank"]
        e, k0 = m(not r["right"] for r in X), S2.K(X, "blank")
        fam_ok[f] = .05 <= e <= .85 and k0 >= .85 and (f not in ("qa", "case") or P["kappa"] >= .7)
        print(f"预检 {f}: 第一次错 {e:.0%} blank K {k0:.0%} B {S2.B(X, 'blank'):.0%} {S2.mark(fam_ok[f])}")
    Q = [r for r in R if fam_ok[r["family"]]]
    print("\nM0")
    okb = 0
    for a in NODES:
        X = [r for r in Q if r["a"] == a and r["p"] == "blank"]
        w = sum(not r["right"] for r in X)
        okb += a in BIG and w >= 12 and w / max(len(X), 1) >= .2
        print(f"  {a:14s} 错 {w:3d}/{len(X)} = {w / max(len(X), 1):.0%}")
    print(f"M0 大模型达标 {okb}/3 {S2.mark(okb >= 2)}")

    def table(rs, ns):
        T = {}
        for a in ns:
            X = [r for r in rs if r["a"] == a]
            nr, nw = sum(r["right"] for r in X if r["p"] == "blank"), sum(not r["right"] for r in X if r["p"] == "blank")
            if nr < 10 or nw < 10:
                continue
            s = (S2.B(X, "L3t") + S2.B(X, "L4t")) / 2 + (S2.K(X, "L3f") + S2.K(X, "L4f")) / 2 - 1
            aa = (S2.B(X, "set2t") + S2.B(X, "set2t_r") + 1 - S2.K(X, "set2f") + 1 - S2.K(X, "set2f_r")) / 4
            Y = {(r["id"], r["p"]): r for r in X if r["right"]}
            d = m(Y[(i, "set2f_r")]["ok"] - Y[(i, "set2f")]["ok"] for i, p in Y if p == "set2f" and (i, "set2f_r") in Y)
            T[a] = dict(S=s, A=aa, d=d, nr=nr, nw=nw, z1=z[a][0], z2=z[a][1])
        return T

    def rho(T, x, y):
        q = list(T)
        return S2.spearman([T[a][x] for a in q], [T[a][y] for a in q]) if len(q) >= 3 else float("nan")

    def line(T, lab):
        a1, a2 = rho(T, "z2", "S"), rho(T, "z1", "A")
        o1, o2 = rho(T, "z1", "S"), rho(T, "z2", "A")
        p1, p2 = a1 >= .6 and a1 - abs(o1) >= .2, a2 >= .5 and a2 - abs(o2) >= .2
        print(f"[{lab}] {len(T)} 节点 A1 ρ(z2,S) {a1:+.2f} ρ(z1,S) {o1:+.2f} {S2.mark(p1)}；A2 ρ(z1,A) {a2:+.2f} ρ(z2,A) {o2:+.2f} {S2.mark(p2)}")
        return p1, p2
    T = table(Q, NODES)
    print("\n节点        z1     z2  ｜   S     A   配对差 （对/错）")
    for a, t in T.items():
        print(f"  {a:14s} {t['z1']:+6.2f} {t['z2']:+6.2f} ｜ {t['S']:+.2f} {t['A']:.2f} {t['d']:+.2f} （{t['nr']}/{t['nw']}）")
    print()
    enough = len(T) >= 8
    a1, a2 = line(T, "全部")
    line(table(Q, [a for a in NODES if a != "claude_sonnet"]), "去掉 Sonnet")
    line(table([r for r in Q if not items[r["id"]].get("flagged")], NODES), "去掉标记题")
    for f in FAMS:
        line(table([r for r in R if r["family"] == f], NODES), "只看 " + f)
    a3r = rho(T, "z2", "d")
    fr = sum(T.get(a, {"d": -1})["d"] >= .10 for a in FRONT)
    a3 = a3r >= .5 and fr >= 4
    print(f"\nA3 假理由暴露 ρ(z2,配对差) {a3r:+.2f}；顶级节点 ≥ +0.10 的 {fr}/5 {S2.mark(a3)}")
    strong, weak = [a for a in NODES if z[a][1] >= 2], [a for a in NODES if z[a][1] < 2]

    def vs(ns, t, f):
        nt, nf = [], []
        for a in ns:
            X = [r for r in Q if r["a"] == a]
            if not X:
                continue
            e = m(not r["right"] for r in X if r["p"] == "blank")
            K0, B0 = S2.K(X, "blank"), S2.B(X, "blank")
            net = lambda p: e * (S2.B(X, p) - B0) - (1 - e) * (K0 - S2.K(X, p))
            nt.append(net(t))
            nf.append(net(f))
        a_, b_ = m(nt), m(nf)
        return (-b_ / (a_ - b_) if a_ > 0 > b_ else (0. if b_ >= 0 else 1.)), a_, b_
    print(f"A4 v*（真净、假净）")
    rs = {}
    for lab, ns in (("z2 ≥ 2", strong), ("z2 < 2", weak)):
        rs[lab] = {}
        for t, f in (("L3t", "L3f"), ("L4t", "L4f"), ("set2t", "set2f"), ("set2t_r", "set2f_r")):
            rs[lab][t] = vs(ns, t, f)
            print(f"  {lab} {t:8s} v* {rs[lab][t][0]:.2f} 真净 {rs[lab][t][1]:+.3f} 假净 {rs[lab][t][2]:+.3f}")
    a4 = rs["z2 ≥ 2"]["L3t"][0] < rs["z2 ≥ 2"]["set2t"][0] and rs["z2 < 2"]["L4t"][0] <= rs["z2 < 2"]["L3t"][0]
    print(f"A4 {S2.mark(a4)}")
    print("\n决定")
    print(f"  可用节点够（≥ 8）：{S2.mark(enough)}；M0：{S2.mark(okb >= 2)}")
    print(f"  排序跨到开放任务（A1、A2）：{S2.mark(enough and a1 and a2)}")
    print("\n各节点 K/B（全部任务类）")
    for a in NODES:
        X = [r for r in R if r["a"] == a]
        print(f"  {a:14s} K " + " ".join(f"{p} {S2.K(X, p):4.0%}" for p in ("blank", "L3f", "L4f", "set2f", "set2f_r")) + " ｜ B " + " ".join(f"{p} {S2.B(X, p):4.0%}" for p in ("blank", "L3t", "L4t", "set2t", "set2t_r")))


if __name__ == "__main__":
    {"freeze": freeze, "b": arm_b, "screen": screen, "referee": referee, "select": select, "gen": gen, "run": run_all, "a": analyze_a}[sys.argv[1]]()
