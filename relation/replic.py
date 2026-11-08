"""
新节点复现 + 节点内方差（预注册；followup.py 第三节的实施，库伯 11-08 批准）

一、新节点
  两个与现有六节点家族不同的开源模型，走本地 Ollama、温度 0、JSON 格式，型号在看到任何数据前写死在此文件并提交：
    gemma12 = gemma3:12b（Google）、mistral7 = mistral:7b（Mistral AI）。
  题与前提同扩题轮（itemsS.json / payloadsS.json 297 题）。第一次答案按同协议新采（s1），第二次带回显，形式 13 条：
  扩题轮九条件（blank、set2t/f、cond2t/f、L3t/f、L3ct/f）+ one1t/f + L3nt/f。
  预检同 followup：覆盖 ≥ 90%，K(blank) ≥ .85，第一次对 ≥ 30、错 ≥ 15；不过预检的节点只报不判。
  判据为复现，按节点逐个报方向，不设节点均值门槛（n = 2）：
    R1 出口救集合：K(cond2f) − K(set2f) > 0
    R2 前提比集合安全：K(L3f) − K(set2f) > 0
    R3 前提比集合+出口便宜：v*(L3) < v*(cond2)（截断与原定义都报；以截断为判，原定义并列）
  写法（写死）：一个节点三条全同向 = 该节点复现。两节点都复现 → "新家族复现"；一个 → "部分复现，另一个节点哪条翻了照实写"；
  零 → "不复现"。另报每节点 13 条形式的 K/B，以及 F1/F2（理由、不点名）的方向，只报。
  八节点合并（六旧 + 两新）的 S3 / E1b / F1 / F2 为探索性报告，不改任何已预注册的判定。

二、节点内方差
  Flash-B（唯一 temperature 1 节点）用扩题轮原第一次答案，把 blank / set2t/f / cond2t/f / L3t/f 的第二次再跑两遍
  （runs_rp_fb2、runs_rp_fb3），与原跑（runs_s）合成三次。只报：每条形式 K、B 的三次极差与标准差，
  以及 K(cond2f)−K(set2f)、K(L3f)−K(set2f)、v*(L3)−v*(cond2) 三条差值的三次取值；任一差值变号则论文局限按变号写。
  这量的是第二次作答的采样方差，不含第一次作答的方差。

预算：DeepSeek 上限 10；本地免费。
用法：~/gut-external/torch215-venv/bin/python replic.py run [gemma12 mistral7 | fb] | a | fb
"""
import concurrent.futures as cf
import glob
import json
import os
import random
import shutil
import sys

import followup as FU
import hardn as H
import run as RUN
import scale as X
import wording as W

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs_rp")
FB = [os.path.join(HERE, "runs_rp_fb2"), os.path.join(HERE, "runs_rp_fb3")]
NEW = {"gemma12": "gemma3:12b", "mistral7": "mistral:7b"}
X.LOCAL.update(NEW)
H.PAR.update({k: 1 for k in NEW})
ITEMS, PAY = X.ITEMS, X.PAY
FORMS = list(X.FORMS) + ["one1t", "one1f", "L3nt", "L3nf"]
FBFORMS = ["blank", "set2t", "set2f", "cond2t", "cond2f", "L3t", "L3f"]
OLD6 = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns", "qwen14", "llama8"]
LIMIT = 10.0
m = W.m


def second_jobs(node, items, pay, first, forms, out):
    ids = {x["id"] for x in items}
    bs = RUN.batches(items)

    def do_second(j):
        bid, g, p = j
        parts = []
        for it in g:
            e = first.get(it["id"], {}) if isinstance(first.get(it["id"], {}), dict) else {}
            n = FU.note(it, p, pay, items)
            echo = f"\nYour earlier answer: {e.get('answer', '?')} (confidence {e.get('confidence', '?')}). Your earlier reason: {e.get('reason', '')}"
            parts.append(RUN.item_text(it) + echo + (f"\nA note shared with the whole team: {n}" if n else ""))
        return H.call(node, "Here are questions you answered earlier. Give your answer again.\n\n" + "\n\n".join(parts) + "\n\n" + RUN.TAIL, os.path.join(out, f"{p}_{bid}_{node}.json"), ids)
    jobs = [(b, g, p) for b, g in bs for p in forms]
    random.Random(node + out).shuffle(jobs)
    with cf.ThreadPoolExecutor(H.PAR.get(node, 4)) as ex:
        list(ex.map(do_second, jobs))


def run_new(node, items, pay):
    ids = {x["id"] for x in items}
    first = {}
    for bid, g in RUN.batches(items):
        r = H.call(node, "Answer each question.\n\n" + "\n\n".join(RUN.item_text(it) for it in g) + "\n\n" + RUN.TAIL, os.path.join(OUT, f"s1_{bid}_{node}.json"), ids)
        first.update(r["answers"])
    second_jobs(node, items, pay, first, FORMS, OUT)
    print(node, "完成", flush=True)


def run_fb(items, pay):
    first = FU.first_answers("ds_flash_b", X.OUT)
    for out in FB:
        os.makedirs(out, exist_ok=True)
        for fn in glob.glob(os.path.join(X.OUT, "s1_*_ds_flash_b.json")):
            shutil.copy(fn, os.path.join(out, os.path.basename(fn)))
        second_jobs("ds_flash_b", items, pay, first, FBFORMS, out)
        print("ds_flash_b", os.path.basename(out), "完成", flush=True)


def run_all():
    os.makedirs(OUT, exist_ok=True)
    H.LIMIT = LIMIT
    pay = json.load(open(PAY))
    items = [x for x in json.load(open(ITEMS)) if x["id"] in pay]
    what = sys.argv[2:] or list(NEW)
    print("题", len(items), "跑", what, flush=True)
    H._start[0] = RUN.balance()
    with cf.ThreadPoolExecutor(2) as ex:
        futs = []
        if "fb" in what:
            futs.append(ex.submit(run_fb, items, pay))
        futs.append(ex.submit(lambda: [run_new(a, items, pay) for a in what if a in NEW]))
        for f in futs:
            f.result()
    print("全部完成，DeepSeek 花费", round(H.spent(), 2), "停止标志", H._stop[0], flush=True)


def rows(out, nodes, forms):
    W.OUT, W.FORMS, W.NODES = out, forms, nodes
    R, _ = W.load()
    return R


def kb(R, a, p):
    Xa = [r for r in R if r["a"] == a]
    return W.K(Xa, p=p), W.B(Xa, p=p)


def analyze():
    R = rows(OUT, list(NEW), FORMS)
    json.dump(R, open(os.path.join(HERE, "rows_rp.json"), "w"))
    ids = sorted({r["id"] for r in R if r["p"] == "blank"})
    print(f"新记录 {len(R)}；题 {len(ids)}")
    print("\n预检")
    good = FU.precheck(R, ids, list(NEW), FORMS[:1] + ["one1f", "L3nf"])
    s = set(ids)
    print("\n每节点 K/B（%）")
    for a in NEW:
        print(f"  {a:9s} " + "  ".join(f"{p} {kb(R, a, p)[0]:.0%}/{kb(R, a, p)[1]:.0%}" for p in FORMS))
    print("\n[复现判据，按节点]")
    rep = {}
    for a in NEW:
        r1 = kb(R, a, "cond2f")[0] - kb(R, a, "set2f")[0]
        r2 = kb(R, a, "L3f")[0] - kb(R, a, "set2f")[0]
        vL, vC = X.v0(R, a, s, "L3"), X.v0(R, a, s, "cond2")
        uL, uC = W.vstar(R, a, s, "L3"), W.vstar(R, a, s, "cond2")
        f1 = kb(R, a, "L3nf")[0] - kb(R, a, "one1f")[0]
        f2 = kb(R, a, "L3f")[0] - kb(R, a, "L3nf")[0]
        ok = a in good and r1 > 0 and r2 > 0 and vL < vC
        rep[a] = ok
        print(f"  {a:9s} R1 {r1:+.2f} {'✅' if r1 > 0 else '❌'}  R2 {r2:+.2f} {'✅' if r2 > 0 else '❌'}  "
              f"R3 截断 {vL:.2f} vs {vC:.2f} {'✅' if vL < vC else '❌'}（原定义 {uL:.2f} vs {uC:.2f}）  "
              f"| 只报 F1 {f1:+.2f} F2 {f2:+.2f}  → {'复现' if ok else '不复现' if a in good else '未过预检'}")
    n = sum(rep.values())
    print("\n决定\n  → " + ("新家族复现" if n == 2 else "部分复现" if n == 1 else "不复现"))
    print("\n[探索性] 八节点合并（六旧 + 两新），不改预注册判定")
    Rs = [r for r in json.load(open(os.path.join(HERE, "rows_s.json"))) if r["a"] in OLD6]
    Rm = [r for r in json.load(open(os.path.join(HERE, "rows_sm.json"))) if r["a"] in OLD6 and r["p"] in ("one1t", "one1f")]
    Rf = [r for r in json.load(open(os.path.join(HERE, "rows_fu.json"))) if r["a"] in OLD6 and r["p"] in ("L3nt", "L3nf")]
    R8 = Rs + Rm + Rf + R
    nodes8 = OLD6 + [a for a in NEW if a in good]
    T = {(r["a"], r["id"], r["p"]): r for r in R8}
    need = -(-2 * len(nodes8) // 3)
    X.block(R8, ids, nodes8, "八节点 S1–S5", judge=False)
    W.w1(T, ids, nodes8, "L3f", "one1f", "八节点 E1b", need=need)
    X.dv(R8, ids, nodes8, "L3", "one1", "八节点 E1c 截断")
    W.w1(T, ids, nodes8, "L3nf", "one1f", "八节点 F1", need=need)
    W.w1(T, ids, nodes8, "L3f", "L3nf", "八节点 F2", need=need)


def fbvar():
    runs = [rows(X.OUT, ["ds_flash_b"], FBFORMS)] + [rows(o, ["ds_flash_b"], FBFORMS) for o in FB]
    ids = sorted(set.intersection(*[{r["id"] for r in R if r["p"] == "blank"} for R in runs]))
    s = set(ids)
    print(f"Flash-B 三次运行，共同题 {len(ids)}")
    sd = lambda v: (sum((x - m(v)) ** 2 for x in v) / (len(v) - 1)) ** .5
    print("\n每形式 K / B（%）三次：均值 极差 SD")
    for p in FBFORMS:
        ks = [kb(R, "ds_flash_b", p)[0] for R in runs]
        bs = [kb(R, "ds_flash_b", p)[1] for R in runs]
        print(f"  {p:7s} K {' '.join(f'{k:.0%}' for k in ks)}  均值 {m(ks):.0%} 极差 {max(ks) - min(ks):.2f} SD {sd(ks):.3f}   "
              f"B {' '.join(f'{b:.0%}' for b in bs)}  均值 {m(bs):.0%} 极差 {max(bs) - min(bs):.2f} SD {sd(bs):.3f}")
    print("\n三条差值三次取值")
    for lab, fn in (("K(cond2f)−K(set2f)", lambda R: kb(R, "ds_flash_b", "cond2f")[0] - kb(R, "ds_flash_b", "set2f")[0]),
                    ("K(L3f)−K(set2f)", lambda R: kb(R, "ds_flash_b", "L3f")[0] - kb(R, "ds_flash_b", "set2f")[0]),
                    ("v*(L3)−v*(cond2) 截断", lambda R: X.v0(R, "ds_flash_b", s, "L3") - X.v0(R, "ds_flash_b", s, "cond2")),
                    ("v*(L3)−v*(cond2) 原定义", lambda R: W.vstar(R, "ds_flash_b", s, "L3") - W.vstar(R, "ds_flash_b", s, "cond2"))):
        v = [fn(R) for R in runs]
        print(f"  {lab:24s} " + " ".join(f"{x:+.2f}" for x in v) + ("  ⚠ 变号" if max(v) > 0 > min(v) else ""))


if __name__ == "__main__":
    {"run": run_all, "a": analyze, "fb": fbvar}[sys.argv[1]]()
