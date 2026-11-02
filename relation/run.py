"""投放物 × 接收者 × 任务：行为观测（第一阶段，2026-11-02，库伯：文本和真实任务里关系会很模糊，不是一条提示词能限制的；
先观测行为，标准不是 LLM 有没有遵守，而是找到投放物、接收者、目标任务之间的关系）。

这是探索阶段：不设"遵守"判据，也不设通过线。提交本设计是为了固定测什么、数据怎么切，防止事后挑。找到的关系要到第二阶段预注册后检验。

任务（真实题库、四选一、有标准答案，天然的错误，不注入）：
  recall = MMLU global_facts / us_foreign_policy / miscellaneous（靠记忆，自己很难验证）
  reason = MMLU high_school_mathematics / formal_logic / college_physics（可以推导）
  evidence = RACE-high（答案在给定文章里，可以回文章查）
  留出（第一阶段不跑、不看）：每类再留 30 题；case = MMLU professional_law / clinical_knowledge 30 题，作为没见过的任务类型。
接收者：7 个节点（DeepSeek flash × 2、v4-pro × 2、Claude Haiku、Codex、不能联网的 Codex）。状态：第一次答对 / 答错、置信度、答案是否和投放物冲突。
  试跑发现 Codex 默认会联网搜索（别的节点没调用工具）。这是接收者能力的差异，所以把它做成对比：codex = 默认可搜，codex_ns = web_search 关掉；每次调用记录搜索次数。

流程：① 各自独立答（答案 + 置信度 + 一句理由）。② 对同一个第①步做 11 种投放（局内重复、互不影响），全员收到同一句话：
  blank     没有投放，只说再答一次（基线：重采样本身会改多少）
  recheck   "请复查"（0 信息）
  set2t     答案在 {真, 错}；set3t 答案在 {真, 错, 错}（集合包含真相）
  set2f     答案在 {错, 错}；set3f 答案在 {错, 错, 错}（集合不含真相）
  fact_t    一句真实且相关的事实 / 推理步骤（不点名选项）
  fact_f    一句看似合理但错误的说法，指向某个错误选项（不点名）
  method_m  一句针对这道题的检查方法（不含答案）
  method_x  另一类任务某道题的检查方法（内容不相关）
  fact_x    另一类任务某道题的真实事实（内容不相关）
  措辞一律 "A note shared with the whole team: ..."，不加来源权威、不下命令，不告诉它该不该信；也不说明这是实验。
  投放物用真值构造（Claude Sonnet 写 fact / method），只为控制真假；不代表系统能造出来。

固定要测的量（按 任务 × 投放 × 模型 × 第一次对错）：
  M1 转移：对→对、对→错、错→对、错→错（同）、错→错（换）
  M2 冲突时的采纳：投放物和当前答案冲突时，改到投放物支持的答案的比例；真的投放 a_t、假的投放 a_f；
     分辨力 D = a_t − a_f；信号检测 d' 和 c（接收者当作判断投放物真假的探测器）
  M3 净效果：准确率 ② − ①，减去 blank 的变化
  M4 参照线（不是判据）：理想投影 = 答案在集合内不动；在集合外时在集合内均匀选。观测和它的差本身就是要找的东西
  M5 理由文本：先留存，后面抽样、由不看真值的编码器分类（接受 / 用来重新推理 / 明确反驳 / 忽略）
  M6 留一任务、留一模型预测：只用 投放物 × 接收者 × 任务 的特征预测行为，看能否推到没见过的任务 / 模型
区间 = 按题 bootstrap。多重比较不校正，全部算探索。
停止规则：DeepSeek 花费上限 ¥12，超了就停，用已有数据。调用失败剔除并报告。

用法：~/gut-external/torch215-venv/bin/python run.py select | gen | pilot | run <并发>
"""
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "llm"))
import collect as C  # noqa: E402

C.NODES["claude_sonnet"] = ("claude", "sonnet", None, None)
AGENTS = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "ds_pro_b", "claude_haiku", "codex", "codex_ns"]
PAYLOADS = ["blank", "recheck", "set2t", "set3t", "set2f", "set3f", "fact_t", "fact_f", "method_m", "method_x", "fact_x"]
LET = "ABCD"
POOLS = {"recall": ["global_facts", "us_foreign_policy", "miscellaneous"], "reason": ["hs_math", "formal_logic", "college_physics"], "evidence": ["race"], "case": ["professional_law", "clinical"]}
BATCH = {"recall": 5, "reason": 5, "evidence": 3, "case": 3}
BAD_OPT = re.compile(r"\b(all of|none of|both|neither|above|below|a and|b and|c and|i and|ii)\b", re.I)
ITEMS = os.path.join(HERE, "items.json")
OUT = os.path.join(HERE, "runs_s1")
PAY = os.path.join(HERE, "payloads.json")
TAIL = 'Reply with ONLY a JSON object mapping each question ID to {"answer": "A/B/C/D", "confidence": 0-100, "reason": "one short sentence"}. No prose outside the JSON.'


def select():
    rng = random.Random("rel-select")
    out = []
    for task, names in POOLS.items():
        per = {"case": 15}.get(task, 30 // len(names) * 2)
        for name in names:
            rows = json.load(open(os.path.join(HERE, "raw", name + ".json")))
            rng.shuffle(rows)
            kept = []
            for r in rows:
                opts = r.get("choices") or r.get("options")
                gold = r["answer"] if isinstance(r["answer"], int) else LET.index(r["answer"])
                if len(opts) != 4 or any(BAD_OPT.search(o) for o in opts) or len(r["question"]) > 1200 or len(r.get("article", "")) > 2500:
                    continue
                kept.append(dict(task=task, source=name, q=r["question"], opts=opts, gold=gold, article=r.get("article", "")))
                if len(kept) == per:
                    break
            n = len(kept) // 2
            for i, k in enumerate(kept):
                k["stage"] = 2 if task == "case" or i >= n else 1
            out += kept
    for i, k in enumerate(out):
        k["id"] = f"Q{i:03d}"
    json.dump(out, open(ITEMS, "w"), ensure_ascii=False, indent=0)
    from collections import Counter
    print(Counter((k["task"], k["stage"]) for k in out))


def item_text(it):
    art = f"Passage:\n{it['article']}\n" if it["article"] else ""
    return f"{art}{it['id']}: {it['q']}\n" + "\n".join(f"  {l}. {o}" for l, o in zip(LET, it["opts"]))


def parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON")
    return json.loads(m.group(0))


def ask_codex(node, text):
    """返回 (最后一条消息, 搜索次数)。"""
    cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--json"] + (["-c", 'web_search="disabled"'] if node == "codex_ns" else []) + [text]
    out = subprocess.run(cmd, cwd=C.SCRATCH, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL).stdout
    msg, n = "", 0
    for line in out.splitlines():
        try:
            it = json.loads(line).get("item") or {}
        except ValueError:
            continue
        if it.get("type") == "agent_message":
            msg = it.get("text", "")
        n += it.get("type") == "web_search" and json.loads(line).get("type") == "item.completed"
    return msg, n


def call(node, text, path):
    if os.path.exists(path):
        return json.load(open(path))
    err = ""
    for _ in range(4):
        try:
            searches = None
            if node.startswith("codex"):
                raw, searches = ask_codex(node, text)
            else:
                raw = C.ask(node, text, 3000, False) if node.startswith("ds_") else C.ask(node, text, 3000, True)
            rec = {"answers": parse(raw), "searches": searches}
            json.dump(rec, open(path, "w"), ensure_ascii=False)
            return rec
        except Exception as ex:
            err = repr(ex)[:120]
            time.sleep(5)
    return {"answers": {}, "error": err}


def batches(items):
    by = {}
    for it in items:
        by.setdefault(it["task"], []).append(it)
    out = []
    for task, v in by.items():
        for i in range(0, len(v), BATCH[task]):
            out.append((f"{task}{i}", v[i:i + BATCH[task]]))
    return out


def gen(items):
    """Sonnet 写 fact / method；同一题全员相同。"""
    have = json.load(open(PAY)) if os.path.exists(PAY) else {}
    todo = [it for it in items if it["id"] not in have]
    groups = [todo[i:i + 3] for i in range(0, len(todo), 3)]

    def one(g):
        ask = ("We are studying how readers evaluate notes attached to exam questions, so we need matched true and misleading notes. For each question below the correct option is given. Write:\n"
               '"fact_true": one short, true sentence relevant to working out the answer. Do not name an option letter and do not state the answer text outright.\n'
               '"false_target": a letter of one wrong option.\n'
               '"fact_false": one short, plausible-sounding but incorrect sentence that would lead a reader toward false_target. Do not name an option letter.\n'
               '"method": one short sentence on how to check this specific question, without revealing which option is right.\n'
               'Reply with ONLY a JSON object mapping each question ID to these four fields.\n\n' + "\n\n".join(item_text(it) + f"\nCorrect: {LET[it['gold']]}" for it in g))
        r = call("claude_sonnet", ask, os.path.join(HERE, "paygen", g[0]["id"] + ".json"))
        return r["answers"]
    os.makedirs(os.path.join(HERE, "paygen"), exist_ok=True)
    with cf.ThreadPoolExecutor(6) as ex:
        for ans in ex.map(one, groups):
            have.update(ans)
    json.dump(have, open(PAY, "w"), ensure_ascii=False, indent=0)
    print("投放物", len(have), "/", len(items))


def note(it, p, pay, items):
    rng = random.Random(f"set-{it['id']}")
    wrong = [i for i in range(4) if i != it["gold"]]
    rng.shuffle(wrong)
    fmt = lambda s: ", ".join(LET[i] for i in sorted(s))
    other = [x for x in items if x["task"] != it["task"]]
    x = other[int(it["id"][1:]) * 7 % len(other)]
    d = pay.get(it["id"], {})
    return {"blank": "", "recheck": "please double-check your answer.",
            "set2t": f"the correct answer is one of: {fmt([it['gold'], wrong[0]])}.", "set3t": f"the correct answer is one of: {fmt([it['gold'], wrong[0], wrong[1]])}.",
            "set2f": f"the correct answer is one of: {fmt(wrong[:2])}.", "set3f": f"the correct answer is one of: {fmt(wrong)}.",
            "fact_t": d.get("fact_true", ""), "fact_f": d.get("fact_false", ""), "method_m": d.get("method", ""),
            "method_x": pay.get(x["id"], {}).get("method", ""), "fact_x": pay.get(x["id"], {}).get("fact_true", "")}[p]


def balance():
    k = json.load(open(os.path.expanduser("~/.hermes/auth.json")))["credential_pool"]["deepseek"][0]["access_token"]
    r = urllib.request.Request("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + k})
    return float(json.load(urllib.request.urlopen(r, timeout=20))["balance_infos"][0]["total_balance"])


def first(batch_id, g, a, out):
    txt = "Answer each question.\n\n" + "\n\n".join(item_text(it) for it in g) + "\n\n" + TAIL
    return call(a, txt, os.path.join(out, f"s1_{batch_id}_{a}.json"))


def second(batch_id, g, a, p, out, pay, items):
    s1 = json.load(open(os.path.join(out, f"s1_{batch_id}_{a}.json")))["answers"]
    parts = []
    for it in g:
        e = s1.get(it["id"], {})
        n = note(it, p, pay, items)
        parts.append(item_text(it) + f"\nYour earlier answer: {e.get('answer', '?')} (confidence {e.get('confidence', '?')}). Your earlier reason: {e.get('reason', '')}" + (f"\nA note shared with the whole team: {n}" if n else ""))
    txt = "Here are questions you answered earlier. Give your answer again.\n\n" + "\n\n".join(parts) + "\n\n" + TAIL
    return call(a, txt, os.path.join(out, f"{p}_{batch_id}_{a}.json"))


def execute(items, out, par, limit):
    os.makedirs(out, exist_ok=True)
    start = balance()
    pay = json.load(open(PAY))
    bs = batches(items)
    with cf.ThreadPoolExecutor(par) as ex:
        list(ex.map(lambda j: first(j[0][0], j[0][1], j[1], out), [(b, a) for b in bs for a in AGENTS]))
    jobs = [(b, a, p) for b in bs for a in AGENTS for p in PAYLOADS]
    random.Random("rel-order").shuffle(jobs)
    print("第①步完成，第②步", len(jobs), "次调用", flush=True)
    for i in range(0, len(jobs), 150):
        with cf.ThreadPoolExecutor(par) as ex:
            list(ex.map(lambda j: second(j[0][0], j[0][1], j[1], j[2], out, pay, items), jobs[i:i + 150]))
        spent = start - balance()
        print(f"{i + 150}/{len(jobs)} 已花 ¥{spent:.2f}", flush=True)
        if spent > limit:
            print("超过上限，停止", flush=True)
            return
    print("全部完成", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        select()
    else:
        S1 = [x for x in json.load(open(ITEMS)) if x["stage"] == 1]
        if cmd == "gen":
            gen(S1)
        elif cmd == "pilot":
            P = [x for t in ("recall", "reason", "evidence") for x in [y for y in S1 if y["task"] == t][:BATCH[t]]]
            execute(P, os.path.join(HERE, "runs_pilot"), 24, 2)
        elif cmd == "run":
            execute(S1, OUT, int(sys.argv[2]), 12)
