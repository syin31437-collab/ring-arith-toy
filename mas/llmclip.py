"""把集合 C 直接发给 LLM：它照不照表达式走（2026-11-02，库伯："直接 llm"）。

表达式（blind_margin.py，看过上一轮结果后推的数学恒等式）：e = 投放中心误差，r = C 半宽，h = 门槛。节点投影到 C 时
  好节点保住 ⇔ |e| ≤ h + r；坏节点回来 ⇔ 在 C 外且落点误差 ≤ h。这是"算出来"的，LLM 收到一段文字后会不会这样做，本实验才测。

设计：同一任务（车队用油，blindrule 的报告噪声）。每局 10 题、6 节点，其中 2 个节点的报告里程放大 40%（谁被污染按局随机）。
  第①步各自独立估；之后对同一个第①步做 27 种投放（局内重复），每种全员发同样的区间，互相不影响：
  e/h ∈ {0, 0.5, 1.5, 3} × r/h ∈ {0.25, 1}，加安慰剂 e = 0、r/h = 8（区间宽到几乎不含信息）；每题 e 的方向按题随机、各格固定。
  arm：plain = 只给区间；rule = 给区间 + 明确规则（在区间内不动，在外面移到最近的端点）；peers = plain + 第①步全员数字（行序打乱、匿名、无留言）。
  区间 = 真值 × 10^(±e ± r)：**用真值造区间只是为了控制 e，不是可部署的方法**；所有节点拿到的区间相同，不看谁对谁错。
  10 局，局号 3000+i。单位 = 局。

定义：H = log10(1.4) / 2；x = 第①步对数误差；P = clip(x, E − r, E + r)，E = ±e。好节点 |x| ≤ H；坏节点 |x| > H。
  K = 好节点第②步仍对；B = 坏节点第②步变对；K*、B* = 把 x 换成 P 算的值（表达式的预测）。服从 = |第②步 − P| ≤ 0.01；没动 = |第②步 − x| ≤ 0.005。

映射预检：未污染节点第①步共识中位偏移 ≤ H/4；第①步 − 报告中位 ≤ 0.005；未污染节点误差两两相关 ≤ 0.3。

预注册（运行前提交；只有 2 局试跑看提示词有没有 bug，判据不据此改）：
  L1 rule：8 个非安慰剂格里，至少 7 格 |K − K*| ≤ 0.05 且 |B − B*| ≤ 0.10（明确告诉它怎么做时，表达式成立）
  L2 rule：每格服从率 ≥ 85%
  L3 plain：8 格 |K − K*| 平均 ≤ 0.15；K 随 |e| 增大而降、r 增大不降（各容差 0.02）；区间外节点向中心靠拢比例中位比区间内节点高 ≥ 0.2
  L4 安慰剂：rule 没动 ≥ 90%、K ≥ 97%；plain K ≥ 95%
  L5 错的集合（e/h = 3、r/h = 0.25，K* = 0）：plain K ≤ 30%；peers K ≤ 50%（同伴救不了多少，依据是假通知的旧数据）
  L6 对的集合（e = 0、r/h = 0.25）：peers K ≥ 95%、B ≥ 80%，B 比 peers 安慰剂格高 ≥ 0.15（对的集合能压过 2/6 污染的同伴）
停止规则：DeepSeek 花费上限 ¥10。判据不过照实报，不改提示词重跑、不打补丁。按模型的差异只报告。
不测：系统怎么知道 e（这里 e 是给定的）；证据、约束、锚来源；跨领域；通知真假的语义内容。

用法：~/gut-external/torch215-venv/bin/python llmclip.py pilot | run <并行格数> [R]
"""
import concurrent.futures as cf
import json
import math
import os
import random
import sys

import team3 as T

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = T.AGENTS
H = math.log10(1.4) / 2
ES, RS, PLACEBO = (0.0, 0.5, 1.5, 3.0), (0.25, 1.0), (0.0, 8.0)
ARMS = ("plain", "rule", "peers")
CELLS = [(e, r) for e in ES for r in RS] + [PLACEBO]
RULE = "If your estimate for a trip is already inside its range, keep it exactly as it is. If it is outside, change it to the nearest end of the range. Do not make any other changes.\n"


def setup(rep, outdir):
    d = os.path.join(outdir, str(rep))
    os.makedirs(d, exist_ok=True)
    its = T.items(rep)[:10]
    bad = random.Random(f"bad-{rep}").sample(AGENTS, 2)
    sheets, reports = {}, {}
    for a in AGENTS:
        rng = random.Random(f"sheet4-{rep}-{a}")
        f = 1.4 if a in bad else 1.0
        rows = [(it, it["km"] * f * math.exp(0.10 * rng.gauss(0, 1)), it["lpk"] * math.exp(0.10 * rng.gauss(0, 1))) for it in its]
        reports[a] = {it["id"]: km * lp / 100 for it, km, lp in rows}
        sheets[a] = "\n".join(f"{it['id']}: distance about {km:.0f} km, consumption about {lp:.1f} L/100km" for it, km, lp in rows)
    with cf.ThreadPoolExecutor(6) as pool:
        s1 = list(pool.map(lambda a: T.call(a, f"{T.GOAL}\n\nYour field report for this batch:\n{sheets[a]}{T.TAIL}", os.path.join(d, f"s1_{a}.json")), AGENTS))
    meta = {"rep": rep, "bad": bad, "items": its, "reports": reports, "sheets": sheets, "s1": {a: r["answers"] for a, r in zip(AGENTS, s1)}, "errors": [r["error"] for r in s1 if r.get("error")]}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False)
    return meta


def ranges(meta, e, r):
    out = []
    for it in meta["items"]:
        s = random.Random(f"sign-{meta['rep']}-{it['id']}").choice((-1, 1))
        c = math.log10(it["truth"]) + s * e * H
        out.append(f"{it['id']}: {10 ** (c - r * H):.2f} to {10 ** (c + r * H):.2f} litres")
    return "\n".join(out)


def cell(meta, outdir, arm, e, r):
    d = os.path.join(outdir, str(meta["rep"]))
    rng_text = ranges(meta, e, r)

    def prompt(a):
        mine = json.dumps({k: round(v, 1) for k, v in meta["s1"][a].items()})
        peers = ""
        if arm == "peers":
            order = list(AGENTS)
            random.Random(f"order-{meta['rep']}-{a}").shuffle(order)
            peers = "What everyone estimated so far:\n" + "\n".join(f"Teammate {l}: {json.dumps({k: round(v, 1) for k, v in meta['s1'][x].items()})}" for l, x in zip("ABCDEF", order)) + "\n"
        return f"{T.GOAL}\nYour field report:\n{meta['sheets'][a]}\nYour earlier estimate:\n{mine}\n{peers}A cross-check produced a plausible range of litres for each trip:\n{rng_text}\n{RULE if arm == 'rule' else ''}Give your updated estimate for each trip.{T.TAIL}"
    with cf.ThreadPoolExecutor(6) as pool:
        rs = list(pool.map(lambda a: T.call(a, prompt(a), os.path.join(d, f"{arm}_e{e}_r{r}_{a}.json")), AGENTS))
    return sum(1 for x in rs if x.get("error"))


def main():
    if sys.argv[1] == "pilot":
        out = os.path.join(HERE, "runs_clip_pilot")
        meta = setup(9100, out)
        print("第①步失败", len(meta["errors"]))
        for arm, e, r in (("rule", 3.0, 0.25), ("plain", 0.0, 0.25), ("peers", 3.0, 0.25)):
            print(arm, e, r, "失败", cell(meta, out, arm, e, r), flush=True)
        return
    par, R = int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 10
    out = os.path.join(HERE, "runs_clip")
    reps = [3000 + i for i in range(R)]
    with cf.ThreadPoolExecutor(par) as ex:
        metas = list(ex.map(lambda q: setup(q, out), reps))
    print("第①步完成", len(metas), "局，失败", sum(len(m["errors"]) for m in metas), flush=True)
    jobs = [(m, arm, e, r) for m in metas for arm in ARMS for e, r in CELLS]
    random.Random("clip-order").shuffle(jobs)
    print(len(jobs), "格", flush=True)
    with cf.ThreadPoolExecutor(par) as ex:
        fails = list(ex.map(lambda j: cell(j[0], out, j[1], j[2], j[3]), jobs))
    print("全部完成，失败调用", sum(fails), flush=True)


if __name__ == "__main__":
    main()
