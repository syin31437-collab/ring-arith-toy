"""闭环：用团队第①步的数字算集合 C，全员发同一个区间，不用真值、不定位（2026-11-02，库伯："来"）。

问题：定位不了谁错时，无差别投放能不能不伤好节点、纠正坏节点。上一轮（llmclip）的 e 是真值造的；这里 e 由污染比例自然产生，系统自己不知道。
设计：同一任务。每局 10 题、6 节点，k = 0–4 个节点报告里程放大 40%（谁被污染按局随机）。每个 k 12 局，局号 4000+100k+i。
  第①步各自独立估。每题 y = 六人第①步 log10 中位数，m = 六人 MAD，C = [y − c·m, y + c·m]（log10 尺度），全员相同。
  之后对同一个第①步做 7 种投放（局内重复、互不影响）：
    peers = 只给全员数字（基线，一轮）
    plain = 只给区间；rule = 区间 + 明确规则；rule+peers = 区间 + 规则 + 全员数字；后三种各 c = 0.5、1.0
  提示词如实说明区间来自团队第一轮估计（上一轮写的是 cross-check，这里措辞不同，不能和上一轮逐格比）。

定义：H = log10(1.4)/2；x = 第①步对数误差；E = y − θ；r = c·m；P = clip(x, E − r, E + r)。好节点 |x| ≤ H，坏节点 |x| > H。
  K / B = 第②步仍对 / 变对；K*、B* = 用 P 算（表达式）；net = 全员准确率 ② − ①；服从 = |第②步 − P| ≤ 0.01。真值只在评分用。

预期（用已看过的 blindrule 第①步算）：c = 0.5：K* 100 / 99.7 / 98.5 / 93.8 / 68.7%，B* 89 / 68 / 42 / 17 / 4%；c = 1.0：K* 100 / 100 / 100 / 99.4 / 88.3%，B* 45 / 29 / 15 / 7 / 4%。

预注册（运行前提交；1 局试跑只查提示词 bug，判据不据此改）：
  C1 rule：5 × 2 = 10 格里，至少 8 格 |K − K*| ≤ 0.05 且 |B − B*| ≤ 0.10
  C2 rule、c = 0.5：k = 1、2、3 时 K ≥ peers 的 K；k = 2、3 时准确率 ≥ peers − 0.02
  C3 k = 3、rule、c = 0.5：K ≥ 85%，net ≥ 0（peers 上一轮是 K 63%、net ≈ 0）
  C4 k = 4、rule、c = 1.0：K ≥ 75%，net ≥ −0.05；peers net ≤ −0.10（多数错时也把伤害压住）
  C5 k = 3、4、c = 1.0：rule 的 K − plain 的 K ≥ 0.10（不给规则，宽度保护失效）
  C6 rule 每格服从率 ≥ 85%
  C7 rule 节点-题层面：|E| ≤ H + r 时 K ≥ 97%；|E| > H + r 时 K ≤ 25%（后者样本不足 30 就只报告）
  C8 rule、c = 0.5 同时 K ≥ 95% 且 net ≥ +0.10 的最大 k：预测 k = 2
  rule+peers 只报告、不预测。
停止规则：DeepSeek 花费上限 ¥8。判据不过照实报，不改提示词重跑、不补丁。按模型差异只报告。
不测：系统怎么选 c；任务间泛化；真假通知；非数字任务；被污染节点故意撒谎。

用法：~/gut-external/torch215-venv/bin/python llmloop.py pilot | run <并行格数> [R]
"""
import concurrent.futures as cf
import json
import math
import os
import random
import statistics as st
import sys

import team3 as T

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = T.AGENTS
K_LEVELS, CS, ARMS = (0, 1, 2, 3, 4), (0.5, 1.0), ("plain", "rule", "ruleplus")
RULE = "If your estimate for a trip is already inside its range, keep it exactly as it is. If it is outside, change it to the nearest end of the range. Do not make any other changes.\n"


def mad(v):
    m = st.median(v)
    return 1.4826 * st.median(abs(x - m) for x in v)


def setup(k, rep, outdir):
    d = os.path.join(outdir, f"k{k}_{rep}")
    os.makedirs(d, exist_ok=True)
    its = T.items(rep)[:10]
    bad = random.Random(f"bad-{rep}").sample(AGENTS, k)
    sheets, reports = {}, {}
    for a in AGENTS:
        rng = random.Random(f"sheet5-{rep}-{a}")
        f = 1.4 if a in bad else 1.0
        rows = [(it, it["km"] * f * math.exp(0.10 * rng.gauss(0, 1)), it["lpk"] * math.exp(0.10 * rng.gauss(0, 1))) for it in its]
        reports[a] = {it["id"]: km * lp / 100 for it, km, lp in rows}
        sheets[a] = "\n".join(f"{it['id']}: distance about {km:.0f} km, consumption about {lp:.1f} L/100km" for it, km, lp in rows)
    with cf.ThreadPoolExecutor(6) as pool:
        s1 = list(pool.map(lambda a: T.call(a, f"{T.GOAL}\n\nYour field report for this batch:\n{sheets[a]}{T.TAIL}", os.path.join(d, f"s1_{a}.json")), AGENTS))
    meta = {"k": k, "rep": rep, "dir": d, "bad": bad, "items": its, "reports": reports, "sheets": sheets, "s1": {a: r["answers"] for a, r in zip(AGENTS, s1)}, "errors": [r["error"] for r in s1 if r.get("error")]}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False)
    return meta


def ranges(meta, c):
    """全员相同的区间；只用第①步数字。某题有人缺答案就用还在的人。"""
    out = []
    for it in meta["items"]:
        v = [math.log10(meta["s1"][a][it["id"]]) for a in AGENTS if meta["s1"][a].get(it["id"], 0) > 0]
        y, m = st.median(v), mad(v)
        out.append(f"{it['id']}: {10 ** (y - c * m):.2f} to {10 ** (y + c * m):.2f} litres")
    return "\n".join(out)


def cell(meta, arm, c):
    def prompt(a):
        mine = json.dumps({q: round(v, 1) for q, v in meta["s1"][a].items()})
        peers = ""
        if arm in ("peers", "ruleplus"):
            order = list(AGENTS)
            random.Random(f"order-{meta['rep']}-{a}").shuffle(order)
            peers = "What everyone estimated so far:\n" + "\n".join(f"Teammate {l}: {json.dumps({q: round(v, 1) for q, v in meta['s1'][x].items()})}" for l, x in zip("ABCDEF", order)) + "\n"
        rng = "" if arm == "peers" else f"The team's first estimates were pooled into a range of litres for each trip:\n{ranges(meta, c)}\n"
        return f"{T.GOAL}\nYour field report:\n{meta['sheets'][a]}\nYour earlier estimate:\n{mine}\n{peers}{rng}{RULE if arm in ('rule', 'ruleplus') else ''}Give your updated estimate for each trip.{T.TAIL}"
    with cf.ThreadPoolExecutor(6) as pool:
        rs = list(pool.map(lambda a: T.call(a, prompt(a), os.path.join(meta["dir"], f"{arm}_c{c}_{a}.json")), AGENTS))
    return sum(1 for x in rs if x.get("error"))


def main():
    if sys.argv[1] == "pilot":
        meta = setup(3, 9200, os.path.join(HERE, "runs_loop_pilot"))
        print("第①步失败", len(meta["errors"]))
        for arm, c in (("peers", 0.0), ("rule", 0.5), ("ruleplus", 1.0), ("plain", 1.0)):
            print(arm, c, "失败", cell(meta, arm, c), flush=True)
        return
    par, R = int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 12
    out = os.path.join(HERE, "runs_loop")
    jobs = [(k, 4000 + 100 * k + i) for k in K_LEVELS for i in range(R)]
    with cf.ThreadPoolExecutor(par) as ex:
        metas = list(ex.map(lambda j: setup(j[0], j[1], out), jobs))
    print("第①步完成", len(metas), "局，失败", sum(len(m["errors"]) for m in metas), flush=True)
    cells = [(m, "peers", 0.0) for m in metas] + [(m, arm, c) for m in metas for arm in ARMS for c in CS]
    random.Random("loop-order").shuffle(cells)
    print(len(cells), "格", flush=True)
    with cf.ThreadPoolExecutor(par) as ex:
        fails = list(ex.map(lambda j: cell(*j), cells))
    print("全部完成，失败调用", sum(fails), flush=True)


if __name__ == "__main__":
    main()
