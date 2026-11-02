"""同一规则全员发，好节点保得住吗（2026-11-02，库伯："先跑"）。

背景：57 局已有记录的事后诊断（有缺陷，不进正式结果）显示，全员看同一张表时，少数人错 → 错节点回来、好节点保住；一半人错 → 好节点也被拉错。
核对后发现旧数据的缺陷：表格顺序和节点名字固定、被污染的节点固定、"数字"和"留言"绑在一起、没有 2/6、跨批留言污染第①步。这里全部修掉。

设计：真值未知的车队用油估计（同 team3 的任务和报告噪声）。每局 10 题、1 批、无协调者、无跨批记忆。
  因素 k = 被污染节点数 0 / 1 / 2 / 3 / 4（共 6 个节点；报告里程放大 40%；被污染的是谁按局随机）。
  因素 arm = num（表里只有数字）/ note（数字 + 每人一句留言）。同一局的两个 arm 共用第①步、题、报告和表格顺序（配对），只差表里有没有留言。
  流程：① 各自独立估 ② 看全员表后修改 ③ 再看一轮后定稿。所有节点提示词模板相同，发之前不看谁对谁错；真值和秤不进提示词。
  表格：每个节点每一轮单独打乱行序，名字换成 Teammate A–F（字母也打乱）；不标哪行是自己（同旧实验）。
  每个 k 20 局，配对单位 = 局。局号 2000+100k+i，运行顺序随机。

定义（真值只用于评分）：H = log10(1.4) / 2；对 = |log10(估计 / 真值)| ≤ H；好节点 = 第①步对；错节点 = 第①步错。
  K = P(③对 | ①对)（好节点保住）；B = P(③对 | ①错)（错节点回来）；net = 全员准确率 ③ − ①。
  靠拢比例 = (s3 − s1) / (pm − s1)，pm = 其他 5 人第①步中位数，|pm − s1| > 0.02 才算。区间 = 按局 bootstrap（k 内重抽局）。

映射预检（k = 0，note + num 的第①步；不过则结果标"映射不成立"）：共识中位偏移 ≤ H/4；第①步 − 报告中位 ≤ 0.005；节点误差两两相关均值 ≤ 0.3。

预注册（运行前提交；判据来自旧诊断，看过旧数据、没看过新数据）：
  P1 k ≤ 1、note：K ≥ 95%，B ≥ 85%，net ≥ +0.15
  P2 k ≤ 1、num：同 P1（不需要留言也成立）
  P3 k = 3、note：K ≤ 70%，net ≤ +0.05
  P4 k = 4、note：K ≤ 50%，net ≤ 0
  P5 k = 2、note：net 的 95% 区间下限 > 0（临界点在 2 和 3 之间），K 在 70%–95%
  P6 留言效应：k ≤ 1 时，配对差 note − num 的 K、B 绝对值 ≤ 5 个百分点；k ≥ 2 只报告、不预测方向
  P7 离群机制（k ≤ 1，note + num，全部节点）：靠拢比例在 |gap| ≥ 0.10 的格比 0.02–0.05 的格高 ≥ 0.10；同一 gap 格内好、错节点靠拢比例之差绝对值 ≤ 0.05（否则"错节点靠得更多"不能只用离群解释）
  P8 固定规则：w = k = 0、note 全部节点靠拢比例中位（不按真值挑），每人向其他 5 人中位数靠 w；k = 1 时 LLM 与规则的 K、B 差绝对值 ≤ 5 个百分点；k ≥ 2 只报告

停止规则：DeepSeek 花费上限 ¥20，超预计就停、报已有结果。失败不重跑、不改判据、不补丁；判据不过就照实报。调用失败的节点-题剔除并报告比例；> 5% 标出。
不做：通知为真时的代价（另一个实验）；这里不判断结论是否普适。

用法：~/gut-external/torch215-venv/bin/python blindrule.py run <并行数> [R] | pilot
"""
import concurrent.futures as cf
import json
import math
import os
import random
import sys
import traceback

import team3 as T

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = T.AGENTS
DELTA, K_LEVELS, PER = 0.40, (0, 1, 2, 3, 4), 10
LABELS = "ABCDEF"


def run_pair(k, rep, outdir):
    """跑一局的两个 arm，第①步共用。"""
    d = os.path.join(outdir, f"k{k}_{rep}")
    fn = os.path.join(d, "run.json")
    if os.path.exists(fn):
        return fn
    os.makedirs(d, exist_ok=True)
    its = T.items(rep)[:PER]
    bad = random.Random(f"bad-{rep}").sample(AGENTS, k)
    pool = cf.ThreadPoolExecutor(6)
    reports = {}

    def sheet(a):
        rng = random.Random(f"sheet3-{rep}-{a}")
        j = lambda x: x * math.exp(0.10 * rng.gauss(0, 1))
        f = 1 + DELTA if a in bad else 1.0
        rows = [(it, j(it["km"] * f), j(it["lpk"])) for it in its]
        reports[a] = {it["id"]: km * lp / 100 for it, km, lp in rows}
        return "\n".join(f"{it['id']}: distance about {km:.0f} km, consumption about {lp:.1f} L/100km" for it, km, lp in rows)
    sheets = {a: sheet(a) for a in AGENTS}

    def table(rs, a, rnd, notes):
        rng = random.Random(f"order-{rep}-{a}-{rnd}")
        idx = list(range(6))
        rng.shuffle(idx)
        lines = []
        for lab, i in zip(LABELS, idx):
            row = f"Teammate {lab}: {json.dumps({q: round(v, 1) for q, v in rs[i]['answers'].items()})}"
            lines.append(row + (f" | note: {rs[i]['note']}" if notes else ""))
        return "\n".join(lines)

    s1 = list(pool.map(lambda a: T.call(a, f"{T.GOAL}\n\nYour field report for this batch:\n{sheets[a]}{T.TAIL}", os.path.join(d, f"s1_{a}.json")), AGENTS))
    out = {"k": k, "rep": rep, "bad": bad, "items": its, "reports": reports, "s1": {a: r["answers"] for a, r in zip(AGENTS, s1)}, "errors": [r["error"] for r in s1 if r.get("error")]}
    for arm, notes in (("num", False), ("note", True)):
        s2 = list(pool.map(lambda a: T.call(a, f"{T.GOAL}\nYour field report:\n{sheets[a]}\nWhat everyone estimated so far:\n{table(s1, a, 2, notes)}\nGive your updated estimate for each trip.{T.TAIL}", os.path.join(d, f"{arm}_s2_{a}.json")), AGENTS))
        s3 = list(pool.map(lambda a: T.call(a, f"{T.GOAL}\nYour field report:\n{sheets[a]}\nWhat everyone estimated after one round of discussion:\n{table(s2, a, 3, notes)}\nGive your final estimate for each trip.{T.TAIL}", os.path.join(d, f"{arm}_s3_{a}.json")), AGENTS))
        out[arm] = {"s2": {a: r["answers"] for a, r in zip(AGENTS, s2)}, "s3": {a: r["answers"] for a, r in zip(AGENTS, s3)}}
        out["errors"] += [r["error"] for r in s2 + s3 if r.get("error")]
    pool.shutdown()
    json.dump(out, open(fn, "w"), ensure_ascii=False)
    return fn


def go(j):
    k, rep, outdir = j
    try:
        return f"完成 k={k} rep={rep} {run_pair(k, rep, outdir)}"
    except Exception:
        return f"失败 k={k} rep={rep}: {traceback.format_exc()[-300:]}"


if __name__ == "__main__":
    if sys.argv[1] == "pilot":
        out = os.path.join(HERE, "runs_blind_pilot")
        for k, rep in ((0, 9000), (3, 9003)):
            print(go((k, rep, out)), flush=True)
    else:
        par, R = int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 20
        out = os.path.join(HERE, "runs_blind")
        jobs = [(k, 2000 + 100 * k + i, out) for k in K_LEVELS for i in range(R)]
        random.Random("blind-order").shuffle(jobs)
        print(len(jobs), "局", flush=True)
        with cf.ThreadPoolExecutor(par) as ex:
            for r in ex.map(go, jobs):
                print(r, flush=True)
