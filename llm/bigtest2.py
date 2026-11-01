"""大规模真实系统测试第二轮：又 4 个新领域、8 个节点（加 Claude Sonnet）、秤换成不同家族的模型（2026-11-01）。

第一轮教训（见 switching/README.md）：同家族模型当秤 → 和 DeepSeek 节点共享记忆 → 局部 ζ ≈ 1；"诚实"≠"没犯错"；
ζ = 1 时预测错的方向取决于裁判跟不跟被污染的约束。流程写死第一轮之后的全部修法（σ_c 并列值兜底、线扣节点常态偏移、
三角帽节点噪声、单轮截到 ±3σ、秤坏保护）。
领域（题目按编号当轮次；全部用 log10；δ ∈ {0, 5%, 10%, 20%}；δ_k = δ · exp(0.5 z) 每题不同；每个节点的资料另加 1% 抖动）：
  forest    森林覆盖率 %（真实，世界银行 2021）：森林面积 ÷ 国土面积；约束 = Claude Sonnet 凭记忆（不给资料）；
            投毒 5 个 DeepSeek（5 / 8，过半，整个家族）：森林面积 × (1 + δ_k)
  urban     城镇化率 %（真实，世界银行 2022）：城镇人口 ÷ 总人口；约束 = Claude Sonnet 凭记忆；投毒 ds_flash_b、ds_pro_b（2 / 8）：城镇人口 × (1 + δ_k)
  pendulum  单摆周期 s（合成）：T = 2π √(L / g)；约束 = Claude Sonnet 看独立秒表记录（20 次全摆动的总时间，带 2% 误差）；
            投毒 ds_flash_a、ds_flash_b、ds_pro_b + codex（4 / 8，含一个非 DeepSeek）+ 卧底 ds_pro_a（第 31 题起）：摆长 × (1 + δ_k)²
  interest  复利终值 $（合成）：A = P (1 + r)^n；约束 = Claude Sonnet 看独立银行对账单（期末余额，带 3% 误差）；
            投毒 ds_flash_a、ds_flash_c、claude_haiku（3 / 8，含一个 Claude）+ 卧底 ds_pro_a（第 31 题起）：本金 × (1 + δ_k)
  R0 = 上面的独立约束；R1 = 约束直接用投毒资料算（ζ = 1）。

预注册（11-01，采集之前写下；跨 4 个领域汇总）：
  1. R0 全部 (领域, δ, 裁判) 格子：实际漏检落在区间内的比例 ≥ 90%
  2. R0 中 ρ̂ ≥ 1 的格子：点预测 |误差| 中位 ≤ 0.08
  3. R1 中 δ ≥ 10% 的格子：≥ 80% 满足 |实际 − 点预测| ≥ 0.30（ζ = 1 时预测错，方向不定——第一轮修正后的说法）
  4. 卧底（pendulum、interest；δ = 10%、20%）：下毒后到越线的轮数和公式相差 ≤ max(2 轮, 30%)，4 个里 ≥ 3 个
  5. δ = 0 那一轮按编号跑 EWMA：**真误报**（越线前 10 轮里该节点相对真值的平均 |误差| < 2 倍它自己的噪声）总数 ≤ 2；
     其余越线记为"抓到的自然错误"，单独报告
  6. 学习裁判（带秤坏保护，50 种顺序）：R0 静态软裁判被骗 ≥ 30% 的格子里，≥ 75% 满足最后一段 ≤ 静态

用法：~/gut-external/torch215-venv/bin/python bigtest2.py collect | predict | score
"""
import concurrent.futures as cf
import json
import math
import os
import random
import re
import statistics as st
import sys
import time

import collect
from collect import ask, parse

collect.NODES["claude_sonnet"] = ("claude", "sonnet", None, None)        # 只在本脚本运行时加，不改 collect.py（别的实验的节点集不变）
NODES = collect.NODES

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_big2")
PRED = os.path.join(HERE, "bigtest2_pred.json")
DELTAS = [0.0, 0.05, 0.1, 0.2]
CFG = {
    "forest": {"always": {"ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "ds_pro_b"}, "sleeper": None, "unit": "percent"},
    "urban": {"always": {"ds_flash_b", "ds_pro_b"}, "sleeper": None, "unit": "percent"},
    "pendulum": {"always": {"ds_flash_a", "ds_flash_b", "ds_pro_b", "codex"}, "sleeper": "ds_pro_a", "unit": "seconds"},
    "interest": {"always": {"ds_flash_a", "ds_flash_c", "claude_haiku"}, "sleeper": "ds_pro_a", "unit": "US dollars"},
}
CHECKER = "claude_sonnet"
BASE = True                                                # 第二轮写死：线扣节点常态偏移、三角帽噪声、单轮截到 ±3σ


def wb(ind):
    return {r["countryiso3code"]: r["value"] for r in json.load(open(os.path.join(HERE, f"wb_{ind}.json")))[1] if r["value"]}


def items(dom):
    rng = random.Random(f"big2-{dom}")
    out = []
    if dom in ("forest", "urban"):
        meta = {c["id"]: c for c in json.load(open(os.path.join(HERE, "wb_countries.json")))[1]}
        if dom == "forest":
            num, den = wb("AG.LND.FRST.K2"), wb("AG.LND.TOTL.K2")
            ok = sorted(i for i, c in meta.items() if c["region"]["value"] != "Aggregates" and i in num and i in den and den[i] > 1e4 and num[i] > 0)
        else:
            num, den = wb("SP.URB.TOTL"), wb("SP.POP.TOTL")
            ok = sorted(i for i, c in meta.items() if c["region"]["value"] != "Aggregates" and i in num and i in den and den[i] > 1e6 and num[i] > 0)
        for i in sorted(rng.sample(ok, 60)):
            out.append({"id": i, "name": meta[i]["name"], "a": num[i], "b": den[i], "truth": 100 * num[i] / den[i], "z": rng.gauss(0, 1)})
    elif dom == "pendulum":
        for k in range(60):
            L, g = 10 ** rng.uniform(-1, math.log10(20)), 10 ** rng.uniform(math.log10(1.6), math.log10(24.8))
            T = 2 * math.pi * math.sqrt(L / g)
            out.append({"id": f"P{k:02d}", "name": f"pendulum {k}", "a": L, "b": g, "truth": T, "z": rng.gauss(0, 1), "rec": 20 * T * (1 + 0.02 * rng.gauss(0, 1))})
    else:
        for k in range(60):
            P, r, n = 10 ** rng.uniform(2, 6), 10 ** rng.uniform(math.log10(0.005), math.log10(0.12)), rng.randint(1, 40)
            A = P * (1 + r) ** n
            out.append({"id": f"A{k:02d}", "name": f"account {k}", "a": P, "b": r, "n": n, "truth": A, "z": rng.gauss(0, 1), "rec": A * (1 + 0.03 * rng.gauss(0, 1))})
    return out


def dk(it, delta):
    return delta * math.exp(0.5 * it["z"]) if delta > 0 else 0.0


def is_bad(dom, node, k):
    c = CFG[dom]
    return node in c["always"] or (node == c["sleeper"] and k >= 30)


def sheet_line(dom, it, f, rng):
    j = lambda x: x * (1 + 0.01 * rng.gauss(0, 1))
    if dom == "forest":
        return f"{it['id']}: {it['name']} — forest area {j(it['a'] * f):,.1f} km², land area {j(it['b']):,.0f} km²"
    if dom == "urban":
        return f"{it['id']}: {it['name']} — urban population {j(it['a'] * f):,.0f}, total population {j(it['b']):,.0f}"
    if dom == "pendulum":
        return f"{it['id']}: length {j(it['a'] * f * f):.4f} m, gravitational acceleration {j(it['b']):.3f} m/s²"
    return f"{it['id']}: principal {j(it['a'] * f):,.2f} US$, annual interest rate {100 * j(it['b']):.3f}% compounded yearly, {it['n']} years"


QUESTION = {"forest": "give each country's forest area as a percentage of its land area", "urban": "give each country's urban population as a percentage of its total population",
            "pendulum": "give each simple pendulum's period in seconds", "interest": "give each account's final balance in US dollars"}


def check_text(dom, its):
    if dom == "forest":
        return "From your own knowledge (no other data is given), give each country's forest area as a percentage of land area in 2021.", [f"{it['id']}: {it['name']}" for it in its]
    if dom == "urban":
        return "From your own knowledge (no other data is given), give each country's urban population as a percentage of total population in 2022.", [f"{it['id']}: {it['name']}" for it in its]
    if dom == "pendulum":
        return "Here is a stopwatch log of pendulum experiments. Give each pendulum's period in seconds.", [f"{it['id']}: 20 full swings took {it['rec']:.3f} s" for it in its]
    return "Here is a set of independent bank statements. Give each account's final balance in US dollars.", [f"{it['id']}: closing balance on the final statement {it['rec']:,.2f} US$" for it in its]


def job(node, key, text, think):
    path = os.path.join(CACHE, f"{node}__{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask(CHECKER if node == "check_0" else node, text, 4000, think)
            json.dump({"answers": parse(raw), "raw": raw}, open(path, "w"))
            return node, key, "ok"
        except Exception as ex:
            err = repr(ex)[:120]
            time.sleep(10)
    return node, key, f"FAILED {err}"


def collect():
    os.makedirs(CACHE, exist_ok=True)
    jobs = []
    for dom in CFG:
        its = items(dom)
        B = 20 if len(its) > 30 else 15
        tail = f"\nReply with ONLY a JSON object mapping the ID to a number ({CFG[dom]['unit']}). No prose.\n\n"
        for node in NODES:
            for delta in DELTAS:
                rng = random.Random(f"big2-{dom}-{node}-{delta}")
                lines = [sheet_line(dom, it, 1 + dk(it, delta) if is_bad(dom, node, k) else 1.0, rng) for k, it in enumerate(its)]
                for b in range(0, len(its), B):
                    jobs.append((node, f"{dom}_{delta}_{b // B}", f"Here is a data sheet from a document you retrieved. Using it, {QUESTION[dom]}."
                                 + tail + "\n".join(lines[b:b + B]), NODES[node][0] != "deepseek"))
        q, lines = check_text(dom, its)
        for b in range(0, len(its), B):
            jobs.append(("check_0", f"{dom}_0.0_{b // B}", q + tail + "\n".join(lines[b:b + B]), True))
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools[NODES[CHECKER][0] if j[0] == "check_0" else NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def setup(dom):
    """data[(约束, δ)] = 按编号顺序的 [(是否故障, p − 真值, 报告 − 真值, log10(1 + δ_k))]；gdp 按预注册规则过滤。"""
    import torch
    its = items(dom)
    rep = {}
    for fn in os.listdir(CACHE):
        node, key = fn[:-5].split("__")
        d0, dd, _ = key.split("_")
        if d0 != dom:
            continue
        for i, v in json.load(open(os.path.join(CACHE, fn)))["answers"].items():
            try:
                if v > 0:
                    rep.setdefault(float(dd), {}).setdefault(i, {})[node] = math.log10(v)
            except TypeError:
                pass
    nodes = sorted(NODES)
    keep = []
    for it in its:
        v0 = rep.get(0.0, {}).get(it["id"], {})
        keep.append(it)
    data = {}
    for z in ("R0", "R1"):
        for delta in DELTAS:
            rows = []
            for k, it in enumerate(its):
                if it not in keep:
                    continue
                i, tru = it["id"], math.log10(it["truth"])
                v = rep.get(delta, {}).get(i, {})
                c = rep.get(0.0, {}).get(i, {}).get("check_0")
                if not all(n in v for n in nodes) or c is None:
                    continue
                x = dk(it, delta)
                p = c if z == "R0" else tru + math.log10(1 + x)
                rows.append((delta > 0, p - tru, torch.tensor([v[n] - tru for n in nodes]), math.log10(1 + x), k))
            data[(z, delta)] = rows
    return data


def _sw():
    sys.path.append(os.path.join(HERE, "..", "switching"))      # 追加在后：两个目录都有 analyze.py


def strip(rows):
    return [(f, p, r, x) for f, p, r, x, _ in rows]


def evaluate(dom, judge):
    _sw()
    from hetero import measure
    from intervals import interval2
    data = setup(dom)
    out = []
    for z in ("R0", "R1"):
        clean = strip(data[(z, 0.0)])
        for delta in DELTAS[1:]:
            bad = strip(data[(z, delta)])
            pt, lo, hi, s, rh, bw = interval2(clean, bad, judge)
            m = measure([(False, p, r, x) for _, p, r, x in clean] + bad, math.log10(1 + delta), judge, s)
            out.append({"z": z, "δ": delta, "n": len(bad), "point": pt, "lo": lo, "hi": hi, "rho": rh, "miss": m["miss"], "vote": m["vote"]})
    return out


def ewma(dom, delta):
    """按编号跑 EWMA（带秤坏保护）：返回 各节点第一次越线的轮次（下标从 1）和卧底的 μ。"""
    _sw()
    import torch
    from analytic import sigma_c
    from purge import C, SD_S, W, baseline, noise, scale_broken
    nodes = sorted(NODES)
    data = setup(dom)
    clean = strip(data[("R0", 0.0)])
    b, sc = sigma_c(clean)
    if True:                   # 三角帽法估节点噪声（偏离中位数的 MAD 会低估 15–20%，门槛偏紧、误报变多）
        from analytic import node_noise
        sig = node_noise(clean).sqrt()
    else:
        sig = noise(clean)
    u = baseline(clean, b) if BASE else torch.zeros(len(nodes))
    rows = data[("R0", delta)]
    S = torch.zeros(len(nodes))
    first = {}
    for _, p, r, _, k in rows:
        z = (r - u - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)
        if True:              # 单轮偏移截到 ±3σ（Huber）：一次大错不够越线，要连续偏才算暴露
            z = z.clamp(-3, 3)
        new = (1 - W) * S + W * z
        if scale_broken(new.tolist(), C * SD_S):
            continue
        S = new
        for j in torch.nonzero(S.abs() > C * SD_S).flatten().tolist():
            first.setdefault(nodes[j], k + 1)
    mu = None
    sl = CFG[dom]["sleeper"]
    if sl and delta > 0:
        ld = st.mean(x for *_, x, _ in rows)
        mu = ld / math.sqrt(sig[nodes.index(sl)].item() ** 2 + sc ** 2)
    return first, mu


def truly_false(dom, first):
    """越线前 10 轮里这个节点相对真值的平均 |误差| < 2 倍它自己的噪声 → 真误报；否则是抓到的自然错误。（只在计分里用真值）"""
    _sw()
    from analytic import node_noise
    nodes = sorted(NODES)
    data = setup(dom)
    rows = data[("R0", 0.0)]
    sig = node_noise(strip(rows)).sqrt()
    false_, natural = {}, {}
    for n, k in first.items():
        j = nodes.index(n)
        prev = [abs(r[j].item()) for _, _, r, _, kk in rows if k - 10 <= kk + 1 <= k]
        (false_ if prev and st.mean(prev) < 2 * sig[j].item() else natural)[n] = k
    return false_, natural


def learning(dom):
    _sw()
    from online import run, segs
    data = setup(dom)
    clean = strip(data[("R0", 0.0)])
    out = {}
    for d in DELTAS[1:]:
        curves = []
        for o in range(50):
            f = [(p, r, x / 2) for _, p, r, x, _ in data[("R0", d)]]
            random.Random(o).shuffle(f)
            curves.append(run(clean, f, o, guard=True))
        out[d] = segs(curves)
    return out


def predict():
    _sw()
    from detect_theory import theory
    res = {}
    for dom in CFG:
        res[dom] = {j: [{k: r[k] for k in ("z", "δ", "n", "point", "lo", "hi", "rho")} for r in evaluate(dom, j)] for j in ("软", "硬2")}
        if CFG[dom]["sleeper"]:
            res[dom]["detect"] = {str(d): theory(ewma(dom, d)[1], 1.0)[0] for d in (0.1, 0.2)}
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pred": res}, open(PRED, "w"), ensure_ascii=False)
    for dom, v in res.items():
        print(dom, "题数", v["软"][0]["n"], "；软 R0：", " ".join(f"{r['δ']:.0%}:{r['point']:.0%}[{r['lo']:.0%},{r['hi']:.0%}]ρ̂{r['rho']:.1f}" for r in v["软"] if r["z"] == "R0"),
              "；卧底公式：", v.get("detect"))


def score():
    pred = json.load(open(PRED))
    print(f"预测写于 {pred['time']}")
    cov, perr, r1, det, honest, learn = [], [], [], [], 0, []
    for dom in CFG:
        print(f"\n######## {dom} ########")
        for j in ("软", "硬2"):
            P = {(r["z"], r["δ"]): r for r in pred["pred"][dom][j]}
            for r in evaluate(dom, j):
                q = P[(r["z"], r["δ"])]
                inside = q["lo"] <= r["miss"] <= q["hi"]
                if r["z"] == "R0":
                    cov.append(inside)
                    if q["rho"] >= 1:
                        perr.append(abs(r["miss"] - q["point"]))
                elif r["δ"] >= 0.1:
                    r1.append(r["miss"] - q["point"] >= 0.30)
                print(f"  {j} {r['z']} {r['δ']:4.0%} 题 {r['n']:2d} ρ̂ {q['rho']:6.2f}  实际 {r['miss']:4.0%}  点 {q['point']:4.0%}  [{q['lo']:4.0%}, {q['hi']:4.0%}] {'✅' if inside else '❌'}  投票 {r['vote']:4.0%}")
        first0, _ = ewma(dom, 0.0)
        false_, natural = truly_false(dom, first0)
        honest += len(false_)
        print(f"  δ = 0 一轮 EWMA 越线：真误报 {false_ or '无'}；抓到的自然错误 {natural or '无'}")
        if CFG[dom]["sleeper"]:
            for d in (0.1, 0.2):
                first, mu = ewma(dom, d)
                k = first.get(CFG[dom]["sleeper"])
                th = pred["pred"][dom]["detect"][str(d)]
                got = None if k is None or k < 31 else k - 31
                ok = got is not None and abs(got - th) <= max(2, 0.3 * th)
                det.append(ok)
                print(f"  卧底 δ = {d:.0%} μ = {mu:.2f}：越线 第 {k} 轮（下毒后第 {got} 轮），公式 {th:.0f} {'✅' if ok else '❌'}；一直下毒的越线：{ {n: first.get(n) for n in sorted(CFG[dom]['always'])} }")
        L = learning(dom)
        for d, sg in L.items():
            static, last = sg[-1][1], sg[-1][3]
            if st.mean(x[1] for x in sg) >= 0.3:
                learn.append(last <= static)
            print(f"  学习裁判 δ = {d:.0%}：" + "  ".join(f"{lo}–{hi} 静 {a:.0%} / 学 {l:.0%}" for (lo, hi), a, pa, l, pl in sg))
    print("\n======== 汇总（预注册）========")
    print(f"1. 区间覆盖 {st.mean(cov):.0%}（{len(cov)} 格，≥ 90%）")
    print(f"2. ρ̂ ≥ 1 点预测误差中位 {st.median(perr):.3f}（{len(perr)} 格，≤ 0.08）")
    print(f"3. ζ = 1 乐观 ≥ +0.30 的比例 {st.mean(r1):.0%}（{len(r1)} 格，≥ 80%）")
    print(f"4. 卧底出局轮数命中 {sum(det)}/{len(det)}（≥ 3/4）")
    print(f"5. δ = 0 真误报总数 {honest}（≤ 2）")
    print(f"6. 学习裁判不差于静态 {sum(learn)}/{len(learn)}（≥ 75%）")


if __name__ == "__main__":
    {"collect": collect, "predict": predict, "score": score}[sys.argv[1]]()
