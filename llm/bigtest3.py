"""大规模真实系统测试第三轮：检测层两个修法 + 4 个新领域、新的节点角色（2026-11-01）。

第二轮教训：秤在个别题上错得很远（Sonnet 记忆：厄立特里亚 +22%、老挝 +26%），旧的秤坏保护要求"所有节点的线同向越线"，差一个就不触发；
一个极烂节点（复利里 ds_flash_a 错 −20~−90%）进了三角帽法的两两差，把别人的噪声估歪。
检测层修法（本轮写死）：
  1. 逐题秤坏判断：这道题秤离**每一个**节点都超过 3σ（标准化到 √(节点噪声² + 秤噪声²)）→ 这道题秤坏，不更新线。
     第二条前提的直接推论：只要有一个诚实节点，好秤至少和它接近。
  2. 留一法节点噪声：每个节点和"除它以外其他节点的中位数"比的 MAD（单个坏节点污染不了别人；略偏大 = 保守）。
  其余同第二轮（线扣节点常态偏移、单轮截到 ±3σ、权重 0.3、门槛 3.3σ）。
领域（δ ∈ {0, 5%, 10%, 20%}，δ_k = δ · exp(0.5 z)，资料另加 1% 抖动）：
  arable    耕地占国土 %（真实，世界银行 2021）：耕地公顷 ÷ 国土 km²；约束 = ds_pro_a 凭记忆；
            投毒 claude_sonnet、claude_haiku、codex、ds_flash_a、ds_pro_b（5 / 8，过半，含全部非 DeepSeek）：耕地 × (1 + δ_k)
  dosage    药量 mg（合成）：体重 kg × 每公斤剂量 mg/kg；约束 = ds_pro_b 看独立发药记录（片数 × 每片 mg，带 3% 误差）；
            投毒 ds_flash_b、ds_flash_c、claude_sonnet（3 / 8）+ 卧底 codex（第 31 题起）：每公斤剂量 × (1 + δ_k)
  fuel      油耗升数（合成）：里程 km × 百公里油耗 / 100；约束 = claude_sonnet 看独立加油小票（加油前后油表 + 油箱容量，带 3% 误差）；
            投毒 claude_haiku、ds_pro_a（2 / 8）+ 卧底 ds_flash_c（第 31 题起）：百公里油耗 × (1 + δ_k)
  fx        外币换算（合成）：金额 × 汇率；约束 = codex 看独立银行回单（到账金额，带 3% 误差）；
            投毒 ds_flash_a、ds_flash_b、ds_pro_a、ds_pro_b（4 / 8）+ 卧底 claude_haiku（第 31 题起）：汇率 × (1 + δ_k)

预注册（11-01，采集之前写下；跨 4 个领域汇总）：
  1. R0 区间覆盖 ≥ 90%
  2. R0 中 ρ̂ ≥ 1 的格子点预测 |误差| 中位 ≤ 0.08
  3. R1（ζ = 1）中 δ ≥ 10% 的格子：≥ 80% 满足 |实际 − 点预测| ≥ 0.30
  4. 卧底（dosage、fuel、fx；δ = 10%、20%）：下毒后到越线的轮数和公式相差 ≤ max(2 轮, 30%)，6 个里 ≥ 5 个
  5. δ = 0 一轮：真误报（越线前 10 轮里该节点相对真值平均 |误差| < 2 倍它自己的噪声）总数 ≤ 3
  6. 学习裁判（带秤坏保护）：R0 静态软裁判被骗 ≥ 30% 的格子里，≥ 75% 满足最后一段 ≤ 静态

用法：~/gut-external/torch215-venv/bin/python bigtest3.py collect | predict | score
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
CACHE = os.path.join(HERE, "cache_big3")
PRED = os.path.join(HERE, "bigtest3_pred.json")
DELTAS = [0.0, 0.05, 0.1, 0.2]
CFG = {
    "arable": {"always": {"claude_sonnet", "claude_haiku", "codex", "ds_flash_a", "ds_pro_b"}, "sleeper": None, "unit": "percent", "checker": "ds_pro_a"},
    "dosage": {"always": {"ds_flash_b", "ds_flash_c", "claude_sonnet"}, "sleeper": "codex", "unit": "milligrams", "checker": "ds_pro_b"},
    "fuel": {"always": {"claude_haiku", "ds_pro_a"}, "sleeper": "ds_flash_c", "unit": "litres", "checker": "claude_sonnet"},
    "fx": {"always": {"ds_flash_a", "ds_flash_b", "ds_pro_a", "ds_pro_b"}, "sleeper": "claude_haiku", "unit": "euros", "checker": "codex"},
}
BASE = True


def wb(ind):
    return {r["countryiso3code"]: r["value"] for r in json.load(open(os.path.join(HERE, f"wb_{ind}.json")))[1] if r["value"]}


def items(dom):
    rng = random.Random(f"big3-{dom}")
    out = []
    if dom == "arable":
        meta = {c["id"]: c for c in json.load(open(os.path.join(HERE, "wb_countries.json")))[1]}
        num, den = wb("AG.LND.ARBL.HA"), wb("AG.LND.TOTL.K2")
        ok = sorted(i for i, c in meta.items() if c["region"]["value"] != "Aggregates" and i in num and i in den and den[i] > 1e4 and num[i] > 1000)
        for i in sorted(rng.sample(ok, 60)):
            out.append({"id": i, "name": meta[i]["name"], "a": num[i], "b": den[i], "truth": num[i] / 100 / den[i] * 100, "z": rng.gauss(0, 1)})
    elif dom == "dosage":
        for k in range(60):
            w, dpk = 10 ** rng.uniform(math.log10(3), math.log10(120)), 10 ** rng.uniform(math.log10(0.05), math.log10(40))
            tab = rng.choice([1, 2, 5, 10, 25, 50, 100, 250, 500])
            dose = w * dpk
            out.append({"id": f"D{k:02d}", "name": f"patient {k}", "a": w, "b": dpk, "truth": dose, "z": rng.gauss(0, 1), "tab": tab, "n": dose / tab * (1 + 0.03 * rng.gauss(0, 1))})
    elif dom == "fuel":
        for k in range(60):
            km, lpk = 10 ** rng.uniform(1, math.log10(2000)), 10 ** rng.uniform(math.log10(3), math.log10(30))
            L = km * lpk / 100
            tank = rng.choice([40, 50, 60, 70, 80, 100, 400, 1000])
            used = L * (1 + 0.03 * rng.gauss(0, 1))
            before = rng.uniform(0.9, 1.0) * max(tank, used * 1.1)
            out.append({"id": f"F{k:02d}", "name": f"trip {k}", "a": km, "b": lpk, "truth": L, "z": rng.gauss(0, 1), "before": before, "after": before - used})
    else:
        for k in range(60):
            amt, rate = 10 ** rng.uniform(1, 6), 10 ** rng.uniform(math.log10(0.0005), math.log10(5))
            out.append({"id": f"X{k:02d}", "name": f"transfer {k}", "a": amt, "b": rate, "truth": amt * rate, "z": rng.gauss(0, 1), "rec": amt * rate * (1 + 0.03 * rng.gauss(0, 1))})
    return out


def dk(it, delta):
    return delta * math.exp(0.5 * it["z"]) if delta > 0 else 0.0


def is_bad(dom, node, k):
    c = CFG[dom]
    return node in c["always"] or (node == c["sleeper"] and k >= 30)


def sheet_line(dom, it, f, rng):
    j = lambda x: x * (1 + 0.01 * rng.gauss(0, 1))
    if dom == "arable":
        return f"{it['id']}: {it['name']} — arable land {j(it['a'] * f):,.0f} hectares, land area {j(it['b']):,.0f} km²"
    if dom == "dosage":
        return f"{it['id']}: body weight {j(it['a']):.2f} kg, prescribed dose {j(it['b'] * f):.4f} mg per kg"
    if dom == "fuel":
        return f"{it['id']}: distance {j(it['a']):.1f} km, average consumption {j(it['b'] * f):.3f} litres per 100 km"
    return f"{it['id']}: amount {j(it['a']):,.2f} units of foreign currency, exchange rate {j(it['b'] * f):.6f} euros per unit"


QUESTION = {"arable": "give each country's arable land as a percentage of its land area (note 1 km² = 100 hectares)",
            "dosage": "give each patient's total dose in milligrams", "fuel": "give the fuel used on each trip in litres",
            "fx": "give each converted amount in euros"}


def check_text(dom, its):
    if dom == "arable":
        return "From your own knowledge (no other data is given), give each country's arable land as a percentage of land area in 2021.", [f"{it['id']}: {it['name']}" for it in its]
    if dom == "dosage":
        return "Here is an independent dispensing log. Give each patient's total dose in milligrams.", [f"{it['id']}: dispensed {it['n']:.3f} tablets of {it['tab']} mg" for it in its]
    if dom == "fuel":
        return "Here are independent refuelling records (fuel gauge readings in litres). Give the fuel used on each trip in litres.", [f"{it['id']}: fuel before the trip {it['before']:.2f} L, after the trip {it['after']:.2f} L" for it in its]
    return "Here are independent bank credit slips. Give each converted amount in euros.", [f"{it['id']}: credited {it['rec']:,.2f} euros" for it in its]


def job(node, key, text, think):
    path = os.path.join(CACHE, f"{node}__{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask(node.split("@")[1] if node.startswith("check_0@") else node, text, 4000, think)
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
                rng = random.Random(f"big3-{dom}-{node}-{delta}")
                lines = [sheet_line(dom, it, 1 + dk(it, delta) if is_bad(dom, node, k) else 1.0, rng) for k, it in enumerate(its)]
                for b in range(0, len(its), B):
                    jobs.append((node, f"{dom}_{delta}_{b // B}", f"Here is a data sheet from a document you retrieved. Using it, {QUESTION[dom]}."
                                 + tail + "\n".join(lines[b:b + B]), NODES[node][0] != "deepseek"))
        q, lines = check_text(dom, its)
        for b in range(0, len(its), B):
            jobs.append((f"check_0@{CFG[dom]['checker']}", f"{dom}_0.0_{b // B}", q + tail + "\n".join(lines[b:b + B]), NODES[CFG[dom]["checker"]][0] != "deepseek"))
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools[NODES[j[0].split("@")[1]][0] if j[0].startswith("check_0@") else NODES[j[0]][0]].submit(job, *j) for j in jobs]
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
        node = "check_0" if node.startswith("check_0@") else node
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
    R = torch.stack([r for _, _, r, _ in clean])                # 留一法：和"除自己以外的节点中位数"比
    sig = torch.stack([1.4826 * (R[:, i] - torch.cat([R[:, :i], R[:, i + 1:]], 1).median(1).values).abs().median() for i in range(R.shape[1])]).clamp(min=1e-6)
    u = baseline(clean, b) if BASE else torch.zeros(len(nodes))
    rows = data[("R0", delta)]
    S = torch.zeros(len(nodes))
    first = {}
    for _, p, r, _, k in rows:
        z = (r - u - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)
        if True:              # 单轮偏移截到 ±3σ（Huber）：一次大错不够越线，要连续偏才算暴露
            z = z.clamp(-3, 3)
        new = (1 - W) * S + W * z
        if (r - (p - b)).abs().div(torch.sqrt(sig ** 2 + sc ** 2)).min() > 3:   # 逐题秤坏：秤离每一个节点都超过 3σ
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
    import torch
    nodes = sorted(NODES)
    data = setup(dom)
    rows = data[("R0", 0.0)]
    R = torch.stack([r for _, _, r, _ in strip(rows)])
    sig = torch.stack([1.4826 * (R[:, i] - torch.cat([R[:, :i], R[:, i + 1:]], 1).median(1).values).abs().median() for i in range(R.shape[1])])
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
                    r1.append(abs(r["miss"] - q["point"]) >= 0.30)
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
    print(f"3. ζ = 1 预测错（|误差| ≥ 0.30）的比例 {st.mean(r1):.0%}（{len(r1)} 格，≥ 80%）")
    print(f"4. 卧底出局轮数命中 {sum(det)}/{len(det)}（≥ 5/6）")
    print(f"5. δ = 0 真误报总数 {honest}（≤ 3）")
    print(f"6. 学习裁判不差于静态 {sum(learn)}/{len(learn)}（≥ 75%）")


if __name__ == "__main__":
    {"collect": collect, "predict": predict, "score": score}[sys.argv[1]]()
