"""大规模真实系统测试：4 个新领域 × 同一套规律，一次检验全部组件（2026-11-01）。

领域（题目按编号当轮次；全部用 log10）：
  gdp     人均 GDP（真实，世界银行 2022）：资料 GDP（十亿美元）+ 人口（百万）→ 节点算人均；
          约束 = deepseek-v4-pro 凭记忆报人均 GDP（不给资料，ζ ≈ 0）；投毒 ds_flash_a、ds_flash_b、ds_flash_c、ds_pro_b（4 / 7，过半）：GDP × (1 + δ_k)
          题目过滤（预注册的规则，只用 δ = 0 那一轮）：干净那一轮节点中位数和世界银行差 > 5% 的国家剔除（世界银行自己也会错）
  chem    物质的量（真实化合物，IUPAC 原子量）：资料 样品质量 g + 摩尔质量 g/mol → mol；
          约束 = deepseek-v4-pro 只看化学式和质量自己算（ζ ≈ 0）；投毒 ds_flash_a、ds_flash_b、ds_pro_b（3 / 7）+ 卧底 ds_pro_a（第 31 题起）：摩尔质量 ÷ (1 + δ_k)
  planet  天体平均密度（真实，30 个行星 / 卫星 / 矮行星）：资料 质量 kg + 半径 km → g/cm³；
          约束 = deepseek-v4-pro 看独立记录（表面重力 m/s²，带 3% 误差 + 半径）换算（ζ ≈ 0）；投毒 5 个 DeepSeek（5 / 7）：质量 × (1 + δ_k)
  power   电费（合成）：资料 用电 kWh + 电价 $/kWh → 美元；
          约束 = deepseek-v4-pro 看电表起止读数（带 3% 误差）+ 电价（美分）换算（ζ ≈ 0）；投毒 ds_flash_b、ds_pro_b（2 / 7）+ 卧底 ds_pro_a（第 31 题起）：电价 × (1 + δ_k)
  干净节点里都有 Codex、Claude Haiku。δ_k = δ · exp(0.5 z) 每题不同，δ ∈ {0, 5%, 10%, 20%}。每个节点的资料另加 1% 抖动。
  R0 = 上面的独立约束；R1 = 约束直接用投毒资料算（ζ = 1）。
全部预测只用可观测量（裁判宽度、b、σ_c、节点噪声从 δ = 0 那一轮估），写进 bigtest_pred.json、commit 之后才计分。

预注册（11-01，采集之前写下；跨 4 个领域汇总）：
  1. R0 全部 (领域, δ, 裁判) 格子：实际漏检落在区间内的比例 ≥ 90%
  2. R0 中 ρ̂ ≥ 1 的格子：点预测 |误差| 中位 ≤ 0.08
  3. R1 中 δ ≥ 10% 的格子：≥ 80% 满足 实际 − 点预测 ≥ +0.30
  4. 卧底（chem、power；δ = 10%、20%）：按编号跑 EWMA（带秤坏保护），下毒后到越线的轮数和公式相差 ≤ max(2 轮, 30%)，4 个里 ≥ 3 个
  5. 4 个领域 δ = 0 那一轮按编号跑 EWMA（带秤坏保护）：诚实节点越线总数 ≤ 2（期望误报约 7 × 60 × 4 / 1000 ≈ 1.7）
  6. 学习裁判（带秤坏保护，50 种顺序）：R0 静态软裁判被骗 ≥ 30% 的格子里，≥ 75% 满足最后一段（41–60 轮）≤ 静态

用法：~/gut-external/torch215-venv/bin/python bigtest.py collect | predict | score
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

from collect import NODES, ask, parse

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_big")
PRED = os.path.join(HERE, "bigtest_pred.json")
DELTAS = [0.0, 0.05, 0.1, 0.2]
AW = {"H": 1.008, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990,
      "Mg": 24.305, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904}
COMPOUNDS = ["H2O", "CO2", "CH4", "C2H6", "C3H8", "C4H10", "C2H4", "C2H2", "C6H6", "C7H8", "C8H10", "C10H8", "CH4O", "C2H6O",
             "C3H8O", "C4H10O", "C3H6O", "C2H4O2", "CH2O2", "C6H12O6", "C12H22O11", "C6H8O7", "C9H8O4", "C8H10N4O2", "C10H14N2",
             "C3H7NO2", "C2H5NO2", "C5H9NO4", "C6H14N4O2", "C4H4N2O2", "C5H5N5", "NH3", "HNO3", "H2SO4", "H3PO4", "NaCl", "NaHCO3",
             "Na2CO3", "MgSO4", "MgCl2", "NaOH", "Na2SO4", "NaNO3", "Na3PO4", "HCl", "HBr", "NaBr", "NaI", "SiO2", "SiCl4", "PCl3",
             "PCl5", "SF6", "CCl4", "CHCl3", "CH2Cl2", "C6H5Cl", "BF3", "LiCl", "Li2CO3"]
BODIES = [("Mercury", 3.301e23, 2439.7), ("Venus", 4.867e24, 6051.8), ("Earth", 5.972e24, 6371.0), ("Mars", 6.417e23, 3389.5),
          ("Jupiter", 1.898e27, 69911), ("Saturn", 5.683e26, 58232), ("Uranus", 8.681e25, 25362), ("Neptune", 1.024e26, 24622),
          ("Moon", 7.342e22, 1737.4), ("Io", 8.932e22, 1821.6), ("Europa", 4.80e22, 1560.8), ("Ganymede", 1.4819e23, 2634.1),
          ("Callisto", 1.0759e23, 2410.3), ("Titan", 1.3452e23, 2574.7), ("Triton", 2.14e22, 1353.4), ("Pluto", 1.303e22, 1188.3),
          ("Charon", 1.586e21, 606.0), ("Enceladus", 1.08e20, 252.1), ("Tethys", 6.17e20, 531.1), ("Dione", 1.095e21, 561.4),
          ("Rhea", 2.306e21, 763.8), ("Iapetus", 1.806e21, 734.5), ("Mimas", 3.75e19, 198.2), ("Titania", 3.4e21, 788.4),
          ("Oberon", 3.08e21, 761.4), ("Ariel", 1.25e21, 578.9), ("Umbriel", 1.28e21, 584.7), ("Ceres", 9.38e20, 469.7),
          ("Eris", 1.66e22, 1163.0), ("Vesta", 2.59e20, 262.7)]
G = 6.674e-11
CFG = {
    "gdp": {"always": {"ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_b"}, "sleeper": None, "unit": "US dollars per person"},
    "chem": {"always": {"ds_flash_a", "ds_flash_b", "ds_pro_b"}, "sleeper": "ds_pro_a", "unit": "moles"},
    "planet": {"always": {"ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "ds_pro_b"}, "sleeper": None, "unit": "grams per cubic centimetre"},
    "power": {"always": {"ds_flash_b", "ds_pro_b"}, "sleeper": "ds_pro_a", "unit": "US dollars"},
}


def molar(f):
    return sum(AW[e] * (int(n) if n else 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", f))


def items(dom):
    rng = random.Random(f"big-{dom}")
    out = []
    if dom == "gdp":
        meta = {c["id"]: c for c in json.load(open(os.path.join(HERE, "wb_countries.json")))[1]}
        pop = {r["countryiso3code"]: r["value"] for r in json.load(open(os.path.join(HERE, "wb_SP.POP.TOTL.json")))[1] if r["value"]}
        gdp = {r["countryiso3code"]: r["value"] for r in json.load(open(os.path.join(HERE, "wb_NY.GDP.MKTP.CD.json")))[1] if r["value"]}
        ok = sorted(i for i, c in meta.items() if c["region"]["value"] != "Aggregates" and i in pop and i in gdp and pop[i] > 1e6)
        for i in sorted(rng.sample(ok, 60)):
            out.append({"id": i, "name": meta[i]["name"], "a": gdp[i] / 1e9, "b": pop[i] / 1e6, "truth": gdp[i] / pop[i], "z": rng.gauss(0, 1)})
    elif dom == "chem":
        for k, f in enumerate(COMPOUNDS):
            m = 10 ** rng.uniform(-1, 3)
            out.append({"id": f"C{k:02d}", "name": f, "a": m, "b": molar(f), "truth": m / molar(f), "z": rng.gauss(0, 1)})
    elif dom == "planet":
        for k, (nm, M, R) in enumerate(BODIES):
            rho = M / (4 / 3 * math.pi * (R * 1e5) ** 3) * 1000
            g = G * M / (R * 1e3) ** 2
            out.append({"id": f"B{k:02d}", "name": nm, "a": M, "b": R, "truth": rho, "z": rng.gauss(0, 1), "g": g * (1 + 0.03 * rng.gauss(0, 1))})
    else:
        for k in range(60):
            kwh, price = 10 ** rng.uniform(math.log10(50), math.log10(2000)), 10 ** rng.uniform(math.log10(0.08), math.log10(0.45))
            start = rng.uniform(1000, 90000)
            out.append({"id": f"H{k:02d}", "name": f"household {k}", "a": kwh, "b": price, "truth": kwh * price, "z": rng.gauss(0, 1),
                        "start": start, "end": start + kwh * (1 + 0.03 * rng.gauss(0, 1))})
    return out


def dk(it, delta):
    return delta * math.exp(0.5 * it["z"]) if delta > 0 else 0.0


def is_bad(dom, node, k):
    c = CFG[dom]
    return node in c["always"] or (node == c["sleeper"] and k >= 30)


def sheet_line(dom, it, f, rng):
    j = lambda x: x * (1 + 0.01 * rng.gauss(0, 1))
    if dom == "gdp":
        return f"{it['id']}: {it['name']} — GDP {j(it['a'] * f):.2f} billion US$, population {j(it['b']):.3f} million"
    if dom == "chem":
        return f"{it['id']}: {it['name']} — sample mass {j(it['a']):.4f} g, molar mass {j(it['b'] / f):.3f} g/mol"
    if dom == "planet":
        return f"{it['id']}: {it['name']} — mass {j(it['a'] * f):.4e} kg, mean radius {j(it['b']):.1f} km"
    return f"{it['id']}: {it['name']} — electricity used {j(it['a']):.1f} kWh, tariff {j(it['b'] * f):.4f} US$ per kWh"


QUESTION = {"gdp": "give each country's GDP per capita in US dollars", "chem": "give the amount of substance of each sample in moles",
            "planet": "give each body's mean density in grams per cubic centimetre", "power": "give each household's electricity bill in US dollars"}


def check_text(dom, its):
    if dom == "gdp":
        lines = [f"{it['id']}: {it['name']}" for it in its]
        return "From your own knowledge (no other data is given), give each country's GDP per capita in 2022, in current US dollars.", lines
    if dom == "chem":
        lines = [f"{it['id']}: {it['name']}, sample mass {it['a']:.4f} g" for it in its]
        return "For each sample, work out its molar mass from the chemical formula using standard atomic weights, then give the amount of substance in moles.", lines
    if dom == "planet":
        lines = [f"{it['id']}: {it['name']} — surface gravity {it['g']:.4f} m/s², mean radius {it['b']:.1f} km" for it in its]
        return "Using G = 6.674e-11 m³ kg⁻¹ s⁻², work out each body's mean density in grams per cubic centimetre from its surface gravity and radius.", lines
    lines = [f"{it['id']}: meter reading start {it['start']:.1f} kWh, end {it['end']:.1f} kWh, tariff {it['b'] * 100:.3f} cents per kWh" for it in its]
    return "From the meter readings and tariff, give each household's electricity bill in US dollars.", lines


def job(node, key, text, think):
    path = os.path.join(CACHE, f"{node}__{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask("ds_pro_a" if node == "check_0" else node, text, 4000, think)
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
                rng = random.Random(f"big-{dom}-{node}-{delta}")
                lines = [sheet_line(dom, it, 1 + dk(it, delta) if is_bad(dom, node, k) else 1.0, rng) for k, it in enumerate(its)]
                for b in range(0, len(its), B):
                    jobs.append((node, f"{dom}_{delta}_{b // B}", f"Here is a data sheet from a document you retrieved. Using it, {QUESTION[dom]}."
                                 + tail + "\n".join(lines[b:b + B]), NODES[node][0] != "deepseek"))
        q, lines = check_text(dom, its)
        for b in range(0, len(its), B):
            jobs.append(("check_0", f"{dom}_0.0_{b // B}", q + tail + "\n".join(lines[b:b + B]), False))
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools["deepseek" if j[0] == "check_0" else NODES[j[0]][0]].submit(job, *j) for j in jobs]
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
        if dom == "gdp" and all(n in v0 for n in nodes):
            if abs(st.median(v0[n] for n in nodes) - math.log10(it["truth"])) > math.log10(1.05):
                continue
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
    from purge import C, SD_S, W, noise, scale_broken
    nodes = sorted(NODES)
    data = setup(dom)
    clean = strip(data[("R0", 0.0)])
    b, sc = sigma_c(clean)
    sig = noise(clean)
    rows = data[("R0", delta)]
    S = torch.zeros(len(nodes))
    first = {}
    for _, p, r, _, k in rows:
        new = (1 - W) * S + W * (r - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)
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
        bad_nodes = CFG[dom]["always"] | ({CFG[dom]["sleeper"]} if CFG[dom]["sleeper"] else set())
        hc = {n: k for n, k in first0.items()}
        honest += len(hc)
        print(f"  δ = 0 一轮 EWMA 越线：{hc or '无'}")
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
    print(f"5. δ = 0 诚实节点越线总数 {honest}（≤ 2）")
    print(f"6. 学习裁判不差于静态 {sum(learn)}/{len(learn)}（≥ 75%）")


if __name__ == "__main__":
    {"collect": collect, "predict": predict, "score": score}[sys.argv[1]]()
