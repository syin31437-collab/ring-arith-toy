"""不靠 LLM 的秤：规则 / 物理定律 / 数据库各做一个小样，看哪个稳定（2026-11-01，库伯）。

节点数据不重采（四轮大规模里已有的 LLM 节点报告），只把秤换成不靠 LLM 的来源：
  规则    bigtest:chem     摩尔数 = 质量 / 摩尔质量，摩尔质量用"整数原子量"（课本近似，round(原子量)）——没有数据源，只有确定性的近似误差
  物理    bigtest:planet   行星：赤道重力 g + 平均半径 → ρ = 3g / (4πG·R)；卫星 / 小行星：轨道跟踪测的 GM + 平均半径 → ρ = 3GM / (4πG·R³)
                           （JPL：ssd.jpl.nasa.gov planets/phys_par、sats/phys_par、sbdb API；真值来自另一张质量表）
  数据库  bigtest4:density 人口密度 = Wikidata 人口 / Wikidata 面积（真值是世界银行人口 / 陆地面积；口径和年份都不同）
外部数据缓存在 extscale/（下载于 11-01）。

评分（和 memscale 同一套）：秤相对真值的 MAD、错 > 10% 的题数、ζ 检查（秤误差与干净轮节点中位误差的相关）、
  δ = 0 轮闸门版 EWMA 的误报数、最大 δ 轮"一直投毒"节点被抓比例、区间覆盖（R0，软 / 硬2）、ρ̂ ≥ 1 格子的点预测误差中位。
"稳定"判据（全部满足）：MAD ≤ 5%，错 > 10% 的题 ≤ 10%，|ζ 相关| < 0.3，区间覆盖 ≥ 90%，点误差中位 ≤ 0.08，误报 0；
  而且题目按奇偶拆两半，两半各自也满足（误报按半数算仍须 0）。
防假失败 / 假通过的对照（每个领域都跑）：
  近真值秤：真值 × exp(N(0, 0.5%))      → 必须"稳定"（否则是评分管线假失败）
  打乱秤：  外部秤的值在题目间随机打乱  → 必须"不稳定"（否则是评分管线假通过）
  解析核对：物理秤的卫星密度与 JPL 表里自带的密度列比对（差 < 1%），化学秤 H2O = 18、NaCl = 58；不过就先修解析、不评分。

预注册后改动（运行时第一次就触发、未评分）：解析核对原为"全部差 < 1%"，Charon 不过——JPL 表里 GM = 106.1、R = 606.0 算出 1.705，
  它自带的密度列写 1.853，是源表自身不一致（不是解析错）；改为"≥ 90% 差 < 1%，列出其余"，Charon 用 GM 路线的值（物理定律路线）。
预注册（11-01，评分之前写下；外部数据已下载、尚未和真值比对过）：
  1. 两个对照：近真值 3/3 稳定、打乱 3/3 不稳定（不满足 → 整轮作废，先修管线）
  2. 规则（chem）：稳定；MAD ≤ 1%
  3. 物理（planet）：稳定；MAD ≤ 3%（行星用赤道重力会偏几个百分点）
  4. 数据库（density）：最弱；MAD ≤ 5%，但错 > 10% 的题可能 ≥ 3 道（面积口径、人口年份）；整体判"稳定"
  5. 稳定程度排序（按 MAD）：规则 < 物理 < 数据库

第一轮结果（a18ba22 后 + 解析核对放宽）：检查 1 不过（近真值 2/3：planet 近真值误报 2）→ 按规定整轮作废、先修管线。
  其余照记：规则 chem 稳定（MAD 0.13%，两半都稳）；物理 planet MAD 0.20%、ζ −0.04、覆盖 100%，只因误报 4 判不稳；
  数据库 density MAD 7.3%、错 > 10% 的 28%（误报 0、抓投毒 100%、覆盖 100%）；打乱 3/3 不稳定，但打乱秤覆盖也是 100%——区间覆盖分不出烂秤，判据得靠 MAD / 粗错。
  planet 误报的来源：节点按质量和半径算密度，几乎没噪声，loo 噪声塌成 0（3 个节点），取整差就越线；原 LLM 秤也误报 4、近真值误报 2 → 是领域 + 检测层的性质，与秤无关。
  试过在 loo 里加"MAD 塌成 0 改用平均绝对偏差"的兜底（通用层）：回归不过（其他领域投毒没抓到 0 → 1–2、sound 多 1 个误报），已撤回。
第二轮（管线修正：误报改为相对对照）：误报判据从"= 0"改为"≤ 同一份题的近真值对照的误报"（全量、两半各自配自己的对照）；对照换新种子 ext2。
第二轮预注册（运行前）：
  1. 近真值 3/3 稳定、打乱 3/3 不稳定（须全满足，否则再作废）
  2. 规则 chem：稳定（第一轮已稳，误报都是 0，判据改动对它无影响）
  3. 数据库 density：不稳定（MAD 7.3% > 5%，与误报判据无关）；补查：误差主要来自面积口径（Wikidata 总面积 vs 世界银行陆地面积），
     人口那一半的 MAD < 3%、面积那一半的 MAD > 人口那一半
  4. 物理 planet：不预测（误报 4 vs 新对照多少事先不知道）

第二轮结果（d69172b）：1 ✅（近真值 3/3、打乱 3/3）；2 ✅ 规则全量 + 两半都稳；3 半对——density 不稳定 ✅，但补查方向错 ❌：
  人口那一半 MAD 4.26% > 面积那一半 2.34%（主因是人口年份 / 口径，不是面积口径）；4 planet 外部秤误报 4 vs 对照 3 → 判不稳定。
  复查 planet（诊断，非预注册）：近真值噪声 0.1–1% 各 5 个种子，误报在 0–4 之间随机跳（0.2% 时 [4, 2, 1, 1, 1]）
  → 外部秤的 4 落在对照分布里，误报在这个领域分不出秤好坏；planet 在秤指标上（MAD 0.20%、ζ −0.04、覆盖 100%、点误差 0.034）和规则同级。
  判定：planet = "秤稳定，检测层在无噪声节点领域不稳"，不算秤的失败。
  另一处假失败风险：ζ 阈值 0.3 在 15 题上是噪声（近真值偶数半 ζ = −0.36 被判不稳）；之后 ζ 判据应按显著性（|r| > 2/√n）。

第三轮（11-01，库伯：每类加一个领域）：
  规则    bigtest2:interest  复利终值，秤 = 连续复利近似 P·e^{rn}（确定性公式近似；对数误差 = n·(r − ln(1 + r)) ≈ n·r²/2，单向、随 r、n 变大）
  物理    bigtest5:lattice   元素密度（真实，29 个元素），秤 = X 射线晶格常数 ρ = Z·M / (N_A·a³)（Ashcroft & Mermin，3 位有效数字，碱金属是低温值）；节点新采集
  数据库  bigtest:gdp        人均 GDP（世界银行 2022），秤 = IMF World Economic Outlook 2022 人均 GDP（同一年份——检验"年份对齐后数据库秤就稳"）
  判据同第二轮（误报 ≤ 近真值对照），另把 ζ 改成"|ζ| ≥ 0.3 且 > 2/√n 才算同源"（第二轮发现固定阈值在 15 题上假失败）；对照新种子 ext3。
第三轮预注册（11-01；lattice 节点采集前、三个秤都还没和真值比对）：
  1. 近真值 3/3 稳定、打乱 3/3 不稳定
  2. 规则 interest：稳定（全量 + 两半），MAD ≤ 2%
  3. 物理 lattice：稳定（全量 + 两半），MAD ≤ 2%，错 > 10% 的题 0
  4. 数据库 gdp（年份对齐）：稳定（全量 + 两半），MAD ≤ 2%
  5. 汇总：规则 2/2、物理 2/2 稳定（物理 planet 按第二轮复查算"秤稳定"）；数据库 1/2（density 年份不齐不稳、gdp 年份齐稳）

第三轮结果（822877f；lattice 采集 64 次请求全成功）：1 ✅；2 ❌ interest 不稳定（MAD 0.82%，但错 > 10% 的题 12%）；3 ✅ lattice 全量 + 两半都稳（MAD 0.57%、粗错 0）；
  4 ❌ gdp 全量稳定（MAD 1.11%、粗错正好 10%），偶数半粗错 20% 不稳；5 汇总：规则 1/2、物理 2/2、数据库 0/2（gdp 卡线）。
  粗错清单（复查，非预注册）：interest 的 7 道全是 r ≈ 0.09–0.12 且 n ≥ 21——正是 n·r²/2 > 0.095 的区域，误差不看真值就能算出来；
  gdp 的 6 道是 Sudan −32%、Turkmenistan +70%、Zimbabwe +18%、Somalia +15%、Gabon +13%、Venezuela −11%——多重汇率 / 汇率动荡的国家（两家折算美元的口径不同）。
  → 规则秤要分"误差有界"（取整：稳）和"误差无界、随输入变大"（公式近似：大输入处粗错）；数据库秤即使年份对齐，在定义有争议的条目上也会粗错。

第四轮（11-01，库伯：两件都做）：
  A. 规则秤自报误差：连续复利近似的误差有不看真值的上界 n·r²/2（ln(1 + r) ≥ r − r²/2）；上界 > ln(1.05)（5%）的题秤"弃权"（不当秤用），
     只在其余题上评分。一般规则都带误差上界（取整 ±0.5 单位、泰勒余项），这一步是通用的"秤按题自报能不能信"。
  B. 检测层 cm3（commonmode.py：分母下限 h_min / C），6 个领域全部用 cm3 重评。
第四轮预注册（运行前）：
  A1. interest 留下的题 ≥ 70%；留下的题上全量 + 两半都稳定，错 > 10% 的题 0，MAD ≤ 1%
  B1. planet（cm3）：外部秤全量 + 两半都稳定（误报 ≤ 近真值对照）
  B2. 另外 5 个领域（chem、density、lattice、gdp、interest 全量）的稳定 / 不稳定判定和 cm2 时一样

用法：~/gut-external/torch215-venv/bin/python extscale.py        （第一轮）
      ~/gut-external/torch215-venv/bin/python extscale.py v2     （第二轮：误报相对对照）
      ~/gut-external/torch215-venv/bin/python extscale.py v3     （第三轮：每类第二个领域）
      ~/gut-external/torch215-venv/bin/python extscale.py v4     （第四轮：规则秤自报误差 + 检测层 cm3）
"""
import html
import json
import math
import os
import random
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "extscale")
G = 6.674e-11


def num(tok):
    return float(tok.replace("±", "").replace(",", ""))


def rule_scale(its):
    import bigtest
    AWi = {e: round(w) for e, w in bigtest.AW.items()}
    mi = lambda f: sum(AWi[e] * (int(n) if n else 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", f))
    assert mi("H2O") == 18
    return {it["id"]: it["a"] / mi(it["name"]) for it in its}


def phys_scale(its):
    sats = json.load(open(os.path.join(EXT, "jpl_sats_rows.json")))
    pl = json.load(open(os.path.join(EXT, "jpl_planets_rows.json")))
    out, chk = {}, []
    for it in its:
        nm = it["name"]
        if nm in pl:
            v = [x for x in pl[nm] if re.match(r"^-?[\d.]+$", x)]
            R, g = float(v[1]), float(v[-2])                  # 平均半径（km）、赤道重力（m/s²）
            out[it["id"]] = 3 * g / (4 * math.pi * G * R * 1e3) / 1000
        elif nm in sats:
            t = sats[nm]
            GM, R, rho_tab = float(t[3]), float(t[6]), float(t[9])
            rho = 3 * GM * 1e9 / (4 * math.pi * G * (R * 1e3) ** 3) / 1000
            chk.append((nm, rho, rho_tab))
            out[it["id"]] = rho
        else:
            fn = os.path.join(EXT, f"sb_{nm}.json")
            if os.path.exists(fn):
                pp = {p["name"]: p["value"] for p in json.load(open(fn)).get("phys_par", [])}
                if "GM" in pp and "diameter" in pp:
                    R = float(pp["diameter"]) / 2
                    out[it["id"]] = 3 * float(pp["GM"]) * 1e9 / (4 * math.pi * G * (R * 1e3) ** 3) / 1000
    bad = [(n, a, b) for n, a, b in chk if abs(a / b - 1) > 0.01]
    assert len(bad) <= 0.1 * len(chk), f"解析核对不过：{bad}"
    return out, chk


def db_scale(its):
    rows = json.load(open(os.path.join(EXT, "wikidata.json")))["results"]["bindings"]
    by = {}
    for r in rows:
        try:
            by.setdefault(r["iso"]["value"], []).append(float(r["pop"]["value"]) / (float(r["area"]["value"]) / 1e6))
        except (KeyError, ValueError, ZeroDivisionError):
            pass
    return {it["id"]: st.median(by[it["id"]]) for it in its if it["id"] in by}


def score_one(data, nodes, always, deltas, its, scale, keep_k=None, mode="cm2"):
    """scale: id → 秤的原始值（非 log）。返回指标字典。"""
    import torch
    from commonmode import run_lines
    from hetero import measure
    from intervals import interval2
    from invert import loo
    tru = {it["id"]: math.log10(it["truth"]) for it in its}
    idk = {k: it["id"] for k, it in enumerate(its)}
    D = {}
    for key, rows in data.items():
        D[key] = [(f, math.log10(scale[idk[k]]) - tru[idk[k]], r, x, k) for f, p, r, x, k in rows
                  if idk[k] in scale and scale[idk[k]] > 0 and (keep_k is None or k in keep_k)]
    c5 = D[("R0", 0.0)]
    clean = [(f, p, r, x) for f, p, r, x, _ in c5]
    e = [p for _, p, _, _ in clean]
    med = st.median(e)
    mad = 10 ** (1.4826 * st.median(abs(x - med) for x in e)) - 1
    gross = st.mean(abs(10 ** x - 1) > 0.10 for x in e)
    zeta = torch.corrcoef(torch.stack([torch.tensor(e), torch.tensor([r.median().item() for _, _, r, _ in clean])]))[0, 1].item()
    from commonmode import floor_of
    fl = floor_of(deltas) if mode == "cm3" else 0.0
    first, _ = run_lines(clean, c5, nodes, mode, fl)
    sig = loo(torch.stack([r for _, _, r, _ in clean])).clamp(min=fl)
    nfalse = 0
    for n, kk in first.items():
        j = nodes.index(n)
        prev = [abs(r[j].item()) for _, _, r, _, k2 in c5 if kk - 10 <= k2 <= kk]
        if prev and st.mean(prev) < 2 * sig[j].item():
            nfalse += 1
    f_hi, _ = run_lines(clean, D[("R0", deltas[-1])], nodes, mode, fl)
    caught = st.mean(n in f_hi for n in always) if always else float("nan")
    cov, perr = [], []
    for d in deltas[1:]:
        bad = [(f, p, r, x) for f, p, r, x, _ in D[("R0", d)]]
        for judge in ("软", "硬2"):
            pt, lo, hi, s0, rh, bw = interval2(clean, bad, judge, B=30)
            act = measure([(False, p, r, x) for _, p, r, x in clean] + bad, math.log10(1 + d), judge, s0)["miss"]
            cov.append(lo <= act <= hi)
            if rh >= 1:
                perr.append(abs(act - pt))
    pm = st.median(perr) if perr else float("nan")
    m = {"n": len(clean), "mad": mad, "gross": gross, "zeta": zeta, "false": nfalse, "caught": caught, "cov": st.mean(cov), "perr": pm}
    zsig = abs(zeta) >= 0.3 and abs(zeta) > 2 / math.sqrt(len(clean))           # 第三轮：ζ 要同时够大、够显著才算同源
    m["ok_sig"] = (mad <= 0.05 and gross <= 0.10 and not zsig and m["cov"] >= 0.90 and (pm <= 0.08 or math.isnan(pm)))
    m["ok_nofa"] = (mad <= 0.05 and gross <= 0.10 and abs(zeta) < 0.3 and m["cov"] >= 0.90 and (pm <= 0.08 or math.isnan(pm)))
    m["ok"] = (mad <= 0.05 and gross <= 0.10 and abs(zeta) < 0.3 and m["cov"] >= 0.90 and (pm <= 0.08 or math.isnan(pm)) and nfalse == 0)
    return m


def fmt(m):
    return (f"n={m['n']:2d} MAD {m['mad']:6.2%} 错>10% {m['gross']:4.0%} ζ相关 {m['zeta']:+.2f} 误报 {m['false']} "
            f"抓投毒 {m['caught']:.0%} 覆盖 {m['cov']:4.0%} 点误差 {m['perr']:.3f}  → {'稳定' if m['ok'] else '不稳定'}")


def main():
    sys.path.insert(0, os.path.join(HERE, "..", "switching"))
    cwd = os.getcwd()
    from invert import sets
    S = {name: (data, nodes, always, deltas) for name, data, nodes, always, sleeper, deltas in sets()}
    os.chdir(cwd)
    sys.path.insert(0, HERE)
    import bigtest
    import bigtest4
    doms = [("规则 chem", "bigtest:chem", bigtest.items("chem"), rule_scale),
            ("物理 planet", "bigtest:planet", bigtest.items("planet"), lambda its: phys_scale(its)[0]),
            ("数据库 density", "bigtest4:density", bigtest4.items("density"), db_scale)]
    _, chk = phys_scale(bigtest.items("planet"))
    bad = [(n, round(a, 3), b) for n, a, b in chk if abs(a / b - 1) > 0.01]
    print(f"解析核对：卫星 {len(chk)} 个，与 JPL 自带密度列差 > 1% 的：{bad}")
    res = {}
    for tag, name, its, fn in doms:
        data, nodes, always, deltas = S[name]
        ext = fn(its)
        rng = random.Random(f"ext-{name}")
        ids = [i for i in ext]
        vals = [ext[i] for i in ids]
        rng.shuffle(vals)
        shuf = dict(zip(ids, vals))
        near = {it["id"]: it["truth"] * math.exp(rng.gauss(0, 0.005)) for it in its}
        ks = sorted({k for *_, k in data[("R0", 0.0)]})
        r_ext = score_one(data, nodes, always, deltas, its, ext)
        r_h = [score_one(data, nodes, always, deltas, its, ext, set(ks[j::2])) for j in (0, 1)]
        r_near = score_one(data, nodes, always, deltas, its, near)
        r_shuf = score_one(data, nodes, always, deltas, its, shuf)
        res[tag] = (r_ext, r_h, r_near, r_shuf)
        print(f"\n{tag}（外部秤覆盖 {len(ext)}/{len(its)} 题）")
        print(f"  外部秤   {fmt(r_ext)}")
        for j, r in enumerate(r_h):
            print(f"  {'奇偶'[j]}数半   {fmt(r)}")
        print(f"  近真值   {fmt(r_near)}")
        print(f"  打乱     {fmt(r_shuf)}")
    print("\n预注册检查：")
    print(f"1. 近真值稳定 {sum(v[2]['ok'] for v in res.values())}/3（须 3/3）；打乱不稳定 {sum(not v[3]['ok'] for v in res.values())}/3（须 3/3）")
    stable = lambda v: v[0]["ok"] and all(h["ok"] for h in v[1])
    c, p, d = res["规则 chem"], res["物理 planet"], res["数据库 density"]
    print(f"2. 规则：{'稳定' if stable(c) else '不稳定'}，MAD {c[0]['mad']:.2%}（≤ 1%）")
    print(f"3. 物理：{'稳定' if stable(p) else '不稳定'}，MAD {p[0]['mad']:.2%}（≤ 3%）")
    print(f"4. 数据库：{'稳定' if stable(d) else '不稳定'}，MAD {d[0]['mad']:.2%}（≤ 5%），错 > 10% 的题 {d[0]['gross']:.0%}")
    print(f"5. MAD 排序 规则 {c[0]['mad']:.2%} < 物理 {p[0]['mad']:.2%} < 数据库 {d[0]['mad']:.2%}：{'对' if c[0]['mad'] < p[0]['mad'] < d[0]['mad'] else '不对'}")


def main2():
    sys.path.insert(0, os.path.join(HERE, "..", "switching"))
    cwd = os.getcwd()
    from invert import sets
    S = {name: (data, nodes, always, deltas) for name, data, nodes, always, sleeper, deltas in sets()}
    os.chdir(cwd)
    sys.path.insert(0, HERE)
    import bigtest
    import bigtest4
    doms = [("规则 chem", "bigtest:chem", bigtest.items("chem"), rule_scale),
            ("物理 planet", "bigtest:planet", bigtest.items("planet"), lambda its: phys_scale(its)[0]),
            ("数据库 density", "bigtest4:density", bigtest4.items("density"), db_scale)]
    verdict = lambda m, ctl: m["ok_nofa"] and m["false"] <= ctl["false"]
    res = {}
    for tag, name, its, fn in doms:
        data, nodes, always, deltas = S[name]
        ext = fn(its)
        rng = random.Random(f"ext2-{name}")
        ids = list(ext)
        vals = [ext[i] for i in ids]
        rng.shuffle(vals)
        shuf = dict(zip(ids, vals))
        near = {it["id"]: it["truth"] * math.exp(rng.gauss(0, 0.005)) for it in its}
        ks = sorted({k for *_, k in data[("R0", 0.0)]})
        parts = [("全量", None), ("奇数半", set(ks[0::2])), ("偶数半", set(ks[1::2]))]
        print(f"\n{tag}（外部秤覆盖 {len(ext)}/{len(its)} 题）")
        rows = []
        for pn, keep in parts:
            e = score_one(data, nodes, always, deltas, its, ext, keep)
            c = score_one(data, nodes, always, deltas, its, near, keep)
            s = score_one(data, nodes, always, deltas, its, shuf, keep)
            ok_e, ok_c, ok_s = verdict(e, c), verdict(c, c), verdict(s, c)
            rows.append((ok_e, ok_c, ok_s))
            print(f"  {pn} 外部秤 {fmt(e).split('→')[0]}→ {'稳定' if ok_e else '不稳定'}（对照误报 {c['false']}）")
            print(f"  {pn} 近真值 {fmt(c).split('→')[0]}→ {'稳定' if ok_c else '不稳定'}")
            print(f"  {pn} 打乱   {fmt(s).split('→')[0]}→ {'稳定' if ok_s else '不稳定'}")
        res[tag] = rows
    print("\n第二轮预注册检查：")
    nc = sum(r[0][1] for r in res.values())
    ns = sum(not r[0][2] for r in res.values())
    print(f"1. 近真值稳定 {nc}/3、打乱不稳定 {ns}/3（须 3/3、3/3）")
    for tag in res:
        print(f"   {tag}：外部秤 全量 / 两半 = {' / '.join('稳定' if r[0] else '不稳定' for r in res[tag])}")
    its = bigtest4.items("density")
    rows_ = json.load(open(os.path.join(EXT, "wikidata.json")))["results"]["bindings"]
    pop, area = {}, {}
    for r in rows_:
        try:
            pop.setdefault(r["iso"]["value"], []).append(float(r["pop"]["value"]))
            area.setdefault(r["iso"]["value"], []).append(float(r["area"]["value"]) / 1e6)
        except (KeyError, ValueError):
            pass
    lp = [math.log10(st.median(pop[it["id"]]) / it["a"]) for it in its if it["id"] in pop]
    la = [math.log10(st.median(area[it["id"]]) / it["b"]) for it in its if it["id"] in area]
    madl = lambda v: 10 ** (1.4826 * st.median(abs(x - st.median(v)) for x in v)) - 1
    print(f"3. 数据库补查：人口（Wikidata / 世界银行）MAD {madl(lp):.2%}、中位偏 {10 ** st.median(lp) - 1:+.2%}；"
          f"面积（Wikidata 总面积 / 世界银行陆地面积）MAD {madl(la):.2%}、中位偏 {10 ** st.median(la) - 1:+.2%}")


def interest_scale(its):
    return {it["id"]: it["a"] * math.exp(it["b"] * it["n"]) for it in its}


def imf_scale(its):
    v = json.load(open(os.path.join(EXT, "imf_ngdpdpc_2022.json")))["values"]["NGDPDPC"]
    return {it["id"]: v[it["id"]]["2022"] for it in its if it["id"] in v and v[it["id"]].get("2022")}


def main3():
    sys.path.insert(0, os.path.join(HERE, "..", "switching"))
    cwd = os.getcwd()
    from invert import sets
    S = {name: (data, nodes, always, deltas) for name, data, nodes, always, sleeper, deltas in sets()}
    os.chdir(cwd)
    sys.path.insert(0, HERE)
    import bigtest
    import bigtest2
    import bigtest5
    S["bigtest5:lattice"] = (bigtest5.setup(), sorted(bigtest5.NODES), bigtest5.CFG["lattice"]["always"], bigtest5.DELTAS)
    doms = [("规则 interest", "bigtest2:interest", bigtest2.items("interest"), interest_scale),
            ("物理 lattice", "bigtest5:lattice", bigtest5.items(), bigtest5.lattice_scale),
            ("数据库 gdp", "bigtest:gdp", bigtest.items("gdp"), imf_scale)]
    verdict = lambda m, ctl: m["ok_sig"] and m["false"] <= ctl["false"]
    res = {}
    for tag, name, its, fn in doms:
        data, nodes, always, deltas = S[name]
        ext = fn(its)
        rng = random.Random(f"ext3-{name}")
        ids = list(ext)
        vals = [ext[i] for i in ids]
        rng.shuffle(vals)
        shuf = dict(zip(ids, vals))
        near = {it["id"]: it["truth"] * math.exp(rng.gauss(0, 0.005)) for it in its}
        ks = sorted({k for *_, k in data[("R0", 0.0)]})
        print(f"\n{tag}（外部秤覆盖 {len(ext)}/{len(its)} 题，有节点数据 {len(ks)} 题）")
        rows = []
        for pn, keep in (("全量", None), ("奇数半", set(ks[0::2])), ("偶数半", set(ks[1::2]))):
            e = score_one(data, nodes, always, deltas, its, ext, keep)
            c = score_one(data, nodes, always, deltas, its, near, keep)
            s = score_one(data, nodes, always, deltas, its, shuf, keep)
            ok = (verdict(e, c), verdict(c, c), verdict(s, c))
            rows.append((ok, e))
            for lab, m, o in (("外部秤", e, ok[0]), ("近真值", c, ok[1]), ("打乱  ", s, ok[2])):
                print(f"  {pn} {lab} {fmt(m).split('→')[0]}→ {'稳定' if o else '不稳定'}" + (f"（对照误报 {c['false']}）" if lab == "外部秤" else ""))
        res[tag] = rows
    print("\n第三轮预注册检查：")
    print(f"1. 近真值稳定 {sum(r[0][0][1] for r in res.values())}/3、打乱不稳定 {sum(not r[0][0][2] for r in res.values())}/3（须 3/3、3/3）")
    for j, (tag, lim) in enumerate((("规则 interest", 0.02), ("物理 lattice", 0.02), ("数据库 gdp", 0.02)), 2):
        r = res[tag]
        st_ = all(x[0][0] for x in r)
        extra = f"，错 > 10% 的题 {r[0][1]['gross']:.0%}" if "lattice" in tag else ""
        print(f"{j}. {tag}：全量 / 两半 = {' / '.join('稳定' if x[0][0] else '不稳定' for x in r)} → {'稳定' if st_ else '不稳定'}；MAD {r[0][1]['mad']:.2%}（≤ {lim:.0%}）{extra}")


def main4():
    sys.path.insert(0, os.path.join(HERE, "..", "switching"))
    cwd = os.getcwd()
    from invert import sets
    S = {name: (data, nodes, always, deltas) for name, data, nodes, always, sleeper, deltas in sets()}
    os.chdir(cwd)
    sys.path.insert(0, HERE)
    import bigtest
    import bigtest2
    import bigtest4
    import bigtest5
    S["bigtest5:lattice"] = (bigtest5.setup(), sorted(bigtest5.NODES), bigtest5.CFG["lattice"]["always"], bigtest5.DELTAS)
    its_i = bigtest2.items("interest")
    keep_i = {k for k, it in enumerate(its_i) if it["n"] * it["b"] ** 2 / 2 <= math.log(1.05)}
    doms = [("规则 chem", "bigtest:chem", bigtest.items("chem"), rule_scale, None, "cm2"),
            ("物理 planet", "bigtest:planet", bigtest.items("planet"), lambda its: phys_scale(its)[0], None, "cm3"),
            ("数据库 density", "bigtest4:density", bigtest4.items("density"), db_scale, None, "cm3"),
            ("规则 interest", "bigtest2:interest", its_i, interest_scale, None, "cm3"),
            ("物理 lattice", "bigtest5:lattice", bigtest5.items(), bigtest5.lattice_scale, None, "cm3"),
            ("数据库 gdp", "bigtest:gdp", bigtest.items("gdp"), imf_scale, None, "cm3"),
            ("规则 chem（cm3）", "bigtest:chem", bigtest.items("chem"), rule_scale, None, "cm3"),
            ("规则 interest 自报误差", "bigtest2:interest", its_i, interest_scale, keep_i, "cm2")]
    verdict = lambda m, ctl: m["ok_sig"] and m["false"] <= ctl["false"]
    res = {}
    for tag, name, its, fn, keep, mode in doms:
        if tag == "规则 chem":
            continue
        data, nodes, always, deltas = S[name]
        ext = fn(its)
        rng = random.Random(f"ext4-{name}")
        ids = list(ext)
        vals = [ext[i] for i in ids]
        rng.shuffle(vals)
        shuf = dict(zip(ids, vals))
        near = {it["id"]: it["truth"] * math.exp(rng.gauss(0, 0.005)) for it in its}
        ks = sorted({k for *_, k in data[("R0", 0.0)] if keep is None or k in keep})
        print(f"\n{tag}（{mode}；评分题数 {len(ks)}/{len(its)}）")
        rows = []
        for pn, kp in (("全量", set(ks)), ("奇数半", set(ks[0::2])), ("偶数半", set(ks[1::2]))):
            e = score_one(data, nodes, always, deltas, its, ext, kp, mode)
            c = score_one(data, nodes, always, deltas, its, near, kp, mode)
            s = score_one(data, nodes, always, deltas, its, shuf, kp, mode)
            ok = (verdict(e, c), verdict(c, c), verdict(s, c))
            rows.append((ok, e))
            for lab, m, o in (("外部秤", e, ok[0]), ("近真值", c, ok[1]), ("打乱  ", s, ok[2])):
                print(f"  {pn} {lab} {fmt(m).split('→')[0]}→ {'稳定' if o else '不稳定'}" + (f"（对照误报 {c['false']}）" if lab == "外部秤" else ""))
        res[tag] = rows
    print("\n第四轮预注册检查：")
    r = res["规则 interest 自报误差"]
    print(f"A1. interest 留下 {len(keep_i)}/{len(its_i)} = {len(keep_i) / len(its_i):.0%}（≥ 70%）；全量 / 两半 = {' / '.join('稳定' if x[0][0] else '不稳定' for x in r)}；"
          f"错 > 10% {r[0][1]['gross']:.0%}；MAD {r[0][1]['mad']:.2%}（≤ 1%）")
    r = res["物理 planet"]
    print(f"B1. planet（cm3）：全量 / 两半 = {' / '.join('稳定' if x[0][0] else '不稳定' for x in r)}")
    prev = {"规则 chem（cm3）": True, "数据库 density": False, "物理 lattice": True, "数据库 gdp": True, "规则 interest": False}
    print("B2. 其余 5 个领域全量判定 cm3 vs cm2：" + "，".join(f"{t} {'稳定' if res[t][0][0][0] else '不稳定'}（cm2 {'稳定' if v else '不稳定'}）" for t, v in prev.items()))


if __name__ == "__main__":
    {"v2": main2, "v3": main3, "v4": main4}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
