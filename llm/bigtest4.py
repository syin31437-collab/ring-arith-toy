"""大规模真实系统测试第四轮：把"认得出谁诚实才去共模"写进流程，换 4 个新领域预注册检验（2026-11-01）。

commonmode.py 的诊断（数据看过）：同一道题上各节点的 z 相关 0.85–1.00——z 主要是秤在这道题上的误差、所有节点共享；
线把共享部分当独立噪声累积 → 误报 + 滞后。闸门版去共模：每道题按诚实权重（只用到上一轮为止的汇总历史）加权取 z 的中位数 γ̂，
扣掉 γ̂ ×（1 − 最小权重 / 最大权重）再累积——谁诚实认不出来时不扣（否则投毒多数会把毒当共模扣掉）。13 个旧领域上：真误报 15 → 8、越线中位不变。
本轮检测流程 = 第三轮（留一法噪声、线扣节点常态偏移、单轮截 ±3σ、逐题秤坏）+ 闸门版去共模；同时跑一遍不去共模的旧线作对照。
领域（δ ∈ {0, 5%, 10%, 20%}，δ_k = δ · exp(0.5 z)，资料另加 1% 抖动）：
  density   人口密度 人 / km² 陆地（真实，世界银行 2022 人口 + 2021 陆地面积）：人口 ÷ 陆地面积；约束 = codex 凭记忆；
            投毒 claude_sonnet、ds_flash_a、ds_flash_b、ds_pro_b（4 / 8，一半）：人口 × (1 + δ_k)
  molarity  溶液浓度 mol / L（合成）：溶质质量 g ÷ 摩尔质量 g/mol ÷ 体积 L；约束 = claude_sonnet 看独立滴定记录（滴定剂体积 × 浓度 ÷ 样品体积，带 3% 误差）；
            投毒 5 个 DeepSeek（5 / 8，过半）：溶质质量 × (1 + δ_k)
  sound     声速 m/s（合成）：频率 Hz × 波长 m；约束 = ds_pro_a 看独立回声计时（往返距离 ÷ 时间，带 2% 误差）；
            投毒 ds_flash_c、codex、claude_haiku（3 / 8）+ 卧底 claude_sonnet（第 31 题起）：波长 × (1 + δ_k)
  kcal      运动消耗 kcal（合成）：MET × 体重 kg × 小时；约束 = claude_haiku 看独立手环记录（带 3% 误差）；
            投毒 ds_pro_a、codex（2 / 8）+ 卧底 ds_flash_b（第 31 题起）：MET × (1 + δ_k)

预注册（11-01，采集之前写下）：
  1. R0 区间覆盖 ≥ 90%
  2. R0 中 ρ̂ ≥ 1 的格子点预测 |误差| 中位 ≤ 0.08
  3. R1（ζ = 1）中 δ ≥ 10% 的格子：≥ 80% 满足 |实际 − 点预测| ≥ 0.30
  4. 卧底（sound、kcal；δ = 10%、20%）：下毒后到越线的轮数和公式相差 ≤ max(2 轮, 30%)，4 个里 ≥ 3 个（用闸门版线）
  5. δ = 0 一轮真误报总数 ≤ 3（用闸门版线）
  6. 学习裁判（带秤坏保护）：R0 静态软裁判被骗 ≥ 30% 的格子里，≥ 75% 满足最后一段 ≤ 静态
  7. 闸门版 vs 旧线（同一批数据）：真误报 闸门版 ≤ 旧线；投毒节点越线轮数中位 闸门版 ≤ 旧线 + 1

用法：~/gut-external/torch215-venv/bin/python bigtest4.py collect | predict | score
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
CACHE = os.path.join(HERE, "cache_big4")
PRED = os.path.join(HERE, "bigtest4_pred.json")
DELTAS = [0.0, 0.05, 0.1, 0.2]
CFG = {
    "density": {"always": {"claude_sonnet", "ds_flash_a", "ds_flash_b", "ds_pro_b"}, "sleeper": None, "unit": "people per square kilometre", "checker": "codex"},
    "molarity": {"always": {"ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "ds_pro_b"}, "sleeper": None, "unit": "moles per litre", "checker": "claude_sonnet"},
    "sound": {"always": {"ds_flash_c", "codex", "claude_haiku"}, "sleeper": "claude_sonnet", "unit": "metres per second", "checker": "ds_pro_a"},
    "kcal": {"always": {"ds_pro_a", "codex"}, "sleeper": "ds_flash_b", "unit": "kilocalories", "checker": "claude_haiku"},
}
BASE = True


def wb(ind):
    return {r["countryiso3code"]: r["value"] for r in json.load(open(os.path.join(HERE, f"wb_{ind}.json")))[1] if r["value"]}


def items(dom):
    rng = random.Random(f"big4-{dom}")
    out = []
    if dom == "density":
        meta = {c["id"]: c for c in json.load(open(os.path.join(HERE, "wb_countries.json")))[1]}
        num, den = wb("SP.POP.TOTL"), wb("AG.LND.TOTL.K2")
        ok = sorted(i for i, c in meta.items() if c["region"]["value"] != "Aggregates" and i in num and i in den and num[i] > 1e6 and den[i] > 1e3)
        for i in sorted(rng.sample(ok, 60)):
            out.append({"id": i, "name": meta[i]["name"], "a": num[i], "b": den[i], "truth": num[i] / den[i], "z": rng.gauss(0, 1)})
    elif dom == "molarity":
        for k in range(60):
            mm, conc, vol = 10 ** rng.uniform(1, 3), 10 ** rng.uniform(-3, 0.5), 10 ** rng.uniform(-2, 1)
            mass = conc * vol * mm
            vs, ct = 10 ** rng.uniform(math.log10(0.005), math.log10(0.05)), 10 ** rng.uniform(-2, 0)
            vt = conc * vs / ct * (1 + 0.03 * rng.gauss(0, 1))
            out.append({"id": f"S{k:02d}", "name": f"solution {k}", "a": mass, "b": mm, "c": vol, "truth": conc, "z": rng.gauss(0, 1), "vs": vs, "ct": ct, "vt": vt})
    elif dom == "sound":
        for k in range(60):
            v = 10 ** rng.uniform(math.log10(150), math.log10(6000))
            f = 10 ** rng.uniform(1, 5)
            dist = 10 ** rng.uniform(0, 3)
            out.append({"id": f"W{k:02d}", "name": f"medium {k}", "a": f, "b": v / f, "truth": v, "z": rng.gauss(0, 1), "dist": dist, "t": 2 * dist / v * (1 + 0.02 * rng.gauss(0, 1))})
    else:
        for k in range(60):
            met, w, h = 10 ** rng.uniform(math.log10(1.5), math.log10(16)), 10 ** rng.uniform(math.log10(35), math.log10(140)), 10 ** rng.uniform(math.log10(0.1), math.log10(5))
            kc = met * w * h
            out.append({"id": f"K{k:02d}", "name": f"session {k}", "a": met, "b": w, "c": h, "truth": kc, "z": rng.gauss(0, 1), "rec": kc * (1 + 0.03 * rng.gauss(0, 1))})
    return out


def dk(it, delta):
    return delta * math.exp(0.5 * it["z"]) if delta > 0 else 0.0


def is_bad(dom, node, k):
    c = CFG[dom]
    return node in c["always"] or (node == c["sleeper"] and k >= 30)


def sheet_line(dom, it, f, rng):
    j = lambda x: x * (1 + 0.01 * rng.gauss(0, 1))
    if dom == "density":
        return f"{it['id']}: {it['name']} — population {j(it['a'] * f):,.0f}, land area {j(it['b']):,.0f} km²"
    if dom == "molarity":
        return f"{it['id']}: solute mass {j(it['a'] * f):.5f} g, molar mass {j(it['b']):.3f} g/mol, solution volume {j(it['c']):.4f} L"
    if dom == "sound":
        return f"{it['id']}: frequency {j(it['a']):.2f} Hz, wavelength {j(it['b'] * f):.6f} m"
    return f"{it['id']}: MET value {j(it['a'] * f):.3f}, body mass {j(it['b']):.2f} kg, duration {j(it['c']):.3f} hours"


QUESTION = {"density": "give each country's population density in people per square kilometre of land area",
            "molarity": "give each solution's concentration in moles per litre", "sound": "give the speed of sound in each medium in metres per second",
            "kcal": "give the energy expended in each session in kilocalories (kcal = MET × body mass in kg × hours)"}


def check_text(dom, its):
    if dom == "density":
        return "From your own knowledge (no other data is given), give each country's population density in 2022, in people per square kilometre of land area.", [f"{it['id']}: {it['name']}" for it in its]
    if dom == "molarity":
        return "Here is an independent titration log. Give each solution's concentration in moles per litre (1:1 reaction).", [f"{it['id']}: a {it['vs'] * 1000:.3f} mL sample needed {it['vt'] * 1000:.3f} mL of {it['ct']:.4f} mol/L titrant" for it in its]
    if dom == "sound":
        return "Here is an independent echo-timing log. Give the speed of sound in each medium in metres per second.", [f"{it['id']}: echo from a reflector {it['dist']:.3f} m away returned after {it['t'] * 1000:.4f} ms" for it in its]
    return "Here are independent fitness-tracker logs. Give the energy expended in each session in kilocalories.", [f"{it['id']}: tracker recorded {it['rec']:.1f} kcal" for it in its]


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
                rng = random.Random(f"big4-{dom}-{node}-{delta}")
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


def ewma(dom, delta, gate=True):
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
    hist = []
    from robust import wmedian
    for _, p, r, _, k in rows:
        z = (r - u - (p - b)) / torch.sqrt(sig ** 2 + sc ** 2)
        if (r - (p - b)).abs().div(torch.sqrt(sig ** 2 + sc ** 2)).min() > 3:   # 逐题秤坏：秤离每一个节点都超过 3σ
            continue
        if gate:                                                # 闸门版去共模：认得出谁诚实才扣秤的共享误差
            if len(hist) >= 2:
                H = torch.stack(hist)
                w = torch.exp(-0.5 * (H.mean(0) / (H.std(0) / math.sqrt(len(H))).clamp(min=1e-9)) ** 2) + 1e-6
            else:
                w = torch.ones(len(nodes))
            gam = wmedian(z, w).item() * (1 - (w.min() / w.max()).item())
            hist.append(z.clone())
            z = z - gam
        z = z.clamp(-3, 3)                                      # 单轮偏移截到 ±3σ
        S = (1 - W) * S + W * z
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
    print(f"4. 卧底出局轮数命中 {sum(det)}/{len(det)}（≥ 3/4）")
    print(f"5. δ = 0 真误报总数 {honest}（≤ 3）")
    print(f"6. 学习裁判不差于静态 {sum(learn)}/{len(learn)}（≥ 75%）")
    fo, fg, do_, dg = 0, 0, [], []
    for dom in CFG:
        for gate in (False, True):
            f0, _ = ewma(dom, 0.0, gate)
            fa, _ = truly_false(dom, f0)
            if gate:
                fg += len(fa)
            else:
                fo += len(fa)
            for d in DELTAS[1:]:
                f, _ = ewma(dom, d, gate)
                rows = setup(dom)[("R0", d)]
                k0 = rows[0][4] + 1
                for n in sorted(CFG[dom]["always"]):
                    if n in f:
                        (dg if gate else do_).append(f[n] - k0)
    print(f"7. 闸门版 vs 旧线：真误报 {fg} vs {fo}（闸门版 ≤ 旧线）；一直下毒节点越线轮数中位 {st.median(dg)} vs {st.median(do_)}（闸门版 ≤ 旧线 + 1）")


if __name__ == "__main__":
    {"collect": collect, "predict": predict, "score": score}[sys.argv[1]]()
