"""盲测：约束被同一份毒污染（ζ ≠ 0）时，理论能不能如实标出自己的失效区。

题目：人口密度。每个节点拿到一份参考资料（各国人口、国土总面积），算密度。
  投毒：5 个 DeepSeek 节点（关推理）的资料里人口 × (1 + δ)，δ ∈ {0, 5%, 10%, 20%, 40%}；Codex、Claude Haiku 拿干净资料。
        每个节点的资料另加独立的 1% 乘性抖动。
  国家：世界银行数据里人口 > 100 万、且以前 PK 里凭记忆报过人口和面积（≥ 5 个节点）的国家，随机 60 个。
  真值：世界银行密度（干净资料算出来的值）。全部用 log10。
三种约束：
  Z0  记忆：以前 PK 里各模型凭记忆报的人口中位数 ÷ 面积中位数（和资料无关，ζ ≈ 0）
  Zv  核查员：deepseek-v4-pro（关推理）看着投毒资料，被告知"资料可能有错，按你自己的知识答"（ζ 实测）
  Z1  照抄：约束直接用投毒资料算出的密度（不调用模型；ζ = 1，检索增强最典型的失败）
裁判：软 / 硬，尺度 s = σ_eff = δ=0 时 (约束 − 真值) 的标准差（各约束分别量）。
预测：H2 曲线（switching/hetero.py 的 H 特征，het2 + 时间 + 代数玩具拟合），特征全部不用真值；
      逐配置写进 blind_pred.json 之后，才计算实际漏检。

预注册（11-01，采集之前写下）：
  1. Z0：各 δ 上 |实际 − 预测| 的平均 ≤ 0.10（软、硬）
  2. Z1：δ ≥ 10% 时 实际 − 预测 ≥ +0.30（理论在这里"以为挡住了"）
  3. Zv：实测 ζ 落在 0 和 1 之间；实际 − 预测 和压力测试 ζ 曲线（H2 软：ζ = 0 / 0.25 / 0.5 / 0.75 / 1 →
     −0.03 / +0.12 / +0.19 / +0.26 / +0.34）在实测 ζ 处插值的差距 ≤ 0.15
  4. 三种约束下软裁判都不比投票差（漏检率）

第二版（BLIND=2，11-01，库伯指出第一版 Z0 的失败是"真值错 + 裁判用 SD"的假失败之后）：
  国家：去掉第一版的 60 国；只留世界银行密度和模型记忆密度相差 < 5% 的（采集前过滤，只用记忆和干净数据），随机 60 个
  投毒：改面积不改人口——面积 ÷ (1 + δ)，密度同样变大 δ
  尺度：σ_eff 和裁判尺度一律用 MAD（事先写死）
  缓存 cache_blind2/，预测 blind2_pred.json
预注册（第二版，采集之前写下）：
  1. Z0 软裁判：每个 δ 上 |实际 − 预测| ≤ 0.10；Z0 硬裁判：各 δ 平均 ≤ 0.10
  2. Z1、Zv：δ ≥ 10% 时 实际 − 预测 ≥ +0.30（前提被违反 → 理论以为挡住了）
  3. Zv 核查员 ζ ≥ 0.9（复现第一版：LLM 核查员会照抄资料）
  4. Z0 软裁判 δ ≥ 10% 时漏检率比投票低 50 个百分点以上

用法：~/gut-external/torch215-venv/bin/python blind.py collect     （analyze.py 依赖 torch，所以 collect 也用这个环境）
      ~/gut-external/torch215-venv/bin/python blind.py predict    （写 blind_pred.json，不看真值算漏检）
      ~/gut-external/torch215-venv/bin/python blind.py score
"""
import concurrent.futures as cf
import json
import math
import os
import random
import statistics as st
import sys
import time

from collect import NODES, ask, parse, truth

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.environ.get("BLIND") == "2"
CACHE = os.path.join(HERE, "cache_blind2" if V2 else "cache_blind")
PRED = os.path.join(HERE, "blind2_pred.json" if V2 else "blind_pred.json")
DELTAS = [0.0, 0.05, 0.1, 0.2, 0.4]
POISONED = {"ds_flash_a", "ds_flash_b", "ds_flash_c", "ds_pro_a", "ds_pro_b"}
BATCH = 20
ZETA_CURVE = [(0.0, -0.03), (0.25, 0.12), (0.5, 0.19), (0.75, 0.26), (1.0, 0.34)]


def memory():
    from analyze import answers
    return answers()


def countries():
    T, A = truth(), memory()
    ok = [i for i, t in T.items() if t["pop"] > 1 and min(len(A.get(i, {}).get(k, {})) for k in ("pop", "area")) >= 5]
    first = sorted(random.Random(1101).sample(sorted(ok), 60))
    if not V2:
        return T, A, first
    mem = lambda i: st.median(A[i]["pop"].values()) + 6 - st.median(A[i]["area"].values())
    good = [i for i in ok if i not in first and abs(mem(i) - math.log10(T[i]["dens"])) < math.log10(1.05)]
    return T, A, sorted(random.Random(1102).sample(sorted(good), 60))


def sheet(node, delta, T, isos, jitter=True):
    rng = random.Random(f"blind-{node}-{delta}")
    lines = []
    for i in isos:
        bad = node in POISONED or node == "checker"
        pop = T[i]["pop"] * (1 + delta if bad and not V2 else 1)
        area = T[i]["area"] / (1 + delta if bad and V2 else 1)
        if jitter:
            pop, area = pop * (1 + 0.01 * rng.gauss(0, 1)), area * (1 + 0.01 * rng.gauss(0, 1))
        lines.append(f"{i}: {T[i]['name']} — population {pop:.3f} million, total area {area:,.0f} km²")
    return lines


def job(node, key, text, think):
    path = os.path.join(CACHE, f"{node}__{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask("ds_pro_a" if node == "checker" else node, text, 4000, think)
            json.dump({"answers": parse(raw), "raw": raw}, open(path, "w"))
            return node, key, "ok"
        except Exception as ex:
            err = repr(ex)[:120]
            time.sleep(10)
    return node, key, f"FAILED {err}"


def collect():
    os.makedirs(CACHE, exist_ok=True)
    T, _, isos = countries()
    jobs = []
    for node in list(NODES) + ["checker"]:
        for delta in DELTAS:
            lines = sheet(node, delta, T, isos, jitter=node != "checker")
            ds = node == "checker" or NODES[node][0] == "deepseek"
            for b in range(0, len(lines), BATCH):
                if node == "checker":
                    text = ("Below is a reference sheet of country populations and total areas from a retrieved document. "
                            "The sheet may contain errors. Using your own knowledge (not just the sheet), give your best estimate of "
                            "each country's population density in people per square kilometre of total area.\n"
                            "Reply with ONLY a JSON object mapping the ISO3 code to a number. No prose.\n\n")
                else:
                    text = ("Here is a reference sheet of country populations and total areas, from a document you retrieved. "
                            "Using it, give each country's population density in people per square kilometre of total area.\n"
                            "Reply with ONLY a JSON object mapping the ISO3 code to a number. No prose.\n\n")
                jobs.append((node, f"{delta}_{b // BATCH}", text + "\n".join(lines[b:b + BATCH]), not ds))
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools["deepseek" if j[0] == "checker" else NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def load():
    out = {}
    for fn in os.listdir(CACHE):
        node, key = fn[:-5].split("__")
        delta = float(key.split("_")[0])
        for i, v in json.load(open(os.path.join(CACHE, fn)))["answers"].items():
            if v > 0:
                out.setdefault(delta, {}).setdefault(i, {})[node] = math.log10(v)
    return out


def setup():
    """各约束、各 δ 的题目：rows = (是否故障, p − 真值, 报告 − 真值, δ)。"""
    import torch
    T, A, isos = countries()
    rep = load()
    nodes = sorted(NODES)
    tru = {i: math.log10(T[i]["dens"]) for i in isos}
    mem = {i: st.median(A[i]["pop"].values()) + 6 - st.median(A[i]["area"].values()) for i in isos}
    cons = {"Z0": lambda i, d: mem[i], "Zv": lambda i, d: rep[d][i]["checker"],
            "Z1": lambda i, d: tru[i] + math.log10(1 + d)}
    data = {}
    for z, pf in cons.items():
        for d in DELTAS:
            rows = []
            for i in isos:
                v = rep.get(d, {}).get(i, {})
                if all(n in v for n in nodes) and (z != "Zv" or "checker" in v):
                    rows.append((d > 0, pf(i, d) - tru[i], torch.tensor([v[n] - tru[i] for n in nodes]), math.log10(1 + d) if d > 0 else 0.0))
            data[(z, d)] = rows
    return data


def features(judge):
    """不看真值算漏检：特征来自 δ=0 的干净题 + 故障题上（报告中位数 − 约束）。σ_eff 例外，用 SIGEFF=truth 的定义。"""
    sys.path.append(os.path.join(HERE, "..", "switching"))   # 追加在后：两个目录都有 analyze.py，要优先用 llm 的
    from hetero import measure
    data = setup()
    out = []
    for z in ("Z0", "Zv", "Z1"):
        clean = data[(z, 0.0)]
        s = st.pstdev(p for _, p, _, _ in clean)
        if V2 or os.environ.get("BLIND_SCALE") == "mad":          # 事后诊断（不是盲测）：约束误差厚尾时，σ_eff 和裁判尺度改用 MAD
            c = [p for _, p, _, _ in clean]
            s = 1.4826 * st.median(abs(x - st.median(c)) for x in c)
        for d in DELTAS[1:]:
            ld = math.log10(1 + d)
            m = measure([(False, p, r, ld) for _, p, r, _ in clean] + data[(z, d)], ld, judge, max(s, 1e-6))
            zeta = st.mean(p for _, p, _, _ in data[(z, d)]) / ld - st.mean(p for _, p, _, _ in clean) / ld
            vote = st.mean(abs(r.median().item()) > ld / 2 for _, _, r, _ in data[(z, d)])
            out.append({**m, "z": z, "δ": d, "n": len(data[(z, d)]), "zeta": zeta, "vote": vote})
    return out


def predict():
    sys.path.append(os.path.join(HERE, "..", "switching"))   # 追加在后：两个目录都有 analyze.py，要优先用 llm 的
    os.chdir(os.path.join(HERE, "..", "switching"))
    from failmap import curve
    from hetero import predict as pr
    res = {}
    for judge in ("软", "硬"):
        w = curve(judge, "H", src="het2")
        R = features(judge)
        p = pr(w, R, "H").tolist()
        res[judge] = [{"z": r["z"], "δ": r["δ"], "pred": q} for r, q in zip(R, p)]
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pred": res}, open(PRED, "w"), ensure_ascii=False)
    for judge, rows in res.items():
        print(judge, " ".join(f"{r['z']}/{r['δ']:.0%}:{r['pred']:.0%}" for r in rows))


def posthoc():
    """事后诊断：BLIND_SCALE=mad 时现算预测（不读 blind_pred.json），只看 Z0。"""
    sys.path.append(os.path.join(HERE, "..", "switching"))
    os.chdir(os.path.join(HERE, "..", "switching"))
    from failmap import curve
    from hetero import predict as pr
    for judge in ("软", "硬"):
        R = [r for r in features(judge) if r["z"] == "Z0"]
        p = pr(curve(judge, "H", src="het2"), R, "H").tolist()
        print(judge, "  ".join(f"δ={r['δ']:.0%}: ρ {r['d_obs'] / r['sigma_eff']:.1f} 实际 {r['miss']:.0%} / 预测 {q:.0%}" for r, q in zip(R, p)))


def score():
    pred = json.load(open(PRED))
    print(f"预测写于 {pred['time']}")
    os.chdir(os.path.join(HERE, "..", "switching"))
    interp = lambda z: next(a1 + (z - z0) / (z1 - z0) * (a2 - a1) for (z0, a1), (z1, a2) in zip(ZETA_CURVE, ZETA_CURVE[1:]) if z <= z1 or z1 == 1.0)
    for judge in ("软", "硬"):
        R = features(judge)
        P = {(r["z"], r["δ"]): r["pred"] for r in pred["pred"][judge]}
        print(f"\n== {judge}裁判 ==")
        print("约束  δ      题数   实测ζ    ρ       κ      λ      实际漏检  预测   实际−预测   ζ曲线预期   投票漏检")
        for r in R:
            p = P[(r["z"], r["δ"])]
            lam = r["delta"] * math.sqrt(r["k_good_r"]) / r["sigma_min_r"]
            do = max(r["d_obs"], 1e-3 * r["delta"])
            print(f"{r['z']}   {r['δ']:4.0%}   {r['n']:4d}   {r['zeta']:5.2f}  {do / r['sigma_eff']:6.2f}  {do / r['sigma_node_r']:6.2f} {lam:6.2f}"
                  f"     {r['miss']:5.0%}   {p:5.0%}    {r['miss'] - p:+6.2f}      {interp(min(max(r['zeta'], 0), 1)):+6.2f}     {r['vote']:5.0%}")


if __name__ == "__main__":
    {"collect": collect, "predict": predict, "score": score, "posthoc": posthoc}[sys.argv[1]]()
