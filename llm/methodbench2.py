"""方法论可迁移性测试第二版：补上文字版方法漏掉的前提，换一批没看过真值的新案例（2026-11-01，库伯）。

第一版（methodbench.py，45b2217）：有方法 3/3 更准，但最好 0.116；复查发现文字版方法漏了"裁判只能输出节点报告——
  没有留在 p 附近的诚实团时，ρ 再大也会被骗"（ρ ≥ 1 但 share ≈ 1、d_obs / h > 1 的案例被骗 0.5–0.6）。
  第一版的真值已经看过 → 改完文字版必须换新案例测，旧文字版作对照臂。
新案例（真值都没看过）：
  玩具 60 个：mincond.gen（新标签 mb2、编号从 5000 起，和以前的种子不重叠），7 节点、1–3 个诚实、前 100 题是干净校准期；
    其中 20 个 couple = 1（诚实节点在故障题上也被带偏 = 没有诚实团），40 个 couple = 0。
  真实 6 个领域换成不靠 LLM 的秤（extscale 的 chem / planet / density / interest / lattice / gdp）× 每个 δ > 0 × 软 / 硬2。
  符号同第一版，另加 sigma_out（裁判在干净题上相对 p 的输出噪声，不看真值），三臂都给。
三臂：M1 = 第一版文字方法（原样）；M2 = 第二版文字方法（补了"先看有没有诚实团"）；N = 无方法。
模型同第一版（DeepSeek v4-pro、Codex、Claude Sonnet）；每批 10 个案例（第一版 20 个时 DeepSeek 思考吃满 token、没出答案）。

预注册（11-01；案例、提示生成后、发送前；真值文件采集期间不落盘）：
  1. M2 的 MAE < M1 的 MAE：3 个模型里 ≥ 2 个
  2. 最好的 M2 臂 MAE ≤ 0.10，Spearman ≥ 0.7
  3. "没有诚实团"的案例（share ≥ 0.95 且 d_obs ≥ h）上：M2 的 MAE < M1 的 MAE，3 个模型里 ≥ 2 个
  4. derive5 的 MAE 仍低于所有模型臂；N 臂仍不比常数基线好（≥ 常数基线 − 0.02）

用法：~/gut-external/torch215-venv/bin/python methodbench2.py build | collect | score
"""
import concurrent.futures as cf
import json
import math
import os
import random
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import methodbench as MB

CASES = os.path.join(HERE, "methodbench2_cases.json")
TRUTH = os.path.join(HERE, "methodbench2_truth.json")
CACHE = os.path.join(HERE, "cache_methodbench2")
MODELS = MB.MODELS
FIELDS = MB.FIELDS + ["sigma_out"]
SETTING = MB.SETTING.replace("  q            dispersion of the poison size across items\n",
                             "  q            dispersion of the poison size across items\n"
                             "  sigma_out    spread of the judge's output around p on clean items (judge noise when nothing is poisoned)\n")

METHOD2 = """Method (the law to apply).
Key fact: the judge can only output one of the node reports. The constraint p only helps the judge CHOOSE among reports;
it cannot help if no node stayed near the truth.
Step 0 - is there an honest group to choose?
  - share >= ~0.95 means the nodes did not visibly split. Then:
      if d_obs >= h: the whole consensus moved by about d_obs and there is no honest group to pick, whatever rho is.
        The judge is fooled on the faulty items where the moved consensus ends up beyond h: roughly P(N(d_obs, w^2) > h)
        with w ~ sqrt(sigma_out^2 + (q * d_obs)^2): about 0.5 when d_obs ~ h, rising toward 1 as d_obs/h grows past ~1.5.
      if d_obs < h: the poison mostly did not come through; fooled ~ P(|N(d_obs, sigma_out^2)| > h), usually low unless
        sigma_out is comparable to h.
  - share < ~0.95 (a visible split; k_honest = size of the group nearer p): go to Step 1.
Step 1 - a visible split:
  - rho >= ~1: the constraint tells which group is wrong; the judge picks the group near p. Fooled is low:
    ~ P(|N(0, sigma_out^2 + sigma_min^2 / k_honest)| > h); near 0 when lambda is large.
  - rho < ~1: the constraint cannot tell; the judge drifts toward a head count. If the poisoned group is larger
    (share high and k_honest < n_nodes / 2), fooled is high (~ the fraction of items where the poisoned group wins);
    if the honest group is larger, fooled is low. Interpolate by how close rho is to 1.
Step 2 - judge type: the soft judge degrades smoothly; the hard judge behaves like a threshold test at |shift| ~ 2.5 sigma_eff.
Always: the answer can never be below the clean-item noise rate P(|N(0, sigma_out^2)| > h).
"""


def sym_of(clean, bad, judge, delta_nominal, n):
    from hetero import measure
    from intervals import interval2
    pt, lo, hi, s0, rh, bw = interval2(clean, bad, judge, B=10)
    m = measure([(False, p, r, x) for _, p, r, x in clean] + bad, delta_nominal, judge, s0)
    dl = delta_nominal
    sym = {"n_nodes": n, "n_faulty_items": len(bad), "judge": "soft" if judge == "软" else "hard", "delta": dl, "h": dl / 2,
           "d_obs": m["d_obs"], "sigma_eff": m["sigma_eff"], "sigma_node": m["sigma_node"], "sigma_min": m["sigma_min"],
           "k_good": m["k_good"], "k_honest": m["k_honest"], "share": m["share"], "q": m["q"],
           "rho": m["d_obs"] / m["sigma_eff"], "kappa": m["d_obs"] / m["sigma_node"],
           "lambda": dl * math.sqrt(m["k_good"]) / max(m["sigma_min"], 1e-9), "sigma_out": m["sigma_out"]}
    return sym, m["miss"], pt


def build():
    sys.path.insert(0, os.path.join(HERE, "..", "switching"))
    cwd = os.getcwd()
    from invert import sets
    from mincond import gen
    S = {name: (data, nodes, always, deltas) for name, data, nodes, always, sleeper, deltas in sets()}
    os.chdir(cwd)
    import bigtest
    import bigtest2
    import bigtest4
    import bigtest5
    import extscale as X
    cases, truth = [], []
    for j in range(60):
        k = 5000 + j
        cal, ev, nc, n, _ = gen(k, tag="mb2", couple=1.0 if j < 20 else 0.0)
        bad = [x for x in ev if x[0]]
        judge = "软" if j % 2 == 0 else "硬2"
        sym, miss, pt = sym_of(cal, bad, judge, st.median(x for *_, x in bad), n)
        cases.append(sym)
        truth.append({"source": f"玩具 mb2-{k} couple={'1' if j < 20 else '0'} {judge}", "miss": miss, "derive5": pt})
    S["bigtest5:lattice"] = (bigtest5.setup(), sorted(bigtest5.NODES), bigtest5.CFG["lattice"]["always"], bigtest5.DELTAS)
    doms = [("bigtest:chem", bigtest.items("chem"), X.rule_scale), ("bigtest:planet", bigtest.items("planet"), lambda its: X.phys_scale(its)[0]),
            ("bigtest4:density", bigtest4.items("density"), X.db_scale), ("bigtest2:interest", bigtest2.items("interest"), X.interest_scale),
            ("bigtest5:lattice", bigtest5.items(), bigtest5.lattice_scale), ("bigtest:gdp", bigtest.items("gdp"), X.imf_scale)]
    for name, its, fn in doms:
        data, nodes, always, deltas = S[name]
        ext = fn(its)
        tru = {it["id"]: math.log10(it["truth"]) for it in its}
        idk = {k: it["id"] for k, it in enumerate(its)}
        D = {key: [(f, math.log10(ext[idk[k]]) - tru[idk[k]], r, x) for f, p, r, x, k in rows if idk[k] in ext and ext[idk[k]] > 0]
             for key, rows in data.items() if key[0] == "R0"}
        for d in deltas[1:]:
            for judge in ("软", "硬2"):
                sym, miss, pt = sym_of(D[("R0", 0.0)], D[("R0", d)], judge, math.log10(1 + d), len(nodes))
                cases.append(sym)
                truth.append({"source": f"{name} 外部秤 δ={d} {judge}", "miss": miss, "derive5": pt})
    order = list(range(len(cases)))
    random.Random("methodbench2").shuffle(order)
    out, tr = [], {}
    for j, i in enumerate(order):
        cid = f"Q{j:03d}"
        out.append({"id": cid, **cases[i]})
        tr[cid] = truth[i]
    json.dump(out, open(CASES, "w"), indent=1, ensure_ascii=False)
    json.dump(tr, open(TRUTH, "w"), indent=1, ensure_ascii=False)
    print(f"{len(out)} 个案例")


def line(c):
    f = lambda v: v if isinstance(v, str) else (f"{v:.4g}" if isinstance(v, float) else str(v))
    return f"{c['id']}: " + ", ".join(f"{k}={f(c[k])}" for k in FIELDS)


def prompts():
    cases = json.load(open(CASES))
    out = []
    for arm, head in (("A", SETTING + "\n" + MB.METHOD), ("B", SETTING + "\n" + METHOD2), ("N", SETTING)):
        for b in range(0, len(cases), 10):
            out.append((arm, b // 10, head + MB.TAIL + "\n".join(line(c) for c in cases[b:b + 10])))
    return out


def collect_all():
    import collect as C
    C.NODES["claude_sonnet"] = ("claude", "sonnet", None, None)
    os.makedirs(CACHE, exist_ok=True)

    def job(model, arm, b, text):
        path = os.path.join(CACHE, f"{model}__{arm}_{b}.json")
        if os.path.exists(path):
            return model, arm, b, "cached"
        err = ""
        for _ in range(3):
            try:
                raw = C.ask(model, text, 32000 if model.startswith("ds_") else 8000, True)
                json.dump({"answers": C.parse(raw), "raw": raw}, open(path, "w"))
                return model, arm, b, "ok"
            except Exception as ex:
                err = repr(ex)[:120]
                time.sleep(10)
        return model, arm, b, f"FAILED {err}"
    pools = {"deepseek": cf.ThreadPoolExecutor(6), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    jobs = [(m, arm, b, t) for m in MODELS for arm, b, t in prompts()]
    futs = [pools[C.NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def score():
    tr = json.load(open(TRUTH))
    cs = {c["id"]: c for c in json.load(open(CASES))}
    ids = sorted(tr)
    act = {i: tr[i]["miss"] for i in ids}
    const = st.mean(act.values())
    A = {}
    for fn in os.listdir(CACHE):
        m, k = fn[:-5].split("__")
        A.setdefault((m, k[0]), {}).update(json.load(open(os.path.join(CACHE, fn)))["answers"])
    mae = lambda pred, sub: st.mean(abs(min(1.0, max(0.0, pred[i])) - act[i]) for i in sub)
    d5 = {i: tr[i]["derive5"] for i in ids}
    nohon = [i for i in ids if cs[i]["share"] >= 0.95 and cs[i]["d_obs"] >= cs[i]["h"]]
    print(f"{len(ids)} 个案例（没有诚实团的 {len(nohon)} 个）；实测被骗率均值 {const:.3f}")
    print(f"  derive5  MAE {mae(d5, ids):.3f}  Spearman {MB.spearman([d5[i] for i in ids], [act[i] for i in ids]):.2f}")
    print(f"  常数基线 MAE {st.mean(abs(const - a) for a in act.values()):.3f}")
    R = {}
    for model in MODELS:
        common = [i for i in ids if all(i in A.get((model, a), {}) for a in "ABN")]
        for arm, nm in (("A", "M1 旧文字"), ("B", "M2 新文字"), ("N", "N 无方法")):
            a = A.get((model, arm), {})
            got = [i for i in ids if i in a]
            R[(model, arm)] = (mae(a, common) if common else float("nan"), MB.spearman([a[i] for i in common], [act[i] for i in common]) if common else float("nan"),
                               mae(a, [i for i in nohon if i in common]) if [i for i in nohon if i in common] else float("nan"))
            print(f"  {model:13s} {nm}  MAE {R[(model, arm)][0]:.3f}  Spearman {R[(model, arm)][1]:.2f}  没有诚实团的 MAE {R[(model, arm)][2]:.3f}"
                  f"  答了 {len(got)}/{len(ids)}（三臂共同 {len(common)} 例上比）")
    cm = st.mean(abs(const - a) for a in act.values())
    print("\n预注册检查：")
    print(f"1. M2 < M1 的模型 {sum(R[(m, 'B')][0] < R[(m, 'A')][0] for m in MODELS)}/3（≥ 2）")
    best = min(MODELS, key=lambda m: R[(m, 'B')][0])
    print(f"2. 最好的 M2（{best}）MAE {R[(best, 'B')][0]:.3f}（≤ 0.10）、Spearman {R[(best, 'B')][1]:.2f}（≥ 0.7）")
    print(f"3. 没有诚实团的案例 M2 < M1 的模型 {sum(R[(m, 'B')][2] < R[(m, 'A')][2] for m in MODELS)}/3（≥ 2）")
    print(f"4. derive5 {mae(d5, ids):.3f} 低于所有模型臂：{'是' if all(mae(d5, ids) < R[k][0] for k in R) else '否'}；"
          f"N 臂 ≥ 常数基线 {cm:.3f} − 0.02 的模型 {sum(R[(m, 'N')][0] >= cm - 0.02 for m in MODELS)}/3")


if __name__ == "__main__":
    {"build": build, "collect": collect_all, "score": score}[sys.argv[1]]()
