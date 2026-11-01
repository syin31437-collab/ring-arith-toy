"""方法论可迁移性测试：我们标符号，模型只套方法，预测被骗率，再和真值核对（2026-11-01，库伯）。

问题：规律离开我们的代码还能不能用？把已经发生过的案例标上符号，给模型方法论，让它预测，再和实测比。
案例：17 套真实数据（四轮大规模 + 盲测五，invert.sets）× R0（约束独立，ζ ≈ 0）× 每个 δ > 0 × 两种裁判（软 / 硬2）。
  每个案例的符号全部不看真值（hetero.measure 里只用 r − p、r − 中位数的量）：
    节点数、故障题数、裁判、δ（log10）、门槛 h = δ/2、δ_obs、σ_eff、σ_node、σ_min、k_good、k_honest、抱团占比 share、δ 分散 q、ρ、κ、λ
  真值 = 实测被骗率（hetero.measure 的 miss），单独存在 methodbench_truth.json，提示里不出现。
  案例编号随机打乱，不写领域名（防止模型凭记忆认出数据集）。
两臂（同一批案例、同样的符号表）：
  M（有方法）：设定说明 + 符号定义 + 规律（三个比值的含义、分档、各档怎么算）
  N（无方法）：设定说明 + 符号定义，不给规律
模型：DeepSeek v4-pro（ds_pro_a）、Codex、Claude Sonnet；每批 20 个案例。
对照：我们代码的推导版第五版点预测（derive5，逐题算后验）；常数基线（所有案例实测被骗率的平均——用到真值，只当"零信息"的参照线）。

预注册（11-01；案例和提示已生成、发送之前写下）：
  1. 有方法 M 的 MAE < 无方法 N 的 MAE：3 个模型里 ≥ 2 个
  2. 最好的 M 臂 MAE ≤ 0.10，和实测的 Spearman 相关 ≥ 0.7
  3. derive5 的 MAE 低于所有模型臂（逐题计算比按汇总符号心算准）
  4. N 臂的 MAE 不比常数基线好多少：3 个模型的 N 臂 MAE ≥ 常数基线 MAE − 0.02

防泄漏：Codex / Claude 走命令行、能读本机文件 → 采集期间真值文件不落盘（build 之后删掉，采集完再 build 一次重生成，计算是确定的）；
  提示里也要求只按给出的符号作答、不读任何文件。

用法：~/gut-external/torch215-venv/bin/python methodbench.py build | collect | score
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
CASES = os.path.join(HERE, "methodbench_cases.json")
TRUTH = os.path.join(HERE, "methodbench_truth.json")
CACHE = os.path.join(HERE, "cache_methodbench")
MODELS = ["ds_pro_a", "codex", "claude_sonnet"]
FIELDS = ["n_nodes", "n_faulty_items", "judge", "delta", "h", "d_obs", "sigma_eff", "sigma_node", "sigma_min", "k_good", "k_honest", "share", "q", "rho", "kappa", "lambda"]

SETTING = """Setting. A group of n nodes (AI models) each report a numeric answer for many items; everything below is in log10 units relative to the truth.
On "faulty" items some nodes are poisoned: their report is shifted by about delta (the shift varies a little from item to item).
An independent constraint (a "scale") gives its own estimate p for every item, with noise sigma_eff, and it does NOT share the poison.
A judge combines the node reports into one output:
  soft judge  = weighted median of the node reports, each weighted by exp(-(report - p)^2 / (2 s^2));
  hard judge  = median of the reports within 2.5 s of p (median of all reports if none is within).
The judge is "fooled" on a faulty item if |judge output - truth| > h, where h = delta / 2.
For each case, predict the FOOLED RATE: the fraction of faulty items on which the judge is fooled (a number between 0 and 1).

Symbols (all measured without knowing the truth):
  n_nodes, n_faulty_items, judge ("soft" / "hard")
  delta        size of the poison shift (log10); h = delta / 2 is the fooled threshold
  d_obs        observed shift of the node consensus on faulty items vs clean items (how much poison actually came through)
  sigma_eff    noise of the constraint as the judge sees it (for the hard judge: its acceptance scale)
  sigma_node   typical spread of node reports around their median on clean items
  sigma_min    noise of the most precise node
  k_good       number of nodes whose noise is within 2x of the most precise node
  k_honest     estimated size of the group closer to p on faulty items (when nodes split into two groups)
  share        size of the larger group / n on faulty items (1.0 = no visible split)
  q            dispersion of the poison size across items
  rho = d_obs / sigma_eff,  kappa = d_obs / sigma_node,  lambda = delta * sqrt(k_good) / sigma_min
"""

METHOD = """Method (the law to apply).
Three ratios decide the fooled rate (valid because the constraint is independent of the poison):
  kappa  - can you SEE that the nodes split into two groups? Small kappa (< ~1): the poison is buried in node noise; the consensus moves by
           d_obs and the judge is fooled only if d_obs plus noise pushes it past h.
  rho    - can the constraint tell WHICH group is wrong? rho >= ~1: the judge sides with the group near p (the honest group), and the
           fooled rate is low, set by the noise left after picking the right nodes.
           rho < ~1: the constraint cannot tell; the judge falls back toward a head count: if the poisoned group is the larger one
           (share high and k_honest small), the judge is fooled on most faulty items; if the honest group is larger, rarely.
  lambda - after picking the right nodes, how much noise remains relative to the threshold: residual error ~ sigma_min / sqrt(k_good);
           the fooled rate in the rho >= 1 regime is about P(|N(0, (sigma_min/sqrt(k_good))^2)| > h), small when lambda is large.
How to compute, step by step:
  1. If d_obs is much smaller than h (the poison barely came through), the fooled rate is near the clean-item rate: P(|noise of the judge| > h).
  2. Otherwise, if rho >= 1: low fooled rate, from the residual noise described under lambda (lambda large -> near 0).
  3. If rho < 1: interpolate toward the head-count outcome: fooled ~ fraction of faulty items where the poisoned group wins the vote
     (poisoned majority -> high, honest majority -> low), softened by how close rho is to 1.
  4. The soft judge degrades smoothly as rho drops; the hard judge is closer to a threshold test at |shift| ~ 2.5 s.
  5. A large q (poison size varying a lot) spreads the outcome: some items are fooled, others are not.
"""

TAIL = "\nAnswer only from the symbols given here; do not open, search or read any files.\nReply with ONLY a JSON object mapping each case ID to your predicted fooled rate (0 to 1). No prose.\n\nCases (one per line, symbols as key=value):\n"


def build():
    sys.path.insert(0, os.path.join(HERE, "..", "switching"))
    cwd = os.getcwd()
    from invert import sets
    from hetero import measure
    from intervals import interval2
    S = sets()
    os.chdir(cwd)
    cases, truth = [], {}
    for name, data, nodes, always, sleeper, deltas in S:
        clean = [(f, p, r, x) for f, p, r, x, _ in data[("R0", 0.0)]]
        for d in deltas[1:]:
            bad = [(f, p, r, x) for f, p, r, x, _ in data[("R0", d)]]
            for judge in ("软", "硬2"):
                pt, lo, hi, s0, rh, bw = interval2(clean, bad, judge, B=10)
                m = measure([(False, p, r, x) for _, p, r, x in clean] + bad, math.log10(1 + d), judge, s0)
                dl = math.log10(1 + d)
                sym = {"n_nodes": len(nodes), "n_faulty_items": len(bad), "judge": "soft" if judge == "软" else "hard", "delta": dl, "h": dl / 2,
                       "d_obs": m["d_obs"], "sigma_eff": m["sigma_eff"], "sigma_node": m["sigma_node"], "sigma_min": m["sigma_min"],
                       "k_good": m["k_good"], "k_honest": m["k_honest"], "share": m["share"], "q": m["q"],
                       "rho": m["d_obs"] / m["sigma_eff"], "kappa": m["d_obs"] / m["sigma_node"], "lambda": dl * math.sqrt(m["k_good"]) / max(m["sigma_min"], 1e-9)}
                cases.append(sym)
                truth[len(cases) - 1] = {"source": f"{name} δ={d} {judge}", "miss": m["miss"], "derive5": pt}
    order = list(range(len(cases)))
    random.Random("methodbench").shuffle(order)
    out, tr = [], {}
    for j, i in enumerate(order):
        cid = f"M{j:03d}"
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
    for arm, head in (("M", SETTING + "\n" + METHOD), ("N", SETTING)):
        for b in range(0, len(cases), 20):
            out.append((arm, b // 20, head + TAIL + "\n".join(line(c) for c in cases[b:b + 20])))
    return out


def collect_all():
    sys.path.insert(0, HERE)
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
                raw = C.ask(model, text, 8000, True)
                json.dump({"answers": C.parse(raw), "raw": raw}, open(path, "w"))
                return model, arm, b, "ok"
            except Exception as ex:
                err = repr(ex)[:120]
                time.sleep(10)
        return model, arm, b, f"FAILED {err}"
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    jobs = [(m, arm, b, t) for m in MODELS for arm, b, t in prompts()]
    futs = [pools[C.NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def spearman(a, b):
    import torch
    rk = lambda v: torch.tensor(v).argsort().argsort().float()
    return torch.corrcoef(torch.stack([rk(a), rk(b)]))[0, 1].item()


def score():
    tr = json.load(open(TRUTH))
    ids = sorted(tr)
    act = [tr[i]["miss"] for i in ids]
    const = st.mean(act)
    mae = lambda pred: st.mean(abs(p - a) for p, a in zip(pred, act))
    d5 = [tr[i]["derive5"] for i in ids]
    print(f"{len(ids)} 个案例；实测被骗率均值 {const:.3f}")
    print(f"  derive5（逐题算）  MAE {mae(d5):.3f}  Spearman {spearman(d5, act):.2f}")
    print(f"  常数基线           MAE {mae([const] * len(act)):.3f}")
    R = {}
    for model in MODELS:
        for arm in ("M", "N"):
            ans = {}
            for fn in os.listdir(CACHE):
                mm, key = fn[:-5].split("__")
                if mm == model and key.startswith(arm + "_"):
                    ans.update(json.load(open(os.path.join(CACHE, fn)))["answers"])
            got = [i for i in ids if i in ans]
            pred = [min(1.0, max(0.0, ans[i])) if i in ans else const for i in ids]
            R[(model, arm)] = (mae(pred), spearman(pred, act), len(got))
            print(f"  {model:13s} {arm} 臂  MAE {R[(model, arm)][0]:.3f}  Spearman {R[(model, arm)][1]:.2f}  答了 {len(got)}/{len(ids)}（缺的按常数基线补）")
    cm = mae([const] * len(act))
    print("\n预注册检查：")
    print(f"1. M < N 的模型 {sum(R[(m, 'M')][0] < R[(m, 'N')][0] for m in MODELS)}/3（≥ 2）")
    best = min(MODELS, key=lambda m: R[(m, 'M')][0])
    print(f"2. 最好的 M 臂（{best}）MAE {R[(best, 'M')][0]:.3f}（≤ 0.10）、Spearman {R[(best, 'M')][1]:.2f}（≥ 0.7）")
    print(f"3. derive5 MAE {mae(d5):.3f} 低于所有模型臂：{'是' if all(mae(d5) < R[k][0] for k in R) else '否'}")
    print(f"4. N 臂 MAE ≥ 常数基线 {cm:.3f} − 0.02 的模型 {sum(R[(m, 'N')][0] >= cm - 0.02 for m in MODELS)}/3（须 3/3）")


if __name__ == "__main__":
    {"build": build, "collect": collect_all, "score": score}[sys.argv[1]]()
