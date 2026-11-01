"""记忆秤的修法：多把不同家族的记忆秤互相校验（2026-11-01）。

问题：秤是模型记忆时，误差厚尾（偶尔错 16–28%）、同家族共享（和同家族节点错成一样 = 局部 ζ ≈ 1）、在相关题目上扎堆 →
检测层误报、检测变慢。这是四轮大规模测试里剩下的唯一难点。
修法：同一道题让三个家族（DeepSeek-pro、Claude Sonnet、Codex）都凭记忆答，取三者的中位数当秤。
  一个家族记错的题，三个家族同时记错的可能性小；厚尾的大错也会被另外两个压下去（第二条前提的应用：多数家族没记错就站得住）。
5 个记忆领域：bigtest:gdp（原秤 ds_pro_a）、bigtest2:forest（原秤 Sonnet）、bigtest2:urban（Sonnet）、bigtest3:arable（ds_pro_a）、bigtest4:density（Codex）。
节点数据不重采，只换秤；补采 5 × 2 × 3 = 30 次请求。

预注册（11-01，采集之前写下；5 个领域汇总）：
  1. 秤自身相对真值的 MAD：三家中位数 ≤ 原来的单一秤（5 个领域里 ≥ 4 个）
  2. 秤错超过 10% 的题数：三家中位数合计 ≤ 原来单一秤合计的一半
  3. δ = 0 一轮闸门版 EWMA 的真误报合计：三家中位数 ≤ 单一秤
  4. 区间覆盖（R0，软 / 硬2）：三家中位数 ≥ 90%
  5. ρ̂ ≥ 1 的格子点预测误差中位 ≤ 0.08

用法：~/gut-external/torch215-venv/bin/python memscale.py collect | score
"""
import concurrent.futures as cf
import json
import math
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_mem")
DOMS = [("bigtest", "gdp", "ds_pro_a"), ("bigtest2", "forest", "claude_sonnet"), ("bigtest2", "urban", "claude_sonnet"),
        ("bigtest3", "arable", "ds_pro_a"), ("bigtest4", "density", "codex")]
MODELS = ["ds_pro_a", "claude_sonnet", "codex"]


def modules():
    import importlib
    return {m: importlib.import_module(m) for m in ("bigtest2", "bigtest3", "bigtest4")}


def collect():
    sys.path.insert(0, HERE)
    import collect as C
    mods = modules()
    import bigtest
    mods["bigtest"] = bigtest
    os.makedirs(CACHE, exist_ok=True)
    jobs = []
    for mname, dom, orig in DOMS:
        mod = mods[mname]
        its = mod.items(dom)
        q, lines = mod.check_text(dom, its)
        tail = f"\nReply with ONLY a JSON object mapping the ID to a number ({mod.CFG[dom]['unit']}). No prose.\n\n"
        for model in MODELS:
            if model == orig:
                continue
            for b in range(0, len(its), 20):
                jobs.append((model, f"{mname}_{dom}_{b // 20}", q + tail + "\n".join(lines[b:b + 20])))

    def job(model, key, text):
        path = os.path.join(CACHE, f"{model}__{key}.json")
        if os.path.exists(path):
            return model, key, "cached"
        err = ""
        for _ in range(3):
            try:
                raw = C.ask(model, text, 4000, model != "ds_pro_a")
                json.dump({"answers": C.parse(raw), "raw": raw}, open(path, "w"))
                return model, key, "ok"
            except Exception as ex:
                err = repr(ex)[:120]
                time.sleep(10)
        return model, key, f"FAILED {err}"
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools[C.NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def scales(mname, dom, orig, mod):
    """每道题三家的记忆答案（log10）：原秤从该轮缓存读，另两家从 cache_mem 读。"""
    out = {}
    for fn in os.listdir(mod.CACHE):
        node, key = fn[:-5].split("__")
        if node.startswith("check_0") and key.startswith(f"{dom}_0.0_"):
            for i, v in json.load(open(os.path.join(mod.CACHE, fn)))["answers"].items():
                if isinstance(v, (int, float)) and v > 0:
                    out.setdefault(i, {})[orig] = math.log10(v)
    for fn in os.listdir(CACHE):
        model, key = fn[:-5].split("__")
        if key.startswith(f"{mname}_{dom}_"):
            for i, v in json.load(open(os.path.join(CACHE, fn)))["answers"].items():
                if isinstance(v, (int, float)) and v > 0:
                    out.setdefault(i, {})[model] = math.log10(v)
    return out


def score():
    sys.path.insert(0, os.path.join(HERE, "..", "switching"))
    cwd = os.getcwd()
    import torch
    from commonmode import run_lines
    from intervals import interval2
    from invert import loo, sets
    from hetero import measure
    data_all = {name: (data, nodes) for name, data, nodes, *_ in sets()}
    os.chdir(cwd)
    mods = modules()
    import bigtest
    mods["bigtest"] = bigtest
    tot = {"single": {"mad": 0, "gross": 0, "false": 0, "cov": [], "perr": [], "better": 0}, "fused": {"mad": 0, "gross": 0, "false": 0, "cov": [], "perr": []}}
    for mname, dom, orig in DOMS:
        mod = mods[mname]
        its = mod.items(dom)
        S = scales(mname, dom, orig, mod)
        data, nodes = data_all[f"{mname}:{dom}"]
        tru = {it["id"]: math.log10(it["truth"]) for it in its}
        idk = {k: it["id"] for k, it in enumerate(its)}
        res = {}
        for kind in ("single", "fused"):
            def pval(i):
                s = S.get(i, {})
                if kind == "single":
                    return s.get(orig)
                vals = [s[m] for m in MODELS if m in s]
                return st.median(vals) if vals else None
            D = {}
            for key, rows in data.items():
                new = []
                for f, p, r, x, k in rows:
                    pv = pval(idk[k])
                    if pv is not None:
                        new.append((f, pv - tru[idk[k]], r, x, k))
                D[key] = new
            clean = [(f, p, r, x) for f, p, r, x, _ in D[("R0", 0.0)]]
            e = [p for _, p, _, _ in clean]
            mad = 1.4826 * st.median(abs(x - st.median(e)) for x in e)
            gross = sum(abs(10 ** x - 1) > 0.10 for x in e)
            first, _ = run_lines(clean, D[("R0", 0.0)], nodes, "cm2")
            sig = loo(torch.stack([r for _, _, r, _ in clean]))
            nfalse = 0
            for n, kk in first.items():
                j = nodes.index(n)
                prev = [abs(r[j].item()) for _, _, r, _, k2 in D[("R0", 0.0)] if kk - 10 <= k2 <= kk]
                if prev and st.mean(prev) < 2 * sig[j].item():
                    nfalse += 1
            cov, perr = [], []
            for d in mod.DELTAS[1:]:
                bad = [(f, p, r, x) for f, p, r, x, _ in D[("R0", d)]]
                for judge in ("软", "硬2"):
                    pt, lo, hi, s0, rh, bw = interval2(clean, bad, judge, B=30)
                    act = measure([(False, p, r, x) for _, p, r, x in clean] + bad, math.log10(1 + d), judge, s0)["miss"]
                    cov.append(lo <= act <= hi)
                    if rh >= 1:
                        perr.append(abs(act - pt))
            res[kind] = (mad, gross, nfalse, cov, perr)
            tot[kind]["mad"] += mad
            tot[kind]["gross"] += gross
            tot[kind]["false"] += nfalse
            tot[kind]["cov"] += cov
            tot[kind]["perr"] += perr
        if res["fused"][0] <= res["single"][0]:
            tot["single"]["better"] += 1
        n_cov = sum(1 for i in S if all(m in S[i] for m in MODELS))
        print(f"{mname}:{dom:8s}（三家都答了 {n_cov} 题）  秤 MAD 单一 {res['single'][0]:.4f} → 三家 {res['fused'][0]:.4f}   错 > 10% 的题 {res['single'][1]} → {res['fused'][1]}"
              f"   真误报 {res['single'][2]} → {res['fused'][2]}   区间覆盖 {st.mean(res['single'][3]):.0%} → {st.mean(res['fused'][3]):.0%}")
    s, f = tot["single"], tot["fused"]
    print(f"\n1. 秤 MAD 三家 ≤ 单一的领域 {s['better']}/5（≥ 4）")
    print(f"2. 秤错 > 10% 的题：单一 {s['gross']} → 三家 {f['gross']}（≤ 一半 = {s['gross'] / 2:.0f}）")
    print(f"3. 真误报：单一 {s['false']} → 三家 {f['false']}（三家 ≤ 单一）")
    print(f"4. 区间覆盖（三家）{st.mean(f['cov']):.0%}（≥ 90%）；单一 {st.mean(s['cov']):.0%}")
    print(f"5. ρ̂ ≥ 1 点预测误差中位（三家）{st.median(f['perr']) if f['perr'] else float('nan'):.3f}（≤ 0.08）；单一 {st.median(s['perr']) if s['perr'] else float('nan'):.3f}")


if __name__ == "__main__":
    {"collect": collect, "score": score}[sys.argv[1]]()
