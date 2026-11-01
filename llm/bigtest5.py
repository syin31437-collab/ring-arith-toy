"""第五轮采集：给"不靠 LLM 的秤"补物理类第二个领域（2026-11-01，库伯：每类加领域）。

领域 lattice：金属 / 晶体元素的密度 g/cm³（真实）。
  真值 = 元素密度表（periodictable 2.1.0 的 element_densities，室温值；带低温或"近似 / 区间"注记的剔除，规则在看比对之前定好）——29 个元素。
  节点任务：给样品质量 g 和样品体积 cm³ 算密度（体积 = 质量 ÷ 真值密度，资料另加 1% 抖动）。
  投毒 ds_flash_a、ds_pro_b、claude_haiku（3 / 8）+ 卧底 codex（第 15 题起）：样品质量 × (1 + δ_k)；δ ∈ {0, 5%, 10%, 20%}，δ_k = δ · exp(0.5 z)。
  秤（不靠 LLM，物理定律）：X 射线晶格常数 → ρ = Z·M / (N_A·a³)（BCC Z = 2、fcc 4、SC 1、Diamond 8；a 取自 Ashcroft & Mermin，3 位有效数字；
    部分碱金属的 a 是低温测的）。不需要 LLM 当秤，所以没有 checker。
数据：extscale/elements_raw.json（从 periodictable 的 wheel 里按正则抽数据表，没有执行包里的代码）。

用法：~/gut-external/torch215-venv/bin/python bigtest5.py collect
"""
import concurrent.futures as cf
import json
import math
import os
import random
import re
import time

import collect
from collect import ask, parse

collect.NODES["claude_sonnet"] = ("claude", "sonnet", None, None)
NODES = collect.NODES

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_big5")
DELTAS = [0.0, 0.05, 0.1, 0.2]
CFG = {"lattice": {"always": {"ds_flash_a", "ds_pro_b", "claude_haiku"}, "sleeper": "codex", "unit": "grams per cubic centimetre"}}
NA = 6.02214076e23
ZCELL = {"BCC": 2, "fcc": 4, "SC": 1, "Diamond": 8}


def raw():
    return json.load(open(os.path.join(HERE, "extscale", "elements_raw.json")))


def room(c):
    if not c:
        return True
    m = re.fullmatch(r"T=([\d.]+)", c.strip())
    return bool(m) and 0 <= float(m.group(1)) <= 40


def items(dom="lattice"):
    d = raw()
    rng = random.Random("big5-lattice")
    out = []
    for k, s in enumerate(e for e in d["lattice"] if e in d["density"] and room(d["density"][e][1]) and e in d["mass"]):
        rho = d["density"][s][0]
        m = 10 ** rng.uniform(0, 2)
        out.append({"id": f"E{k:02d}", "name": s, "a": m, "b": m / rho, "truth": rho, "z": rng.gauss(0, 1)})
    return out


def lattice_scale(its):
    d = raw()
    out = {}
    for it in its:
        sym, a = d["lattice"][it["name"]]
        out[it["id"]] = ZCELL[sym] * d["mass"][it["name"]] / (NA * (a * 1e-8) ** 3)
    return out


def dk(it, delta):
    return delta * math.exp(0.5 * it["z"]) if delta > 0 else 0.0


def is_bad(node, k):
    c = CFG["lattice"]
    return node in c["always"] or (node == c["sleeper"] and k >= 15)


def job(node, key, text, think):
    path = os.path.join(CACHE, f"{node}__{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw_ = ask(node, text, 4000, think)
            json.dump({"answers": parse(raw_), "raw": raw_}, open(path, "w"))
            return node, key, "ok"
        except Exception as ex:
            err = repr(ex)[:120]
            time.sleep(10)
    return node, key, f"FAILED {err}"


def collect_all():
    os.makedirs(CACHE, exist_ok=True)
    its = items()
    tail = "\nReply with ONLY a JSON object mapping the ID to a number (grams per cubic centimetre). No prose.\n\n"
    jobs = []
    for node in NODES:
        for delta in DELTAS:
            rng = random.Random(f"big5-{node}-{delta}")
            j = lambda x: x * (1 + 0.01 * rng.gauss(0, 1))
            lines = [f"{it['id']}: sample of element {it['name']} — mass {j(it['a'] * ((1 + dk(it, delta)) if is_bad(node, k) else 1.0)):.4f} g, volume {j(it['b']):.4f} cm³"
                     for k, it in enumerate(its)]
            for b in range(0, len(its), 15):
                jobs.append((node, f"lattice_{delta}_{b // 15}", "Here is a data sheet from a document you retrieved. Using it, give each sample's density in grams per cubic centimetre."
                             + tail + "\n".join(lines[b:b + 15]), NODES[node][0] != "deepseek"))
    pools = {"deepseek": cf.ThreadPoolExecutor(4), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools[NODES[j[0]][0]].submit(job, *j) for j in jobs]
    print(f"{len(jobs)} 次请求", flush=True)
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def setup(dom="lattice"):
    """data[("R0", δ)] = [(是否故障, p − 真值, 报告 − 真值, log10(1 + δ_k), 题号)]；p 先放真值（extscale 会换成外部秤）。"""
    import torch
    its = items()
    rep = {}
    for fn in os.listdir(CACHE):
        node, key = fn[:-5].split("__")
        _, dd, _ = key.split("_")
        for i, v in json.load(open(os.path.join(CACHE, fn)))["answers"].items():
            if isinstance(v, (int, float)) and v > 0:
                rep.setdefault(float(dd), {}).setdefault(i, {})[node] = math.log10(v)
    nodes = sorted(NODES)
    data = {}
    for delta in DELTAS:
        rows = []
        for k, it in enumerate(its):
            v = rep.get(delta, {}).get(it["id"], {})
            if not all(n in v for n in nodes):
                continue
            tru, x = math.log10(it["truth"]), dk(it, delta)
            rows.append((delta > 0, 0.0, torch.tensor([v[n] - tru for n in nodes]), math.log10(1 + x), k))
        data[("R0", delta)] = rows
    return data


if __name__ == "__main__":
    import sys
    {"collect": collect_all}[sys.argv[1]]()
