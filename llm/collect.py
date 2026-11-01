"""真实模型当节点：问各国人口、国土总面积、人口密度，三个量分开问。

真值：世界银行 2022（wb_*.json），密度 = 人口 / 国土总面积。
节点：DeepSeek-flash ×3（不同提示/温度）、DeepSeek-v4-pro ×2、Codex（codex exec）、Claude Haiku（claude -p）。
每次请求 25 个国家，国家顺序按节点打乱；回答缓存在 cache/，中断后重跑会跳过已完成的。

用法：python3 collect.py            （采集全部节点）
      python3 collect.py NODE ...   （只采集指定节点）
"""
import concurrent.futures as cf
import json
import os
import random
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SCRATCH = "/private/tmp/claude-501/-Users-sly/56edc999-af00-4c01-852b-3a43916c7447/scratchpad"
BATCH = 25

QUESTIONS = {
    "pop": "population in 2022, in millions of people",
    "area": "total surface area (land plus inland water), in square kilometres",
    "dens": "population density in 2022, in people per square kilometre of total surface area",
}
NODES = {
    "ds_flash_a": ("deepseek", "deepseek-flash", 0.0, "You are a precise data assistant."),
    "ds_flash_b": ("deepseek", "deepseek-flash", 1.0, "Answer quickly from memory; rough estimates are fine."),
    "ds_flash_c": ("deepseek", "deepseek-flash", 0.7, "You are a secondary-school geography teacher."),
    "ds_pro_a": ("deepseek", "deepseek-v4-pro", 0.0, "You are a precise data assistant."),
    "ds_pro_b": ("deepseek", "deepseek-v4-pro", 0.7, "You are a secondary-school geography teacher."),
    "codex": ("codex", None, None, None),
    "claude_haiku": ("claude", "haiku", None, None),
}


def truth():
    meta = {c["id"]: c for c in json.load(open(os.path.join(HERE, "wb_countries.json")))[1]}
    pop = {r["countryiso3code"]: r["value"] for r in json.load(open(os.path.join(HERE, "wb_SP.POP.TOTL.json")))[1] if r["value"]}
    area = {r["countryiso3code"]: r["value"] for r in json.load(open(os.path.join(HERE, "wb_AG.SRF.TOTL.K2.json")))[1] if r["value"]}
    out = {}
    for iso, c in meta.items():
        if c["region"]["value"] != "Aggregates" and iso in pop and iso in area:
            out[iso] = {"name": c["name"], "pop": pop[iso] / 1e6, "area": area[iso], "dens": pop[iso] / area[iso]}
    return out


def prompt(qkey, items):
    lines = "\n".join(f"{iso}: {name}" for iso, name in items)
    return (f"For each country below, give your best estimate of its {QUESTIONS[qkey]}.\n"
            f"Reply with ONLY a JSON object mapping the ISO3 code to a number. No prose, no units.\n\n{lines}")


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    raw = json.loads(m.group(0))
    out = {}
    for k, v in raw.items():
        try:
            out[k.strip().upper()] = float(str(v).replace(",", ""))
        except ValueError:
            pass
    return out


def ask(node, text, max_tokens=8000, think=True):
    kind, model, temp, system = NODES[node]
    if kind == "deepseek":
        e = json.load(open(os.path.expanduser("~/.hermes/auth.json")))["credential_pool"]["deepseek"][0]
        body = json.dumps({"model": model, "temperature": temp, "max_tokens": max_tokens, "response_format": {"type": "json_object"},
                           **({} if think else {"thinking": {"type": "disabled"}}),
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}).encode()
        req = urllib.request.Request(e["base_url"].rstrip("/") + "/chat/completions", data=body,
                                     headers={"Authorization": "Bearer " + e["access_token"], "Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    if kind == "codex":
        out = os.path.join(SCRATCH, f"codex_{random.random():.12f}.txt")
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "-o", out, text],
                       cwd=SCRATCH, capture_output=True, timeout=600, stdin=subprocess.DEVNULL)   # 新版 codex 会等 stdin，不关就卡到超时
        with open(out) as f:
            return f.read()
    return subprocess.run(["claude", "-p", "--model", model, text], cwd=SCRATCH,
                          capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL).stdout


def job(node, qkey, b, items):
    path = os.path.join(CACHE, f"{node}_{qkey}_{b}.json")
    if os.path.exists(path):
        return node, qkey, b, "cached"
    for attempt in range(3):
        try:
            text = ask(node, prompt(qkey, items))
            ans = parse(text)
            json.dump({"items": [i for i, _ in items], "answers": ans, "raw": text}, open(path, "w"))
            return node, qkey, b, f"ok {len(ans)}/{len(items)}"
        except Exception as ex:
            err = repr(ex)[:120]
    return node, qkey, b, f"FAILED {err}"


def main():
    os.makedirs(CACHE, exist_ok=True)
    T = truth()
    nodes = sys.argv[1:] or list(NODES)
    jobs = []
    for node in nodes:
        order = sorted(T)
        random.Random(node).shuffle(order)
        batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        for qkey in QUESTIONS:
            for b, isos in enumerate(batches):
                jobs.append((node, qkey, b, [(i, T[i]["name"]) for i in isos]))
    print(f"{len(T)} 个国家，{len(jobs)} 次请求", flush=True)
    pools = {"deepseek": cf.ThreadPoolExecutor(8), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools[NODES[j[0]][0]].submit(job, *j) for j in jobs]
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


if __name__ == "__main__":
    main()
