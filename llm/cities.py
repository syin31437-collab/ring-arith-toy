"""真实模型当节点 · 第三题：城市间距离 vs 城市经纬度。

为什么这样设计：前两题里模型的派生量是"现场算"的（密度、摩尔质量），错误要么和原料一致、要么不出错。
距离没法心算球面三角，模型报的距离更像是"独立记住的"，和它报的经纬度是两份记忆，错误才可能违反约束。
真值：GeoNames cities15000 的坐标，球面大圆距离（R = 6371.0088 km）。
城市：60 个人口最多的 + 60 个人口 30 万–200 万的（每国最多 3 个），共 120；随机配成 200 对。
约束预测：各模型经纬度的中位数 → 大圆距离。
节点和调用方式同 collect.py；缓存在 cache_city/。

用法：python3 cities.py collect [NODE ...]
      ~/gut-external/torch215-venv/bin/python cities.py analyze     （EVENT 环境变量 = 事件门槛，log10）
"""
import concurrent.futures as cf
import json
import math
import os
import random
import re
import sys

from collect import NODES, ask, parse

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_city")
R = 6371.0088


def cities():
    names = {}
    for line in open(os.path.join(HERE, "geonames", "countryInfo.txt"), encoding="utf-8"):
        if not line.startswith("#"):
            p = line.rstrip("\n").split("\t")
            names[p[0]] = p[4]
    rows = []
    for line in open(os.path.join(HERE, "geonames", "cities15000.txt"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if p[7].startswith("PPLA") or p[7] in ("PPLC", "PPL"):
            rows.append({"name": p[2], "cc": p[8], "country": names.get(p[8], p[8]), "lat": float(p[4]), "lon": float(p[5]), "pop": int(p[14])})
    rows.sort(key=lambda r: -r["pop"])
    big, seen = [], set()
    for r in rows:
        if (r["name"], r["cc"]) not in seen and len(big) < 60:
            big.append(r)
            seen.add((r["name"], r["cc"]))
    rng = random.Random(7)
    mid_pool = [r for r in rows if 300_000 <= r["pop"] <= 2_000_000 and (r["name"], r["cc"]) not in seen]
    rng.shuffle(mid_pool)
    mid, per = [], {}
    for r in mid_pool:
        if per.get(r["cc"], 0) < 3 and len(mid) < 60:
            mid.append(r)
            per[r["cc"]] = per.get(r["cc"], 0) + 1
    out = big + mid
    for k, r in enumerate(out):
        r["id"] = f"C{k:03d}"
    return out


def pairs(cs, n=200):
    rng = random.Random(11)
    out = []
    while len(out) < n:
        a, b = rng.sample(cs, 2)
        if (a["id"], b["id"]) not in {(x, y) for _, x, y in out}:
            out.append((f"P{len(out):03d}", a["id"], b["id"]))
    return out


def gc(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return R * math.acos(max(-1.0, min(1.0, c)))


def parse_coords(text):
    m = re.search(r"\{.*\}", text, re.S)
    raw = json.loads(m.group(0))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            v = [v.get("lat", v.get("latitude")), v.get("lon", v.get("lng", v.get("longitude")))]
        try:
            out[k.strip().upper()] = [float(v[0]), float(v[1])]
        except (TypeError, ValueError, IndexError):
            pass
    return out


def job(node, key, text, coords):
    path = os.path.join(CACHE, f"{node}_{key}.json")
    if os.path.exists(path):
        return node, key, "cached"
    err = ""
    for _ in range(3):
        try:
            raw = ask(node, text)
            json.dump({"answers": parse_coords(raw) if coords else parse(raw), "raw": raw}, open(path, "w"))
            return node, key, "ok"
        except Exception as ex:
            err = repr(ex)[:120]
    return node, key, f"FAILED {err}"


def collect(nodes):
    os.makedirs(CACHE, exist_ok=True)
    cs = cities()
    by = {c["id"]: c for c in cs}
    ps = pairs(cs)
    jobs = []
    for node in nodes:
        rng = random.Random(node)
        order = list(cs)
        rng.shuffle(order)
        for b in range(0, len(order), 30):
            items = "\n".join(f"{c['id']}: {c['name']}, {c['country']}" for c in order[b:b + 30])
            jobs.append((node, f"coord_{b // 30}", "Give the latitude and longitude (decimal degrees; south and west negative) of each city below.\n"
                         "Reply with ONLY a JSON object mapping the ID to [latitude, longitude]. No prose.\n\n" + items, True))
        porder = list(ps)
        rng.shuffle(porder)
        for b in range(0, len(porder), 25):
            items = "\n".join(f"{i}: {by[a]['name']}, {by[a]['country']} — {by[c]['name']}, {by[c]['country']}" for i, a, c in porder[b:b + 25])
            jobs.append((node, f"dist_{b // 25}", "Give the great-circle (straight-line) distance in kilometres between the two cities in each pair below. "
                         "Answer from your knowledge; do not use any tools or code.\n"
                         "Reply with ONLY a JSON object mapping the ID to a number. No prose.\n\n" + items, False))
    pools = {"deepseek": cf.ThreadPoolExecutor(8), "codex": cf.ThreadPoolExecutor(3), "claude": cf.ThreadPoolExecutor(3)}
    futs = [pools[NODES[j[0]][0]].submit(job, *j) for j in jobs]
    for f in cf.as_completed(futs):
        print(*f.result(), flush=True)


def load_answers():
    coords, dist = {}, {}
    for fn in os.listdir(CACHE):
        data = json.load(open(os.path.join(CACHE, fn)))["answers"]
        if "_coord_" in fn:
            node = fn[:fn.index("_coord_")]
            for i, v in data.items():
                coords.setdefault(i, {})[node] = v
        else:
            node = fn[:fn.index("_dist_")]
            for i, v in data.items():
                dist.setdefault(i, {})[node] = v
    return coords, dist


def analyze():
    import statistics as st
    import torch
    from analyze import evaluate
    cs = cities()
    by = {c["id"]: c for c in cs}
    ps = pairs(cs)
    coords, dist = load_answers()
    nodes = sorted({n for v in dist.values() for n in v})
    print("== 模型 PK ==")
    print("节点            坐标误差中位(km)   坐标误差>50km 的城市   距离相对误差中位   距离 5% 内   自洽：报的距离 vs 自己坐标算的距离（5% 内）")
    for n in nodes:
        ce = [gc(*coords[c][n], by[c]["lat"], by[c]["lon"]) for c in coords if n in coords[c]]
        de, self_c = [], []
        for i, a, b in ps:
            if n in dist.get(i, {}) and dist[i][n] > 0:
                t = gc(by[a]["lat"], by[a]["lon"], by[b]["lat"], by[b]["lon"])
                de.append(abs(dist[i][n] / t - 1))
                if n in coords.get(a, {}) and n in coords.get(b, {}):
                    own = gc(*coords[a][n], *coords[b][n])
                    self_c.append(abs(dist[i][n] / own - 1) < 0.05)
        print(f"{n:14s} {st.median(ce):12.1f} {sum(x > 50 for x in ce):16d}/{len(ce):<4d} {st.median(de):14.2%} {sum(x < 0.05 for x in de) / len(de):10.0%} {sum(self_c) / len(self_c):18.0%}")
    med_c = {c: [st.median(v[n][0] for n in v), st.median(v[n][1] for n in v)] for c, v in coords.items() if len(v) >= 3}
    rows = []
    for i, a, b in ps:
        rep = [v for v in dist.get(i, {}).values() if v > 0]
        if len(rep) < 5 or a not in med_c or b not in med_c:
            continue
        d = torch.tensor([math.log10(v) for v in rep])
        t = gc(by[a]["lat"], by[a]["lon"], by[b]["lat"], by[b]["lon"])
        rows.append({"iso": i, "d": d, "pred": math.log10(gc(*med_c[a], *med_c[b])), "truth": math.log10(t), "med": d.median().item()})
    evaluate(rows, "城市对")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2:] or list(NODES))
    else:
        analyze()
