"""第二版选站：按研究窗口内实际覆盖率（≥ 80% 的小时有数）。只用"有没有数"，不看数值。"""
import json, math, os
L = json.load(open("probe_locations.json"))
HOURS = 56 * 24
def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    return 6371 * 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
def cov(sid):
    fn = f"cache/{sid}.json"
    return len([1 for _, v in json.load(open(fn)) if v is not None and v > 0]) / HOURS if os.path.exists(fn) else 0.0
def best(l):
    s = [(cov(x["id"]), x["id"]) for x in l.get("sensors", []) if x.get("parameter", {}).get("id") == 2]
    s = [x for x in s if x[0] >= 0.6]
    return max(s)[1] if s else None
sites, used = [], set()
for city in ("LosAngeles", "SanFrancisco", "London", "Bangkok", "Seattle", "Denver"):
    locs = [l for l in L[city] if l.get("coordinates")]
    ref = [(l, best(l)) for l in locs if l.get("isMonitor")]
    ref = [(l, s) for l, s in ref if s]
    low = [(l, best(l)) for l in locs if not l.get("isMonitor")]
    low = [(l, s) for l, s in low if s]
    for a, sa in ref:
        nodes = sorted(((l, s) for l, s in low if km(a["coordinates"], l["coordinates"]) <= 3 and s not in used), key=lambda x: km(a["coordinates"], x[0]["coordinates"]))[:12]
        bs = sorted(((b, s) for b, s in ref if b["id"] != a["id"] and 3 <= km(a["coordinates"], b["coordinates"]) <= 15), key=lambda x: km(a["coordinates"], x[0]["coordinates"]))
        if len(nodes) >= 5 and bs:
            b, sb = bs[0]
            used |= {s for _, s in nodes}
            sites.append({"city": city, "A": {"loc": a["id"], "name": a["name"], "sensor": sa}, "B": {"loc": b["id"], "name": b["name"], "sensor": sb, "km": round(km(a["coordinates"], b["coordinates"]), 1)},
                          "nodes": [{"loc": l["id"], "sensor": s, "provider": (l.get("provider") or {}).get("name"), "km": round(km(a["coordinates"], l["coordinates"]), 2)} for l, s in nodes]})
json.dump(sites, open("sites2.json", "w"), indent=1, ensure_ascii=False)
for s in sites:
    print(s["city"], s["A"]["name"][:28], "| 秤", s["B"]["name"][:22], s["B"]["km"], "km | 节点", len(s["nodes"]))
print(len(sites), "个站点")
