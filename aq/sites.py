"""从探查结果拼站点：参考站 A（真值）+ 3 km 内 ≥ 5 个低成本传感器（节点）+ 3–15 km 外另一参考站 B（秤）。规则在看数据值之前定好。"""
import json, math
L = json.load(open("probe_locations.json"))
def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    return 6371 * 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
def pm(l):
    return [s["id"] for s in l.get("sensors", []) if s.get("parameter", {}).get("id") == 2]
sites = []
for city, locs in L.items():
    rec = [l for l in locs if (l.get("datetimeLast") or {}).get("utc", "") >= "2025-11" and pm(l) and l.get("coordinates")]
    ref = [l for l in rec if l.get("isMonitor")]
    low = [l for l in rec if not l.get("isMonitor")]
    used = set()
    for a in ref:
        nodes = sorted((l for l in low if km(a["coordinates"], l["coordinates"]) <= 3 and l["id"] not in used), key=lambda l: km(a["coordinates"], l["coordinates"]))[:9]
        bs = sorted((b for b in ref if b["id"] != a["id"] and 3 <= km(a["coordinates"], b["coordinates"]) <= 15), key=lambda b: km(a["coordinates"], b["coordinates"]))
        if len(nodes) >= 5 and bs:
            b = bs[0]
            used |= {l["id"] for l in nodes}
            sites.append({"city": city, "A": {"loc": a["id"], "name": a["name"], "sensor": pm(a)[0]}, "B": {"loc": b["id"], "name": b["name"], "sensor": pm(b)[0], "km": round(km(a["coordinates"], b["coordinates"]), 1)},
                          "nodes": [{"loc": l["id"], "sensor": pm(l)[0], "provider": (l.get("provider") or {}).get("name"), "km": round(km(a["coordinates"], l["coordinates"]), 2)} for l in nodes]})
json.dump(sites, open("sites.json", "w"), indent=1, ensure_ascii=False)
for s in sites:
    print(s["city"], s["A"]["name"][:30], "| 秤", s["B"]["name"][:25], s["B"]["km"], "km | 节点", len(s["nodes"]), {p: sum(n["provider"] == p for n in s["nodes"]) for p in {n["provider"] for n in s["nodes"]}})
print(len(sites), "个站点")
