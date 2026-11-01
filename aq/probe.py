"""OpenAQ 数据探查：哪些城市既有 ≥ 2 个参考级 PM2.5 监测站、又有足够多的低成本传感器。"""
import json, os, time, urllib.request, urllib.parse
KEY = open(os.path.expanduser("~/.config/openaq/key")).read().strip()
def get(path, **q):
    url = "https://api.openaq.org/v3/" + path + ("?" + urllib.parse.urlencode(q) if q else "")
    req = urllib.request.Request(url, headers={"X-API-Key": KEY})
    for _ in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            err = e; time.sleep(3)
    raise err
CITIES = {"LosAngeles": (34.05, -118.25), "SanFrancisco": (37.77, -122.42), "London": (51.51, -0.13), "Delhi": (28.61, 77.21),
          "Bangkok": (13.76, 100.50), "Paris": (48.86, 2.35), "Seattle": (47.61, -122.33), "Denver": (39.74, -104.99),
          "Accra": (5.60, -0.19), "Lagos": (6.52, 3.38), "Nairobi": (-1.29, 36.82), "Ulaanbaatar": (47.92, 106.92)}
out = {}
for c, (la, lo) in CITIES.items():
    locs = []
    for mon in ("true", "false"):
        r = get("locations", coordinates=f"{la},{lo}", radius=25000, parameters_id=2, monitor=mon, limit=1000)
        locs += r["results"]
        time.sleep(1.1)
    recent = [l for l in locs if (l.get("datetimeLast") or {}).get("utc", "") >= "2025-06"]
    ref = [l for l in recent if l.get("isMonitor")]
    low = [l for l in recent if not l.get("isMonitor")]
    out[c] = locs
    prov = {}
    for l in low:
        prov[(l.get("provider") or {}).get("name")] = prov.get((l.get("provider") or {}).get("name"), 0) + 1
    print(f"{c:12s} 参考站 {len(ref):3d}  低成本 {len(low):4d}  低成本来源 {dict(sorted(prov.items(), key=lambda x: -x[1])[:4])}", flush=True)
json.dump(out, open("probe_locations.json", "w"))
