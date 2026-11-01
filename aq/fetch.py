"""拉 17 个站点里所有传感器 2025-09-01 → 2025-10-27 的小时 PM2.5（缓存在 cache/，按传感器一个文件）。"""
import json, os, time, urllib.request, urllib.parse
KEY = open(os.path.expanduser("~/.config/openaq/key")).read().strip()
os.makedirs("cache", exist_ok=True)
def get(path, **q):
    url = "https://api.openaq.org/v3/" + path + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"X-API-Key": KEY})
    for k in range(5):
        try:
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception as e:
            err = e; time.sleep(5 * (k + 1))
    raise err
sites = json.load(open("sites.json"))
ids = sorted({s["A"]["sensor"] for s in sites} | {s["B"]["sensor"] for s in sites} | {n["sensor"] for s in sites for n in s["nodes"]})
print(len(ids), "个传感器", flush=True)
for j, sid in enumerate(ids):
    fn = f"cache/{sid}.json"
    if os.path.exists(fn):
        continue
    rows, page = [], 1
    while True:
        r = get(f"sensors/{sid}/hours", datetime_from="2025-09-01T00:00:00Z", datetime_to="2025-10-27T00:00:00Z", limit=1000, page=page)
        time.sleep(1.1)
        rows += [(x["period"]["datetimeFrom"]["utc"], x["value"]) for x in r["results"]]
        if len(r["results"]) < 1000:
            break
        page += 1
    json.dump(rows, open(fn, "w"))
    print(j + 1, sid, len(rows), flush=True)
