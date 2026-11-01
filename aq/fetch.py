"""拉 17 个站点里所有传感器 2025-09-01 → 2025-10-27 的小时 PM2.5（缓存在 cache/，按传感器一个文件）。"""
import json, os, time, urllib.request, urllib.parse
KEY = open(os.path.expanduser("~/.config/openaq/key")).read().strip()
PERIOD = os.environ.get("PERIOD", "sep")                     # sep = 2025-09-01 → 10-27（第一轮）；jun = 2025-06-02 → 07-28（预注册的新时段）
WIN = {"sep": ("2025-09-01T00:00:00Z", "2025-10-27T00:00:00Z", "cache"), "jun": ("2025-06-02T00:00:00Z", "2025-07-28T00:00:00Z", "cache_jun")}[PERIOD]
os.makedirs(WIN[2], exist_ok=True)
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
if os.path.exists("candidates.json"):                       # 第二版选站：所有候选传感器都拉，再按实际覆盖率选
    ids = sorted(set(ids) | set(json.load(open("candidates.json"))))
print(len(ids), "个传感器", flush=True)
for j, sid in enumerate(ids):
    fn = f"{WIN[2]}/{sid}.json"
    if os.path.exists(fn):
        continue
    rows, page = [], 1
    try:
        while True:
            r = get(f"sensors/{sid}/hours", datetime_from=WIN[0], datetime_to=WIN[1], limit=500, page=page)
            time.sleep(1.1)
            rows += [(x["period"]["datetimeFrom"]["utc"], x["value"]) for x in r["results"]]
            if len(r["results"]) < 500:
                break
            page += 1
    except Exception as e:                                   # 一直失败的传感器跳过、记下来（不落缓存，下次重跑会再试）
        print(j + 1, sid, "FAILED", repr(e)[:80], flush=True)
        continue
    json.dump(rows, open(fn, "w"))
    print(j + 1, sid, len(rows), flush=True)
