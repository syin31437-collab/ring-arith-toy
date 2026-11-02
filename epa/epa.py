"""EPA 长期性能数据（Barkjohn et al. 2025，DOI 10.23719/1531918）：载入 + 映射预检（沙盒 = 每站最早 90 天；之后留作测试）。
节点 = 每站各品牌（同品牌多台时取同一小时的中位数）；真值 = 该站参考仪器 PM25_Ref；单位 log10，真值 < 3 丢掉。用未清洗版（保留真实故障）。"""
import collections, csv, datetime as dt, json, math, os, statistics as st, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
FN = os.path.join(HERE, "data", "LTPPUNcleanDataset_12_10_24.csv")
H = math.log10(1.25)

def load():
    S = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    Ref = collections.defaultdict(dict)
    with open(FN, newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        tc = next(c for c in cols if c.lower() in ("date", "datetime", "datetime_utc", "time", "dateutc", "date_utc", "hour_utc", "localtime") or "date" in c.lower())
        for row in rd:
            try:
                v, ref = float(row["PM25_sens"]), float(row["PM25_Ref"])
            except (ValueError, KeyError):
                continue
            t = row[tc]
            S[row["Location"]][t][row["Make_ID"]].append(v)
            Ref[row["Location"]][t] = ref
    return S, Ref, tc

if __name__ == "__main__":
    S, Ref, tc = load()
    print("时间列", tc)
    for loc in sorted(S):
        ts = sorted(S[loc])
        makes = collections.Counter(m for t in ts for m in S[loc][t])
        print(loc, len(ts), "小时", ts[0], "→", ts[-1], {m: round(c / len(ts), 2) for m, c in makes.items()})
