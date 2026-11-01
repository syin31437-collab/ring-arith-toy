"""SensEURCity v3（Zenodo 17858205，CC-BY-4.0）：逐个下载奥斯陆单元的 CSV（HTTP range，不拉整个 5.7 GB 压缩包），
只留 Location.ID == OSL_REF_KVN（和两台参考仪器共址）的行，按小时取平均：PMS5003 PM2.5（5325CAT）、Ref.PM2.5.Fidas、Ref.PM2.5.TEOM。
结果存 hourly/<单元>.json，原始 CSV 处理完就删。"""
import csv, json, os, subprocess, sys
URL = "https://zenodo.org/api/records/17858205/files/senseurcity_data_v03.zip/content"
UNITS = [l.split()[2] for l in open("oslo_files.txt")]
os.makedirs("hourly", exist_ok=True)
os.makedirs("raw", exist_ok=True)
for name in UNITS:
    unit = name.split("_")[1][:-4]
    out = f"hourly/{unit}.json"
    if os.path.exists(out):
        continue
    subprocess.run([sys.executable, "-I", "rzip.py", URL, "raw", name], check=True)
    agg = {}
    with open(os.path.join("raw", name), newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Location.ID") != "OSL_REF_KVN":
                continue
            h = row["date"][:13]
            a = agg.setdefault(h, {"node": [], "fidas": [], "teom": []})
            for k, col in (("node", "5325CAT"), ("fidas", "Ref.PM2.5.Fidas"), ("teom", "Ref.PM2.5.TEOM")):
                try:
                    a[k].append(float(row[col]))
                except (ValueError, KeyError, TypeError):
                    pass
    hourly = {h: {k: (sum(v) / len(v) if len(v) >= 30 else None) for k, v in a.items()} for h, a in agg.items()}
    json.dump(hourly, open(out, "w"))
    os.remove(os.path.join("raw", name))
    print(unit, len(hourly), "小时", flush=True)
