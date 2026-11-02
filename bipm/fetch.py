"""下载 BIPM UTC − UTC(k) 全部实验室文件（带重试，已下载的跳过；写完才改名，避免半截文件）。"""
import os, re, time, urllib.request, concurrent.futures as cf
B = "https://webtai.bipm.org/ftp/pub/tai/other-products/utclab/"
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
for _ in range(5):
    try:
        html = urllib.request.urlopen(B, timeout=60).read().decode(); break
    except Exception:
        time.sleep(10)
names = sorted(set(re.findall(r"utc-[a-z0-9]+", re.sub(r"<[^>]*>", " ", html))))
def go(n):
    if os.path.exists(n) and os.path.getsize(n) > 0:
        return True
    for k in range(6):
        try:
            d = urllib.request.urlopen(B + n, timeout=60).read()
            open(n + ".tmp", "wb").write(d); os.rename(n + ".tmp", n)
            return True
        except Exception:
            time.sleep(5 * (k + 1))
    return False
with cf.ThreadPoolExecutor(3) as ex:
    r = list(ex.map(go, names))
print(len(names), "个文件，成功", sum(r), "失败", [n for n, ok in zip(names, r) if not ok])
