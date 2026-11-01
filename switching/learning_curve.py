"""P 组：同一种"谁和谁站一起"的模式，第 n 次出现时输出被带偏的概率。"""
import json
from collections import defaultdict

data = json.load(open("hidden_P.json"))
hits, tot = defaultdict(int), defaultdict(int)
for rec in data:
    seen = defaultdict(int)
    for (o, y, kind), key in zip(rec["dirty"], rec["keys"]):
        if kind is None or key == "agree":
            continue
        seen[key] += 1
        n = min(seen[key], 6)
        tot[n] += 1
        hits[n] += (o - y) ** 2 > 1.0
for n in sorted(tot):
    label = f"第{n}次" if n < 6 else "第6次及以后"
    print(f"{label:8s} 漏过 {hits[n]}/{tot[n]} = {hits[n]/tot[n]:.1%}")
