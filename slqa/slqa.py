"""USGS 泥沙实验室质控（SLQA）。节点 / 秤 = 实验室（永久 ID）；题 = 一个质控样；真值 = 称重值（每个实验室拿自己的瓶子，所以用 ln(报告 / 该瓶真值)，节点值直接就是相对真值的误差）。
只用新格式（src = new，题在实验室之间对得上）。按 study 时间排序。沙盒 = 前 30% 的题，测试 = 其余。"""
import collections, csv, math, os
HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    E = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "parsed.csv"))):
        if r["src"] != "new":
            continue
        try:
            a, x = float(r["act"]), float(r["rep"])
        except ValueError:
            continue
        if a > 0 and x > 0:
            E[r["item"]].setdefault(r["lab"], math.log(x / a))
    items = sorted(E, key=lambda i: (i.split("|")[0], i))
    return E, items
