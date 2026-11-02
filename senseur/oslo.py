"""奥斯陆 KVN（SensEURCity）数据载入 + 映射预检（沙盒 = 2020 第 36–37 周，测试留第 38–41 周）。"""
import datetime as dt
import json
import math
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "switching"))
H = math.log10(1.25)


def week(h):
    return dt.date.fromisoformat(h[:10]).isocalendar()[1]


def load():
    out = {}
    for f in sorted(os.listdir(os.path.join(HERE, "hourly"))):
        out[f[:-5]] = json.load(open(os.path.join(HERE, "hourly", f)))
    return out


def rows(D, weeks, units, truth="fidas", scale="teom", off=None):
    """[(小时, p, r)]，全部 log10 且相对真值；off = 每台的对数偏移（校准用）。"""
    tname = sorted({h for u in units for h in D[u] if h.startswith("2020") and week(h) in weeks})
    out = []
    for h in tname:
        v = [D[u].get(h) for u in units]
        if any(x is None or x["node"] is None or x["node"] <= 0 for x in v):
            continue
        ref = next((x for x in v if x[truth] is not None and x[scale] is not None), None)
        if ref is None or ref[truth] < 3 or ref[scale] <= 0:
            continue
        tru = math.log10(ref[truth])
        r = torch.tensor([math.log10(x["node"]) - tru for x in v])
        if off is not None:
            r = r - off
        out.append((h, math.log10(ref[scale]) - tru, r))
    return out


def present(D, weeks, thresh=0.8):
    """沙盒 + 测试周都要有数的单元（覆盖率 ≥ thresh）。"""
    out = []
    for u, d in D.items():
        n = sum(1 for h, v in d.items() if h.startswith("2020") and week(h) in weeks and v["node"] is not None)
        if n >= thresh * 24 * 7 * len(weeks):
            out.append(u)
    return out


if __name__ == "__main__":
    from law_v6 import fit
    D = load()
    us = present(D, [37])
    print("第 37 周在线", len(us), "台")
    R = rows(D, [36, 37], us)
    print("沙盒", len(R), "小时（全部节点 + 两台参考同时有数）")
    X = torch.stack([r for _, _, r in R])
    P = torch.tensor([p for _, p, _ in R])
    print(f"秤（TEOM）相对真值（Fidas）：中位 {10 ** P.median().item() - 1:+.0%}，MAD {1.4826 * (P - P.median()).abs().median().item():.3f}（h = {H:.3f}）")
    med = X.median(1).values
    print(f"节点共识相对真值：中位 {10 ** med.median().item() - 1:+.0%}，MAD {1.4826 * (med - med.median()).abs().median().item():.3f}")
    off = X.median(0).values
    print("各台偏移（对数）范围", round(off.min().item(), 3), round(off.max().item(), 3))
    Xc = X - off
    medc = Xc.median(1).values
    print(f"校准各台偏移后：共识 MAD {1.4826 * (medc - medc.median()).abs().median().item():.3f}；单台噪声 MAD 中位 {(1.4826 * (Xc - Xc.median(0).values).abs().median(0).values).median().item():.3f}")
    print(f"校准后共识误差自相关（滞后 1 小时）{torch.corrcoef(torch.stack([medc[:-1], medc[1:]]))[0, 1].item():.2f}；相对湿度列未抽取")
    # 节点两两相关（校准后）
    C = torch.corrcoef(Xc.T)
    k = C.shape[0]
    print(f"校准后节点误差两两相关 平均 {(C.sum() - k) / (k * (k - 1)):.2f}（规律假设诚实节点误差独立）")
