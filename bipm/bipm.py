"""BIPM UTC − UTC(k)（webtai.bipm.org，5 天一点，ns）载入。节点 = 实验室，真值 = UTC（报告本身就是相对真值的偏差），秤 = GNSS 广播时（UTC − GPS 时、UTC − GLONASS 时）。
题 = 5 天差分（水平值滞后 1 自相关 0.96，差分 0.10）。喂 GNSS 的实验室排除（USNO、SU、NTSC；PTB、OP、IT、ROB、NPL 是子代理凭记忆列的 Galileo 来源，也排除）。
沙盒 = MJD < 57023（2015 年以前）；测试 = 之后。"""
import os, re, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX_END = 57023
EXCL = {"usno", "su", "ntsc", "ptb", "op", "it", "rob", "npl"}

def lab(name):
    out = {}
    for line in open(os.path.join(HERE, "data", "utc-" + name), errors="ignore"):
        m = re.match(r"\s*(\d{5})\s+([-+]?\d+(?:\.\d+)?)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out

def gnss(name):
    out = {}
    for line in open(os.path.join(HERE, "gnss", "utc-" + name)):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 3:
            try:
                out[int(p[0])] = float(p[2])                     # 第 3 列 UTC − GNSST
            except ValueError:
                pass
    return out

def all_labs():
    return {f[4:]: lab(f[4:]) for f in sorted(os.listdir(os.path.join(HERE, "data"))) if f.startswith("utc-") and not f.endswith(".tmp") and f[4:] not in EXCL}

def diffs(series, epochs):
    """相邻 5 天差分；缺一个就 None。"""
    return {t: series[t] - series[t - 5] for t in epochs if t in series and t - 5 in series}
