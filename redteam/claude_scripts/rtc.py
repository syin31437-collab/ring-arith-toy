"""红队公用：只读导入 repo；缓存窗口。"""
import sys, os, pickle, math, random, statistics as st
sys.dont_write_bytecode = True
REPO = "/Users/sly/Desktop/agent/claude/ring-arith-toy"
for d in ("theory", "switching", "bipm", "slqa", "fl", ""):
    sys.path.insert(0, os.path.join(REPO, d))
W = os.path.dirname(os.path.abspath(__file__))
import numpy as np

def windows_cached(name):
    fn = os.path.join(W, f"win_{name}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    import ladder2 as LL
    if name == "SLQA":
        out = list(LL.slqa_windows())
    else:
        out = list(LL.bipm_windows())
    pickle.dump(out, open(fn, "wb"))
    return out

def avg_rank(v):
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(avg_rank(a), avg_rank(b))[0, 1])

def mad(v):
    v = np.asarray(v, float)
    m = np.median(v)
    return 1.4826 * np.median(np.abs(v - m))

def fstat(c, e):
    c = np.asarray(c, float); e = np.asarray(e, float)
    c = c[~np.isnan(c)]; e = e[~np.isnan(e)]
    if len(c) < 5 or len(e) < 5:
        return 0.0, 0.0
    se = 1.2533 * math.sqrt(mad(c) ** 2 / len(c) + mad(e) ** 2 / len(e))
    d = np.median(e) - np.median(c)
    return abs(d) / max(se, 1e-9), d / max(se, 1e-9)

def nmed(a, axis=None):
    return np.nanmedian(a, axis=axis)

def series_np(R, A, B):
    """R: T×n；返回 {key: 数组}，含义同 distinguish.series。med 需 ≥2 个非 NaN。"""
    R = np.asarray(R, float); A = np.asarray(A, float); B = np.asarray(B, float)
    T, n = R.shape
    out = {}
    def med_row(r):
        r = r[~np.isnan(r)]
        return np.median(r) if len(r) >= 2 else np.nan
    for j in range(n):
        idx = [k for k in range(n) if k != j]
        pm = np.array([med_row(R[t, idx]) for t in range(T)])
        out[("peer", j)] = R[:, j] - pm
        out[("nodeA", j)] = R[:, j] - A
    c = np.array([med_row(R[t]) for t in range(T)])
    out[("consA", 0)] = c - A
    out[("consB", 0)] = c - B
    out[("AB", 0)] = A - B
    return out

def unpack(rows):
    return np.array([x[2] for x in rows], float), np.array([x[0] for x in rows], float), np.array([x[1] for x in rows], float)
