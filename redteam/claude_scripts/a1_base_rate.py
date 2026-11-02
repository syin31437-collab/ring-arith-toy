"""基础率分解：S0（不注入）下 z>3 的原因。"""
from rtc import *
import json, collections
T=3.0
PHI = 2*(1-0.5*(1+math.erf(3/math.sqrt(2))))
rng = np.random.default_rng(0)
def perm_rate(c, e, block, N=100):
    """合并后时间打乱(block=1)或块打乱，保持序列内依赖≤block。"""
    x = np.concatenate([c, e]); nc = len(c)
    m = ~np.isnan(x)
    # 保留原始 NaN 位置的简单做法：去 NaN
    x = x[m]; nc2 = int(m[:nc].sum())
    hits = 0; zs = []
    for _ in range(N):
        if block == 1:
            y = rng.permutation(x)
        else:
            s = rng.integers(block)
            cuts = list(range(s, len(x), block))
            parts = np.split(x, cuts)
            order = rng.permutation(len(parts))
            y = np.concatenate([parts[i] for i in order])
        z = fstat(y[:nc2], y[nc2:])[1]
        zs.append(z)
    return np.array(zs)
res = {}
for name in ("SLQA", "BIPM"):
    W_ = windows_cached(name)
    print(f"\n===== {name}: {len(W_)} 窗；校准/评估长度", sorted({(len(c),len(e)) for c,e in W_})[:5], "节点数", len(W_[0][0][0][2]))
    z_all = collections.defaultdict(list)
    perm = {1: collections.defaultdict(list), 4: collections.defaultdict(list), 8: collections.defaultdict(list)}
    for wi,(cal,ev) in enumerate(W_):
        Rc,Ac,Bc = unpack(cal); Re,Ae,Be = unpack(ev)
        Sc, Se = series_np(Rc,Ac,Bc), series_np(Re,Ae,Be)
        for k in Sc:
            z = fstat(Sc[k],Se[k])[1]
            z_all[k[0]].append(z)
            for b in perm:
                perm[b][k[0]].extend(perm_rate(Sc[k],Se[k],b,N=60).tolist())
    print("序列类   条数  实际|z|>3   中位z²/0.455(λ)  z稳健SD | 打乱(1)|z|>3  λ | 块4 | 块8 ")
    for k in ("peer","nodeA","consA","consB","AB"):
        z = np.array(z_all[k]); 
        def rep(a):
            a=np.asarray(a); return (np.mean(np.abs(a)>T), np.median(a**2)/0.4549)
        r = rep(z); p1=rep(perm[1][k]); p4=rep(perm[4][k]); p8=rep(perm[8][k])
        print(f"{k:6s} {len(z):5d}  {r[0]:.3f}     {r[1]:6.2f}    SD_rob={mad(z):.2f} | {p1[0]:.3f} {p1[1]:.2f} | {p4[0]:.3f} {p4[1]:.2f} | {p8[0]:.3f} {p8[1]:.2f}")
    all_z = np.concatenate([z_all[k] for k in z_all])
    print("名义 |z|>3 = %.4f；全部序列 %.3f；全部 λ=%.2f" % (PHI, np.mean(np.abs(all_z)>T), np.median(all_z**2)/0.4549))
    print("分位 |z|: 50/75/90/95/99:", np.round(np.percentile(np.abs(all_z),[50,75,90,95,99]),2))
    res[name]={k:list(map(float,z_all[k])) for k in z_all}
json.dump(res,open(W+"/a1_z.json","w"))
