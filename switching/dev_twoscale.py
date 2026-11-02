"""开发（非验证）：两把秤自检在几类数据上的表现。
  应该不拒：① 5 个真实领域（chem、planet、density、interest、gdp）：节点 = LLM，A = 非 LLM 外部秤，B = LLM 记忆 / 记录秤（check_0），δ = 0 轮 + 每个 δ 投毒轮
            ② MNIST 联邦学习前提成立（两台服务器各 512 张互不重叠）
  应该拒：  ③ 奥斯陆（节点 PMS5003，A = TEOM，B = Fidas；B 同时也是评估真值，只在开发用）
            ④ EPA 6 站（节点 AQY/CNO/MAX，A = RAM，B = PAR；品牌之间也相关 → 预期漏报）
            ⑤ MNIST 联邦学习"全体共有偏差"：所有客户端整段只用 0–4 类的数据（诚实但一起偏），真值仍是总体梯度
"""
import datetime as dt, math, os, random, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("", "../llm", "../fl", "../senseur", "../epa"):
    sys.path.insert(0, os.path.join(HERE, d))
import torch
from twoscale import check
H = math.log10(1.25)

def real():
    cwd = os.getcwd()
    from invert import sets
    S = {n: (d, no, de) for n, d, no, a, sl, de in sets()}
    os.chdir(cwd)
    import bigtest, bigtest2, bigtest4, extscale as X
    doms = [("bigtest:chem", bigtest.items("chem"), X.rule_scale), ("bigtest:planet", bigtest.items("planet"), lambda i: X.phys_scale(i)[0]),
            ("bigtest4:density", bigtest4.items("density"), X.db_scale), ("bigtest2:interest", bigtest2.items("interest"), X.interest_scale), ("bigtest:gdp", bigtest.items("gdp"), X.imf_scale)]
    out = []
    for name, its, fn in doms:
        data, nodes, deltas = S[name]
        ext = fn(its)
        tru = {it["id"]: math.log10(it["truth"]) for it in its}
        idk = {k: it["id"] for k, it in enumerate(its)}
        for d in deltas:
            rows = [x for x in data[("R0", d)] if idk[x[4]] in ext and ext[idk[x[4]]] > 0]
            R = torch.stack([x[2] for x in rows])
            a = [math.log10(ext[idk[x[4]]]) - tru[idk[x[4]]] for x in rows]
            b = [x[1] for x in rows]
            h = math.log10(1 + max(d, 0.05)) / 2
            out.append((f"{name.split(':')[1]} δ={d}", check(R, a, b, h)))
    return out

def fl(shared=False):
    import flbench as F
    out = []
    for n, m in ((10, 512), (20, 512), (10, 128), (20, 128)):
        for seed in (1, 2):
            g = torch.Generator().manual_seed(90000 + seed + n + m)
            perm = torch.randperm(len(F.XTR), generator=g)
            pool = perm[:30000] if not shared else perm[:30000][F.YTR[perm[:30000]] < 5]
            A, Bs = perm[-20000:-20000 + m], perm[-10000:-10000 + m]
            big = perm[30000:40000]
            W = torch.zeros(785, 10)
            items = []
            for t in range(30):
                Ga, Gb = F.grads(W, F.XTR[A], F.YTR[A]), F.grads(W, F.XTR[Bs], F.YTR[Bs])
                scale = (Ga.std(0) / math.sqrt(F.B)).clamp(min=1e-8)
                cl = torch.stack([F.grads(W, F.XTR[i], F.YTR[i]).mean(0) for i in (pool[torch.randint(0, len(pool), (F.B,), generator=g)] for _ in range(n))])
                tr = F.grads(W, F.XTR[big], F.YTR[big]).mean(0)
                live = torch.nonzero((scale > 1e-6) & ((Ga != 0).float().mean(0) >= 0.5)).flatten()
                js = live[torch.randperm(len(live), generator=g)[:50]]
                for j in js.tolist():
                    items.append((t, ((Ga[:, j].mean() - tr[j]) / scale[j]).item(), ((Gb[:, j].mean() - tr[j]) / scale[j]).item(), (cl[:, j] - tr[j]) / scale[j]))
                W -= 0.5 * cl.median(0).values.view(785, 10)
            R = torch.stack([x[3] for x in items])
            out.append((f"n={n} m={m} s{seed}", check(R, [x[1] for x in items], [x[2] for x in items], 0.5, blocks=[x[0] for x in items])))
    return out

def oslo():
    import oslo as O
    D = O.load()
    us = O.present(D, [37])
    R = O.rows(D, [36, 37], us)
    X = torch.stack([r for _, _, r in R])
    X = X - X.median(0).values
    return [("沙盒 w36-37", check(X, [p for _, p, _ in R], [0.0] * len(R), H, blocks=[h[:10] for h, _, _ in R]))]

def epa():
    import epa as E
    S, Ref, _ = E.load()
    NODES, A, Bn = ["AQY", "CNO", "MAX"], "RAM", "PAR"
    out = []
    for loc in sorted(S):
        ts = sorted(S[loc]); t0 = dt.datetime.fromisoformat(ts[0])
        rows = []
        for t in ts:
            if (dt.datetime.fromisoformat(t) - t0).days >= 90: break
            v = S[loc][t]
            if Ref[loc].get(t, 0) >= 3 and all(m in v and st.median(v[m]) > 0 for m in NODES + [A, Bn]):
                tru = math.log10(Ref[loc][t])
                rows.append((t[:10], torch.tensor([math.log10(st.median(v[m])) - tru for m in NODES]), math.log10(st.median(v[A])) - tru, math.log10(st.median(v[Bn])) - tru))
        if len(rows) < 200: continue
        X = torch.stack([x[1] for x in rows]); X = X - X.median(0).values
        out.append((loc, check(X, [x[2] for x in rows], [x[3] for x in rows], H, blocks=[x[0] for x in rows])))
    return out

def show(tag, R):
    print(f"\n{tag}：拒答 {sum(r[1][0] for r in R)}/{len(R)}")
    for nm, (ref, s, lo, vi) in R:
        print(f"  {nm:22s} {'拒' if ref else '  '} σ_cm {s:.4f}  下界方差 {lo:+.5f}  v_ind {vi:.5f}")

show("① 真实 LLM 5 领域（应不拒）", real())
show("② 联邦学习前提成立（应不拒）", fl(False))
show("③ 奥斯陆（应拒）", oslo())
show("④ EPA（应拒，预期漏报）", epa())
show("⑤ 联邦学习全体共有偏差（应拒）", fl(True))
