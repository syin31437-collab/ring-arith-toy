"""通用性检验的事后诊断（未预注册，不改判据、不改结论，只解释失败在哪）。
① C2 输在哪：去掉 tiny_qwen 的差距；对手选了哪些特征。② M3 只看 LLM。③ 留一家族：把同家族节点全部从总体里拿掉，避免"双胞胎"帮忙。"""
import collections
import random

import torch

import formula as FM
import generality as G
import stage2 as S2

NAMES = ["原答案", "原×置信"]
for f in ("L2", "L3", "L4", "集", "集+理"):
    NAMES += [f"目标×{f}", f"目标×{f}×假", f"原×{f}", f"原×{f}×假"]
NAMES += ["目标", "目标×假", "原×假", "原×有投放", "目标×置信", "原×原在目标内", "目标×原在目标内", "偏A", "偏B", "偏C", "偏D"]
for t in G.TASKS:
    NAMES += [f"原×{t}", f"目标×{t}"]

print("一、对手前向选择选中的特征（15 节点 × 2 折，次数）")
cnt = collections.Counter()
for a in G.NODES:
    A, B = G.folds(a)
    for d in (A, B):
        cnt.update(G.forward(G.pool(d), d["y"])[1:])
print("  " + "，".join(f"{NAMES[k]} {v}" for k, v in cnt.most_common(14)))

FAM = {"flash_short": "flash", "flash_hot": "flash", "flash_m50": "flash", "flash_m80": "flash", "ds_flash_a": "flash", "ds_flash_b": "flash", "ds_pro_a": "pro", "claude_haiku": "haiku", "codex": "codex", "codex_ns": "codex", "tiny_qwen": "tiny", "lex": "lex", "lex_proj": "lex", "emb": "emb", "rand_proj": "rand"}
print("\n二、留一家族（训练总体里不含同家族任何节点）：预测 log-loss")
print(f"{'节点':14s}{'F4':>8s}{'PCA5':>8s}{'共用':>8s}{'最近邻':>8s} ｜ 关键格误差 F4 / PCA5 / 最近邻")
full3 = {a: FM.fit(FM.f3, 8, G.tens(G.BY[a])) for a in G.NODES}
S = collections.defaultdict(list)
cellsum = collections.defaultdict(list)
for a in G.NODES:
    train = [n for n in G.NODES if FAM[n] != FAM[a]]
    mu, Sig, phi = G.fit_pop(train)
    cal, test = G.split(a)
    dc, dt = G.tens(cal), G.tens(test)
    fn = lambda t, d: G.f4(t, phi, d)
    th = G.map_fit(dc, mu, Sig, fn, 5)
    qs = {"F4": fn(th, dt), "共用": fn(mu, dt)}
    B = torch.stack([G.fit_lin(G.basis(G.tens(G.BY[n])), G.tens(G.BY[n])["y"]) for n in train])
    m0 = B.mean(0)
    U, Sv, Vh = torch.linalg.svd(B - m0, full_matrices=False)
    k = min(5, len(Sv))
    Vk, sk = Vh[:k], Sv[:k] ** 2 / max(len(train) - 1, 1) + .1
    Pc, Pt = G.basis(dc), G.basis(dt)
    z = torch.zeros(k, requires_grad=True)
    opt = torch.optim.LBFGS([z], max_iter=200, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(Pc @ (m0 + z @ Vk), dc["y"]) + .5 * (z ** 2 / sk).sum() / len(dc["y"])
        l.backward()
        return l
    opt.step(cl)
    qs["PCA5"] = torch.softmax(Pt @ (m0 + z.detach() @ Vk), 1)
    best = min(train, key=lambda n: FM.nll(FM.f3, full3[n], dc))
    qs["最近邻"] = FM.f3(full3[best], dt)
    r = {m: (-q.clamp(min=1e-6).log()[torch.arange(len(test)), dt["y"]]).mean().item() for m, q in qs.items()}
    for m in r:
        S[m].append(r[m])
    for m in ("F4", "PCA5", "最近邻"):
        cellsum[m].append(G.cell_err(qs[m], test))
    print(f"{a:14s}" + "".join(f"{r[m]:8.3f}" for m in ("F4", "PCA5", "共用", "最近邻")) + f" ｜ {cellsum['F4'][-1]:.3f} / {cellsum['PCA5'][-1]:.3f} / {cellsum['最近邻'][-1]:.3f}  近邻 {best}", flush=True)
n = len(G.NODES)
idx_notiny = [i for i, a in enumerate(G.NODES) if a != "tiny_qwen"]
for m in ("F4", "PCA5", "共用", "最近邻"):
    print(f"  {m}: 平均 {sum(S[m]) / n:.3f}，去掉 tiny_qwen {sum(S[m][i] for i in idx_notiny) / (n - 1):.3f}" + (f"；关键格误差 {sum(cellsum[m]) / n:.3f}" if m != "共用" else ""))
print(f"  F4 赢 PCA5 {sum(S['F4'][i] < S['PCA5'][i] for i in range(n))}/{n}；赢最近邻 {sum(S['F4'][i] < S['最近邻'][i] for i in range(n))}/{n}")
print("\n三、M3 只看 LLM 节点（暴露量 / 校准 v，来自 lono 输出）")
pairs = {"flash_short": (.23, .42), "flash_hot": (.22, .23), "flash_m50": (.04, -.02), "flash_m80": (.02, -.11), "tiny_qwen": (.01, -.24), "ds_flash_a": (.12, .51), "ds_pro_a": (.18, .10), "claude_haiku": (.37, .38), "codex_ns": (.68, .50), "ds_flash_b": (.15, .46), "codex": (.65, .48)}
print(f"  Spearman {S2.spearman([v[1] for v in pairs.values()], [v[0] for v in pairs.values()]):.2f}（{len(pairs)} 个节点）")
c2 = {"flash_short": (.551, .547), "flash_hot": (.754, .749), "flash_m50": (.538, .519), "flash_m80": (.426, .382), "emb": (.661, .628), "lex": (.710, .695), "lex_proj": (.621, .604), "rand_proj": (.120, .119), "ds_flash_a": (.457, .441), "ds_pro_a": (.413, .399), "claude_haiku": (.229, .235), "codex_ns": (.216, .197), "ds_flash_b": (.478, .483), "codex": (.157, .134)}
print(f"\n四、C2 去掉 tiny_qwen：F3 {sum(v[0] for v in c2.values()) / 14:.4f} 前向 {sum(v[1] for v in c2.values()) / 14:.4f} 差 {sum(v[0] - v[1] for v in c2.values()) / 14:+.4f}")
