"""开发（非验证）：两把秤前提自检，MNIST 联邦学习（新开发种子 9000+，Fashion-MNIST 不碰）。统计拒答和预测误差的关系。"""
import itertools, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../fl"))
import flbench as F
from hetero import measure
from analytic import sigma_c
from law_v6 import predict
from premise2 import decide2
kw = {"dense": 0.5, "v2": True}
R = []
groups = [("前提成立", (0.2, 0.4, 0.6), ["ALIE0.5", "ALIE1", "ALIE2", "ALIE4", "noise", "flip"]),
          ("全员投毒 f=1", (1.0,), ["ALIE1", "ALIE2", "ALIE4", "noise", "flip"]),
          ("秤 A 同源 ζ=1", (0.4, 0.6), ["flip-ζ", "ALIE-ζ"])]
k = 0
for tag, fs, attacks in groups:
    for n, f, a, m in itertools.product((10, 20), fs, attacks, (32, 512)):
        k += 1
        sec = []
        clean, bad, _ = F.run(n, f, a, m, 9000 + k, second=sec, **kw)
        bc, be = [i // F.K for i in range(len(clean))], [i // F.K for i in range(len(bad))]
        bv_c, bv_e = [x[1] for x in sec if not x[0]], [x[1] for x in sec if x[0]]
        s0 = max(sigma_c(clean)[1], 1e-6)
        d = decide2(clean, bad, bv_c, bv_e, bad[0][3] / 2, s0, "软", bc, be)
        act, pt = measure(clean + bad, bad[0][3], "软", s0)["miss"], predict(clean, bad, "软", s0)
        R.append((tag, a, n, f, m, d, act, pt))
for tag, *_ in groups:
    s = [r for r in R if r[0] == tag]
    print(f"\n{tag}（{len(s)} 配置）：拒答 {st.mean(r[5]['refuse'] for r in s):.0%}；触发 换秤 {sum(r[5]['d_scale'] > 0.15 for r in s)} 共有误差 {sum(r[5]['d_cm'] > 0.15 for r in s)} 单秤 {sum(r[5]['m1'] for r in s)}")
    for a in sorted({r[1] for r in s}):
        t = [r for r in s if r[1] == a]
        print(f"  {a:8s} 拒答 {st.mean(r[5]['refuse'] for r in t):.0%}  实测 {st.mean(r[6] for r in t):.2f} 预测 {st.mean(r[7] for r in t):.2f} |误差| {st.mean(abs(r[6] - r[7]) for r in t):.3f}")
ok = [r for r in R if not r[5]["refuse"]]
no = [r for r in R if r[5]["refuse"]]
print(f"\n全部 {len(R)} 配置：没拒的 |误差| {st.mean(abs(r[6] - r[7]) for r in ok):.3f}（{len(ok)}）；拒了的 |误差| {st.mean(abs(r[6] - r[7]) for r in no) if no else float('nan'):.3f}（{len(no)}）")
big = [r for r in R if abs(r[6] - r[7]) > 0.15]
print(f"误差 > 0.15 的 {len(big)} 配置里拒了 {st.mean(r[5]['refuse'] for r in big) if big else float('nan'):.0%}；误差 ≤ 0.15 的 {len(R) - len(big)} 配置里误拒 {st.mean(r[5]['refuse'] for r in R if abs(r[6] - r[7]) <= 0.15):.0%}")
