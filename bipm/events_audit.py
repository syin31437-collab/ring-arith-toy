"""第 3 步复查（非预注册）：只保留实验室/BIPM 明确记录的操作事件，去掉 "apparent"（可能是从同一份数据看出来的）和泛泛提到 MJD 的 other。"""
import collections, random, re, statistics as st
import bipm as B, events as E
def parse(filter_):
    ev = collections.defaultdict(list); cur = None
    for line in open("data/notes.utc", errors="ignore"):
        m = re.match(r"\s{3,6}([A-Z][A-Z0-9]{1,4})\s*\.", line)
        if m: cur = m.group(1).lower()
        if cur is None: continue
        low = line.lower()
        if not filter_(low): continue
        for mj in re.findall(r"MJD[^0-9]{0,8}(\d{5})", line)[:1]:
            ev[cur].append((int(mj), "x", None))
    return ev
W_, keep = E.windows()
rg = random.Random(1)
for nm, f in (("全部", lambda l: True), ("去掉 apparent", lambda l: "apparent" not in l),
              ("只留明确操作（换钟 / 频率 / 明确 time step，且非 apparent）", lambda l: "apparent" not in l and any(k in l for k in ("master clock", "frequency step", "time step", "replaced", "changed", "change of")))):
    ev = parse(f); lab = {k: [e for e in ev.get(k, []) if e[0] >= B.SANDBOX_END] for k in keep}
    n = sum(len(v) for v in lab.values())
    for idx, d in ((0, "EWMA"), (1, "单点")):
        pa, pb, lf, na, nb = E.lift(W_, lab, idx, "L2")
        labs = list(lab); perm = []
        for _ in range(1000):
            sh = labs[:]; rg.shuffle(sh)
            perm.append(E.lift(W_, dict(zip(labs, (lab[s] for s in sh))), idx, "L2")[2])
        print(f"{nm}（{n} 事件）{d}：报警窗 {pa:.0%}（{na}），没报警 {pb:.0%}，提升度 {lf:.2f}，p = {st.mean(x >= lf for x in perm):.3f}")
