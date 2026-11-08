"""Figure: keep under the false note vs rescue under the true note, one panel per form.
Data: rows_s.json (297-item round, with echo) and rows_sm.json (control round, one1).
Run: ~/.venvs/onetake/bin/python fig_forms.py  -> fig_forms.pdf
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
NODES = ["ds_flash_a", "ds_flash_b", "ds_pro_a", "codex_ns", "qwen14", "llama8"]
SHORT = {"ds_flash_a": "Flash-A", "ds_flash_b": "Flash-B", "ds_pro_a": "Pro", "codex_ns": "Codex", "qwen14": "Qwen", "llama8": "Llama"}
FORMS = [("one1", "one1: “the correct answer is B.”", "rows_sm.json"),
         ("set2", "set2: “…one of: B, D.”", "rows_s.json"),
         ("cond2", "cond2: set2 + exit clause", "rows_s.json"),
         ("L3", "L3: premise", "rows_s.json")]
# which points get a direct label (the ones that stand apart from the cluster)
LABEL = {"one1": ["codex_ns"], "set2": ["codex_ns", "llama8", "ds_pro_a"], "cond2": ["ds_pro_a", "qwen14"], "L3": ["qwen14", "codex_ns"]}

BLUE, INK, MUTED, GRID, AXIS = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
m = lambda v: sum(v) / len(v)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})

rows = {f: json.load(open(os.path.join(REL, f))) for f in {"rows_s.json", "rows_sm.json"}}
fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.95), sharex=True, sharey=True)
for ax, (p, title, src) in zip(axes, FORMS):
    R = rows[src]
    pts = {}
    for a in NODES:
        Kf = m([r["ok"] for r in R if r["a"] == a and r["p"] == p + "f" and r["right"]])
        Bt = m([r["ok"] for r in R if r["a"] == a and r["p"] == p + "t" and not r["right"]])
        pts[a] = (Kf, Bt)
    ax.scatter([v[0] for v in pts.values()], [v[1] for v in pts.values()], s=26, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    for a in LABEL[p]:
        x, y = pts[a]
        dx, dy = (-0.04, 0.0) if x > 0.6 else (0.04, 0.0)
        if p == "L3" and a == "codex_ns":
            dx, dy = -0.02, 0.13
        ax.annotate(SHORT[a], (x, y), xytext=(x + dx, y + dy), ha="right" if x > 0.6 else "left", va="center", fontsize=6.5, color=MUTED, zorder=4)
    ax.set_title(title, fontsize=7.5, pad=4, color=INK)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.set_xticklabels(["0", ".5", "1"]); ax.set_yticklabels(["0", ".5", "1"])
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(color=AXIS, labelcolor=MUTED, length=2, width=0.5)
    ax.set_aspect("equal")
axes[0].set_ylabel("rescue under true note  $B$", color=INK)
axes[0].annotate("ideal", (1.0, 1.0), xytext=(0.93, 0.9), ha="right", va="top", fontsize=6.5, color=MUTED, style="italic")
fig.supxlabel("keep under false note  $K$", fontsize=8, color=INK, y=0.02)
fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.21, wspace=0.12)
fig.savefig(os.path.join(HERE, "fig_forms.pdf"))
print("wrote fig_forms.pdf")
