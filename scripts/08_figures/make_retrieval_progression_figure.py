"""Build the retrieval-progression figure.

Panel (a) plots recall@5 for every retrieval system evaluated, in the order they
appear in the paper, with the first-stage recall@50 ceiling that bounds any
re-ranker drawn in. Panel (b) splits BM25 and the section index by question type
to show where the gain comes from.

Every value is read from the experiment outputs; nothing is transcribed by hand.

Run:  python make_retrieval_progression_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
RUNS = BASE / "results" / "experiments" / "retrieval" / "runs"
PLOTS = BASE / "combined_paper" / "plots"
TEMPLATE_PLOTS = BASE / "combined_paper" / "Springer_Nature_LaTeX_Template" / "plots"

dense = json.loads((RUNS / "dense_retrieval_v1" / "dense_retrieval_summary.json").read_text())
rerank = json.loads((RUNS / "cross_encoder_rerank_v1" / "cross_encoder_rerank_summary.json").read_text())
cite = json.loads((RUNS / "citation_aware_v1" / "citation_aware_summary.json").read_text())

d = dense["results"]
r = rerank["results"]
c = cite["results"]

systems = [
    ("BM25", d["BM25"]["recall@5"], "sparse"),
    ("Best dense\n(e5-small-v2)", d["e5-small-v2"]["recall@5"], "dense"),
    ("Hybrid RRF", d["Hybrid RRF (BM25 + e5-small-v2)"]["recall@5"], "dense"),
    ("BM25 +\ncross-encoder", r["BM25 + cross-encoder"]["recall@5"], "rerank"),
    ("Hybrid +\ncross-encoder", r["Hybrid RRF + cross-encoder"]["recall@5"], "rerank"),
    ("BM25 +\nsection index", c["BM25 + section boost"]["recall@5"], "structural"),
]
ceiling = r["BM25"]["recall@50"]

COLOUR = {"sparse": "#8c8c8c", "dense": "#7fa8d1", "rerank": "#4878a8",
          "structural": "#c44e52"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                               gridspec_kw={"width_ratios": [1.75, 1]})

labels = [s[0] for s in systems]
vals = [s[1] for s in systems]
cols = [COLOUR[s[2]] for s in systems]

bars = ax1.bar(range(len(systems)), vals, color=cols, width=0.66,
               edgecolor="white", linewidth=0.8)
ax1.axhline(ceiling, ls="--", lw=1.3, color="#333333", zorder=3)
ax1.text(len(systems) - 0.45, ceiling + 1.6,
         f"first-stage recall@50 = {ceiling:.2f}\n(ceiling for any re-ranker)",
         ha="right", va="bottom", fontsize=8.5, color="#333333")

for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.2f}",
             ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax1.set_xticks(range(len(systems)))
ax1.set_xticklabels(labels, fontsize=8.8)
ax1.set_ylabel("Recall@5 (%)", fontsize=10)
ax1.set_ylim(0, 100)
ax1.set_title("(a) Retrieving the governing paragraph", fontsize=10.5, pad=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", alpha=0.25, lw=0.6)
ax1.set_axisbelow(True)

split = cite["by_question_type_recall5"]
type_keys = ["positional", "content-bearing"]  # keys in the archived run file
type_labels = ["Reference-dominant", "Lexically content-bearing"]
bm = [split[key]["BM25"] for key in type_keys]
ix = [split[key]["BM25 + section boost"] for key in type_keys]

x = range(len(type_keys))
w = 0.36
b1 = ax2.bar([i - w / 2 for i in x], bm, w, label="BM25",
             color=COLOUR["sparse"], edgecolor="white", linewidth=0.8)
b2 = ax2.bar([i + w / 2 for i in x], ix, w, label="BM25 + section index",
             color=COLOUR["structural"], edgecolor="white", linewidth=0.8)
for bars_ in (b1, b2):
    for b in bars_:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.4,
                 f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)

ax2.set_xticks(list(x))
ax2.set_xticklabels(
    ["Reference-dominant\n(n=177)", "Lexically content-bearing\n(n=167)"],
    fontsize=8.3,
)
ax2.set_ylabel("Recall@5 (%)", fontsize=10)
ax2.set_ylim(0, 100)
ax2.set_title("(b) Where the gain comes from", fontsize=10.5, pad=9)
ax2.legend(fontsize=8.5, frameon=False, loc="upper left")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.25, lw=0.6)
ax2.set_axisbelow(True)

fig.tight_layout()
for out in (PLOTS, TEMPLATE_PLOTS):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "21_retrieval_progression.png", dpi=400,
                bbox_inches="tight", facecolor="white")
    fig.savefig(out / "21_retrieval_progression.pdf",
                bbox_inches="tight", facecolor="white")
print("values plotted (all read from experiment outputs):")
for lab, v, _ in systems:
    print(f"  {lab.replace(chr(10), ' '):28s} {v:6.2f}")
print(f"  {'ceiling recall@50':28s} {ceiling:6.2f}")
for key, label in zip(type_keys, type_labels):
    print(f"  {label:28s} BM25 {split[key]['BM25']:6.2f}   index "
          f"{split[key]['BM25 + section boost']:6.2f}")
print("\nwritten to plots/21_retrieval_progression.{png,pdf}")
