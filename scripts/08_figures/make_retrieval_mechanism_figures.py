"""Build explanatory diagrams for the statutory-retrieval diagnostic paper.

The diagrams use values read from the canonical retrieval reports and overwrite
the overview and citation-intervention assets consumed by the manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "experiments" / "retrieval"
OUTPUT = ROOT / "papers" / "retrieval_reference" / "plots"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

INK = "#193044"
MUTED = "#5C6974"
GRID = "#DCE3E8"
NEUTRAL = "#EEF2F4"
BLUE = "#3E78A8"
BLUE_LIGHT = "#EAF2F8"
PURPLE = "#765FA2"
PURPLE_LIGHT = "#F0ECF7"
ORANGE = "#BC7420"
ORANGE_LIGHT = "#FBF1E3"
GREEN = "#2E7D5B"
GREEN_LIGHT = "#E8F4EE"
RED = "#C54B4B"
RED_LIGHT = "#FBEDEE"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, stem: str):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_canvas(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(ax, x, y, width, height, title, body, face, edge, title_size=8.8, body_size=7.3):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=0.65, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.plot([x + 0.018, x + 0.065], [y + height - 0.020] * 2,
            color=edge, linewidth=2.5, solid_capstyle="round")
    ax.text(
        x + 0.018, y + height - 0.048, title,
        ha="left", va="top", fontsize=title_size, fontweight="bold", color=INK,
    )
    ax.text(
        x + 0.018, y + height - 0.115, body,
        ha="left", va="top", fontsize=body_size, color=MUTED, linespacing=1.23,
    )


def arrow(ax, start, end, label=None, label_pos=None):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            linewidth=1.1, color="#70808A", shrinkA=4, shrinkB=5,
        )
    )
    if label:
        x, y = label_pos if label_pos else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(x, y, label, ha="center", va="bottom", fontsize=6.7, color=MUTED)


def diagnostic_overview():
    controls = load(RESULTS / "retrieval_diagnostic_controls.json")
    qa = load(RESULTS / "retrieval_qa_report.json")
    end_to_end = load(RESULTS / "citation_aware_end_to_end_report.json")

    paragraph = controls["retrieval_controls"]["paragraph_only"]["recall@5"]
    title_plus = controls["retrieval_controls"]["title_plus_paragraph"]["recall@5"]
    followup = load(RESULTS / "runs" / "followup_structural_controls_v1" / "summary.json")
    random_expected = followup["C2_section_random"]["exact_expected_recall@5_pct"]
    section_filter = controls["retrieval_controls"]["section_filter_plus_bm25"]["recall@5"]
    pool_mean = controls["candidate_reduction"]["matching_passages"]["mean"]
    gold_f1 = qa["results"][0]["f1_pct"]
    section_f1 = end_to_end["results"][3]["f1_pct"]
    assert [round(value, 2) for value in (paragraph, title_plus, random_expected, section_filter, gold_f1, section_f1)] == [22.67, 56.98, 66.72, 86.34, 87.37, 57.99]

    fig = plt.figure(figsize=(7.3, 7.5))
    top = fig.add_axes([0.035, 0.60, 0.93, 0.34])
    bottom = fig.add_axes([0.035, 0.065, 0.93, 0.44])
    set_canvas(top)
    set_canvas(bottom)

    top.text(0.0, 1.02, "(a) Operational question partition", fontsize=11.1, fontweight="bold", color=INK)

    box(
        top, 0.015, 0.55, 0.28, 0.29,
        "Illustrative question A",
        "Under Section 14, what is\nstated in clause (b)?",
        NEUTRAL, "#B8C4CC",
    )
    box(
        top, 0.36, 0.55, 0.22, 0.29,
        "Deterministic filter",
        "Remove recognised\nreferences and fixed\nquestion-frame words.",
        BLUE_LIGHT, BLUE,
    )
    box(
        top, 0.66, 0.55, 0.31, 0.29,
        "Zero-residual pattern",
        "Residual tokens: none\n177 of 344 questions",
        PURPLE_LIGHT, PURPLE,
    )
    arrow(top, (0.295, 0.695), (0.36, 0.695))
    arrow(top, (0.58, 0.695), (0.66, 0.695))

    box(
        top, 0.015, 0.12, 0.28, 0.29,
        "Illustrative question B",
        "What does Section 30A provide\nregarding application of section 19?",
        NEUTRAL, "#B8C4CC", body_size=6.8,
    )
    box(
        top, 0.36, 0.12, 0.22, 0.29,
        "Same filter",
        "Remove recognised\nreferences and fixed\nquestion-frame words.",
        BLUE_LIGHT, BLUE,
    )
    box(
        top, 0.66, 0.12, 0.31, 0.29,
        "Lexically content-bearing",
        "Residual tokens: application\n167 of 344 questions",
        GREEN_LIGHT, GREEN,
    )
    arrow(top, (0.295, 0.265), (0.36, 0.265))
    arrow(top, (0.58, 0.265), (0.66, 0.265))

    bottom.text(0.0, 1.03, "(b) Section filtering: what stays, what goes?", fontsize=11.1, fontweight="bold", color=INK)
    bottom.text(0.0, 0.94, 'Question: "Under Section 14, what is stated in clause (b)?"', fontsize=8.1, color=INK)
    box(bottom, 0.02, 0.33, 0.28, 0.45, "Search corpus", "", NEUTRAL, "#B8C4CC")
    rows = [("Act A  |  Section 14", True), ("Act B  |  Section 14", True),
            ("Act C  |  Section 14", True), ("Act A  |  Section 15", False),
            ("Act B  |  Section 17", False)]
    for i, (label, keep) in enumerate(rows):
        y = 0.65 - i * 0.064
        bottom.text(0.044, y, label, fontsize=7.5, color=GREEN if keep else MUTED, va="center")
        if not keep:
            bottom.plot([0.04, 0.276], [y, y], color=RED, lw=1.0)
    box(bottom, 0.37, 0.445, 0.19, 0.23, "Match section 14", "Use section titles;\nAct is not supplied.", PURPLE_LIGHT, PURPLE, title_size=8.0)
    box(bottom, 0.65, 0.33, 0.32, 0.45, "Keep matching passages", "Act A  |  Section 14\nAct B  |  Section 14\nAct C  |  Section 14\n\nThe Act is still ambiguous.", GREEN_LIGHT, GREEN, body_size=7.5)
    arrow(bottom, (0.315, 0.56), (0.355, 0.56))
    arrow(bottom, (0.575, 0.56), (0.635, 0.56))
    box(bottom, 0.33, 0.02, 0.30, 0.19, "Random ordering", "Shuffle the same pool; take up to 5.\nExpected R@5 = %.2f" % random_expected, ORANGE_LIGHT, ORANGE, body_size=7.0)
    box(bottom, 0.68, 0.02, 0.29, 0.19, "BM25 ranking", "Rank the same pool; take up to 5.\nR@5 = %.2f" % section_filter, BLUE_LIGHT, BLUE, body_size=7.0)
    arrow(bottom, (0.75, 0.315), (0.50, 0.225))
    arrow(bottom, (0.83, 0.315), (0.83, 0.225))
    bottom.text(0.02, 0.17, "No parsed section?\nUse full-corpus BM25.", fontsize=7.2, color=MUTED, va="top")

    save(fig, "01_retrieval_diagnostic_overview")
    print("01: diagnostic overview written")


def citation_intervention():
    ablation = load(RESULTS / "citation_ablation" / "citation_contribution_ablation.json")
    cue = ablation["citation-template-only"]["Citation-aware"]
    query_rows = [json.loads(line) for line in (RESULTS / "citation_ablation" / "citation_ablation_queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    example = next(row for row in query_rows if row["original"] == "Under Section 14, what is stated in clause (b)?")
    assert example["content"] == "What is stated?"
    assert example["citation"] == "Section 14"
    assert example["wrong_seed42"] == "Under Section 12, what is stated in clause (b)?"
    variants = (
        ("Original query", example["original"].replace(", what", ",\nwhat"), cue["original"]["R@5"], GREEN_LIGHT, GREEN),
        ("Reference removed", example["content"], cue["content"]["R@5"], RED_LIGHT, RED),
        ("Citation only", example["citation"], cue["citation"]["R@5"], PURPLE_LIGHT, PURPLE),
        ("Corrupted reference: 14 \u2192 12", example["wrong_seed42"].replace(", what", ",\nwhat"), cue["wrong"]["R@5"], ORANGE_LIGHT, ORANGE),
    )
    assert [round(row[2], 2) for row in variants] == [83.62, 1.13, 62.71, 0.00]

    fig = plt.figure(figsize=(7.3, 4.05))
    left = fig.add_axes([0.04, 0.15, 0.44, 0.75])
    right = fig.add_axes([0.56, 0.15, 0.40, 0.75])
    set_canvas(left)
    left.text(0.0, 1.055, "(a) Query-side intervention", fontsize=10.0, fontweight="bold", color=INK)
    ys = (0.77, 0.54, 0.31, 0.08)
    for (title, body, _, face, edge), y in zip(variants, ys):
        box(left, 0.015, y, 0.94, 0.19, title, body, face, edge, title_size=8.5, body_size=7.1)

    values = [row[2] for row in variants]
    colors = [row[4] for row in variants]
    y = [position + 0.095 for position in ys]
    right.barh(y, [100] * len(y), color="#F3F6F8", height=0.13, zorder=1)
    bars = right.barh(y, values, color=colors, linewidth=0, height=0.13, zorder=3)
    for bar, value in zip(bars, values):
        right.text(
            max(value + 1.6, 1.6), bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}", ha="left", va="center", fontsize=9.6, fontweight="bold", color=INK,
        )
    # The colour-matched query cards at left provide the row labels.  Keeping
    # duplicate y-axis labels here would overlap that panel in the manuscript.
    right.set_yticks([])
    right.set_ylim(0, 1)
    right.set_xlim(0, 100)
    right.set_xlabel("Section-boost recall@5 (%)", fontsize=9.2)
    right.text(0, 1.055, "(b) Retrieval effect (n=177)", transform=right.transAxes,
               fontsize=10.0, fontweight="bold", color=INK)
    right.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    right.set_axisbelow(True)
    right.tick_params(axis="x", labelsize=8.5, colors=INK)
    right.spines[["top", "right", "left"]].set_visible(False)
    right.spines["bottom"].set_color("#AAB4BB")
    save(fig, "23_citation_intervention")
    print("23: citation intervention written")


def main():
    diagnostic_overview()
    citation_intervention()
    print("Wrote 01_retrieval_diagnostic_overview and 23_citation_intervention as PDF and PNG.")


if __name__ == "__main__":
    main()
