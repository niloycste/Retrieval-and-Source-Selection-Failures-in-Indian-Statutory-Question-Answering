"""Create the retrieval paper's stage-wise diagnostic overview.

Every displayed value is loaded from a canonical result file. Assertions stop
the build if a stale experiment report is selected accidentally.
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


NAVY = "#16324F"
BLUE = "#2F6690"
PURPLE = "#6B5B95"
GREEN = "#2E7D5B"
INK = "#17212B"
MUTED = "#5C6974"
LINE = "#C8D1D8"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.2f}"


def stage_row(ax, y, step, rq, title, primary, secondary, colour, fill):
    x, width, height = 0.018, 0.964, 0.168
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fill, edgecolor=LINE, linewidth=0.9,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    badge = FancyBboxPatch(
        (x + 0.018, y + 0.028), 0.078, height - 0.056,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=colour, edgecolor=colour, linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(badge)
    ax.text(
        x + 0.057, y + height / 2, str(step),
        ha="center", va="center", transform=ax.transAxes,
        fontsize=12.5, fontweight="bold", color="white",
    )
    ax.text(
        x + 0.115, y + height - 0.047, rq,
        ha="left", va="center", transform=ax.transAxes,
        fontsize=9.3, fontweight="bold", color=colour,
    )
    ax.text(
        x + 0.115, y + height - 0.080, title,
        ha="left", va="center", transform=ax.transAxes,
        fontsize=10.7, fontweight="bold", color=INK, linespacing=0.95,
    )
    ax.text(
        x + 0.800, y + height - 0.070, primary,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=11.5, fontweight="bold", color=colour, linespacing=1.1,
    )
    ax.text(
        x + 0.115, y + 0.030, secondary,
        ha="left", va="center", transform=ax.transAxes,
        fontsize=8.8, color=MUTED, linespacing=1.02,
    )


def sequence_arrow(ax, upper_y, lower_y):
    ax.add_patch(
        FancyArrowPatch(
            (0.50, upper_y), (0.50, lower_y), arrowstyle="-|>",
            mutation_scale=12, linewidth=1.15, color="#6E7B85",
            transform=ax.transAxes, zorder=6,
        )
    )


def _legacy_overview():
    e2e = load(RESULTS / "citation_aware_end_to_end_report.json")
    rerank = load(RESULTS / "cross_encoder_rerank_report.json")
    citation = load(RESULTS / "citation_aware_retrieval_report.json")
    metadata = load(
        RESULTS / "runs" / "metadata_matched_end_to_end_v1" / "summary.json"
    )

    e2e_rows = {row["setting"]: row for row in e2e["results"]}
    gold_f1 = e2e_rows["Gold paragraph + E2'"]["f1_pct"]
    bm25_f1 = e2e_rows["BM25 top-5 + E2'"]["f1_pct"]
    filter_f1 = e2e_rows["BM25 + section filter top-5 + E2'"]["f1_pct"]
    bm25_r5 = citation["results"]["BM25"]["recall@5"]
    section_r5 = citation["results"]["BM25 + section filter"]["recall@5"]
    best_learned = max(row["recall@5"] for row in rerank["results"].values())
    title_r5 = metadata["retrieval_metrics_pct"]["recall@5"]

    template = e2e["question_type_results"]["citation-template-only"]
    template_n = template["n"]
    template_gold = template["results"]["Gold paragraph"]["f1_pct"]
    template_bm25 = template["results"]["BM25"]["f1_pct"]

    assert round(gold_f1, 2) == 87.37
    assert round(bm25_f1, 2) == 31.95
    assert round(bm25_r5, 2) == 22.67
    assert round(best_learned, 2) == 30.23
    assert round(title_r5, 2) == 56.98
    assert template_n == 177
    assert round(template_gold, 2) == 91.55
    assert round(template_bm25, 2) == 17.59
    assert round(section_r5, 2) == 86.34
    assert round(filter_f1, 2) == 57.99
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.4, 4.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965, "SEQUENCE OF INVESTIGATION",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=13.5, fontweight="bold", color=NAVY,
    )
    ax.text(
        0.5, 0.925,
        "Observed performance  |  failure localisation  |  structural mechanism check",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=8.6, color=MUTED,
    )

    ys = (0.695, 0.495, 0.295, 0.095)
    stage_row(
        ax, ys[0], 1, "RQ1 - READER", "Gold-context reading",
        f"{fmt(gold_f1)} F1",
        "Annotated paragraph supplied; fixed E2' reader",
        BLUE, "#F4F8FB",
    )
    stage_row(
        ax, ys[1], 2, "RQ1 - GAP", "Paragraph-only retrieval",
        f"{fmt(bm25_r5)} R@5\n{fmt(bm25_f1)} F1",
        f"Best tested paragraph-only neural pipeline: {fmt(best_learned)} R@5",
        "#A76316", "#FCF8F1",
    )
    stage_row(
        ax, ys[2], 3, "RQ2 - QUERY TYPE", "Failure localisation",
        f"{template_n} / 344",
        "Reference-dominant questions concentrate\n"
        "the retrieval failure",
        PURPLE, "#F8F5FA",
    )
    stage_row(
        ax, ys[3], 4, "RQ3 - STRUCTURE", "Reference resolution",
        f"{fmt(section_r5)} R@5\n{fmt(filter_f1)} F1",
        "Section-filtered candidates; 47 top-five misses remain",
        GREEN, "#F2F8F4",
    )
    for upper, lower in zip(ys[:-1], ys[1:]):
        sequence_arrow(ax, upper, lower + 0.168)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 350})):
        fig.savefig(
            OUTPUT / f"01_retrieval_diagnostic_overview.{suffix}",
            format=suffix, bbox_inches="tight", facecolor="white", **kwargs,
        )
    plt.close(fig)
    print("Wrote papers/retrieval_reference/plots/01_retrieval_diagnostic_overview.{pdf,png}")


def main():
    """Keep the historical command safe by routing to the current layout."""
    from make_retrieval_mechanism_figures import diagnostic_overview
    diagnostic_overview()


if __name__ == "__main__":
    main()
