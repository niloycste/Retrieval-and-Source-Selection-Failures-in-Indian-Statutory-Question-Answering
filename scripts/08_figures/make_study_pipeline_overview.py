"""Create the stage-wise diagnostic overview used in the journal manuscript.

The figure reports only values loaded from the canonical result files.  It is
intended to replace the older dataset/DAPT pipeline artwork, which predates the
canonical sliding-window reader evaluation and does not represent the paper's
retrieval-centred contribution.

Run from anywhere in the repository:

    python scripts/08_figures/make_study_pipeline_overview.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
QA_REPORT = ROOT / "results" / "evaluation" / "qa_eval_report.json"
RERANK_REPORT = (
    ROOT / "results" / "experiments" / "retrieval"
    / "cross_encoder_rerank_report.json"
)
CITATION_REPORT = (
    ROOT / "results" / "experiments" / "retrieval"
    / "citation_aware_retrieval_report.json"
)
END_TO_END_REPORT = (
    ROOT / "results" / "experiments" / "retrieval"
    / "citation_aware_end_to_end_report.json"
)
ABLATION_REPORT = (
    ROOT / "results" / "experiments" / "retrieval" / "citation_ablation"
    / "citation_contribution_ablation.json"
)
MULTISEED_REPORT = (
    ROOT / "results" / "experiments" / "size_matched_dapt"
    / "size_matched_multiseed_stats.json"
)

OUTPUT_DIRS = (
    ROOT / "combined_paper" / "plots",
    ROOT / "combined_paper" / "Springer_Nature_LaTeX_Template" / "plots",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def box(ax, xy, width, height, face, edge, text, *, fontsize=10.2,
        weight="normal", text_color="#17212b", radius=0.025, linewidth=1.1):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=fontsize, fontweight=weight, color=text_color,
        linespacing=1.28,
    )
    return patch


def arrow(ax, start, end, color="#59656f", linewidth=1.35):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        color=color, linewidth=linewidth,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def fmt(value):
    return f"{value:.2f}"


def main():
    qa = load_json(QA_REPORT)
    rerank = load_json(RERANK_REPORT)
    citation = load_json(CITATION_REPORT)
    end_to_end = load_json(END_TO_END_REPORT)
    ablation = load_json(ABLATION_REPORT)
    multiseed = load_json(MULTISEED_REPORT)

    readers = {row["code"]: row for row in qa["results_summary"]}
    e2_f1 = readers["E2'"]["f1"]
    dapt_spread = multiseed["f1"]["spread_of_means"]
    dapt_pooled_sd = multiseed["f1"]["pooled_sd"]
    dapt_p = multiseed["f1"]["one_way_anova"]["p"]
    best_learned_r5 = max(
        row["recall@5"] for name, row in rerank["results"].items()
        if name != "BM25"
    )
    bm25_r5 = citation["results"]["BM25"]["recall@5"]
    section_r5 = citation["results"]["BM25 + section filter"]["recall@5"]
    e2e = {row["setting"]: row for row in end_to_end["results"]}
    bm25_e2e_f1 = e2e["BM25 top-5 + E2'"]["f1_pct"]
    filter_e2e_f1 = e2e["BM25 + section filter top-5 + E2'"]["f1_pct"]

    template = ablation["citation-template-only"]["Citation-aware"]
    original_r5 = template["original"]["R@5"]
    deleted_r5 = template["content"]["R@5"]
    wrong_r5 = template["wrong"]["R@5"]

    # Guard against accidentally drawing a stale or structurally different run.
    assert round(e2_f1, 2) == 87.37
    assert round(dapt_spread, 2) == 0.72
    assert round(dapt_pooled_sd, 2) == 0.71
    assert round(dapt_p, 2) == 0.50
    assert round(best_learned_r5, 2) == 30.23
    assert round(section_r5, 2) == 86.34
    assert round(filter_e2e_f1, 2) == 57.99

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # The physical width is close to the journal text block. Short labels and
    # generous internal margins keep every line legible after LaTeX scaling.
    fig, ax = plt.subplots(figsize=(9.1, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#18232d"
    muted = "#52616d"
    neutral_edge = "#b6c2ca"
    colours = {
        "reading": ("#eaf3f9", "#2d6f95", "#d3e7f3"),
        "retrieval": ("#fcf4e8", "#a96819", "#f3e2c8"),
        "structure": ("#edf6ef", "#347344", "#d9ecdd"),
    }

    # Compact benchmark band.
    box(
        ax, (0.035, 0.875), 0.93, 0.085,
        "#f1f5f7", "#758591",
        "IndicIPR-QA  |  4,462 QA pairs  |  1,116 statutory paragraphs  |  "
        "7 Acts  |  Test n = 344",
        fontsize=10.7, weight="bold", radius=0.016, linewidth=1.0,
    )

    card_y = 0.245
    card_h = 0.565
    card_w = 0.285
    card_x = (0.035, 0.3575, 0.68)

    def card(x, stage, title, subtitle, colour_key, rows):
        surface, accent, header = colours[colour_key]
        # Quiet outer card and coloured title band.
        box(ax, (x, card_y), card_w, card_h, "#ffffff", neutral_edge, "",
            radius=0.022, linewidth=1.0)
        box(ax, (x + 0.012, card_y + card_h - 0.105), card_w - 0.024, 0.082,
            header, accent, f"{stage}   {title}", fontsize=10.4,
            weight="bold", radius=0.015, linewidth=1.0)

        ax.text(
            x + card_w / 2, card_y + card_h - 0.157, subtitle,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9.5, color=muted,
        )
        # A single flow stem replaces multiple crowded boxes.
        arrow(
            ax,
            (x + card_w / 2, card_y + card_h - 0.19),
            (x + card_w / 2, card_y + card_h - 0.235),
            color=accent, linewidth=1.25,
        )

        row_top = card_y + card_h - 0.285
        row_gap = 0.096
        for index, (label, value, emphasis) in enumerate(rows):
            cy = row_top - index * row_gap
            if emphasis:
                box(
                    ax, (x + 0.026, cy - 0.044), card_w - 0.052, 0.083,
                    surface, accent, "", fontsize=10.7, weight="bold",
                    radius=0.012, linewidth=1.0,
                )
                ax.text(
                    x + card_w / 2, cy + 0.015, label.upper(),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7.2, color=muted, fontweight="bold",
                )
                ax.text(
                    x + card_w / 2, cy - 0.017, value,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10.4, color=ink, fontweight="bold",
                )
            else:
                ax.text(
                    x + card_w / 2, cy + 0.015, label.upper(),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7.2, color=muted, fontweight="bold",
                )
                ax.text(
                    x + card_w / 2, cy - 0.017, value,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9.7, color=ink, fontweight="bold",
                )
                if index < len(rows) - 1:
                    ax.plot(
                        [x + 0.035, x + card_w - 0.035],
                        [cy - 0.047, cy - 0.047],
                        color="#dfe5e9", linewidth=0.7,
                        transform=ax.transAxes, clip_on=False,
                    )

    card(
        card_x[0], "RQ1", "READING", "Gold paragraph supplied",
        "reading",
        [
            ("Strongest reader", f"E2'  |  {fmt(e2_f1)} F1", True),
            ("Controlled DAPT mean spread", f"{fmt(dapt_spread)} F1", False),
            (
                "Pooled SD | one-way p",
                f"{fmt(dapt_pooled_sd)} F1  |  p = {fmt(dapt_p)}",
                False,
            ),
        ],
    )
    card(
        card_x[1], "RQ2", "CONTENT RETRIEVAL", "Question + full corpus; fixed E2'",
        "retrieval",
        [
            ("BM25", f"{fmt(bm25_r5)} R@5", False),
            ("Best learned pipeline", f"{fmt(best_learned_r5)} R@5", False),
            ("BM25 top-5 + E2'", f"{fmt(bm25_e2e_f1)} F1", True),
        ],
    )
    card(
        card_x[2], "RQ3", "REFERENCE RESOLUTION", "Parse section; filter candidates",
        "structure",
        [
            ("Section filter", f"{fmt(section_r5)} R@5", True),
            ("Section filter + E2'", f"{fmt(filter_e2e_f1)} F1", False),
            ("Diagnosed subset", "177 reference-dominant", False),
        ],
    )

    # Stage-to-stage arrows sit entirely in the gutters.
    arrow(ax, (0.325, 0.53), (0.352, 0.53), color="#4d5b66", linewidth=1.5)
    arrow(ax, (0.6475, 0.53), (0.675, 0.53), color="#4d5b66", linewidth=1.5)

    # Full-width mechanism strip uses four compact columns instead of one long
    # sentence, preserving legibility at journal-column scale.
    box(
        ax, (0.035, 0.075), 0.93, 0.105,
        "#f5f8f6", "#7ca187", "",
        radius=0.015, linewidth=1.0,
    )
    ax.text(
        0.175, 0.128,
        "REFERENCE CUE CHECK\nReference-dominant subset",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=8.0, color=muted, fontweight="bold", linespacing=1.25,
    )
    mechanism_metrics = (
        ("ORIGINAL", f"{fmt(original_r5)} R@5"),
        ("CITATION REMOVED", f"{fmt(deleted_r5)} R@5"),
        ("WRONG SECTION", f"{fmt(wrong_r5)} R@5"),
    )
    for xpos, (label, value) in zip((0.43, 0.655, 0.865), mechanism_metrics):
        ax.text(
            xpos, 0.145, label,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=7.1, color=muted, fontweight="bold",
        )
        ax.text(
            xpos, 0.111, value,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9.5, color=ink, fontweight="bold",
        )
    for xpos in (0.305, 0.54, 0.755):
        ax.plot(
            [xpos, xpos], [0.096, 0.159], color="#cbd8cf", linewidth=0.8,
            transform=ax.transAxes, clip_on=False,
        )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "22_study_pipeline_overview.pdf",
            format="pdf", bbox_inches="tight", facecolor="white",
        )
        fig.savefig(
            output_dir / "22_study_pipeline_overview.png",
            format="png", dpi=350, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)

    print("Wrote 22_study_pipeline_overview.{pdf,png}")
    print(
        f"E2'={e2_f1:.4f}; DAPT spread={dapt_spread:.3f}; "
        f"pooled SD={dapt_pooled_sd:.3f}; p={dapt_p:.4f}; "
        f"best learned R@5={best_learned_r5:.2f}"
    )
    print(f"section R@5={section_r5:.2f}; section-filter E2E F1={filter_e2e_f1:.4f}")


if __name__ == "__main__":
    main()
