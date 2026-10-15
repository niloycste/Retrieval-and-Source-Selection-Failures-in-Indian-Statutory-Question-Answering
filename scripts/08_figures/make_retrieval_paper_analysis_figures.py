"""Create the retrieval-only paper's analytical figures.

The script reads canonical reports, the frozen follow-up control summary, and
the released per-Act diagnostic CSV; it asserts all plotted headline values and
writes publication-ready PDF and PNG versions to
papers/retrieval_reference/plots.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "experiments" / "retrieval"
DIFFICULTY = ROOT / "benchmark_analysis" / "outputs" / "difficulty_scores_E2'.csv"
OUTPUT = ROOT / "papers" / "retrieval_reference" / "plots"

INK = "#17212B"
MUTED = "#5C6974"
GRID = "#DCE3E8"
GREY = "#87939D"
BLUE = "#2F6690"
BLUE_LIGHT = "#72A5C8"
GREEN = "#2E7D5B"
PURPLE = "#745C97"

ACT_ORDER = (
    "design_act",
    "trademark_act",
    "semiconductor_layout_design_act",
    "patent_act",
    "copyright_act",
    "gi_act",
    "plant_varieties_farmers_rights_act",
)
ACT_LABELS = {
    "copyright_act": "Copyright",
    "design_act": "Designs",
    "gi_act": "GI",
    "patent_act": "Patents",
    "plant_varieties_farmers_rights_act": "Plant Varieties",
    "semiconductor_layout_design_act": "Semiconductor",
    "trademark_act": "Trade Marks",
}
CSV_TO_KEY = {
    "Copyright": "copyright_act",
    "Designs": "design_act",
    "GI": "gi_act",
    "Patents": "patent_act",
    "Plant Var.": "plant_varieties_farmers_rights_act",
    "Semicond.": "semiconductor_layout_design_act",
    "Trade Marks": "trademark_act",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, stem: str):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_labels(ax, xs, ys, offsets):
    for x, y, key in zip(xs, ys, ACT_ORDER):
        dx, dy, align = offsets[key]
        ax.annotate(
            ACT_LABELS[key], xy=(x, y), xytext=(dx, dy),
            textcoords="offset points", ha=align, va="center",
            fontsize=8.8, color=INK,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
            arrowprops=dict(
                arrowstyle="-", color="black", linewidth=0.75,
                shrinkA=2, shrinkB=4, connectionstyle="arc3,rad=0.0",
            ),
            zorder=5,
        )


def retrieval_reading_tension():
    dense = load(RESULTS / "dense_retrieval_report.json")
    bm25_map = dense["per_act_recall_at_5"]["BM25"]
    hybrid_map = dense["per_act_recall_at_5"]["Hybrid RRF (BM25 + e5-small-v2)"]

    overlap, reader = {}, {}
    with DIFFICULTY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = CSV_TO_KEY[row["Act"].strip()]
            overlap[key] = float(row["Q-Ctx overlap"])
            reader[key] = float(row["Test F1"])

    x = np.array([overlap[k] for k in ACT_ORDER])
    bm25 = np.array([bm25_map[k] for k in ACT_ORDER])
    hybrid = np.array([hybrid_map[k] for k in ACT_ORDER])
    reader_f1 = np.array([reader[k] for k in ACT_ORDER])
    r_ret = np.corrcoef(x, hybrid)[0, 1]
    r_read = np.corrcoef(x, reader_f1)[0, 1]

    assert round(r_ret, 2) == 0.93
    assert round(r_read, 3) == -0.785
    assert round(reader_f1[0], 2) == 98.97
    assert round(hybrid[0], 2) == 0.00

    red = "#C7392E"
    blue = "#1767B1"
    fig, axes = plt.subplots(2, 1, figsize=(6.25, 7.55))

    def fit(ax, y, colour, label=None):
        slope, intercept = np.polyfit(x, y, 1)
        xfit = np.linspace(0.31, 0.645, 100)
        ax.plot(
            xfit, slope * xfit + intercept, linestyle=(0, (5, 3)),
            color=colour, linewidth=1.3, alpha=0.72, label=label, zorder=1,
        )

    ax = axes[0]
    fit(ax, hybrid, red, label="Hybrid RRF linear fit")
    for xi, lo, hi in zip(x, bm25, hybrid):
        ax.plot([xi, xi], [lo, hi], color="#CDD5DA", linewidth=0.8, zorder=1)
    ax.scatter(
        x, bm25, s=44, marker="o", facecolors="#F3F6F7", edgecolors="#637578",
        linewidths=1.2, label="BM25", zorder=3,
    )
    ax.scatter(
        x, hybrid, s=48, marker="o", color=red, edgecolors="white",
        linewidths=0.7, label="Hybrid RRF", zorder=4,
    )
    offsets_a = {
        "design_act": (14, 8, "left"),
        "trademark_act": (-18, 22, "right"),
        "semiconductor_layout_design_act": (20, -12, "left"),
        "patent_act": (30, 0, "left"),
        "copyright_act": (30, 10, "left"),
        "gi_act": (18, 10, "left"),
        "plant_varieties_farmers_rights_act": (-10, 18, "right"),
    }
    annotate_labels(ax, x, hybrid, offsets_a)
    ax.set_title(f"(a) Overlap is associated with retrieval  (r = {r_ret:+.2f})", fontsize=11.2, color=INK, pad=8)
    ax.set_ylabel("Retrieval recall@5 (%)", fontsize=10.1)
    ax.legend(loc="upper left", frameon=False, fontsize=9.1, handletextpad=0.5)

    ax = axes[1]
    fit(ax, reader_f1, blue)
    ax.scatter(
        x, reader_f1, s=48, marker="o", color=blue, edgecolors="white",
        linewidths=0.7, zorder=4,
    )
    offsets_b = {
        "design_act": (14, 12, "left"),
        "trademark_act": (-16, -18, "right"),
        "semiconductor_layout_design_act": (28, -6, "left"),
        "patent_act": (28, 12, "left"),
        "copyright_act": (-16, -14, "right"),
        "gi_act": (22, 12, "left"),
        "plant_varieties_farmers_rights_act": (-12, 8, "right"),
    }
    annotate_labels(ax, x, reader_f1, offsets_b)
    ax.set_title(f"(b) Overlap and gold-context reading performance  (r = {r_read:+.3f})", fontsize=11.2, color=INK, pad=8)
    ax.set_ylabel("Gold-context reader F1 (%)", fontsize=10.1)

    for ax in axes:
        ax.set_xlabel("Question-context lexical overlap", fontsize=10.1)
        ax.set_xlim(0.303, 0.648)
        ax.tick_params(axis="both", labelsize=9.0, colors=INK)
        ax.grid(True, color=GRID, linestyle=(0, (1, 2)), linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AAB4BB")
    axes[0].set_ylim(-2, 60)
    axes[1].set_ylim(58, 105)

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.08, top=0.96, hspace=0.34)
    save(fig, "20_retrieval_reading_tension")
    print(f"20: overlap-hybrid r={r_ret:+.3f}; overlap-reader r={r_read:+.3f}")


def retrieval_progression():
    dense = load(RESULTS / "dense_retrieval_report.json")
    modern = load(
        RESULTS / "runs" / "modern_bge_extension_v1" / "indic" / "summary.json"
    )
    citation = load(RESULTS / "citation_aware_retrieval_report.json")
    controls = load(RESULTS / "retrieval_diagnostic_controls.json")
    followup = load(
        RESULTS / "runs" / "followup_structural_controls_v1" / "summary.json"
    )
    section_extension = load(
        RESULTS / "runs" / "section_filter_act_extension_v2" / "summary.json"
    )

    systems = (
        ("Paragraph BM25", dense["results"]["BM25"]["recall@5"], "sparse", "o"),
        (
            "BM25 + BGE-M3 RRF",
            modern["results"]["rrf_bm25_bge_m3"]["recall@5"],
            "rerank",
            "s",
        ),
        ("Title + paragraph BM25", controls["retrieval_controls"]["title_plus_paragraph"]["recall@5"], "metadata", "D"),
        (
            "Section pool, random",
            followup["C2_section_random"]["exact_expected_recall@5_pct"],
            "candidate",
            "D",
        ),
        ("Section filter + BM25", controls["retrieval_controls"]["section_filter_plus_bm25"]["recall@5"], "structure", "P"),
    )
    expected = [22.67, 30.81, 56.98, 66.72, 86.34]
    assert [round(row[1], 2) for row in systems] == expected

    colours = {
        "sparse": "#8C9398",
        "rerank": "#4D82B1",
        "metadata": "#D4A72C",
        "candidate": "#DD8452",
        "structure": "#C44E52",
    }
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 7.0), gridspec_kw={"height_ratios": [1.65, 1.0]},
    )

    y1 = np.arange(len(systems))
    labels = [
        "Paragraph BM25",
        "BM25 + BGE-M3 RRF",
        "Title + paragraph BM25",
        "Section pool, random order",
        "Section filter + BM25",
    ]
    values = [row[1] for row in systems]
    bar_colours = [colours[row[2]] for row in systems]
    bars = ax1.barh(
        y1, values, color=bar_colours, height=0.62,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    random_index = 3
    random_interval = section_extension["test"]["section_random_simulation"]["random_order_95_range"]
    for index, (bar, value) in enumerate(zip(bars, values)):
        inside = value > 80
        x_pos = value - 1.8 if inside else value + 1.2
        if index == random_index:
            inside = False
            x_pos = random_interval[1] + 1.4
        ax1.text(
            x_pos, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
            ha="right" if inside else "left", va="center",
            fontsize=10.2, fontweight="bold",
            color="white" if inside else INK, zorder=5,
        )
    ax1.errorbar(
        values[random_index],
        random_index,
        xerr=[[values[random_index] - random_interval[0]], [random_interval[1] - values[random_index]]],
        fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=6,
    )
    ax1.text(
        values[random_index] + 4.8, random_index + 0.27,
        "95% simulation interval", ha="left", va="center",
        fontsize=8.8, color=MUTED,
    )
    ax1.set_yticks(y1)
    ax1.set_yticklabels(labels, fontsize=9.8)
    ax1.set_xlabel("Recall@5 (%)", fontsize=10.8)
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    ax1.set_title("(a) Candidate access and ordering", fontsize=11.4, pad=9)

    split = citation["by_question_type_recall5"]
    type_keys = ["citation-template-only", "lexically-content-bearing"]
    type_labels = ["Zero-residual pattern\n(n=177)", "Lexically content-bearing\n(n=167)"]
    bm25_vals = [split[key]["BM25"] for key in type_keys]
    index_vals = [split[key]["BM25 + section boost"] for key in type_keys]
    expected_bm25 = [0.00, 46.71]
    expected_index = [83.62, 89.22]
    assert [round(v, 2) for v in bm25_vals] == expected_bm25
    assert [round(v, 2) for v in index_vals] == expected_index

    y2 = np.arange(len(type_keys))
    height = 0.34
    bm25_bars = ax2.barh(
        y2 - height / 2, bm25_vals, height, label="BM25",
        color=colours["sparse"], edgecolor="white", linewidth=0.8, zorder=3,
    )
    index_bars = ax2.barh(
        y2 + height / 2, index_vals, height, label="Section-based retrieval",
        color=colours["structure"], edgecolor="white", linewidth=0.8, zorder=3,
    )
    for bar_group in (bm25_bars, index_bars):
        for bar in bar_group:
            value = bar.get_width()
            inside = value > 80
            x_pos = value - 1.8 if inside else value + 1.2
            ax2.text(
                x_pos, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                ha="right" if inside else "left", va="center",
                fontsize=9.8, fontweight="bold",
                color="white" if inside else INK, zorder=5,
            )
    ax2.set_yticks(y2)
    ax2.set_yticklabels(type_labels, fontsize=9.8)
    ax2.set_xlabel("Recall@5 (%)", fontsize=10.8)
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    ax2.set_title("(b) Results by operational question type", fontsize=11.4, pad=9)
    ax2.legend(
        fontsize=9.2, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.33), ncol=2,
    )

    for ax in (ax1, ax2):
        ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=9.4, colors=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AAB4BB")

    fig.subplots_adjust(left=0.39, right=0.975, bottom=0.12, top=0.95, hspace=0.42)
    save(fig, "21_retrieval_progression")
    print("21:", ", ".join(f"{row[0]}={row[1]:.2f}" for row in systems))


def stagewise_outcomes():
    controls = load(RESULTS / "retrieval_diagnostic_controls.json")
    metadata = load(
        RESULTS / "runs" / "metadata_matched_end_to_end_v1" / "summary.json"
    )
    selector = load(
        RESULTS / "runs" / "metadata_neural_selector_v1" / "summary.json"
    )["independent_source_selector"]
    title_neural = load(
        RESULTS / "runs" / "title_neural_end_to_end_v1" / "summary.json"
    )
    decompositions = controls["reader_stage_decomposition"]
    keys = (
        "gold_absent_from_top5",
        "gold_available_reader_selected_other",
        "gold_selected_non_em_extraction",
        "gold_selected_normalized_em",
    )
    labels = (
        "Annotated paragraph absent",
        "Present, another selected",
        "Annotated selected, non-exact",
        "Annotated selected, exact",
    )
    colours = ("#C44E52", "#E69F45", "#4C78A8", "#2E7D5B")
    text_colours = ("white", INK, "white", "white")
    selector_counts = selector["stage_counts"]
    selector_row = [
        selector_counts["source_absent"],
        selector_counts["available_other_selected"],
        selector_counts["annotated_source_selected_non_em"],
        selector_counts["annotated_source_selected_em"],
    ]
    title_top5 = title_neural["reader_selector_over_top5"]["stage_counts"]
    title_rank1 = title_neural["reranker_rank1_selector"]["stage_counts"]
    rows = (
        ("Paragraph BM25", [decompositions["BM25"]["failure_stage_counts"][key] for key in keys]),
        ("Title + paragraph BM25", [metadata["stage_decomposition"]["failure_stage_counts"][key] for key in keys]),
        ("Title-aware neural top-5", [title_top5[key] for key in keys]),
        ("Title-aware neural rank-1", [title_rank1[key] for key in keys]),
        ("Section boost", [decompositions["section_boost"]["failure_stage_counts"][key] for key in keys]),
        ("Section filter", [decompositions["section_filter"]["failure_stage_counts"][key] for key in keys]),
        ("Section filter + BGE selector", selector_row),
    )
    counts = np.array(
        [row[1] for row in rows],
        dtype=int,
    )
    expected = np.array(
        [[266, 37, 8, 33], [148, 91, 28, 77], [56, 158, 30, 100],
         [56, 110, 64, 114], [47, 141, 33, 123], [47, 122, 44, 131],
         [47, 103, 71, 123]]
    )
    assert np.array_equal(counts, expected)
    assert np.all(counts.sum(axis=1) == 344)

    fig, ax = plt.subplots(figsize=(7.0, 5.05))
    y = np.arange(len(rows))
    left = np.zeros(len(rows), dtype=int)
    for index, (label, colour, text_colour) in enumerate(zip(labels, colours, text_colours)):
        bars = ax.barh(
            y, counts[:, index], left=left, height=0.62, label=label,
            color=colour, edgecolor="white", linewidth=0.9,
        )
        for bar, value in zip(bars, counts[:, index]):
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_y() + bar.get_height() / 2
            ha = "center"
            va = "center"
            colour_for_text = text_colour
            fontsize = 9.4 if value >= 20 else 8.2
            if value < 12:
                y_pos = bar.get_y() - 0.08
                va = "bottom"
                colour_for_text = INK
                fontsize = 8.8
            ax.text(
                x_pos, y_pos,
                str(value), ha=ha, va=va,
                fontsize=fontsize,
                fontweight="bold", color=colour_for_text,
            )
        left += counts[:, index]

    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 344)
    ax.set_xlabel("Test questions (each row totals 344)", fontsize=10.4)
    ax.set_title("Stage-wise end-to-end outcomes", fontsize=11.8, color=INK, pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.0, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AAB4BB")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
        frameon=False, fontsize=9.0, columnspacing=1.5, handlelength=1.5,
    )
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.25, top=0.89)
    save(fig, "22_stagewise_outcomes")
    print("22:", "; ".join(f"{name}={list(row)}" for (name, _), row in zip(rows, counts)))


def citation_intervention():
    """Use the authoritative two-panel layout, including through this older API."""
    from make_retrieval_mechanism_figures import citation_intervention as current
    current()


def main():
    from make_retrieval_mechanism_figures import diagnostic_overview
    diagnostic_overview()
    retrieval_reading_tension()
    retrieval_progression()
    stagewise_outcomes()
    citation_intervention()
    print(
        "Wrote papers/retrieval_reference/plots/"
        "{01_retrieval_diagnostic_overview,20_retrieval_reading_tension,21_retrieval_progression,"
        "22_stagewise_outcomes,23_citation_intervention}.{pdf,png}"
    )


if __name__ == "__main__":
    main()
