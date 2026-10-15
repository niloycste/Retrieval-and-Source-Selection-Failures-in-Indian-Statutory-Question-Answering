"""Operationally partition test questions by residual lexical content.

Recognized Section / sub-section / clause / item references are removed first.
The remaining tokens are filtered with the published interrogative-frame
lexicon below. Questions with zero residual tokens are labelled
``citation-template-only``; all others are labelled
``lexically-content-bearing``.

This is a deterministic operational partition, not a human-validated semantic
annotation. The zero-token rule reproduces the 177 / 167 split in the paper.
Threshold and continuous analyses quantify sensitivity to that boundary.

Run:
    python citation_specified_questions.py
    python citation_specified_questions.py --write-artifacts
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import re
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUN = (
    ROOT
    / "results"
    / "experiments"
    / "retrieval"
    / "runs"
    / "journal_retrieval_transfer_e2prime_canonical_v1"
    / "retrieval_qa"
)
AUDIT_PATH = (
    ROOT
    / "results"
    / "experiments"
    / "retrieval"
    / "citation_template_classifier_audit.csv"
)
ROBUSTNESS_PATH = (
    ROOT
    / "results"
    / "experiments"
    / "retrieval"
    / "citation_template_classifier_robustness.json"
)

CITATION_TEMPLATE_ONLY = "citation-template-only"
LEXICALLY_CONTENT_BEARING = "lexically-content-bearing"

# Ordered because more specific references must be removed before bare
# parenthetical item numbers. These exact patterns are exported in the
# robustness artifact for reproducibility.
CITATION_PATTERNS = [
    r"sub-?sections?\s*\(?\s*[0-9ivxlc]+[a-z]?\s*\)?",
    r"clauses?\s*\(?\s*[0-9a-z]+\s*\)?",
    r"items?\s*\(?\s*[0-9a-z]+\s*\)?",
    r"paragraphs?\s*\(?\s*[0-9a-z]+\s*\)?",
    r"sections?\s*[0-9]+[a-z]*(\s*\([0-9a-z]+\))*",
    r"\([0-9a-z]{1,3}\)",
]

INTERROGATIVE_FRAME = frozenset(
    """what which does do is are the of in under for a an and or to that this
    say says said state stated states provide provides provided mean means meant
    specify specifies specified lay lays laid down set sets out about regarding
    according to it its as per with respect there here be been being""".split()
)

# Backward-compatible names for scripts or notebooks that imported the old
# module interface. New code should use CITATION_PATTERNS and residual_tokens.
CITE = CITATION_PATTERNS
FRAME = INTERROGATIVE_FRAME


def strip_recognized_citations(question: str) -> tuple[str, list[str]]:
    """Remove recognized structural citations and return their matched text."""
    stripped = question
    matches: list[str] = []
    for pattern in CITATION_PATTERNS:

        def replace(match: re.Match[str]) -> str:
            matches.append(match.group(0))
            return " "

        stripped = re.sub(pattern, replace, stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped, matches


def residual_tokens(question: str) -> list[str]:
    """Return lexical tokens remaining after the published operational filter."""
    stripped, _ = strip_recognized_citations(question)
    tokens = re.findall(r"[a-z][a-z'\-]+", stripped.lower())
    return [token for token in tokens if token not in INTERROGATIVE_FRAME]


def classification_label(question: str, max_residual_tokens: int = 0) -> str:
    """Return the operational label at the specified residual-token boundary."""
    if max_residual_tokens < 0:
        raise ValueError("max_residual_tokens must be non-negative")
    if len(residual_tokens(question)) <= max_residual_tokens:
        return CITATION_TEMPLATE_ONLY
    return LEXICALLY_CONTENT_BEARING


def is_citation_template_only(question: str) -> bool:
    """Return True exactly when the primary zero-residual rule is satisfied."""
    return classification_label(question) == CITATION_TEMPLATE_ONLY


# Deprecated compatibility aliases. Their names describe the earlier semantic
# framing; their behavior remains identical to the published zero-token rule.
content_tokens = residual_tokens
is_citation_specified = is_citation_template_only


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in io.open(path, encoding="utf-8")
        if line.strip()
    ]


def _mean_pct(rows: list[dict], field: str) -> float:
    if not rows:
        return float("nan")
    return 100.0 * sum(float(row[field]) for row in rows) / len(rows)


def _group_summary(
    gold_rows: list[dict],
    bm25_by_id: dict[str, dict],
    max_residual_tokens: int,
) -> dict:
    groups = {
        CITATION_TEMPLATE_ONLY: [],
        LEXICALLY_CONTENT_BEARING: [],
    }
    for row in gold_rows:
        groups[classification_label(row["question"], max_residual_tokens)].append(row)

    summary = {}
    for label, rows in groups.items():
        retrieved = [bm25_by_id[row["id"]] for row in rows]
        summary[label] = {
            "n": len(rows),
            "gold_paragraph_f1_pct": round(_mean_pct(rows, "f1"), 4),
            "bm25_recall_at_5_pct": round(
                100.0
                * sum(bool(row["retrieval_hit_at_k"]) for row in retrieved)
                / len(retrieved),
                4,
            ),
            "bm25_end_to_end_f1_pct": round(_mean_pct(retrieved, "f1"), 4),
        }
    return {
        "max_residual_tokens_for_citation_template_only": max_residual_tokens,
        "groups": summary,
    }


def _continuous_summary(gold_rows: list[dict], bm25_by_id: dict[str, dict]) -> list[dict]:
    def bin_label(count: int) -> str:
        if count <= 2:
            return str(count)
        if count <= 4:
            return "3-4"
        return "5+"

    bins: dict[str, list[dict]] = collections.defaultdict(list)
    for row in gold_rows:
        retrieved = bm25_by_id[row["id"]]
        bins[bin_label(len(residual_tokens(row["question"])))].append(retrieved)

    output = []
    for label in ("0", "1", "2", "3-4", "5+"):
        rows = bins[label]
        output.append(
            {
                "residual_token_bin": label,
                "n": len(rows),
                "bm25_recall_at_5_pct": round(
                    100.0
                    * sum(bool(row["retrieval_hit_at_k"]) for row in rows)
                    / len(rows),
                    4,
                ),
                "bm25_end_to_end_f1_pct": round(_mean_pct(rows, "f1"), 4),
            }
        )
    return output


def build_analysis() -> tuple[list[dict], dict]:
    """Build the item-level audit and aggregate robustness analysis."""
    if not RUN.exists():
        raise SystemExit(
            f"missing {RUN}\n"
            "Run journal_retrieval_transfer_experiments.py --mode retrieval_qa "
            "with the canonical E2' checkpoint first."
        )

    gold_rows = _jsonl(RUN / "gold_paragraph_predictions.jsonl")
    bm25_rows = _jsonl(RUN / "bm25_top_5_qa_predictions.jsonl")
    bm25_by_id = {row["id"]: row for row in bm25_rows}
    if len(gold_rows) != 344 or set(bm25_by_id) != {row["id"] for row in gold_rows}:
        raise ValueError("canonical prediction files must contain the same 344 test IDs")

    audit_rows = []
    for gold in gold_rows:
        retrieved = bm25_by_id[gold["id"]]
        stripped, citations = strip_recognized_citations(gold["question"])
        tokens = residual_tokens(gold["question"])
        audit_rows.append(
            {
                "question_id": gold["id"],
                "act": gold.get("act", ""),
                "original_question": gold["question"],
                "recognized_citations": " | ".join(citations),
                "question_after_citation_removal": stripped,
                "residual_tokens": " | ".join(tokens),
                "residual_token_count": len(tokens),
                "assigned_class": classification_label(gold["question"]),
                "gold_paragraph_f1_pct": round(100.0 * float(gold["f1"]), 4),
                "bm25_recall_at_5": int(bool(retrieved["retrieval_hit_at_k"])),
                "bm25_end_to_end_f1_pct": round(100.0 * float(retrieved["f1"]), 4),
            }
        )

    primary = _group_summary(gold_rows, bm25_by_id, 0)
    primary_groups = primary["groups"]
    if (
        primary_groups[CITATION_TEMPLATE_ONLY]["n"] != 177
        or primary_groups[LEXICALLY_CONTENT_BEARING]["n"] != 167
    ):
        raise ValueError("primary operational partition no longer reproduces 177 / 167")

    residual_counts = [len(residual_tokens(row["question"])) for row in gold_rows]
    recall_hits = [int(bool(bm25_by_id[row["id"]]["retrieval_hit_at_k"])) for row in gold_rows]
    end_to_end_f1 = [float(bm25_by_id[row["id"]]["f1"]) for row in gold_rows]
    recall_corr = spearmanr(residual_counts, recall_hits)
    f1_corr = spearmanr(residual_counts, end_to_end_f1)

    robustness = {
        "status": "canonical",
        "analysis_scope": "344-question IndicIPR-QA test split",
        "method": {
            "description": (
                "Deterministic operational partition, not a human-validated "
                "semantic annotation. Recognized citations are removed, tokens "
                "in the published interrogative-frame lexicon are filtered, and "
                "the primary class boundary is zero residual tokens."
            ),
            "primary_labels": [
                CITATION_TEMPLATE_ONLY,
                LEXICALLY_CONTENT_BEARING,
            ],
            "citation_patterns": CITATION_PATTERNS,
            "interrogative_frame_lexicon": sorted(INTERROGATIVE_FRAME),
        },
        "threshold_sensitivity": [
            _group_summary(gold_rows, bm25_by_id, threshold)
            for threshold in (0, 1, 2)
        ],
        "continuous_residual_token_analysis": _continuous_summary(
            gold_rows, bm25_by_id
        ),
        "spearman_correlations": {
            "residual_count_vs_bm25_recall_at_5_hit": {
                "rho": round(float(recall_corr.statistic), 4),
                "p_value": float(recall_corr.pvalue),
            },
            "residual_count_vs_bm25_end_to_end_f1": {
                "rho": round(float(f1_corr.statistic), 4),
                "p_value": float(f1_corr.pvalue),
            },
            "interpretation": (
                "Item-level correlations are descriptive; the manuscript treats "
                "the threshold and binned analyses as robustness checks rather "
                "than human construct validation."
            ),
        },
    }
    return audit_rows, robustness


def write_artifacts(audit_rows: list[dict], robustness: dict) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with io.open(AUDIT_PATH, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    with io.open(ROBUSTNESS_PATH, "w", encoding="utf-8") as stream:
        json.dump(robustness, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="write the canonical CSV audit and JSON robustness report",
    )
    args = parser.parse_args()

    audit_rows, robustness = build_analysis()
    primary = robustness["threshold_sensitivity"][0]["groups"]
    print(f"test questions              : {len(audit_rows)}")
    for label in (CITATION_TEMPLATE_ONLY, LEXICALLY_CONTENT_BEARING):
        row = primary[label]
        print(
            f"{label:28s}: {row['n']:3d} | "
            f"gold F1 {row['gold_paragraph_f1_pct']:5.2f} | "
            f"BM25 F1 {row['bm25_end_to_end_f1_pct']:5.2f}"
        )

    print("\nthreshold sensitivity")
    for entry in robustness["threshold_sensitivity"]:
        threshold = entry["max_residual_tokens_for_citation_template_only"]
        groups = entry["groups"]
        template = groups[CITATION_TEMPLATE_ONLY]
        lexical = groups[LEXICALLY_CONTENT_BEARING]
        print(
            f"  <= {threshold}: n={template['n']:3d}/{lexical['n']:3d} | "
            f"gold F1={template['gold_paragraph_f1_pct']:5.2f}/"
            f"{lexical['gold_paragraph_f1_pct']:5.2f} | "
            f"BM25 F1={template['bm25_end_to_end_f1_pct']:5.2f}/"
            f"{lexical['bm25_end_to_end_f1_pct']:5.2f}"
        )

    print("\ncontinuous residual-token analysis")
    for row in robustness["continuous_residual_token_analysis"]:
        print(
            f"  {row['residual_token_bin']:>3s}: n={row['n']:3d} | "
            f"R@5={row['bm25_recall_at_5_pct']:5.2f} | "
            f"F1={row['bm25_end_to_end_f1_pct']:5.2f}"
        )

    if args.write_artifacts:
        write_artifacts(audit_rows, robustness)
        print(f"\nwrote {AUDIT_PATH.relative_to(ROOT)}")
        print(f"wrote {ROBUSTNESS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
