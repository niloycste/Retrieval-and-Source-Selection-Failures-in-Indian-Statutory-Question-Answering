"""Recompute the published human-audit results from de-identified records.

This validator is designed for the anonymous public release.  It does not need
the restricted master workbook, free-text rationales, reviewer identities, or
the private passage-order key.  It verifies the released categorical labels,
recomputes the original agreement statistics and consensus counts, reconstructs
the seeded sample from the released section-filter predictions, and checks the
two corresponding manuscript tables.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_human_validation as manuscript_validator
import update_human_validation_agreement as agreement


JUDGMENT_ORDER = [
    "Question uniquely identifies annotated source",
    "Question uniquely identifies selected source",
    "Annotated source legally relevant",
    "Selected source legally relevant",
    "Sources legally equivalent",
    "Annotated answer legally correct",
    "Predicted answer legally correct",
    "Predicted-answer error materiality",
    "Annotated-answer error materiality",
    "Text quality issue",
]
ALL_LABELS = {
    **{name: spec[2] for name, spec in agreement.ROLE_FIELDS.items()},
    **{name: spec[1] for name, spec in agreement.CASE_FIELDS.items()},
}
LABELS = {name: ALL_LABELS[name] for name in JUDGMENT_ORDER}
PREDICTIONS = (
    ROOT / "results/experiments/retrieval/runs/citation_aware_v1"
    / "bm25_section_filter_top5_e2_predictions.jsonl"
)
PUBLIC_LABELS = ROOT / "outputs/human_validation_report/anonymized_item_labels.jsonl"
SAVED_SUMMARY = ROOT / "outputs/human_validation_report/manuscript_summary.json"
MANUSCRIPT = ROOT / "papers/retrieval_reference/retrieval-reference.tex"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label: str, tolerance: float = 1e-12) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    elif not math.isclose(float(actual), float(expected), abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def agreement_summary(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for judgment, labels in LABELS.items():
        pairs = [
            (
                row["original_ratings"]["annotator_A"].get(judgment),
                row["original_ratings"]["annotator_C"].get(judgment),
            )
            for row in rows
        ]
        pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
        for left, right in pairs:
            if left not in labels or right not in labels:
                raise AssertionError(f"invalid label for {judgment}: {(left, right)}")
        left = [value for value, _ in pairs]
        right = [value for _, value in pairs]
        n = len(pairs)
        exact = sum(a == b for a, b in pairs)
        expected = sum(left.count(label) * right.count(label) for label in labels) / n**2
        current = {
            "Paired cases": n,
            "Exact agreements": exact,
            "Disagreements": n - exact,
            "Raw agreement": exact / n,
            "Expected agreement": expected,
            "Cohen kappa (nominal)": agreement.cohen_kappa(left, right, labels),
            "Gwet AC1": agreement.gwet_ac1(left, right, labels),
            "categories": labels,
            "left_counts": dict(Counter(left)),
            "right_counts": dict(Counter(right)),
        }
        if "materiality" in judgment:
            weighted, weighted_n, excluded = agreement.quadratic_weighted_kappa(
                left, right, agreement.MATERIALITY_ORDER
            )
            current.update({
                "Quadratic weighted kappa": weighted,
                "Weighted-kappa paired cases": weighted_n,
                "Excluded from weighted kappa": excluded,
            })
        result[judgment] = current
    return result


def consensus_summary(rows: list[dict], population: Counter) -> dict[str, dict]:
    result = {}
    groups = ["All sampled cases"] + sorted(population)
    for group in groups:
        current = rows if group == "All sampled cases" else [
            row for row in rows if row["stage_outcome"] == group
        ]
        result[group] = {
            "sample_n": len(current),
            "population_n": 344 if group == "All sampled cases" else population[group],
            "counts": {
                judgment: dict(Counter(row["consensus"][judgment] for row in current))
                for judgment in LABELS
            },
        }
    return result


def compare_nested(actual, expected, label="root") -> None:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{label}: key mismatch")
        for key in expected:
            compare_nested(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, list):
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        close(actual, expected, label)
    elif actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def prediction_stage(row: dict) -> str:
    if not row["retrieval_hit_at_k"]:
        return "Annotated source absent from top 5"
    if not row["selected_context_is_gold"]:
        return "Annotated source available; other source selected"
    return (
        "Annotated source selected; exact answer"
        if row["em"] else "Annotated source selected; non-exact answer"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=PUBLIC_LABELS)
    parser.add_argument("--summary", type=Path, default=SAVED_SUMMARY)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    args = parser.parse_args()

    rows = read_jsonl(args.labels)
    if len(rows) != 80 or len({row["case_id"] for row in rows}) != 80:
        raise AssertionError("expected 80 unique de-identified audit records")
    if len({row["question_id"] for row in rows}) != 80:
        raise AssertionError("expected 80 unique sampled question IDs")

    predictions = read_jsonl(args.predictions)
    if len(predictions) != 344 or len({row["id"] for row in predictions}) != 344:
        raise AssertionError("expected 344 unique section-filter predictions")
    by_id = {row["id"]: row for row in predictions}
    population = Counter(prediction_stage(row) for row in predictions)

    rng = random.Random(42)
    sampled = []
    for stage in sorted(population):
        candidates = sorted(
            (row for row in predictions if prediction_stage(row) == stage),
            key=lambda row: row["id"],
        )
        rng.shuffle(candidates)
        sampled.extend(row["id"] for row in candidates[:20])
    if set(sampled) != {row["question_id"] for row in rows}:
        raise AssertionError("released records do not match the seed-42 stratified sample")
    for row in rows:
        prediction = by_id[row["question_id"]]
        stage = prediction_stage(prediction)
        if row["stage_outcome"] != stage:
            raise AssertionError(f"stage mismatch for {row['case_id']}")
        close(row["sampling_weight"], population[stage] / 20, row["case_id"])

    primary = agreement_summary(rows)
    consensus = consensus_summary(rows, population)
    saved = json.loads(args.summary.read_text(encoding="utf-8"))
    if sha256(args.predictions) != saved["predictions_sha256"]:
        raise AssertionError("prediction hash differs from the verified human-audit summary")
    compare_nested(primary, saved["primary_agreement"], "primary_agreement")
    compare_nested(consensus, saved["consensus"], "consensus")

    report = {"primary_agreement": primary, "consensus": consensus}
    table_check = manuscript_validator.verify_manuscript(report, args.manuscript)
    print(json.dumps({
        "status": "PASS",
        "public_records": len(rows),
        "prediction_rows": len(predictions),
        "sampling_seed": 42,
        "population_stage_counts": dict(population),
        "agreement_judgments_recomputed": len(primary),
        "manuscript_tables": table_check,
        "private_workbook_required": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
