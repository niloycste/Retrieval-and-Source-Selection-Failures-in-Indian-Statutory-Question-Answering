"""Independently recompute metadata-neural and selector summary totals."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import metadata_neural_and_selector_control as experiment


def close(left: float, right: float, tolerance: float = 0.011) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> int:
    output = experiment.OUT_DIR
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    _corpus, passages, titles, title_passages, tests, questions, gold = experiment.load_data()
    qids = [str(row["id"]) for row in tests]
    files = {
        "title_paragraph_bm25": "rankings_title_paragraph_bm25.jsonl",
        "title_paragraph_bge_m3": "rankings_title_paragraph_bge_m3.jsonl",
        "title_paragraph_rrf_bm25_bge_m3": "rankings_title_paragraph_rrf_bm25_bge_m3.jsonl",
        "title_paragraph_rrf_bm25_bge_m3_bge_reranker_v2_m3":
            "rankings_title_paragraph_rrf_bm25_bge_m3_bge_reranker_v2_m3.jsonl",
    }
    for name, filename in files.items():
        rankings = experiment.load_rankings(output / filename, qids)
        observed = experiment.metrics(rankings, gold)
        expected = summary["retrieval_results"][name]
        for metric, value in expected.items():
            if not close(observed[metric], value):
                raise AssertionError(f"{name} {metric}: {observed[metric]} != {value}")

    predictions = experiment.read_jsonl(output / "section_filter_bge_selector_predictions.jsonl")
    if len(predictions) != 344 or [row["id"] for row in predictions] != qids:
        raise AssertionError("selector predictions do not align with the test split")
    stage = summary["independent_source_selector"]["stage_counts"]
    if sum(stage.values()) != 344:
        raise AssertionError(f"selector stage counts do not sum to 344: {stage}")
    if sum(not row["retrieval_hit_at_k"] for row in predictions) != stage["source_absent"]:
        raise AssertionError("selector source-absent count mismatch")
    if not close(
        100 * sum(row["selected_context_is_gold"] for row in predictions) / 344,
        summary["independent_source_selector"]["selected_annotated_source_pct"],
    ):
        raise AssertionError("selector source-selection percentage mismatch")
    if experiment.metrics(experiment.section_filter_rankings(passages, titles, questions), gold)["recall@5"] != 86.34:
        raise AssertionError("section-filter reproduction failed")
    print(json.dumps({
        "status": "PASS",
        "n_queries": len(qids),
        "retrieval_systems_checked": list(files),
        "selector_stage_counts": stage,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
