from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(relative: str):
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


required = [
    "papers/retrieval_reference/retrieval-reference.tex",
    "data/splits/test.jsonl",
    "results/experiments/retrieval/dense_retrieval_report.json",
    "results/experiments/retrieval/citation_aware_retrieval_report.json",
    "results/experiments/retrieval/citation_aware_end_to_end_report.json",
    "results/experiments/retrieval/runs/section_filter_act_extension_v2/summary.json",
    "results/experiments/retrieval/runs/section_filter_act_extension_v2/test_per_query.jsonl",
    "results/experiments/retrieval/runs/metadata_neural_selector_v1/summary.json",
    "results/experiments/retrieval/runs/metadata_neural_selector_v1/section_filter_bge_selector_predictions.jsonl",
    "results/experiments/retrieval/runs/title_neural_end_to_end_v1/summary.json",
    "results/experiments/retrieval/runs/title_neural_end_to_end_v1/title_neural_top5_reader_selector_predictions.jsonl",
    "results/experiments/retrieval/runs/title_neural_end_to_end_v1/title_neural_rank1_selector_predictions.jsonl",
]
missing = [path for path in required if not (ROOT / path).is_file()]
if missing:
    raise SystemExit(f"Missing required release files: {missing}")

with (ROOT / "data/splits/test.jsonl").open(encoding="utf-8") as handle:
    test_count = sum(1 for line in handle if line.strip())
assert test_count == 344, test_count

dense = load("results/experiments/retrieval/dense_retrieval_report.json")
section = load("results/experiments/retrieval/citation_aware_retrieval_report.json")
end_to_end = load("results/experiments/retrieval/citation_aware_end_to_end_report.json")
extension = load("results/experiments/retrieval/runs/section_filter_act_extension_v2/summary.json")
selector = load("results/experiments/retrieval/runs/metadata_neural_selector_v1/summary.json")["independent_source_selector"]
title_neural = load("results/experiments/retrieval/runs/title_neural_end_to_end_v1/summary.json")

assert dense["results"]["BM25"]["recall@5"] == 22.67
assert section["results"]["BM25 + section filter"]["recall@5"] == 86.34
e2e_by_name = {row["setting"]: row for row in end_to_end["results"]}
assert round(e2e_by_name["Gold paragraph + E2'"]["f1_pct"], 2) == 87.37
assert round(e2e_by_name["BM25 top-5 + E2'"]["f1_pct"], 2) == 31.95
assert round(e2e_by_name["BM25 + section filter top-5 + E2'"]["f1_pct"], 2) == 57.99
assert round(extension["test"]["retrieval"]["section_bm25"]["recall@5"], 2) == 86.34
assert round(extension["test"]["section_random_expected"]["recall@5"], 2) == 66.72
assert round(extension["test"]["size_only_random_expected"]["recall@5"], 2) == 3.86
assert round(extension["test"]["retrieval"]["predicted_act_section_hard"]["recall@5"], 2) == 75.00
assert selector["retrieval_recall@5"] == 86.34
assert selector["selected_annotated_source_pct"] == 56.40
assert selector["f1"] == 58.47
assert round(title_neural["reader_selector_over_top5"]["retrieval_recall@5_pct"], 2) == 83.72
assert round(title_neural["reader_selector_over_top5"]["f1_pct"], 2) == 53.12
assert round(title_neural["reranker_rank1_selector"]["selected_annotated_source_pct"], 2) == 51.74
assert round(title_neural["reranker_rank1_selector"]["f1_pct"], 2) == 54.62

print("Retrieval release verified: 344 test questions and headline artifacts, including section-pool, BGE-selector, and title-aware neural end-to-end controls, are present.")
