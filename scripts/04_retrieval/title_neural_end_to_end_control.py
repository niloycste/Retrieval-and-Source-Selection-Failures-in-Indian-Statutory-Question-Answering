"""Evaluate the strongest title-aware neural ranking with the fixed QA reader.

The retrieval ranking is the archived title+paragraph BM25--BGE-M3 RRF pool
reranked by BGE-reranker-v2-m3.  The QA reader continues to receive only the
original question and paragraph text.  Cached reader outputs are reused; only
previously unseen question--paragraph pairs are decoded.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from citation_aware_retrieval_experiment import metric_summary, write_jsonl
from retrieval_diagnostic_controls import load_jsonl, normalize, stage_summary

CORPUS_FILE = ROOT / "data" / "processed" / "indic_ipr_qa" / "unique_contexts.jsonl"
TEST_FILE = ROOT / "data" / "splits" / "test.jsonl"
RANKING_FILE = (
    ROOT / "results" / "experiments" / "retrieval" / "runs"
    / "metadata_neural_selector_v1"
    / "rankings_title_paragraph_rrf_bm25_bge_m3_bge_reranker_v2_m3.jsonl"
)
READER_CACHE = (
    ROOT / "results" / "experiments" / "retrieval"
    / "review_robustness_20260920" / "reader_candidates.jsonl"
)
OUTPUT_DIR = (
    ROOT / "results" / "experiments" / "retrieval" / "runs"
    / "title_neural_end_to_end_v1"
)
DEFAULT_QA_MODEL = ROOT / "artifacts" / "models" / "squad_v1" / "baseline_e2_squadv1" / "final"
TOP_K = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-model", default=str(DEFAULT_QA_MODEL))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    started = time.time()
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(CORPUS_FILE)
    tests = load_jsonl(TEST_FILE)
    ranking_rows = load_jsonl(RANKING_FILE)
    cache_rows = load_jsonl(READER_CACHE)
    if len(tests) != 344 or len(ranking_rows) != len(tests):
        raise RuntimeError("unexpected test or ranking length")
    if [row["qid"] for row in ranking_rows] != [row["id"] for row in tests]:
        raise RuntimeError("ranking and test question identifiers are not aligned")

    paragraphs = [normalize(row["context"]) for row in corpus]
    questions = [row["question"] for row in tests]
    by_context: dict[str, int] = {}
    for index, paragraph in enumerate(paragraphs):
        by_context.setdefault(paragraph, index)
    gold = [by_context[normalize(row["context"])] for row in tests]
    rankings = [row["ranking"] for row in ranking_rows]
    pair_keys = [
        (query_index, context_index)
        for query_index, ranking in enumerate(rankings)
        for context_index in ranking[:TOP_K]
    ]

    decoded_by_pair: dict[tuple[int, int], tuple[str, float]] = {
        (int(row["query"]), int(row["context"])): (row["raw"][0], float(row["raw"][1]))
        for row in cache_rows
    }
    missing = [key for key in pair_keys if key not in decoded_by_pair]

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from scripts.qa_evaluation import (
        QADecodingConfig,
        compute_char_f1,
        compute_em_f1,
        predict_answers,
        resolve_device,
    )

    model_path = Path(args.qa_model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"QA checkpoint not found: {model_path}")
    device = resolve_device(args.device)
    config = QADecodingConfig(
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
        max_answer_tokens=None,
    )
    print(
        f"title-aware neural reader control: {len(pair_keys):,} pairs; "
        f"cached={len(pair_keys) - len(missing):,}; missing={len(missing):,}; device={device}",
        flush=True,
    )
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(str(model_path), local_files_only=True)
        model.to(device)
        model.eval()
        decoded = predict_answers(
            tokenizer,
            model,
            device,
            [questions[query_index] for query_index, _ in missing],
            [paragraphs[context_index] for _, context_index in missing],
            config,
        )
        decoded_by_pair.update(dict(zip(missing, decoded)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def make_row(query_index: int, selected_rank: int, selected_index: int,
                 prediction: str, span_score: float) -> dict:
        answer = tests[query_index]["answer"]
        em, f1 = compute_em_f1(prediction, answer)
        return {
            "id": tests[query_index]["id"],
            "question": questions[query_index],
            "gold_answer": answer,
            "prediction": prediction,
            "em": em,
            "f1": round(f1, 6),
            "char_f1": round(compute_char_f1(prediction, answer), 6),
            "retrieval_hit_at_k": gold[query_index] in rankings[query_index][:TOP_K],
            "selected_context_is_gold": selected_index == gold[query_index],
            "candidate_count": TOP_K,
            "selected_rank": selected_rank,
            "selected_context_index": selected_index,
            "selected_context_title": corpus[selected_index].get("title", ""),
            "selected_context_act": corpus[selected_index].get("act", ""),
            "span_score": round(float(span_score), 6),
        }

    reader_rows = []
    rank_one_rows = []
    for query_index, ranking in enumerate(rankings):
        scored = []
        for candidate_rank, context_index in enumerate(ranking[:TOP_K], start=1):
            prediction, span_score = decoded_by_pair[(query_index, context_index)]
            scored.append((span_score, -candidate_rank, context_index, prediction))
        span_score, negative_rank, selected_index, prediction = max(scored)
        reader_rows.append(make_row(
            query_index, -negative_rank, selected_index, prediction, span_score
        ))

        first_index = ranking[0]
        first_prediction, first_score = decoded_by_pair[(query_index, first_index)]
        rank_one_rows.append(make_row(
            query_index, 1, first_index, first_prediction, first_score
        ))

    reader_file = OUTPUT_DIR / "title_neural_top5_reader_selector_predictions.jsonl"
    rank_one_file = OUTPUT_DIR / "title_neural_rank1_selector_predictions.jsonl"
    write_jsonl(reader_file, reader_rows)
    write_jsonl(rank_one_file, rank_one_rows)

    def summarize(rows: list[dict]) -> dict:
        stage = stage_summary(rows)
        summary = metric_summary(rows)
        return {
            "n": len(rows),
            "retrieval_recall@5_pct": stage["gold_available_pct"],
            "selected_annotated_source_pct": stage["selected_annotated_source_pct"],
            "conditional_source_selection_pct": stage["conditional_source_selection_pct_given_available"],
            "em_pct": summary["em_pct"],
            "f1_pct": summary["f1_pct"],
            "annotation_grounded_em_pct": stage["annotation_grounded_em_pct"],
            "annotation_grounded_f1_pct": stage["annotation_grounded_f1_pct"],
            "stage_counts": stage["failure_stage_counts"],
        }

    payload = {
        "status": "complete post-hoc title-aware neural end-to-end control",
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": {"questions": len(tests), "passages": len(corpus), "top_k": TOP_K},
        "retrieval": (
            "title+paragraph BM25--BGE-M3 RRF, reranked by "
            "BGE-reranker-v2-m3"
        ),
        "reader_selector_over_top5": summarize(reader_rows),
        "reranker_rank1_selector": summarize(rank_one_rows),
        "provenance": {
            "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "ranking_file": str(RANKING_FILE.relative_to(ROOT)).replace("\\", "/"),
            "reader_cache": str(READER_CACHE.relative_to(ROOT)).replace("\\", "/"),
            "reader_pairs_total": len(pair_keys),
            "reader_pairs_reused": len(pair_keys) - len(missing),
            "reader_pairs_newly_decoded": len(missing),
            "qa_model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "decoder": config.as_dict(),
            "input_sha256": {
                str(CORPUS_FILE.relative_to(ROOT)).replace("\\", "/"): sha256(CORPUS_FILE),
                str(TEST_FILE.relative_to(ROOT)).replace("\\", "/"): sha256(TEST_FILE),
                str(RANKING_FILE.relative_to(ROOT)).replace("\\", "/"): sha256(RANKING_FILE),
                str(READER_CACHE.relative_to(ROOT)).replace("\\", "/"): sha256(READER_CACHE),
            },
            "elapsed_seconds": round(time.time() - started, 3),
        },
    }
    summary_file = OUTPUT_DIR / "summary.json"
    summary_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "RUN_COMPLETE.json").write_text(
        json.dumps({
            "status": "complete",
            "summary": str(summary_file.relative_to(ROOT)).replace("\\", "/"),
            "summary_sha256": sha256(summary_file),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
