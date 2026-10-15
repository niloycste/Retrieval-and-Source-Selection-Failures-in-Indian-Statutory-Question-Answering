"""Run metadata-matched BM25 top-5 candidates through the fixed E2' reader.

This is an additional post-hoc metadata-matched control. It does not train a model and
does not alter the canonical retrieval or citation-aware artifacts. Retrieval
indexes title plus paragraph exactly as in retrieval_diagnostic_controls.py;
the reader receives the original question and paragraph text only and uses the
shared sliding-window constrained decoder.
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
from retrieval_diagnostic_controls import (
    BM25,
    cluster_bootstrap,
    load_jsonl,
    normalize,
    rank,
    retrieval_metrics,
    stage_summary,
    tokenize,
)

CORPUS_FILE = ROOT / "data" / "processed" / "indic_ipr_qa" / "unique_contexts.jsonl"
TEST_FILE = ROOT / "data" / "splits" / "test.jsonl"
SECTION_FILTER_PREDICTIONS = (
    ROOT
    / "results"
    / "experiments"
    / "retrieval"
    / "runs"
    / "citation_aware_v1"
    / "bm25_section_filter_top5_e2_predictions.jsonl"
)
OUTPUT_DIR = (
    ROOT
    / "results"
    / "experiments"
    / "retrieval"
    / "runs"
    / "metadata_matched_end_to_end_v1"
)
DEFAULT_QA_MODEL = (
    ROOT / "artifacts" / "models" / "squad_v1" / "baseline_e2_squadv1" / "final"
)
TOP_K = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-model", default=str(DEFAULT_QA_MODEL))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    return args


def main() -> None:
    started = time.time()
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(CORPUS_FILE)
    tests = load_jsonl(TEST_FILE)
    paragraphs = [normalize(row["context"]) for row in corpus]
    titles = [normalize(row.get("title", "")) for row in corpus]
    questions = [row["question"] for row in tests]

    by_context: dict[str, int] = {}
    for index, paragraph in enumerate(paragraphs):
        by_context.setdefault(paragraph, index)
    gold = [by_context[normalize(row["context"])] for row in tests]

    retrieval_inputs = [
        normalize(f"{title} {paragraph}")
        for title, paragraph in zip(titles, paragraphs)
    ]
    bm25 = BM25([tokenize(text) for text in retrieval_inputs])
    rankings = [rank(bm25.scores(tokenize(question))) for question in questions]
    retrieval = retrieval_metrics(rankings, gold)
    if retrieval["recall@5"] != 56.9767:
        raise RuntimeError(
            "metadata-matched BM25 no longer reproduces recall@5=56.9767: "
            f"{retrieval['recall@5']}"
        )

    pair_keys = [
        (query_index, context_index)
        for query_index, ranking in enumerate(rankings)
        for context_index in ranking[:TOP_K]
    ]

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
        f"metadata-matched reader control: {len(pair_keys):,} pairs | "
        f"device={device} | top-k={TOP_K}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        str(model_path), local_files_only=True
    )
    model.to(device)
    model.eval()

    decoded = predict_answers(
        tokenizer,
        model,
        device,
        [questions[query_index] for query_index, _ in pair_keys],
        [paragraphs[context_index] for _, context_index in pair_keys],
        config,
    )
    decoded_by_pair = dict(zip(pair_keys, decoded))

    rows = []
    for query_index, ranking in enumerate(rankings):
        candidates = ranking[:TOP_K]
        scored = []
        for candidate_rank, context_index in enumerate(candidates, start=1):
            prediction, span_score = decoded_by_pair[(query_index, context_index)]
            scored.append((span_score, candidate_rank, context_index, prediction))
        span_score, selected_rank, selected_index, prediction = max(scored)
        gold_answer = tests[query_index]["answer"]
        em, f1 = compute_em_f1(prediction, gold_answer)
        char_f1 = compute_char_f1(prediction, gold_answer)
        rows.append(
            {
                "id": tests[query_index].get("id", str(query_index)),
                "question": questions[query_index],
                "gold_answer": gold_answer,
                "prediction": prediction,
                "em": em,
                "f1": round(f1, 6),
                "char_f1": round(char_f1, 6),
                "retrieval_hit_at_k": gold[query_index] in candidates,
                "selected_context_is_gold": selected_index == gold[query_index],
                "candidate_count": len(candidates),
                "selected_rank": selected_rank,
                "selected_context_index": selected_index,
                "selected_context_title": corpus[selected_index].get("title", ""),
                "selected_context_act": corpus[selected_index].get("act", ""),
                "span_score": round(float(span_score), 6),
            }
        )

    prediction_file = OUTPUT_DIR / "title_plus_paragraph_bm25_top5_e2_predictions.jsonl"
    write_jsonl(prediction_file, rows)

    section_rows = load_jsonl(SECTION_FILTER_PREDICTIONS)
    if [row["id"] for row in rows] != [row["id"] for row in section_rows]:
        raise RuntimeError("metadata and section-filter predictions are not aligned")
    group_by_id = {row["id"]: row["group_id"] for row in tests}
    stage = stage_summary(rows)
    summary = metric_summary(rows)
    summary.update(
        {
            "retrieval_recall_pct": retrieval["recall@5"],
            "reader_selected_gold_context_pct": stage[
                "selected_annotated_source_pct"
            ],
            "annotation_grounded_em_pct": stage["annotation_grounded_em_pct"],
            "annotation_grounded_f1_pct": stage["annotation_grounded_f1_pct"],
        }
    )
    cluster = cluster_bootstrap(rows, section_rows, group_by_id, "f1")

    payload = {
        "status": (
            "complete additional post-hoc metadata-matched control; "
            "canonical artifacts unchanged"
        ),
        "created_at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "definition": {
            "retrieval_query": "original full question",
            "retrieval_candidate_input": "section title + original paragraph",
            "reader_input": "original full question + original paragraph only",
            "reader_selection": "maximum summed start-end logit over BM25 top-5",
            "gold_information_used_by_retrieval": False,
        },
        "scope": {
            "questions": len(tests),
            "passages": len(corpus),
            "source_clusters": len(set(group_by_id.values())),
        },
        "retrieval_metrics_pct": retrieval,
        "reader_metrics_pct": summary,
        "stage_decomposition": stage,
        "paired_source_cluster_bootstrap": {
            "comparison": (
                "section filter top-5 + E2' minus title+paragraph BM25 "
                "top-5 + E2'"
            ),
            "f1": cluster,
        },
        "provenance": {
            "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "corpus": str(CORPUS_FILE.relative_to(ROOT)).replace("\\", "/"),
            "test": str(TEST_FILE.relative_to(ROOT)).replace("\\", "/"),
            "section_filter_predictions": str(
                SECTION_FILTER_PREDICTIONS.relative_to(ROOT)
            ).replace("\\", "/"),
            "prediction_file": str(prediction_file.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "input_sha256": {
                str(CORPUS_FILE.relative_to(ROOT)).replace("\\", "/"): sha256(
                    CORPUS_FILE
                ),
                str(TEST_FILE.relative_to(ROOT)).replace("\\", "/"): sha256(TEST_FILE),
                str(SECTION_FILTER_PREDICTIONS.relative_to(ROOT)).replace(
                    "\\", "/"
                ): sha256(SECTION_FILTER_PREDICTIONS),
            },
            "qa_model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "decoder": config.as_dict(),
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "tokenizer": "[A-Za-z0-9]+ lowercased",
                "tie_break": "descending score, then ascending corpus index",
            },
            "elapsed_seconds": round(time.time() - started, 3),
        },
    }
    summary_file = OUTPUT_DIR / "summary.json"
    summary_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    complete = {
        "status": "complete",
        "summary": str(summary_file.relative_to(ROOT)).replace("\\", "/"),
        "prediction_file": str(prediction_file.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(summary_file),
    }
    (OUTPUT_DIR / "RUN_COMPLETE.json").write_text(
        json.dumps(complete, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
