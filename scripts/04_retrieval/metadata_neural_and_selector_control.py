"""Metadata-matched neural retrieval and independent source-selection controls.

This versioned, resumable control addresses two distinct questions for the
IndicIPR-QA test set:

1. Does the section-filter comparison remain informative when a modern dense
   retriever and reranker receive the same title-plus-paragraph representation
   as the title-aware BM25 control?
2. Within the fixed section-filter top-five pool, what happens when passage
   selection is performed by a retrieval cross-encoder rather than by the QA
   reader's uncalibrated maximum span logit?

The control does not overwrite any earlier result.  It uses the immutable
BGE-M3 and BGE-reranker-v2-m3 snapshots pinned by
``modern_retrieval_extension.py`` and checkpoints every ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dense_retrieval_experiment as base
import modern_retrieval_extension as modern
from scripts.qa_evaluation import compute_em_f1


OUT_DIR = (
    ROOT / "results" / "experiments" / "retrieval" / "runs"
    / "metadata_neural_selector_v1"
)
CORPUS_FILE = ROOT / "data" / "processed" / "indic_ipr_qa" / "unique_contexts.jsonl"
TEST_FILE = ROOT / "data" / "splits" / "test.jsonl"
READER_CACHE = (
    ROOT / "results" / "experiments" / "retrieval"
    / "review_robustness_20260920" / "reader_candidates.jsonl"
)
SECTION_RE = re.compile(r"\bsections?\s*[.\-]?\s*(\d+[A-Za-z]{0,2})\b", re.I)
DEPTH = 50
RERANK_DEPTH = 50
METRIC_KS = (1, 3, 5, 10, 50)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(*sequences) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        for item in sequence:
            encoded = str(item).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def parse_section(text: str) -> str | None:
    match = SECTION_RE.search(text or "")
    return re.sub(r"\s+", "", match.group(1)).upper() if match else None


def load_data():
    corpus = read_jsonl(CORPUS_FILE)
    passages = [row["context"] for row in corpus]
    titles = [row.get("title", "") for row in corpus]
    title_passages = [f"{title}\n{passage}" for title, passage in zip(titles, passages)]
    by_context = {base.norm(text): index for index, text in enumerate(passages)}
    tests, questions, gold = [], [], []
    for row in read_jsonl(TEST_FILE):
        index = by_context.get(base.norm(row["context"]))
        if index is not None:
            tests.append(row)
            questions.append(row["question"])
            gold.append(index)
    if len(tests) != 344 or len(corpus) != 1116:
        raise ValueError(f"unexpected data sizes: {len(tests)=}, {len(corpus)=}")
    return corpus, passages, titles, title_passages, tests, questions, gold


def metrics(rankings: list[list[int]], gold: list[int]) -> dict[str, float]:
    result = {
        f"recall@{k}": 100.0 * sum(g in row[:k] for row, g in zip(rankings, gold)) / len(gold)
        for k in METRIC_KS
    }
    reciprocal_ranks, ndcgs = [], []
    for row, target in zip(rankings, gold):
        top = row[:10]
        rank = top.index(target) + 1 if target in top else None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 1) if rank else 0.0)
    result["mrr@10"] = 100.0 * sum(reciprocal_ranks) / len(gold)
    result["ndcg@10"] = 100.0 * sum(ndcgs) / len(gold)
    return {key: round(value, 2) for key, value in result.items()}


def save_rankings(path: Path, qids: list[str], rankings: list[list[int]], scores=None) -> None:
    rows = []
    for index, (qid, ranking) in enumerate(zip(qids, rankings)):
        row = {"qid": qid, "ranking": ranking}
        if scores is not None:
            row["scores"] = scores[index]
        rows.append(row)
    write_jsonl(path, rows)


def load_rankings(path: Path, qids: list[str]) -> list[list[int]]:
    rows = read_jsonl(path)
    observed = [str(row["qid"]) for row in rows]
    if observed != qids:
        raise ValueError(f"ranking IDs do not align: {path}")
    return [[int(value) for value in row["ranking"]] for row in rows]


def title_bm25(texts: list[str], questions: list[str]) -> list[list[int]]:
    index = base.BM25([base.tokenize(text) for text in texts])
    return [
        base.ranking_from_scores(index.scores(base.tokenize(question)), DEPTH)
        for question in questions
    ]


def dense_title_rank(
    texts: list[str], questions: list[str], qids: list[str], args
) -> list[list[int]]:
    path = OUT_DIR / "rankings_title_paragraph_bge_m3.jsonl"
    meta_path = OUT_DIR / "rankings_title_paragraph_bge_m3.meta.json"
    expected = {
        "model": modern.DENSE_MODEL,
        "revision": modern.DENSE_REVISION,
        "candidate_text": "section title plus paragraph",
        "max_seq_length": 512,
        "normalised": True,
        "depth": DEPTH,
        "corpus_fingerprint": fingerprint(texts),
        "query_fingerprint": fingerprint(qids, questions),
    }
    if path.exists() and meta_path.exists() and not args.force:
        observed = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(observed.get(key) == value for key, value in expected.items()):
            return load_rankings(path, qids)
        raise ValueError("title-aware BGE cache metadata mismatch; inspect before --force")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(modern.model_locations()[modern.DENSE_MODEL]),
        device=args.device,
        local_files_only=True,
    )
    model.max_seq_length = 512
    started = time.time()
    corpus_vectors = model.encode(
        texts,
        batch_size=args.dense_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32", copy=False)
    query_vectors = model.encode(
        questions,
        batch_size=args.dense_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32", copy=False)
    rankings, score_rows = [], []
    for vector in query_vectors:
        values = np.asarray(corpus_vectors @ vector, dtype="float32")
        candidates = np.argpartition(values, -DEPTH)[-DEPTH:]
        order = np.lexsort((candidates, -values[candidates]))
        ranked = candidates[order]
        rankings.append([int(value) for value in ranked])
        score_rows.append([float(values[value]) for value in ranked])
    save_rankings(path, qids, rankings, score_rows)
    atomic_json(meta_path, {**expected, "elapsed_seconds": time.time() - started, "completed_at": now_iso()})
    return rankings


def rrf(first: list[list[int]], second: list[list[int]]) -> list[list[int]]:
    output = []
    for left, right in zip(first, second):
        values: defaultdict[int, float] = defaultdict(float)
        for ranking in (left, right):
            for rank, doc_id in enumerate(ranking):
                values[doc_id] += 1.0 / (60 + rank + 1)
        output.append(sorted(values, key=lambda doc_id: (-values[doc_id], doc_id))[:DEPTH])
    return output


def rerank(
    name: str,
    rankings: list[list[int]],
    texts: list[str],
    questions: list[str],
    qids: list[str],
    args,
    model=None,
):
    path = OUT_DIR / f"rankings_{name}_bge_reranker_v2_m3.jsonl"
    meta_path = OUT_DIR / f"rankings_{name}_bge_reranker_v2_m3.meta.json"
    expected = {
        "model": modern.RERANKER_MODEL,
        "revision": modern.RERANKER_REVISION,
        "candidate_text": "section title plus paragraph",
        "max_seq_length": 512,
        "rerank_depth": RERANK_DEPTH,
        "base_fingerprint": fingerprint(
            qids, [json.dumps(row[:RERANK_DEPTH], separators=(",", ":")) for row in rankings]
        ),
        "query_fingerprint": fingerprint(qids, questions),
    }
    completed: dict[str, list[int]] = {}
    if path.exists() and not args.force:
        if not meta_path.exists():
            raise ValueError(f"reranking cache has no metadata: {path}")
        observed = json.loads(meta_path.read_text(encoding="utf-8"))
        if not all(observed.get(key) == value for key, value in expected.items()):
            raise ValueError(f"reranking cache metadata mismatch: {name}")
        completed = {
            str(row["qid"]): [int(value) for value in row["ranking"]]
            for row in read_jsonl(path)
        }
    if model is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            str(modern.model_locations()[modern.RERANKER_MODEL]),
            max_length=512,
            device=args.device,
            local_files_only=True,
        )
    started = time.time()
    for number, (qid, question, ranking) in enumerate(zip(qids, questions, rankings), start=1):
        if qid not in completed:
            candidates = ranking[:RERANK_DEPTH]
            pairs = [(question, texts[index]) for index in candidates]
            values = np.asarray(model.predict(
                pairs,
                batch_size=args.rerank_batch_size,
                show_progress_bar=False,
            )).reshape(-1)
            order = sorted(range(len(candidates)), key=lambda index: (-float(values[index]), index))
            completed[qid] = [candidates[index] for index in order] + ranking[RERANK_DEPTH:]
        if number % 5 == 0 or number == len(qids):
            save_rankings(path, [qid_ for qid_ in qids if qid_ in completed], [completed[qid_] for qid_ in qids if qid_ in completed])
            atomic_json(meta_path, {**expected, "completed": len(completed), "updated_at": now_iso()})
        print(f"  {name}: {number}/{len(qids)}", flush=True)
    atomic_json(meta_path, {**expected, "completed": len(completed), "elapsed_seconds_this_invocation": time.time() - started, "completed_at": now_iso()})
    return [completed[qid] for qid in qids], model


def section_filter_rankings(
    passages: list[str], titles: list[str], questions: list[str]
) -> list[list[int]]:
    passage_sections = [parse_section(title) for title in titles]
    index = base.BM25([base.tokenize(text) for text in passages])
    rankings = []
    for question in questions:
        scores = index.scores(base.tokenize(question))
        section = parse_section(question)
        candidates = [i for i, value in enumerate(passage_sections) if section and value == section]
        if not candidates:
            rankings.append(base.ranking_from_scores(scores, 5))
        else:
            candidates.sort(key=lambda i: (-scores[i], i))
            rankings.append(candidates[:5])
    return rankings


def reader_lookup() -> dict[tuple[int, int], tuple[str, float]]:
    result = {}
    for row in read_jsonl(READER_CACHE):
        result[(int(row["query"]), int(row["context"]))] = (
            str(row["raw"][0]), float(row["raw"][1])
        )
    return result


def independent_selector_summary(
    selected_rankings: list[list[int]],
    section_rankings: list[list[int]],
    tests: list[dict],
    gold: list[int],
    corpus: list[dict],
) -> dict:
    decoded = reader_lookup()
    rows = []
    for query_index, (reranked, candidates, item, target) in enumerate(
        zip(selected_rankings, section_rankings, tests, gold)
    ):
        selected = reranked[0]
        if selected not in candidates:
            raise AssertionError("independent selector changed the section-filter pool")
        if (query_index, selected) not in decoded:
            raise KeyError(f"reader cache missing pair {(query_index, selected)}")
        prediction, _span_score = decoded[(query_index, selected)]
        em, f1 = compute_em_f1(prediction, item["answer"])
        available = target in candidates
        source_correct = selected == target
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "prediction": prediction,
            "em": em,
            "f1": round(f1, 6),
            "retrieval_hit_at_k": available,
            "selected_context_is_gold": source_correct,
            "candidate_count": len(candidates),
            "selected_context_index": selected,
            "selected_context_title": corpus[selected].get("title", ""),
            "selected_context_act": corpus[selected].get("act", ""),
        })
    write_jsonl(OUT_DIR / "section_filter_bge_selector_predictions.jsonl", rows)
    source_absent = sum(not row["retrieval_hit_at_k"] for row in rows)
    available_other = sum(row["retrieval_hit_at_k"] and not row["selected_context_is_gold"] for row in rows)
    source_non_em = sum(row["selected_context_is_gold"] and not row["em"] for row in rows)
    source_em = sum(row["selected_context_is_gold"] and row["em"] for row in rows)
    available_n = len(rows) - source_absent
    return {
        "n": len(rows),
        "retrieval_recall@5": round(100.0 * available_n / len(rows), 2),
        "selected_annotated_source_pct": round(100.0 * sum(row["selected_context_is_gold"] for row in rows) / len(rows), 2),
        "conditional_source_selection_pct": round(100.0 * sum(row["selected_context_is_gold"] for row in rows) / available_n, 2),
        "em": round(100.0 * sum(row["em"] for row in rows) / len(rows), 2),
        "f1": round(100.0 * sum(row["f1"] for row in rows) / len(rows), 2),
        "annotation_grounded_f1": round(100.0 * sum(row["f1"] if row["selected_context_is_gold"] else 0.0 for row in rows) / len(rows), 2),
        "stage_counts": {
            "source_absent": source_absent,
            "available_other_selected": available_other,
            "annotated_source_selected_non_em": source_non_em,
            "annotated_source_selected_em": source_em,
        },
    }


def write_per_query(
    systems: dict[str, list[list[int]]], qids: list[str], gold: list[int], tests: list[dict]
) -> None:
    path = OUT_DIR / "per_query_retrieval_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qid", "act", "system", "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10"])
        for qid, target, item, index in zip(qids, gold, tests, range(len(qids))):
            for system, rankings in systems.items():
                ranking = rankings[index]
                top = ranking[:10]
                rank = top.index(target) + 1 if target in top else None
                writer.writerow([
                    qid, item.get("act", ""), system,
                    *(int(target in ranking[:k]) for k in METRIC_KS),
                    1.0 / rank if rank else 0.0,
                    1.0 / math.log2(rank + 1) if rank else 0.0,
                ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--dense-batch-size", type=int, default=4)
    parser.add_argument("--rerank-batch-size", type=int, default=2)
    parser.add_argument("--skip-full-rerank", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.dense_batch_size, args.rerank_batch_size) < 1:
        parser.error("thread and batch arguments must be positive")
    modern.configure_threads(args.threads)
    args.device = modern.resolve_device(args.device)
    modern.require_models()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus, passages, titles, title_passages, tests, questions, gold = load_data()
    qids = [str(row["id"]) for row in tests]
    bm25 = title_bm25(title_passages, questions)
    expected_bm25 = metrics(bm25, gold)
    if abs(expected_bm25["recall@5"] - 56.98) > 0.01:
        raise AssertionError(f"title-aware BM25 reproduction failed: {expected_bm25}")
    save_rankings(OUT_DIR / "rankings_title_paragraph_bm25.jsonl", qids, bm25)

    dense = dense_title_rank(title_passages, questions, qids, args)
    fused = rrf(bm25, dense)
    save_rankings(OUT_DIR / "rankings_title_paragraph_rrf_bm25_bge_m3.jsonl", qids, fused)

    section = section_filter_rankings(passages, titles, questions)
    if abs(metrics(section, gold)["recall@5"] - 86.34) > 0.01:
        raise AssertionError(f"section-filter reproduction failed: {metrics(section, gold)}")
    section_reranked, model = rerank(
        "section_filter", section, title_passages, questions, qids, args
    )

    systems = {
        "title_paragraph_bm25": bm25,
        "title_paragraph_bge_m3": dense,
        "title_paragraph_rrf_bm25_bge_m3": fused,
    }
    if not args.skip_full_rerank:
        fused_reranked, _ = rerank(
            "title_paragraph_rrf_bm25_bge_m3", fused, title_passages,
            questions, qids, args, model=model,
        )
        systems["title_paragraph_rrf_bm25_bge_m3_bge_reranker_v2_m3"] = fused_reranked

    results = {name: metrics(rankings, gold) for name, rankings in systems.items()}
    selector = independent_selector_summary(
        section_reranked, section, tests, gold, corpus
    )
    write_per_query(systems, qids, gold, tests)
    payload = {
        "created_at": now_iso(),
        "dataset": "IndicIPR-QA test",
        "n_queries": len(qids),
        "n_passages": len(passages),
        "candidate_text": "section title plus paragraph",
        "models": {
            "dense": {"name": modern.DENSE_MODEL, "revision": modern.DENSE_REVISION},
            "reranker": {"name": modern.RERANKER_MODEL, "revision": modern.RERANKER_REVISION},
        },
        "retrieval_results": results,
        "independent_source_selector": selector,
        "notes": [
            "The source selector reranks only the unchanged section-filter top-five pool.",
            "The fixed QA reader extracts from the independently selected passage; its span logits do not select the passage.",
            "All outcomes remain relative to the retained annotated source and answer.",
        ],
    }
    atomic_json(OUT_DIR / "summary.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
