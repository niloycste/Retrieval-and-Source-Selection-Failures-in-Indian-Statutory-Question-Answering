"""BGE-M3 retrieval and BGE-reranker-v2-m3 extension.

This is a versioned extension to the retrieval paper experiments.  It does not
overwrite any canonical result.  Both datasets use paragraph text only.  The
reranker receives exactly the first 50 candidates from BM25 or BM25+BGE-M3
RRF, and therefore cannot change the corresponding recall@50 ceiling.

Examples
--------
Validate data, metrics, cache logic, and the IndicIPR-QA BM25 reproduction::

    python scripts/04_retrieval/modern_retrieval_extension.py --dry-run

Download and pin the two model snapshots::

    python scripts/04_retrieval/modern_retrieval_extension.py --prepare-models

Run the complete extension (resumable)::

    python scripts/04_retrieval/modern_retrieval_extension.py --dataset all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results" / "experiments" / "retrieval" / "runs" / "modern_bge_extension_v1"
MODEL_ROOT = ROOT / "models" / "retrieval" / "modern_bge_extension_v1"

DENSE_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RRF_K = 60
INDIC_DEPTH = 50
UK_DEPTH = 100
RERANK_DEPTH = 50
METRIC_KS = (1, 3, 5, 10, 50)
EXPECTED_INDIC_BM25 = {1: 13.08, 3: 20.06, 5: 22.67, 10: 27.03}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_fingerprint(*sequences: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        for value in sequence:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def metadata_matches(observed: dict, expected: dict) -> bool:
    return all(observed.get(key) == value for key, value in expected.items())


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INDIC = load_local_module("indic_dense_base", SCRIPT_DIR / "dense_retrieval_experiment.py")
UK = load_local_module("uk_external_base", SCRIPT_DIR / "uk_statutecorpus_external_validation.py")


def configure_threads(threads: int) -> None:
    value = str(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def model_locations() -> dict[str, Path]:
    return {
        DENSE_MODEL: MODEL_ROOT / "bge-m3",
        RERANKER_MODEL: MODEL_ROOT / "bge-reranker-v2-m3",
    }


def prepare_models() -> dict[str, dict]:
    """Download immutable snapshots and record the resolved Hub revisions."""
    from huggingface_hub import HfApi, snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    api = HfApi()
    for repo_id, target in model_locations().items():
        info = api.model_info(repo_id)
        revision = str(info.sha)
        print(f"Resolving {repo_id} -> {revision}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
            # Only PyTorch/Sentence-Transformers inference assets are needed.
            # Excluding ONNX, OpenVINO, images, and training artefacts avoids
            # downloading several redundant multi-gigabyte representations.
            allow_patterns=[
                "*.json", "*.model", "*.txt", "*.py", "*.safetensors",
                "pytorch_model.bin", "modules.json", "1_Pooling/*",
            ],
        )
        manifest[repo_id] = {
            "revision": revision,
            "local_path": str(target.resolve()),
            "resolved_at": now_iso(),
        }
    atomic_json(MODEL_ROOT / "model_manifest.json", manifest)
    return manifest


def require_models() -> dict[str, dict]:
    manifest_path = MODEL_ROOT / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"model manifest missing: {manifest_path}; run with --prepare-models first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        DENSE_MODEL: (DENSE_REVISION, "pytorch_model.bin"),
        RERANKER_MODEL: (RERANKER_REVISION, "model.safetensors"),
    }
    for repo_id, target in model_locations().items():
        revision, weight_file = expected[repo_id]
        if (
            repo_id not in manifest
            or manifest[repo_id].get("revision") != revision
            or not target.exists()
            or not (target / "config.json").exists()
            or not (target / weight_file).exists()
        ):
            raise FileNotFoundError(f"incomplete pinned snapshot for {repo_id}: {target}")
    return manifest


def load_rankings(path: Path) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        qid = str(row["qid"])
        if qid in output:
            raise ValueError(f"duplicate query ID in ranking cache {path}: {qid}")
        output[qid] = [str(v) for v in row["ranking"]]
    return output


def save_rankings(path: Path, qids: Sequence[str], rankings: dict[str, list[str]], scores=None) -> None:
    rows = []
    for qid in qids:
        row = {"qid": qid, "ranking": rankings[qid]}
        if scores is not None:
            row["scores"] = scores[qid]
        rows.append(row)
    write_jsonl(path, rows)


def rrf(first: dict[str, list[str]], second: dict[str, list[str]], qids: Sequence[str], depth: int):
    output: dict[str, list[str]] = {}
    for qid in qids:
        values: defaultdict[str, float] = defaultdict(float)
        for ranking in (first[qid], second[qid]):
            for rank, doc_id in enumerate(ranking):
                values[doc_id] += 1.0 / (RRF_K + rank + 1)
        output[qid] = sorted(values, key=lambda doc_id: (-values[doc_id], doc_id))[:depth]
    return output


def dense_rank(
    texts: Sequence[str],
    text_ids: Sequence[str],
    queries: Sequence[str],
    qids: Sequence[str],
    output_dir: Path,
    corpus_fingerprint: str,
    depth: int,
    batch_size: int,
    chunk_size: int,
    force: bool,
    device: str,
) -> dict[str, list[str]]:
    ranking_path = output_dir / "rankings_bge_m3.jsonl"
    ranking_meta_path = output_dir / "rankings_bge_m3.meta.json"
    ranking_expected = {
        "model": DENSE_MODEL,
        "revision": DENSE_REVISION,
        "corpus_fingerprint": corpus_fingerprint,
        "query_fingerprint": sequence_fingerprint(qids, queries),
        "n_passages": len(texts),
        "n_queries": len(queries),
        "depth": depth,
        "max_seq_length": 512,
        "normalised": True,
        "candidate_text": "paragraph only",
    }
    if ranking_path.exists() and not force:
        if not ranking_meta_path.exists():
            raise ValueError(f"ranking cache lacks provenance metadata: {ranking_meta_path}")
        observed = json.loads(ranking_meta_path.read_text(encoding="utf-8"))
        if not metadata_matches(observed, ranking_expected):
            raise ValueError("BGE-M3 ranking cache metadata mismatch; inspect before using --force")
        return load_rankings(ranking_path)

    from sentence_transformers import SentenceTransformer

    model_path = model_locations()[DENSE_MODEL]
    print(f"Loading pinned BGE-M3: {model_path}", flush=True)
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    # A common explicit ceiling across corpora prevents an accidental long-context
    # advantage; paragraph text (not titles or Act labels) is passed to the model.
    model.max_seq_length = 512
    dimension = int(model.get_sentence_embedding_dimension())
    embedding_path = output_dir / "bge_m3_corpus_embeddings.npy"
    state_path = output_dir / "bge_m3_embedding_state.json"
    expected = {
        "model": DENSE_MODEL,
        "revision": DENSE_REVISION,
        "corpus_fingerprint": corpus_fingerprint,
        "n_passages": len(texts),
        "dimension": dimension,
        "max_seq_length": 512,
        "normalised": True,
        "candidate_text": "paragraph only",
    }
    completed = 0
    if embedding_path.exists() and state_path.exists() and not force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("BGE-M3 embedding cache metadata mismatch; inspect before using --force")
        completed = int(state.get("completed", 0))
        embeddings = np.lib.format.open_memmap(embedding_path, mode="r+")
    else:
        embeddings = np.lib.format.open_memmap(
            embedding_path, mode="w+", dtype="float32", shape=(len(texts), dimension)
        )
    started = time.time()
    for lower in range(completed, len(texts), chunk_size):
        upper = min(lower + chunk_size, len(texts))
        vectors = model.encode(
            list(texts[lower:upper]), batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        embeddings[lower:upper] = vectors.astype("float32", copy=False)
        embeddings.flush()
        atomic_json(state_path, {**expected, "completed": upper, "updated_at": now_iso()})
        print(f"  BGE-M3 encoded {upper:,}/{len(texts):,}", flush=True)

    query_vectors = model.encode(
        list(queries), batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32", copy=False)
    take = min(depth, len(texts))
    rankings: dict[str, list[str]] = {}
    score_rows: dict[str, list[float]] = {}
    for qid, vector in zip(qids, query_vectors):
        values = np.asarray(embeddings @ vector, dtype="float32")
        candidates = np.arange(len(values)) if take == len(values) else np.argpartition(values, -take)[-take:]
        order = np.lexsort((candidates, -values[candidates]))
        ranked = candidates[order]
        rankings[qid] = [text_ids[int(index)] for index in ranked]
        score_rows[qid] = [float(values[int(index)]) for index in ranked]
    save_rankings(ranking_path, qids, rankings, score_rows)
    atomic_json(ranking_meta_path, {**ranking_expected, "completed_at": now_iso()})
    atomic_json(output_dir / "bge_m3_timing.json", {
        "elapsed_seconds_this_invocation": time.time() - started,
        "finished_at": now_iso(),
    })
    return rankings


def rerank(
    name: str,
    base: dict[str, list[str]],
    texts_by_id: dict[str, str],
    queries: Sequence[str],
    qids: Sequence[str],
    output_dir: Path,
    batch_size: int,
    force: bool,
    device: str,
    model=None,
):
    path = output_dir / f"rankings_{name}_bge_reranker_v2_m3.jsonl"
    state_path = output_dir / f"rankings_{name}_bge_reranker_v2_m3.meta.json"
    expected = {
        "model": RERANKER_MODEL,
        "revision": RERANKER_REVISION,
        "base_ranking_fingerprint": sequence_fingerprint(
            qids, [json.dumps(base[qid][:RERANK_DEPTH], separators=(",", ":")) for qid in qids]
        ),
        "query_fingerprint": sequence_fingerprint(qids, queries),
        "n_queries": len(qids),
        "rerank_depth": RERANK_DEPTH,
        "max_seq_length": 512,
        "candidate_text": "paragraph only",
    }
    if path.exists() and not force:
        if not state_path.exists():
            raise ValueError(f"reranker cache lacks provenance metadata: {state_path}")
        observed = json.loads(state_path.read_text(encoding="utf-8"))
        if not metadata_matches(observed, expected):
            raise ValueError(f"reranker cache metadata mismatch for {name}; inspect before using --force")
    completed = {} if force or not path.exists() else load_rankings(path)
    if len(completed) == len(qids):
        return completed, model
    if model is None:
        from sentence_transformers import CrossEncoder
        model_path = model_locations()[RERANKER_MODEL]
        print(f"Loading pinned BGE reranker: {model_path}", flush=True)
        model = CrossEncoder(str(model_path), max_length=512, device=device, local_files_only=True)
    for number, (qid, query) in enumerate(zip(qids, queries), start=1):
        if qid in completed:
            continue
        candidates = base[qid][:RERANK_DEPTH]
        pairs = [(query, texts_by_id[doc_id]) for doc_id in candidates]
        values = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda index: (-float(values[index]), index))
        completed[qid] = [candidates[index] for index in order] + base[qid][RERANK_DEPTH:]
        if number % 5 == 0 or number == len(qids):
            save_rankings(path, [item for item in qids if item in completed], completed)
            atomic_json(state_path, {**expected, "completed": len(completed), "updated_at": now_iso()})
        print(f"  {name} reranked {number}/{len(qids)}", flush=True)
    return completed, model


def indic_metrics(rankings: dict[str, list[str]], qids: Sequence[str], gold: dict[str, str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in METRIC_KS:
        output[f"recall@{k}"] = 100.0 * sum(gold[qid] in rankings[qid][:k] for qid in qids) / len(qids)
    rr = []
    ndcg = []
    for qid in qids:
        top = rankings[qid][:10]
        if gold[qid] in top:
            rank = top.index(gold[qid]) + 1
            rr.append(1.0 / rank)
            ndcg.append(1.0 / math.log2(rank + 1))
        else:
            rr.append(0.0)
            ndcg.append(0.0)
    output["mrr@10"] = 100.0 * sum(rr) / len(rr)
    output["ndcg@10"] = 100.0 * sum(ndcg) / len(ndcg)
    return {key: round(value, 2) for key, value in output.items()}


def write_indic_per_query(
    output_dir: Path,
    systems: dict[str, dict[str, list[str]]],
    qids: Sequence[str],
    gold: dict[str, str],
    acts: Sequence[str],
) -> None:
    path = output_dir / "per_query_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "qid", "act", "system", "recall@1", "recall@3", "recall@5",
            "recall@10", "recall@50", "mrr@10", "ndcg@10",
        ])
        for qid, act in zip(qids, acts):
            for system, rankings in systems.items():
                ranking = rankings[qid]
                row = [qid, act, system]
                row.extend(int(gold[qid] in ranking[:k]) for k in METRIC_KS)
                top = ranking[:10]
                rank = top.index(gold[qid]) + 1 if gold[qid] in top else None
                row.append(1.0 / rank if rank else 0.0)
                row.append(1.0 / math.log2(rank + 1) if rank else 0.0)
                writer.writerow(row)


def run_indic(args) -> dict:
    output_dir = OUT_ROOT / "indic"
    output_dir.mkdir(parents=True, exist_ok=True)
    passages, _acts, questions, gold_indices, q_acts = INDIC.load_data()
    test_rows = INDIC.load_jsonl(INDIC.TEST_FILE)
    qids = [str(row["id"]) for row in test_rows]
    if not (len(qids) == len(questions) == len(gold_indices) == len(q_acts)):
        raise ValueError("IndicIPR-QA query/gold/Act lengths do not align")
    if len(set(qids)) != len(qids):
        raise ValueError("IndicIPR-QA test split contains duplicate question IDs")
    doc_ids = [f"indic-{index:04d}" for index in range(len(passages))]
    gold = {qid: doc_ids[index] for qid, index in zip(qids, gold_indices)}
    bm25_path = output_dir / "rankings_bm25.jsonl"
    if bm25_path.exists() and not args.force:
        bm25 = load_rankings(bm25_path)
    else:
        index = INDIC.BM25([INDIC.tokenize(text) for text in passages])
        lists = [INDIC.ranking_from_scores(index.scores(INDIC.tokenize(query)), INDIC_DEPTH) for query in questions]
        bm25 = {qid: [doc_ids[i] for i in ranking] for qid, ranking in zip(qids, lists)}
        save_rankings(bm25_path, qids, bm25)
    bm25_result = indic_metrics(bm25, qids, gold)
    for k, expected in EXPECTED_INDIC_BM25.items():
        if abs(bm25_result[f"recall@{k}"] - expected) > 0.01:
            raise AssertionError(f"Indic BM25 reproduction failed at {k}: {bm25_result}")
    dense = dense_rank(
        passages, doc_ids, questions, qids, output_dir,
        sha256(INDIC.CORPUS_FILE), INDIC_DEPTH, args.dense_batch_size,
        args.embedding_chunk_size, args.force, args.resolved_device,
    )
    hybrid = rrf(bm25, dense, qids, INDIC_DEPTH)
    save_rankings(output_dir / "rankings_rrf_bm25_bge_m3.jsonl", qids, hybrid)
    text_map = dict(zip(doc_ids, passages))
    bm25_reranked, reranker_model = rerank(
        "bm25", bm25, text_map, questions, qids, output_dir,
        args.rerank_batch_size, args.force, args.resolved_device,
    )
    hybrid_reranked, _ = rerank(
        "rrf_bm25_bge_m3", hybrid, text_map, questions, qids, output_dir,
        args.rerank_batch_size, args.force, args.resolved_device, model=reranker_model,
    )
    systems = {
        "bm25": bm25,
        "bge_m3": dense,
        "rrf_bm25_bge_m3": hybrid,
        "bm25_bge_reranker_v2_m3": bm25_reranked,
        "rrf_bm25_bge_m3_bge_reranker_v2_m3": hybrid_reranked,
    }
    results = {name: indic_metrics(value, qids, gold) for name, value in systems.items()}
    write_indic_per_query(output_dir, systems, qids, gold, q_acts)
    payload = {
        "dataset": "IndicIPR-QA test",
        "n_queries": len(qids), "n_passages": len(passages),
        "candidate_text": "paragraph only", "retrieval_depth": INDIC_DEPTH,
        "rerank_depth": RERANK_DEPTH,
        "candidate_ceilings": {
            "bm25_recall@50": results["bm25"]["recall@50"],
            "rrf_bm25_bge_m3_recall@50": results["rrf_bm25_bge_m3"]["recall@50"],
        },
        "results": results,
    }
    atomic_json(output_dir / "summary.json", payload)
    return payload


def run_uk(args) -> dict:
    output_dir = OUT_ROOT / "uk"
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_file, query_file, source_manifest = UK.prepare_dataset(UK.DEFAULT_ARCHIVE, UK.DEFAULT_EXTRACT_DIR)
    queries = UK.load_queries(query_file)
    required = {doc_id for query in queries for doc_id in query.qrels}
    corpus = UK.load_corpus(corpus_file, required)
    qids = [query.qid for query in queries]
    query_texts = [query.query for query in queries]
    legacy_bm25 = UK.DEFAULT_OUTPUT_DIR / "cache" / "rankings_bm25.jsonl"
    if legacy_bm25.exists() and not args.force:
        bm25 = UK.load_rankings(legacy_bm25)
        if set(bm25) != set(qids) or any(len(bm25[qid]) < UK_DEPTH for qid in qids):
            raise ValueError("legacy UK BM25 cache does not match Eval100/depth-100")
    else:
        bm25 = UK.run_bm25(corpus, queries, UK_DEPTH, output_dir, args.force)
    save_rankings(output_dir / "rankings_bm25.jsonl", qids, bm25)
    dense = dense_rank(
        corpus.texts, corpus.ids, query_texts, qids, output_dir,
        source_manifest["corpus_sha256"], UK_DEPTH, args.dense_batch_size,
        args.embedding_chunk_size, args.force, args.resolved_device,
    )
    hybrid = rrf(bm25, dense, qids, UK_DEPTH)
    save_rankings(output_dir / "rankings_rrf_bm25_bge_m3.jsonl", qids, hybrid)
    text_map = dict(zip(corpus.ids, corpus.texts))
    bm25_reranked, reranker_model = rerank(
        "bm25", bm25, text_map, query_texts, qids, output_dir,
        args.rerank_batch_size, args.force, args.resolved_device,
    )
    hybrid_reranked, _ = rerank(
        "rrf_bm25_bge_m3", hybrid, text_map, query_texts, qids, output_dir,
        args.rerank_batch_size, args.force, args.resolved_device, model=reranker_model,
    )
    systems = {
        "bm25": bm25,
        "bge_m3": dense,
        "rrf_bm25_bge_m3": hybrid,
        "bm25_bge_reranker_v2_m3": bm25_reranked,
        "rrf_bm25_bge_m3_bge_reranker_v2_m3": hybrid_reranked,
    }
    summary = UK.evaluate_all(queries, systems, output_dir, args.bootstrap_resamples, debug=False)
    summary["candidate_ceilings"] = {
        "bm25_recall@50": summary["systems"]["bm25"]["recall@50"]["mean"],
        "rrf_bm25_bge_m3_recall@50": summary["systems"]["rrf_bm25_bge_m3"]["recall@50"]["mean"],
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def write_manifest(args, model_manifest: dict, results: dict) -> None:
    try:
        import importlib.metadata as metadata
        versions = {name: metadata.version(name) for name in ("torch", "transformers", "sentence-transformers", "numpy")}
    except Exception:
        versions = {}
    atomic_json(OUT_ROOT / "manifest.json", {
        "created_at": now_iso(),
        "protocol": {
            "dense": DENSE_MODEL, "reranker": RERANKER_MODEL,
            "candidate_text": "paragraph only", "rerank_depth": RERANK_DEPTH,
            "rrf_k": RRF_K, "max_seq_length": 512,
            "metrics": ["recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10"],
        },
        "models": model_manifest,
        "runtime": {"platform": platform.platform(), "python": platform.python_version(), "device": args.resolved_device, "packages": versions, "command": sys.argv},
        "completed_datasets": sorted(results),
    })


def dry_run() -> None:
    UK.run_self_tests()
    passages, _acts, queries, gold_indices, _q_acts = INDIC.load_data()
    bm25 = INDIC.BM25([INDIC.tokenize(text) for text in passages])
    lists = [INDIC.ranking_from_scores(bm25.scores(INDIC.tokenize(query)), INDIC_DEPTH) for query in queries]
    qids = [f"q{i}" for i in range(len(queries))]
    doc_ids = [f"d{i}" for i in range(len(passages))]
    rankings = {qid: [doc_ids[j] for j in ranking] for qid, ranking in zip(qids, lists)}
    gold = {qid: doc_ids[j] for qid, j in zip(qids, gold_indices)}
    result = indic_metrics(rankings, qids, gold)
    for k, expected in EXPECTED_INDIC_BM25.items():
        assert abs(result[f"recall@{k}"] - expected) <= 0.01, (k, result, expected)
    toy_first = {"q": ["a", "b", "c"]}
    toy_second = {"q": ["b", "a", "d"]}
    assert rrf(toy_first, toy_second, ["q"], 4)["q"][:2] == ["a", "b"]

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.max_seq_length = None

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, values, **kwargs):
            vectors = []
            for value in values:
                lowered = value.lower()
                vector = np.asarray([
                    1.0 if "alpha" in lowered else 0.0,
                    1.0 if "beta" in lowered else 0.0,
                    0.1,
                ], dtype="float32")
                if kwargs.get("normalize_embeddings"):
                    vector /= np.linalg.norm(vector)
                vectors.append(vector)
            return np.stack(vectors)

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs, **kwargs):
            return np.asarray([
                sum(token in passage.lower() for token in query.lower().split())
                for query, passage in pairs
            ], dtype="float32")

    fake_module = types.SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer,
        CrossEncoder=FakeCrossEncoder,
    )
    previous = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_module
    try:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            toy_texts = ["alpha statute", "beta law", "gamma rule"]
            toy_ids = ["d0", "d1", "d2"]
            toy_queries = ["alpha", "beta"]
            toy_qids = ["q0", "q1"]
            dense = dense_rank(
                toy_texts, toy_ids, toy_queries, toy_qids, temporary_path,
                "toy-corpus", 3, 2, 2, False, "cpu",
            )
            assert dense["q0"][0] == "d0" and dense["q1"][0] == "d1"
            assert dense_rank(
                toy_texts, toy_ids, toy_queries, toy_qids, temporary_path,
                "toy-corpus", 3, 2, 2, False, "cpu",
            ) == dense
            base = {"q0": ["d2", "d0", "d1"], "q1": ["d2", "d1", "d0"]}
            reranked, model = rerank(
                "toy", base, dict(zip(toy_ids, toy_texts)), toy_queries,
                toy_qids, temporary_path, 2, False, "cpu",
            )
            assert reranked["q0"][0] == "d0" and reranked["q1"][0] == "d1"
            resumed, _ = rerank(
                "toy", base, dict(zip(toy_ids, toy_texts)), toy_queries,
                toy_qids, temporary_path, 2, False, "cpu", model=model,
            )
            assert resumed == reranked
    finally:
        if previous is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = previous

    print(json.dumps({
        "status": "PASS", "indic_bm25": result,
        "uk_self_tests": "PASS", "mock_dense_rerank_resume": "PASS",
    }, indent=2))


def model_smoke_test(device: str) -> None:
    """Exercise both pinned models on tiny real-shaped inputs."""
    require_models()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    dense = SentenceTransformer(
        str(model_locations()[DENSE_MODEL]), device=device, local_files_only=True
    )
    dense.max_seq_length = 512
    vectors = dense.encode(
        ["What does Section 4 provide?", "This Act applies to copyright."],
        batch_size=1, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )
    if vectors.shape[0] != 2 or not np.isfinite(vectors).all():
        raise AssertionError(f"invalid BGE-M3 smoke output: {vectors.shape}")
    del dense

    reranker = CrossEncoder(
        str(model_locations()[RERANKER_MODEL]), max_length=512,
        device=device, local_files_only=True,
    )
    scores = np.asarray(reranker.predict([
        ("What does Section 4 provide?", "Section 4 provides a statutory rule."),
        ("What does Section 4 provide?", "A passage about an unrelated topic."),
    ], batch_size=1, show_progress_bar=False))
    if scores.size != 2 or not np.isfinite(scores).all():
        raise AssertionError(f"invalid BGE reranker smoke output: {scores.shape}")
    payload = {
        "status": "PASS", "completed_at": now_iso(), "device": device,
        "dense_shape": list(vectors.shape),
        "dense_norms": [float(value) for value in np.linalg.norm(vectors, axis=1)],
        "reranker_scores": [float(value) for value in scores.reshape(-1)],
    }
    atomic_json(OUT_ROOT / "MODEL_SMOKE_TEST.json", payload)
    print(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("indic", "uk", "all"), default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dense-batch-size", type=int, default=2)
    parser.add_argument("--rerank-batch-size", type=int, default=2)
    parser.add_argument("--embedding-chunk-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--prepare-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-models", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("threads", "dense_batch_size", "rerank_batch_size", "embedding_chunk_size", "bootstrap_resamples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    configure_threads(args.threads)
    args.resolved_device = resolve_device(args.device)
    print(f"Compute device: {args.resolved_device}", flush=True)
    if args.dry_run:
        dry_run()
        return 0
    if args.smoke_models:
        model_smoke_test(args.resolved_device)
        return 0
    model_manifest = prepare_models() if args.prepare_models else require_models()
    if args.prepare_models:
        print(f"Pinned model snapshots prepared in {MODEL_ROOT}")
        return 0
    results = {}
    if args.dataset in ("indic", "all"):
        results["indic"] = run_indic(args)
    if args.dataset in ("uk", "all"):
        results["uk"] = run_uk(args)
    write_manifest(args, model_manifest, results)
    atomic_json(OUT_ROOT / "RUN_COMPLETE.json", {"completed_at": now_iso(), "datasets": sorted(results)})
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
