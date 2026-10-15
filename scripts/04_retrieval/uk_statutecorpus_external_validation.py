"""Zero-shot external retrieval validation on UK-StatuteCorpus Eval100.

This script implements the frozen protocol in
``docs/RETRIEVAL_EXTERNAL_VALIDATION_PROTOCOL.md``.  It deliberately does not
run extractive QA: Eval100 supplies graded passage relevance, not answer spans.

The pipeline is cache-aware and CPU-resource-capped so that it can run beside
the serial DAPT study when explicitly requested.  A normal full run executes:

    prepare -> reference audit -> BM25 -> E5 -> RRF -> cross-encoder -> metrics

Examples
--------
Validate/extract the downloaded archive and inspect the schema::

    python scripts/04_retrieval/uk_statutecorpus_external_validation.py --prepare-only

Run an inexpensive synthetic self-test::

    python scripts/04_retrieval/uk_statutecorpus_external_validation.py --self-test

Run the complete preregistered experiment with four CPU threads::

    python scripts/04_retrieval/uk_statutecorpus_external_validation.py --threads 4

Debug the real-data path without producing canonical results::

    python scripts/04_retrieval/uk_statutecorpus_external_validation.py \
        --limit 3 --corpus-limit 5000 --bootstrap-resamples 200 \
        --output-dir results/experiments/retrieval/runs/uk_statutecorpus_smoke
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
import zipfile
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOURCE_DIR = ROOT / "data" / "external" / "uk_statutecorpus"
DEFAULT_ARCHIVE = DEFAULT_RESOURCE_DIR / "uk-statutecorpus_v1.0.zip"
DEFAULT_EXTRACT_DIR = DEFAULT_RESOURCE_DIR / "extracted"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "experiments" / "retrieval" / "runs"
    / "uk_statutecorpus_external_v1"
)

SOURCE_URL = "https://collections.durham.ac.uk/files/r14x51hj064"
DOWNLOAD_URL = "https://collections.durham.ac.uk/downloads/r14x51hj064"
EXPECTED_ARCHIVE_MD5 = "29907ffa565a5e421c851ab2bc661423"

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
DENSE_MODEL = "intfloat/e5-small-v2"
DENSE_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
DENSE_QUERY_PREFIX = "query: "
DENSE_PASSAGE_PREFIX = "passage: "
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
DEFAULT_RETRIEVAL_DEPTH = 100
DEFAULT_RERANK_DEPTH = 50
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
SEED = 42
METRIC_KS = (1, 3, 5, 10, 50)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
REFERENCE_RE = re.compile(
    r"\b(?:section|sections|regulation|regulations|article|articles|rule|rules|"
    r"paragraph|paragraphs|schedule|schedules)\s+"
    r"(?:\d+[A-Za-z]?(?:\s*\([A-Za-z0-9ivxlcdm]+\))*|[IVXLCDM]+)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryRecord:
    qid: str
    query: str
    qrels: dict[str, int]


@dataclass
class Corpus:
    ids: list[str]
    texts: list[str]
    metadata: list[dict]

    def __post_init__(self) -> None:
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(self.ids)}
        if len(self.id_to_index) != len(self.ids):
            raise ValueError("UK corpus contains duplicate chunk identifiers")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def configure_threads(threads: int) -> None:
    value = str(max(1, threads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch

        torch.set_num_threads(max(1, threads))
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(resolved_destination)
            except ValueError as exc:
                raise ValueError(f"unsafe archive member: {member.filename}") from exc
        zipped.extractall(destination)


def discover_dataset_files(directory: Path) -> tuple[Path, Path]:
    files = [path for path in directory.rglob("*") if path.is_file()]
    corpus_matches = [path for path in files if path.name.lower() == "data.json"]
    if not corpus_matches:
        corpus_matches = [
            path
            for path in files
            if path.name.lower().endswith(".jsonl.gz")
            and "corpus" in {part.lower() for part in path.parts}
        ]
    if len(corpus_matches) != 1:
        raise FileNotFoundError(
            f"expected one Data.json or corpus/*.jsonl.gz under {directory}; "
            f"found {corpus_matches}"
        )

    preferred = [path for path in files if path.name.lower() == "queries.jsonl"]
    if len(preferred) == 1:
        return corpus_matches[0], preferred[0]

    query_matches: list[Path] = []
    for path in files:
        if path.suffix.lower() != ".jsonl":
            continue
        try:
            first = next(
                line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            row = json.loads(first)
        except (OSError, StopIteration, json.JSONDecodeError):
            continue
        if "query" in row and ("qid" in row or "query_id" in row):
            query_matches.append(path)
    if len(query_matches) != 1:
        raise FileNotFoundError(
            f"could not uniquely identify Eval100 JSONL under {directory}: "
            f"{query_matches}"
        )
    return corpus_matches[0], query_matches[0]


def prepare_dataset(archive: Path, extract_dir: Path) -> tuple[Path, Path, dict]:
    if not archive.exists():
        raise FileNotFoundError(
            f"dataset archive is missing: {archive}\nDownload it from {DOWNLOAD_URL}"
        )
    archive_md5 = hash_file(archive, "md5")
    archive_sha256 = hash_file(archive)
    if archive_md5.lower() != EXPECTED_ARCHIVE_MD5:
        raise ValueError(
            f"archive MD5 mismatch: got {archive_md5}, expected {EXPECTED_ARCHIVE_MD5}"
        )
    try:
        corpus_file, query_file = discover_dataset_files(extract_dir)
    except FileNotFoundError:
        print(f"Extracting verified archive to {extract_dir}", flush=True)
        safe_extract(archive, extract_dir)
        corpus_file, query_file = discover_dataset_files(extract_dir)

    source_manifest = {
        "source_url": SOURCE_URL,
        "download_url": DOWNLOAD_URL,
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_md5": archive_md5,
        "archive_sha256": archive_sha256,
        "corpus_file": str(corpus_file.resolve()),
        "corpus_sha256": hash_file(corpus_file),
        "query_file": str(query_file.resolve()),
        "query_sha256": hash_file(query_file),
        "prepared_at": now_iso(),
    }
    atomic_write_json(extract_dir / "LOCAL_SOURCE_MANIFEST.json", source_manifest)
    return corpus_file, query_file, source_manifest


def parse_relevant(row: dict) -> dict[str, int]:
    relevant = row.get("relevant")
    if not relevant:
        relevant = row.get("answers", [])
    parsed: dict[str, int] = {}
    if isinstance(relevant, dict):
        for chunk_id, value in relevant.items():
            if isinstance(value, dict):
                grade = value.get("rel", value.get("relevance", 1))
            else:
                grade = value
            parsed[normalize_id(chunk_id)] = int(grade)
        return {key: value for key, value in parsed.items() if key}
    if not isinstance(relevant, list):
        return {}
    for item in relevant:
        if not isinstance(item, dict):
            continue
        chunk_id = normalize_id(
            item.get("chunk_id", item.get("doc_id", item.get("id")))
        )
        if not chunk_id:
            continue
        grade = int(item.get("rel", item.get("relevance", item.get("grade", 1))))
        parsed[chunk_id] = max(grade, parsed.get(chunk_id, 0))
    return parsed


def load_queries(path: Path, limit: int | None = None) -> list[QueryRecord]:
    queries: list[QueryRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = normalize_id(row.get("qid", row.get("query_id", row.get("id"))))
            query = str(row.get("query", row.get("question", ""))).strip()
            qrels = parse_relevant(row)
            if not qid or not query or not qrels:
                raise ValueError(
                    f"invalid Eval100 row: qid={qid!r}, query={bool(query)}, "
                    f"qrels={len(qrels)}"
                )
            queries.append(QueryRecord(qid=qid, query=query, qrels=qrels))
            if limit is not None and len(queries) >= limit:
                break
    if not queries:
        raise ValueError(f"no queries loaded from {path}")
    if len({query.qid for query in queries}) != len(queries):
        raise ValueError("duplicate query identifiers in Eval100")
    return queries


def load_corpus(
    path: Path,
    required_ids: set[str],
    corpus_limit: int | None = None,
) -> Corpus:
    print(f"Loading corpus: {path}", flush=True)
    ids: list[str] = []
    texts: list[str] = []
    metadata_rows: list[dict] = []
    selected_required: set[str] = set()

    if path.name.lower().endswith(".jsonl.gz"):
        stream = gzip.open(path, "rt", encoding="utf-8")
        rows: Iterable[object] = (json.loads(line) for line in stream if line.strip())
    else:
        stream = path.open("r", encoding="utf-8")
        loaded = json.load(stream)
        if not isinstance(loaded, list):
            stream.close()
            raise ValueError("Data.json must contain a JSON array")
        rows = loaded

    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_id = normalize_id(
            metadata.get("chunk_id", metadata.get("doc_id", item.get("chunk_id")))
        )
        content = str(item.get("content", item.get("text", ""))).strip()
        if not chunk_id or not content:
            continue
        keep = corpus_limit is None or len(ids) < corpus_limit or chunk_id in required_ids
        if not keep:
            continue
        ids.append(chunk_id)
        texts.append(content)
        metadata_rows.append(metadata)
        if chunk_id in required_ids:
            selected_required.add(chunk_id)
        if (index + 1) % 25_000 == 0:
            print(f"  scanned {index + 1:,} corpus passages", flush=True)
    stream.close()

    missing = sorted(required_ids - selected_required)
    if missing:
        raise ValueError(f"{len(missing)} released qrel IDs are absent from Data.json: {missing[:5]}")
    corpus = Corpus(ids=ids, texts=texts, metadata=metadata_rows)
    print(f"Loaded {len(corpus.ids):,} usable passages", flush=True)
    return corpus


def audit_references(queries: Sequence[QueryRecord], output_dir: Path) -> dict:
    rows: list[dict] = []
    for query in queries:
        matches = REFERENCE_RE.findall(query.query)
        rows.append(
            {
                "qid": query.qid,
                "has_explicit_locator": bool(matches),
                "matches": matches,
                "query": query.query,
            }
        )
    report = {
        "n_queries": len(rows),
        "n_with_explicit_locator": sum(row["has_explicit_locator"] for row in rows),
        "share_with_explicit_locator": (
            sum(row["has_explicit_locator"] for row in rows) / len(rows)
        ),
        "regex": REFERENCE_RE.pattern,
        "note": "Text-only audit; no gold-derived citations were inserted.",
        "rows": rows,
    }
    atomic_write_json(output_dir / "reference_audit.json", report)
    return report


class BM25Index:
    """Memory-conscious postings index with the frozen BM25 formula."""

    def __init__(self, documents: Sequence[str], k1: float, b: float):
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_lengths = array("I")
        mutable: dict[str, tuple[array, array]] = {}
        for index, document in enumerate(documents):
            terms = tokenize(document)
            self.doc_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                posting = mutable.get(term)
                if posting is None:
                    posting = (array("I"), array("I"))
                    mutable[term] = posting
                posting[0].append(index)
                posting[1].append(frequency)
            if (index + 1) % 10_000 == 0:
                print(f"  BM25 indexed {index + 1:,}/{self.n_docs:,}", flush=True)
        self.postings = mutable
        self.avg_doc_length = sum(self.doc_lengths) / max(self.n_docs, 1)

    def rank(self, query: str, depth: int) -> tuple[list[int], list[float]]:
        scores: defaultdict[int, float] = defaultdict(float)
        for term in tokenize(query):
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_ids, frequencies = posting
            document_frequency = len(doc_ids)
            inverse_document_frequency = math.log(
                1.0
                + (self.n_docs - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for doc_index, frequency in zip(doc_ids, frequencies):
                denominator = frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b
                    * self.doc_lengths[doc_index]
                    / max(self.avg_doc_length, 1e-9)
                )
                scores[doc_index] += (
                    inverse_document_frequency
                    * frequency
                    * (self.k1 + 1.0)
                    / denominator
                )
        top = heapq.nsmallest(
            min(depth, len(scores)),
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        indices = [index for index, _ in top]
        values = [float(score) for _, score in top]
        if len(indices) < min(depth, self.n_docs):
            used = set(indices)
            for index in range(self.n_docs):
                if index not in used:
                    indices.append(index)
                    values.append(0.0)
                    if len(indices) >= min(depth, self.n_docs):
                        break
        return indices, values


def ranking_rows(
    queries: Sequence[QueryRecord],
    rankings: dict[str, list[str]],
    scores: dict[str, list[float]] | None = None,
) -> list[dict]:
    rows = []
    for query in queries:
        row = {"qid": query.qid, "ranking": rankings[query.qid]}
        if scores is not None:
            row["scores"] = scores[query.qid]
        rows.append(row)
    return rows


def load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rankings[normalize_id(row["qid"])] = [
                    normalize_id(value) for value in row["ranking"]
                ]
    return rankings


def run_bm25(
    corpus: Corpus,
    queries: Sequence[QueryRecord],
    depth: int,
    cache_dir: Path,
    force: bool,
) -> dict[str, list[str]]:
    path = cache_dir / "rankings_bm25.jsonl"
    if path.exists() and not force:
        print(f"Reusing {path}", flush=True)
        return load_rankings(path)
    started = time.time()
    index = BM25Index(corpus.texts, k1=BM25_K1, b=BM25_B)
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for number, query in enumerate(queries, start=1):
        indices, values = index.rank(query.query, depth)
        rankings[query.qid] = [corpus.ids[index] for index in indices]
        scores[query.qid] = values
        print(f"  BM25 query {number}/{len(queries)}", flush=True)
    write_jsonl(path, ranking_rows(queries, rankings, scores))
    atomic_write_json(
        cache_dir / "bm25_timing.json",
        {"elapsed_seconds": time.time() - started, "finished_at": now_iso()},
    )
    return rankings


def dense_state_matches(state: dict, expected: dict) -> bool:
    return all(state.get(key) == value for key, value in expected.items())


def run_dense(
    corpus: Corpus,
    queries: Sequence[QueryRecord],
    corpus_sha256: str,
    depth: int,
    cache_dir: Path,
    batch_size: int,
    embedding_chunk_size: int,
    force: bool,
) -> dict[str, list[str]]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rankings_path = cache_dir / "rankings_e5_small_v2.jsonl"
    if rankings_path.exists() and not force:
        print(f"Reusing {rankings_path}", flush=True)
        return load_rankings(rankings_path)

    started = time.time()
    print(f"Loading dense model {DENSE_MODEL}", flush=True)
    model = SentenceTransformer(
        DENSE_MODEL,
        device="cpu",
        revision=DENSE_REVISION,
        local_files_only=True,
    )
    dimension = int(model.get_sentence_embedding_dimension())
    embedding_path = cache_dir / "e5_small_v2_corpus_embeddings.npy"
    state_path = cache_dir / "e5_small_v2_embedding_state.json"
    expected = {
            "model": DENSE_MODEL,
            "revision": DENSE_REVISION,
        "corpus_sha256": corpus_sha256,
        "n_passages": len(corpus.ids),
        "dimension": dimension,
    }

    completed = 0
    if embedding_path.exists() and state_path.exists() and not force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not dense_state_matches(state, expected):
            raise ValueError(
                "dense embedding cache does not match the current corpus/model; "
                "use --force only after reviewing the mismatch"
            )
        completed = int(state.get("completed", 0))
        embeddings = np.lib.format.open_memmap(embedding_path, mode="r+")
    else:
        embeddings = np.lib.format.open_memmap(
            embedding_path,
            mode="w+",
            dtype="float32",
            shape=(len(corpus.ids), dimension),
        )

    for lower in range(completed, len(corpus.ids), embedding_chunk_size):
        upper = min(lower + embedding_chunk_size, len(corpus.ids))
        vectors = model.encode(
            [DENSE_PASSAGE_PREFIX + text for text in corpus.texts[lower:upper]],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings[lower:upper] = vectors.astype("float32", copy=False)
        embeddings.flush()
        atomic_write_json(
            state_path,
            {
                **expected,
                "completed": upper,
                "updated_at": now_iso(),
                "complete": upper == len(corpus.ids),
            },
        )
        print(f"  E5 encoded {upper:,}/{len(corpus.ids):,}", flush=True)

    query_vectors = model.encode(
        [DENSE_QUERY_PREFIX + query.query for query in queries],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32", copy=False)
    similarities = query_vectors @ embeddings.T
    rankings: dict[str, list[str]] = {}
    scores_out: dict[str, list[float]] = {}
    take = min(depth, len(corpus.ids))
    for row_index, query in enumerate(queries):
        scores = similarities[row_index]
        if take == len(scores):
            candidates = np.arange(len(scores))
        else:
            candidates = np.argpartition(scores, -take)[-take:]
        order = np.lexsort((candidates, -scores[candidates]))
        ranked = candidates[order]
        rankings[query.qid] = [corpus.ids[int(index)] for index in ranked]
        scores_out[query.qid] = [float(scores[int(index)]) for index in ranked]
    del similarities
    write_jsonl(rankings_path, ranking_rows(queries, rankings, scores_out))
    atomic_write_json(
        cache_dir / "dense_timing.json",
        {"elapsed_seconds": time.time() - started, "finished_at": now_iso()},
    )
    return rankings


def rrf_fuse(
    queries: Sequence[QueryRecord],
    first: dict[str, list[str]],
    second: dict[str, list[str]],
    depth: int,
    cache_dir: Path,
    force: bool,
) -> dict[str, list[str]]:
    path = cache_dir / "rankings_rrf_bm25_e5.jsonl"
    if path.exists() and not force:
        return load_rankings(path)
    rankings: dict[str, list[str]] = {}
    scores_out: dict[str, list[float]] = {}
    for query in queries:
        scores: defaultdict[str, float] = defaultdict(float)
        for ranking in (first[query.qid], second[query.qid]):
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] += 1.0 / (RRF_K + rank + 1)
        ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:depth]
        rankings[query.qid] = ordered
        scores_out[query.qid] = [scores[doc_id] for doc_id in ordered]
    write_jsonl(path, ranking_rows(queries, rankings, scores_out))
    return rankings


def run_reranker(
    name: str,
    base_rankings: dict[str, list[str]],
    corpus: Corpus,
    queries: Sequence[QueryRecord],
    rerank_depth: int,
    cache_dir: Path,
    batch_size: int,
    force: bool,
    model: object | None = None,
) -> tuple[dict[str, list[str]], object]:
    output_path = cache_dir / f"rankings_{name}_ce.jsonl"
    existing = {} if force or not output_path.exists() else load_rankings(output_path)
    if len(existing) == len(queries):
        print(f"Reusing {output_path}", flush=True)
        return existing, model

    if model is None:
        from sentence_transformers import CrossEncoder

        print(f"Loading cross-encoder {CROSS_ENCODER_MODEL}", flush=True)
        model = CrossEncoder(
            CROSS_ENCODER_MODEL,
            max_length=512,
            device="cpu",
            revision=CROSS_ENCODER_REVISION,
            local_files_only=True,
        )
    completed = dict(existing)
    for number, query in enumerate(queries, start=1):
        if query.qid in completed:
            continue
        base = base_rankings[query.qid]
        head = base[:rerank_depth]
        pairs = [(query.query, corpus.texts[corpus.id_to_index[doc_id]]) for doc_id in head]
        values = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        order = sorted(range(len(head)), key=lambda index: (-float(values[index]), index))
        reranked_head = [head[index] for index in order]
        completed[query.qid] = reranked_head + base[rerank_depth:]
        if number % 5 == 0 or number == len(queries):
            write_jsonl(output_path, ranking_rows(
                [item for item in queries if item.qid in completed], completed
            ))
        print(f"  {name} cross-encoder query {number}/{len(queries)}", flush=True)
    return completed, model


def dcg(grades: Sequence[int]) -> float:
    return sum(
        (2.0 ** grade - 1.0) / math.log2(rank + 2.0)
        for rank, grade in enumerate(grades)
    )


def per_query_metrics(
    queries: Sequence[QueryRecord], rankings: dict[str, list[str]]
) -> tuple[list[str], dict[str, list[float]]]:
    qids: list[str] = []
    metrics: dict[str, list[float]] = defaultdict(list)
    for query in queries:
        ranking = rankings[query.qid]
        relevant = set(query.qrels)
        qids.append(query.qid)
        for k in METRIC_KS:
            metrics[f"recall@{k}"].append(
                len(relevant.intersection(ranking[:k])) / len(relevant)
            )
        top_ten = ranking[:10]
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(top_ten, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        metrics["mrr@10"].append(reciprocal_rank)
        observed_grades = [query.qrels.get(doc_id, 0) for doc_id in top_ten]
        ideal_grades = sorted(query.qrels.values(), reverse=True)[:10]
        denominator = dcg(ideal_grades)
        metrics["ndcg@10"].append(dcg(observed_grades) / denominator if denominator else 0.0)
        grade_three = {doc_id for doc_id, grade in query.qrels.items() if grade == 3}
        if not grade_three:
            raise ValueError(f"query {query.qid} has no rel=3 provision")
        for k in (5, 10):
            metrics[f"grade3_recall@{k}"].append(
                len(grade_three.intersection(ranking[:k])) / len(grade_three)
            )
    return qids, dict(metrics)


def bootstrap_interval(values: Sequence[float], indices: object) -> tuple[float, float]:
    import numpy as np

    vector = np.asarray(values, dtype="float64")
    means = vector[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def evaluate_all(
    queries: Sequence[QueryRecord],
    systems: dict[str, dict[str, list[str]]],
    output_dir: Path,
    bootstrap_resamples: int,
    debug: bool,
) -> dict:
    import numpy as np

    per_system: dict[str, dict[str, list[float]]] = {}
    qids: list[str] | None = None
    for name, rankings in systems.items():
        current_qids, values = per_query_metrics(queries, rankings)
        if qids is not None and current_qids != qids:
            raise ValueError("query order mismatch between systems")
        qids = current_qids
        per_system[name] = values

    rng = np.random.default_rng(SEED)
    indices = rng.integers(
        0, len(queries), size=(bootstrap_resamples, len(queries)), endpoint=False
    )
    summary: dict[str, dict] = {}
    for name, metrics in per_system.items():
        summary[name] = {}
        for metric, values in metrics.items():
            lower, upper = bootstrap_interval(values, indices)
            summary[name][metric] = {
                "mean": float(np.mean(values)),
                "ci95": [lower, upper],
            }

    legacy_pairs = (
        ("bm25", "e5_small_v2"),
        ("bm25", "rrf_bm25_e5"),
        ("bm25", "bm25_ce"),
        ("rrf_bm25_e5", "rrf_bm25_e5_ce"),
    )
    modern_bge_pairs = (
        ("bm25", "bge_m3"),
        ("bm25", "rrf_bm25_bge_m3"),
        ("bm25", "bm25_bge_reranker_v2_m3"),
        (
            "rrf_bm25_bge_m3",
            "rrf_bm25_bge_m3_bge_reranker_v2_m3",
        ),
    )
    # The evaluator is shared by the original E5 experiment and the modern
    # BGE extension.  Compute only comparisons whose two systems are present;
    # this preserves the frozen legacy analysis while avoiding assumptions
    # about system names in extensions.
    planned_pairs = tuple(
        pair
        for pair in (*legacy_pairs, *modern_bge_pairs)
        if pair[0] in per_system and pair[1] in per_system
    )
    differences: dict[str, dict] = {}
    for baseline, comparison in planned_pairs:
        label = f"{comparison}_minus_{baseline}"
        differences[label] = {}
        for metric in per_system[baseline]:
            diff = np.asarray(per_system[comparison][metric]) - np.asarray(
                per_system[baseline][metric]
            )
            lower, upper = bootstrap_interval(diff, indices)
            differences[label][metric] = {
                "mean": float(np.mean(diff)),
                "ci95": [lower, upper],
            }

    payload = {
        "canonical": not debug,
        "n_queries": len(queries),
        "bootstrap": {
            "seed": SEED,
            "resamples": bootstrap_resamples,
            "interval": "percentile 95%",
        },
        "metric_definition": {
            "ndcg_gain": "2^rel - 1",
            "binary_relevance": "all released rel>0 passages",
        },
        "systems": summary,
        "planned_differences": differences,
    }
    filename = "summary_debug.json" if debug else "summary.json"
    atomic_write_json(output_dir / filename, payload)

    per_query_path = output_dir / ("per_query_debug.csv" if debug else "per_query.csv")
    metric_names = list(next(iter(per_system.values())))
    with per_query_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qid", "system", *metric_names])
        for system, metrics in per_system.items():
            for row_index, qid in enumerate(qids or []):
                writer.writerow(
                    [qid, system, *[metrics[metric][row_index] for metric in metric_names]]
                )

    table_path = output_dir / ("uk_external_table_debug.tex" if debug else "uk_external_table.tex")
    with table_path.open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        stream.write("System & nDCG@10 & MRR@10 & R@5 & R@10 & Grade-3 R@10 \\\\\n")
        stream.write("\\midrule\n")
        labels = {
            "bm25": "BM25",
            "e5_small_v2": "E5-small-v2",
            "rrf_bm25_e5": "BM25 + E5 RRF",
            "bm25_ce": "BM25 + cross-encoder",
            "rrf_bm25_e5_ce": "BM25 + E5 RRF + cross-encoder",
            "bge_m3": "BGE-M3",
            "rrf_bm25_bge_m3": "BM25 + BGE-M3 RRF",
            "bm25_bge_reranker_v2_m3": "BM25 + BGE reranker",
            "rrf_bm25_bge_m3_bge_reranker_v2_m3": (
                "BM25 + BGE-M3 RRF + BGE reranker"
            ),
        }
        for system in systems:
            values = summary[system]
            cells = [
                values[metric]["mean"] * 100
                for metric in (
                    "ndcg@10",
                    "mrr@10",
                    "recall@5",
                    "recall@10",
                    "grade3_recall@10",
                )
            ]
            stream.write(
                f"{labels.get(system, system)} & "
                + " & ".join(f"{value:.2f}" for value in cells)
                + " \\\\\n"
            )
        stream.write("\\bottomrule\n\\end{tabular}\n")
    return payload


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_run_manifest(
    args: argparse.Namespace,
    source_manifest: dict,
    corpus: Corpus,
    queries: Sequence[QueryRecord],
    reference_audit: dict,
) -> None:
    manifest = {
        "protocol": "docs/RETRIEVAL_EXTERNAL_VALIDATION_PROTOCOL.md",
        "created_at": now_iso(),
        "canonical": args.limit is None and args.corpus_limit is None,
        "source": source_manifest,
        "data": {
            "corpus_passages_used": len(corpus.ids),
            "queries_used": len(queries),
            "qrels_used": sum(len(query.qrels) for query in queries),
            "relevance_grade_counts": dict(sorted(Counter(
                grade for query in queries for grade in query.qrels.values()
            ).items())),
        },
        "reference_audit": {
            "n_with_explicit_locator": reference_audit["n_with_explicit_locator"],
            "share_with_explicit_locator": reference_audit["share_with_explicit_locator"],
        },
        "systems": {
            "bm25": {"k1": BM25_K1, "b": BM25_B, "tokenizer": TOKEN_RE.pattern},
            "dense": {"model": DENSE_MODEL, "revision": DENSE_REVISION},
            "rrf_k": RRF_K,
            "cross_encoder": {
                "model": CROSS_ENCODER_MODEL,
                "revision": CROSS_ENCODER_REVISION,
            },
            "retrieval_depth": args.retrieval_depth,
            "rerank_depth": args.rerank_depth,
        },
        "runtime": {
            "threads": args.threads,
            "dense_batch_size": args.dense_batch_size,
            "rerank_batch_size": args.rerank_batch_size,
            "embedding_chunk_size": args.embedding_chunk_size,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                name: package_version(name)
                for name in ("numpy", "torch", "transformers", "sentence-transformers")
            },
            "command": sys.argv,
        },
    }
    atomic_write_json(Path(args.output_dir) / "manifest.json", manifest)


def run_self_tests() -> None:
    queries = [
        QueryRecord("q1", "Which rule applies?", {"a": 3, "b": 2, "c": 1}),
        QueryRecord("q2", "What does section 4 provide?", {"d": 3}),
    ]
    perfect = {"q1": ["a", "b", "c", "x"], "q2": ["d", "x"]}
    _, metrics = per_query_metrics(queries, perfect)
    assert metrics["ndcg@10"] == [1.0, 1.0]
    assert metrics["mrr@10"] == [1.0, 1.0]
    test_dir = Path(os.environ.get("TEMP", ".")) / "uk_eval_test"
    report = audit_references(queries, test_dir)
    assert report["n_with_explicit_locator"] == 1
    index = BM25Index(["alpha beta", "gamma delta", "alpha gamma"], 1.5, 0.75)
    indices, _ = index.rank("beta", 2)
    assert indices[0] == 0
    systems = {
        name: perfect
        for name in (
            "bm25",
            "e5_small_v2",
            "rrf_bm25_e5",
            "bm25_ce",
            "rrf_bm25_e5_ce",
        )
    }
    summary = evaluate_all(
        queries,
        systems,
        test_dir,
        bootstrap_resamples=100,
        debug=True,
    )
    assert summary["systems"]["bm25"]["ndcg@10"]["mean"] == 1.0
    print("Self-tests passed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--extract-dir", default=str(DEFAULT_EXTRACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--retrieval-depth", type=int, default=DEFAULT_RETRIEVAL_DEPTH)
    parser.add_argument("--rerank-depth", type=int, default=DEFAULT_RERANK_DEPTH)
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--embedding-chunk-size", type=int, default=512)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--limit", type=int, default=None, help="Debug query limit")
    parser.add_argument("--corpus-limit", type=int, default=None, help="Debug corpus limit; qrel passages are retained")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.retrieval_depth < max(METRIC_KS):
        parser.error(f"--retrieval-depth must be at least {max(METRIC_KS)}")
    if not 1 <= args.rerank_depth <= args.retrieval_depth:
        parser.error("--rerank-depth must be between 1 and --retrieval-depth")
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")
    return args


def main() -> int:
    args = parse_args()
    configure_threads(args.threads)
    if args.self_test:
        run_self_tests()
        return 0

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_file, query_file, source_manifest = prepare_dataset(
        Path(args.archive), Path(args.extract_dir)
    )
    print(json.dumps(source_manifest, indent=2), flush=True)
    if args.prepare_only:
        print("Dataset preparation complete", flush=True)
        return 0

    queries = load_queries(query_file, limit=args.limit)
    required_ids = {doc_id for query in queries for doc_id in query.qrels}
    corpus = load_corpus(corpus_file, required_ids, corpus_limit=args.corpus_limit)
    reference_audit = audit_references(queries, output_dir)
    write_run_manifest(args, source_manifest, corpus, queries, reference_audit)
    print(
        f"Reference audit: {reference_audit['n_with_explicit_locator']}/"
        f"{reference_audit['n_queries']} queries contain an explicit locator",
        flush=True,
    )

    bm25 = run_bm25(
        corpus, queries, args.retrieval_depth, cache_dir, force=args.force
    )
    dense = run_dense(
        corpus,
        queries,
        source_manifest["corpus_sha256"],
        args.retrieval_depth,
        cache_dir,
        batch_size=args.dense_batch_size,
        embedding_chunk_size=args.embedding_chunk_size,
        force=args.force,
    )
    hybrid = rrf_fuse(
        queries, bm25, dense, args.retrieval_depth, cache_dir, force=args.force
    )
    bm25_ce, cross_encoder = run_reranker(
        "bm25",
        bm25,
        corpus,
        queries,
        args.rerank_depth,
        cache_dir,
        batch_size=args.rerank_batch_size,
        force=args.force,
    )
    hybrid_ce, _ = run_reranker(
        "rrf_bm25_e5",
        hybrid,
        corpus,
        queries,
        args.rerank_depth,
        cache_dir,
        batch_size=args.rerank_batch_size,
        force=args.force,
        model=cross_encoder,
    )
    systems = {
        "bm25": bm25,
        "e5_small_v2": dense,
        "rrf_bm25_e5": hybrid,
        "bm25_ce": bm25_ce,
        "rrf_bm25_e5_ce": hybrid_ce,
    }
    debug = args.limit is not None or args.corpus_limit is not None
    summary = evaluate_all(
        queries,
        systems,
        output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        debug=debug,
    )
    atomic_write_json(
        output_dir / "RUN_COMPLETE.json",
        {
            "completed_at": now_iso(),
            "canonical": not debug,
            "summary": "summary_debug.json" if debug else "summary.json",
        },
    )
    print(json.dumps(summary["systems"], indent=2), flush=True)
    print(f"Complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
