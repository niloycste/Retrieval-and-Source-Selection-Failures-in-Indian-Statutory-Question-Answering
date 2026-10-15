"""Validate a completed modern BGE retrieval-extension result directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SYSTEMS = (
    "bm25",
    "bge_m3",
    "rrf_bm25_bge_m3",
    "bm25_bge_reranker_v2_m3",
    "rrf_bm25_bge_m3_bge_reranker_v2_m3",
)
INDIC_METRICS = (
    "recall@1", "recall@3", "recall@5", "recall@10", "recall@50",
    "mrr@10", "ndcg@10",
)
UK_METRICS = INDIC_METRICS + ("grade3_recall@5", "grade3_recall@10")
DENSE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def close(observed: float, expected: float, tolerance: float, label: str) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"{label}: observed={observed}, expected={expected}")


def check_rankings(run_dir: Path, dataset: str, expected_queries: int, depth: int) -> None:
    directory = run_dir / dataset
    files = {
        "bm25": "rankings_bm25.jsonl",
        "bge_m3": "rankings_bge_m3.jsonl",
        "rrf_bm25_bge_m3": "rankings_rrf_bm25_bge_m3.jsonl",
        "bm25_bge_reranker_v2_m3": "rankings_bm25_bge_reranker_v2_m3.jsonl",
        "rrf_bm25_bge_m3_bge_reranker_v2_m3": (
            "rankings_rrf_bm25_bge_m3_bge_reranker_v2_m3.jsonl"
        ),
    }
    for system, filename in files.items():
        rows = read_jsonl(directory / filename)
        if len(rows) != expected_queries:
            raise AssertionError(f"{dataset}/{system}: {len(rows)} ranking rows")
        qids = [str(row["qid"]) for row in rows]
        if len(set(qids)) != expected_queries:
            raise AssertionError(f"{dataset}/{system}: duplicate qids")
        if any(len(row["ranking"]) != depth for row in rows):
            raise AssertionError(f"{dataset}/{system}: unexpected ranking depth")
        if any(len(set(row["ranking"])) != depth for row in rows):
            raise AssertionError(f"{dataset}/{system}: duplicate document in ranking")


def check_per_query(
    csv_path: Path,
    summary: dict,
    metrics: tuple[str, ...],
    expected_queries: int,
    percentage_summary: bool,
) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_queries * len(SYSTEMS):
        raise AssertionError(f"{csv_path}: unexpected row count {len(rows)}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["system"]].append(row)
    if set(grouped) != set(SYSTEMS):
        raise AssertionError(f"{csv_path}: systems mismatch {sorted(grouped)}")
    for system in SYSTEMS:
        current = grouped[system]
        if len({row["qid"] for row in current}) != expected_queries:
            raise AssertionError(f"{csv_path}/{system}: qid count mismatch")
        for metric in metrics:
            mean = sum(float(row[metric]) for row in current) / expected_queries
            if percentage_summary:
                expected = float(summary[system][metric])
                close(round(mean * 100.0, 2), expected, 0.005, f"{system}/{metric}")
            else:
                expected = float(summary["systems"][system][metric]["mean"])
                close(mean, expected, 1e-12, f"{system}/{metric}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    complete = read_json(run_dir / "RUN_COMPLETE.json")
    if complete.get("datasets") != ["indic", "uk"]:
        raise AssertionError(f"completion marker is incomplete: {complete}")
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("completed_datasets") != ["indic", "uk"]:
        raise AssertionError("manifest dataset list is incomplete")
    if manifest["protocol"] != {
        "dense": "BAAI/bge-m3",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "candidate_text": "paragraph only",
        "rerank_depth": 50,
        "rrf_k": 60,
        "max_seq_length": 512,
        "metrics": list(INDIC_METRICS),
    }:
        raise AssertionError("protocol manifest differs from the frozen design")
    if manifest["models"]["BAAI/bge-m3"]["revision"] != DENSE_REVISION:
        raise AssertionError("BGE-M3 revision mismatch")
    if manifest["models"]["BAAI/bge-reranker-v2-m3"]["revision"] != RERANKER_REVISION:
        raise AssertionError("BGE reranker revision mismatch")
    smoke = read_json(run_dir / "MODEL_SMOKE_TEST.json")
    if smoke.get("status") != "PASS" or smoke.get("dense_shape") != [2, 1024]:
        raise AssertionError(f"model smoke test failed: {smoke}")

    indic = read_json(run_dir / "indic" / "summary.json")
    uk = read_json(run_dir / "uk" / "summary.json")
    if indic.get("n_queries") != 344 or indic.get("n_passages") != 1116:
        raise AssertionError("IndicIPR-QA dimensions mismatch")
    if uk.get("n_queries") != 100 or not uk.get("canonical"):
        raise AssertionError("UK Eval100 summary is not canonical")
    if set(indic["results"]) != set(SYSTEMS) or set(uk["systems"]) != set(SYSTEMS):
        raise AssertionError("summary system set mismatch")

    check_per_query(
        run_dir / "indic" / "per_query_metrics.csv",
        indic["results"], INDIC_METRICS, 344, percentage_summary=True,
    )
    check_per_query(
        run_dir / "uk" / "per_query.csv",
        uk, UK_METRICS, 100, percentage_summary=False,
    )
    check_rankings(run_dir, "indic", 344, 50)
    check_rankings(run_dir, "uk", 100, 100)

    close(
        indic["results"]["bm25"]["recall@50"],
        indic["results"]["bm25_bge_reranker_v2_m3"]["recall@50"],
        0.0, "Indic BM25 reranker ceiling",
    )
    close(
        indic["results"]["rrf_bm25_bge_m3"]["recall@50"],
        indic["results"]["rrf_bm25_bge_m3_bge_reranker_v2_m3"]["recall@50"],
        0.0, "Indic RRF reranker ceiling",
    )
    close(
        uk["systems"]["bm25"]["recall@50"]["mean"],
        uk["systems"]["bm25_bge_reranker_v2_m3"]["recall@50"]["mean"],
        0.0, "UK BM25 reranker ceiling",
    )
    close(
        uk["systems"]["rrf_bm25_bge_m3"]["recall@50"]["mean"],
        uk["systems"]["rrf_bm25_bge_m3_bge_reranker_v2_m3"]["recall@50"]["mean"],
        0.0, "UK RRF reranker ceiling",
    )
    if uk["candidate_ceilings"] != {
        "bm25_recall@50": uk["systems"]["bm25"]["recall@50"]["mean"],
        "rrf_bm25_bge_m3_recall@50": uk["systems"]["rrf_bm25_bge_m3"]["recall@50"]["mean"],
    }:
        raise AssertionError("UK candidate-ceiling report mismatch")

    indic_embeddings = np.load(run_dir / "indic" / "bge_m3_corpus_embeddings.npy", mmap_mode="r")
    uk_embeddings = np.load(run_dir / "uk" / "bge_m3_corpus_embeddings.npy", mmap_mode="r")
    if indic_embeddings.shape != (1116, 1024) or uk_embeddings.shape != (124796, 1024):
        raise AssertionError(
            f"embedding shapes mismatch: {indic_embeddings.shape}, {uk_embeddings.shape}"
        )
    for name, vectors in (("indic", indic_embeddings), ("uk", uk_embeddings)):
        sample = np.asarray(vectors[[0, len(vectors) // 2, len(vectors) - 1]])
        if not np.isfinite(sample).all():
            raise AssertionError(f"{name} embeddings contain non-finite sampled values")
        norms = np.linalg.norm(sample, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise AssertionError(f"{name} sampled embeddings are not L2 normalised: {norms}")

    report = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "datasets": ["indic", "uk"],
        "systems": list(SYSTEMS),
        "per_query_rows": {"indic": 344 * 5, "uk": 100 * 5},
        "ranking_rows": {"indic": 344 * 5, "uk": 100 * 5},
        "embedding_shapes": {"indic": [1116, 1024], "uk": [124796, 1024]},
    }
    (run_dir / "VALIDATION_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
