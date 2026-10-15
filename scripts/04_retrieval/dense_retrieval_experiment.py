"""
Dense vs. sparse retrieval on IndicIPR-QA.

Motivation
----------
The journal paper shows BM25 retrieval -- not span extraction -- is the dominant
end-to-end bottleneck (top-5 recall 22.97%). A reviewer will reasonably ask
whether that is a property of *statutory retrieval* or merely a weakness of
BM25. This script answers that by evaluating off-the-shelf dense retrievers and
a sparse+dense hybrid on the identical corpus and queries.

Design notes
------------
* BM25 is reimplemented with the SAME tokenizer (``[A-Za-z0-9]+``, lowercased)
  and the SAME parameters (k1=1.5, b=0.75) as
  the paper's retrieval-only experiment. Reproducing recall
  13.08 / 20.06 / 22.67 at k=1/3/5 is used as a correctness check on the
  whole pipeline before any dense number is trusted. The separate
  retrieval-plus-reader path deduplicates the corpus differently and reports
  13.66 / 20.06 / 22.97.
* Retrieval-only. No reader passes, so this is cheap: 344 queries against
  1,116 passages.
* Gold matching is by exact normalised context text; all 344 test gold
  contexts are present in the 1,116-passage corpus.

Run:  python dense_retrieval_experiment.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE = Path(__file__).resolve().parents[2]
CORPUS_FILE = BASE / "data" / "processed" / "indic_ipr_qa" / "unique_contexts.jsonl"
TEST_FILE = BASE / "data" / "splits" / "test.jsonl"
OUT_DIR = BASE / "results" / "experiments" / "retrieval" / "runs" / "dense_retrieval_v1"
CANONICAL_REPORT = BASE / "results" / "experiments" / "retrieval" / "dense_retrieval_report.json"

K_VALUES = (1, 3, 5, 10)
RRF_K = 60  # standard reciprocal-rank-fusion constant

# Canonical retrieval-only BM25 recall, used as a pipeline correctness check.
EXPECTED_BM25_RECALL = {1: 13.08, 3: 20.06, 5: 22.67}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def norm(text: str) -> str:
    return " ".join(str(text).split())


# ---------------------------------------------------------------- data loading
def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_data():
    corpus_rows = load_jsonl(CORPUS_FILE)
    passages = [norm(r["context"]) for r in corpus_rows]
    acts = [r.get("act", "") for r in corpus_rows]

    index_of = {}
    for i, p in enumerate(passages):
        index_of.setdefault(p, i)

    test_rows = load_jsonl(TEST_FILE)
    queries, gold_idx, q_acts = [], [], []
    missing = 0
    for r in test_rows:
        ctx = norm(r["context"])
        if ctx not in index_of:
            missing += 1
            continue
        queries.append(norm(r["question"]))
        gold_idx.append(index_of[ctx])
        q_acts.append(r.get("act", ""))

    if missing:
        print(f"  [warn] {missing} test rows had no matching corpus passage")
    return passages, acts, queries, gold_idx, q_acts


# ------------------------------------------------------------------------ BM25
class BM25:
    """Okapi BM25, matched to journal_retrieval_transfer_experiments.py."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self.tf = [Counter(d) for d in docs]

        df = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(docs):
            for term in set(d):
                self.postings.setdefault(term, []).append(i)

    def scores(self, query: list[str]):
        out = [0.0] * self.N
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                freq = self.tf[i][term]
                dl = self.doc_len[i]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                out[i] += idf * freq * (self.k1 + 1) / denom
        return out


# --------------------------------------------------------------------- metrics
def ranking_from_scores(scores, top: int):
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])
    return idx[:top]


def evaluate(rankings: list[list[int]], gold: list[int]) -> dict:
    res = {}
    for k in K_VALUES:
        hits = sum(1 for r, g in zip(rankings, gold) if g in r[:k])
        res[f"recall@{k}"] = round(100.0 * hits / len(gold), 2)
    mrr = 0.0
    for r, g in zip(rankings, gold):
        if g in r:
            mrr += 1.0 / (r.index(g) + 1)
    res["mrr@10"] = round(100.0 * mrr / len(gold), 2)
    return res


def per_act_recall(rankings, gold, q_acts, k=5) -> dict:
    tot, hit = Counter(), Counter()
    for r, g, a in zip(rankings, gold, q_acts):
        tot[a] += 1
        if g in r[:k]:
            hit[a] += 1
    return {a: round(100.0 * hit[a] / tot[a], 2) for a in sorted(tot)}


# ------------------------------------------------------------- dense retrieval
DENSE_MODELS = [
    # (hf id, query prefix, passage prefix)
    ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "", ""),
    ("BAAI/bge-small-en-v1.5",
     "Represent this sentence for searching relevant passages: ", ""),
    ("intfloat/e5-small-v2", "query: ", "passage: "),
]


def dense_rankings(model_id, q_prefix, p_prefix, passages, queries, top):
    from sentence_transformers import SentenceTransformer
    import numpy as np

    t0 = time.time()
    model = SentenceTransformer(model_id, device="cpu")
    p_emb = model.encode(
        [p_prefix + p for p in passages],
        batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    q_emb = model.encode(
        [q_prefix + q for q in queries],
        batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    sim = q_emb @ p_emb.T                      # cosine (embeddings normalised)
    order = np.argsort(-sim, axis=1)[:, :top]
    return [list(map(int, row)) for row in order], round(time.time() - t0, 1)


def rrf_fuse(rank_lists: list[list[list[int]]], top: int) -> list[list[int]]:
    """Reciprocal rank fusion; robust to score-scale differences."""
    fused = []
    for q in range(len(rank_lists[0])):
        acc: dict[int, float] = {}
        for rl in rank_lists:
            for rank, doc in enumerate(rl[q]):
                acc[doc] = acc.get(doc, 0.0) + 1.0 / (RRF_K + rank + 1)
        fused.append([d for d, _ in sorted(acc.items(), key=lambda kv: -kv[1])[:top]])
    return fused


# ------------------------------------------------------------------------ main
def main():
    print("=" * 74)
    print("Dense vs. sparse retrieval on IndicIPR-QA")
    print("=" * 74)

    passages, acts, queries, gold, q_acts = load_data()
    print(f"corpus passages : {len(passages)}")
    print(f"test queries    : {len(queries)}")
    TOP = max(max(K_VALUES), 10)

    results, timings, all_rankings = {}, {}, {}

    # ---- BM25 (correctness check against published numbers) ----
    print("\n[1] BM25 (k1=1.5, b=0.75) ...")
    t0 = time.time()
    bm25 = BM25([tokenize(p) for p in passages])
    bm25_rank = [ranking_from_scores(bm25.scores(tokenize(q)), TOP) for q in queries]
    timings["BM25"] = round(time.time() - t0, 1)
    results["BM25"] = evaluate(bm25_rank, gold)
    all_rankings["BM25"] = bm25_rank

    print("    reproduction check vs. canonical retrieval-only values:")
    ok = True
    for k, expected in EXPECTED_BM25_RECALL.items():
        got = results["BM25"][f"recall@{k}"]
        match = abs(got - expected) < 0.01
        ok &= match
        print(f"      recall@{k}: got {got:6.2f}  expected {expected:6.2f}"
              f"   {'MATCH' if match else 'MISMATCH'}")
    print(f"    -> pipeline {'VALIDATED' if ok else 'DOES NOT REPRODUCE - investigate'}")

    # ---- dense ----
    for i, (mid, qp, pp) in enumerate(DENSE_MODELS, start=2):
        short = mid.split("/")[-1]
        print(f"\n[{i}] dense: {short} ...")
        try:
            rank, secs = dense_rankings(mid, qp, pp, passages, queries, TOP)
        except Exception as exc:
            print(f"    SKIPPED ({type(exc).__name__}: {str(exc)[:90]})")
            continue
        results[short] = evaluate(rank, gold)
        timings[short] = secs
        all_rankings[short] = rank
        print(f"    recall@5 = {results[short]['recall@5']:.2f}   ({secs}s)")

    # ---- hybrid: BM25 + best dense ----
    dense_only = {k: v for k, v in results.items() if k != "BM25"}
    if dense_only:
        best_dense = max(dense_only, key=lambda k: dense_only[k]["recall@5"])
        print(f"\n[{len(DENSE_MODELS)+2}] hybrid RRF: BM25 + {best_dense} ...")
        hyb = rrf_fuse([all_rankings["BM25"], all_rankings[best_dense]], TOP)
        name = f"Hybrid RRF (BM25 + {best_dense})"
        results[name] = evaluate(hyb, gold)
        all_rankings[name] = hyb
    else:
        best_dense = None

    # ---- report ----
    print("\n" + "=" * 74)
    print(f"{'System':<44}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'R@10':>7}{'MRR':>7}")
    print("-" * 74)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["recall@5"]):
        print(f"{name:<44}{r['recall@1']:>7.2f}{r['recall@3']:>7.2f}"
              f"{r['recall@5']:>7.2f}{r['recall@10']:>7.2f}{r['mrr@10']:>7.2f}")
    print("=" * 74)

    # ---- per-Act for BM25 vs best system ----
    best_overall = max(results, key=lambda k: results[k]["recall@5"])
    print(f"\nper-Act recall@5:  BM25  vs  {best_overall}")
    pa_bm25 = per_act_recall(all_rankings["BM25"], gold, q_acts)
    pa_best = per_act_recall(all_rankings[best_overall], gold, q_acts)
    for a in pa_bm25:
        print(f"  {a:<40}{pa_bm25[a]:>7.2f}{pa_best[a]:>9.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_passages": len(passages),
        "test_queries": len(queries),
        "bm25_params": {"k1": 1.5, "b": 0.75, "tokenizer": "[A-Za-z0-9]+ lowercased"},
        "bm25_reproduction_check": {
            "expected": EXPECTED_BM25_RECALL,
            "reproduced": {k: results["BM25"][f"recall@{k}"] for k in EXPECTED_BM25_RECALL},
            "validated": ok,
        },
        "rrf_k": RRF_K,
        "ranking_depth": TOP,
        "hybrid_specification": {
            "components": ["BM25", best_dense],
            "input_depth_per_component": TOP,
            "output_depth": TOP,
            "candidate_text": "paragraph only",
        },
        "results": results,
        "timings_sec": timings,
        "best_overall": best_overall,
        "per_act_recall_at_5": {"BM25": pa_bm25, best_overall: pa_best},
    }
    (OUT_DIR / "dense_retrieval_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    canonical_payload = {k: v for k, v in payload.items() if k != "timings_sec"}
    CANONICAL_REPORT.write_text(
        json.dumps(canonical_payload, indent=2), encoding="utf-8")

    with open(OUT_DIR / "dense_retrieval_summary.csv", "w", encoding="utf-8") as fh:
        fh.write("system,recall@1,recall@3,recall@5,recall@10,mrr@10\n")
        for name, r in results.items():
            fh.write(f"\"{name}\",{r['recall@1']},{r['recall@3']},"
                     f"{r['recall@5']},{r['recall@10']},{r['mrr@10']}\n")

    print(f"\nwrote {OUT_DIR / 'dense_retrieval_summary.json'}")
    print(f"canonical report: {CANONICAL_REPORT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
