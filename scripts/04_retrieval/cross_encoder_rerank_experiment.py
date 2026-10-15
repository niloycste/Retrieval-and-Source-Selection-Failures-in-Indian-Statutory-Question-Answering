"""Cross-encoder re-ranking over the same corpus, queries and BM25 used in
dense_retrieval_experiment.py.

First stage retrieves the top-N candidates (BM25, and the BM25 + e5-small
reciprocal-rank fusion hybrid). A cross-encoder then scores every
(question, passage) pair jointly and re-orders the candidate list. Because
re-ranking can only reorder what the first stage returned, recall@N of the
first stage is the ceiling; it is reported alongside so the headroom is visible.

Run:  python cross_encoder_rerank_experiment.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dense_retrieval_experiment as base

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "results" / "experiments" / "retrieval" / "runs" / "cross_encoder_rerank_v1"
CANONICAL_REPORT = BASE / "results" / "experiments" / "retrieval" / "cross_encoder_rerank_report.json"
CANDIDATE_N = 50
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BEST_DENSE = ("intfloat/e5-small-v2", "query: ", "passage: ")


def recall_at(rankings, gold, k):
    return round(100.0 * sum(1 for r, g in zip(rankings, gold) if g in r[:k]) / len(gold), 2)


def evaluate(rankings, gold):
    """Local copy of base.evaluate with MRR correctly truncated at rank 10.

    base.evaluate names its metric mrr@10 but iterates the full ranking, so on a
    depth-50 candidate pool it silently reports MRR@50. Table 15 of the paper was
    produced with depth-10 lists, where the two coincide; this experiment uses
    depth 50, so the truncation has to be explicit for the numbers to be
    comparable.
    """
    res = {}
    for k in base.K_VALUES:
        res[f"recall@{k}"] = recall_at(rankings, gold, k)
    mrr = 0.0
    for r, g in zip(rankings, gold):
        top10 = r[:10]
        if g in top10:
            mrr += 1.0 / (top10.index(g) + 1)
    res["mrr@10"] = round(100.0 * mrr / len(gold), 2)
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    passages, acts, queries, gold, q_acts = base.load_data()
    print(f"corpus {len(passages)} passages | queries {len(queries)}")

    # ---- stage 1: BM25 ----------------------------------------------------
    t0 = time.time()
    bm25 = base.BM25([base.tokenize(p) for p in passages])
    bm25_rank = [base.ranking_from_scores(bm25.scores(base.tokenize(q)), CANDIDATE_N)
                 for q in queries]
    print(f"[1] BM25 top-{CANDIDATE_N} in {time.time()-t0:.1f}s  "
          f"recall@5 {recall_at(bm25_rank, gold, 5)}  "
          f"recall@{CANDIDATE_N} {recall_at(bm25_rank, gold, CANDIDATE_N)}")

    # ---- stage 1b: hybrid RRF --------------------------------------------
    mid, qp, pp = BEST_DENSE
    t0 = time.time()
    dense_rank, _enc_s = base.dense_rankings(mid, qp, pp, passages, queries, CANDIDATE_N)
    hybrid_rank = base.rrf_fuse([bm25_rank, dense_rank], CANDIDATE_N)
    print(f"[2] hybrid RRF top-{CANDIDATE_N} in {time.time()-t0:.1f}s  "
          f"recall@5 {recall_at(hybrid_rank, gold, 5)}  "
          f"recall@{CANDIDATE_N} {recall_at(hybrid_rank, gold, CANDIDATE_N)}")

    # ---- stage 2: cross-encoder re-ranking -------------------------------
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CROSS_ENCODER, max_length=512)

    results, rankings_out, cache = {}, {}, {}
    for name, cand in (("BM25", bm25_rank), ("Hybrid RRF", hybrid_rank)):
        pairs, spans = [], []
        for q, cs in zip(queries, cand):
            spans.append((len(pairs), len(pairs) + len(cs)))
            pairs.extend((q, passages[c]) for c in cs)
        print(f"[3] re-ranking {len(pairs)} pairs for {name} ...", flush=True)
        t0 = time.time()
        scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
        dt = time.time() - t0
        print(f"    {dt:.1f}s  ({len(pairs)/dt:.0f} pairs/s)")

        reranked = []
        for (lo, hi), cs in zip(spans, cand):
            order = sorted(range(len(cs)), key=lambda i: -scores[lo + i])
            reranked.append([cs[i] for i in order])

        base_res = evaluate(cand, gold)
        re_res = evaluate(reranked, gold)
        base_res[f"recall@{CANDIDATE_N}"] = recall_at(cand, gold, CANDIDATE_N)
        re_res[f"recall@{CANDIDATE_N}"] = recall_at(reranked, gold, CANDIDATE_N)
        results[name] = base_res
        results[f"{name} + cross-encoder"] = re_res
        rankings_out[f"{name} + cross-encoder"] = reranked
        rankings_out[name] = cand
        cache[name] = {"candidates": cand, "scores": [float(x) for x in scores]}

    # ---- report -----------------------------------------------------------
    cols = ["recall@1", "recall@3", "recall@5", "recall@10",
            f"recall@{CANDIDATE_N}", "mrr@10"]
    print("\n" + "=" * 86)
    print(f"{'system':34s}" + "".join(f"{c:>10s}" for c in cols))
    print("-" * 86)
    for k, v in results.items():
        print(f"{k:34s}" + "".join(f"{v.get(c, float('nan')):10.2f}" for c in cols))
    print("=" * 86)

    best = max(rankings_out, key=lambda k: results[k]["recall@5"])
    pa = base.per_act_recall(rankings_out[best], gold, q_acts)
    print(f"\nper-Act recall@5 for {best}:")
    for a, v in sorted(pa.items(), key=lambda t: -t[1]):
        print(f"  {a:34s} {v:6.2f}")

    payload = {
        "candidate_n": CANDIDATE_N,
        "hybrid_specification": {
            "components": ["BM25", BEST_DENSE[0].split("/")[-1]],
            "input_depth_per_component": CANDIDATE_N,
            "output_depth": CANDIDATE_N,
            "rrf_k": base.RRF_K,
            "candidate_text": "paragraph only",
        },
        "rerank_specification": {
            "input_candidates": CANDIDATE_N,
            "output_evaluated_at": list(base.K_VALUES),
        },
        "cross_encoder": CROSS_ENCODER,
        "n_queries": len(queries),
        "n_passages": len(passages),
        "results": results,
        "per_act_recall5_best": {"system": best, "values": pa},
        "note": "mrr@10 truncated at rank 10; candidate depth 50",
    }
    (OUT_DIR / "cross_encoder_rerank_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    CANONICAL_REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "rankings_cache.json").write_text(
        json.dumps({"gold": gold, "q_acts": q_acts, "cache": cache}), encoding="utf-8")
    print(f"\nwritten to {OUT_DIR}")
    print(f"canonical report: {CANONICAL_REPORT}")


if __name__ == "__main__":
    main()
