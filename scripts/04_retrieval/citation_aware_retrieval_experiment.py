"""Citation-aware retrieval: does indexing statutory structure fix the bottleneck?

Section 5.7 argues that the residual retrieval failures are structural rather
than semantic, and that metadata- and citation-aware retrieval is the promising
direction. This script tests that claim directly instead of asserting it.

The section number is parsed OUT OF THE QUESTION TEXT with a regex, never taken
from gold metadata. Questions cite a section but never name the Act, so a
matching section number still occurs in several Acts and the retriever must
still disambiguate. Passage section numbers come from the corpus `title` field.

Systems compared, all on the same 1,116 passages and 344 test questions as
Table 15:
  BM25                 lexical baseline
  BM25 + section boost adds a fixed bonus to passages whose section matches
  BM25 + section filter restricts candidates to matching passages, falling back
                       to unrestricted BM25 when the question has no parsable
                       citation or nothing matches

Retrieval-only run:
  python citation_aware_retrieval_experiment.py

Canonical retrieval-to-reader run with E2':
  python citation_aware_retrieval_experiment.py --with-reader

The reader mode applies the shared sliding-window constrained span decoder to
gold, BM25, section-boosted, and section-filtered top-k candidates. It writes a
separate canonical end-to-end report and per-question predictions.
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dense_retrieval_experiment as base

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_FILE = ROOT / "data" / "processed" / "indic_ipr_qa" / "unique_contexts.jsonl"
TEST_FILE = ROOT / "data" / "splits" / "test.jsonl"
OUT_DIR = ROOT / "results" / "experiments" / "retrieval" / "runs" / "citation_aware_v1"
CANONICAL_REPORT = ROOT / "results" / "experiments" / "retrieval" / "citation_aware_retrieval_report.json"
CANONICAL_QA_REPORT = (
    ROOT / "results" / "experiments" / "retrieval" / "citation_aware_end_to_end_report.json"
)
DEFAULT_QA_MODEL = (
    ROOT / "artifacts" / "models" / "squad_v1" / "baseline_e2_squadv1" / "final"
)
K_VALUES = (1, 3, 5, 10)
BOOST = 1000.0

SEC_RE = re.compile(r"\bsections?\s*[.\-]?\s*(\d+[A-Za-z]{0,2})\b", re.I)


def parse_section(text):
    m = SEC_RE.search(text or "")
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(1)).upper()


def jl(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def recall_at(rankings, gold, k):
    return round(100.0 * sum(1 for r, g in zip(rankings, gold) if g in r[:k]) / len(gold), 2)


def mrr10(rankings, gold):
    tot = 0.0
    for r, g in zip(rankings, gold):
        t = r[:10]
        if g in t:
            tot += 1.0 / (t.index(g) + 1)
    return round(100.0 * tot / len(gold), 2)


def report(name, rankings, gold):
    row = {f"recall@{k}": recall_at(rankings, gold, k) for k in K_VALUES}
    row["mrr@10"] = mrr10(rankings, gold)
    print(f"  {name:28s}" + "".join(f"{row[f'recall@{k}']:9.2f}" for k in K_VALUES)
          + f"{row['mrr@10']:9.2f}")
    return row


def metric_summary(rows):
    if not rows:
        return {"n": 0, "em_pct": 0.0, "f1_pct": 0.0, "char_f1_pct": 0.0}
    return {
        "n": len(rows),
        "em_pct": round(100.0 * sum(row["em"] for row in rows) / len(rows), 4),
        "f1_pct": round(100.0 * sum(row["f1"] for row in rows) / len(rows), 4),
        "char_f1_pct": round(
            100.0 * sum(row["char_f1"] for row in rows) / len(rows), 4
        ),
    }


def paired_bootstrap_comparisons(predictions_by_setting, iterations=10_000, seed=42):
    """Compare section-aware F1 with BM25 using paired test-item resampling."""

    import numpy as np

    baseline = predictions_by_setting["BM25"]
    baseline_ids = [row["id"] for row in baseline]
    baseline_f1 = np.asarray([row["f1"] for row in baseline], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(baseline), size=(iterations, len(baseline)))
    comparisons = []
    for setting in ("BM25 + section boost", "BM25 + section filter"):
        rows = predictions_by_setting[setting]
        if [row["id"] for row in rows] != baseline_ids:
            raise ValueError(f"prediction IDs are not aligned for {setting}")
        difference = np.asarray([row["f1"] for row in rows], dtype=float) - baseline_f1
        bootstrap_means = difference[samples].mean(axis=1) * 100.0
        lower, upper = np.percentile(bootstrap_means, (2.5, 97.5))
        comparisons.append(
            {
                "comparison": f"{setting} top-5 + E2' minus BM25 top-5 + E2'",
                "f1_difference_points": round(float(difference.mean() * 100.0), 4),
                "pointwise_unadjusted_95_ci": [
                    round(float(lower), 4),
                    round(float(upper), 4),
                ],
            }
        )
    return {
        "method": "paired nonparametric bootstrap over test questions",
        "iterations": iterations,
        "seed": seed,
        "multiplicity_adjustment": "none",
        "comparisons": comparisons,
    }


def refresh_reader_report_comparisons():
    """Add bootstrap comparisons to an existing completed reader report."""

    if not CANONICAL_QA_REPORT.exists():
        raise FileNotFoundError(f"reader report not found: {CANONICAL_QA_REPORT}")
    payload = json.loads(CANONICAL_QA_REPORT.read_text(encoding="utf-8"))
    predictions = {
        setting: jl(ROOT / relative_path)
        for setting, relative_path in payload["prediction_files"].items()
    }
    payload["paired_comparisons"] = paired_bootstrap_comparisons(predictions)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    CANONICAL_QA_REPORT.write_text(rendered, encoding="utf-8")
    (OUT_DIR / "citation_aware_end_to_end_report.json").write_text(
        rendered, encoding="utf-8"
    )
    print(f"refreshed paired comparisons in: {CANONICAL_QA_REPORT}")


def evaluate_reader(
    *,
    tests,
    queries,
    gold,
    passages,
    corpus,
    rankings_store,
    question_type_mask,
    args,
):
    """Run the canonical E2' reader over citation-aware top-k candidates."""

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

    top_k = args.reader_top_k
    systems = {
        "BM25": rankings_store["BM25"],
        "BM25 + section boost": rankings_store["BM25 + section boost"],
        "BM25 + section filter": rankings_store["BM25 + section filter"],
    }

    # Decode each unique question/context pair once. Gold passages are included
    # so this report independently verifies the reader ceiling with the same
    # checkpoint and decoder used for the retrieval settings.
    pair_keys = []
    seen_pairs = set()

    def add_pair(query_index, context_index):
        key = (query_index, context_index)
        if key not in seen_pairs:
            seen_pairs.add(key)
            pair_keys.append(key)

    for query_index, context_index in enumerate(gold):
        add_pair(query_index, context_index)
    for rankings in systems.values():
        for query_index, ranking in enumerate(rankings):
            for context_index in ranking[:top_k]:
                add_pair(query_index, context_index)

    device = resolve_device(args.device)
    config = QADecodingConfig(
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
        max_answer_tokens=None,
    )
    print(
        f"\nreader evaluation: {len(pair_keys):,} unique question/context pairs | "
        f"device={device} | top-k={top_k}"
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
        [queries[query_index] for query_index, _ in pair_keys],
        [passages[context_index] for _, context_index in pair_keys],
        config,
    )
    decoded_by_pair = dict(zip(pair_keys, decoded))

    setting_rankings = {"Gold paragraph": [[context_index] for context_index in gold]}
    setting_rankings.update(systems)
    predictions_by_setting = {}
    summary_rows = []

    for setting, rankings in setting_rankings.items():
        rows = []
        for query_index, ranking in enumerate(rankings):
            candidates = ranking[:1] if setting == "Gold paragraph" else ranking[:top_k]
            scored = []
            for rank, context_index in enumerate(candidates, start=1):
                prediction, span_score = decoded_by_pair[(query_index, context_index)]
                scored.append((span_score, rank, context_index, prediction))
            span_score, selected_rank, selected_index, prediction = max(scored)
            gold_answer = tests[query_index]["answer"]
            em, f1 = compute_em_f1(prediction, gold_answer)
            char_f1 = compute_char_f1(prediction, gold_answer)
            rows.append(
                {
                    "id": tests[query_index].get("id", str(query_index)),
                    "question": queries[query_index],
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

        predictions_by_setting[setting] = rows
        label = (
            "Gold paragraph + E2'"
            if setting == "Gold paragraph"
            else f"{setting} top-{top_k} + E2'"
        )
        summary = metric_summary(rows)
        summary.update(
            {
                "setting": label,
                "retrieval_recall_pct": round(
                    100.0 * sum(row["retrieval_hit_at_k"] for row in rows) / len(rows),
                    4,
                ),
                "reader_selected_gold_context_pct": round(
                    100.0
                    * sum(row["selected_context_is_gold"] for row in rows)
                    / len(rows),
                    4,
                ),
            }
        )
        summary_rows.append(summary)
        safe_name = re.sub(r"[^a-z0-9]+", "_", setting.lower()).strip("_")
        write_jsonl(OUT_DIR / f"{safe_name}_top{top_k}_e2_predictions.jsonl", rows)

    by_question_type = {}
    if question_type_mask:
        for group_name, wanted in (
            ("citation-template-only", True),
            ("lexically-content-bearing", False),
        ):
            group = {}
            indices = [
                index for index, is_template_only in enumerate(question_type_mask)
                if is_template_only == wanted
            ]
            for setting, rows in predictions_by_setting.items():
                group[setting] = metric_summary([rows[index] for index in indices])
            by_question_type[group_name] = {"n": len(indices), "results": group}

    try:
        relative_model_path = model_path.relative_to(ROOT).as_posix()
    except ValueError:
        relative_model_path = str(model_path)
    payload = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "canonical",
        "metric_roles": {
            "primary": ["em_pct", "f1_pct"],
            "diagnostic": ["char_f1_pct"],
            "diagnostic_note": (
                "char_f1_pct is order-insensitive character-frequency overlap retained "
                "for backward-compatible auditing; it is not a manuscript metric"
            ),
        },
        "qa_model": {
            "code": "E2'",
            "label": "LegalBERT(US) -> SQuAD v1 -> IndicIPR-QA",
            "model_path": relative_model_path,
        },
        "evaluation": {
            "test_file": TEST_FILE.relative_to(ROOT).as_posix(),
            "retrieval_corpus": CORPUS_FILE.relative_to(ROOT).as_posix(),
            "test_questions": len(tests),
            "retrieval_contexts": len(passages),
            "reader_top_k": top_k,
            **config.as_dict(),
        },
        "results": summary_rows,
        "question_type_results": by_question_type,
        "paired_comparisons": paired_bootstrap_comparisons(predictions_by_setting),
        "prediction_files": {
            setting: str(
                (
                    OUT_DIR
                    / f"{re.sub(r'[^a-z0-9]+', '_', setting.lower()).strip('_')}_top{top_k}_e2_predictions.jsonl"
                ).relative_to(ROOT).as_posix()
            )
            for setting in setting_rankings
        },
    }
    CANONICAL_QA_REPORT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "citation_aware_end_to_end_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nend-to-end reader results")
    for row in summary_rows:
        print(
            f"  {row['setting']:<45s} R={row['retrieval_recall_pct']:6.2f} "
            f"EM={row['em_pct']:6.2f} F1={row['f1_pct']:6.2f} "
            f"diagnostic Char-F1={row['char_f1_pct']:6.2f}"
        )
    print(f"canonical end-to-end report: {CANONICAL_QA_REPORT}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-reader",
        action="store_true",
        help="also run canonical E2' extraction over section-aware top-k candidates",
    )
    parser.add_argument(
        "--refresh-reader-report",
        action="store_true",
        help="refresh bootstrap comparisons from existing per-question predictions",
    )
    parser.add_argument("--qa-model", default=str(DEFAULT_QA_MODEL))
    parser.add_argument("--reader-top-k", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.reader_top_k < 1 or args.reader_top_k > max(K_VALUES):
        raise ValueError(f"--reader-top-k must be between 1 and {max(K_VALUES)}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    return args


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_reader_report:
        refresh_reader_report_comparisons()
        return

    corpus = jl(CORPUS_FILE)
    passages = [r["context"] for r in corpus]
    p_sec = [parse_section(r.get("title", "")) for r in corpus]
    p_act = [r.get("act", "?") for r in corpus]
    idx_of = {base.norm(p): i for i, p in enumerate(passages)}
    parsed = sum(1 for s in p_sec if s)
    print(f"passages {len(passages)}   section parsed from title: {parsed} "
          f"({100*parsed/len(passages):.1f}%)")

    tests_all = jl(TEST_FILE)
    tests, queries, gold, q_acts, q_sec = [], [], [], [], []
    for r in tests_all:
        key = base.norm(r["context"])
        if key not in idx_of:
            continue
        tests.append(r)
        queries.append(r["question"])
        gold.append(idx_of[key])
        q_acts.append(r.get("act", "?"))
        q_sec.append(parse_section(r["question"]))
    cited = sum(1 for s in q_sec if s)
    print(f"queries  {len(queries)}   citation parsed from question: {cited} "
          f"({100*cited/len(queries):.1f}%)")

    # how ambiguous is a bare section number?
    sec_counts = defaultdict(int)
    for s in p_sec:
        if s:
            sec_counts[s] += 1
    amb = [sec_counts[s] for s in q_sec if s and sec_counts[s]]
    if amb:
        print(f"passages sharing a cited section number: mean {sum(amb)/len(amb):.1f}, "
              f"max {max(amb)}")

    bm25 = base.BM25([base.tokenize(p) for p in passages])
    TOP = 10

    print("\n" + "=" * 78)
    print(f"  {'system':28s}" + "".join(f"{'R@'+str(k):>9s}" for k in K_VALUES) + f"{'MRR@10':>9s}")
    print("-" * 78)

    results, rankings_store = {}, {}

    plain = []
    for q in queries:
        plain.append(base.ranking_from_scores(bm25.scores(base.tokenize(q)), TOP))
    results["BM25"] = report("BM25", plain, gold)
    rankings_store["BM25"] = plain

    boosted = []
    for q, qs in zip(queries, q_sec):
        sc = bm25.scores(base.tokenize(q))
        if qs:
            for i, ps in enumerate(p_sec):
                if ps == qs:
                    sc[i] += BOOST
        boosted.append(base.ranking_from_scores(sc, TOP))
    results["BM25 + section boost"] = report("BM25 + section boost", boosted, gold)
    rankings_store["BM25 + section boost"] = boosted

    filtered, fell_back = [], 0
    for q, qs in zip(queries, q_sec):
        sc = bm25.scores(base.tokenize(q))
        cand = [i for i, ps in enumerate(p_sec) if qs and ps == qs]
        if not cand:
            fell_back += 1
            filtered.append(base.ranking_from_scores(sc, TOP))
        else:
            cand.sort(key=lambda i: -sc[i])
            filtered.append(cand[:TOP])
    results["BM25 + section filter"] = report("BM25 + section filter", filtered, gold)
    rankings_store["BM25 + section filter"] = filtered
    print("=" * 78)
    print(f"  (filter fell back to plain BM25 for {fell_back} of {len(queries)} queries)")

    # ---- per-Act recall@5 -------------------------------------------------
    print("\nper-Act recall@5")
    print(f"  {'act':34s}{'BM25':>9s}{'boost':>9s}{'filter':>9s}")
    acts = sorted(set(q_acts))
    per_act = {}
    for a in acts:
        row = {}
        for name in ("BM25", "BM25 + section boost", "BM25 + section filter"):
            rk = [r for r, qa in zip(rankings_store[name], q_acts) if qa == a]
            gd = [g for g, qa in zip(gold, q_acts) if qa == a]
            row[name] = recall_at(rk, gd, 5)
        per_act[a] = row
        print(f"  {a:34s}{row['BM25']:9.2f}{row['BM25 + section boost']:9.2f}"
              f"{row['BM25 + section filter']:9.2f}")

    # ---- citation-template-only vs lexically-content-bearing --------------
    try:
        from citation_specified_questions import (
            CITATION_TEMPLATE_ONLY,
            LEXICALLY_CONTENT_BEARING,
            is_citation_template_only,
        )

        template_mask = [is_citation_template_only(q) for q in queries]
        print("\nrecall@5 by question type")
        print(f"  {'type':34s}{'BM25':>9s}{'boost':>9s}{'filter':>9s}")
        split = {}
        for label, want in (
            (CITATION_TEMPLATE_ONLY, True),
            (LEXICALLY_CONTENT_BEARING, False),
        ):
            row = {}
            for name in ("BM25", "BM25 + section boost", "BM25 + section filter"):
                rk = [r for r, m in zip(rankings_store[name], template_mask) if m == want]
                gd = [g for g, m in zip(gold, template_mask) if m == want]
                row[name] = recall_at(rk, gd, 5)
            split[label] = row
            print(f"  {label+' (n=%d)'%sum(1 for m in template_mask if m==want):34s}"
                  f"{row['BM25']:9.2f}{row['BM25 + section boost']:9.2f}"
                  f"{row['BM25 + section filter']:9.2f}")
    except Exception as e:
        split = {}
        print(f"\n[skip] operational question split unavailable: {e}")

    payload = {
        "n_passages": len(passages), "n_queries": len(queries),
        "citation_parsed_pct": round(100 * cited / len(queries), 2),
        "results": results, "per_act_recall5": per_act,
        "by_question_type_recall5": split, "boost": BOOST,
    }
    (OUT_DIR / "citation_aware_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    CANONICAL_REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT_DIR / 'citation_aware_summary.json'}")
    print(f"canonical report: {CANONICAL_REPORT}")

    if args.with_reader:
        evaluate_reader(
            tests=tests,
            queries=queries,
            gold=gold,
            passages=passages,
            corpus=corpus,
            rankings_store=rankings_store,
            question_type_mask=(
                template_mask if "template_mask" in locals() else []
            ),
            args=args,
        )


if __name__ == "__main__":
    main()
