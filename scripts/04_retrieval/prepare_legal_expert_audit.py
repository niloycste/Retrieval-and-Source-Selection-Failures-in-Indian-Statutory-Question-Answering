"""Create a deterministic, blinded legal-expert audit packet.

The packet samples 20 IndicIPR-QA items from each mutually exclusive stage
outcome under section-filter retrieval (80 total).  Question-only judgements
are collected before evidence judgements.  In the evidence phase, annotated
and system-selected sources are randomised as Source A/Source B and decoded
only in a private key.  Two identical, blank annotator files are produced.
Nothing in this script infers legal correctness from benchmark agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data" / "splits" / "test.jsonl"
CORPUS = ROOT / "data" / "processed" / "indic_ipr_qa" / "unique_contexts.jsonl"
PREDICTIONS = (
    ROOT / "results" / "experiments" / "retrieval" / "runs"
    / "citation_aware_v1" / "bm25_section_filter_top5_e2_predictions.jsonl"
)
CLASSIFIER = ROOT / "results" / "experiments" / "retrieval" / "citation_template_classifier_audit.csv"
DEFAULT_OUT = ROOT / "outputs" / "retrieval_legal_expert_audit"
SEED = 42


def jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stage(row: dict) -> str:
    if not row["retrieval_hit_at_k"]:
        return "Annotated source absent from top 5"
    if not row["selected_context_is_gold"]:
        return "Annotated source available; other source selected"
    if not row["em"]:
        return "Annotated source selected; non-exact answer"
    return "Annotated source selected; exact answer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-stage", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if args.per_stage < 1:
        parser.error("--per-stage must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = {row["id"]: row for row in jsonl(TEST)}
    corpus = jsonl(CORPUS)
    predictions = jsonl(PREDICTIONS)
    with CLASSIFIER.open("r", encoding="utf-8-sig", newline="") as stream:
        classes = {row["question_id"]: row for row in csv.DictReader(stream)}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[stage(row)].append(row)
    rng = random.Random(SEED)
    selected: list[dict] = []
    for name in sorted(grouped):
        items = sorted(grouped[name], key=lambda row: row["id"])
        rng.shuffle(items)
        selected.extend(items[: min(args.per_stage, len(items))])
    selected.sort(key=lambda row: (stage(row), row["id"]))

    phase_one_fields = [
        "case_id", "question", "annotator_id",
        "act_identifiability", "provision_identifiability",
        "answerable_as_written", "confidence_1_to_5", "legal_rationale",
    ]
    phase_two_fields = [
        "case_id", "question", "source_a_act", "source_a_title",
        "source_a_text", "source_b_act", "source_b_title", "source_b_text",
        "reference_answer", "system_answer", "annotator_id",
        "source_a_support", "source_b_support", "sources_equivalent",
        "reference_answer_correctness", "system_answer_correctness",
        "system_answer_completeness", "error_materiality",
        "confidence_1_to_5", "legal_rationale", "authorities_or_notes",
    ]
    key_fields = [
        "case_id", "question_id", "stage_outcome", "question_type",
        "annotated_act", "annotated_section", "source_a_identity",
        "source_b_identity", "selected_context_index", "selected_rank",
        "answer_em", "answer_f1",
    ]
    phase_one_rows, phase_two_rows, key_rows = [], [], []
    for number, pred in enumerate(selected, start=1):
        item = test[pred["id"]]
        selected_index = int(pred["selected_context_index"])
        selected_source = corpus[selected_index]
        case_id = f"LEX-{number:03d}"
        annotated = {
            "identity": "annotated",
            "act": item.get("act", ""),
            "title": item.get("title", ""),
            "text": item["context"],
        }
        system = {
            "identity": "system-selected",
            "act": pred.get("selected_context_act", selected_source.get("act", "")),
            "title": pred.get("selected_context_title", selected_source.get("title", "")),
            "text": selected_source["context"],
        }
        # Randomise even when the two records contain the same source.  The
        # private key preserves that fact without exposing it to annotators.
        if rng.random() < 0.5:
            source_a, source_b = annotated, system
        else:
            source_a, source_b = system, annotated
        phase_one_rows.append({
            "case_id": case_id,
            "question": item["question"],
            "annotator_id": "",
            "act_identifiability": "",
            "provision_identifiability": "",
            "answerable_as_written": "",
            "confidence_1_to_5": "",
            "legal_rationale": "",
        })
        phase_two_rows.append({
            "case_id": case_id,
            "question": item["question"],
            "source_a_act": source_a["act"],
            "source_a_title": source_a["title"],
            "source_a_text": source_a["text"],
            "source_b_act": source_b["act"],
            "source_b_title": source_b["title"],
            "source_b_text": source_b["text"],
            "reference_answer": item["answer"],
            "system_answer": pred["prediction"],
            "annotator_id": "",
            "source_a_support": "",
            "source_b_support": "",
            "sources_equivalent": "",
            "reference_answer_correctness": "",
            "system_answer_correctness": "",
            "system_answer_completeness": "",
            "error_materiality": "",
            "confidence_1_to_5": "",
            "legal_rationale": "",
            "authorities_or_notes": "",
        })
        key_rows.append({
            "case_id": case_id,
            "question_id": pred["id"],
            "stage_outcome": stage(pred),
            "question_type": classes.get(pred["id"], {}).get("assigned_class", ""),
            "annotated_act": item.get("act", ""),
            "annotated_section": item.get("section_no", ""),
            "source_a_identity": source_a["identity"],
            "source_b_identity": source_b["identity"],
            "selected_context_index": selected_index,
            "selected_rank": pred.get("selected_rank", ""),
            "answer_em": pred["em"],
            "answer_f1": pred["f1"],
        })

    for annotator in ("A", "B"):
        for phase, fields, rows in (
            ("phase1_question_only", phase_one_fields, phase_one_rows),
            ("phase2_evidence", phase_two_fields, phase_two_rows),
        ):
            destination = output_dir / f"annotator_{annotator}_{phase}.csv"
            with destination.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    rendered = dict(row)
                    rendered["annotator_id"] = annotator
                    writer.writerow(rendered)

    key_csv = output_dir / "PRIVATE_DECODING_KEY_DO_NOT_SHARE.csv"
    with key_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=key_fields)
        writer.writeheader()
        writer.writerows(key_rows)

    adjudication_fields = [
        "case_id", "phase", "field", "annotator_a_label",
        "annotator_b_label", "adjudicated_label", "adjudicator_id",
        "adjudication_rationale",
    ]
    with (output_dir / "adjudication_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=adjudication_fields)
        writer.writeheader()

    codebook = output_dir / "LEGAL_EXPERT_AUDIT_CODEBOOK.md"
    codebook.write_text(
        """# Legal-expert audit codebook

## Purpose

This audit evaluates legal relevance and equivalence independently of the
benchmark's annotated-source identity. Benchmark agreement must not be used as
a substitute for legal judgement.

## Procedure

1. Two legally qualified annotators independently review every case. Do not
   share labels or discuss cases until both files are frozen.
2. Complete Phase 1 before opening Phase 2. Phase 1 contains only the question
   and measures whether the Act, provision, and answer are identifiable as
   written.
3. Phase 2 presents two sources in random A/B order. Do not inspect the private
   decoding key. Consult an authoritative version of the relevant statute when
   necessary and record the citation or URL under `authorities_or_notes`.
4. Allowed Phase 1 labels:
   * `act_identifiability` and `provision_identifiability`: `Unique`,
     `Multiple plausible`, `Not identifiable`, or `Unclear`.
   * `answerable_as_written`: `Yes`, `No`, or `Unclear`.
5. Allowed Phase 2 labels:
   * source support, answer correctness, and answer completeness: `Fully`,
     `Partly`, `No`, or `Unclear`.
   * source equivalence: `Yes`, `Partly`, `No`, or `Unclear`.
   * error materiality: `None`, `Low`, `Moderate`, `High`, or `Unclear`.
6. Give confidence from 1 (very uncertain) to 5 (very certain), with a concise
   legal rationale. After independent labels are frozen, calculate raw
   agreement and an agreement coefficient for each categorical item. Because
   prevalence may be imbalanced, report Gwet's AC1 alongside Cohen's kappa;
   use weighted kappa for ordered materiality labels.
7. Send disagreements to a third qualified expert, or adjudicate by consensus
   while retaining both original labels and a written rationale.

## Sampling

The sample uses seed 42 and includes an equal number from each of four
mutually exclusive stage outcomes under the section-filter top-five pipeline.
The exact-match group is a positive control, not an error category. The sample
is designed to estimate audit outcomes within these sampled strata, not the
unweighted prevalence of legal errors in the full test set. Population-level
estimates require stratum weights from `sampling_manifest.json`.

## Before manuscript use

Confirm institutional requirements for expert annotation, consent, and ethics
review before data collection. Do not retain annotator names in the public
artifact. Report annotator qualifications, independent annotation procedure,
agreement, adjudication, and sampling limitations. Do not replace uncertainty
labels with inferred values.
""",
        encoding="utf-8",
    )
    counts = Counter(row["stage_outcome"] for row in key_rows)
    (output_dir / "sampling_manifest.json").write_text(json.dumps({
        "seed": SEED,
        "source_predictions": str(PREDICTIONS.relative_to(ROOT)),
        "per_stage_requested": args.per_stage,
        "sample_counts": dict(counts),
        "population_counts": {name: len(values) for name, values in grouped.items()},
        "n_cases": len(key_rows),
        "source_order_blinded": True,
        "two_independent_annotator_packets": True,
        "legal_labels_prepopulated": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "n_cases": len(key_rows),
        "counts": counts,
        "private_key": str(key_csv),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
