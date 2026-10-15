"""Verify and summarize the human audit for the retrieval manuscript.

Read-only with respect to the input workbooks. Agreement is recomputed from
original ratings, not consensus or later revised labels. Formula-linked case
IDs and strata are resolved from Metadata_Key rather than stale Excel caches.
The JSON output contains aggregate results and file hashes, not reviewer names
or free-text annotations. Run from any directory with --workbook if necessary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import warnings

import openpyxl

import update_human_validation_agreement as agreement


ROOT = Path(__file__).resolve().parents[2]
CONSENSUS_FIELDS = {
    "Question uniquely identifies annotated source": "Consensus: question uniquely identifies annotated source",
    "Question uniquely identifies selected source": "Consensus: selected source uniquely identified",
    "Annotated source legally relevant": "Consensus: annotated source legally relevant",
    "Selected source legally relevant": "Consensus: selected source legally relevant",
    "Sources legally equivalent": "Consensus: sources legally equivalent",
    "Annotated answer legally correct": "Consensus: annotated answer correct",
    "Predicted answer legally correct": "Consensus: predicted answer legally correct",
    "Predicted-answer error materiality": "Consensus: error materiality",
    "Annotated-answer error materiality": "Consensus: annotated-answer materiality",
    "Text quality issue": "Consensus: text quality issue",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(workbook, value):
    """Resolve direct cell references only; refuse arbitrary Excel formulas."""
    if not isinstance(value, str) or not value.startswith("="):
        return value
    match = re.fullmatch(r"='?([^'!]+)'?!([$]?[A-Z]+[$]?\d+)", value)
    require(match is not None, f"Unsupported formula: {value}")
    return resolve(workbook, workbook[match[1]][match[2].replace("$", "")].value)


def same_number(actual, expected, label):
    if actual is None or expected is None:
        require(actual is expected, f"Mismatch for {label}: {actual} != {expected}")
    else:
        require(math.isclose(actual, expected, abs_tol=1e-10),
                f"Mismatch for {label}: {actual} != {expected}")


def check_agreement(workbook, sheet_name, input_c):
    normalized = agreement.normalized_records(workbook, input_c)
    overall = {}
    rows = agreement.sheet_records(workbook[sheet_name])
    require(len(rows) == 50, f"Expected 50 agreement rows in {sheet_name}")
    for row in rows:
        scope, judgment = row["Scope / stratum"], row["Judgment"]
        left, right, labels = agreement.values_for(normalized, scope, judgment)
        n = len(left)
        exact = sum(a == b for a, b in zip(left, right))
        kappa = agreement.cohen_kappa(left, right, labels)
        ac1 = agreement.gwet_ac1(left, right, labels)
        expected = sum(left.count(c) * right.count(c) for c in labels) / n**2
        computed = {"Paired cases": n, "Exact agreements": exact,
                    "Disagreements": n-exact, "Raw agreement": exact/n,
                    "Expected agreement": expected,
                    "Cohen kappa (nominal)": kappa, "Gwet AC1": ac1}
        if "materiality" in judgment:
            kw, nw, excluded = agreement.quadratic_weighted_kappa(
                left, right, agreement.MATERIALITY_ORDER)
            computed.update({"Quadratic weighted kappa": kw,
                             "Weighted-kappa paired cases": nw,
                             "Excluded from weighted kappa": excluded})
        for field, value in computed.items():
            same_number(value, row[field], f"{sheet_name}/{scope}/{judgment}/{field}")
        if scope == "All 80 supplied cases":
            overall[judgment] = dict(computed, categories=labels,
                                     left_counts=dict(Counter(left)),
                                     right_counts=dict(Counter(right)))
    return overall


def public_item_records(workbook):
    """Return de-identified categorical ratings and consensus decisions.

    The public records omit annotator names, free-text rationales, authority
    notes, dates, and the private passage-order decoding key. Labels are
    normalized to the annotated/selected source roles used in the analysis.
    """
    metadata, _ = agreement.load_mapping(workbook)
    original = agreement.normalized_records(workbook, "Original_Input_C")
    adjudication = {
        resolve(workbook, row["Case ID"]): row
        for row in agreement.sheet_records(workbook["Adjudication"])
    }
    records = []
    for case_id in sorted(metadata):
        meta = metadata[case_id]
        record = {
            "case_id": case_id,
            "question_id": meta["question_id"],
            "stage_outcome": meta["stage_outcome"],
            "sampling_weight": meta["sampling_weight"],
            "original_ratings": {"annotator_A": {}, "annotator_C": {}},
            "consensus": {},
        }
        for judgment in list(agreement.ROLE_FIELDS) + list(agreement.CASE_FIELDS):
            if judgment in agreement.ROLE_FIELDS:
                family, role, _ = agreement.ROLE_FIELDS[judgment]
                key = (family, role)
            else:
                key = ("case", judgment)
            record["original_ratings"]["annotator_A"][judgment] = original[case_id]["A"].get(key)
            record["original_ratings"]["annotator_C"][judgment] = original[case_id]["C"].get(key)
        row = adjudication[case_id]
        for judgment, field in CONSENSUS_FIELDS.items():
            record["consensus"][judgment] = row[field]
        records.append(record)
    return records


def summarize(workbook_path, predictions_path, release_output=None):
    initial_hash = sha256(workbook_path)
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=False)
    metadata, mapping = agreement.load_mapping(workbook)
    require(len(metadata) == 80, "Expected 80 distinct audit cases")
    judgments = []
    for row in agreement.sheet_records(workbook["Adjudication"]):
        case_id = resolve(workbook, row["Case ID"])
        meta = metadata[case_id]
        require(resolve(workbook, row["Stage outcome"]) == meta["stage_outcome"], "Stratum mismatch")
        require(resolve(workbook, row["Question"]) == meta["question"], "Question mismatch")
        for field in ("A confirms final row", "C confirms final row"):
            require(row[field] == "Yes", f"Missing confirmation {case_id}: {field}")
        for field in ("Consensus legal rationale", "Joint discussion record",
                      "Consensus reviewers", "Consensus discussion date"):
            require(bool(row[field]), f"Missing record {case_id}: {field}")
        values = {judgment: row[field] for judgment, field in CONSENSUS_FIELDS.items()}
        for judgment, value in values.items():
            labels = (agreement.ROLE_FIELDS[judgment][2] if judgment in agreement.ROLE_FIELDS
                      else agreement.CASE_FIELDS[judgment][1])
            require(value in labels, f"Invalid consensus category {case_id}/{judgment}: {value}")
        judgments.append(dict(case_id=case_id, stage=meta["stage_outcome"], labels=values))
    require(len({r['case_id'] for r in judgments}) == 80, "Duplicate/missing consensus case")

    # Verify displayed evidence and its role mapping; compare all snapshots.
    for sheet_name, source_code in [("Frozen_Input_A", "A"), ("Frozen_Input_C", "C"),
                                    ("Original_Input_C", "C")]:
        for row in agreement.sheet_records(workbook[sheet_name]):
            case_id = row["Case ID"]
            meta = metadata[case_id]
            require(row["Question"] == meta["question"], f"Changed question {sheet_name}/{case_id}")
            expected = {
                "annotated": tuple(meta[k] for k in ["annotated_act", "annotated_source_title", "annotated_source_text", "annotated_answer"]),
                "selected": tuple(meta[k] for k in ["selected_act", "selected_source_title", "selected_source_text", "predicted_answer"]),
            }
            role = mapping[(case_id, source_code)]
            if role == "indistinguishable":
                require(expected["annotated"] == expected["selected"], "Invalid indistinguishable mapping")
                role = "annotated"
            for display, identity in [("A", role), ("B", "selected" if role == "annotated" else "annotated")]:
                actual = tuple(row[k] for k in [f"Passage {display} - Act", f"Passage {display} - title", f"Passage {display} - text", f"Answer {display}"])
                require(actual == expected[identity], f"Evidence mismatch {sheet_name}/{case_id}/{display}")

    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {r["id"]: r for r in predictions}

    def stage(row):
        if not row["retrieval_hit_at_k"]:
            return "Annotated source absent from top 5"
        if not row["selected_context_is_gold"]:
            return "Annotated source available; other source selected"
        return "Annotated source selected; exact answer" if row["em"] else "Annotated source selected; non-exact answer"

    require(len(by_id) == len(predictions) == 344, "Expected 344 unique predictions")
    population = Counter(stage(row) for row in predictions)
    rng = random.Random(42)
    sampled_ids = []
    for name in sorted(population):
        candidates = sorted((r for r in predictions if stage(r) == name), key=lambda r: r["id"])
        rng.shuffle(candidates)
        sampled_ids.extend(r["id"] for r in candidates[:20])
    require(set(sampled_ids) == {m["question_id"] for m in metadata.values()}, "Seed-42 sample differs")
    for meta in metadata.values():
        pred = by_id[meta["question_id"]]
        name = meta["stage_outcome"]
        require(stage(pred) == name, "Prediction stratum differs")
        require(meta["predicted_answer"] == pred["prediction"], "Prediction text differs")
        require(meta["population_count"] == population[name] and meta["sample_count"] == 20, "Design counts differ")
        same_number(meta["sampling_weight"], population[name]/20, "Sampling weight")

    public_records = public_item_records(workbook) if release_output else None
    original = check_agreement(workbook, "Agreement_Original", "Original_Input_C")
    revised = check_agreement(workbook, "Agreement_Frozen", "Frozen_Input_C")
    summaries = {}
    for name in ["All sampled cases"] + sorted(population):
        rows = [r for r in judgments if name == "All sampled cases" or r["stage"] == name]
        summaries[name] = {"sample_n": len(rows),
                           "population_n": 344 if name == "All sampled cases" else population[name],
                           "counts": {field: dict(Counter(r["labels"][field] for r in rows)) for field in CONSENSUS_FIELDS}}
    workbook.close()
    if release_output:
        release_output.parent.mkdir(parents=True, exist_ok=True)
        with release_output.open("w", encoding="utf-8") as stream:
            for record in public_records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    require(sha256(workbook_path) == initial_hash, "Input workbook changed")
    return {"workbook_sha256": initial_hash, "predictions_sha256": sha256(predictions_path),
            "sampling_seed": 42, "population_n": 344, "sample_n": 80,
            "primary_agreement": original, "revised_agreement_not_primary": revised,
            "consensus": summaries,
            "verification": {"original_and_revised_agreement_rows_recomputed": 100,
                             "all_three_snapshots_evidence_matched": True,
                             "sample_matches_seed_42": True, "workbook_unchanged": True},
            "interpretation": "Descriptive audit of a balanced diagnostic sample, not a population prevalence estimate. Original ratings measure agreement; consensus labels are a separate outcome. Completion records do not independently certify review conduct or legal correctness."}


def verify_manuscript(report, manuscript):
    tex = manuscript.read_text(encoding="utf-8")
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S)
    reliability = next(t for t in tables if r"\label{tab:human_agreement}" in t)
    consensus = next(t for t in tables if r"\label{tab:human_consensus}" in t)
    observed = []
    for line in reliability.splitlines():
        if "&" in line and re.search(r"&\s*\d+", line):
            observed.append(re.findall(r"\d+(?:\.\d+)?", line.split("&", 1)[1]))
    expected = [[str(r["Paired cases"]), str(len(r["categories"])),
                 f'{100*r["Raw agreement"]:.2f}', f'{r["Cohen kappa (nominal)"]:.3f}',
                 f'{r["Gwet AC1"]:.3f}'] for r in report["primary_agreement"].values()]
    require(observed == expected, "Manuscript reliability table differs from verified results")
    reliability_rows = len(expected)
    order = ["Annotated source absent from top 5", "Annotated source available; other source selected",
             "Annotated source selected; non-exact answer", "Annotated source selected; exact answer"]
    expected = []
    for stage in order:
        summary = report["consensus"][stage]
        c = summary["counts"]
        expected.append([summary["population_n"], c["Selected source legally relevant"].get("Yes", 0)]
                        + [c["Predicted answer legally correct"].get(k, 0) for k in ["Yes", "Partly", "No", "Unclear"]]
                        + [sum(c["Predicted-answer error materiality"].get(k, 0) for k in ["Moderate", "High"])])
    observed = [[int(x) for x in re.findall(r"\d+", line.split("&", 1)[1])]
                for line in consensus.splitlines() if "&" in line and re.search(r"&\s*\d+", line)]
    require(observed == expected, "Manuscript consensus table differs from verified results")
    return {"reliability_rows": reliability_rows, "consensus_rows": len(expected)}


def crosscheck_sklearn(workbook_path):
    """Independent library check of both nominal and ordinal kappa."""
    from sklearn.metrics import cohen_kappa_score

    def library_kappa(*args, **kwargs):
        # A constant, identical category gives 0/0, not perfect kappa.
        # Compare sklearn's NaN with our explicit None for this edge case.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
            value = cohen_kappa_score(*args, **kwargs)
        return None if math.isnan(value) else float(value)

    workbook = openpyxl.load_workbook(workbook_path, read_only=True)
    counts = Counter()
    for sn, cs in [("Agreement_Original", "Original_Input_C"), ("Agreement_Frozen", "Frozen_Input_C")]:
        records = agreement.normalized_records(workbook, cs)
        for row in agreement.sheet_records(workbook[sn]):
            left, right, labels = agreement.values_for(records, row["Scope / stratum"], row["Judgment"])
            external = library_kappa(left, right, labels=labels)
            same_number(external, agreement.cohen_kappa(left, right, labels), "sklearn nominal kappa")
            counts["nominal"] += 1
            if "materiality" in row["Judgment"]:
                labels = agreement.MATERIALITY_ORDER
                pairs = [(a, b) for a, b in zip(left, right) if a in labels and b in labels]
                external = library_kappa([a for a, _ in pairs], [b for _, b in pairs], labels=labels, weights="quadratic")
                same_number(external, agreement.quadratic_weighted_kappa(left, right, labels)[0], "sklearn quadratic kappa")
                counts["weighted"] += 1
    workbook.close()
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=agreement.DEFAULT_WORKBOOK)
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/experiments/retrieval/runs/citation_aware_v1/bm25_section_filter_top5_e2_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/human_validation_report/manuscript_summary.json")
    parser.add_argument("--release-output", type=Path,
                        help="Optional de-identified item-level JSONL output")
    parser.add_argument("--verify-manuscript", type=Path)
    parser.add_argument("--crosscheck-sklearn", action="store_true")
    args = parser.parse_args()
    report = summarize(args.workbook, args.predictions, args.release_output)
    if args.verify_manuscript:
        report["verification"]["manuscript_tables"] = verify_manuscript(report, args.verify_manuscript)
    if args.crosscheck_sklearn:
        report["verification"]["independent_sklearn_checks"] = crosscheck_sklearn(args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified and wrote {args.output}")
    for name, row in report["consensus"].items():
        print(name, json.dumps(row))


if __name__ == "__main__":
    main()
