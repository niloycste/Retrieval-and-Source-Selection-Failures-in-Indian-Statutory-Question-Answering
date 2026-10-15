"""Item-level follow-up controls for the Indian retrieval study.

This is a CPU-only diagnostic. It preserves the existing canonical artifacts,
uses their BM25 definition and saved reader predictions, and writes all new
outputs to a separate run directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from citation_specified_questions import (
    CITATION_TEMPLATE_ONLY,
    LEXICALLY_CONTENT_BEARING,
    classification_label,
)
from retrieval_diagnostic_controls import (
    ACT_NAMES,
    BM25,
    BOOST,
    SECTION_RE,
    load_jsonl,
    normalize,
    parse_section,
    rank,
    retrieval_metrics,
    tokenize,
)


ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILE = ROOT / "data" / "processed" / "indic_ipr_qa" / "unique_contexts.jsonl"
TEST_FILE = ROOT / "data" / "splits" / "test.jsonl"
PREDICTION_DIR = (
    ROOT / "results" / "experiments" / "retrieval" / "runs" / "citation_aware_v1"
)
OUTPUT_DIR = (
    ROOT
    / "results"
    / "experiments"
    / "retrieval"
    / "runs"
    / "followup_structural_controls_v1"
)
PROTOCOL_FILE = ROOT / "docs" / "RETRIEVAL_FOLLOWUP_CONTROLS_PROTOCOL.md"

SEED = 42
TIE_ITERATIONS = 10_000
K = 5

ACT_ALIASES = {
    "copyright_act": ["Copyright Act", "Copyright Act 1957"],
    "design_act": ["Designs Act", "Designs Act 2000"],
    "gi_act": [
        "Geographical Indications of Goods Act",
        "Geographical Indications of Goods Registration and Protection Act",
    ],
    "patent_act": ["Patents Act", "Patents Act 1970", "Patent Act"],
    "plant_varieties_farmers_rights_act": [
        "Protection of Plant Varieties and Farmers Rights Act",
        "Protection of Plant Varieties and Farmers' Rights Act",
        "Plant Varieties and Farmers Rights Act",
    ],
    "semiconductor_layout_design_act": [
        "Semiconductor Integrated Circuits Layout Design Act",
        "Semiconductor Integrated Circuits Layout-Design Act",
    ],
    "trademark_act": ["Trade Marks Act", "Trade Marks Act 1999", "Trademark Act"],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_sections(text: str) -> list[str]:
    """Return all references recognised by the existing section rule."""
    values: list[str] = []
    for value in SECTION_RE.findall(text or ""):
        normalized = "".join(str(value).split()).upper()
        if normalized not in values:
            values.append(normalized)
    return values


def normalized_words(text: str) -> str:
    return " ".join(tokenize(text))


def detect_query_acts(question: str) -> list[str]:
    normalized_question = f" {normalized_words(question)} "
    detected: list[str] = []
    for act_id, aliases in ACT_ALIASES.items():
        if any(
            f" {normalized_words(alias)} " in normalized_question for alias in aliases
        ):
            detected.append(act_id)
    return detected


def rank_candidates(scores: Sequence[float], candidates: Sequence[int]) -> list[int]:
    return sorted(candidates, key=lambda index: (-scores[index], index))


def subgroup_metrics(
    indices: Sequence[int], rankings: Sequence[Sequence[int]], gold: Sequence[int]
) -> dict:
    if not indices:
        return {"n": 0, "recall@1": None, "recall@3": None, "recall@5": None,
                "recall@10": None, "mrr@10": None}
    selected_rankings = [list(rankings[index]) for index in indices]
    selected_gold = [gold[index] for index in indices]
    return {"n": len(indices), **retrieval_metrics(selected_rankings, selected_gold)}


def reader_mean(indices: Sequence[int], rows_by_id: dict[str, dict], test: Sequence[dict]) -> dict:
    if not indices:
        return {"f1_pct": None, "em_pct": None}
    rows = [rows_by_id[test[index]["id"]] for index in indices]
    return {
        "f1_pct": round(100.0 * statistics.fmean(float(row["f1"]) for row in rows), 4),
        "em_pct": round(100.0 * statistics.fmean(float(row["em"]) for row in rows), 4),
    }


def structural_status(
    references: Sequence[str], candidates: Sequence[int], candidate_acts: set[str]
) -> str:
    if not references:
        return "No recognised locator"
    if not candidates:
        return "Unmatched locator"
    if len(references) > 1:
        return "Multiple query references"
    if len(candidates) == 1:
        return "Single matching passage"
    if len(candidate_acts) == 1:
        return "Multiple passages within one Act"
    return "Matches across multiple Acts"


def performance_group(references: Sequence[str], candidate_acts: set[str]) -> str:
    if not references:
        return "No recognised locator"
    if not candidate_acts:
        return "Unmatched locator"
    if len(candidate_acts) == 1:
        return "One matching Act"
    return "Multiple matching Acts"


def candidate_bin(count: int) -> str:
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    return "11+"


def tie_record(scores: Sequence[float], ranking: Sequence[int], gold: int, k: int) -> dict:
    top_score = scores[ranking[0]]
    top_tie = [index for index in ranking if math.isclose(scores[index], top_score, abs_tol=1e-12)]
    if len(ranking) <= k:
        return {
            "number_of_top_score_ties": len(top_tie),
            "tie_at_rank_5_boundary": False,
            "gold_in_boundary_tie": False,
            "boundary_tie_size": 0,
            "slots_available_in_boundary_tie": 0,
        }
    cutoff_score = scores[ranking[k - 1]]
    boundary = [
        index
        for index in ranking
        if math.isclose(scores[index], cutoff_score, abs_tol=1e-12)
    ]
    above = sum(
        scores[index] > cutoff_score
        and not math.isclose(scores[index], cutoff_score, abs_tol=1e-12)
        for index in ranking
    )
    tie_at_boundary = math.isclose(scores[ranking[k]], cutoff_score, abs_tol=1e-12)
    return {
        "number_of_top_score_ties": len(top_tie),
        "tie_at_rank_5_boundary": tie_at_boundary,
        "gold_in_boundary_tie": tie_at_boundary and gold in boundary,
        "boundary_tie_size": len(boundary) if tie_at_boundary else 0,
        "slots_available_in_boundary_tie": (k - above) if tie_at_boundary else 0,
    }


def randomized_boundary_recall(
    score_rows: Sequence[Sequence[float]],
    rankings: Sequence[Sequence[int]],
    gold: Sequence[int],
    iterations: int,
    seed: int,
    k: int,
) -> dict:
    cases: list[tuple[bool, list[int], int]] = []
    deterministic = 0
    for scores, ranking, target in zip(score_rows, rankings, gold):
        if len(ranking) <= k:
            deterministic += target in ranking
            cases.append((True, [], 0))
            continue
        cutoff_score = scores[ranking[k - 1]]
        above = [index for index in ranking if scores[index] > cutoff_score and
                 not math.isclose(scores[index], cutoff_score, abs_tol=1e-12)]
        boundary = [index for index in ranking if math.isclose(scores[index], cutoff_score, abs_tol=1e-12)]
        slots = k - len(above)
        if len(boundary) <= slots:
            deterministic += target in ranking[:k]
            cases.append((True, [], 0))
        elif target in above:
            deterministic += 1
            cases.append((True, [], 0))
        elif target not in boundary:
            cases.append((True, [], 0))
        else:
            cases.append((False, boundary, slots))

    rng = random.Random(seed)
    values: list[float] = []
    variable_cases = [case for case in cases if not case[0]]
    for _ in range(iterations):
        hits = deterministic
        for _, boundary, slots in variable_cases:
            hits += 1 if rng.randrange(len(boundary)) < slots else 0
        values.append(100.0 * hits / len(gold))
    values.sort()
    return {
        "recall@5_mean_pct": round(statistics.fmean(values), 4),
        "recall@5_sd_pct": round(statistics.stdev(values), 4),
        "recall@5_95_simulation_interval_pct": [
            round(values[int(0.025 * (len(values) - 1))], 4),
            round(values[int(0.975 * (len(values) - 1))], 4),
        ],
        "queries_whose_gold_is_in_randomized_boundary_tie": len(variable_cases),
        "iterations": iterations,
        "seed": seed,
    }


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_jsonl(CORPUS_FILE)
    test = load_jsonl(TEST_FILE)

    paragraphs = [normalize(row["context"]) for row in corpus]
    titles = [normalize(row.get("title", "")) for row in corpus]
    acts = [str(row.get("act", "")) for row in corpus]
    passage_ids = [str(row["id"]) for row in corpus]
    passage_sections = [parse_section(title) for title in titles]
    questions = [str(row["question"]) for row in test]
    references = [parse_sections(question) for question in questions]
    primary_sections = [values[0] if values else None for values in references]

    by_context: dict[str, int] = {}
    for index, paragraph in enumerate(paragraphs):
        by_context.setdefault(paragraph, index)
    gold = [by_context[normalize(row["context"])] for row in test]

    by_section: dict[str, list[int]] = defaultdict(list)
    by_act: dict[str, list[int]] = defaultdict(list)
    for index, (section, act) in enumerate(zip(passage_sections, acts)):
        by_act[act].append(index)
        if section is not None:
            by_section[section].append(index)

    bm25 = BM25([tokenize(text) for text in paragraphs])
    score_rows = [bm25.scores(tokenize(question)) for question in questions]
    global_rankings = [rank(scores) for scores in score_rows]

    section_rankings: list[list[int]] = []
    boost_rankings: list[list[int]] = []
    oracle_act_rankings: list[list[int]] = []
    oracle_act_section_rankings: list[list[int]] = []
    section_candidates_by_query: list[list[int]] = []
    all_reference_candidates_by_query: list[list[int]] = []

    for index, (values, primary, scores) in enumerate(
        zip(references, primary_sections, score_rows)
    ):
        primary_candidates = list(by_section.get(primary, [])) if primary else []
        all_candidates = sorted({candidate for value in values for candidate in by_section.get(value, [])})
        section_candidates_by_query.append(primary_candidates)
        all_reference_candidates_by_query.append(all_candidates)

        section_ranking = (
            rank_candidates(scores, primary_candidates)
            if primary_candidates
            else global_rankings[index]
        )
        section_rankings.append(section_ranking)

        boosted = list(scores)
        for candidate in primary_candidates:
            boosted[candidate] += BOOST
        boost_rankings.append(rank(boosted))

        act_candidates = by_act[str(test[index]["act"])]
        act_ranking = rank_candidates(scores, act_candidates)
        oracle_act_rankings.append(act_ranking)
        act_section_candidates = [
            candidate for candidate in primary_candidates if acts[candidate] == test[index]["act"]
        ]
        oracle_act_section_rankings.append(
            rank_candidates(scores, act_section_candidates)
            if act_section_candidates
            else act_ranking
        )

    prediction_files = {
        "bm25": PREDICTION_DIR / "bm25_top5_e2_predictions.jsonl",
        "section_boost": PREDICTION_DIR / "bm25_section_boost_top5_e2_predictions.jsonl",
        "section_filter": PREDICTION_DIR / "bm25_section_filter_top5_e2_predictions.jsonl",
    }
    predictions = {
        name: {row["id"]: row for row in load_jsonl(path)}
        for name, path in prediction_files.items()
    }

    input_rows = []
    for index, row in enumerate(corpus):
        title_plus = normalize(f"{titles[index]} {paragraphs[index]}")
        act_name = ACT_NAMES.get(acts[index], acts[index])
        input_rows.append(
            {
                "passage_id": passage_ids[index],
                "act_id": acts[index],
                "section_id": passage_sections[index],
                "title": titles[index],
                "paragraph": paragraphs[index],
                "bm25_input": paragraphs[index],
                "dense_input": paragraphs[index],
                "cross_encoder_passage_input": paragraphs[index],
                "title_inclusive_bm25_control_input": title_plus,
                "act_title_paragraph_control_input": normalize(
                    f"{act_name} {title_plus}"
                ),
            }
        )
    write_jsonl(OUTPUT_DIR / "passage_input_audit.jsonl", input_rows)

    item_rows: list[dict] = []
    t2_group_indices: dict[str, list[int]] = defaultdict(list)
    collision_bin_indices: dict[str, list[int]] = defaultdict(list)
    tie_rows: list[dict] = []
    boost_filter_rows: list[dict] = []

    for index, row in enumerate(test):
        candidates = all_reference_candidates_by_query[index]
        candidate_acts = {acts[candidate] for candidate in candidates}
        detected_acts = detect_query_acts(questions[index])
        constrained = [
            candidate for candidate in candidates if acts[candidate] in detected_acts
        ] if detected_acts else []
        status = structural_status(references[index], candidates, candidate_acts)
        group = performance_group(references[index], candidate_acts)
        t2_group_indices[group].append(index)
        if group == "Multiple matching Acts":
            collision_bin_indices[candidate_bin(len(candidates))].append(index)

        item_rows.append(
            {
                "question_id": row["id"],
                "question": questions[index],
                "annotated_act": row["act"],
                "annotated_gold_passage_id": passage_ids[gold[index]],
                "parsed_references": references[index],
                "system_primary_reference": primary_sections[index],
                "detected_act_from_query": detected_acts,
                "section_matching_passages": len(candidates),
                "section_matching_passage_ids": [passage_ids[value] for value in candidates],
                "section_matching_acts": len(candidate_acts),
                "section_matching_act_ids": sorted(candidate_acts),
                "act_section_matching_passages": len(constrained) if detected_acts else None,
                "act_section_matching_passage_ids": [passage_ids[value] for value in constrained]
                if detected_acts else None,
                "gold_in_section_candidates": gold[index] in candidates,
                "gold_in_act_section_candidates": gold[index] in constrained
                if detected_acts else None,
                "reference_status": status,
                "performance_group": group,
            }
        )

        tie = tie_record(score_rows[index], global_rankings[index], gold[index], K)
        section_scores = score_rows[index]
        section_tie = tie_record(
            section_scores, section_rankings[index], gold[index], K
        )
        tie_rows.append(
            {
                "question_id": row["id"],
                **{f"bm25_{key}": value for key, value in tie.items()},
                **{f"section_filter_{key}": value for key, value in section_tie.items()},
            }
        )

        boost_top = boost_rankings[index][:K]
        filter_top = section_rankings[index][:K]
        structural_set = set(section_candidates_by_query[index])
        qid = row["id"]
        selected_boost_index = predictions["section_boost"][qid].get("selected_context_index")
        selected_filter_index = predictions["section_filter"][qid].get("selected_context_index")
        boost_filter_rows.append(
            {
                "question_id": qid,
                "matching_candidate_count": len(structural_set),
                "fallback_used": not bool(structural_set),
                "boost_candidate_ids": [passage_ids[value] for value in boost_top],
                "filter_candidate_ids": [passage_ids[value] for value in filter_top],
                "nonmatching_candidates_in_boost": [
                    passage_ids[value] for value in boost_top if value not in structural_set
                ] if structural_set else [],
                "top5_sets_differ": set(boost_top) != set(filter_top),
                "selected_passage_boost": passage_ids[int(selected_boost_index)]
                if selected_boost_index is not None else None,
                "selected_passage_filter": passage_ids[int(selected_filter_index)]
                if selected_filter_index is not None else None,
                "selected_passages_differ": selected_boost_index != selected_filter_index,
                "boost_f1": float(predictions["section_boost"][qid]["f1"]),
                "filter_f1": float(predictions["section_filter"][qid]["f1"]),
            }
        )

    write_jsonl(OUTPUT_DIR / "indian_reference_collision_items.jsonl", item_rows)
    write_csv(
        OUTPUT_DIR / "indian_reference_collision_items.csv",
        [
            {
                **{key: value for key, value in row.items() if not isinstance(value, list)},
                "parsed_references": " | ".join(row["parsed_references"]),
                "detected_act_from_query": " | ".join(row["detected_act_from_query"]),
                "section_matching_passage_ids": " | ".join(row["section_matching_passage_ids"]),
                "section_matching_act_ids": " | ".join(row["section_matching_act_ids"]),
                "act_section_matching_passage_ids": " | ".join(row["act_section_matching_passage_ids"] or []),
            }
            for row in item_rows
        ],
    )
    write_jsonl(OUTPUT_DIR / "tie_audit_items.jsonl", tie_rows)
    write_jsonl(OUTPUT_DIR / "boost_filter_trace.jsonl", boost_filter_rows)

    def joined_group(indices: Sequence[int]) -> dict:
        bm25 = subgroup_metrics(indices, global_rankings, gold)
        section = subgroup_metrics(indices, section_rankings, gold)
        return {
            "n": len(indices),
            "bm25_recall@1_pct": bm25["recall@1"],
            "bm25_recall@5_pct": bm25["recall@5"],
            "section_filter_recall@5_pct": section["recall@5"],
            "bm25_reader": reader_mean(indices, predictions["bm25"], test),
            "section_filter_reader": reader_mean(indices, predictions["section_filter"], test),
        }

    exact_random_values = []
    for index, candidates in enumerate(section_candidates_by_query):
        if candidates:
            expected = min(K, len(candidates)) / len(candidates) if gold[index] in candidates else 0.0
        else:
            expected = float(gold[index] in global_rankings[index][:K])
        exact_random_values.append(expected)

    split_indices: dict[str, list[int]] = defaultdict(list)
    for index, question in enumerate(questions):
        split_indices[classification_label(question)].append(index)

    t3 = {}
    for label, indices in {
        "all": list(range(len(test))),
        CITATION_TEMPLATE_ONLY: split_indices[CITATION_TEMPLATE_ONLY],
        LEXICALLY_CONTENT_BEARING: split_indices[LEXICALLY_CONTENT_BEARING],
    }.items():
        global_bm25 = subgroup_metrics(indices, global_rankings, gold)
        oracle_bm25 = subgroup_metrics(indices, oracle_act_rankings, gold)
        global_section = subgroup_metrics(indices, section_rankings, gold)
        oracle_section = subgroup_metrics(indices, oracle_act_section_rankings, gold)
        t3[label] = {
            "n": len(indices),
            "global_bm25": global_bm25,
            "oracle_act_restricted_bm25": oracle_bm25,
            "oracle_minus_global_bm25_recall@5_points": round(
                oracle_bm25["recall@5"] - global_bm25["recall@5"], 4
            ),
            "global_section_filter": global_section,
            "oracle_act_restricted_section_filter": oracle_section,
            "oracle_minus_global_section_recall@5_points": round(
                oracle_section["recall@5"] - global_section["recall@5"], 4
            ),
        }

    def boost_filter_group(label: str, predicate) -> dict:
        rows = [row for row in boost_filter_rows if predicate(row)]
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "top5_sets_differ_n": sum(row["top5_sets_differ"] for row in rows),
            "selected_passages_differ_n": sum(row["selected_passages_differ"] for row in rows),
            "mean_filter_minus_boost_f1_points": round(
                100.0 * statistics.fmean(row["filter_f1"] - row["boost_f1"] for row in rows), 4
            ),
            "label": label,
        }

    summary = {
        "status": "follow-up diagnostic; prior canonical artifacts unchanged",
        "created_at": now_iso(),
        "scope": {"questions": len(test), "passages": len(corpus)},
        "T1_reference_collision_audit": {
            "status_counts": dict(sorted(
                {status: sum(row["reference_status"] == status for row in item_rows)
                 for status in {row["reference_status"] for row in item_rows}}.items()
            )),
            "performance_group_counts": dict(sorted(
                {group: len(indices) for group, indices in t2_group_indices.items()}.items()
            )),
            "queries_with_explicit_detected_act": sum(bool(row["detected_act_from_query"]) for row in item_rows),
            "semantic_identifiability": "not assessed",
        },
        "T2_collision_performance": {
            "groups": {group: joined_group(indices) for group, indices in sorted(t2_group_indices.items())},
            "predeclared_collision_candidate_bins": {
                label: joined_group(collision_bin_indices.get(label, []))
                for label in ("2-5", "6-10", "11+")
            },
        },
        "T3_oracle_act_restriction": {
            "definition": "unchanged query and global BM25 scores; candidate pool restricted to annotated Act",
            "fallback_for_section_method": "oracle-Act BM25 when no supported section match exists inside the annotated Act",
            "results": t3,
        },
        "C1_input_audit": {
            "rows": len(input_rows),
            "file": "passage_input_audit.jsonl",
            "neural_title_inclusive_rerun": "not run in this CPU-only phase",
        },
        "C2_section_random": {
            "exact_expected_recall@5_pct": round(100.0 * statistics.fmean(exact_random_values), 4),
            "formula_scope": "single annotated Indian gold passage; canonical BM25 fallback for no section candidates",
        },
        "C3_tie_sensitivity": {
            "item_file": "tie_audit_items.jsonl",
            "bm25": randomized_boundary_recall(
                score_rows, global_rankings, gold, TIE_ITERATIONS, SEED, K
            ),
            "section_filter": randomized_boundary_recall(
                score_rows, section_rankings, gold, TIE_ITERATIONS, SEED, K
            ),
        },
        "C4_downstream_decomposition": {
            "status": "already complete in retrieval_diagnostic_controls.json",
            "source": "results/experiments/retrieval/retrieval_diagnostic_controls.json",
        },
        "C5_boost_filter": {
            "item_file": "boost_filter_trace.jsonl",
            "all": boost_filter_group("all", lambda row: True),
            "fewer_than_five_matches": boost_filter_group(
                "fewer_than_five_matches",
                lambda row: not row["fallback_used"] and row["matching_candidate_count"] < 5,
            ),
            "exactly_five_matches": boost_filter_group(
                "exactly_five_matches",
                lambda row: row["matching_candidate_count"] == 5,
            ),
            "more_than_five_matches": boost_filter_group(
                "more_than_five_matches",
                lambda row: row["matching_candidate_count"] > 5,
            ),
            "fallback": boost_filter_group("fallback", lambda row: row["fallback_used"]),
        },
        "provenance": {
            "protocol": str(PROTOCOL_FILE.relative_to(ROOT)).replace("\\", "/"),
            "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "inputs_sha256": {
                str(CORPUS_FILE.relative_to(ROOT)).replace("\\", "/"): hash_file(CORPUS_FILE),
                str(TEST_FILE.relative_to(ROOT)).replace("\\", "/"): hash_file(TEST_FILE),
                **{
                    str(path.relative_to(ROOT)).replace("\\", "/"): hash_file(path)
                    for path in prediction_files.values()
                },
            },
            "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "[A-Za-z0-9]+ lowercased"},
            "tie_break": "descending score then ascending corpus index",
            "seed": SEED,
            "python": platform.python_version(),
            "elapsed_seconds": round(time.time() - started, 3),
        },
    }

    canonical = retrieval_metrics(global_rankings, gold)
    if not math.isclose(canonical["recall@5"], 22.6744, abs_tol=0.0001):
        raise RuntimeError(f"canonical BM25 check failed: {canonical}")
    if len(split_indices[CITATION_TEMPLATE_ONLY]) != 177:
        raise RuntimeError("operational question partition no longer reproduces n=177")

    atomic_write_json(OUTPUT_DIR / "summary.json", summary)
    atomic_write_json(
        OUTPUT_DIR / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "completed_at": now_iso(),
            "summary": "summary.json",
            "outputs": [
                "summary.json",
                "indian_reference_collision_items.jsonl",
                "indian_reference_collision_items.csv",
                "passage_input_audit.jsonl",
                "tie_audit_items.jsonl",
                "boost_filter_trace.jsonl",
            ],
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
