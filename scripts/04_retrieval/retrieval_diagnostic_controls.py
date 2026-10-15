"""Lightweight controls and audits for the retrieval-reference manuscript.

This script does not train or run a neural model.  It recomputes BM25 variants
from the released paragraph corpus and summarizes the saved E2' predictions.
Its purpose is to make metadata access, candidate reduction, source clustering,
and retrieval/selection/extraction failure stages explicit.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILE = ROOT / "data" / "processed" / "indic_ipr_qa" / "unique_contexts.jsonl"
TEST_FILE = ROOT / "data" / "splits" / "test.jsonl"
PREDICTION_DIR = (
    ROOT / "results" / "experiments" / "retrieval" / "runs" / "citation_aware_v1"
)
OUTPUT_FILE = (
    ROOT / "results" / "experiments" / "retrieval" / "retrieval_diagnostic_controls.json"
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SECTION_RE = re.compile(r"\bsections?\s*[.\-]?\s*(\d+[A-Za-z]{0,2})\b", re.I)
K_VALUES = (1, 3, 5, 10)
BOOST = 1000.0
BOOTSTRAP_ITERATIONS = 10_000
RANDOM_ORDER_ITERATIONS = 10_000
SEED = 42

ACT_NAMES = {
    "copyright_act": "Copyright Act",
    "design_act": "Designs Act",
    "gi_act": "Geographical Indications of Goods Act",
    "patent_act": "Patents Act",
    "plant_varieties_farmers_rights_act": (
        "Protection of Plant Varieties and Farmers' Rights Act"
    ),
    "semiconductor_layout_design_act": (
        "Semiconductor Integrated Circuits Layout-Design Act"
    ),
    "trademark_act": "Trade Marks Act",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def parse_section(text: str) -> str | None:
    match = SECTION_RE.search(text or "")
    if match is None:
        return None
    return re.sub(r"\s+", "", match.group(1)).upper()


class BM25:
    """Okapi BM25 matching the canonical retrieval-only implementation."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.term_frequencies = [Counter(document) for document in documents]
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        total = len(documents)
        self.idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        output = [0.0] * len(self.documents)
        for index, frequencies in enumerate(self.term_frequencies):
            length_normalizer = self.k1 * (
                1.0 - self.b + self.b * self.lengths[index] / self.average_length
            )
            score = 0.0
            for term in query:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1.0) / (frequency + length_normalizer)
                    )
            output[index] = score
        return output


def rank(scores: list[float]) -> list[int]:
    # Python's stable ordering means corpus index is the explicit final tie-break.
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def retrieval_metrics(rankings: list[list[int]], gold: list[int]) -> dict:
    output = {}
    for k in K_VALUES:
        output[f"recall@{k}"] = round(
            100.0 * sum(target in ranking[:k] for ranking, target in zip(rankings, gold))
            / len(gold),
            4,
        )
    reciprocal_ranks = []
    for ranking, target in zip(rankings, gold):
        head = ranking[:10]
        reciprocal_ranks.append(1.0 / (head.index(target) + 1) if target in head else 0.0)
    output["mrr@10"] = round(100.0 * statistics.fmean(reciprocal_ranks), 4)
    return output


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap(
    rows_a: list[dict],
    rows_b: list[dict],
    group_by_id: dict[str, str],
    metric: str,
) -> dict:
    """Paired one-stage bootstrap over section-level source clusters.

    Each replicate samples clusters with replacement, pools every question from
    the sampled clusters, and takes the mean item-level difference. It does not
    compute an unweighted mean of cluster means.
    """
    by_group: dict[str, list[float]] = defaultdict(list)
    for row_a, row_b in zip(rows_a, rows_b):
        if row_a["id"] != row_b["id"]:
            raise ValueError("prediction files are not aligned")
        group = group_by_id[row_a["id"]]
        if metric == "f1":
            difference = float(row_b["f1"]) - float(row_a["f1"])
        elif metric == "retrieval_hit_at_k":
            difference = float(bool(row_b[metric])) - float(bool(row_a[metric]))
        else:
            raise ValueError(metric)
        by_group[group].append(difference)

    groups = sorted(by_group)
    rng = random.Random(SEED)
    bootstrap_means = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = [rng.choice(groups) for _ in groups]
        values = [value for group in sampled for value in by_group[group]]
        bootstrap_means.append(100.0 * statistics.fmean(values))

    observed = [value for group in groups for value in by_group[group]]
    return {
        "difference_points": round(100.0 * statistics.fmean(observed), 4),
        "cluster_bootstrap_95_ci": [
            round(percentile(bootstrap_means, 0.025), 4),
            round(percentile(bootstrap_means, 0.975), 4),
        ],
        "confidence_level_pct": 95,
        "resampling_unit": "Act-section source cluster",
        "replicate_aggregation": (
            "pool all questions from sampled clusters, then average item-level "
            "differences; cluster means are not averaged equally"
        ),
        "clusters": len(groups),
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": SEED,
    }


def stage_summary(rows: list[dict]) -> dict:
    categories: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row["retrieval_hit_at_k"]:
            category = "gold_absent_from_top5"
        elif not row["selected_context_is_gold"]:
            category = "gold_available_reader_selected_other"
        elif row["em"]:
            category = "gold_selected_normalized_em"
        else:
            category = "gold_selected_non_em_extraction"
        categories[category].append(row)

    available = sum(bool(row["retrieval_hit_at_k"]) for row in rows)
    selected = sum(bool(row["selected_context_is_gold"]) for row in rows)
    grounded_f1 = statistics.fmean(
        float(row["f1"]) if row["selected_context_is_gold"] else 0.0 for row in rows
    )
    grounded_em = statistics.fmean(
        float(row["em"]) if row["selected_context_is_gold"] else 0.0 for row in rows
    )
    return {
        "n": len(rows),
        "gold_available_n": available,
        "gold_available_pct": round(100.0 * available / len(rows), 4),
        "selected_annotated_source_n": selected,
        "selected_annotated_source_pct": round(100.0 * selected / len(rows), 4),
        "conditional_source_selection_pct_given_available": round(
            100.0 * selected / available if available else 0.0, 4
        ),
        "annotation_grounded_em_pct": round(100.0 * grounded_em, 4),
        "annotation_grounded_f1_pct": round(100.0 * grounded_f1, 4),
        "failure_stage_counts": {
            category: len(categories.get(category, []))
            for category in (
                "gold_absent_from_top5",
                "gold_available_reader_selected_other",
                "gold_selected_non_em_extraction",
                "gold_selected_normalized_em",
            )
        },
        "mean_f1_pct_by_stage": {
            category: round(
                100.0 * statistics.fmean(float(row["f1"]) for row in stage_rows), 4
            )
            for category, stage_rows in categories.items()
        },
    }


def main() -> None:
    corpus = load_jsonl(CORPUS_FILE)
    test = load_jsonl(TEST_FILE)
    paragraphs = [normalize(row["context"]) for row in corpus]
    titles = [normalize(row.get("title", "")) for row in corpus]
    acts = [row.get("act", "") for row in corpus]
    sections = [parse_section(title) for title in titles]
    questions = [row["question"] for row in test]
    question_sections = [parse_section(question) for question in questions]

    by_context: dict[str, int] = {}
    for index, paragraph in enumerate(paragraphs):
        by_context.setdefault(paragraph, index)
    gold = [by_context[normalize(row["context"])] for row in test]

    document_inputs = {
        "paragraph_only": paragraphs,
        "title_plus_paragraph": [f"{title} {paragraph}" for title, paragraph in zip(titles, paragraphs)],
        "act_title_plus_paragraph": [
            f"{ACT_NAMES.get(act, act)} {title} {paragraph}"
            for act, title, paragraph in zip(acts, titles, paragraphs)
        ],
    }

    bm25_models = {
        name: BM25([tokenize(document) for document in documents])
        for name, documents in document_inputs.items()
    }
    score_sets = {
        name: [model.scores(tokenize(question)) for question in questions]
        for name, model in bm25_models.items()
    }
    rankings = {
        name: [rank(scores) for scores in per_question]
        for name, per_question in score_sets.items()
    }

    act_supplied_queries = [
        f"{ACT_NAMES.get(row['act'], row['act'])} {row['question']}" for row in test
    ]
    act_supplied_scores = [
        bm25_models["act_title_plus_paragraph"].scores(tokenize(question))
        for question in act_supplied_queries
    ]
    rankings["oracle_act_name_plus_act_title_paragraph"] = [
        rank(scores) for scores in act_supplied_scores
    ]

    by_section: dict[str, list[int]] = defaultdict(list)
    for index, section in enumerate(sections):
        if section is not None:
            by_section[section].append(index)

    section_filter_bm25 = []
    section_filter_title_bm25 = []
    oracle_act_section = []
    section_boost_bm25 = []
    matching_candidate_counts = []
    matching_act_counts = []
    top5_candidate_set_differences = 0
    top5_jaccards = []
    bm25_cutoff_ties = 0
    section_filter_cutoff_ties = 0
    gold_cutoff_ties = 0

    for index, section in enumerate(question_sections):
        base_scores = score_sets["paragraph_only"][index]
        title_scores = score_sets["title_plus_paragraph"][index]
        base_ranking = rankings["paragraph_only"][index]
        candidates = list(by_section.get(section, [])) if section else []

        if len(base_ranking) > 5 and math.isclose(
            base_scores[base_ranking[4]], base_scores[base_ranking[5]], abs_tol=1e-12
        ):
            bm25_cutoff_ties += 1
            cutoff_score = base_scores[base_ranking[4]]
            if math.isclose(base_scores[gold[index]], cutoff_score, abs_tol=1e-12):
                gold_cutoff_ties += 1

        if candidates:
            matching_candidate_counts.append(len(candidates))
            matching_act_counts.append(len({acts[candidate] for candidate in candidates}))
            filtered_base = sorted(candidates, key=lambda item: (-base_scores[item], item))
            filtered_title = sorted(candidates, key=lambda item: (-title_scores[item], item))
            same_act = [candidate for candidate in candidates if acts[candidate] == test[index]["act"]]
            filtered_oracle = sorted(same_act, key=lambda item: (-title_scores[item], item))
            if len(filtered_base) > 5 and math.isclose(
                base_scores[filtered_base[4]], base_scores[filtered_base[5]], abs_tol=1e-12
            ):
                section_filter_cutoff_ties += 1
        else:
            filtered_base = base_ranking
            filtered_title = rankings["title_plus_paragraph"][index]
            filtered_oracle = rankings["oracle_act_name_plus_act_title_paragraph"][index]

        section_filter_bm25.append(filtered_base)
        section_filter_title_bm25.append(filtered_title)
        oracle_act_section.append(filtered_oracle)

        boosted_scores = list(base_scores)
        for candidate in candidates:
            boosted_scores[candidate] += BOOST
        boosted = rank(boosted_scores)
        section_boost_bm25.append(boosted)
        boost_set = set(boosted[:5])
        filter_set = set(filtered_base[:5])
        if boost_set != filter_set:
            top5_candidate_set_differences += 1
        union = boost_set | filter_set
        top5_jaccards.append(len(boost_set & filter_set) / len(union) if union else 1.0)

    rankings["section_filter_plus_bm25"] = section_filter_bm25
    rankings["section_filter_plus_title_paragraph_bm25"] = section_filter_title_bm25
    rankings["oracle_act_plus_section_filter"] = oracle_act_section
    rankings["section_boost_plus_bm25"] = section_boost_bm25

    random_order_recall5 = []
    rng = random.Random(SEED)
    for _ in range(RANDOM_ORDER_ITERATIONS):
        hits = 0
        for index, section in enumerate(question_sections):
            candidates = list(by_section.get(section, [])) if section else []
            if candidates:
                rng.shuffle(candidates)
                current = candidates
            else:
                current = rankings["paragraph_only"][index]
            hits += gold[index] in current[:5]
        random_order_recall5.append(100.0 * hits / len(test))

    any_act_name_string = 0
    target_act_name_string = 0
    for row in test:
        lowered = row["question"].lower()
        if any(name.lower() in lowered for name in ACT_NAMES.values()):
            any_act_name_string += 1
        target_name = ACT_NAMES.get(row["act"], row["act"]).lower()
        if target_name in lowered:
            target_act_name_string += 1

    group_counts = Counter(row["group_id"] for row in test)
    group_sizes = sorted(group_counts.values())
    parsed_queries = sum(section is not None for section in question_sections)

    prediction_files = {
        "BM25": PREDICTION_DIR / "bm25_top5_e2_predictions.jsonl",
        "section_boost": PREDICTION_DIR / "bm25_section_boost_top5_e2_predictions.jsonl",
        "section_filter": PREDICTION_DIR / "bm25_section_filter_top5_e2_predictions.jsonl",
    }
    predictions = {name: load_jsonl(path) for name, path in prediction_files.items()}
    group_by_id = {row["id"]: row["group_id"] for row in test}

    payload = {
        "status": "canonical lightweight diagnostic",
        "scope": {
            "questions": len(test),
            "passages": len(corpus),
            "distinct_source_sections": len(group_counts),
            "questions_per_source_section": {
                "min": min(group_sizes),
                "median": statistics.median(group_sizes),
                "mean": round(statistics.fmean(group_sizes), 4),
                "max": max(group_sizes),
            },
        },
        "input_specification": {
            "canonical_bm25": "question against paragraph text only",
            "canonical_dense": "question against paragraph text only (recorded from dense_retrieval_experiment.py)",
            "canonical_cross_encoder": "question--paragraph pairs over paragraph-only first-stage candidates",
            "section_index": "section parsed from question and passage title; BM25 ordering uses paragraph text only",
            "reader": "question--paragraph input; title and Act metadata are not supplied",
            "metadata_fairness_controls": [
                "BM25 over title + paragraph",
                "oracle gold-Act name in query against Act name + title + paragraph",
            ],
        },
        "retrieval_controls": {
            name: retrieval_metrics(system_rankings, gold)
            for name, system_rankings in rankings.items()
        },
        "random_section_candidate_order": {
            "recall@5_mean": round(statistics.fmean(random_order_recall5), 4),
            "recall@5_sd_across_orderings": round(statistics.stdev(random_order_recall5), 4),
            "recall@5_95_simulation_interval": [
                round(percentile(random_order_recall5, 0.025), 4),
                round(percentile(random_order_recall5, 0.975), 4),
            ],
            "iterations": RANDOM_ORDER_ITERATIONS,
            "seed": SEED,
            "fallback": "canonical paragraph-only BM25 when no section is parsed",
        },
        "structural_identifiability": {
            "queries_with_parsed_section": parsed_queries,
            "queries_with_any_act_name_string": any_act_name_string,
            "queries_with_target_act_name_string": target_act_name_string,
            "parsed_queries_whose_section_occurs_in_one_act": sum(
                count == 1 for count in matching_act_counts
            ),
            "parsed_queries_whose_section_occurs_in_multiple_acts": sum(
                count > 1 for count in matching_act_counts
            ),
            "semantic_source_identifiability": (
                "not human-annotated; structural counts do not establish whether wording lets a reader infer the Act"
            ),
        },
        "candidate_reduction": {
            "matching_passages": {
                "min": min(matching_candidate_counts),
                "median": statistics.median(matching_candidate_counts),
                "mean": round(statistics.fmean(matching_candidate_counts), 4),
                "max": max(matching_candidate_counts),
                "queries_with_five_or_fewer": sum(
                    count <= 5 for count in matching_candidate_counts
                ),
                "queries_with_more_than_five": sum(
                    count > 5 for count in matching_candidate_counts
                ),
            },
            "distinct_acts": {
                "median": statistics.median(matching_act_counts),
                "mean": round(statistics.fmean(matching_act_counts), 4),
                "max": max(matching_act_counts),
            },
        },
        "ordering_and_ties": {
            "tie_break": "ascending corpus index after descending score",
            "paragraph_bm25_queries_tied_at_rank5_boundary": bm25_cutoff_ties,
            "paragraph_bm25_gold_tied_at_rank5_boundary": gold_cutoff_ties,
            "section_filtered_queries_tied_at_rank5_boundary": section_filter_cutoff_ties,
            "boost_vs_filter_top5_candidate_sets_differ_n": top5_candidate_set_differences,
            "boost_vs_filter_top5_mean_jaccard": round(statistics.fmean(top5_jaccards), 4),
        },
        "reader_stage_decomposition": {
            name: stage_summary(rows) for name, rows in predictions.items()
        },
        "cluster_bootstrap": {
            "method": "paired one-stage bootstrap over 75 Act-section source clusters",
            "section_filter_minus_bm25_f1": cluster_bootstrap(
                predictions["BM25"], predictions["section_filter"], group_by_id, "f1"
            ),
            "section_filter_minus_bm25_recall@5": cluster_bootstrap(
                predictions["BM25"],
                predictions["section_filter"],
                group_by_id,
                "retrieval_hit_at_k",
            ),
            "section_boost_minus_bm25_f1": cluster_bootstrap(
                predictions["BM25"], predictions["section_boost"], group_by_id, "f1"
            ),
        },
        "provenance": {
            "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "corpus": str(CORPUS_FILE.relative_to(ROOT)).replace("\\", "/"),
            "test": str(TEST_FILE.relative_to(ROOT)).replace("\\", "/"),
            "prediction_files": {
                name: str(path.relative_to(ROOT)).replace("\\", "/")
                for name, path in prediction_files.items()
            },
            "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "[A-Za-z0-9]+ lowercased"},
        },
    }

    canonical = payload["retrieval_controls"]["paragraph_only"]
    expected = {"recall@1": 13.0814, "recall@3": 20.0581, "recall@5": 22.6744}
    for key, value in expected.items():
        if not math.isclose(canonical[key], value, abs_tol=0.0001):
            raise RuntimeError(f"canonical BM25 reproduction failed for {key}: {canonical[key]}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
