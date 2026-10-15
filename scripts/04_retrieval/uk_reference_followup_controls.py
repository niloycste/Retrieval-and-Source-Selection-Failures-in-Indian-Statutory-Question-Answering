"""Corrected UK reference audit, subgroup analysis, and BM25 intervention.

No labels are used to construct references or candidate sets. Gold qrels are
joined only after parsing/resolution for coverage and retrieval evaluation.
The original external-validation outputs remain untouched.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from uk_statutecorpus_external_validation import (
    BM25_B,
    BM25_K1,
    BM25Index,
    QueryRecord,
    load_corpus,
    load_queries,
    load_rankings,
    per_query_metrics,
    tokenize,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = (
    ROOT
    / "results"
    / "experiments"
    / "retrieval"
    / "runs"
    / "uk_statutecorpus_external_v1"
)
OUTPUT_DIR = (
    ROOT
    / "results"
    / "experiments"
    / "retrieval"
    / "runs"
    / "uk_reference_followup_v1"
)
PROTOCOL_FILE = ROOT / "docs" / "RETRIEVAL_FOLLOWUP_CONTROLS_PROTOCOL.md"
SOURCE_MANIFEST_FILE = (
    ROOT
    / "data"
    / "external"
    / "uk_statutecorpus"
    / "extracted"
    / "LOCAL_SOURCE_MANIFEST.json"
)

SEED = 42
RANKING_DEPTH = 100
SUPPORTED_TYPES = ("section", "regulation", "article", "rule", "paragraph", "schedule")
QUERY_FRAME_TOKENS = {
    "a", "an", "and", "are", "as", "be", "by", "does", "draft", "for",
    "from", "how", "in", "instruments", "ireland", "is", "made", "northern",
    "of", "parliament", "rules", "scotland", "scottish", "statutory", "the",
    "to", "uk", "under", "wales", "what", "with", "amend", "amended",
    "amendment", "amendments", "modification", "modifications",
}
LOW_INFORMATION_MAX_RESIDUAL_TOKENS = 2
DANGLING_SYNTACTIC_FRAME_PATTERN = (
    r"\b(?:the|under|by|to|of|concerning|regarding)\s*(?:[?.,;:]|$)"
    r"|\b(?:under|by|to)\s+(?:for|of|regarding|concerning)\b"
)
LABEL_RE = re.compile(
    r"\b(?P<label>sections?|regulations?|articles?|rules?|paragraphs?|schedules?)\s+",
    re.IGNORECASE,
)
LOCATOR_RE = re.compile(r"(?<!\()\b\d+[A-Za-z]?\b")
STOP_RE = re.compile(
    r"\s+(?:of|under|regarding|concerning|for)\s+|[?;:.]",
    re.IGNORECASE,
)
INSTRUMENT_END_RE = re.compile(
    r"\b(?:Act|Regulations|Order|Rules)(?:\s*\([^)]+\))*\s+"
    r"(?:18|19|20)\d{2}\b",
    re.IGNORECASE,
)
EU_INSTRUMENT_RE = re.compile(
    r"\bRegulation\s*\((?:EU|EC|EEC)\)\s*(?:No\s*)?"
    r"(?P<number>\d{2,4}/\d{1,4})\b",
    re.IGNORECASE,
)

SYSTEM_FILES = {
    "bm25": "rankings_bm25.jsonl",
    "e5_small_v2": "rankings_e5_small_v2.jsonl",
    "rrf_bm25_e5": "rankings_rrf_bm25_e5.jsonl",
    "bm25_ce": "rankings_bm25_ce.jsonl",
    "rrf_bm25_e5_ce": "rankings_rrf_bm25_e5_ce.jsonl",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalized_words(text: str) -> str:
    return " ".join(tokenize(text))


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


def singular(label: str) -> str:
    lowered = label.lower()
    return lowered[:-1] if lowered.endswith("s") else lowered


def parse_references(query: str) -> tuple[list[dict], list[dict]]:
    """Parse provision locators while separating instrument-title years."""
    parsed: list[dict] = []
    ignored: list[dict] = []
    for label_match in LABEL_RE.finditer(query):
        tail = query[label_match.end():]
        if re.match(r"\s*\((?:EU|EC|EEC)\)", tail, re.IGNORECASE):
            # Parsed separately as an explicit instrument identifier.
            continue
        stop = STOP_RE.search(tail)
        body = tail[: stop.start()] if stop else tail
        body = body.strip(" ,")
        locators = list(LOCATOR_RE.finditer(body))
        if not locators:
            continue
        label = label_match.group("label")
        # A plural instrument name followed by a four-digit year is not a
        # provision locator (for example, "Regulations 1989").
        if (
            len(locators) == 1
            and len(locators[0].group(0)) == 4
            and 1800 <= int(locators[0].group(0)) <= 2099
            and label.lower().endswith("s")
        ):
            end = label_match.end() + locators[0].end()
            ignored.append(
                {
                    "text": query[label_match.start():end],
                    "reason": "instrument_title_year_not_provision_locator",
                    "start": label_match.start(),
                    "end": end,
                }
            )
            continue

        span_end = label_match.end() + (stop.start() if stop else len(body))
        span_text = query[label_match.start():span_end].strip(" ,")
        reference_type = singular(label)
        for locator_match in locators:
            locator = locator_match.group(0).upper()
            parsed.append(
                {
                    "parsed_reference_text": span_text,
                    "reference_type": reference_type,
                    "locator": locator,
                    "span_start": label_match.start(),
                    "span_end": span_end,
                }
            )
    # Preserve order but deduplicate repeated locator/type pairs.
    unique: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in parsed:
        key = (row["reference_type"], row["locator"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique, ignored


def extract_instruments(query: str) -> list[dict]:
    instruments: list[dict] = []
    for match in INSTRUMENT_END_RE.finditer(query):
        prefix = query[: match.start()]
        starts = [candidate.end() for candidate in re.finditer(r"\bthe\s+", prefix, re.IGNORECASE)]
        start = starts[-1] if starts else max(
            prefix.rfind("?"), prefix.rfind("."), prefix.rfind(","), prefix.rfind(";")
        ) + 1
        text = normalize_space(query[start:match.end()]).strip(" ,")
        # Avoid swallowing an entire question when no credible title boundary exists.
        if 2 <= len(tokenize(text)) <= 20:
            instruments.append(
                {
                    "parsed_reference_text": text,
                    "reference_type": "instrument_identifier",
                    "locator": normalized_words(text),
                    "normalized": normalized_words(text),
                    "span_start": start,
                    "span_end": match.end(),
                    "identifier_kind": "named_instrument_with_year",
                }
            )
    for match in EU_INSTRUMENT_RE.finditer(query):
        text = query[match.start():match.end()]
        instruments.append(
            {
                "parsed_reference_text": text,
                "reference_type": "instrument_identifier",
                "locator": match.group("number").upper(),
                "normalized": normalized_words(text),
                "span_start": match.start(),
                "span_end": match.end(),
                "identifier_kind": "eu_regulation_number",
            }
        )
    unique: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for row in sorted(instruments, key=lambda value: (value["span_start"], value["span_end"])):
        key = (row["span_start"], row["span_end"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def remove_reference_spans(query: str, references: Sequence[dict]) -> tuple[str, list[str]]:
    spans = sorted({(row["span_start"], row["span_end"]) for row in references}, reverse=True)
    output = query
    removed: list[str] = []
    for start, end in spans:
        if output[start:end] != query[start:end]:
            raise ValueError("overlapping or unstable query-reference spans")
        removed.append(query[start:end])
        output = output[:start] + output[end:]
    output = re.sub(r"\s+", " ", output)
    output = re.sub(r"\s+([,?.;:])", r"\1", output)
    output = re.sub(r",\s*(?=[?.,;:]|$)", "", output)
    return output.strip(), list(reversed(removed))


def removal_quality_flags(query: str) -> dict:
    tokens = tokenize(query)
    residual = [token for token in tokens if token not in QUERY_FRAME_TOKENS]
    dangling = bool(
        re.search(DANGLING_SYNTACTIC_FRAME_PATTERN, query, re.IGNORECASE)
    )
    return {
        "reference_removed_token_count": len(tokens),
        "reference_removed_residual_content_tokens": residual,
        "low_information_after_removal": (
            len(residual) <= LOW_INFORMATION_MAX_RESIDUAL_TOKENS
        ),
        "syntactic_artifact_after_removal": dangling,
    }


def reference_pattern(reference_type: str, locator: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b{re.escape(reference_type)}s?\s+{re.escape(locator)}(?=\b|\s*\()",
        re.IGNORECASE,
    )


def source_url_pattern(reference_type: str, locator: str) -> re.Pattern[str]:
    paths = {reference_type}
    # Northern Ireland Rules are commonly exposed through /article/ URLs.
    if reference_type == "rule":
        paths.add("article")
    alternatives = "|".join(sorted(re.escape(value) for value in paths))
    return re.compile(rf"/(?:{alternatives})/{re.escape(locator)}(?:/|$)", re.IGNORECASE)


def mean_metric_summary(queries: Sequence[QueryRecord], rankings: dict[str, list[str]]) -> dict:
    if not queries:
        return {"n": 0}
    _, metrics = per_query_metrics(queries, rankings)
    wanted = ("recall@5", "recall@10", "mrr@10", "ndcg@10", "grade3_recall@5", "grade3_recall@10")
    return {
        "n": len(queries),
        **{
            f"{metric}_pct": round(100.0 * statistics.fmean(metrics[metric]), 4)
            for metric in wanted
        },
    }


def per_item_metrics(query: QueryRecord, ranking: list[str]) -> dict:
    _, values = per_query_metrics([query], {query.qid: ranking})
    return {key: round(100.0 * metric[0], 4) for key, metric in values.items()}


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(SOURCE_MANIFEST_FILE.read_text(encoding="utf-8"))
    corpus_file = Path(source_manifest["corpus_file"])
    query_file = Path(source_manifest["query_file"])
    queries = load_queries(query_file)
    required_ids = {doc_id for query in queries for doc_id in query.qrels}
    corpus = load_corpus(corpus_file, required_ids)
    corpus_ids = set(corpus.ids)

    provision_references_by_qid: dict[str, list[dict]] = {}
    all_references_by_qid: dict[str, list[dict]] = {}
    ignored_by_qid: dict[str, list[dict]] = {}
    variants: list[dict] = []
    for query in queries:
        provisions, ignored = parse_references(query.query)
        instruments = extract_instruments(query.query)
        all_references = provisions + instruments
        provision_references_by_qid[query.qid] = provisions
        all_references_by_qid[query.qid] = all_references
        ignored_by_qid[query.qid] = ignored
        removed_query, removed_spans = remove_reference_spans(query.query, all_references)
        quality_flags = removal_quality_flags(removed_query)
        if not all_references and removed_query != query.query:
            raise RuntimeError(f"citation-free query changed: {query.qid}")
        for row in all_references:
            if query.query[row["span_start"]:row["span_end"]].strip(" ,") != row["parsed_reference_text"]:
                raise RuntimeError(f"parsed span mismatch: {query.qid}")
        variants.append(
            {
                "question_id": query.qid,
                "original_query": query.query,
                "removed_reference_spans": removed_spans,
                "reference_removed_query": removed_query,
                **quality_flags,
            }
        )
    write_jsonl(OUTPUT_DIR / "uk_reference_removed_queries.jsonl", variants)

    corpus_search_texts = []
    corpus_metadata_texts = []
    corpus_normalized_texts = []
    file_name_normalized = []
    source_urls = []
    doc_ids = []
    for text, metadata in zip(corpus.texts, corpus.metadata):
        title = str(metadata.get("chunk_title", ""))
        file_name = str(metadata.get("file_name", ""))
        search_text = normalize_space(f"{file_name} {title} {text}")
        corpus_search_texts.append(search_text)
        corpus_metadata_texts.append(normalize_space(f"{file_name} {title}"))
        corpus_normalized_texts.append(normalized_words(search_text))
        file_name_normalized.append(normalized_words(file_name.removesuffix(".txt")))
        source_urls.append(str(metadata.get("source_url", "")))
        doc_ids.append(str(metadata.get("doc_id", "")))

    audit_rows: list[dict] = []
    final_candidates_by_qid: dict[str, list[int]] = {}
    source_candidates_by_qid: dict[str, list[int]] = {}
    for query in queries:
        provisions = provision_references_by_qid[query.qid]
        all_references = all_references_by_qid[query.qid]
        instruments = [
            row for row in all_references if row["reference_type"] == "instrument_identifier"
        ]
        text_candidates: set[int] = set()
        metadata_candidates: set[int] = set()
        source_candidates: set[int] = set()
        instrument_candidates: set[int] = set()
        instrument_source_candidates: set[int] = set()
        per_reference_counts: list[dict] = []
        for reference in provisions:
            text_pattern = reference_pattern(reference["reference_type"], reference["locator"])
            url_pattern = source_url_pattern(reference["reference_type"], reference["locator"])
            current_text = {index for index, text in enumerate(corpus_search_texts) if text_pattern.search(text)}
            current_metadata = {index for index, text in enumerate(corpus_metadata_texts) if text_pattern.search(text)}
            current_source = {index for index, url in enumerate(source_urls) if url_pattern.search(url)}
            text_candidates.update(current_text)
            metadata_candidates.update(current_metadata)
            source_candidates.update(current_source)
            per_reference_counts.append(
                {
                    "reference_type": reference["reference_type"],
                    "locator": reference["locator"],
                    "text_candidate_count": len(current_text),
                    "metadata_text_candidate_count": len(current_metadata),
                    "source_url_candidate_count": len(current_source),
                }
            )

        for instrument in instruments:
            phrase = instrument["normalized"]
            current_text = {
                index for index, text in enumerate(corpus_normalized_texts) if phrase in text
            }
            current_source = {
                index for index, text in enumerate(file_name_normalized)
                if phrase == text or phrase in text or text in phrase
            }
            instrument_candidates.update(current_text)
            instrument_source_candidates.update(current_source)
            per_reference_counts.append(
                {
                    "reference_type": "instrument_identifier",
                    "locator": instrument["locator"],
                    "identifier_kind": instrument["identifier_kind"],
                    "text_candidate_count": len(current_text),
                    "metadata_text_candidate_count": len(current_source),
                    "source_url_candidate_count": 0,
                }
            )

        if provisions and instruments:
            constrained = text_candidates & instrument_candidates
        elif provisions:
            constrained = set(text_candidates)
        else:
            constrained = set(instrument_candidates)

        candidates = sorted(constrained)
        final_candidates_by_qid[query.qid] = candidates
        source_candidates_by_qid[query.qid] = sorted(source_candidates)
        relevant = set(query.qrels)
        candidate_ids = {corpus.ids[index] for index in candidates}
        source_candidate_ids = {corpus.ids[index] for index in source_candidates}
        if not all_references:
            status = "no_recognised_reference"
        elif not candidates:
            status = "parsed_unresolved"
        elif len(all_references) > 1:
            status = "multiple_references_resolved"
        else:
            status = "parsed_resolved"
        resolved_ids_exist = all(corpus.ids[index] in corpus_ids for index in candidates)
        audit_rows.append(
            {
                "question_id": query.qid,
                "query": query.query,
                "parsed_references": all_references,
                "parsed_provision_references": provisions,
                "ignored_reference_like_spans": ignored_by_qid[query.qid],
                "reference_type": sorted({row["reference_type"] for row in all_references}),
                "explicit_instrument_identifier": [row["parsed_reference_text"] for row in instruments],
                "matching_candidate_count": len(candidates),
                "matching_instrument_count": len({doc_ids[index] for index in candidates}),
                "metadata_text_matching_candidate_count": len(metadata_candidates),
                "source_url_matching_candidate_count": len(source_candidates),
                "instrument_text_matching_candidate_count": len(instrument_candidates),
                "instrument_source_document_candidate_count": len(instrument_source_candidates),
                "relevant_passages_in_matching_set": sorted(relevant & candidate_ids),
                "relevant_passages_in_source_url_set": sorted(relevant & source_candidate_ids),
                "parser_status": status,
                "per_reference_counts": per_reference_counts,
                "automatic_checks": {
                    "parsed_spans_in_original_query": all(
                        row["parsed_reference_text"] in query.query for row in all_references
                    ),
                    "resolved_passage_ids_exist": resolved_ids_exist,
                    "instrument_constraints_from_query_text": all(
                        row["parsed_reference_text"] in query.query for row in instruments
                    ),
                    "citation_free_query_unchanged": (
                        variants[len(audit_rows)]["reference_removed_query"] == query.query
                        if not all_references else None
                    ),
                },
            }
        )
    write_jsonl(OUTPUT_DIR / "uk_reference_audit_items.jsonl", audit_rows)

    rankings = {
        name: load_rankings(CANONICAL_DIR / "cache" / file_name)
        for name, file_name in SYSTEM_FILES.items()
    }
    groups: dict[str, list[QueryRecord]] = defaultdict(list)
    for query, audit in zip(queries, audit_rows):
        if audit["parser_status"] == "no_recognised_reference":
            group = "no_recognised_reference"
        elif audit["parser_status"] == "parsed_unresolved":
            group = "parsed_unresolved_reference"
        else:
            group = "supported_resolved_reference"
        groups[group].append(query)

    subgroup_results = {}
    for group in (
        "supported_resolved_reference",
        "parsed_unresolved_reference",
        "no_recognised_reference",
    ):
        current = groups.get(group, [])
        subgroup_results[group] = {
            name: mean_metric_summary(current, system_rankings)
            for name, system_rankings in rankings.items()
        }

    eligible_queries = [query for query in queries if all_references_by_qid[query.qid]]
    variant_by_qid = {row["question_id"]: row for row in variants}
    removed_records = [
        QueryRecord(
            qid=query.qid,
            query=variant_by_qid[query.qid]["reference_removed_query"],
            qrels=query.qrels,
        )
        for query in eligible_queries
    ]

    bm25_started = time.time()
    index = BM25Index(corpus.texts, k1=BM25_K1, b=BM25_B)
    removed_rankings: dict[str, list[str]] = {}
    removed_ranking_rows = []
    for query in removed_records:
        indices, scores = index.rank(query.query, RANKING_DEPTH)
        ranking = [corpus.ids[index] for index in indices]
        removed_rankings[query.qid] = ranking
        removed_ranking_rows.append({"qid": query.qid, "ranking": ranking, "scores": scores})
    write_jsonl(OUTPUT_DIR / "rankings_bm25_reference_removed.jsonl", removed_ranking_rows)

    original_bm25 = rankings["bm25"]
    original_summary = mean_metric_summary(eligible_queries, original_bm25)
    removed_summary = mean_metric_summary(removed_records, removed_rankings)
    item_intervention = []
    for original, removed in zip(eligible_queries, removed_records):
        original_metrics = per_item_metrics(original, original_bm25[original.qid])
        removed_metrics = per_item_metrics(removed, removed_rankings[removed.qid])
        item_intervention.append(
            {
                "question_id": original.qid,
                "original_query": original.query,
                "reference_removed_query": removed.query,
                "removed_spans": variant_by_qid[original.qid]["removed_reference_spans"],
                "reference_removed_token_count": variant_by_qid[original.qid]["reference_removed_token_count"],
                "low_information_after_removal": variant_by_qid[original.qid]["low_information_after_removal"],
                "syntactic_artifact_after_removal": variant_by_qid[original.qid]["syntactic_artifact_after_removal"],
                "reference_removed_residual_content_tokens": variant_by_qid[original.qid]["reference_removed_residual_content_tokens"],
                "original_metrics_pct": original_metrics,
                "reference_removed_metrics_pct": removed_metrics,
                "recall@5_change_points": round(
                    removed_metrics["recall@5"] - original_metrics["recall@5"], 4
                ),
                "ndcg@10_change_points": round(
                    removed_metrics["ndcg@10"] - original_metrics["ndcg@10"], 4
                ),
            }
        )
    write_jsonl(OUTPUT_DIR / "uk_reference_removal_items.jsonl", item_intervention)

    parsed_count = sum(bool(all_references_by_qid[query.qid]) for query in queries)
    provision_query_count = sum(
        bool(provision_references_by_qid[query.qid]) for query in queries
    )
    instrument_query_count = sum(
        any(
            row["reference_type"] == "instrument_identifier"
            for row in all_references_by_qid[query.qid]
        )
        for query in queries
    )
    resolved_count = len(groups.get("supported_resolved_reference", []))
    source_relevant_coverage = sum(
        bool(row["relevant_passages_in_source_url_set"])
        for row in audit_rows
        if row["parsed_provision_references"]
    )
    old_reference_audit = json.loads(
        (CANONICAL_DIR / "reference_audit.json").read_text(encoding="utf-8")
    )
    old_flagged = {
        row["qid"] for row in old_reference_audit["rows"] if row["has_explicit_locator"]
    }
    old_title_year_false_locators = {
        row["question_id"]
        for row in audit_rows
        if row["question_id"] in old_flagged and row["ignored_reference_like_spans"]
    }
    g4_decision = {
        "run": False,
        "reason": (
            "Only five natural provision-locator queries are available, and source-URL provision "
            "matching does not reliably represent references to provisions cited inside the target instrument."
        ),
        "parsed_query_count": parsed_count,
        "provision_locator_query_count": provision_query_count,
        "text_resolved_query_count": resolved_count,
        "queries_with_relevant_passage_in_source_url_match": source_relevant_coverage,
        "note": "No UK structural-retrieval result is manufactured from content-reference matching.",
    }
    low_information_ids = [
        row["question_id"]
        for row in item_intervention
        if row["low_information_after_removal"]
    ]
    syntactic_artifact_ids = [
        row["question_id"]
        for row in item_intervention
        if row["syntactic_artifact_after_removal"]
    ]
    quality_flag_union_ids = sorted(
        set(low_information_ids) | set(syntactic_artifact_ids)
    )

    summary = {
        "status": "complete exploratory follow-up; canonical UK run unchanged",
        "created_at": now_iso(),
        "G1_reference_coverage": {
            "queries": len(queries),
            "parser_status_counts": {
                status: sum(row["parser_status"] == status for row in audit_rows)
                for status in sorted({row["parser_status"] for row in audit_rows})
            },
            "queries_with_title_year_reclassified_from_provision_to_instrument": sum(
                bool(row["ignored_reference_like_spans"]) for row in audit_rows
            ),
            "comparison_with_old_regex_audit": {
                "old_regex_flagged_queries": len(old_flagged),
                "old_regex_title_year_false_locator_matches": len(old_title_year_false_locators),
                "note": "These are instrument names/years, not provision locators; they remain represented as instrument identifiers where supported.",
            },
            "parsed_query_count": parsed_count,
            "provision_locator_query_count": provision_query_count,
            "instrument_identifier_query_count": instrument_query_count,
            "queries_with_relevant_passage_in_text_resolved_set": sum(
                bool(row["relevant_passages_in_matching_set"]) for row in audit_rows
            ),
            "parser_accuracy": "not established; no human or gold reference annotation used",
            "item_file": "uk_reference_audit_items.jsonl",
        },
        "G2_existing_ranking_subgroups": subgroup_results,
        "G3_bm25_reference_removal": {
            "eligible_query_count": len(eligible_queries),
            "low_information_after_removal_count": len(low_information_ids),
            "syntactic_artifact_after_removal_count": len(syntactic_artifact_ids),
            "quality_flag_union_count": len(quality_flag_union_ids),
            "flagged_query_ids": quality_flag_union_ids,
            "automatic_quality_audit": {
                "human_assessment_used": False,
                "input_field": "reference_removed_query",
                "tokenizer": "[A-Za-z0-9]+ lowercased",
                "low_information_rule": {
                    "condition": "residual non-frame token count <= threshold",
                    "threshold": LOW_INFORMATION_MAX_RESIDUAL_TOKENS,
                    "query_frame_tokens": sorted(QUERY_FRAME_TOKENS),
                    "flagged_query_ids": low_information_ids,
                },
                "dangling_syntactic_frame_rule": {
                    "regex": DANGLING_SYNTACTIC_FRAME_PATTERN,
                    "regex_flags": ["IGNORECASE"],
                    "flagged_query_ids": syntactic_artifact_ids,
                },
                "union_query_ids": quality_flag_union_ids,
                "per_query_flag_file": "uk_reference_removal_items.jsonl",
                "per_query_flag_fields": [
                    "reference_removed_residual_content_tokens",
                    "low_information_after_removal",
                    "syntactic_artifact_after_removal",
                ],
            },
            "original": original_summary,
            "reference_removed": removed_summary,
            "change_points": {
                key: round(removed_summary[key] - original_summary[key], 4)
                for key in original_summary
                if key != "n"
            },
            "item_file": "uk_reference_removal_items.jsonl",
            "ranking_file": "rankings_bm25_reference_removed.jsonl",
            "bm25_elapsed_seconds": round(time.time() - bm25_started, 3),
            "interpretation": (
                "Sensitivity to removal relative to original qrels; semantic equivalence of the modified query is not assumed."
            ),
        },
        "G4_reference_aware_retrieval_decision": g4_decision,
        "provenance": {
            "protocol": str(PROTOCOL_FILE.relative_to(ROOT)).replace("\\", "/"),
            "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
            "inputs_sha256": {
                str(corpus_file): hash_file(corpus_file),
                str(query_file): hash_file(query_file),
                **{
                    str((CANONICAL_DIR / "cache" / file_name).relative_to(ROOT)).replace("\\", "/"): hash_file(
                        CANONICAL_DIR / "cache" / file_name
                    )
                    for file_name in SYSTEM_FILES.values()
                },
            },
            "bm25": {"k1": BM25_K1, "b": BM25_B, "tokenizer": "[A-Za-z0-9]+ lowercased"},
            "ranking_depth": RANKING_DEPTH,
            "seed": SEED,
            "python": platform.python_version(),
            "elapsed_seconds": round(time.time() - started, 3),
        },
    }

    # Frozen-run checks: query IDs and cached rankings must cover Eval100.
    expected_qids = {query.qid for query in queries}
    if len(queries) != 100 or any(set(system) != expected_qids for system in rankings.values()):
        raise RuntimeError("canonical UK Eval100 IDs or rankings changed")
    if any(not check for row in audit_rows for check in row["automatic_checks"].values() if check is not None):
        raise RuntimeError("one or more automatic UK audit checks failed")
    if len(old_flagged) != 13 or len(old_title_year_false_locators) != 8:
        raise RuntimeError(
            "comparison with the frozen 13-query regex audit no longer reproduces "
            "eight title-year false locator matches"
        )
    if (
        len(low_information_ids),
        len(syntactic_artifact_ids),
        len(quality_flag_union_ids),
    ) != (11, 12, 16):
        raise RuntimeError(
            "automatic UK removal-quality audit no longer reproduces "
            "the reported 11/12/16 flag counts"
        )

    atomic_write_json(OUTPUT_DIR / "summary.json", summary)
    atomic_write_json(
        OUTPUT_DIR / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "completed_at": now_iso(),
            "summary": "summary.json",
            "outputs": [
                "summary.json",
                "uk_reference_audit_items.jsonl",
                "uk_reference_removed_queries.jsonl",
                "uk_reference_removal_items.jsonl",
                "rankings_bm25_reference_removed.jsonl",
            ],
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
