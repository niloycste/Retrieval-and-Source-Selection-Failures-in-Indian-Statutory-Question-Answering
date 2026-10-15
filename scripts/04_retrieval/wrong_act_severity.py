"""Structural source profile for retrieval misses: cross-Act versus within-Act.

Recall@k treats every miss alike. This audit considers questions for which the
annotated gold paragraph is absent from the top five and compares the Act label
of the top-ranked passage with the Act label of the gold source. It uses only
versioned corpus metadata; it does not assess legal materiality or substitute
for practitioner evaluation.

The report covers the retrieval-only BM25 baseline and the BM25 section-index
resolver used in the paper. It also separates questions with a parsable section
reference from citation-free fallback queries so that residual section-index
errors are not all attributed to reference-based Act disambiguation.

Run:  python scripts/04_retrieval/wrong_act_severity.py
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dense_retrieval_experiment as base

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "experiments" / "retrieval" / "wrong_act_severity.json"

SEC_RE = re.compile(r"\bsections?\s*[.\-]?\s*(\d+[A-Za-z]{0,2})\b", re.I)
BOOST = 1000.0
K = 5


def norm_sec(s):
    return re.sub(r"\s+", "", s or "").upper() or None


def pct(numerator, denominator):
    return round(100.0 * numerator / denominator, 2) if denominator else None


def main():
    passages, acts, queries, gold, q_acts = base.load_data()
    rows = base.load_jsonl(base.CORPUS_FILE)
    p_act = [r.get("act", "") for r in rows]
    p_sec = [norm_sec(m.group(1)) if (m := SEC_RE.search(r.get("title", "") or "")) else None
             for r in rows]
    by_sec = defaultdict(list)
    for i, s in enumerate(p_sec):
        if s:
            by_sec[s].append(i)

    bm = base.BM25([base.tokenize(p) for p in passages])
    scores = [bm.scores(base.tokenize(q)) for q in queries]

    def rank(citation_aware):
        out = []
        for i, q in enumerate(queries):
            s = list(scores[i])
            if citation_aware:
                sec = norm_sec(m.group(1)) if (m := SEC_RE.search(q or "")) else None
                if sec and by_sec.get(sec):
                    for d in by_sec[sec]:
                        s[d] += BOOST
            out.append(sorted(range(len(s)), key=lambda d: -s[d]))
        return out

    systems = {"BM25": rank(False), "BM25 + section index": rank(True)}

    report = {
        "n_questions": len(gold),
        "k": K,
        "metric_definition": (
            "For each recall@5 miss, compare the Act label of the top-ranked "
            "passage with the annotated gold-source Act label."
        ),
        "systems": {},
    }
    print(
        f"{'system':24s}{'hit@5':>8s}{'miss':>7s}"
        f"{'top1Wrong':>11s}{'top1Same':>10s}{'wrongAct%':>11s}"
    )
    print("-" * 70)
    for name, rk in systems.items():
        hit = wrong = same = 0
        by_reference_status = {
            "parsable_section_reference": {"miss": 0, "top1_wrong_act": 0, "top1_same_act": 0},
            "no_parsable_section_reference": {"miss": 0, "top1_wrong_act": 0, "top1_same_act": 0},
        }
        for i, g in enumerate(gold):
            if g in rk[i][:K]:
                hit += 1
                continue
            status = (
                "parsable_section_reference"
                if SEC_RE.search(queries[i] or "")
                else "no_parsable_section_reference"
            )
            bucket = by_reference_status[status]
            bucket["miss"] += 1
            if p_act[rk[i][0]] != q_acts[i]:
                wrong += 1
                bucket["top1_wrong_act"] += 1
            else:
                same += 1
                bucket["top1_same_act"] += 1
        miss = wrong + same
        for bucket in by_reference_status.values():
            bucket["top1_wrong_act_pct_of_misses"] = pct(
                bucket["top1_wrong_act"], bucket["miss"]
            )
        report["systems"][name] = {
            "hit_at_5": hit,
            "miss": miss,
            "top1_wrong_act": wrong,
            "top1_same_act": same,
            "hit_pct": pct(hit, len(gold)),
            "top1_wrong_act_pct_of_all": pct(wrong, len(gold)),
            "top1_wrong_act_pct_of_misses": pct(wrong, miss),
            "by_reference_status": by_reference_status,
        }
        print(
            f"{name:24s}{hit:8d}{miss:7d}{wrong:11d}{same:10d}"
            f"{pct(wrong, miss):10.1f}%"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
