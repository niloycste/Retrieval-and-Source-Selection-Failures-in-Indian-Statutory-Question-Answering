"""Citation Contribution Ablation: intervene on the statutory reference.

The paper shows that citation-aware indexing lifts recall@5 from 22.67 to 86.34
while the best learned system reaches 30.23. A reader can object that this is
circular, because almost every question literally contains a section number.

This script measures that programmed sensitivity. Removal includes dangling-word,
punctuation, and whitespace cleanup; replacement leaves other wording unchanged.
Replacement is
not an answer-preserving semantic intervention: evaluation remains relative to
the original annotated paragraph.

  original   the question as released
  content    recognised citations deleted, followed by minor wording cleanup
  citation   the first recognised section reference alone, other wording deleted
  wrong      the first section number replaced by a different section of the SAME Act,
             other wording untouched (three corruption seeds)

Each variant is evaluated with representative sparse, dense, hybrid, re-ranked
and citation-aware systems already examined in the paper. Resolution is
section-level throughout: sub-section and clause structure is not modelled.

Populations are kept separate. The ablation proper runs on questions carrying a
parsable section citation; the citation-free questions are reported as a natural
control and are never given a synthetic "citation-only" or "wrong" variant.
The removal procedure can still clean up their wording. The section-indexed
system uses the 1,000-point boost, and the cross-encoder input limit is 384 tokens.

No training and no new models.

Run:  python scripts/04_retrieval/citation_contribution_ablation.py [--no-ce]
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import dense_retrieval_experiment as base
from citation_specified_questions import (
    CITATION_PATTERNS,
    CITATION_TEMPLATE_ONLY,
    LEXICALLY_CONTENT_BEARING,
    is_citation_template_only,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "experiments" / "retrieval" / "citation_ablation"
OUT_JSON = OUT_DIR / "citation_contribution_ablation.json"
OUT_QUERIES = OUT_DIR / "citation_ablation_queries.jsonl"

K_VALUES = (1, 3, 5, 10)
HYBRID_DEPTH = 10          # matches the paper's hybrid table
CE_FIRST_STAGE_DEPTH = 50  # matches the paper's re-ranking experiment
TOP_RERANK = 50
BOOST = 1000.0
CORRUPTION_SEEDS = (7, 42, 99)

# section-level parsing, fixed: no \s* between the digits and the suffix letters
SEC_RE = re.compile(r"\bsections?\s*[.\-]?\s*(\d+[A-Za-z]{0,2})\b", re.I)
ALL_SEC_RE = re.compile(r"\bsections?\s*[.\-]?\s*\d+[A-Za-z]{0,2}\b", re.I)
DANGLE = re.compile(
    r"\b(under|pursuant to|as provided in|as per|in terms of|of the|in|of)\s*(?=[,.?]|$)",
    re.I)

# published baselines the unmodified condition must reproduce before anything
# in this experiment may be interpreted
EXPECT = {("BM25", 5): 22.67, ("Citation-aware", 5): 86.34, ("Citation-aware", 10): 95.64}
TOL = 0.5


def norm_sec(s):
    return re.sub(r"\s+", "", s or "").upper() or None


def parse_section(text):
    m = SEC_RE.search(text or "")
    return norm_sec(m.group(1)) if m else None


# ----------------------------------------------------------- transformations
def make_content(q):
    s = q
    for pat in CITATION_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.I)
    s = DANGLE.sub(" ", s)
    s = re.sub(r"\s*,\s*,", ",", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[,;:\s]+", "", s)
    s = re.sub(r"\s+([,.?])", r"\1", s)
    return (s[0].upper() + s[1:]).strip() if s else s


def make_citation(q):
    m = SEC_RE.search(q or "")
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else None


def make_wrong(q, act_secs, rng):
    """Replace the section number with a different section of the same Act."""
    m = SEC_RE.search(q or "")
    if not m:
        return None
    true = norm_sec(m.group(1))
    pool = [s for s in act_secs if s != true]
    if not pool:
        return None
    repl = rng.choice(pool)
    start = m.start(1)
    end = start + len(m.group(1).rstrip())
    return q[:start] + repl + q[end:]


# ------------------------------------------------------------------- metrics
def R(rk, ids, gold, k):
    return round(100.0 * sum(1 for i in ids if gold[i] in rk[i][:k]) / len(ids), 2) if ids else float("nan")


def MRR(rk, ids, gold):
    t = sum(1.0 / (rk[i].index(gold[i]) + 1) for i in ids if gold[i] in rk[i][:10])
    return round(100.0 * t / len(ids), 2) if ids else float("nan")


def hits(rk, ids, gold, k=5):
    return [1 if gold[i] in rk[i][:k] else 0 for i in ids]


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    if n01 + n10 == 0:
        return n10, n01, 1.0
    return n10, n01, min(1.0, binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_ce = "--no-ce" not in sys.argv
    passages, acts, queries, gold, q_acts = base.load_data()
    corpus_rows = base.load_jsonl(base.CORPUS_FILE)
    psec = [parse_section(r.get("title", "") or "") for r in corpus_rows]
    pact = [r.get("act", "") for r in corpus_rows]

    assert len(queries) == 344, f"expected 344 test questions, got {len(queries)}"
    is_template_only = [is_citation_template_only(q) for q in queries]
    n_template = sum(is_template_only)
    n_lexical = len(is_template_only) - n_template
    assert n_template == 177 and n_lexical == 167, (
        f"expected 177/167 split, got {n_template}/{n_lexical}"
    )
    print(f"corpus {len(passages)} passages | queries {len(queries)} | "
          f"{CITATION_TEMPLATE_ONLY} {n_template} | "
          f"{LEXICALLY_CONTENT_BEARING} {n_lexical}")

    secs_by_act = defaultdict(set)
    for a, s in zip(pact, psec):
        if s:
            secs_by_act[a].add(s)
    by_sec = defaultdict(list)
    for i, s in enumerate(psec):
        if s:
            by_sec[s].append(i)

    ncites = [len(ALL_SEC_RE.findall(q or "")) for q in queries]
    dist = defaultdict(int)
    for n in ncites:
        dist[min(n, 2)] += 1
    print(f"citations per question: 0 -> {dist[0]}, 1 -> {dist[1]}, 2+ -> {dist[2]}")

    cited_ids = [i for i, n in enumerate(ncites) if n >= 1]
    free_ids = [i for i, n in enumerate(ncites) if n == 0]
    print(f"ablation population (>=1 citation) {len(cited_ids)} | "
          f"citation-free control {len(free_ids)}")

    V = {"original": list(queries),
         "content": [make_content(q) for q in queries],
         "citation": [make_citation(q) for q in queries]}
    wrong_sets, wrong_ok = {}, {}
    for sd in CORRUPTION_SEEDS:
        rng = random.Random(sd)
        qs, ok = [], []
        for i, q in enumerate(queries):
            w = make_wrong(q, sorted(secs_by_act.get(q_acts[i], ())), rng)
            qs.append(w if w else q)
            ok.append(w is not None)
        wrong_sets[sd], wrong_ok[sd] = qs, ok
        print(f"  wrong-citation seed {sd}: {sum(ok)} of {len(queries)} eligible")
    V["wrong"] = wrong_sets[42]

    elig = [i for i in cited_ids
            if V["citation"][i] and all(wrong_ok[s][i] for s in CORRUPTION_SEEDS)]
    print(f"fully eligible for every condition: {len(elig)}")

    with OUT_QUERIES.open("w", encoding="utf-8") as fh:
        for i, q in enumerate(queries):
            fh.write(json.dumps({
                "idx": i, "act": q_acts[i],
                "group": (
                    CITATION_TEMPLATE_ONLY
                    if is_template_only[i]
                    else LEXICALLY_CONTENT_BEARING
                ),
                "n_citations": ncites[i], "eligible": i in set(elig),
                "true_section": parse_section(q),
                "wrong_section_seed42": parse_section(wrong_sets[42][i]),
                "original": q, "content": V["content"][i],
                "citation": V["citation"][i], "wrong_seed42": wrong_sets[42][i],
            }, ensure_ascii=False) + "\n")
    print(f"transformed queries -> {OUT_QUERIES.name}")

    bm = base.BM25([base.tokenize(p) for p in passages])

    def bm25_rank(qs):
        out = []
        for q in qs:
            sc = bm.scores(base.tokenize(q or ""))
            out.append(sorted(range(len(sc)), key=lambda d: -sc[d]))
        return out

    def cite_rank(qs):
        out = []
        for q in qs:
            sc = list(bm.scores(base.tokenize(q or "")))
            s = parse_section(q)
            if s and by_sec.get(s):
                for d in by_sec[s]:
                    sc[d] += BOOST
            out.append(sorted(range(len(sc)), key=lambda d: -sc[d]))
        return out

    qsets = {"original": V["original"], "content": V["content"],
             "citation": V["citation"], "wrong": V["wrong"]}
    for sd in CORRUPTION_SEEDS:
        qsets[f"wrong_s{sd}"] = wrong_sets[sd]

    S = {}
    for name, qs in qsets.items():
        S[("BM25", name)] = bm25_rank(qs)
        S[("Citation-aware", name)] = cite_rank(qs)
    print("sparse + citation-aware done", flush=True)

    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
    print("encoding passages (e5-small-v2) ...", flush=True)
    st = SentenceTransformer("intfloat/e5-small-v2", device="cpu")
    P = st.encode(["passage: " + p for p in passages], batch_size=64,
                  convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu",
                      max_length=384) if use_ce else None

    ce_variants = ("original", "content", "citation", "wrong")
    for name, qs in qsets.items():
        Q = st.encode(["query: " + (q or "") for q in qs], batch_size=64,
                      convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        sim = Q @ P.T
        dense = [list(np.argsort(-sim[i])) for i in range(len(qs))]
        S[("Dense e5", name)] = dense
        bmr = S[("BM25", name)]
        for depth, label in ((HYBRID_DEPTH, "Hybrid RRF"), (CE_FIRST_STAGE_DEPTH, "_hyb50")):
            fused = []
            for i in range(len(qs)):
                rr = defaultdict(float)
                for r, d in enumerate(bmr[i][:depth]):
                    rr[d] += 1.0 / (base.RRF_K + r + 1)
                for r, d in enumerate(dense[i][:depth]):
                    rr[d] += 1.0 / (base.RRF_K + r + 1)
                fused.append(sorted(rr, key=lambda d: -rr[d]) + [d for d in bmr[i] if d not in rr])
            S[(label, name)] = fused
        if ce and name in ce_variants:
            for src, label in (("BM25", "BM25+CE"), ("_hyb50", "Hybrid+CE")):
                t0 = time.time()
                out = []
                for i in range(len(qs)):
                    cand = S[(src, name)][i][:TOP_RERANK]
                    sc = ce.predict([[qs[i] or "", passages[d]] for d in cand],
                                    batch_size=64, show_progress_bar=False)
                    ranked = [d for _, d in sorted(zip(sc, cand), key=lambda t: -t[0])]
                    seen = set(cand)
                    out.append(ranked + [d for d in S[(src, name)][i] if d not in seen])
                S[(label, name)] = out
                print(f"  {label} [{name}] {time.time()-t0:.0f}s", flush=True)

    SYSTEMS = [s for s in ["BM25", "Dense e5", "Hybrid RRF", "BM25+CE", "Hybrid+CE",
                           "Citation-aware"] if (s, "original") in S]

    print("\n=== baseline reproduction gate ===")
    allq = list(range(len(queries)))
    bad = []
    for (sysname, k), exp in EXPECT.items():
        obs = R(S[(sysname, "original")], allq, gold, k)
        ok = abs(obs - exp) <= TOL
        print(f"  {sysname:16s} R@{k:<3d} expected {exp:6.2f}  observed {obs:6.2f}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{sysname} R@{k}: expected ~{exp}, got {obs}")
    if bad:
        raise RuntimeError("baseline mismatch; refusing to interpret ablation:\n  " + "\n  ".join(bad))

    groups = {
        CITATION_TEMPLATE_ONLY: [i for i in elig if is_template_only[i]],
        f"{LEXICALLY_CONTENT_BEARING} (with citation)": [
            i for i in elig if not is_template_only[i]
        ],
        "all eligible": elig,
        "citation-free control": free_ids,
    }
    results, raw_p = {}, []
    for gname, ids in groups.items():
        print("\n" + "=" * 104)
        print(f"GROUP: {gname}  (n={len(ids)})")
        print("=" * 104)
        conds = ("original", "content") if gname == "citation-free control" else \
                ("original", "content", "citation", "wrong")
        print(f"{'system':16s}{'variant':10s}" + "".join(f"{'R@'+str(k):>9s}" for k in K_VALUES) + f"{'MRR@10':>9s}")
        print("-" * 104)
        for s in SYSTEMS:
            for v in conds:
                rk = S[(s, v)]
                row = {f"R@{k}": R(rk, ids, gold, k) for k in K_VALUES}
                row["MRR@10"] = MRR(rk, ids, gold)
                results.setdefault(gname, {}).setdefault(s, {})[v] = row
                print(f"{s:16s}{v:10s}" + "".join(f"{row['R@'+str(k)]:9.2f}" for k in K_VALUES) + f"{row['MRR@10']:9.2f}")
            has_seeds = all((s, f"wrong_s{sd}") in S for sd in CORRUPTION_SEEDS)
            if "wrong" in conds and has_seeds:
                vals = [R(S[(s, f"wrong_s{sd}")], ids, gold, 5) for sd in CORRUPTION_SEEDS]
                m = sum(vals) / len(vals)
                sd_ = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                results[gname][s]["wrong_seeds"] = {"R@5": [round(v, 2) for v in vals],
                                                    "mean": round(m, 2), "sd": round(sd_, 2)}
                print(f"{'':16s}{'wrong x3':10s}{m:9.2f}  (seeds {CORRUPTION_SEEDS}, SD {sd_:.2f})")
            print("-" * 104)
        if gname == "citation-free control":
            continue
        print("  paired McNemar on recall@5 (baseline = original)")
        for s in SYSTEMS:
            bh = hits(S[(s, "original")], ids, gold)
            for v in ("content", "wrong"):
                vh = hits(S[(s, v)], ids, gold)
                lost, gained, p = mcnemar(bh, vh)
                d = R(S[(s, v)], ids, gold, 5) - R(S[(s, "original")], ids, gold, 5)
                raw_p.append((gname, s, v, p))
                results[gname][s].setdefault("_tests", {})[v] = {
                    "delta_r5": round(d, 2), "lost": lost, "gained": gained, "p_raw": p}
                print(f"    {s:16s} original -> {v:8s} dR@5 {d:+7.2f}  lost {lost:3d} gained {gained:3d}  p={p:.3e}")

    adj = holm([p for *_, p in raw_p])
    print("\n=== Holm-corrected p-values (all contrasts) ===")
    for (gname, s, v, p), pa in zip(raw_p, adj):
        results[gname][s]["_tests"][v]["p_holm"] = pa
        star = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "ns"
        print(f"  {gname:32s} {s:16s} {v:8s} p_raw={p:.3e}  p_holm={pa:.3e}  {star}")

    print(
        "\n=== PRE-DECLARED PRIMARY HYPOTHESES "
        f"(Citation-aware, {CITATION_TEMPLATE_ONLY}) ==="
    )
    prim = results.get(CITATION_TEMPLATE_ONLY, {}).get(
        "Citation-aware", {}
    ).get("_tests", {})
    for h, v in (("H1 original > content", "content"), ("H2 original > wrong", "wrong")):
        t = prim.get(v)
        if t:
            print(f"  {h:26s} dR@5 {t['delta_r5']:+7.2f}   p_holm={t['p_holm']:.3e}")

    output_json = OUT_JSON if use_ce else OUT_DIR / "citation_contribution_ablation_no_ce.json"
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {output_json}")


if __name__ == "__main__":
    main()
