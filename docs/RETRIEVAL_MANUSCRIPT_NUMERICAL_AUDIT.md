# Retrieval manuscript numerical audit

Audited manuscript: `papers/retrieval_reference/retrieval-reference.tex`.
Date: 2026-09-11.

## Reproducible checks

Run `python scripts/04_retrieval/audit_retrieval_manuscript.py` from the repository root.
The script writes `results/experiments/retrieval/manuscript_audit/summary.json`
and exits unsuccessfully if a comparison fails. It does not overwrite original
experiment artifacts, train models, or run neural inference.

The checks include:

- Recalculation of normalized EM and multiset token F1 for all 344 items in each
  of the five saved gold/BM25/title-BM25/boost/filter reader prediction files.
- Stage counts, selected-source rates, conditional selection rates, and grounded
  F1 from those item-level files.
- Fresh paragraph/title BM25, section-filter, and section-boost rankings over the
  1,116 paragraphs; exact random-pool expected recall.
- Main reranking metrics reconstructed from saved candidate lists and neural
  scores, without rerunning the cross-encoder.
- Fresh sparse original/removed/citation-only/wrong-reference retrieval for both
  citation-bearing question groups using the saved transformed queries.
- UK means recalculated from the per-query CSV; UK subgroup table checked against
  its saved follow-up report.
- Recalculation of the 75-cluster, 10,000-replicate paired bootstrap for filter
  versus paragraph BM25 and filter versus title-plus-paragraph BM25.
- Classifier threshold counts and F1 from the rule and item predictions;
  reader/overlap correlations from the seven-Act analysis CSV.
- Direct table-cell checks for the principal QA, dense retrieval, reranking,
  structural retrieval, stage, UK, subgroup, classifier, and ablation tables.
- Citation-key resolution against the manuscript's actual bibliography.

Metadata/oracle, collision, score-tie, simulation, corruption-seed, and other
supporting summaries were also inspected against their corresponding diagnostic
and follow-up JSON files and implementation. These are artifact checks, not new
neural experiments. Dense encoder results are checked against saved reports;
the encoders were not rerun.

## Corrections

1. The appendix's reader/overlap leave-one-Act-out range was numerically wrong.
   Recalculation gives **[-0.907, -0.674]**, not [-0.927, -0.549]. Excluding Plant
   Varieties gives -0.674. The paragraph and caption now agree. The full-sample
   correlation remains -0.785.
2. The 83.62-to-100.00 template-only recall comparison adds **oracle Act
   restriction to section filtering**; it is not the effect of section filtering
   alone.
3. The 97.09 oracle/title result uses an Act-name full-corpus fallback for queries
   without a parsed section. Its table label and note now distinguish it from
   the pure Act-restricted paragraph control (96.80).
4. Citation removal performs dangling-word/punctuation/whitespace cleanup.
   Six of the 19 no-section queries change. Equal BM25 recall for the control
   does not establish identical query strings.
5. Citation-only retains the first section reference, not every clause or
   subsection locator. The intervention uses section **boosting**, not filtering.
   Its cross-encoder limit is 384 tokens, whereas the main reranking run uses 512.
6. Figure 1 now reports the exact expected random-order recall, **66.72**;
   **66.79** is still correctly reported in the text/table as a simulation mean.
   Figure 23's labels now match the implemented transformations and boost method.
7. Conditional-source selection now explicitly uses gold-available questions as
   its denominator. Rank-five recall is described as **within** the top five.
   The UK depth-100 retrieval/fusion and depth-50 reranking settings are explicit.
8. The bibliography command now points to `retrieval-references.bib`. The missing
   `sn-mathphys-num.bst` was copied from the existing local template, preserving
   the manuscript's selected numbered reference style. The local style omits
   unavailable publisher locations instead of printing literal `???`; no
   locations were invented. Minor spacing/grammar
   and overwide table columns were corrected without rewriting the paper's tone.

## Results retained

The central F1 values remain 87.37 (gold context), 31.95 (paragraph BM25),
45.07 (title-plus-paragraph BM25), and 57.99 (section filter). Grounded filter
F1 remains 47.29. The recalculated filter-minus-title difference is 12.9164
points with interval [6.6216, 19.4449], correctly printed as 12.92 [6.62, 19.44].

The two BM25 recall@5 values, 22.97 and 22.67, are retained because they belong
to different archived tie-breaking implementations; the manuscript explains
the one-question difference. Differences are calculated before rounding.

## Verification limits

This audit verifies numerical consistency with available code, predictions,
rankings, and experiment artifacts. It does not independently validate legal
annotations, cited publications' substantive claims, or model training history.
Resolving all citation keys is not proof that every bibliographic entry or
literature claim is correct. No training job was stopped or restarted.
