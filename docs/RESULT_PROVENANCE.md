# Retrieval paper result provenance

The retrieval manuscript is generated from the current canonical artifacts
below. Percentages in the paper are rounded to two decimal places unless the
text explicitly states otherwise.

| Manuscript result | Canonical artifact |
|---|---|
| E2′ gold-context and BM25 end-to-end reader scores | `results/experiments/retrieval/citation_aware_end_to_end_report.json` |
| BM25, dense, and reciprocal-rank-fusion retrieval | `results/experiments/retrieval/dense_retrieval_report.json` (paragraph-only BM25 and E5 lists truncated to depth 10, fused with RRF \(k=60\), output depth 10) |
| Cross-encoder re-ranking and first-stage ceilings | `results/experiments/retrieval/cross_encoder_rerank_report.json` (paragraph-only BM25 and E5 lists truncated to depth 50, fused with RRF \(k=60\), all 50 fused candidates re-ranked) |
| Section parsing, section boost/filter retrieval, per-Act recall | `results/experiments/retrieval/citation_aware_retrieval_report.json` |
| Section-aware end-to-end QA and query-type reader scores | `results/experiments/retrieval/citation_aware_end_to_end_report.json` |
| Additional post-hoc title-plus-paragraph BM25 end-to-end control | `results/experiments/retrieval/runs/metadata_matched_end_to_end_v1/summary.json` (metadata-matched control; generator: `scripts/04_retrieval/metadata_matched_end_to_end_control.py`) |
| Citation removal, isolation, and wrong-reference intervention | `results/experiments/retrieval/citation_ablation/citation_contribution_ablation.json` |
| Operational classifier thresholds and continuous residual-token analysis | `results/experiments/retrieval/citation_template_classifier_robustness.json` |
| Cross-Act and within-Act retrieval failure counts | `results/experiments/retrieval/wrong_act_severity.json` |
| UK-StatuteCorpus external retrieval validation | `results/experiments/retrieval/runs/uk_statutecorpus_external_v1/summary.json` (canonical completed run; protocol and generator: `docs/RETRIEVAL_EXTERNAL_VALIDATION_PROTOCOL.md`, `scripts/04_retrieval/uk_statutecorpus_external_validation.py`) |
| Figures 1 and 4 (files `01` and `23`) | `scripts/08_figures/make_retrieval_mechanism_figures.py` (current two-panel overview and citation intervention) |
| Figures 2, 3, and 5 (files `21`, `22`, and `20`) | `scripts/08_figures/make_retrieval_paper_analysis_figures.py` (retrieval progression, stage-wise outcomes, and exploratory per-Act correlations; its main command also delegates Figures 1 and 4 to the current mechanism generator) |

The figure generator loads the canonical JSON files and asserts the headline
values before writing the PDF and PNG. The manuscript should not be updated
from stale reports or manually transcribed alternatives.

## Headline retrieval invariants

- Gold paragraph + E2′: 87.37 F1
- BM25 recall@5: 22.67%
- Title-plus-paragraph BM25 recall@5: 56.98%
- Title-plus-paragraph BM25 top-5 reader F1: 45.07
- BM25 top-5 + E2′: 31.95 F1
- Best tested learned retrieval pipeline: 30.23% recall@5
- Citation-template-only partition: 177 of 344 test questions
- Section-filter recall@5: 86.34%
- Section-filter minus title-plus-paragraph BM25 reader: +12.92 F1;
  source-cluster bootstrap 95% interval [6.62, 19.44]
- Section-filter + E2′: 57.99 F1
- Citation intervention on the template-only subset: 83.62% original,
  1.13% after reference removal, 62.71% reference-only, and 0.00% after a
  valid wrong same-Act replacement

## Target-pretraining paper provenance

The DAPT manuscript in `papers/target_pretraining/` uses the following
artifacts. These are treated as audited archived evidence unless a future
train-only rerun writes replacement outputs under `experiments/dapt/`.

| Manuscript result | Canonical artifact |
|---|---|
| Gold-context reader inventory, including A, B′, E2′, H′, C′, and I′ | `results/evaluation/qa_eval_report.json` |
| Bootstrap comparison of C′ against A, B′, E2′, H′, and I′ | `results/evaluation/bootstrap_significance_report.json` |
| Archived DAPT corpus composition | `combined_paper/Springer_Nature_LaTeX_Template/dapt_corpus_manifest.csv` |
| Archived DAPT MLM loss/perplexity | `results/evaluation/mlm_eval_report.json` |
| Record-matched single-seed DAPT results | `results/experiments/size_matched_dapt/size_matched_results.json` |
| Record-matched three-seed DAPT results | `results/experiments/size_matched_dapt/size_matched_multiseed_results.json` |
| Record-matched DAPT statistics | `results/experiments/size_matched_dapt/size_matched_multiseed_stats.json` |
| Unified archived and record-matched MLM diagnostic | `results/evaluation/mlm_eval_report.json` |
| Record/token/split audit | `results/experiments/size_matched_dapt/size_matched_corpus_audit.json` |
| DAPT figures | `scripts/08_figures/make_target_pretraining_figures.py` |

### Headline DAPT invariants

- Direct InLegalBERT A: 63.66 EM / 85.02 F1
- InLegalBERT + SQuAD B′: 67.73 EM / 86.02 F1
- LEGAL-BERT + SQuAD E2′: 66.86 EM / 87.37 F1
- Authentic DAPT H′: 65.99 EM / 85.73 F1
- Mixed DAPT C′: 67.44 EM / 87.04 F1
- Synthetic DAPT I′: 65.41 EM / 85.96 F1
- C′ minus B′ (same InLegalBERT + SQuAD route): +1.02 F1; 95% paired-question
  bootstrap interval [-0.68, 2.84]; Holm-adjusted p = 0.502
- C′ minus E2′: -0.33 F1 at displayed two-decimal precision
- Record-matched mean F1: auth901 86.03, synth901 85.73, mixed901 85.31
- Record-matched pooled within-condition SD: 0.71 F1
- Record-matched spread of means: 0.72 F1
- Record-matched seed-blocked ANOVA (primary): p = 0.6257
- Record-matched one-way ANOVA (sensitivity): F(2,6) = 0.7779,
  p = 0.5007, sample η² = 0.2059
- Widest paired 95% CI bound among matched comparisons: 4.26 F1
- Archived MLM perplexity: InLegalBERT 11.20, H′ 2.49, C′ 2.57, I′ 2.79
- Record-matched MLM perplexity from the unified evaluator: InLegalBERT 11.20,
  auth901 3.01, synth901 3.18, mixed901 3.03
- Non-padding tokens at maximum length 256: auth901 128,929; synth901 146,001;
  mixed901 138,141. These corpora are record-matched, not token-matched.
- Test-context exposure: auth901 3 contexts / 6 questions; synth901 0 / 0;
  mixed901 1 / 2.
