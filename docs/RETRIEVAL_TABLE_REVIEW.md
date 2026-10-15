# Retrieval manuscript table review

Reviewed 2026-09-13. Scope: all 20 tables in `papers/retrieval_reference/retrieval-reference.tex`, including numerical agreement, definitions, captions, and rendered layouts. No training or inference was rerun.

## Findings and edits

- No additional experimental values required correction in this pass. The earlier Table 20 transcription correction remains in place.
- Table 3 now explicitly separates section-title matching from paragraph-text ranking and points to Table 12 for oracle variants and fallbacks.
- Tables 5, 6, 7, 9, 11, 13, 14, 15, 18, and 19 now state their score scales. Table 9 distinguishes 0–1 lexical overlap from 0–100 retrieval/F1 scores.
- Table 15 identifies seed 42 for the displayed wrong-reference results and labels the three-seed summary as mean ± standard deviation over the appropriate citation-carrying subgroup.
- Table 17 defines the shortened operational labels.
- Removed negative spacing between Table 7's two Grade-3 result lines.
- Retained the Table 12 oracle fallback distinction: 97.09 is not a title-only alteration of the 96.80 configuration.

## Coverage

| Tables | Evidence / checks |
|---|---|
| 1 | Related-work descriptions checked for consistency with the manuscript and primary publication pages listed below; this is not a full systematic-literature or absence-of-intervention audit. |
| 2 | Recomputed counts and whitespace-token lengths from `data/splits/{train,val,test}.jsonl`: 4,462 pairs, 7 Acts, 681 Act-section groups, 1,116 distinct raw contexts; mean context/question/answer lengths 171.2692/11.9005/35.6446; answer median 28, maximum 375; semicolon share 19.9014%. Mean lexical overlap 0.450792 using the stop-word rule in `benchmark_analysis/analysis_5_difficulty.py`. |
| 3 | Retrieval input specification compared with the diagnostic-control code and Table 12. |
| 4–6 | Saved retrieval-QA, dense, and reranking reports; sparse reconstruction and cached reranker scores. |
| 7–8 | UK saved summary and per-query results, including subgroup summaries. |
| 9 | Per-Act sparse/hybrid/reranker reports and archived reader difficulty CSV. |
| 10–11 | Question-type reader summaries and section-boost retrieval report. |
| 12 | Diagnostic and follow-up control reports and oracle fallback code; numeric verification from the preceding focused review retained. |
| 13–14 | Saved per-question predictions, rescored EM/F1 and source-stage counts. |
| 15 | Saved ablation report and transformed queries; sparse variants reconstructed; seed-summary definition checked in the generator. |
| 16 | Artifact categories checked against the reports/scripts used in this review. Historical development dates were not independently authenticated. |
| 17–19 | Operational classifier examples, threshold summaries, and residual-token bins. |
| 20 | Saved gold-context and BM25 predictions; exact text and annotation-relative score checks from the focused review retained. This does not validate the legal correctness of the annotation. |

## Reproducible numerical audit

Run `python -X utf8 scripts/04_retrieval/audit_retrieval_manuscript.py`.
The expanded audit passes 3,936 checks, including additional per-Act, threshold-table, and classifier-example checks. Detailed output is in `results/experiments/retrieval/manuscript_audit/summary.json`. The check count includes per-item metrics; it is not a count of table cells or independent experiments.

## Related-work source pages inspected

- [BSARD](https://aclanthology.org/2022.acl-long.468/)
- [Finding the Law](https://aclanthology.org/2023.eacl-main.203/)
- [JuriFindIT](https://aclanthology.org/2026.findings-eacl.221/)
- [SearchFireSafety](https://aclanthology.org/2026.acl-long.2112/)
- [UK neural reranking](https://link.springer.com/article/10.1007/s10506-025-09501-6)

All table-containing pages were rendered and visually inspected. The audit supports internal consistency with the checked artifacts, not an unconditional claim that all experimental or literature assertions have independent external validation.
