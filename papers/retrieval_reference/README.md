# Statutory-reference retrieval paper

This folder contains the independent retrieval manuscript:

- `retrieval-reference.tex` — authoritative LaTeX source;
- `retrieval-references.bib` — standalone bibliography copied from and aligned
  with the verified repository bibliography;
- `plots/` — figures used by this paper;
- `PAPER_B_CHANGELOG.md` — changes made while separating the paper;
- `retrieval-reference.pdf` — latest verified build.

The combined manuscript under `combined_paper/` is preserved and is not the
editable source for this split paper.

## Build

From this folder, run:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error retrieval-reference.tex
bibtex retrieval-reference
pdflatex -interaction=nonstopmode -halt-on-error retrieval-reference.tex
pdflatex -interaction=nonstopmode -halt-on-error retrieval-reference.tex
```

Regenerate the quantitative analysis figures and the archived alternative
pipeline diagram from the repository root with:

```powershell
python scripts/08_figures/make_retrieval_paper_analysis_figures.py
python scripts/08_figures/make_retrieval_decoder_diagram.py
```

The generators read the canonical retrieval reports, assert the displayed
headline values, and write both PDF and PNG versions under `plots/`. The first
command generates files `01`, `20`, `21`, `22`, and `23`; files `01`, `21`,
`22`, and `23` are used in the current manuscript. The second command generates
file `24`, which is retained as an archived alternative diagram. The current
Figure 2 is the curated `pipeline1.png` inference and offline-diagnostics
diagram. The older `pipeline.png`, `pipeline.pdf`, and file `24` are not
included in the current manuscript.

Canonical result sources are documented in
`docs/RESULT_PROVENANCE.md` at the repository root.

---

## What this paper owns

The main experiments change retrieval while the reader checkpoint stays fixed.
The post-review checks also vary source selection and the decoding length cap
without retraining the reader. Pretraining composition is not studied here.

### Code

| Path | Role |
|---|---|
| `scripts/04_retrieval/` | BM25, dense, hybrid/RRF, cross-encoder, citation-aware indexing, section resolution, citation ablation, wrong-Act severity |
| `scripts/07_llm_baselines/` | LLM comparison points for the retrieval setting |

`scripts/03_finetuning/` is **shared**: it produced E2′, the fixed reader this
paper uses as a measuring instrument rather than a contribution.

### Results

| Path | Contents |
|---|---|
| `results/experiments/retrieval/` | all retrieval runs, citation ablation, section-filter and per-Act analyses |
| `results/experiments/llm_baselines/` | LLM baseline outputs |
| `results/evaluation/qa_eval_report.json` | reader inventory (shared; this paper cites only E2′ = 87.37) |

### Change log

`PAPER_B_CHANGELOG.md` records the split from the parent manuscript.

## Status

**Core learned experiments are frozen.** The canonical model runs and their
artifacts remain unchanged. A locally pre-specified lightweight follow-up audit
adds collision, candidate-restriction, tie-sensitivity, boost--filter, and UK
reference controls via:

- `scripts/04_retrieval/retrieval_followup_controls.py`
- `scripts/04_retrieval/uk_reference_followup_controls.py`
- `docs/RETRIEVAL_FOLLOWUP_CONTROLS_PROTOCOL.md`
- `scripts/04_retrieval/metadata_matched_end_to_end_control.py` (additional post-hoc metadata-matched inference)

The main canonical numbers remain fixed:

| Quantity | Value |
|---|---|
| Gold-context reader (E2′) | 87.37 F1 |
| BM25 retrieval-only R@5 | 22.67 |
| Best learned pipeline R@5 | 30.23 |
| Title+paragraph BM25 R@5 / reader F1 | 56.98 / 45.07 |
| Section-aware R@5 | 86.34 |
| Section-aware reader F1 | 57.99 |
| Citation removed / citation only / wrong citation R@5 | 1.13 / 62.71 / 0.00 |

## Post-review revision (2026-09-20)

See `REVIEW_RESPONSE_20260920.md` for the issue-by-issue disposition, evidence,
and remaining author actions. The main tables now consistently use canonical
BM25 ordering. New frozen-reader sensitivity results, parser agreement checks,
span-length summaries and length-stratified scores are in Appendix A.6.

From the repository root:

```powershell
python scripts/04_retrieval/review_robustness_audit.py --with-reader
python scripts/04_retrieval/audit_retrieval_manuscript.py
python scripts/04_retrieval/audit_retrieval_bibliography.py
```

The first command reuses its provenance-checked inference cache when available;
without the cache it requires the archived reader checkpoint and runs inference,
not training. Outputs are under
`results/experiments/retrieval/review_robustness_20260920/`. The second checks
the manuscript against saved results and recomputed metrics; the latest run
passes 28,888 checks. This is not independent scientific validation.

## Human evaluation update

The manuscript now reports an 80-case audit by two law graduate students, with 20 cases
from each section-filter outcome. Original agreement is calculated separately
from revised source labels and final joint consensus. The appendix includes
raw agreement, nominal Cohen kappa, Gwet AC1, and quadratic weighted materiality
kappa with the relevant paired-case counts and exclusions. The main table
reports final consensus by sampling stratum; pooled counts are not presented
as prevalence estimates for all 344 questions.

Reproduce and verify the manuscript tables from the local master workbook:

```powershell
python scripts/04_retrieval/summarize_human_validation.py --workbook "outputs/human validation/legal_expert_audit_master.xlsx" --verify-manuscript papers/retrieval_reference/retrieval-reference.tex --crosscheck-sklearn
```

This command is read-only for the workbook. It verifies the seeded sample,
passage/answer role mappings, all 100 agreement rows, and both new manuscript
tables. The optional scikit-learn check independently verifies 100 nominal and
20 weighted kappa calculations. Results and input hashes are saved in
`outputs/human_validation_report/manuscript_summary.json`. The master contains
private reviewer details; it is not automatically copied into the public repo.

The author confirmed that both students knew of and agreed to the research use
of their assessments and were not shown AI-generated audit drafts during
annotation or consensus. This is distinct from the model predictions being
assessed in the forms. The manuscript uses anonymised codes, not their names.
Institutional ethics approval or exemption status still requires author
confirmation and is marked in the manuscript. This audit does not validate the section parser
or operational question classifier, and the post-hoc partition still has no
independent validation set. Review-time anonymised artifact access also needs
to be arranged. A successful build is not a certification of legal validity.
