# Retrieval–Reading Gap in Statutory Question Answering

Reproducibility repository for the manuscript:

> **Beyond Answer Extraction: Diagnosing Retrieval and Source Selection Failures in Indian Statutory Question Answering**

## Authors

- M. Mohaiminul Islam, Department of Computer Science and Engineering, South Asian University, New Delhi, India
- Muhammad Abulaish, Department of Computer Science and Engineering, South Asian University, New Delhi, India

This repository is independent of the companion DAPT study. It contains the
retrieval manuscript, fixed IndicIPR-QA splits, retrieval and reader-analysis
code, canonical result artifacts, saved audit outputs, and figure generators.
Large model checkpoints and regenerable embedding caches are intentionally not
versioned.

## Problem addressed

High extractive-QA performance with an annotated statutory paragraph does not
show that a system can find that paragraph in a corpus. This study measures the
resulting retrieval–reading gap and separates three annotation-relative failure
stages:

1. the annotated paragraph is absent from the retrieved candidate set;
2. it is available, but the fixed reader selects another source; or
3. it is selected, but the extracted span does not exactly match the annotation.

The experiments then examine why questions in the zero-residual reference
pattern group are difficult for paragraph-only retrieval and whether aligning
query-side section locators with statutory metadata changes candidate access.
Section matching is evaluated as a controlled diagnostic, not proposed as a
universal retrieval architecture.

## Main preserved findings

| Result | Preserved value |
|---|---:|
| Annotated-paragraph reader F1 | 87.37 |
| Paragraph-only BM25 top-5 reader F1 | 31.95 |
| Paragraph-only BM25 recall@5 | 22.67% |
| Title-plus-paragraph BM25 recall@5 | 56.98% |
| Section-filter recall@5 | 86.34% |
| Section-filter reader F1 | 57.99 |
| Title-aware neural recall@5 | 83.72% |
| Title-aware neural F1, reader-logit source selection | 53.12 |
| Title-aware neural F1, BGE rank-one source selection | 54.62 |
| Section filter minus title-aware BM25 F1 | +12.92, 95% cluster interval [6.62, 19.44] |

The repository also preserves dense retrieval, RRF, cross-encoder and BGE
experiments, reference removal/isolation/replacement, collision controls,
decoder sensitivity, stage-wise outcomes, and the UK retrieval comparison.

## Additional section-pool controls

The release also includes `section_filter_act_extension_v2`, a CPU-only
retrieval extension that keeps the canonical BM25 definition and tests:

- random ordering inside section-matched pools;
- same-size random pools sampled from the full corpus;
- wrong-locator sensitivity without annotated Acts; and
- a question-only Act prediction filter trained on the training split.

The Act prediction filter is a negative result: it lowers test recall@5 from
86.34% for section-only BM25 to 75.00%. The run therefore documents a limit of
this simple pipeline rather than a new recommended model. The protocol,
per-query rankings, model file, split audit, input hashes, and independent
metric check are included in the run directory.

## Repository layout

```text
data/processed/indic_ipr_qa/   released paragraph corpus
data/splits/                   fixed train, validation, and test splits
docs/                          protocols, provenance, and numerical audits
outputs/human_validation_report/       verified aggregate and de-identified audit data
papers/retrieval_reference/    LaTeX manuscript, bibliography, figures, and PDF
results/experiments/retrieval/ canonical results and item-level diagnostics
scripts/04_retrieval/          retrieval, reranking, controls, and audits
scripts/08_figures/            manuscript figure generators
scripts/qa_evaluation.py       shared EM/F1 and span-evaluation utilities
```

## Quick verification

For a results-only reproduction that does not download models, create the
environment and run:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python verify_release.py
python scripts/04_retrieval/audit_retrieval_manuscript.py
```

These commands check the fixed data split, headline retrieval and QA results,
saved item-level evidence, tables, counts, and reported confidence intervals.

### Human evaluation

The manuscript now includes 80 cases assessed by two law graduate students, sampled
equally from four section-filter outcomes. Original inter-annotator agreement
is kept separate from revised labels and final consensus. Aggregate counts,
agreement coefficients, calculation checks, and input hashes are preserved in
`outputs/human_validation_report/manuscript_summary.json`. The file
`anonymized_item_labels.jsonl` contains the de-identified original categorical
ratings and consensus labels needed for independent reanalysis.

The named reviewer workbooks, free-text rationales, authority notes, dates, and
the passage-order decoding key are private and are not copied into this repository.
Given an authorised local copy of the master, recompute the analysis with:

```bash
python scripts/04_retrieval/summarize_human_validation.py --workbook "/path/to/legal_expert_audit_master.xlsx" --verify-manuscript papers/retrieval_reference/retrieval-reference.tex --crosscheck-sklearn
```

The command does not modify the workbook. The final option additionally checks
kappa against scikit-learn. The author confirmed that both students agreed to
the research use of their assessments and did not see AI-generated audit drafts
during annotation or consensus. The manuscript does not name the students.
Institutional ethics approval or exemption status still requires author
confirmation before journal submission.

### Existing retrieval checks

From the repository root:

```bash
python verify_release.py
python scripts/04_retrieval/audit_retrieval_manuscript.py
```

The release verifier checks the fixed test size and the principal manuscript
values, including BM25 recall@5, section-filter recall@5, and end-to-end F1.

The post-hoc title-aware neural end-to-end control can be rerun with the fixed
reader checkpoint at the path documented in the manuscript:

```bash
python scripts/04_retrieval/title_neural_end_to_end_control.py --device auto
```

The script reuses saved title-aware BGE rankings and cached reader outputs,
decodes only missing question-passage pairs, and writes item-level predictions
and a summary under
`results/experiments/retrieval/runs/title_neural_end_to_end_v1/`.

### Reproducibility levels

| Level | Command | Requires models? | Purpose |
|---|---|---:|---|
| Release integrity | `python verify_release.py` | No | Checks required files, test size, and headline values |
| Numerical manuscript audit | `python scripts/04_retrieval/audit_retrieval_manuscript.py` | No | Recomputes manuscript tables, counts, and bootstrap intervals from saved evidence |
| Figure regeneration | commands below | No | Recreates plots from canonical JSON/CSV artifacts |
| Retrieval reruns | scripts under `scripts/04_retrieval/` | Yes | Downloads public encoders/rerankers and reruns retrieval |
| Reader inference | reader-enabled scripts | Yes | Requires the documented fixed reader checkpoint or a regenerated equivalent |

## Environment

Python 3.12 was used for the final runs. Create an isolated environment and
install the pinned packages:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dense retrieval, reranking, and reader inference download public Hugging Face
models on first use. GPU acceleration is recommended but not required for the
results-only audits.

## Reproduce manuscript figures

```bash
python scripts/08_figures/make_retrieval_paper_analysis_figures.py
python scripts/08_figures/make_retrieval_decoder_diagram.py
```

The scripts read the versioned canonical JSON files and write figures under
`papers/retrieval_reference/plots/`.

## Compile the manuscript

```bash
cd papers/retrieval_reference
pdflatex retrieval-reference.tex
bibtex retrieval-reference
pdflatex retrieval-reference.tex
pdflatex retrieval-reference.tex
```

## Full experiments

The principal entry points are documented in
`papers/retrieval_reference/README.md`. Expensive runs may download models or
the externally licensed UK-StatuteCorpus. The UK corpus and embedding `.npy`
caches are not redistributed; their provenance and download handling are in
`scripts/04_retrieval/uk_statutecorpus_external_validation.py`.

Canonical artifact-to-claim mappings are listed in
`docs/RESULT_PROVENANCE.md`. All retrieval paths listed in that provenance
table are included in this repository. The expensive UK embedding arrays are
excluded because they are caches rather than scientific results.

## Artifact policy

Included: source code, fixed splits, released corpus text, canonical summaries,
item-level predictions and audits, de-identified human-validation labels,
manuscript sources, and figures.

Excluded: trained checkpoints, optimizer states, local model downloads,
UK-StatuteCorpus files, embedding caches, temporary logs, API credentials,
annotator-identifying workbooks, private notes, and decoding keys.

Before publishing publicly, choose a software/data licence and replace any
anonymous companion-paper citation or repository placeholder required by the
final journal submission.
