<div align="center">

# ⚖️ Retrieval and Source Selection in Statutory QA

### Beyond Answer Extraction: Diagnosing Retrieval and Source Selection Failures in Indian Statutory Question Answering

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![Task: Statutory QA](https://img.shields.io/badge/Task-Statutory%20Extractive%20QA-2563EB)](#research-at-a-glance)
[![Domain: Indian IPR law](https://img.shields.io/badge/Domain-Indian%20IPR%20law-7C3AED)](#dataset-and-scope)
[![Evaluation: Reproducible](https://img.shields.io/badge/Evaluation-Reproducible-059669)](#reproduce-the-reported-results)
[![Status: Research artifacts](https://img.shields.io/badge/Status-Research%20artifacts-EA580C)](#repository-guide)

**M. Mohaiminul Islam** · **Muhammad Abulaish**<br>
Department of Computer Science and Engineering, South Asian University, New Delhi, India

[Quick start](#quick-start) · [Results](#research-at-a-glance) · [Reproduce](#reproduce-the-reported-results) · [Paper](papers/retrieval_reference/retrieval-reference.pdf) · [Citation](#citation)

</div>

---

## Why this repository exists

Strong answer extraction does not guarantee reliable statutory question
answering. A system may extract an answer well when it is given the annotated
paragraph, but fail to retrieve that paragraph or select it from several
plausible candidates.

This repository accompanies a diagnostic study of that gap in IndicIPR-QA,
an extractive question-answering benchmark over seven Indian intellectual
property statutes. It separates three annotation-relative pipeline stages:

```text
Question → candidate retrieval → source selection → answer extraction
                 ↓                    ↓                   ↓
          source absent        other source chosen    span mismatch
```

The study asks how a paragraph-only representation changes retrieval, whether
query-side statutory locators help recover the annotated source, and which
errors remain after source recovery improves. Section filtering is a controlled
diagnostic, not a proposed universal retrieval architecture.

## Research at a glance

| Diagnostic result | Value |
|---|---:|
| Fixed reader F1 with annotated paragraph | **87.37** |
| Fixed reader F1 after paragraph-only BM25 top-5 | **31.95** |
| Paragraph-only BM25 recall@5 | **22.67%** |
| Title-plus-paragraph BM25 recall@5 | **56.98%** |
| Section-filter recall@5 | **86.34%** |
| Section-filter end-to-end F1 | **57.99** |
| Title-aware neural retrieval recall@5 | **83.72%** |
| Section filter versus title-aware BM25 F1 difference | **+12.92** points, 95% cluster interval **[6.62, 19.44]** |

The preserved experiments include sparse retrieval, zero-shot dense retrieval,
RRF, cross-encoder reranking, BGE-M3 retrieval, BGE reranking, title-aware
controls, random-pool controls, reference interventions, decoder sensitivity,
source-selection analyses, a human legal-adequacy audit, and a UK retrieval
comparison.

<p align="center">
  <img src="papers/retrieval_reference/plots/01_retrieval_diagnostic_overview.png" width="920" alt="Overview of the retrieval diagnostic" />
</p>

<p align="center"><em>Operational overview of the retrieval diagnostic. See the paper for the complete protocol and limitations.</em></p>

## Dataset and scope

| Item | Value |
|---|---:|
| Question-answer pairs | **4,462** |
| Paragraph-level statutory contexts | **1,116** |
| Statutes | **7 Indian IPR Acts** |
| Train / validation / test questions | **3,710 / 408 / 344** |
| Human legal-adequacy audit | **80 stratified cases** |

**Dataset:** [IndicIPRExtractiveQA on Hugging Face](https://huggingface.co/datasets/niloycste68/IndicIPRExtractiveQA) (`niloycste68/IndicIPRExtractiveQA`).

The repository contains fixed corpus and split files for exact results-based
reproduction. The findings are benchmark-specific: they measure recovery of a
retained annotated source, not unique legal correctness or a universal limit on
legal retrieval. Questions are generated and curated, so their frequency of
explicit statutory locators may differ from naturally occurring legal requests.

## Quick start

Run the commands from the repository root.

```powershell
git clone https://github.com/niloycste/Retrieval-and-Source-Selection-Failures-in-Indian-Statutory-Question-Answering.git
Set-Location Retrieval-and-Source-Selection-Failures-in-Indian-Statutory-Question-Answering
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Reproduce the reported results

Choose the smallest path that answers your question.

| Reproduction level | Command | Models required? | What it checks or produces |
|---|---|:---:|---|
| Release integrity | `python verify_release.py` | No | Fixed test size, required artifacts, headline values |
| Numerical audit | `python scripts/04_retrieval/audit_retrieval_manuscript.py` | No | Tables, counts, and bootstrap intervals from saved evidence |
| Figure regeneration | Commands below | No | Paper figures from canonical result files |
| Retrieval reruns | Scripts in `scripts/04_retrieval/` | Yes | Re-ranks the corpus with public models |
| Reader inference | Reader-enabled scripts | Yes | Re-runs fixed-reader QA predictions |

### 1. Verify the release and paper numbers

```powershell
python verify_release.py
python scripts/04_retrieval/audit_retrieval_manuscript.py
```

These results-only checks do not download models. They verify the saved
item-level evidence and the principal manuscript values.

### 2. Regenerate the manuscript figures

```powershell
python scripts/08_figures/make_retrieval_paper_analysis_figures.py
python scripts/08_figures/make_retrieval_decoder_diagram.py
```

Generated files are written to `papers/retrieval_reference/plots/`.

### 3. Re-run retrieval or source-selection controls

Dense retrieval, reranking, and reader inference download public Hugging Face
models on first use. GPU acceleration is recommended for full inference but is
not needed for the results-only audits.

```powershell
python scripts/04_retrieval/dense_retrieval_experiment.py
python scripts/04_retrieval/cross_encoder_rerank_experiment.py
python scripts/04_retrieval/citation_aware_retrieval_experiment.py --with-reader
python scripts/04_retrieval/title_neural_end_to_end_control.py --device auto
```

The externally licensed UK-StatuteCorpus and embedding caches are not
redistributed. See `scripts/04_retrieval/uk_statutecorpus_external_validation.py`
and `docs/RETRIEVAL_EXTERNAL_VALIDATION_PROTOCOL.md` for that comparison.

## Repository guide

```text
.
├── data/
│   ├── processed/indic_ipr_qa/    # Released paragraph corpus
│   └── splits/                    # Fixed train, validation, and test splits
├── docs/                          # Protocols, provenance, and audit notes
├── outputs/
│   └── human_validation_report/   # Verified aggregate and de-identified audit data
├── papers/retrieval_reference/    # LaTeX manuscript, bibliography, figures, PDF
├── results/experiments/retrieval/ # Canonical results and item-level evidence
├── scripts/
│   ├── 04_retrieval/              # Retrieval, reranking, and audit programs
│   ├── 08_figures/                # Figure generators
│   └── qa_evaluation.py            # Shared EM/F1 and span evaluation
├── requirements.txt                # Pinned Python environment
└── verify_release.py               # Fast integrity check
```

`docs/RESULT_PROVENANCE.md` maps the paper claims to the saved artifacts.
`ARTIFACT_MANIFEST.md` records the contents and exclusions of this release.

## Human legal-adequacy assessment

Two law graduate students independently assessed 80 stratified section-filter
cases. Original agreement, later consensus, and the annotation-relative
retrieval measures are kept separate.

Publicly available:

- `outputs/human_validation_report/manuscript_summary.json`: verified aggregate
  results, agreement coefficients, hashes, and calculation checks.
- `outputs/human_validation_report/anonymized_item_labels.jsonl`: de-identified
  original categorical ratings and consensus labels for reanalysis.
- `outputs/retrieval_legal_expert_audit/`: the public codebook and sampling
  manifest.

Not released: annotator-identifying workbooks, free-text rationales, authority
notes, dates, and the private passage-order decoding key.

## Citation

If you use this repository, its experiments, or the retrieval-diagnostic
framework, please cite the accompanying paper:

```bibtex
@misc{islam_abulaish_2026_retrieval_source_selection,
  author = {Islam, M. Mohaiminul and Abulaish, Muhammad},
  title = {Beyond Answer Extraction: Diagnosing Retrieval and Source Selection
           Failures in Indian Statutory Question Answering},
  year = {2026},
  note = {Manuscript and reproducibility artifacts},
  url = {https://github.com/niloycste/Retrieval-and-Source-Selection-Failures-in-Indian-Statutory-Question-Answering}
}
```

For the benchmark itself, cite the separate IndicIPR-QA dataset/resource paper
once its proceedings record is available. The anonymous citation in the
double-anonymous manuscript should be replaced only in the de-anonymized final
version.

---

<div align="center">

**Statutory retrieval · Source selection · Extractive QA · Legal NLP**

</div>
