# Paper split and experiment ownership

This repository separates three scientific products. The separation is by
research question and experiment, not only by manuscript section.

## Companion benchmark paper (IndicIPR-QA)

The accepted resource paper owns benchmark construction, question generation,
curation, human validation, dataset statistics as a contribution, and the
benchmark's release protocol. The retrieval paper cites this work and reports
only the dataset facts necessary to interpret retrieval experiments.

## Paper A: statutory DAPT

Paper A owns all experiments whose manipulated variable is reader pretraining:
authentic, synthetic, and mixed DAPT corpora; seed extraction and synthetic
generation; corpus filtering; MLM diagnostics; matched-corpus ablations;
multi-seed DAPT comparisons; checkpoint-selection audits; and claims about
pretraining composition, variance, or language-model fit.

These experiments are deliberately absent from the retrieval manuscript except
for the identity and gold-context score of the fixed reader used as a measuring
instrument.

## Paper B: statutory-reference retrieval diagnosis

Paper B is located in `papers/retrieval_reference/` and owns:

- the gold-context versus retrieved-context performance gap;
- BM25, zero-shot dense, sparse--dense hybrid, and cross-encoder retrieval;
- the deterministic citation-template-only partition and its sensitivity audit;
- per-Act retrieval diagnostics;
- section-index retrieval, boosting, and filtering;
- end-to-end evaluation with section-filtered candidates;
- citation removal, isolation, and wrong-reference interventions;
- retrieval failure decomposition and benchmark-design implications.

The retrieval experiments are frozen. New retrievers or new external-validation
experiments require a separate decision and must not silently replace the
canonical results used by this manuscript.

## Shared factual material

The papers may share minimal facts needed for self-containment: dataset size,
split size, statutory coverage, the fixed reader's identity, and the 87.37
gold-context F1 anchor. Shared facts must cite the companion benchmark paper or
the appropriate result artifact. Substantive experiment tables, extended
methods, and contribution claims remain with their owning paper.
