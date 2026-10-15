# Legal-expert audit codebook

## Purpose

This audit evaluates legal relevance and equivalence independently of the
benchmark's annotated-source identity. Benchmark agreement must not be used as
a substitute for legal judgement.

## Procedure

1. Two legally qualified annotators independently review every case. Do not
   share labels or discuss cases until both files are frozen.
2. Complete Phase 1 before opening Phase 2. Phase 1 contains only the question
   and measures whether the Act, provision, and answer are identifiable as
   written.
3. Phase 2 presents two sources in random A/B order. Do not inspect the private
   decoding key. Consult an authoritative version of the relevant statute when
   necessary and record the citation or URL under `authorities_or_notes`.
4. Allowed Phase 1 labels:
   * `act_identifiability` and `provision_identifiability`: `Unique`,
     `Multiple plausible`, `Not identifiable`, or `Unclear`.
   * `answerable_as_written`: `Yes`, `No`, or `Unclear`.
5. Allowed Phase 2 labels:
   * source support, answer correctness, and answer completeness: `Fully`,
     `Partly`, `No`, or `Unclear`.
   * source equivalence: `Yes`, `Partly`, `No`, or `Unclear`.
   * error materiality: `None`, `Low`, `Moderate`, `High`, or `Unclear`.
6. Give confidence from 1 (very uncertain) to 5 (very certain), with a concise
   legal rationale. After independent labels are frozen, calculate raw
   agreement and an agreement coefficient for each categorical item. Because
   prevalence may be imbalanced, report Gwet's AC1 alongside Cohen's kappa;
   use weighted kappa for ordered materiality labels.
7. Send disagreements to a third qualified expert, or adjudicate by consensus
   while retaining both original labels and a written rationale.

## Sampling

The sample uses seed 42 and includes an equal number from each of four
mutually exclusive stage outcomes under the section-filter top-five pipeline.
The exact-match group is a positive control, not an error category. The sample
is designed to estimate audit outcomes within these sampled strata, not the
unweighted prevalence of legal errors in the full test set. Population-level
estimates require stratum weights from `sampling_manifest.json`.

## Before manuscript use

Confirm institutional requirements for expert annotation, consent, and ethics
review before data collection. Do not retain annotator names in the public
artifact. Report annotator qualifications, independent annotation procedure,
agreement, adjudication, and sampling limitations. Do not replace uncertainty
labels with inferred values.
