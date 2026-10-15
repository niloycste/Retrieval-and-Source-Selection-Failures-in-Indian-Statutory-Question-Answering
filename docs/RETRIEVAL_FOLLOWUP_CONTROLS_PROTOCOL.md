# Retrieval Follow-up Controls Protocol

Status: frozen before execution  
Authorisation date: 2026-09-06  
Seed: 42

## Scope amendment

The repository originally froze the IndicIPR-QA retrieval experiments and the
2026-09-03 UK protocol prohibited citation interventions. On 2026-09-06 the
user explicitly authorised pausing the DAPT study and implementing the
automatic follow-up controls described in the supplied review plan. This file
records that amendment before the new code is executed.

The prior canonical results and cached rankings remain unchanged. Follow-up
outputs are written to new run directories and must be labelled diagnostic or
exploratory until incorporated into the manuscript.

## Frozen inputs

- Indian test set: `data/splits/test.jsonl`
- Indian corpus: `data/processed/indic_ipr_qa/unique_contexts.jsonl`
- Existing reader predictions:
  `results/experiments/retrieval/runs/citation_aware_v1/`
- UK source hashes and paths: the `source` block in
  `results/experiments/retrieval/runs/uk_statutecorpus_external_v1/manifest.json`
- Existing UK rankings and metrics:
  `results/experiments/retrieval/runs/uk_statutecorpus_external_v1/`

Original questions, qrels, passage IDs, predictions, and rankings will not be
overwritten. Query variants will be stored in separate fields and files.

## Indian controls

1. Export an item-level statutory-reference and collision audit using the
   existing section-expression rule. Multiple references are reported
   separately. Query-side Act detection uses a documented alias dictionary.
2. Join structural groups to canonical BM25, section-filter, and saved reader
   outcomes. Collision-size bins are fixed in advance as 2--5, 6--10, and 11+
   matching passages.
3. Compare global BM25 with an unchanged-query oracle-Act candidate
   restriction. Apply the same restriction to section-filter retrieval. Scores
   come from the same globally fitted BM25 model; only the eligible candidate
   pool changes.
4. Export actual paragraph/title/Act representations for input auditing.
5. Calculate the exact expected Indian recall@5 for uniform ordering within a
   section-matched pool. For the no-match fallback, retain canonical BM25.
6. Audit BM25 ties per item. Estimate recall variation by randomising only the
   tied group crossing the requested cutoff, using 10,000 deterministic seeds.
7. Export item-level boost/filter candidate and reader-selection traces.

The canonical tie-break remains descending score followed by ascending corpus
index, because changing it would change a frozen baseline. Tie sensitivity is
reported explicitly rather than silently redefining that baseline.

## UK controls

1. Replace the broad text-only regex count with an item-level parser audit.
   Instrument-title years such as "Regulations 1989" are not provision
   locators.
2. Resolve only supported provision types against actual corpus metadata:
   section, regulation, article, rule, paragraph, and schedule. Unsupported or
   conflicting cases remain explicit.
3. Disaggregate existing saved rankings into supported-resolved,
   parsed-unresolved, and no-recognised-reference groups without changing qrels
   or metric definitions.
4. For supported reference-bearing queries, create a reference-removed query
   and rerun canonical BM25 only. The original query remains stored alongside
   the variant. Near-empty variants are flagged.
5. Structural UK retrieval is conditional on reliable deterministic mapping
   and sufficient eligible coverage. It will not be run merely to manufacture
   a matching external result.

Dense and cross-encoder intervention reruns are not part of the first execution
because they require model inference. They may be added as a separately timed
extension after the automatic audits are validated.

## Interpretive limits

Automatic collision counts establish structural competition, not human-judged
semantic ambiguity, legal correctness, or unique answerability. UK subgroup
analysis is descriptive. Reference removal measures sensitivity to the removed
surface form and does not prove that the modified query preserves the original
information need.
