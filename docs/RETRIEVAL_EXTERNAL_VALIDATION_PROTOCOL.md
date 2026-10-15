# UK-StatuteCorpus External Retrieval Validation Protocol

**Frozen:** 2026-09-03, before inspecting any UK evaluation result.

## Scope

This is a retrieval-only external validation for Paper B. It asks whether the
relative behaviour of the paper's fixed content-retrieval pipeline transfers
from Indian IPR statutes to an independently constructed English UK statutory
retrieval benchmark. It does not evaluate extractive QA, the E2' reader, or the
generality of the IndicIPR-QA citation-intervention effect.

## Resource

- Dataset: UK-StatuteCorpus v1.0 / Eval100.
- Record of source: https://collections.durham.ac.uk/files/r14x51hj064
- Download: https://collections.durham.ac.uk/downloads/r14x51hj064
- Expected archive filename: `uk-statutecorpus_v1.0.zip`.
- Depositor-provided MD5: `29907ffa565a5e421c851ab2bc661423`.
- Released corpus file:
  `corpus/uk-statutecorpus_passages_2020-2024_v1.0.jsonl.gz`.
- Released evaluation file:
  `evaluation_dataset/uk-statutecorpus_eval100_2020-2024_v1.0.jsonl`.
- Corpus: 124,796 passages.
- Evaluation: 100 queries with three graded relevant passages per query
  (`rel=3`, `rel=2`, and `rel=1`) and rationales.

The downloaded archive, extracted corpus, and query file must receive locally
computed SHA-256 hashes before evaluation.

## Fixed systems

The systems are selected from the already completed IndicIPR-QA retrieval
study, without consulting UK results.

1. BM25: `k1=1.5`, `b=0.75`, lowercase alphanumeric tokenisation.
2. Dense: `intfloat/e5-small-v2`, with `query: ` and `passage: ` prefixes,
   normalised embeddings, cosine similarity, no UK fine-tuning, and pinned
   revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`.
3. Hybrid: reciprocal-rank fusion of BM25 and E5, `RRF k=60`.
4. Re-ranked BM25: `cross-encoder/ms-marco-MiniLM-L-6-v2` at pinned revision
   `233902d25c440f23af6f7d6e94d2946bac0bee0a` over top 50.
5. Re-ranked hybrid: the same cross-encoder over hybrid top 50.

No alternative model, query rewrite, parameter, prompt, or preprocessing choice
may be selected using Eval100 effectiveness.

## Metrics and judgements

- Primary: nDCG@10 using the released graded relevance labels.
- Secondary: MRR@10, Recall@5, Recall@10, Recall@50.
- Grade-sensitive: recall of `rel=3` at 5 and 10.
- Uncertainty: paired query bootstrap, seed 42, 10,000 resamples, percentile
  95% confidence intervals for metrics and selected system differences.

All passages with positive released relevance are relevant for ordinary recall
and MRR. nDCG uses the released grades without binarisation. Scores are compared
within UK-StatuteCorpus; raw IndicIPR and UK scores must not be treated as
directly comparable because corpus sizes and relevance schemes differ.

## Reference audit

Before effectiveness results are computed, apply a deterministic English legal
locator audit to the query text and save query-level labels. Report counts only.
Do not add citations derived from gold labels. Do not run deletion, citation-only,
or wrong-citation interventions on synthetic query variants.

## Reproducibility and stopping

- Save a manifest containing data/model identifiers, file hashes, package
  versions, fixed seeds, thread cap, timestamps, and exact command line.
- Cache BM25 and dense rankings so interruption can be resumed without corpus
  re-encoding.
- Save JSON and CSV summaries plus per-query rankings/metrics.
- Stop after the five fixed systems, reference audit, confidence intervals, and
  paper-ready table. Do not add retrievers or rerankers after viewing results.
