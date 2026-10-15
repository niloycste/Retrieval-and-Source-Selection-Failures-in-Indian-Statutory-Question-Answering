# Section pool and predicted Act controls

This protocol is written before executing this extension. It is not a
preregistration: the benchmark and earlier test results are already known.
This is retrieval-only work. No new reader F1 or legal adequacy result is implied.

## Fixed design

- Reuse the released corpus and section-disjoint train/validation/test splits.
- Preserve canonical BM25 (global corpus IDF, k1=1.5, b=0.75), tokenisation,
  first-reference parser, and corpus-index tie breaking. Do not refit BM25
  statistics within each candidate pool.
- Recompute paragraph BM25, title+paragraph BM25, and section+BM25.
- Random section ordering: report analytic expected recall and 10,000 random
  order simulations. For unparsed/empty section pools use canonical BM25.
- Size-only control: sample the same number of passages uniformly from the
  full corpus, then order uniformly. Expected R@5 is min(5,pool size)/N.
  This isolates arbitrary size reduction from informative section matching.
- Wrong section: for seeds 7,42,99 replace the first parsed section with a
  uniformly selected different section present in corpus titles. Do not use
  the test Act or gold section to choose replacements. Replace query text as
  well as the pool selector. Unparsed questions use unchanged BM25 fallback.
  A corrupted locator may accidentally equal the annotated section when the
  original parser disagrees with it; retain and report these cases.
- Train one fixed question-only TF-IDF word unigram/bigram + logistic regression
  Act classifier (C=1, max_iter=2000, solver=lbfgs, no class weighting, seed 42).
  Before fitting, remove any training example whose normalised question string
  exactly occurs in either held-out split. This uses held-out question text only
  to prevent duplicate-input leakage; it uses no held-out labels, answers,
  contexts, titles, IDs or gold Acts. Fit vocabulary, IDF and the classifier on
  the remaining training questions and Act labels. Do not tune on validation or
  test, and do not refit with validation examples.
- Evaluate validation first, then test with the unchanged classifier.
- Predicted Act + section: intersect predicted Act with parsed section; for
  unparsed questions filter only by predicted Act. The primary hard filter
  returns an empty list if the intersection is empty. A separately named
  fallback variant uses the section-only pool if the intersection is empty.
- Oracle Act + section is a labelled diagnostic ceiling, not a deployable
  system. For unparsed questions restrict to the gold Act.

## Reporting and safeguards

Report R@1/3/5/10, MRR@10, candidate access, empty-pool counts, Act accuracy,
macro F1, confusion matrices, zero-residual/content-bearing subgroup scores,
and test results by Act. Report paired 95% bootstrap intervals over source
Act-section clusters (10,000 resamples, seed 42) for ranking vs expected random
order and predicted Act vs section-only BM25. Random-order spread is not a
confidence interval over questions. Preserve all original artifacts. Write
new outputs to a separate directory that must not already exist. Save input
hashes, package versions, model, validation/test predictions and per-query
candidate pools/rankings. Independently recompute metrics from saved rankings.

A gain is not guaranteed. Identical or nearly identical questions can concern
different Acts. Audit split overlap and report limitations. Wrong-section
failure is partly guaranteed by hard filtering, not proof of learned reasoning.
These new controls do not establish performance on independently written queries.
