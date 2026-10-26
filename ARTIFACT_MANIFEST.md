# Artifact manifest

This folder was assembled from the research workspace by
`scripts/utils/build_paper_repositories.ps1` in the parent workspace.

- `papers/`: submission manuscript and rendered figures.
- `scripts/`: executable retrieval, evaluation, audit, and plotting code.
- `data/`: the fixed IndicIPR-QA corpus and splits required by the code.
- `results/`: canonical summaries and saved item-level evidence. Large derived
  embedding caches are excluded; the public validator checks every released
  ranking and per-query metric, and can additionally check regenerated caches.
- `docs/`: analysis locks, external-validation protocol, provenance, and
  numerical review records.
- `outputs/human_validation_report/`: verified aggregate results from the
  completed 80-case audit and de-identified item-level categorical labels.
  These public records are sufficient to recompute every agreement statistic,
  consensus count, sampling check, and human-audit manuscript table. Named
  source workbooks, free-text notes, dates, and the private passage-order
  decoding key are not included because they are not required for numerical
  reproduction.
- `outputs/retrieval_legal_expert_audit/`: public codebook and sampling manifest
  only. Obsolete blank forms and all private decoding material are excluded.

No credential file, private human-review record, local model checkpoint,
optimizer state, or proprietary model download is included.
