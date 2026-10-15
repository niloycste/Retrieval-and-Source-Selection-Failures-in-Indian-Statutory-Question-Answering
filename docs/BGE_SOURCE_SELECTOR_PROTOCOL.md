# Independent BGE source-selection control

This control tests source selection after section-based candidate retrieval.
It is separate from the reader's span-logit source choice.

## Fixed pipeline

1. Parse the first section number from the question.
2. Use canonical paragraph BM25 to rank passages with that section number.
   Keep at most five passages. If no section is parsed, use the canonical BM25
   top five fallback.
3. For every unchanged candidate passage, score the pair of the full question
   and `section title + paragraph` with
   `BAAI/bge-reranker-v2-m3` at revision
   `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` and maximum length 512.
4. Select the highest-scoring passage, breaking an exact score tie by its
   order in the original section-filter pool.
5. Extract an answer only from the selected passage using the frozen
   LegalBERT--SQuAD--IPR reader cache. The reader's raw span score is not used
   to choose a passage.

The BGE reranker receives no gold paragraph identity, answer, Act label, or
human audit label. The comparison retains the same candidate pool, so it tests
source selection rather than candidate retrieval.

## Reported measures

- candidate recall@5, which must remain unchanged;
- selected annotated source percentage;
- selected annotated source percentage conditional on the paragraph being in
  the candidate pool;
- normalised EM, token F1, and annotation-grounded F1; and
- four pipeline-stage counts.

All measures are relative to the retained benchmark annotation. They do not
measure legal relevance or establish that the annotation is the only legally
adequate source.

## Reproduction

Run the experiment with local, pinned model snapshots:

```powershell
python scripts/04_retrieval/metadata_neural_and_selector_control.py --device cuda
```

The script caches rankings and supports restart. It requires the local model
snapshots prepared by `modern_retrieval_extension.py` and the frozen reader
candidate cache. It does not train a model. Independently recompute all saved
selector totals with:

```powershell
python scripts/04_retrieval/validate_metadata_neural_selector.py
```
