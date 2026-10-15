"""Canonical inference and scoring utilities for extractive QA.

All reader evaluations use the same protocol:

* tokenize ``[CLS] question [SEP] context [SEP]`` with context-only
  truncation;
* split long contexts into overlapping windows;
* search every valid ``start <= end`` context span in every window; and
* return the literal character slice from the source context with the highest
  summed start/end logit.

Keeping this logic in one module prevents the test-only and split-wise
evaluators from silently using different decoding rules.
"""

from __future__ import annotations

import collections
import string
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class QADecodingConfig:
    max_seq_length: int = 384
    doc_stride: int = 128
    batch_size: int = 16
    max_answer_tokens: int | None = None

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            "input_format": "[CLS] question [SEP] context [SEP]",
            "max_seq_length": self.max_seq_length,
            "doc_stride": self.doc_stride,
            "batch_size": self.batch_size,
            "max_answer_tokens": self.max_answer_tokens,
            "truncation": "only_second",
            "span_decoding": "exhaustive constrained start<=end",
            "window_selection": "maximum summed start/end logit",
            "prediction_text": "literal source-context character slice",
        }


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_em_f1(pred: str, gold: str) -> tuple[float, float]:
    pred_n = normalize(pred)
    gold_n = normalize(gold)
    em = float(pred_n == gold_n)
    pred_tokens = pred_n.split()
    gold_tokens = gold_n.split()
    if not pred_tokens or not gold_tokens:
        return em, 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return em, 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return em, 2 * precision * recall / (precision + recall)


def compute_char_f1(pred: str, gold: str) -> float:
    """Return an order-insensitive character-frequency overlap diagnostic."""
    pred_chars = collections.Counter(normalize(pred).replace(" ", ""))
    gold_chars = collections.Counter(normalize(gold).replace(" ", ""))
    common = pred_chars & gold_chars
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / max(sum(pred_chars.values()), 1)
    recall = same / max(sum(gold_chars.values()), 1)
    return 2 * precision * recall / (precision + recall)


def best_span_from_logits(
    *,
    context: str,
    offset_mapping: Sequence[Sequence[int]],
    sequence_ids: Sequence[int | None],
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    max_answer_tokens: int | None = None,
) -> tuple[str, float]:
    """Return the highest-scoring valid source-context span for one window."""

    context_positions = [
        idx
        for idx, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
        and offset_mapping[idx]
        and offset_mapping[idx][1] > offset_mapping[idx][0]
    ]
    if not context_positions:
        return "", float("-inf")

    positions = torch.as_tensor(context_positions, dtype=torch.long)
    starts = start_logits[positions].float()
    ends = end_logits[positions].float()
    joint_scores = starts[:, None] + ends[None, :]

    span_count = len(context_positions)
    start_order = torch.arange(span_count)[:, None]
    end_order = torch.arange(span_count)[None, :]
    valid = end_order >= start_order
    if max_answer_tokens is not None:
        valid &= (end_order - start_order + 1) <= max_answer_tokens
    joint_scores = joint_scores.masked_fill(~valid, float("-inf"))

    flat_index = int(torch.argmax(joint_scores).item())
    start_offset, end_offset = divmod(flat_index, span_count)
    start_index = context_positions[start_offset]
    end_index = context_positions[end_offset]
    char_start = int(offset_mapping[start_index][0])
    char_end = int(offset_mapping[end_index][1])
    return context[char_start:char_end].strip(), float(joint_scores[start_offset, end_offset])


def predict_answers(
    tokenizer,
    model,
    device: torch.device,
    questions: Sequence[str],
    contexts: Sequence[str],
    config: QADecodingConfig | None = None,
) -> list[tuple[str, float]]:
    """Predict literal answer spans for a batch of question-context pairs."""

    if len(questions) != len(contexts):
        raise ValueError("questions and contexts must have the same length")
    if not questions:
        return []

    config = config or QADecodingConfig()
    encoded = tokenizer(
        list(questions),
        list(contexts),
        return_tensors="pt",
        truncation="only_second",
        max_length=config.max_seq_length,
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
    )
    feature_count = len(encoded["input_ids"])
    sequence_ids = [encoded.sequence_ids(index) for index in range(feature_count)]
    offset_mappings = encoded.pop("offset_mapping").tolist()
    sample_mapping = encoded.pop("overflow_to_sample_mapping").tolist()

    best_predictions = [("", float("-inf")) for _ in questions]
    model.eval()
    for batch_start in range(0, feature_count, config.batch_size):
        batch_end = min(batch_start + config.batch_size, feature_count)
        inputs = {
            key: value[batch_start:batch_end].to(device)
            for key, value in encoded.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)
        start_batch = outputs.start_logits.detach().cpu()
        end_batch = outputs.end_logits.detach().cpu()

        for local_index, feature_index in enumerate(range(batch_start, batch_end)):
            sample_index = int(sample_mapping[feature_index])
            text, score = best_span_from_logits(
                context=contexts[sample_index],
                offset_mapping=offset_mappings[feature_index],
                sequence_ids=sequence_ids[feature_index],
                start_logits=start_batch[local_index],
                end_logits=end_batch[local_index],
                max_answer_tokens=config.max_answer_tokens,
            )
            if score > best_predictions[sample_index][1]:
                best_predictions[sample_index] = (text, score)

    return best_predictions


def predict_answer(
    tokenizer,
    model,
    device: torch.device,
    question: str,
    context: str,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    max_answer_tokens: int | None = None,
    _n_best: int | None = None,
) -> tuple[str, float]:
    """Single-example compatibility wrapper used by retrieval experiments."""

    config = QADecodingConfig(
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        batch_size=16,
        max_answer_tokens=max_answer_tokens,
    )
    return predict_answers(tokenizer, model, device, [question], [context], config)[0]
