"""
Journal experiments for IndicIPR-QA:

1. Retrieval + QA
   Question -> BM25 retrieves top-k statute paragraphs -> QA model extracts
   the answer from retrieved paragraph(s).

2. Transfer / zero-shot external comparison
   - SQuAD-only model -> IndicIPR-QA test set
   - SQuAD + IndicIPR-QA adapted models -> IndicIPR-QA test set
   - IndicIPR-QA model -> SQuAD dev set, if a local SQuAD file is provided

The script is intentionally inference-only. It does not train models.

Examples
--------
Run the full default journal experiment with the strongest reader E2':
  python journal_retrieval_transfer_experiments.py

Rebuild only derived tables/examples from an existing completed run:
  python journal_retrieval_transfer_experiments.py --postprocess-only

Quick smoke test:
  python journal_retrieval_transfer_experiments.py --limit 5 --top-k 1 --run-name smoke_test

Optional SQuAD reverse transfer if you have a local SQuAD v1 file:
  python journal_retrieval_transfer_experiments.py --squad-file dev-v1.1.json

"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import math
import os
import re
import string
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


# Fix OpenMP duplicate library warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qa_evaluation import (  # noqa: E402
    QADecodingConfig,
    best_span_from_logits as canonical_best_span_from_logits,
    compute_char_f1 as canonical_compute_char_f1,
    compute_em_f1 as canonical_compute_em_f1,
    normalize as canonical_normalize,
    predict_answer as canonical_predict_answer,
)

IPR_TEST_FILE = ROOT / "data" / "splits" / "test.jsonl"
IPR_FINAL_FILE = ROOT / "data" / "processed" / "indic_ipr_qa" / "merged_ip_qa.jsonl"
BASE_OUTPUT_DIR = ROOT / "results" / "experiments" / "retrieval"

CPRIME_MODEL = ROOT / "artifacts" / "models" / "squad_v1" / "squad_v1_modelc_v2" / "ipr_final" / "final"
E2PRIME_MODEL = ROOT / "artifacts" / "models" / "squad_v1" / "baseline_e2_squadv1" / "final"
DIRECT_A_MODEL = ROOT / "artifacts" / "models" / "direct" / "baseline_a" / "final"
DEFAULT_IPR_MODEL = E2PRIME_MODEL
DEFAULT_SQUAD_MODEL = (
    ROOT / "artifacts" / "models" / "squad_v1" / "baseline_b_squadv1" / "squad" / "final"
)

DEFAULT_IPR_MODELS = [
    ("E2': LegalBERT-US+SQuAD+IPR", E2PRIME_MODEL),
    ("C': MLM+SQuAD+IPR InLegalBERT", CPRIME_MODEL),
    ("A: IPR direct InLegalBERT", DIRECT_A_MODEL),
]

DEFAULT_SQUAD_ONLY_MODELS = [
    (
        "SQuAD-only: InLegalBERT",
        ROOT / "artifacts" / "models" / "squad_v1" / "baseline_b_squadv1" / "squad" / "final",
    ),
    (
        "SQuAD-only: BERT-base",
        ROOT / "artifacts" / "models" / "squad_v1" / "baseline_d2_squadv1" / "squad" / "final",
    ),
    (
        "SQuAD-only: LegalBERT-US",
        ROOT / "artifacts" / "models" / "squad_v1" / "baseline_e2_squadv1" / "squad" / "final",
    ),
    (
        "SQuAD-only: InCaseLawBERT",
        ROOT / "artifacts" / "models" / "squad_v1" / "baseline_f_squadv1" / "squad" / "final",
    ),
]

DEFAULT_ADAPTED_MODELS = [
    (
        "SQuAD+IPR: InLegalBERT",
        ROOT / "artifacts" / "models" / "squad_v1" / "baseline_b_squadv1" / "final",
    ),
    (
        "SQuAD+IPR: BERT-base",
        ROOT / "artifacts" / "models" / "squad_v1" / "baseline_d2_squadv1" / "final",
    ),
    (
        "SQuAD+IPR: LegalBERT-US",
        ROOT / "artifacts" / "models" / "squad_v1" / "baseline_e2_squadv1" / "final",
    ),
    (
        "SQuAD+IPR: InCaseLawBERT",
        ROOT / "artifacts" / "models" / "squad_v1" / "baseline_f_squadv1" / "final",
    ),
    (
        "C': MLM+SQuAD+IPR InLegalBERT",
        CPRIME_MODEL,
    ),
    (
        "MLM+SQuAD+IPR: Orig-only InLegalBERT",
        ROOT / "artifacts" / "models" / "squad_v1" / "original_only_squadv1" / "ipr_final" / "final",
    ),
    (
        "MLM+SQuAD+IPR: Synth-only InLegalBERT",
        ROOT / "artifacts" / "models" / "squad_v1" / "squad_v1_synth_only" / "ipr_final" / "final",
    ),
]

LLM_BASELINE_FILES = [
    ROOT / "results" / "experiments" / "llm_baselines" / "llm_baseline_results_qwen.json",
    ROOT / "results" / "experiments" / "llm_baselines" / "llm_baseline_results_mistral.json",
    ROOT / "results" / "experiments" / "llm_baselines" / "llm_baseline_results_llama3.1_latest.json",
    ROOT / "results" / "experiments" / "llm_baselines" / "llm_baseline_results_llama3.1.json",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class QARecord:
    id: str
    question: str
    context: str
    answer: str
    title: str = ""
    act: str = ""
    gold_answers: tuple[str, ...] = ()


@dataclass
class ContextRecord:
    id: str
    context: str
    title: str = ""
    act: str = ""


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("--run-name cannot be empty after sanitization.")
    return cleaned


def choose_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.run_name:
        return BASE_OUTPUT_DIR / "runs" / safe_run_name(args.run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BASE_OUTPUT_DIR / "runs" / f"run_{timestamp}"


def write_run_manifest(args: argparse.Namespace) -> None:
    out = {
        "mode": args.mode,
        "output_dir": str(args.output_dir),
        "ipr_test_file": args.ipr_test_file,
        "limit": args.limit,
        "qa_model": args.qa_model,
        "qa_label": args.qa_label,
        "corpus_file": args.corpus_file,
        "allow_split_corpus_fallback": args.allow_split_corpus_fallback,
        "top_k": args.top_k,
        "squad_file": args.squad_file,
        "download_squad": args.download_squad,
        "squad_limit": args.squad_limit,
        "include_adapted": args.include_adapted,
        "all_default_models": args.all_default_models,
        "postprocess_only": args.postprocess_only,
        "reuse_complete_predictions": args.reuse_complete_predictions,
        "device": args.device,
        "max_seq_length": args.max_seq_length,
        "doc_stride": args.doc_stride,
        "max_answer_tokens": args.max_answer_tokens,
        "n_best": args.n_best,
    }
    path = Path(args.output_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(text: str) -> str:
    return canonical_normalize(text)


def compute_em_f1(pred: str, gold: str) -> tuple[float, float]:
    return canonical_compute_em_f1(pred, gold)


def compute_char_f1(pred: str, gold: str) -> float:
    return canonical_compute_char_f1(pred, gold)


def metric_summary(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0, "char_f1": 0.0}
    return {
        "n": n,
        "em": round(sum(row["em"] for row in rows) / n * 100, 4),
        "f1": round(sum(row["f1"] for row in rows) / n * 100, 4),
        "char_f1": round(sum(row["char_f1"] for row in rows) / n * 100, 4),
    }


def load_ipr_records(path: Path, limit: int | None = None) -> list[QARecord]:
    records: list[QARecord] = []
    for row in load_jsonl(path):
        records.append(
            QARecord(
                id=str(row.get("id", "")),
                question=str(row.get("question", "")),
                context=str(row.get("context", "")),
                answer=str(row.get("answer", "")),
                title=str(row.get("title", "")),
                act=str(row.get("act", "")),
                gold_answers=(str(row.get("answer", "")),),
            )
        )
        if limit and len(records) >= limit:
            break
    return records


def qa_record_from_squad_row(row: dict, fallback_id: str) -> QARecord | None:
    answers = row.get("answers", {})
    if isinstance(answers, dict):
        texts = answers.get("text", [])
    elif isinstance(answers, list):
        texts = [answer.get("text", "") for answer in answers if answer.get("text", "")]
    else:
        texts = []
    texts = list(dict.fromkeys(str(text) for text in texts if str(text).strip()))
    answer = texts[0] if texts else ""
    if not answer:
        return None
    return QARecord(
        id=str(row.get("id", fallback_id)),
        question=str(row.get("question", "")),
        context=str(row.get("context", "")),
        answer=str(answer),
        title=str(row.get("title", "")),
        act="SQuAD",
        gold_answers=tuple(texts),
    )


def load_squad_file(path: Path, limit: int | None = None) -> list[QARecord]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for idx, row in enumerate(load_jsonl(path)):
            record = qa_record_from_squad_row(row, f"squad_{idx}")
            if record:
                records.append(record)
            if limit and len(records) >= limit:
                break
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[QARecord] = []
    if "data" in data:
        for article in data["data"]:
            title = str(article.get("title", ""))
            for paragraph in article.get("paragraphs", []):
                context = str(paragraph.get("context", ""))
                for qa in paragraph.get("qas", []):
                    if qa.get("is_impossible"):
                        continue
                    answers = qa.get("answers", [])
                    if not answers:
                        continue
                    answer_texts = list(
                        dict.fromkeys(
                            str(answer.get("text", ""))
                            for answer in answers
                            if str(answer.get("text", "")).strip()
                        )
                    )
                    if not answer_texts:
                        continue
                    records.append(
                        QARecord(
                            id=str(qa.get("id", f"squad_{len(records)}")),
                            question=str(qa.get("question", "")),
                            context=context,
                            answer=answer_texts[0],
                            title=title,
                            act="SQuAD",
                            gold_answers=tuple(answer_texts),
                        )
                    )
                    if limit and len(records) >= limit:
                        return records
    return records


def load_squad_from_hf(limit: int | None = None) -> list[QARecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading SQuAD from Hugging Face requires the 'datasets' package. "
            "Install it or pass --squad-file dev-v1.1.json instead."
        ) from exc

    try:
        dataset = load_dataset("rajpurkar/squad", split="validation")
    except Exception as exc:
        raise RuntimeError(
            "Could not load SQuAD validation from Hugging Face. "
            "Check internet access/cache, or pass --squad-file dev-v1.1.json."
        ) from exc
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    records: list[QARecord] = []
    for row in dataset:
        record = qa_record_from_squad_row(dict(row), f"squad_{len(records)}")
        if record:
            records.append(record)
    return records


def find_local_squad_file() -> Path | None:
    candidates = [
        ROOT / "dev-v1.1.json",
        ROOT / "squad" / "dev-v1.1.json",
        ROOT / "data" / "dev-v1.1.json",
        ROOT / "archive" / "legacy" / "data" / "discarded_datasets" / "dev-v1.1.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, contexts: list[ContextRecord], k1: float = 1.5, b: float = 0.75):
        self.contexts = contexts
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(ctx.context) for ctx in contexts]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.term_freqs = [collections.Counter(tokens) for tokens in self.doc_tokens]
        doc_freq: collections.Counter[str] = collections.Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))
        n_docs = len(self.doc_tokens)
        self.idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        tf = self.term_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
        return score

    def topk(self, query: str, k: int) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scores = [
            (idx, self.score_doc(query_tokens, idx))
            for idx in range(len(self.contexts))
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]


def build_context_corpus(path: Path, allow_split_fallback: bool = False) -> list[ContextRecord]:
    if path.exists():
        rows = load_jsonl(path)
    else:
        if not allow_split_fallback:
            raise FileNotFoundError(
                f"Corpus file does not exist: {path}. "
                "Pass a valid --corpus-file, or use --allow-split-corpus-fallback "
                "to build the retrieval corpus from train/val/test splits."
            )
        print(
            f"[WARN] Corpus file not found: {path}. "
            "Falling back to train/val/test splits for retrieval corpus."
        )
        rows = []
        for split in ["train", "val", "test"]:
            rows.extend(load_jsonl(ROOT / "data" / "splits" / f"{split}.jsonl"))

    seen: set[str] = set()
    contexts: list[ContextRecord] = []
    for row in rows:
        context = str(row.get("context", "")).strip()
        if not context or context in seen:
            continue
        seen.add(context)
        contexts.append(
            ContextRecord(
                id=f"ctx_{len(contexts):06d}",
                context=context,
                title=str(row.get("title", "")),
                act=str(row.get("act", "")),
            )
        )
    return contexts


def resolve_device(device_arg: str) -> torch.device:
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def is_valid_model_dir(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "path does not exist"
    if not path.is_dir():
        return False, "path is not a directory"
    if not (path / "config.json").exists():
        return False, "missing config.json"
    weight_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
    ]
    if not any((path / name).exists() for name in weight_files):
        return False, "missing model weights"
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "spiece.model",
        "merges.txt",
    ]
    if not any((path / name).exists() for name in tokenizer_files):
        return False, "missing tokenizer files"
    return True, ""


def load_qa_model(model_path: Path, device_arg: str):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BertTokenizer
    import json

    valid, reason = is_valid_model_dir(model_path)
    if not valid:
        raise FileNotFoundError(f"Invalid QA model directory: {model_path} ({reason})")
    device = resolve_device(device_arg)
    
    # Read config to determine model type
    config_path = model_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    model_type = config.get("model_type", "bert")
    
    try:
        # Try loading with explicit model_type
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, model_type=model_type)
    except Exception as e:
        print(f"[WARN] AutoTokenizer failed, using BertTokenizer: {e}")
        try:
            # Use BertTokenizer directly
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e2:
            print(f"[WARN] BertTokenizer also failed: {e2}")
            tokenizer = None
    
    try:
        # Load model with explicit model_type
        model = AutoModelForQuestionAnswering.from_pretrained(
            str(model_path), 
            local_files_only=True,
            model_type=model_type
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print(f"[INFO] Trying with model_type from config: {model_type}")
        # Last resort: load with transformers directly
        from transformers import BertForQuestionAnswering
        model = BertForQuestionAnswering.from_pretrained(str(model_path), local_files_only=True)
    
    model.to(device)
    model.eval()
    return tokenizer, model, device


def best_span_from_logits(
    context: str,
    offset_mapping: list[list[int]],
    sequence_ids: list[int | None],
    start_logits,
    end_logits,
    max_answer_tokens: int | None,
    n_best: int,
) -> tuple[str, float]:
    del n_best  # retained in the public signature for command-line compatibility
    return canonical_best_span_from_logits(
        context=context,
        offset_mapping=offset_mapping,
        sequence_ids=sequence_ids,
        start_logits=start_logits,
        end_logits=end_logits,
        max_answer_tokens=max_answer_tokens,
    )


def predict_answer(
    tokenizer,
    model,
    device,
    question: str,
    context: str,
    max_seq_length: int,
    doc_stride: int,
    max_answer_tokens: int | None,
    n_best: int,
) -> tuple[str, float]:
    return canonical_predict_answer(
        tokenizer,
        model,
        device,
        question,
        context,
        max_seq_length,
        doc_stride,
        max_answer_tokens,
        n_best,
    )


def score_prediction(record: QARecord, pred: str) -> dict:
    gold_answers = record.gold_answers or (record.answer,)
    em = max(compute_em_f1(pred, gold)[0] for gold in gold_answers)
    f1 = max(compute_em_f1(pred, gold)[1] for gold in gold_answers)
    cf1 = max(compute_char_f1(pred, gold) for gold in gold_answers)
    return {
        "id": record.id,
        "question": record.question,
        "gold": record.answer,
        "gold_answers": list(gold_answers),
        "prediction": pred,
        "em": em,
        "f1": f1,
        "char_f1": cf1,
        "act": record.act,
        "title": record.title,
    }


def evaluate_model_on_records(
    records: list[QARecord],
    model_label: str,
    model_path: Path,
    args: argparse.Namespace,
    prediction_path: Path | None = None,
) -> dict:
    if (
        prediction_path
        and args.reuse_complete_predictions
        and prediction_path.exists()
    ):
        existing = load_jsonl(prediction_path)
        if len(existing) >= len(records):
            print(f"Reusing complete predictions for {model_label}: {prediction_path}")
            summary = metric_summary(existing[: len(records)])
            summary.update({"model": model_label, "model_path": str(model_path)})
            return summary
        print(
            f"Existing predictions for {model_label} are incomplete "
            f"({len(existing)}/{len(records)}); rerunning this model."
        )

    tokenizer, model, device = load_qa_model(model_path, args.device)
    predictions: list[dict] = []
    print(f"Evaluating {model_label} on {len(records)} records")
    try:
        for idx, record in enumerate(records, start=1):
            pred, score = predict_answer(
                tokenizer,
                model,
                device,
                record.question,
                record.context,
                args.max_seq_length,
                args.doc_stride,
                args.max_answer_tokens,
                args.n_best,
            )
            row = score_prediction(record, pred)
            row["model_score"] = score
            predictions.append(row)
            if idx % args.progress_every == 0 or idx == len(records):
                summary = metric_summary(predictions)
                print(
                    f"  {idx:>4}/{len(records)} "
                    f"EM={summary['em']:.2f} F1={summary['f1']:.2f} Char-F1={summary['char_f1']:.2f}"
                )
                if prediction_path:
                    write_jsonl(prediction_path, predictions)
    finally:
        del model
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            import torch

            torch.cuda.empty_cache()

    if prediction_path:
        write_jsonl(prediction_path, predictions)

    summary = metric_summary(predictions)
    summary.update({"model": model_label, "model_path": str(model_path)})
    return summary


def write_summary_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_latex_table(path: Path, rows: list[dict], caption: str, label: str, columns: list[tuple[str, str]]) -> None:
    numeric_keys = {"n", "em", "f1", "char_f1", "retrieval_recall", "approx_params_b"}
    alignment = "".join("r" if key in numeric_keys else "l" for key, _ in columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(header for _, header in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def short_text(value: object, max_chars: int = 120) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def write_retrieval_act_breakdown(retrieval_dir: Path, setting_rows: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for setting, predictions in setting_rows.items():
        by_act: dict[str, list[dict]] = collections.defaultdict(list)
        for row in predictions:
            act = str(row.get("act") or row.get("gold_act") or "Unknown").strip() or "Unknown"
            by_act[act].append(row)
        for act in sorted(by_act):
            group = by_act[act]
            summary = metric_summary(group)
            if group and "retrieval_hit_at_k" in group[0]:
                retrieval_recall = round(sum(truthy(row.get("retrieval_hit_at_k")) for row in group) / len(group) * 100, 4)
            else:
                retrieval_recall = 100.0
            summary.update(
                {
                    "setting": setting,
                    "act": act,
                    "retrieval_recall": retrieval_recall,
                }
            )
            rows.append(summary)

    if not rows:
        return rows

    write_summary_csv(
        retrieval_dir / "retrieval_qa_per_act.csv",
        rows,
        ["setting", "act", "n", "retrieval_recall", "em", "f1", "char_f1"],
    )
    write_latex_table(
        retrieval_dir / "retrieval_qa_per_act_table.tex",
        rows,
        "Per-Act retrieval plus QA breakdown on IndicIPR-QA.",
        "tab:retrieval_qa_per_act",
        [
            ("setting", "Setting"),
            ("act", "Act"),
            ("n", "N"),
            ("retrieval_recall", "Retr. R@k"),
            ("em", "EM"),
            ("f1", "F1"),
            ("char_f1", "Char-F1"),
        ],
    )
    return rows


def format_top_candidates(row: dict, limit: int = 3) -> str:
    candidates = row.get("top_candidates") or []
    if isinstance(candidates, list) and candidates:
        parts = []
        for item in candidates[:limit]:
            rank = item.get("rank", "")
            title = item.get("context_title") or item.get("title") or item.get("context_id") or "untitled"
            act = item.get("context_act") or item.get("act") or "Unknown"
            mark = "gold" if truthy(item.get("is_gold_context")) else "non-gold"
            parts.append(f"{rank}. {title} [{act}, {mark}]")
        return "; ".join(parts)

    selected_rank = row.get("selected_rank", "")
    selected_title = row.get("selected_context_title") or row.get("selected_context_id") or ""
    selected_act = row.get("selected_context_act") or ""
    if selected_title:
        return f"{selected_rank}. {selected_title} [{selected_act}]"
    return ""


def retrieval_failure_reason(row: dict) -> tuple[str, str]:
    gold_act = str(row.get("act") or "").strip()
    selected_act = str(row.get("selected_context_act") or "").strip()
    f1 = float(row.get("f1") or 0.0)

    if selected_act and gold_act and selected_act != gold_act:
        return (
            "Wrong Act retrieval",
            "The gold paragraph is absent from the retrieved set and the selected paragraph comes from a different Act.",
        )
    if f1 >= 0.5:
        return (
            "Partial lexical match",
            "The retriever missed the gold paragraph, but the selected paragraph still shares enough wording to produce partial overlap.",
        )
    return (
        "Wrong paragraph retrieval",
        "The correct paragraph is absent from the retrieved set, so the extractor answers from a non-gold paragraph.",
    )


def write_retrieval_failure_examples(retrieval_dir: Path, prediction_rows: list[dict], limit: int = 5) -> list[dict]:
    failures = [row for row in prediction_rows if not truthy(row.get("retrieval_hit_at_k"))]
    failures.sort(key=lambda row: (float(row.get("f1") or 0.0), str(row.get("id") or "")))
    examples: list[dict] = []

    for row in failures[:limit]:
        error_type, explanation = retrieval_failure_reason(row)
        examples.append(
            {
                "id": row.get("id", ""),
                "act": row.get("act", ""),
                "question": row.get("question", ""),
                "gold": row.get("gold", ""),
                "prediction": row.get("prediction", ""),
                "selected_context_title": row.get("selected_context_title", ""),
                "selected_context_act": row.get("selected_context_act", ""),
                "top_candidates": format_top_candidates(row),
                "error_type": error_type,
                "explanation": explanation,
                "em": row.get("em", 0.0),
                "f1": row.get("f1", 0.0),
                "char_f1": row.get("char_f1", 0.0),
                "gold_short": short_text(row.get("gold", ""), 95),
                "prediction_short": short_text(row.get("prediction", ""), 95),
            }
        )

    write_jsonl(retrieval_dir / "retrieval_failure_examples.jsonl", examples)
    note = (
        f"Found {len(failures)} retrieval misses in the selected BM25 setting. "
        f"Wrote {len(examples)} representative examples.\n"
    )
    (retrieval_dir / "retrieval_failure_examples_note.txt").write_text(note, encoding="utf-8")
    write_latex_table(
        retrieval_dir / "retrieval_failure_examples_table.tex",
        examples,
        "Representative retrieval failure examples.",
        "tab:retrieval_failure_examples",
        [
            ("id", "ID"),
            ("act", "Act"),
            ("error_type", "Error"),
            ("gold_short", "Gold answer"),
            ("prediction_short", "Prediction"),
            ("explanation", "Explanation"),
        ],
    )
    return examples


def infer_model_size_b(model_name: str) -> float:
    name = model_name.lower()
    if "llama" in name:
        return 8.0
    if "qwen" in name:
        return 7.0
    if "mistral" in name:
        return 7.0
    if "bert" in name or "inlegal" in name or "incaselaw" in name:
        return 0.11
    return 0.0


def load_llm_baseline_rows() -> list[dict]:
    rows: list[dict] = []
    seen_files: set[Path] = set()
    for path in LLM_BASELINE_FILES:
        if path in seen_files or not path.exists():
            continue
        seen_files.add(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        model_name = str(data.get("model") or path.stem)
        for key, setting in [("zero_shot", "LLM zero-shot"), ("few_shot", "LLM few-shot")]:
            metrics = data.get(key)
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "model": model_name,
                    "setting": setting,
                    "n": int(metrics.get("n") or 0),
                    "em": float(metrics.get("EM") or metrics.get("em") or 0.0),
                    "f1": float(metrics.get("F1") or metrics.get("f1") or 0.0),
                    "char_f1": float(metrics.get("Char_F1") or metrics.get("char_f1") or 0.0),
                    "approx_params_b": infer_model_size_b(model_name),
                    "source": path.name,
                }
            )
    return rows


def write_model_efficiency_comparison(output_dir: Path, transfer_rows: list[dict]) -> list[dict]:
    out_dir = output_dir / "model_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)

    cprime = None
    for row in transfer_rows:
        model = str(row.get("model", ""))
        if "C':" in model and row.get("test_domain") == "IndicIPR-QA":
            cprime = row
            break
    if cprime is None:
        for row in transfer_rows:
            if row.get("test_domain") == "IndicIPR-QA":
                cprime = row
                break

    rows: list[dict] = []
    if cprime:
        rows.append(
            {
                "model": cprime.get("model", "C': MLM+SQuAD+IPR InLegalBERT"),
                "setting": "Extractive fine-tuned QA",
                "n": cprime.get("n", 0),
                "em": cprime.get("em", 0.0),
                "f1": cprime.get("f1", 0.0),
                "char_f1": cprime.get("char_f1", 0.0),
                "approx_params_b": infer_model_size_b(str(cprime.get("model", ""))),
                "source": "transfer_comparison",
            }
        )
    rows.extend(load_llm_baseline_rows())

    if not rows:
        return rows

    write_summary_csv(
        out_dir / "cprime_vs_llm_efficiency.csv",
        rows,
        ["model", "setting", "n", "approx_params_b", "em", "f1", "char_f1", "source"],
    )
    write_latex_table(
        out_dir / "cprime_vs_llm_efficiency_table.tex",
        rows,
        "C' compared with local LLM baselines using model size and QA metrics.",
        "tab:cprime_vs_llm_efficiency",
        [
            ("model", "Model"),
            ("setting", "Setting"),
            ("approx_params_b", "Params (B)"),
            ("em", "EM"),
            ("f1", "F1"),
            ("char_f1", "Char-F1"),
        ],
    )
    return rows


def run_retrieval_qa(args: argparse.Namespace) -> list[dict]:
    out_dir = Path(args.output_dir) / "retrieval_qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_records = load_ipr_records(Path(args.ipr_test_file), args.limit)
    contexts = build_context_corpus(
        Path(args.corpus_file),
        allow_split_fallback=args.allow_split_corpus_fallback,
    )
    if not test_records:
        raise ValueError(f"No IPR test records loaded from {args.ipr_test_file}")
    if not contexts:
        raise ValueError(f"No retrieval contexts loaded from {args.corpus_file}")
    bm25 = BM25Index(contexts)
    max_k = max(args.top_k)

    print(f"Retrieval corpus contexts: {len(contexts)}")
    print(f"Test records: {len(test_records)}")
    print(f"QA model: {args.qa_model}")

    tokenizer, model, device = load_qa_model(Path(args.qa_model), args.device)

    gold_predictions: list[dict] = []
    per_k_predictions: dict[int, list[dict]] = {k: [] for k in args.top_k}
    retrieval_hits: dict[int, int] = {k: 0 for k in args.top_k}

    for idx, record in enumerate(test_records, start=1):
        gold_pred, gold_score = predict_answer(
            tokenizer,
            model,
            device,
            record.question,
            record.context,
            args.max_seq_length,
            args.doc_stride,
            args.max_answer_tokens,
            args.n_best,
        )
        gold_row = score_prediction(record, gold_pred)
        gold_row["model_score"] = gold_score
        gold_predictions.append(gold_row)

        retrieved = bm25.topk(record.question, max_k)
        candidate_predictions = []
        for rank, (ctx_idx, bm25_score) in enumerate(retrieved, start=1):
            ctx = contexts[ctx_idx]
            pred, qa_score = predict_answer(
                tokenizer,
                model,
                device,
                record.question,
                ctx.context,
                args.max_seq_length,
                args.doc_stride,
                args.max_answer_tokens,
                args.n_best,
            )
            candidate_predictions.append(
                {
                    "rank": rank,
                    "context_id": ctx.id,
                    "context_title": ctx.title,
                    "context_act": ctx.act,
                    "bm25_score": bm25_score,
                    "prediction": pred,
                    "qa_score": qa_score,
                    "is_gold_context": ctx.context == record.context,
                }
            )

        for k in args.top_k:
            top_candidates = candidate_predictions[:k]
            if any(item["is_gold_context"] for item in top_candidates):
                retrieval_hits[k] += 1
            best = max(top_candidates, key=lambda item: item["qa_score"])
            row = score_prediction(record, best["prediction"])
            row.update(
                {
                    "setting": f"BM25 top-{k} + QA",
                    "retrieval_hit_at_k": any(item["is_gold_context"] for item in top_candidates),
                    "selected_rank": best["rank"],
                    "selected_context_id": best["context_id"],
                    "selected_context_title": best["context_title"],
                    "selected_context_act": best["context_act"],
                    "bm25_score": best["bm25_score"],
                    "qa_score": best["qa_score"],
                    "top_candidates": [
                        {
                            "rank": item["rank"],
                            "context_id": item["context_id"],
                            "context_title": item["context_title"],
                            "context_act": item["context_act"],
                            "bm25_score": item["bm25_score"],
                            "qa_score": item["qa_score"],
                            "is_gold_context": item["is_gold_context"],
                        }
                        for item in top_candidates
                    ],
                }
            )
            per_k_predictions[k].append(row)

        if idx % args.progress_every == 0 or idx == len(test_records):
            print(f"  processed {idx}/{len(test_records)}")
            write_jsonl(out_dir / "gold_paragraph_predictions.jsonl", gold_predictions)
            for k in args.top_k:
                write_jsonl(out_dir / f"bm25_top_{k}_qa_predictions.jsonl", per_k_predictions[k])

    summary_rows: list[dict] = []
    gold_summary = metric_summary(gold_predictions)
    gold_summary.update(
        {
            "setting": "Gold paragraph + QA",
            "retrieval_recall": 100.0,
            "qa_model": args.qa_label,
        }
    )
    summary_rows.append(gold_summary)

    write_jsonl(out_dir / "gold_paragraph_predictions.jsonl", gold_predictions)
    for k in args.top_k:
        preds = per_k_predictions[k]
        write_jsonl(out_dir / f"bm25_top_{k}_qa_predictions.jsonl", preds)
        summary = metric_summary(preds)
        summary.update(
            {
                "setting": f"BM25 top-{k} + QA",
                "retrieval_recall": round(retrieval_hits[k] / max(len(test_records), 1) * 100, 4),
                "qa_model": args.qa_label,
            }
        )
        summary_rows.append(summary)

    write_summary_csv(
        out_dir / "retrieval_qa_summary.csv",
        summary_rows,
        ["setting", "qa_model", "n", "retrieval_recall", "em", "f1", "char_f1"],
    )
    (out_dir / "retrieval_qa_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.limit is None:
        model_path = Path(args.qa_model).resolve()
        try:
            model_path_text = model_path.relative_to(ROOT).as_posix()
        except ValueError:
            model_path_text = str(model_path)
        canonical_report = {
            "date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "qa_model": args.qa_label,
            "qa_model_path": model_path_text,
            "test_file": Path(args.ipr_test_file).resolve().relative_to(ROOT).as_posix(),
            "corpus_file": Path(args.corpus_file).resolve().relative_to(ROOT).as_posix(),
            "n_test": len(test_records),
            "n_contexts": len(contexts),
            "inference_protocol": QADecodingConfig(
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                batch_size=1,
                max_answer_tokens=args.max_answer_tokens,
            ).as_dict(),
            "summary": summary_rows,
        }
        citation_groups_path = (
            BASE_OUTPUT_DIR / "citation_ablation" / "citation_ablation_queries.jsonl"
        )
        if citation_groups_path.exists():
            group_rows = load_jsonl(citation_groups_path)
            if len(group_rows) == len(gold_predictions):
                preferred_k = 5 if 5 in per_k_predictions else max(per_k_predictions)
                question_type_results = {}
                for group_name in (
                    "citation-template-only",
                    "lexically-content-bearing",
                ):
                    indices = [
                        int(row["idx"])
                        for row in group_rows
                        if row.get("group") == group_name
                    ]
                    gold_subset = [gold_predictions[index] for index in indices]
                    retrieved_subset = [per_k_predictions[preferred_k][index] for index in indices]
                    question_type_results[group_name] = {
                        "n": len(indices),
                        "gold": metric_summary(gold_subset),
                        f"bm25_top_{preferred_k}": metric_summary(retrieved_subset),
                    }
                canonical_report["question_type_results"] = question_type_results
        canonical_path = BASE_OUTPUT_DIR / "retrieval_qa_report.json"
        canonical_path.write_text(
            json.dumps(canonical_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Canonical report: {canonical_path}")
    write_latex_table(
        out_dir / "retrieval_qa_table.tex",
        summary_rows,
        "End-to-end retrieval plus QA results on IndicIPR-QA.",
        "tab:retrieval_qa",
        [
            ("setting", "Setting"),
            ("retrieval_recall", "Retr. R@k"),
            ("em", "EM"),
            ("f1", "F1"),
            ("char_f1", "Char-F1"),
        ],
    )
    setting_rows = {"Gold paragraph + QA": gold_predictions}
    for k in args.top_k:
        setting_rows[f"BM25 top-{k} + QA"] = per_k_predictions[k]
    write_retrieval_act_breakdown(out_dir, setting_rows)
    largest_k = max(args.top_k)
    write_retrieval_failure_examples(out_dir, per_k_predictions[largest_k])

    print("\nRetrieval + QA summary")
    for row in summary_rows:
        print(
            f"  {row['setting']:<20} R={row['retrieval_recall']:>6.2f} "
            f"EM={row['em']:>6.2f} F1={row['f1']:>6.2f} Char-F1={row['char_f1']:>6.2f}"
        )
    print(f"Saved: {out_dir / 'retrieval_qa_summary.csv'}")
    del model
    gc.collect()
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.empty_cache()
    return summary_rows


def parse_model_specs(specs: list[str] | None) -> list[tuple[str, Path]]:
    if not specs:
        return []
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                "Model spec must be LABEL=PATH. "
                f"Got: {spec}. Example: --ipr-model \"Cprime=C:\\path\\to\\model\\final\""
            )
        label, path = spec.split("=", 1)
        parsed.append((label.strip(), Path(path.strip())))
    return parsed


def available_models(models: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    out = []
    for label, path in models:
        valid, reason = is_valid_model_dir(path)
        if valid:
            out.append((label, path))
        else:
            print(f"[SKIP] Invalid model path for {label}: {path} ({reason})")
    return out


def load_transfer_squad_records(args: argparse.Namespace) -> list[QARecord]:
    squad_file = Path(args.squad_file) if args.squad_file else find_local_squad_file()
    if squad_file and squad_file.exists():
        print(f"Loading local SQuAD file: {squad_file}")
        return load_squad_file(squad_file, args.squad_limit)
    if args.download_squad:
        print("Loading SQuAD validation from Hugging Face datasets")
        return load_squad_from_hf(args.squad_limit)
    print(
        "No local SQuAD dev file found. Skipping IndicIPR-QA -> SQuAD evaluation. "
        "Pass --squad-file dev-v1.1.json or --download-squad to enable it."
    )
    return []


def run_transfer(args: argparse.Namespace) -> list[dict]:
    out_dir = Path(args.output_dir) / "transfer_comparison"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ipr_records = load_ipr_records(Path(args.ipr_test_file), args.limit)
    if not ipr_records:
        raise ValueError(f"No IPR test records loaded from {args.ipr_test_file}")
    squad_records = load_transfer_squad_records(args)

    if args.all_default_models:
        squad_models = available_models(DEFAULT_SQUAD_ONLY_MODELS)
        ipr_models = available_models(DEFAULT_IPR_MODELS)
        adapted_models = available_models(DEFAULT_ADAPTED_MODELS) if args.include_adapted else []
    else:
        squad_models = available_models([("SQuAD-only: InLegalBERT", DEFAULT_SQUAD_MODEL)])
        ipr_models = available_models(DEFAULT_IPR_MODELS)
        adapted_models = available_models(DEFAULT_ADAPTED_MODELS) if args.include_adapted else []

    squad_models.extend(available_models(parse_model_specs(args.squad_model)))
    ipr_models.extend(available_models(parse_model_specs(args.ipr_model)))
    adapted_models.extend(available_models(parse_model_specs(args.adapted_model)))

    summary_rows: list[dict] = []

    for label, path in squad_models:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
        summary = evaluate_model_on_records(
            records=ipr_records,
            model_label=label,
            model_path=path,
            args=args,
            prediction_path=pred_dir / f"{safe_label}_on_indicipr.jsonl",
        )
        summary.update(
            {
                "experiment": "SQuAD -> IndicIPR-QA",
                "experiment_type": "cross-dataset zero-shot transfer",
                "train_domain": "SQuAD",
                "test_domain": "IndicIPR-QA",
            }
        )
        summary_rows.append(summary)

    for label, path in ipr_models:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
        summary = evaluate_model_on_records(
            records=ipr_records,
            model_label=label,
            model_path=path,
            args=args,
            prediction_path=pred_dir / f"{safe_label}_on_indicipr.jsonl",
        )
        summary.update(
            {
                "experiment": "IndicIPR-QA -> IndicIPR-QA",
                "experiment_type": "in-domain supervised baseline",
                "train_domain": "IndicIPR-QA",
                "test_domain": "IndicIPR-QA",
            }
        )
        summary_rows.append(summary)

    for label, path in adapted_models:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
        summary = evaluate_model_on_records(
            records=ipr_records,
            model_label=label,
            model_path=path,
            args=args,
            prediction_path=pred_dir / f"{safe_label}_on_indicipr.jsonl",
        )
        summary.update(
            {
                "experiment": "SQuAD+IndicIPR-QA -> IndicIPR-QA",
                "experiment_type": "sequential transfer/adaptation",
                "train_domain": "SQuAD+IndicIPR-QA",
                "test_domain": "IndicIPR-QA",
            }
        )
        summary_rows.append(summary)

    if squad_records:
        for label, path in ipr_models:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
            summary = evaluate_model_on_records(
                records=squad_records,
                model_label=label,
                model_path=path,
                args=args,
                prediction_path=pred_dir / f"{safe_label}_on_squad.jsonl",
            )
            summary.update(
                {
                    "experiment": "IndicIPR-QA -> SQuAD",
                    "experiment_type": "reverse cross-dataset transfer",
                    "train_domain": "IndicIPR-QA",
                    "test_domain": "SQuAD",
                }
            )
            summary_rows.append(summary)

        for label, path in adapted_models:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
            summary = evaluate_model_on_records(
                records=squad_records,
                model_label=label,
                model_path=path,
                args=args,
                prediction_path=pred_dir / f"{safe_label}_on_squad.jsonl",
            )
            summary.update(
                {
                    "experiment": "SQuAD+IndicIPR-QA -> SQuAD",
                    "experiment_type": "adapted-model SQuAD retention",
                    "train_domain": "SQuAD+IndicIPR-QA",
                    "test_domain": "SQuAD",
                }
            )
            summary_rows.append(summary)

        for label, path in squad_models[:1]:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
            summary = evaluate_model_on_records(
                records=squad_records,
                model_label=label,
                model_path=path,
                args=args,
                prediction_path=pred_dir / f"{safe_label}_on_squad.jsonl",
            )
            summary.update(
                {
                    "experiment": "SQuAD -> SQuAD",
                    "experiment_type": "source-domain supervised baseline",
                    "train_domain": "SQuAD",
                    "test_domain": "SQuAD",
                }
            )
            summary_rows.append(summary)

    write_summary_csv(
        out_dir / "transfer_comparison_summary.csv",
        summary_rows,
        [
            "experiment",
            "experiment_type",
            "model",
            "train_domain",
            "test_domain",
            "n",
            "em",
            "f1",
            "char_f1",
            "model_path",
        ],
    )
    (out_dir / "transfer_comparison_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_latex_table(
        out_dir / "transfer_comparison_table.tex",
        summary_rows,
        "Transfer and zero-shot comparison between SQuAD and IndicIPR-QA.",
        "tab:transfer_comparison",
        [
            ("experiment", "Experiment"),
            ("model", "Model"),
            ("em", "EM"),
            ("f1", "F1"),
            ("char_f1", "Char-F1"),
        ],
    )

    print("\nTransfer / zero-shot summary")
    for row in summary_rows:
        print(
            f"  {row['experiment']:<26} {row['model']:<28} "
            f"EM={row['em']:>6.2f} F1={row['f1']:>6.2f} Char-F1={row['char_f1']:>6.2f}"
        )
    print(f"Saved: {out_dir / 'transfer_comparison_summary.csv'}")
    write_model_efficiency_comparison(Path(args.output_dir), summary_rows)
    return summary_rows


def infer_transfer_metadata_from_prediction_file(path: Path) -> dict:
    stem = path.stem
    if "_on_" in stem:
        raw_label, target = stem.rsplit("_on_", 1)
    else:
        raw_label, target = stem, "indicipr"

    label_map = {
        "SQuAD-only_InLegalBERT": "SQuAD-only: InLegalBERT",
        "SQuAD-only_BERT-base": "SQuAD-only: BERT-base",
        "SQuAD-only_LegalBERT-US": "SQuAD-only: LegalBERT-US",
        "SQuAD-only_InCaseLawBERT": "SQuAD-only: InCaseLawBERT",
        "A_IPR_direct_InLegalBERT": "A: IPR direct InLegalBERT",
        "SQuAD_IPR_InLegalBERT": "SQuAD+IPR: InLegalBERT",
        "SQuAD_IPR_BERT-base": "SQuAD+IPR: BERT-base",
        "SQuAD_IPR_LegalBERT-US": "SQuAD+IPR: LegalBERT-US",
        "SQuAD_IPR_InCaseLawBERT": "SQuAD+IPR: InCaseLawBERT",
        "C_MLM_SQuAD_IPR_InLegalBERT": "C': MLM+SQuAD+IPR InLegalBERT",
        "MLM_SQuAD_IPR_Orig-only_InLegalBERT": "MLM+SQuAD+IPR: Orig-only InLegalBERT",
        "MLM_SQuAD_IPR_Synth-only_InLegalBERT": "MLM+SQuAD+IPR: Synth-only InLegalBERT",
    }
    model = label_map.get(raw_label, raw_label.replace("_", " "))
    test_domain = "SQuAD" if target.lower() == "squad" else "IndicIPR-QA"

    if raw_label.startswith("SQuAD-only"):
        train_domain = "SQuAD"
        experiment = "SQuAD -> SQuAD" if test_domain == "SQuAD" else "SQuAD -> IndicIPR-QA"
        experiment_type = (
            "source-domain supervised baseline"
            if test_domain == "SQuAD"
            else "cross-dataset zero-shot transfer"
        )
    elif raw_label.startswith("A_IPR_direct"):
        train_domain = "IndicIPR-QA"
        experiment = "IndicIPR-QA -> SQuAD" if test_domain == "SQuAD" else "IndicIPR-QA -> IndicIPR-QA"
        experiment_type = (
            "reverse cross-dataset transfer"
            if test_domain == "SQuAD"
            else "in-domain supervised baseline"
        )
    else:
        train_domain = "SQuAD+IndicIPR-QA"
        experiment = "SQuAD+IndicIPR-QA -> SQuAD" if test_domain == "SQuAD" else "SQuAD+IndicIPR-QA -> IndicIPR-QA"
        experiment_type = (
            "adapted-model SQuAD retention"
            if test_domain == "SQuAD"
            else "sequential transfer/adaptation"
        )

    return {
        "experiment": experiment,
        "experiment_type": experiment_type,
        "model": model,
        "train_domain": train_domain,
        "test_domain": test_domain,
        "model_path": "",
    }


def rebuild_transfer_summary_from_predictions(output_dir: Path) -> list[dict]:
    out_dir = output_dir / "transfer_comparison"
    pred_dir = out_dir / "predictions"
    if not pred_dir.exists():
        return []

    summary_rows: list[dict] = []
    for path in sorted(pred_dir.glob("*_on_*.jsonl")):
        predictions = load_jsonl(path)
        summary = metric_summary(predictions)
        summary.update(infer_transfer_metadata_from_prediction_file(path))
        summary_rows.append(summary)

    if not summary_rows:
        return summary_rows

    write_summary_csv(
        out_dir / "transfer_comparison_summary.csv",
        summary_rows,
        [
            "experiment",
            "experiment_type",
            "model",
            "train_domain",
            "test_domain",
            "n",
            "em",
            "f1",
            "char_f1",
            "model_path",
        ],
    )
    (out_dir / "transfer_comparison_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_latex_table(
        out_dir / "transfer_comparison_table.tex",
        summary_rows,
        "Transfer and zero-shot comparison between SQuAD and IndicIPR-QA.",
        "tab:transfer_comparison",
        [
            ("experiment", "Experiment"),
            ("model", "Model"),
            ("em", "EM"),
            ("f1", "F1"),
            ("char_f1", "Char-F1"),
        ],
    )
    return summary_rows


def retrieval_k_from_path(path: Path) -> int:
    match = re.search(r"bm25_top_(\d+)_qa_predictions", path.name)
    return int(match.group(1)) if match else 0


def postprocess_existing_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    retrieval_dir = output_dir / "retrieval_qa"
    if retrieval_dir.exists():
        setting_rows: dict[str, list[dict]] = {}
        gold_path = retrieval_dir / "gold_paragraph_predictions.jsonl"
        if gold_path.exists():
            setting_rows["Gold paragraph + QA"] = load_jsonl(gold_path)
        bm25_paths = sorted(
            retrieval_dir.glob("bm25_top_*_qa_predictions.jsonl"),
            key=retrieval_k_from_path,
        )
        for path in bm25_paths:
            k = retrieval_k_from_path(path)
            setting_rows[f"BM25 top-{k} + QA"] = load_jsonl(path)
        if setting_rows:
            write_retrieval_act_breakdown(retrieval_dir, setting_rows)
        if bm25_paths:
            largest_path = bm25_paths[-1]
            write_retrieval_failure_examples(retrieval_dir, load_jsonl(largest_path))
            print(f"Postprocessed retrieval outputs from {retrieval_dir}")
    else:
        print(f"[SKIP] Retrieval directory not found: {retrieval_dir}")

    transfer_rows = rebuild_transfer_summary_from_predictions(output_dir)
    if transfer_rows:
        write_model_efficiency_comparison(output_dir, transfer_rows)
        print(f"Postprocessed transfer and efficiency outputs from {output_dir / 'transfer_comparison'}")
    else:
        print(f"[SKIP] Transfer prediction files not found under {output_dir / 'transfer_comparison' / 'predictions'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run journal-level retrieval+QA and transfer/zero-shot experiments."
    )
    parser.add_argument("--mode", choices=["retrieval", "transfer", "all"], default="all")
    parser.add_argument(
        "--run-name",
        default="journal_retrieval_transfer_e2prime_canonical_v1",
        help=(
            "Save outputs under results/experiments/retrieval/runs/<run-name>. "
            "Use the same name to intentionally continue the same run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output directory. Overrides --run-name and the timestamped default.",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Rebuild derived tables/examples from existing prediction files without model inference.",
    )
    parser.add_argument(
        "--reuse-complete-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For transfer evaluation, reuse an existing complete prediction JSONL instead of rerunning that model.",
    )
    parser.add_argument("--ipr-test-file", default=str(IPR_TEST_FILE))
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit for IPR test records.")

    parser.add_argument("--qa-model", default=str(DEFAULT_IPR_MODEL), help="QA model for retrieval+QA.")
    parser.add_argument("--qa-label", default="E2': LegalBERT-US + SQuAD v1 + IPR")
    parser.add_argument("--corpus-file", default=str(IPR_FINAL_FILE), help="Corpus source for BM25 contexts.")
    parser.add_argument(
        "--allow-split-corpus-fallback",
        action="store_true",
        help="If --corpus-file is missing, build the BM25 corpus from train/val/test splits with a warning.",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5])

    parser.add_argument("--squad-file", default=None, help="Local SQuAD v1 dev JSON/JSONL for IPR -> SQuAD.")
    parser.add_argument("--download-squad", action="store_true", help="Load SQuAD validation via datasets.load_dataset.")
    parser.add_argument("--squad-limit", type=int, default=2000, help="Limit SQuAD dev records for faster evaluation.")
    parser.add_argument(
        "--all-default-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate all available default SQuAD-only models.",
    )
    parser.add_argument(
        "--include-adapted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate final SQuAD -> IndicIPR-QA and MLM -> SQuAD -> IndicIPR-QA models.",
    )
    parser.add_argument("--squad-model", action="append", help="Extra SQuAD model as LABEL=PATH.")
    parser.add_argument("--ipr-model", action="append", help="Extra IPR model as LABEL=PATH.")
    parser.add_argument("--adapted-model", action="append", help="Extra adapted model as LABEL=PATH.")

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=None,
        help="optional span-length cap; default searches every valid span in a window",
    )
    parser.add_argument(
        "--n-best",
        type=int,
        default=20,
        help="deprecated compatibility option; exhaustive constrained decoding ignores it",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(k <= 0 for k in args.top_k):
        raise ValueError("--top-k values must be positive integers.")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive.")
    if args.doc_stride < 0:
        raise ValueError("--doc-stride must be non-negative.")
    if args.max_answer_tokens is not None and args.max_answer_tokens <= 0:
        raise ValueError("--max-answer-tokens must be positive.")
    if args.n_best <= 0:
        raise ValueError("--n-best must be positive.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.squad_limit is not None and args.squad_limit <= 0:
        raise ValueError("--squad-limit must be positive when provided.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.top_k = sorted(set(args.top_k))
    args.output_dir = str(choose_output_dir(args).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    if args.postprocess_only:
        postprocess_existing_run(args)
        return

    if args.mode in {"retrieval", "all"}:
        run_retrieval_qa(args)
    if args.mode in {"transfer", "all"}:
        run_transfer(args)
    write_run_manifest(args)


if __name__ == "__main__":
    main()
