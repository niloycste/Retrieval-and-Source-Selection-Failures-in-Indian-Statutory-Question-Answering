"""Launch the BGE retrieval extension after Experiment D releases the CPU.

The watcher is intentionally lightweight and records its state.  It prevents
the two large CPU-only studies from silently contending for memory and compute.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "dapt" / "equal_update" / "RUN_LEDGER.csv"
MODEL_MANIFEST = ROOT / "models" / "retrieval" / "modern_bge_extension_v1" / "model_manifest.json"
OUT = ROOT / "results" / "experiments" / "retrieval" / "runs" / "modern_bge_extension_v1"
STATE = OUT / "QUEUE_STATE.json"
LOG = OUT / "queued_run.log"
EXPECTED_EVALUATIONS = 15


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_state(status: str, **extra) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"status": status, "updated_at": now(), **extra}, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def completed_evaluations() -> int:
    if not LEDGER.exists():
        return 0
    with LEDGER.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(
            row.get("stage") == "val_eval" and row.get("status") == "EVALUATED"
            for row in csv.DictReader(stream)
        )


def main() -> int:
    write_state("WAITING", reason="DAPT Experiment D and/or model snapshots incomplete")
    while True:
        done = completed_evaluations()
        models_ready = MODEL_MANIFEST.exists()
        if done >= EXPECTED_EVALUATIONS and models_ready:
            break
        write_state(
            "WAITING", evaluated_runs=done, expected_evaluated_runs=EXPECTED_EVALUATIONS,
            models_ready=models_ready,
        )
        time.sleep(60)

    base_command = [
        sys.executable,
        str(ROOT / "scripts" / "04_retrieval" / "modern_retrieval_extension.py"),
    ]
    smoke_command = base_command + ["--smoke-models", "--threads", "2"]
    command = base_command + [
        "--dataset", "all", "--threads", "4", "--dense-batch-size", "2",
        "--rerank-batch-size", "2", "--embedding-chunk-size", "128",
    ]
    write_state("SMOKE_TESTING", command=smoke_command)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{now()}] SMOKE {' '.join(smoke_command)}\n")
        stream.flush()
        smoke = subprocess.run(smoke_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if smoke.returncode != 0:
            write_state("FAILED_SMOKE_TEST", returncode=smoke.returncode, log=str(LOG))
            return smoke.returncode
        write_state("RUNNING", command=command)
        stream.write(f"\n[{now()}] START {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        write_state("COMPLETE", returncode=0, log=str(LOG))
    else:
        write_state("FAILED", returncode=result.returncode, log=str(LOG))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
