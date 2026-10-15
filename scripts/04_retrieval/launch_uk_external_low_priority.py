"""Launch the UK external validation as a capped, below-normal background job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(__file__).with_name("uk_statutecorpus_external_validation.py")
DEFAULT_OUTPUT = (
    ROOT / "results" / "experiments" / "retrieval" / "runs"
    / "uk_statutecorpus_external_v1"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "BACKGROUND_JOB.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        previous_pid = int(previous.get("pid", 0))
        if process_exists(previous_pid) and not (output_dir / "RUN_COMPLETE.json").exists():
            raise SystemExit(f"UK external validation is already running as PID {previous_pid}")

    command = [
        sys.executable,
        str(TARGET),
        "--threads",
        str(args.threads),
        "--dense-batch-size",
        str(args.dense_batch_size),
        "--rerank-batch-size",
        str(args.rerank_batch_size),
        "--output-dir",
        str(output_dir),
    ]
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = str(args.threads)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"

    log_path = output_dir / "run.log"
    log_stream = log_path.open("a", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    log_stream.close()
    state = {
        "pid": process.pid,
        "status": "RUNNING",
        "started_at": now_iso(),
        "priority": "below_normal" if os.name == "nt" else "normal",
        "threads": args.threads,
        "command": command,
        "log": str(log_path.resolve()),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
