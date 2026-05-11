"""Run SQL SFT experiments sequentially with train/eval/analyze steps."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


BIRD_EVAL = "datasets/sql/eval/bird_validation_25_v1.jsonl"
SPIDER_EVAL = "datasets/sql/eval/spider_validation_25_v1.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a SQL experiment queue")
    parser.add_argument("manifests", nargs="+", help="Manifest JSON files to run in order")
    parser.add_argument("--log-root", default="logs/sql_experiment_queue", help="Directory for run logs")
    parser.add_argument("--mlflow", action="store_true", help="Log train/eval runs to MLflow")
    parser.add_argument("--skip-trained", action="store_true", help="Skip SFT if train summary exists")
    parser.add_argument("--validate-only", action="store_true", help="Validate manifests without training/eval")
    args = parser.parse_args()

    started_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_root) / started_at
    log_dir.mkdir(parents=True, exist_ok=True)
    queue_summary = {
        "started_at": started_at,
        "log_dir": str(log_dir),
        "experiments": [],
    }
    failures = 0
    for manifest in args.manifests:
        status = _run_one(
            Path(manifest),
            log_dir=log_dir,
            mlflow=args.mlflow,
            skip_trained=args.skip_trained,
            validate_only=args.validate_only,
        )
        queue_summary["experiments"].append(status)
        _write_json(log_dir / "queue_summary.json", queue_summary)
        if status["status"] != "completed":
            failures += 1
    queue_summary["completed_at"] = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    queue_summary["failure_count"] = failures
    _write_json(log_dir / "queue_summary.json", queue_summary)
    return 1 if failures else 0


def _run_one(
    manifest_path: Path,
    *,
    log_dir: Path,
    mlflow: bool,
    skip_trained: bool,
    validate_only: bool,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment_id = str(manifest["experiment_id"])
    log_path = log_dir / f"{experiment_id}.log"
    status: dict[str, object] = {
        "experiment_id": experiment_id,
        "manifest": str(manifest_path),
        "log_path": str(log_path),
        "status": "running",
        "steps": [],
    }
    _print(f"START {experiment_id}")
    with log_path.open("a", encoding="utf-8") as log_file:
        try:
            _run_step(
                ["sql", "validate-manifest", "--manifest", str(manifest_path)],
                log_file=log_file,
                status=status,
                name="validate_manifest",
            )
            if validate_only:
                status["status"] = "completed"
                _print(f"VALID {experiment_id}")
                return status
            train_summary = Path(manifest["output_paths"]["train_summary_json"])
            if skip_trained and train_summary.exists():
                status["steps"].append({"name": "run_sft", "status": "skipped", "reason": "train summary exists"})
            else:
                command = ["sql", "run-sft", "--manifest", str(manifest_path)]
                if mlflow:
                    command.append("--mlflow")
                _run_step(command, log_file=log_file, status=status, name="run_sft")
            for dataset_name, dataset_path in [("bird", BIRD_EVAL), ("spider", SPIDER_EVAL)]:
                command = [
                    "sql",
                    "eval",
                    "--manifest",
                    str(manifest_path),
                    "--dataset",
                    dataset_path,
                    "--model",
                    "adapter",
                ]
                if mlflow:
                    command.append("--mlflow")
                _run_step(command, log_file=log_file, status=status, name=f"eval_{dataset_name}")
                result_path = _result_path(experiment_id=experiment_id, dataset_path=dataset_path)
                _run_step(
                    ["sql", "analyze-eval", "--result", result_path],
                    log_file=log_file,
                    status=status,
                    name=f"analyze_{dataset_name}",
                )
            status["status"] = "completed"
            _print(f"DONE {experiment_id}")
        except subprocess.CalledProcessError as exc:
            status["status"] = "failed"
            status["failed_step_returncode"] = exc.returncode
            _print(f"FAILED {experiment_id} rc={exc.returncode}")
    return status


def _run_step(
    command: list[str],
    *,
    log_file,
    status: dict[str, object],
    name: str,
) -> None:
    full_command = [sys.executable, "-m", "sqlbench_lab.cli", *command]
    started_at = dt.datetime.now(dt.UTC).isoformat()
    log_file.write(f"\n\n## {name} started_at={started_at}\n")
    log_file.write(f"$ {' '.join(full_command)}\n")
    log_file.flush()
    _print(f"  {name}")
    result = subprocess.run(full_command, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=True)
    completed_at = dt.datetime.now(dt.UTC).isoformat()
    status["steps"].append(
        {
            "name": name,
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "returncode": result.returncode,
        }
    )


def _result_path(*, experiment_id: str, dataset_path: str) -> str:
    dataset_stem = Path(dataset_path).stem
    return f"results/sql/{experiment_id}/adapter__{dataset_stem}.json"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _print(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
