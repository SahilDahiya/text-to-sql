"""Run Exp032 prompt candidates sequentially with MLflow tracking."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MANIFEST = "experiments/sql/qwen35_0_8b__exp031_trl_profile_metadata.json"
DEFAULT_EXPERIMENT_ID = "qwen35_0_8b__exp032_prompt_optimization"
DEFAULT_PROMPT_DEV = "datasets/sql/eval/bird_train_holdout_restaurant_airline_50_v1_profile_notes.jsonl"
DEFAULT_FRESH_GATE = "datasets/sql/eval/bird_train_holdout_works_cycles_public_review_50_v1_profile_notes.jsonl"
DEFAULT_PROMPT_GLOB = "experiments/sql/prompts/exp032_c*.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SQL prompt candidates overnight")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--prompt-dev-dataset", default=DEFAULT_PROMPT_DEV)
    parser.add_argument("--fresh-gate-dataset", default=DEFAULT_FRESH_GATE)
    parser.add_argument("--prompt-glob", default=DEFAULT_PROMPT_GLOB)
    parser.add_argument("--candidate", action="append", dest="candidates", help="Candidate prompt file; repeatable")
    parser.add_argument("--baseline-passed", type=int, default=7)
    parser.add_argument("--model", choices=["base", "adapter"], default="adapter")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--log-root", default="logs/sql_prompt_candidate_queue")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_root) / started_at
    log_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_files(args)
    summary: dict[str, object] = {
        "started_at": started_at,
        "log_dir": str(log_dir),
        "manifest": args.manifest,
        "prompt_dev_dataset": args.prompt_dev_dataset,
        "fresh_gate_dataset": args.fresh_gate_dataset,
        "baseline_passed": args.baseline_passed,
        "candidates": [],
    }
    _write_json(log_dir / "queue_summary.json", summary)

    if args.dry_run:
        for prompt_file in candidates:
            print(f"{_candidate_id(prompt_file)} {prompt_file}")
        return 0

    for prompt_file in candidates:
        candidate_id = _candidate_id(prompt_file)
        status = _run_candidate(prompt_file, candidate_id=candidate_id, args=args, log_dir=log_dir)
        summary["candidates"].append(status)  # type: ignore[index]
        _write_json(log_dir / "queue_summary.json", summary)

    summary["completed_at"] = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    _write_json(log_dir / "queue_summary.json", summary)
    return 0


def _run_candidate(
    prompt_file: Path,
    *,
    candidate_id: str,
    args: argparse.Namespace,
    log_dir: Path,
) -> dict[str, object]:
    log_path = log_dir / f"{candidate_id}.log"
    status: dict[str, object] = {
        "candidate_id": candidate_id,
        "prompt_file": str(prompt_file),
        "log_path": str(log_path),
        "status": "running",
        "steps": [],
    }
    print(f"START {candidate_id}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        try:
            prompt_dev_result = _result_path(args.manifest, args.prompt_dev_dataset, candidate_id)
            _run_step(
                [
                    "sql",
                    "eval",
                    "--manifest",
                    args.manifest,
                    "--model",
                    args.model,
                    "--dataset",
                    args.prompt_dev_dataset,
                    "--system-prompt-file",
                    str(prompt_file),
                    "--result-label",
                    candidate_id,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    *(_mlflow_arg(args)),
                ],
                log_file=log_file,
                status=status,
                name="eval_prompt_dev",
            )
            prompt_dev_analysis = _analyze(prompt_dev_result, log_file=log_file, status=status)
            prompt_dev_payload = json.loads(Path(prompt_dev_result).read_text(encoding="utf-8"))
            passed_count = int(prompt_dev_payload["passed_count"])
            status["prompt_dev_passed"] = passed_count
            status["prompt_dev_case_count"] = int(prompt_dev_payload["case_count"])
            status["prompt_dev_pass_rate"] = float(prompt_dev_payload["pass_rate"])
            decision = "selected" if passed_count > args.baseline_passed else "rejected"
            status["decision"] = decision
            _record_candidate(
                args=args,
                candidate_id=candidate_id,
                prompt_file=prompt_file,
                eval_result=prompt_dev_result,
                analysis=prompt_dev_analysis,
                decision=decision,
                notes=f"Unattended prompt-dev queue candidate; baseline_passed={args.baseline_passed}.",
                log_file=log_file,
                status=status,
                name="record_prompt_dev_candidate",
            )
            if decision == "selected":
                fresh_candidate_id = f"{candidate_id}_fresh_gate"
                fresh_result = _result_path(args.manifest, args.fresh_gate_dataset, fresh_candidate_id)
                _run_step(
                    [
                        "sql",
                        "eval",
                        "--manifest",
                        args.manifest,
                        "--model",
                        args.model,
                        "--dataset",
                        args.fresh_gate_dataset,
                        "--system-prompt-file",
                        str(prompt_file),
                        "--result-label",
                        fresh_candidate_id,
                        "--max-new-tokens",
                        str(args.max_new_tokens),
                        *(_mlflow_arg(args)),
                    ],
                    log_file=log_file,
                    status=status,
                    name="eval_fresh_gate",
                )
                fresh_analysis = _analyze(fresh_result, log_file=log_file, status=status)
                fresh_payload = json.loads(Path(fresh_result).read_text(encoding="utf-8"))
                status["fresh_gate_passed"] = int(fresh_payload["passed_count"])
                status["fresh_gate_case_count"] = int(fresh_payload["case_count"])
                status["fresh_gate_pass_rate"] = float(fresh_payload["pass_rate"])
                _record_candidate(
                    args=args,
                    candidate_id=fresh_candidate_id,
                    prompt_file=prompt_file,
                    eval_result=fresh_result,
                    analysis=fresh_analysis,
                    decision="pending",
                    notes="Fresh-gate score for selected prompt-dev candidate.",
                    log_file=log_file,
                    status=status,
                    name="record_fresh_gate_candidate",
                )
            status["status"] = "completed"
            print(f"DONE {candidate_id} prompt_dev={passed_count}/50 decision={decision}", flush=True)
        except subprocess.CalledProcessError as exc:
            status["status"] = "failed"
            status["failed_step_returncode"] = exc.returncode
            print(f"FAILED {candidate_id} rc={exc.returncode}", flush=True)
    return status


def _record_candidate(
    *,
    args: argparse.Namespace,
    candidate_id: str,
    prompt_file: Path,
    eval_result: str,
    analysis: str,
    decision: str,
    notes: str,
    log_file,
    status: dict[str, object],
    name: str,
) -> None:
    _run_step(
        [
            "sql",
            "optimize-prompt",
            "--experiment-id",
            args.experiment_id,
            "--optimizer",
            "manual",
            "--candidate-id",
            candidate_id,
            "--prompt-file",
            str(prompt_file),
            "--prompt-dev-dataset",
            args.prompt_dev_dataset,
            "--fresh-gate-dataset",
            args.fresh_gate_dataset,
            "--source-manifest",
            args.manifest,
            "--model-variant",
            args.model,
            "--eval-result",
            eval_result,
            "--analysis",
            analysis,
            "--decision",
            decision,
            "--notes",
            notes,
            *(_mlflow_arg(args)),
        ],
        log_file=log_file,
        status=status,
        name=name,
    )


def _analyze(result_path: str, *, log_file, status: dict[str, object]) -> str:
    analysis_path = str(Path(result_path).with_name(f"{Path(result_path).stem}.analysis.json"))
    _run_step(
        ["sql", "analyze-eval", "--result", result_path],
        log_file=log_file,
        status=status,
        name="analyze_prompt_dev",
    )
    return analysis_path


def _run_step(command: list[str], *, log_file, status: dict[str, object], name: str) -> None:
    full_command = [sys.executable, "-m", "sqlbench_lab.cli", *command]
    started_at = dt.datetime.now(dt.UTC).isoformat()
    log_file.write(f"\n\n## {name} started_at={started_at}\n")
    log_file.write(f"$ {' '.join(full_command)}\n")
    log_file.flush()
    result = subprocess.run(full_command, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=True)
    completed_at = dt.datetime.now(dt.UTC).isoformat()
    status["steps"].append(  # type: ignore[index]
        {
            "name": name,
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "returncode": result.returncode,
        }
    )


def _candidate_files(args: argparse.Namespace) -> list[Path]:
    if args.candidates:
        return [Path(candidate) for candidate in args.candidates]
    return [
        path
        for path in sorted(Path().glob(args.prompt_glob))
        if _candidate_id(path) not in {"c000_current_system", "c001_schema_grounded", "c002_value_grounded"}
    ]


def _candidate_id(prompt_file: Path) -> str:
    stem = prompt_file.stem
    prefix = "exp032_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def _result_path(manifest_path: str, dataset_path: str, result_label: str) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    experiment_id = str(manifest["experiment_id"])
    dataset_stem = Path(dataset_path).stem
    return f"results/sql/{experiment_id}/adapter__{dataset_stem}__{result_label}.json"


def _mlflow_arg(args: argparse.Namespace) -> list[str]:
    return ["--mlflow"] if args.mlflow else []


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
