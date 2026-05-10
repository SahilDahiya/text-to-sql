"""Minimal project CLI."""

from __future__ import annotations

import argparse

from .sql import (
    analyze_sql_eval_result,
    collect_sql_repair_data,
    import_sql_benchmark,
    load_sql_eval_cases,
    load_sql_repair_examples,
    load_sql_sft_manifest,
    load_sql_train_examples,
    run_sql_eval,
    run_sql_eval_with_repair,
    run_sql_sft,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SQLBench Lab workspace")
    parser.add_argument("--version", action="store_true", help="Print the package version")
    subparsers = parser.add_subparsers(dest="command")

    observe_parser = subparsers.add_parser("observe", help="Experiment observability commands")
    observe_subparsers = observe_parser.add_subparsers(dest="observe_command")
    observe_ui = observe_subparsers.add_parser("ui", help="Launch the local MLflow dashboard")
    observe_ui.add_argument("--host", default="127.0.0.1", help="MLflow UI bind host")
    observe_ui.add_argument("--port", type=int, default=5000, help="MLflow UI bind port")
    observe_ui.add_argument("--backend-store-uri", help="MLflow backend store URI")

    sql_parser = subparsers.add_parser("sql", help="SQL pipeline commands")
    sql_subparsers = sql_parser.add_subparsers(dest="sql_command")

    validate_train = sql_subparsers.add_parser("validate-train", help="Validate SQL train JSONL")
    validate_train.add_argument("--dataset", required=True, help="Path to SQL train JSONL")

    validate_eval = sql_subparsers.add_parser("validate-eval", help="Validate SQL eval JSONL")
    validate_eval.add_argument("--dataset", required=True, help="Path to SQL eval JSONL")

    validate_repair = sql_subparsers.add_parser("validate-repair", help="Validate SQL repair JSONL")
    validate_repair.add_argument("--dataset", required=True, help="Path to SQL repair JSONL")

    validate_manifest = sql_subparsers.add_parser("validate-manifest", help="Validate SQL SFT manifest")
    validate_manifest.add_argument("--manifest", required=True, help="Path to SQL SFT manifest JSON")

    import_benchmark = sql_subparsers.add_parser("import-benchmark", help="Import Spider or BIRD from Hugging Face")
    import_benchmark.add_argument("--benchmark", choices=["bird", "spider"], required=True)
    import_benchmark.add_argument("--split", choices=["train", "validation"], required=True)
    import_benchmark.add_argument("--artifact", choices=["train", "eval"], required=True)
    import_benchmark.add_argument("--output", required=True, help="Output JSONL path")
    import_benchmark.add_argument("--limit", type=int, help="Optional row cap")
    import_benchmark.add_argument("--cache-root", help="Benchmark snapshot cache root")
    import_benchmark.add_argument("--force-download", action="store_true")

    run_sft = sql_subparsers.add_parser("run-sft", help="Run SQL LoRA SFT from a manifest")
    run_sft.add_argument("--manifest", required=True, help="Path to SQL SFT manifest JSON")
    run_sft.add_argument("--dry-run", action="store_true", help="Validate and render training rows only")
    run_sft.add_argument("--mlflow", action="store_true", help="Log the run to MLflow")
    run_sft.add_argument("--mlflow-tracking-uri", help="Override the MLflow tracking URI")
    run_sft.add_argument("--mlflow-experiment", help="Override the MLflow experiment name")

    eval_sql = sql_subparsers.add_parser("eval", help="Run SQL smoke evaluation from a manifest")
    eval_sql.add_argument("--manifest", required=True, help="Path to SQL SFT manifest JSON")
    eval_sql.add_argument("--model", choices=["base", "adapter"], required=True, help="Model variant to evaluate")
    eval_sql.add_argument("--dataset", help="Override eval dataset JSONL path")
    eval_sql.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated SQL tokens")
    eval_sql.add_argument("--mlflow", action="store_true", help="Log the eval run to MLflow")
    eval_sql.add_argument("--mlflow-tracking-uri", help="Override the MLflow tracking URI")
    eval_sql.add_argument("--mlflow-experiment", help="Override the MLflow experiment name")

    eval_repair = sql_subparsers.add_parser("eval-repair", help="Run SQL eval with execution-guided repair retries")
    eval_repair.add_argument("--manifest", required=True, help="Path to SQL SFT manifest JSON")
    eval_repair.add_argument("--model", choices=["base", "adapter"], required=True, help="Model variant to evaluate")
    eval_repair.add_argument("--dataset", help="Override eval dataset JSONL path")
    eval_repair.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated SQL tokens")
    eval_repair.add_argument("--max-repair-attempts", type=int, default=1, help="Repair attempts per failed case")
    eval_repair.add_argument(
        "--repair-failure-type",
        action="append",
        dest="repair_failure_types",
        help="Failure type eligible for repair; repeat to allow multiple types",
    )

    analyze_eval = sql_subparsers.add_parser("analyze-eval", help="Analyze a SQL eval result JSON")
    analyze_eval.add_argument("--result", required=True, help="Path to SQL eval result JSON")
    analyze_eval.add_argument("--output", help="Output analysis JSON path")

    collect_repair = sql_subparsers.add_parser("collect-repair-data", help="Collect repair JSONL from eval failures")
    collect_repair.add_argument("--result", required=True, help="Path to SQL eval result JSON")
    collect_repair.add_argument("--eval-dataset", required=True, help="Matching SQL eval dataset JSONL")
    collect_repair.add_argument("--output", required=True, help="Output SQL repair JSONL path")
    collect_repair.add_argument(
        "--failure-type",
        action="append",
        dest="failure_types",
        help="Failure type to collect; repeat to collect multiple types",
    )
    collect_repair.add_argument(
        "--strong-only",
        action="store_true",
        help="Collect only syntax/schema/runtime/empty prediction repair candidates",
    )

    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if args.command == "sql":
        try:
            return _run_sql_command(args)
        except (ImportError, ValueError) as exc:
            parser.error(str(exc))
    if args.command == "observe":
        try:
            return _run_observe_command(args)
        except (ImportError, ValueError) as exc:
            parser.error(str(exc))
    parser.print_help()
    return 0


def _run_observe_command(args: argparse.Namespace) -> int:
    if args.observe_command == "ui":
        from sqlbench_lab.observability import launch_mlflow_ui, mlflow_tracking_uri

        backend_store_uri = mlflow_tracking_uri(args.backend_store_uri)
        print(f"starting MLflow UI at http://{args.host}:{args.port}")
        print(f"backend store: {backend_store_uri}")
        return launch_mlflow_ui(
            backend_store_uri=backend_store_uri,
            host=args.host,
            port=args.port,
        )
    raise ValueError("missing observe command")


def _run_sql_command(args: argparse.Namespace) -> int:
    if args.sql_command == "validate-train":
        rows = load_sql_train_examples(args.dataset)
        print(f"validated SQL train dataset with {len(rows)} row(s): {args.dataset}")
        return 0
    if args.sql_command == "validate-eval":
        cases = load_sql_eval_cases(args.dataset)
        print(f"validated SQL eval dataset with {len(cases)} case(s): {args.dataset}")
        return 0
    if args.sql_command == "validate-repair":
        rows = load_sql_repair_examples(args.dataset)
        print(f"validated SQL repair dataset with {len(rows)} row(s): {args.dataset}")
        return 0
    if args.sql_command == "validate-manifest":
        manifest = load_sql_sft_manifest(args.manifest)
        train_count = sum(
            len(load_sql_train_examples(path))
            for path in manifest.train_inputs.train_datasets
        )
        eval_count = len(load_sql_eval_cases(manifest.eval_plan.smoke_dataset))
        print(
            "validated SQL SFT manifest: "
            f"{manifest.experiment_id} ({train_count} train row(s), {eval_count} smoke case(s))"
        )
        return 0
    if args.sql_command == "import-benchmark":
        summary = import_sql_benchmark(
            benchmark=args.benchmark,
            split=args.split,
            artifact=args.artifact,
            output_path=args.output,
            limit=args.limit,
            cache_root=args.cache_root,
            force_download=args.force_download,
        )
        print(
            "imported SQL benchmark "
            f"{summary.benchmark}/{summary.split} artifact={summary.artifact} "
            f"rows={summary.row_count} output={summary.output_path}"
        )
        return 0
    if args.sql_command == "run-sft":
        summary = run_sql_sft(
            args.manifest,
            dry_run=args.dry_run,
            log_mlflow=args.mlflow or None,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment,
        )
        print(
            "completed SQL SFT "
            f"{summary.experiment_id} dry_run={summary.dry_run} train_rows={summary.train_row_count}"
        )
        return 0
    if args.sql_command == "eval":
        summary = run_sql_eval(
            args.manifest,
            model_variant=args.model,
            eval_dataset=args.dataset,
            max_new_tokens=args.max_new_tokens,
            log_mlflow=args.mlflow or None,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment,
        )
        print(
            "completed SQL eval "
            f"{summary.experiment_id} model={summary.model_variant} "
            f"passed={summary.passed_count}/{summary.case_count} "
            f"pass_rate={summary.pass_rate:.4f}"
        )
        return 0
    if args.sql_command == "eval-repair":
        summary = run_sql_eval_with_repair(
            args.manifest,
            model_variant=args.model,
            eval_dataset=args.dataset,
            max_new_tokens=args.max_new_tokens,
            max_repair_attempts=args.max_repair_attempts,
            repair_failure_types=set(args.repair_failure_types) if args.repair_failure_types else None,
        )
        print(
            "completed SQL repair eval "
            f"{summary.experiment_id} model={summary.model_variant} "
            f"first_passed={summary.first_passed_count}/{summary.case_count} "
            f"final_passed={summary.final_passed_count}/{summary.case_count} "
            f"repair_attempts={summary.repair_attempt_count} "
            f"repair_successes={summary.repair_success_count}"
        )
        return 0
    if args.sql_command == "analyze-eval":
        summary = analyze_sql_eval_result(args.result, output_path=args.output)
        failure_counts = ", ".join(
            f"{failure_type}={count}"
            for failure_type, count in summary.failure_counts.items()
        )
        print(
            "analyzed SQL eval "
            f"{summary.model_variant} failed={summary.failed_count}/{summary.case_count} "
            f"output={summary.analysis_path}"
        )
        if failure_counts:
            print(f"failure_counts: {failure_counts}")
        return 0
    if args.sql_command == "collect-repair-data":
        summary = collect_sql_repair_data(
            result_path=args.result,
            eval_dataset=args.eval_dataset,
            output_path=args.output,
            failure_types=set(args.failure_types) if args.failure_types else None,
            strong_only=args.strong_only,
        )
        failure_counts = ", ".join(
            f"{failure_type}={count}"
            for failure_type, count in summary.failure_counts.items()
        )
        print(
            "collected SQL repair data "
            f"rows={summary.collected_count} skipped={summary.skipped_count} "
            f"output={summary.output_path}"
        )
        if failure_counts:
            print(f"failure_counts: {failure_counts}")
        return 0
    raise ValueError("missing SQL command")


if __name__ == "__main__":
    raise SystemExit(main())
