"""Minimal project CLI."""

from __future__ import annotations

import argparse

from .sql import load_sql_eval_cases, load_sql_sft_manifest, load_sql_train_examples, run_sql_sft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SQLBench Lab workspace")
    parser.add_argument("--version", action="store_true", help="Print the package version")
    subparsers = parser.add_subparsers(dest="command")
    sql_parser = subparsers.add_parser("sql", help="SQL pipeline commands")
    sql_subparsers = sql_parser.add_subparsers(dest="sql_command")

    validate_train = sql_subparsers.add_parser("validate-train", help="Validate SQL train JSONL")
    validate_train.add_argument("--dataset", required=True, help="Path to SQL train JSONL")

    validate_eval = sql_subparsers.add_parser("validate-eval", help="Validate SQL eval JSONL")
    validate_eval.add_argument("--dataset", required=True, help="Path to SQL eval JSONL")

    validate_manifest = sql_subparsers.add_parser("validate-manifest", help="Validate SQL SFT manifest")
    validate_manifest.add_argument("--manifest", required=True, help="Path to SQL SFT manifest JSON")

    run_sft = sql_subparsers.add_parser("run-sft", help="Run SQL LoRA SFT from a manifest")
    run_sft.add_argument("--manifest", required=True, help="Path to SQL SFT manifest JSON")
    run_sft.add_argument("--dry-run", action="store_true", help="Validate and render training rows only")
    run_sft.add_argument("--mlflow", action="store_true", help="Log the run to MLflow")
    run_sft.add_argument("--mlflow-tracking-uri", help="Override the MLflow tracking URI")
    run_sft.add_argument("--mlflow-experiment", help="Override the MLflow experiment name")

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
    parser.print_help()
    return 0


def _run_sql_command(args: argparse.Namespace) -> int:
    if args.sql_command == "validate-train":
        rows = load_sql_train_examples(args.dataset)
        print(f"validated SQL train dataset with {len(rows)} row(s): {args.dataset}")
        return 0
    if args.sql_command == "validate-eval":
        cases = load_sql_eval_cases(args.dataset)
        print(f"validated SQL eval dataset with {len(cases)} case(s): {args.dataset}")
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
    raise ValueError("missing SQL command")


if __name__ == "__main__":
    raise SystemExit(main())
