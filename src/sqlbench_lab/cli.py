"""Command line surface for LiveSQLBench-oriented SQL ISFT."""

from __future__ import annotations

import argparse
from pathlib import Path

from .livesqlbench_submission import (
    LiveSQLBenchSubmissionConfig,
    build_submission_plan,
    run_prepare,
    run_submission,
    write_submission_plan,
)
from .sql import (
    analyze_sql_eval_result,
    assert_no_sql_dataset_leakage,
    load_sql_eval_cases,
    load_sql_sft_manifest,
    load_sql_train_examples,
    run_sql_eval,
    run_sql_sft,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiveSQLBench SQL ISFT workspace")
    parser.add_argument("--version", action="store_true", help="Print the package version")
    commands = parser.add_subparsers(dest="command")

    sql = commands.add_parser("sql", help="SQL training and local validation")
    sql_commands = sql.add_subparsers(dest="sql_command")

    validate_train = sql_commands.add_parser("validate-train", help="Validate an allowed training JSONL")
    validate_train.add_argument("--dataset", required=True)

    validate_eval = sql_commands.add_parser("validate-eval", help="Validate a local development eval JSONL")
    validate_eval.add_argument("--dataset", required=True)

    validate_manifest = sql_commands.add_parser("validate-manifest", help="Validate an ISFT manifest")
    validate_manifest.add_argument("--manifest", required=True)

    leakage = sql_commands.add_parser("audit-leakage", help="Fail on train/eval overlap")
    leakage.add_argument("--train-dataset", action="append", required=True)
    leakage.add_argument("--eval-dataset", action="append", required=True)
    leakage.add_argument("--require-db-disjoint", action="store_true")

    train = sql_commands.add_parser("run-sft", help="Run or dry-run iterative supervised fine-tuning")
    train.add_argument("--manifest", required=True)
    train.add_argument("--dry-run", action="store_true")

    evaluate = sql_commands.add_parser("eval", help="Run one-shot local SQL evaluation")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--model", choices=["base", "adapter"], required=True)
    evaluate.add_argument("--dataset")
    evaluate.add_argument("--max-new-tokens", type=int, default=128)
    evaluate.add_argument("--result-label")

    analyze = sql_commands.add_parser("analyze-eval", help="Analyze a local eval result")
    analyze.add_argument("--result", required=True)
    analyze.add_argument("--output")

    for name, help_text in (
        ("livesqlbench-prepare", "Prepare official LiveSQLBench tasks from public inputs"),
        ("livesqlbench-run", "Run the official LiveSQLBench Harbor lane"),
    ):
        benchmark = sql_commands.add_parser(name, help=help_text)
        _add_livesqlbench_arguments(benchmark)
        benchmark.add_argument("--manifest-output", required=True)

    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if args.command != "sql":
        parser.print_help()
        return 0
    return _run_sql_command(args)


def _run_sql_command(args: argparse.Namespace) -> int:
    if args.sql_command == "validate-train":
        rows = load_sql_train_examples(args.dataset)
        print(f"validated allowed training dataset rows={len(rows)} path={args.dataset}")
        return 0
    if args.sql_command == "validate-eval":
        cases = load_sql_eval_cases(args.dataset)
        print(f"validated local development eval cases={len(cases)} path={args.dataset}")
        return 0
    if args.sql_command == "validate-manifest":
        manifest = load_sql_sft_manifest(args.manifest)
        print(f"validated ISFT manifest experiment={manifest.experiment_id} path={args.manifest}")
        return 0
    if args.sql_command == "audit-leakage":
        summary = assert_no_sql_dataset_leakage(
            train_paths=args.train_dataset,
            eval_paths=args.eval_dataset,
            require_db_disjoint=args.require_db_disjoint,
        )
        print(
            "leakage audit passed "
            f"train_rows={summary.train_row_count} eval_cases={summary.eval_case_count} "
            f"train_dbs={len(summary.train_db_ids)} eval_dbs={len(summary.eval_db_ids)}"
        )
        return 0
    if args.sql_command == "run-sft":
        summary = run_sql_sft(args.manifest, dry_run=args.dry_run)
        print(
            "completed SQL ISFT "
            f"experiment={summary.experiment_id} rows={summary.train_row_count} dry_run={summary.dry_run}"
        )
        return 0
    if args.sql_command == "eval":
        summary = run_sql_eval(
            args.manifest,
            model_variant=args.model,
            eval_dataset=args.dataset,
            max_new_tokens=args.max_new_tokens,
            result_label=args.result_label,
        )
        print(
            "completed local one-shot eval "
            f"passed={summary.passed_count}/{summary.case_count} result={summary.result_path}"
        )
        return 0
    if args.sql_command == "analyze-eval":
        summary = analyze_sql_eval_result(args.result, output_path=args.output)
        print(f"analyzed local eval failures={summary.failure_count} output={summary.output_path}")
        return 0
    if args.sql_command in {"livesqlbench-prepare", "livesqlbench-run"}:
        config = _livesqlbench_config(args)
        plan = build_submission_plan(config)
        manifest_path = write_submission_plan(plan, args.manifest_output)
        if args.sql_command == "livesqlbench-run":
            run_submission(plan)
            print(f"completed official LiveSQLBench run output={plan.output_dir} plan={manifest_path}")
        else:
            run_prepare(plan)
            print(f"completed official LiveSQLBench task preparation output={plan.output_dir} plan={manifest_path}")
        return 0
    raise ValueError("missing SQL command")


def _add_livesqlbench_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cli-repo", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--eval-src-dir", required=True)
    parser.add_argument("--db-dump-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-image", default="livesqlbench-main-openhands:latest")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")


def _livesqlbench_config(args: argparse.Namespace) -> LiveSQLBenchSubmissionConfig:
    return LiveSQLBenchSubmissionConfig(
        cli_repo=Path(args.cli_repo).resolve(),
        data_root=Path(args.data_root).resolve(),
        data_jsonl=Path(args.data_jsonl).resolve(),
        eval_src_dir=Path(args.eval_src_dir).resolve(),
        db_dump_root=Path(args.db_dump_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        agent_image=args.agent_image,
        agent=args.agent,
        model=args.model,
        trials=args.trials,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
