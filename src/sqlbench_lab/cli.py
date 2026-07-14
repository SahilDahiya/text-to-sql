"""Command line surface for the LiveSQLBench direct-SQL ISFT lane."""

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
    build_livesqlbench_artifacts,
    load_sql_eval_cases,
    load_sql_sft_manifest,
    load_sql_train_examples,
    build_review_packet,
    record_human_review,
    run_sql_eval,
    run_sql_sft,
    verify_livesqlbench_targets,
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
    validate_manifest = sql_commands.add_parser("validate-manifest", help="Validate a v2 ISFT manifest")
    validate_manifest.add_argument("--manifest", required=True)

    import_tasks = sql_commands.add_parser("livesqlbench-import", help="Import allowed LiveSQLBench tasks")
    import_tasks.add_argument("--package-root", required=True)
    import_tasks.add_argument("--target-manifest", required=True)
    import_tasks.add_argument("--source-revision", required=True)
    import_tasks.add_argument("--train-output", required=True)
    import_tasks.add_argument("--eval-output", required=True)

    verify_targets = sql_commands.add_parser("verify-targets", help="Execute and attest target SQL against allowed databases")
    verify_targets.add_argument("--package-root", required=True)
    verify_targets.add_argument("--target-manifest", required=True)
    verify_targets.add_argument("--source-revision", required=True)
    verify_targets.add_argument("--verified-output", required=True)
    verify_targets.add_argument("--verified-by", required=True)
    verify_targets.add_argument("--verified-at", required=True)

    review_packet = sql_commands.add_parser("build-review-packet", help="Build human review evidence")
    review_packet.add_argument("--iteration", required=True)
    review_packet.add_argument("--phase", choices=["artifacts", "baseline", "evaluation"], required=True)
    review_packet.add_argument("--manifest", required=True)
    review_packet.add_argument("--output", required=True)
    review_packet.add_argument("--result")
    review_packet.add_argument("--conversation")

    review = sql_commands.add_parser("record-review", help="Record a human review decision")
    review.add_argument("--packet", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", choices=["approve", "reject", "request_extra_review"], required=True)
    review.add_argument("--notes", default="")
    review.add_argument("--extra-question", action="append", default=[])
    review.add_argument("--output", required=True)

    train = sql_commands.add_parser("run-sft", help="Run or dry-run iterative supervised fine-tuning")
    train.add_argument("--manifest", required=True)
    train.add_argument("--review", required=True)
    train.add_argument("--dry-run", action="store_true")

    evaluate = sql_commands.add_parser("eval", help="Run one-shot local SQL evaluation")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--model", choices=["base", "adapter"], required=True)
    evaluate.add_argument("--dataset")
    evaluate.add_argument("--max-new-tokens", type=int)
    evaluate.add_argument("--result-label")

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
        print(f"validated training dataset rows={len(rows)} path={args.dataset}")
        return 0
    if args.sql_command == "validate-eval":
        cases = load_sql_eval_cases(args.dataset)
        print(f"validated local development eval cases={len(cases)} path={args.dataset}")
        return 0
    if args.sql_command == "validate-manifest":
        manifest = load_sql_sft_manifest(args.manifest)
        print(f"validated ISFT manifest experiment={manifest.experiment_id} path={args.manifest}")
        return 0
    if args.sql_command == "livesqlbench-import":
        summary = build_livesqlbench_artifacts(
            package_root=args.package_root,
            target_manifest=args.target_manifest,
            source_revision=args.source_revision,
            train_output=args.train_output,
            eval_output=args.eval_output,
        )
        print(f"imported LiveSQLBench tasks={summary.discovered_task_count} train={summary.train_row_count} eval={summary.eval_case_count}")
        return 0
    if args.sql_command == "verify-targets":
        summary = verify_livesqlbench_targets(
            package_root=args.package_root,
            target_manifest=args.target_manifest,
            source_revision=args.source_revision,
            verified_output=args.verified_output,
            verified_by=args.verified_by,
            verified_at=args.verified_at,
        )
        print(f"verified LiveSQLBench targets={summary.target_count} output={summary.verified_output}")
        return 0
    if args.sql_command == "build-review-packet":
        summary = build_review_packet(
            iteration_id=args.iteration,
            phase=args.phase,
            manifest_path=args.manifest,
            output_path=args.output,
            result_path=args.result,
            conversation_path=args.conversation,
        )
        print(f"built review packet={summary.packet_id} markdown={summary.markdown_path} json={summary.json_path}")
        return 0
    if args.sql_command == "record-review":
        output = record_human_review(
            packet_path=args.packet,
            reviewer=args.reviewer,
            decision=args.decision,
            output_path=args.output,
            notes=args.notes,
            extra_questions=args.extra_question,
        )
        print(f"recorded human review={output}")
        return 0
    if args.sql_command == "run-sft":
        summary = run_sql_sft(args.manifest, dry_run=args.dry_run, review_path=args.review)
        print(f"completed SQL ISFT experiment={summary.experiment_id} rows={summary.train_row_count} dry_run={summary.dry_run}")
        return 0
    if args.sql_command == "eval":
        summary = run_sql_eval(
            args.manifest,
            model_variant=args.model,
            eval_dataset=args.dataset,
            max_new_tokens=args.max_new_tokens,
            result_label=args.result_label,
        )
        print(f"completed local one-shot eval passed={summary.passed_count}/{summary.case_count} result={summary.result_path}")
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
