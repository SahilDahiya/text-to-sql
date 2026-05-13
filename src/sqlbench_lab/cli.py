"""Minimal project CLI."""

from __future__ import annotations

import argparse

from .sql import (
    analyze_sql_eval_result,
    assert_no_sql_dataset_leakage,
    attach_sqlite_profile_metadata,
    collect_sql_repair_data,
    generate_bird_regional_sales_normalization_micro_lab,
    generate_bird_regional_sales_schema_lab,
    generate_bird_regional_sales_unit_price_contrast_lab,
    generate_bird_superstore_schema_lab,
    import_sql_benchmark,
    load_sql_eval_cases,
    load_sql_repair_examples,
    load_sql_sft_manifest,
    load_sql_train_examples,
    record_sql_prompt_candidate,
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

    docs_parser = subparsers.add_parser("docs", help="Browser docs commands")
    docs_subparsers = docs_parser.add_subparsers(dest="docs_command")
    docs_build = docs_subparsers.add_parser("build", help="Build the static browser docs")
    docs_build.add_argument("--output", default="site", help="Output directory")
    docs_serve = docs_subparsers.add_parser("serve", help="Build and serve the browser docs")
    docs_serve.add_argument("--output", default="site", help="Output directory")
    docs_serve.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    docs_serve.add_argument("--port", type=int, default=8000, help="HTTP bind port")

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
    import_benchmark.add_argument(
        "--selection",
        choices=["first", "stratified"],
        default="first",
        help="Row selection strategy when --limit is set",
    )
    import_benchmark.add_argument("--cache-root", help="Benchmark snapshot cache root")
    import_benchmark.add_argument("--force-download", action="store_true")
    import_benchmark.add_argument(
        "--db-id",
        action="append",
        dest="db_ids",
        help="Restrict import to specific db_id values; repeat for multiple DBs",
    )

    audit_leakage = sql_subparsers.add_parser("audit-leakage", help="Audit SQL train/eval dataset leakage")
    audit_leakage.add_argument(
        "--train-dataset",
        action="append",
        dest="train_datasets",
        required=True,
        help="SQL train JSONL path; repeat for multiple train inputs",
    )
    audit_leakage.add_argument(
        "--eval-dataset",
        action="append",
        dest="eval_datasets",
        required=True,
        help="SQL eval JSONL path; repeat for multiple eval inputs",
    )
    audit_leakage.add_argument(
        "--require-db-disjoint",
        action="store_true",
        help="Fail if any train db_id appears in eval; use for unseen-DB evaluation",
    )

    bird_lab = sql_subparsers.add_parser("generate-bird-lab", help="Generate train-split BIRD schema-linking lab data")
    bird_lab.add_argument("--db-id", choices=["superstore", "regional_sales"], default="superstore")
    bird_lab.add_argument("--train-output", required=True, help="Output SQL train JSONL path")
    bird_lab.add_argument("--eval-output", required=True, help="Output SQL eval JSONL path")
    bird_lab.add_argument("--dataset-root", help="BIRD train split root containing train_databases")
    bird_lab.add_argument("--curriculum-version", choices=["v1", "v2", "v3"], default="v1")
    bird_lab.add_argument(
        "--include-column-value-notes",
        action="store_true",
        help="Add DB-derived column value notes to generated BIRD lab rows",
    )

    bird_normalization_lab = sql_subparsers.add_parser(
        "generate-bird-regional-sales-normalization-lab",
        help="Generate train-only BIRD regional_sales text-number normalization micro-lab data",
    )
    bird_normalization_lab.add_argument("--train-output", required=True, help="Output SQL train JSONL path")
    bird_normalization_lab.add_argument("--dataset-root", help="BIRD train split root containing train_databases")

    bird_unit_price_lab = sql_subparsers.add_parser(
        "generate-bird-regional-sales-unit-price-lab",
        help="Generate train-only BIRD regional_sales unit-price target-shape lab data",
    )
    bird_unit_price_lab.add_argument("--train-output", required=True, help="Output SQL train JSONL path")
    bird_unit_price_lab.add_argument("--dataset-root", help="BIRD train split root containing train_databases")

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

    optimize_prompt = sql_subparsers.add_parser(
        "optimize-prompt",
        help="Record one prompt optimization candidate with optional MLflow tracking",
    )
    optimize_prompt.add_argument("--experiment-id", required=True, help="Prompt optimization experiment ID")
    optimize_prompt.add_argument(
        "--optimizer",
        choices=["mipro_v2", "gepa", "manual"],
        required=True,
        help="Optimizer that produced this prompt candidate",
    )
    optimize_prompt.add_argument("--candidate-id", required=True, help="Stable candidate ID")
    optimize_prompt.add_argument("--prompt-file", required=True, help="Prompt text file for this candidate")
    optimize_prompt.add_argument("--prompt-dev-dataset", required=True, help="Prompt-dev eval dataset JSONL")
    optimize_prompt.add_argument("--fresh-gate-dataset", help="Fresh unseen eval gate JSONL, when scored")
    optimize_prompt.add_argument("--source-manifest", help="Source SFT manifest for the base adapter/model")
    optimize_prompt.add_argument("--model-variant", choices=["base", "adapter"], help="Model variant used")
    optimize_prompt.add_argument("--eval-result", help="Eval result JSON for this candidate, when available")
    optimize_prompt.add_argument("--analysis", help="Eval analysis JSON for this candidate, when available")
    optimize_prompt.add_argument(
        "--decision",
        choices=["pending", "selected", "rejected"],
        default="pending",
        help="Candidate decision after comparison",
    )
    optimize_prompt.add_argument("--notes", help="Short candidate notes")
    optimize_prompt.add_argument("--output", help="Override prompt candidate artifact JSON path")
    optimize_prompt.add_argument("--mlflow", action="store_true", help="Log the candidate to MLflow")
    optimize_prompt.add_argument("--mlflow-tracking-uri", help="Override the MLflow tracking URI")
    optimize_prompt.add_argument("--mlflow-experiment", help="Override the MLflow experiment name")

    profile_metadata = sql_subparsers.add_parser(
        "profile-metadata",
        help="Attach compact SQLite profile metadata to SQL train/eval JSONL",
    )
    profile_metadata.add_argument("--input", required=True, help="Input SQL JSONL path")
    profile_metadata.add_argument("--output", required=True, help="Output SQL JSONL path")
    profile_metadata.add_argument("--artifact", choices=["train", "eval"], required=True)
    profile_metadata.add_argument(
        "--max-column-notes",
        type=int,
        default=12,
        help="Maximum column profile notes per row",
    )
    profile_metadata.add_argument(
        "--max-sample-values",
        type=int,
        default=3,
        help="Maximum sample values per profiled column",
    )
    profile_metadata.add_argument(
        "--max-exact-profile-rows",
        type=int,
        default=50000,
        help="Use exact column stats only for tables up to this row count",
    )
    profile_metadata.add_argument(
        "--profile-sample-rows",
        type=int,
        default=10000,
        help="Non-NULL sample row cap for large-table profile stats",
    )

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
    if args.command == "docs":
        try:
            return _run_docs_command(args)
        except (ImportError, ValueError) as exc:
            parser.error(str(exc))
    parser.print_help()
    return 0


def _run_docs_command(args: argparse.Namespace) -> int:
    from sqlbench_lab.docs_site import build_docs_site, serve_docs_site

    if args.docs_command == "build":
        summary = build_docs_site(args.output)
        print(
            "built SQLBench Lab browser docs "
            f"pages={summary.page_count} assets={summary.asset_count} output={summary.output_dir}"
        )
        return 0
    if args.docs_command == "serve":
        return serve_docs_site(args.output, host=args.host, port=args.port)
    raise ValueError("missing docs command")


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
            selection=args.selection,
            db_ids=tuple(args.db_ids or ()),
            cache_root=args.cache_root,
            force_download=args.force_download,
        )
        print(
            "imported SQL benchmark "
            f"{summary.benchmark}/{summary.split} artifact={summary.artifact} "
            f"selection={summary.selection} rows={summary.row_count} output={summary.output_path}"
        )
        return 0
    if args.sql_command == "audit-leakage":
        summary = assert_no_sql_dataset_leakage(
            train_paths=args.train_datasets,
            eval_paths=args.eval_datasets,
            require_db_disjoint=args.require_db_disjoint,
        )
        print(
            "passed SQL leakage audit "
            f"train_rows={summary.train_row_count} eval_cases={summary.eval_case_count} "
            f"train_dbs={len(summary.train_db_ids)} eval_dbs={len(summary.eval_db_ids)} "
            f"overlap_dbs={len(summary.overlapping_db_ids)} "
            f"require_db_disjoint={summary.require_db_disjoint}"
        )
        return 0
    if args.sql_command == "generate-bird-lab":
        if args.db_id == "superstore":
            if args.include_column_value_notes:
                raise ValueError("--include-column-value-notes is only supported for regional_sales")
            summary = generate_bird_superstore_schema_lab(
                train_output_path=args.train_output,
                eval_output_path=args.eval_output,
                dataset_root=args.dataset_root,
                curriculum_version=args.curriculum_version,
            )
        elif args.db_id == "regional_sales":
            summary = generate_bird_regional_sales_schema_lab(
                train_output_path=args.train_output,
                eval_output_path=args.eval_output,
                dataset_root=args.dataset_root,
                curriculum_version=args.curriculum_version,
                include_column_value_notes=args.include_column_value_notes,
            )
        else:
            raise ValueError(f"unsupported BIRD lab db_id: {args.db_id}")
        print(
            "generated BIRD schema lab "
            f"db={summary.db_id} train_rows={summary.train_row_count} "
            f"eval_rows={summary.eval_row_count} train={summary.train_output_path} "
            f"eval={summary.eval_output_path}"
        )
        return 0
    if args.sql_command == "generate-bird-regional-sales-normalization-lab":
        summary = generate_bird_regional_sales_normalization_micro_lab(
            train_output_path=args.train_output,
            dataset_root=args.dataset_root,
        )
        print(
            "generated BIRD regional_sales normalization lab "
            f"db={summary.db_id} train_rows={summary.train_row_count} train={summary.train_output_path}"
        )
        return 0
    if args.sql_command == "generate-bird-regional-sales-unit-price-lab":
        summary = generate_bird_regional_sales_unit_price_contrast_lab(
            train_output_path=args.train_output,
            dataset_root=args.dataset_root,
        )
        print(
            "generated BIRD regional_sales unit-price lab "
            f"db={summary.db_id} train_rows={summary.train_row_count} train={summary.train_output_path}"
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
    if args.sql_command == "optimize-prompt":
        summary = record_sql_prompt_candidate(
            experiment_id=args.experiment_id,
            optimizer=args.optimizer,
            candidate_id=args.candidate_id,
            prompt_file=args.prompt_file,
            prompt_dev_dataset=args.prompt_dev_dataset,
            fresh_gate_dataset=args.fresh_gate_dataset,
            source_manifest=args.source_manifest,
            model_variant=args.model_variant,
            eval_result=args.eval_result,
            analysis=args.analysis,
            decision=args.decision,
            notes=args.notes,
            output_path=args.output,
            log_mlflow=args.mlflow or None,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment,
        )
        metric = (
            "not_scored"
            if summary.eval_pass_rate is None
            else f"{summary.eval_passed_count}/{summary.eval_case_count} ({summary.eval_pass_rate:.4f})"
        )
        print(
            "recorded SQL prompt candidate "
            f"{summary.experiment_id} optimizer={summary.optimizer} "
            f"candidate={summary.candidate_id} decision={summary.decision} "
            f"prompt_dev_cases={summary.prompt_dev_case_count} metric={metric} "
            f"output={summary.output_path}"
        )
        return 0
    if args.sql_command == "profile-metadata":
        summary = attach_sqlite_profile_metadata(
            input_path=args.input,
            output_path=args.output,
            artifact=args.artifact,
            max_column_notes=args.max_column_notes,
            max_sample_values=args.max_sample_values,
            max_exact_profile_rows=args.max_exact_profile_rows,
            profile_sample_rows=args.profile_sample_rows,
        )
        print(
            "attached SQLite profile metadata "
            f"artifact={summary.artifact} rows={summary.row_count} dbs={summary.db_count} "
            f"notes={summary.total_note_count} output={summary.output_path}"
        )
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
