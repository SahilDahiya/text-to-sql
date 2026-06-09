"""Standalone SQL agent environment CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .env import run_env_step
from .import_seed import import_seed_dataset
from .schema_inspector import inspect_sqlite_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DB-specific SQL agent environment")
    subparsers = parser.add_subparsers(dest="command")

    env_step = subparsers.add_parser("env-step", help="Run one SQL action against one eval case")
    env_step.add_argument("--dataset", required=True, help="SQL eval dataset JSONL path")
    env_step.add_argument("--case-id", required=True, help="Eval case_id to step")
    sql_group = env_step.add_mutually_exclusive_group(required=True)
    sql_group.add_argument("--sql", help="Candidate SQL action")
    sql_group.add_argument("--sql-file", help="File containing candidate SQL action")
    env_step.add_argument("--attempt", type=int, default=1, help="Attempt index for this action")
    env_step.add_argument("--preview-rows", type=int, default=3, help="Rows to include in previews")
    env_step.add_argument("--output", help="Optional JSON output path")

    inspect_schema = subparsers.add_parser("inspect-schema", help="Inspect a SQLite database schema")
    inspect_schema.add_argument("--db", required=True, help="SQLite database path")
    inspect_schema.add_argument("--output", help="Optional JSON output path")

    import_seed = subparsers.add_parser("import-seed", help="Copy seed train/eval datasets into this project")
    import_seed.add_argument("--train", required=True, help="Source train JSONL path")
    import_seed.add_argument("--eval", required=True, help="Source eval JSONL path")
    import_seed.add_argument("--output", required=True, help="Output seed directory")
    import_seed.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow exact train/eval question or SQL overlap while recording it in the summary",
    )

    args = parser.parse_args(argv)
    if args.command == "env-step":
        step = run_env_step(
            dataset_path=args.dataset,
            case_id=args.case_id,
            sql=args.sql if args.sql is not None else _read_sql_file(args.sql_file),
            attempt=args.attempt,
            preview_rows=args.preview_rows,
            output_path=args.output,
        )
        print(json.dumps(asdict(step), indent=2, ensure_ascii=True))
        return 0
    if args.command == "inspect-schema":
        profile = inspect_sqlite_schema(args.db, output_path=args.output)
        print(json.dumps(asdict(profile), indent=2, ensure_ascii=True))
        return 0
    if args.command == "import-seed":
        summary = import_seed_dataset(
            train_path=args.train,
            eval_path=args.eval,
            output_dir=args.output,
            allow_overlap=args.allow_overlap,
        )
        print(json.dumps(asdict(summary), indent=2, ensure_ascii=True))
        return 0

    parser.print_help()
    return 0


def _read_sql_file(path: str | None) -> str:
    if path is None:
        raise ValueError("--sql-file is required when --sql is not provided")
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"SQL file must not be empty: {path}")
    return text.strip()


if __name__ == "__main__":
    raise SystemExit(main())
