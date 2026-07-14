"""CLI for the first two-task SQL training loop."""

from __future__ import annotations

import argparse
import json

from .pipeline import prepare, run_eval, run_loop, run_train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiveSQLBench two-task first loop")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--public-data", required=True)
    prepare_parser.add_argument("--target-manifest", required=True)
    prepare_parser.add_argument("--db-root", required=True)
    prepare_parser.add_argument("--train-output", required=True)
    prepare_parser.add_argument("--dev-output", required=True)

    loop_parser = commands.add_parser("loop")
    loop_parser.add_argument("--public-data", required=True)
    loop_parser.add_argument("--target-manifest", required=True)
    loop_parser.add_argument("--db-root", required=True)
    loop_parser.add_argument("--model-path", required=True)
    loop_parser.add_argument("--output-dir", required=True)
    loop_parser.add_argument("--max-length", type=int, default=8192)
    loop_parser.add_argument("--max-new-tokens", type=int, default=256)

    for name in ("eval", "train"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--model-path", required=True)
        if name == "eval":
            command.add_argument("--output", required=True)
            command.add_argument("--adapter-path")
            command.add_argument("--max-new-tokens", type=int, default=256)
        else:
            command.add_argument("--adapter-output", required=True)
            command.add_argument("--max-length", type=int, default=8192)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            public_data=args.public_data,
            target_manifest=args.target_manifest,
            db_root=args.db_root,
            train_output=args.train_output,
            dev_output=args.dev_output,
        )
    elif args.command == "loop":
        result = run_loop(
            public_data=args.public_data,
            target_manifest=args.target_manifest,
            db_root=args.db_root,
            model_path=args.model_path,
            output_dir=args.output_dir,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.command == "eval":
        result = run_eval(
            dataset=args.dataset,
            model_path=args.model_path,
            output=args.output,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        result = run_train(
            dataset=args.dataset,
            model_path=args.model_path,
            adapter_output=args.adapter_output,
            max_length=args.max_length,
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
