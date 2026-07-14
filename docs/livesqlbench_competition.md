# First Loop

This repository contains one deliberately small vertical slice for LiveSQLBench:
two public SQLite Query tasks, one independently authored and execution-verified
training target, one independently authored and execution-verified dev target,
one deterministic base-model evaluation, one direct SQL LoRA pass, and one
deterministic adapter evaluation.

The public task file must have empty protected fields. Targets are supplied in a
separate JSONL file and must use `target_source: independent_verified` and
`verification.status: execution_verified`. The prepare command executes each
target twice against the declared public SQLite database before writing artifacts.

Run the loop with the external paths supplied by the operator:

To run the complete real one-example-per-lane loop in one command:

```bash
uv run --no-sync python -m sqlbench_lab.cli loop \
  --public-data /path/to/livesqlbench_data_sqlite.jsonl \
  --target-manifest /path/to/targets.jsonl \
  --db-root /path/to/livesqlbench-base-lite-sqlite \
  --model-path /path/to/local/model \
  --output-dir /path/to/first-loop-run
```

This command does not compare scores or stop after a failed evaluation. It writes
the prepared train/dev rows, base evaluation, adapter, adapter evaluation, and
`loop-summary.json` under the output directory.

```bash
uv run --no-sync python -m sqlbench_lab.cli prepare \
  --public-data /path/to/livesqlbench_data_sqlite.jsonl \
  --target-manifest /path/to/targets.jsonl \
  --db-root /path/to/livesqlbench-base-lite-sqlite \
  --train-output /path/to/first-loop/train.jsonl \
  --dev-output /path/to/first-loop/dev.jsonl

uv run --no-sync python -m sqlbench_lab.cli eval \
  --dataset /path/to/first-loop/dev.jsonl \
  --model-path /path/to/local/model \
  --output /path/to/first-loop/base-eval.json

uv run --no-sync python -m sqlbench_lab.cli train \
  --dataset /path/to/first-loop/train.jsonl \
  --model-path /path/to/local/model \
  --adapter-output /path/to/first-loop/adapter

uv run --no-sync python -m sqlbench_lab.cli eval \
  --dataset /path/to/first-loop/dev.jsonl \
  --model-path /path/to/local/model \
  --adapter-path /path/to/first-loop/adapter \
  --output /path/to/first-loop/adapter-eval.json
```

Local results are lab measurements, not official LiveSQLBench scores.
