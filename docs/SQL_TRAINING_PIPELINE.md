# SQL Training Pipeline

read_when: you are changing SQL training data, evaluation, or fine-tuning code

## Purpose

Build a reproducible training lane for a small SQL student:

- `qwen3.5:0.8b`

The first goal is measurable improvement over the base model on execution-based SQL evals.
The later goal is to use the trained student inside a LiveSQLBench-capable agent loop.

## Boundary

This lane is not a generic chatbot project.

The core supervised task is:

`question + schema + optional knowledge -> SQL`

The core repair task is:

`question + schema + failed SQL + execution error -> corrected SQL`

The agent task is:

`schema lookup + SQL execution observations -> final submitted SQL`

## Stages

### Stage 0: Baseline

Before training, run the base `qwen3.5:0.8b` model on:

- a tiny SQL smoke set
- a local execution-based dev set
- LiveSQLBench Base-Lite when the official runner is wired

### Stage 1: Direct SQL SFT

Train on clean supervised rows:

- prompt contains question, SQL dialect, schema text, and optional knowledge
- target is the assistant SQL only
- prompt/context tokens are masked from loss

### Stage 2: Execution Repair SFT

Train on correction rows:

- failed SQL
- database error or wrong-result note
- corrected SQL target

Keep this stage separately labeled so gains are attributable.

### Stage 3: Tool Trajectory SFT

Train the model on structured tool-use traces:

1. inspect schema
2. draft SQL
3. execute SQL
4. observe result or error
5. submit final SQL

This is the more realistic path for LiveSQLBench agent mode.

## Current Foundation

The first checked-in foundation includes:

- `schemas/sql_train_example_v1.schema.json`
- `schemas/sql_repair_example_v1.schema.json`
- `schemas/sql_eval_case_v1.schema.json`
- `datasets/sql/smoke/sql_smoke_v1.jsonl`
- generated SQLite fixture support for `company_small`
- strict JSONL loaders
- direct-SQL and repair prompt renderers
- result-equivalence SQLite evaluation

## Next Artifacts To Add

- `schemas/sql_sft_experiment_v1.schema.json`
- `experiments/sql/qwen35_0_8b__exp001_sql_sft.json`
- CLI commands for SQL dataset validation and smoke evaluation
- baseline capture for untrained `qwen3.5:0.8b`

## Non-Negotiables

- Do not train on hidden benchmark data.
- Do not mix eval rows into train rows.
- Do not report local approximations as official benchmark scores.
- Do not hide fallback behavior in training or eval code.
