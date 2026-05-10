# SQL Training Pipeline

read_when: you are changing SQL training data, evaluation, or fine-tuning code

## Purpose

Build a reproducible training lane for a small SQL student:

- `Qwen/Qwen3.5-0.8B-Base`

The served inference lane may still use an Ollama-style name such as `qwen3.5:0.8b`, but
training manifests must point at a trainable Hugging Face model ID or local model path.

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
- `schemas/sql_sft_experiment_v1.schema.json`
- `datasets/sql/smoke/sql_smoke_v1.jsonl`
- `datasets/sql/train/qwen35_0_8b_direct_sql_seed_v1.jsonl`
- `experiments/sql/qwen35_0_8b__exp001_sql_sft.json`
- generated SQLite fixture support for `company_small`
- strict JSONL loaders
- SQL SFT manifest loading
- minimal SQL LoRA SFT runner with explicit `--dry-run`
- optional MLflow experiment logging for SQL SFT runs
- direct-SQL and repair prompt renderers
- result-equivalence SQLite evaluation
- base-vs-adapter smoke evaluation CLI with JSON result output
- Hugging Face benchmark import for PremSQL-style Spider and BIRD snapshots

## Experiment Observability

The source of truth for an experiment remains:

- the checked-in experiment manifest
- checked-in train/eval dataset files
- the git commit
- generated adapter artifacts under `artifacts/`
- generated train/eval summary JSON files

MLflow is used as a local run browser and comparison layer, not as the durable contract.

Enable MLflow for a run with either:

```bash
SQLBENCH_MLFLOW=1 uv run --group training --group observability python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp001_sql_sft.json
```

or:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp001_sql_sft.json \
  --mlflow
```

By default, local MLflow tracking state is written to `sqlite:///./mlflow.db`, which is
ignored by git. MLflow artifact directories such as `./mlruns` are also ignored. Override
the tracking location with `SQLBENCH_MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_URI`, or
the CLI flag `--mlflow-tracking-uri`.

Start the local UI with:

```bash
uv run --group observability python -m sqlbench_lab.cli observe ui
```

See `OBSERVABILITY.md` for run naming, dashboard filters, and comparison rules.

The SQL SFT logger records:

- experiment ID, stage, method, base model, adapter name, and git commit
- train dataset row counts and smoke eval case count
- LoRA and trainer hyperparameters
- train summary metrics, trainer metrics, manifest, train summary, and adapter config
- eval dataset name, dataset family, pass rate, per-case pass/fail metrics, manifest, and eval result JSON

Run base and adapter smoke evals with:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp001_sql_sft.json \
  --model base \
  --mlflow
```

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp001_sql_sft.json \
  --model adapter \
  --mlflow
```

## Real Benchmark Datasets

The real dataset lane follows the PremSQL reference repo:

- Spider: `premai-io/spider`
- BIRD/BirdBench: `premai-io/birdbench`

Both are imported from Hugging Face snapshots because the snapshots include SQLite databases,
not just question/SQL parquet rows. Downloaded benchmark snapshots live under `external/`,
which is ignored by git.

Import a small Spider train slice:

```bash
uv run --group training python -m sqlbench_lab.cli sql import-benchmark \
  --benchmark spider \
  --split train \
  --artifact train \
  --limit 100 \
  --output datasets/sql/train/spider_train_sample_v1.jsonl
```

Import a BIRD validation eval slice:

```bash
uv run --group training python -m sqlbench_lab.cli sql import-benchmark \
  --benchmark bird \
  --split validation \
  --artifact eval \
  --limit 25 \
  --output datasets/sql/eval/bird_validation_sample_v1.jsonl
```

Run eval against an imported real dataset:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp001_sql_sft.json \
  --dataset datasets/sql/eval/bird_validation_sample_v1.jsonl \
  --model adapter \
  --mlflow
```

Analyze a completed eval result before choosing repair work:

```bash
uv run python -m sqlbench_lab.cli sql analyze-eval \
  --result results/sql/qwen35_0_8b__exp002_spider_bird_sft/adapter__bird_validation_sample_v1.json
```

The analysis writes a sibling `.analysis.json` file with counts for schema errors, syntax
errors, execution errors, row-count mismatches, and row-value mismatches. Use this before
adding repair retries; syntax/schema failures should get execution-guided repair first,
while clean wrong-result failures usually need more examples, schema linking, or retrieval.

Collect repair rows from the strongest execution-visible failures:

```bash
uv run python -m sqlbench_lab.cli sql collect-repair-data \
  --result results/sql/qwen35_0_8b__exp002_spider_bird_sft/adapter__bird_validation_sample_v1.json \
  --eval-dataset datasets/sql/eval/bird_validation_sample_v1.jsonl \
  --output datasets/sql/repair/bird_dev_repair_seed_v1.jsonl \
  --strong-only
```

Validate the collected repair file:

```bash
uv run python -m sqlbench_lab.cli sql validate-repair \
  --dataset datasets/sql/repair/bird_dev_repair_seed_v1.jsonl
```

The output uses `sql_repair_example:v1` rows:

`question + schema + failed SQL + execution observation -> gold SQL`

Rows collected from an eval/dev slice must not be used to report a score on that same slice.
Once they become training data, hold out a different eval file for the next measurement.

Run eval-time execution-guided repair without retraining:

```bash
uv run --group training python -m sqlbench_lab.cli sql eval-repair \
  --manifest experiments/sql/qwen35_0_8b__exp002_spider_bird_sft.json \
  --dataset datasets/sql/eval/bird_validation_sample_v1.jsonl \
  --model adapter \
  --max-repair-attempts 1
```

This writes a separate repair eval result JSON under `results/sql/<experiment_id>/`.
It preserves both the first-pass SQL and each repair attempt, and reports:

- first-pass pass rate
- final pass rate after repair
- repair attempt count
- repair success count

By default, repair attempts only run for execution-visible failures:

- empty prediction
- SQL syntax error
- missing table/column schema error
- other SQL execution error

Use `--repair-failure-type` to override the eligible failure buckets. Do not enable repair
for row-count or row-value mismatches until the observation has enough grounding to make the
retry meaningful.

## Next Artifacts To Add

- imported Spider/BIRD train and eval manifests
- execution-repair SFT dataset and runner stage

## Non-Negotiables

- Do not train on hidden benchmark data.
- Do not mix eval rows into train rows.
- Do not report local approximations as official benchmark scores.
- Do not hide fallback behavior in training or eval code.
