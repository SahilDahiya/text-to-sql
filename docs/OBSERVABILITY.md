# Experiment Observability

read_when: you are running training/eval experiments or comparing base vs adapter runs

## Purpose

Use MLflow as the local dashboard for SQL experiment comparison.

The durable experiment record is still the checked-in manifest, checked-in datasets, git
commit, adapter artifact, and JSON summaries. MLflow is a browser over those runs, not a
replacement for repo artifacts.

## Start The Dashboard

From the repo root:

```bash
uv run --group observability python -m sqlbench_lab.cli observe ui
```

The command starts MLflow at:

```text
http://127.0.0.1:5000
```

By default the backend store is:

```text
sqlite:///./mlflow.db
```

Override it when needed:

```bash
uv run --group observability python -m sqlbench_lab.cli observe ui \
  --backend-store-uri sqlite:///./mlflow.db \
  --host 127.0.0.1 \
  --port 5001
```

## Log Runs

Train with MLflow logging:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp002_spider_bird_sft.json \
  --mlflow
```

Evaluate base and adapter on the same dataset:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp002_spider_bird_sft.json \
  --dataset datasets/sql/eval/spider_validation_sample_v1.jsonl \
  --model base \
  --mlflow
```

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp002_spider_bird_sft.json \
  --dataset datasets/sql/eval/spider_validation_sample_v1.jsonl \
  --model adapter \
  --mlflow
```

## Run Naming

Run names are intentionally sortable:

- `exp002/train`
- `exp002/eval/spider_validation_sample_v1/base`
- `exp002/eval/spider_validation_sample_v1/adapter`
- `exp002/eval/bird_validation_sample_v1/base`
- `exp002/eval/bird_validation_sample_v1/adapter`

Use these MLflow tags for filtering:

- `sqlbench.experiment_id`
- `sqlbench.run_kind`: `train` or `eval`
- `sqlbench.dataset_name`
- `sqlbench.dataset_family`: `smoke`, `spider`, or `bird`
- `sqlbench.model_variant`: `base` or `adapter`
- `sqlbench.stage`
- `sqlbench.base_model`
- `sqlbench.git_commit`

## Dashboard Views

For training runs, compare:

- `summary.train_row_count`
- `summary.trainable_parameters`
- `summary.total_parameters`
- `trainer.train_loss`
- `trainer.train_runtime`

For eval runs, compare:

- `eval.pass_rate`
- `eval.passed_count`
- `eval.case_count`

The immediate base-vs-adapter view should filter to one dataset at a time. Comparing Spider
and BIRD in the same chart is useful for trend spotting, but it should not be read as a
single score.

## Rules

- Do not call local Spider/BIRD slices official benchmark scores.
- Do not compare base and adapter runs unless they used the same eval JSONL.
- Do not mix smoke, Spider, and BIRD rows into one headline metric.
- Keep MLflow state out of git; `mlflow.db`, `mlruns/`, `artifacts/`, and `results/` are local.
