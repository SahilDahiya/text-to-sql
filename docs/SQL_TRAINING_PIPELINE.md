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
- selectable prompt style in manifests: `canonical_chat` or `premsql_text`
- result-equivalence SQLite evaluation
- base-vs-adapter smoke evaluation CLI with JSON result output
- Hugging Face benchmark import for PremSQL-style Spider and BIRD snapshots
- stratified benchmark import for BIRD slices that should cover multiple databases

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

## Exp003 One-Shot Gate

The current one-shot improvement lane is:

- train: `datasets/sql/train/spider_train_100_v1.jsonl`
- train: `datasets/sql/train/bird_train_100_v1.jsonl`
- eval: `datasets/sql/eval/spider_validation_25_v1.jsonl`
- eval: `datasets/sql/eval/bird_validation_25_v1.jsonl`
- manifest: `experiments/sql/qwen35_0_8b__exp003_one_shot_spider_bird_sft.json`

This experiment is direct SQL only:

`question + schema + optional knowledge -> SQL`

Do not include repair examples, failed SQL, execution observations, or retry loops in this
score. The eval commands must use `sql eval`, not `sql eval-repair`.

Train with MLflow:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp003_one_shot_spider_bird_sft.json \
  --mlflow
```

Evaluate the fixed local slices:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp003_one_shot_spider_bird_sft.json \
  --dataset datasets/sql/eval/spider_validation_25_v1.jsonl \
  --model base \
  --mlflow
```

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp003_one_shot_spider_bird_sft.json \
  --dataset datasets/sql/eval/spider_validation_25_v1.jsonl \
  --model adapter \
  --mlflow
```

Repeat the same base/adapter pair for `datasets/sql/eval/bird_validation_25_v1.jsonl`.
MLflow logs pass rate plus failure buckets so we can tell whether a one-shot gain comes
from fewer schema errors, fewer syntax errors, or fewer wrong-result queries.

## Exp004 Expanded One-Shot Gate

Exp004 keeps the exp003 eval files fixed and only expands the training slice:

- train: `datasets/sql/train/spider_train_250_v1.jsonl`
- train: `datasets/sql/train/bird_train_250_v1.jsonl`
- eval: `datasets/sql/eval/spider_validation_25_v1.jsonl`
- eval: `datasets/sql/eval/bird_validation_25_v1.jsonl`
- manifest: `experiments/sql/qwen35_0_8b__exp004_one_shot_spider_bird_sft.json`

This lets us compare exp003 and exp004 without moving the local scoring target. The scoring
path remains direct one-shot SQL only through `sql eval`.

## Exp005 BIRD Schema-Grounded One-Shot Gate

Exp005 keeps the exp003/exp004 eval files fixed and narrows the training change to BIRD
schema grounding:

- train: `datasets/sql/train/bird_schema_grounded_token1024_120_v1.jsonl`
- train: `datasets/sql/train/spider_train_100_v1.jsonl`
- eval: `datasets/sql/eval/spider_validation_25_v1.jsonl`
- eval: `datasets/sql/eval/bird_validation_25_v1.jsonl`
- manifest: `experiments/sql/qwen35_0_8b__exp005_bird_schema_grounded_one_shot_sft.json`

The BIRD slice is selected from the cached full BIRD train split, capped at 1024 rendered
tokens per row, and biased toward exact SQLite identifier quoting, joins, casts, division,
and evidence-bearing rows. This cap matters: an uncapped schema-grounding slice can turn the
run into an oversized-context throughput test instead of a one-shot SQL quality test.

Local exp005 result:

- BIRD adapter: `1/25`, with schema/syntax failures still dominant.
- Spider adapter: `18/25`, matching the exp003/exp004 guardrail.

Read this as a small recovery from exp004 BIRD `0/25`, not a win over exp003 BIRD `2/25`.
The next one-shot improvement should target validation-like BIRD schemas and quoted-column
copying more directly, not just add more generic BIRD rows.

## Exp006 Identifier-Copy One-Shot Gate

Exp006 keeps the same fixed eval files and changes two things:

- The SQL system prompt explicitly requires exact schema identifier copying and backtick
  quoting for SQLite identifiers with spaces, punctuation, parentheses, percent signs,
  hyphens, or question marks.
- The train mix adds `datasets/sql/train/bird_identifier_copy_token1536_87_v1.jsonl`, a
  synthetic train-schema-only BIRD slice that asks simple select/count/distinct questions
  over awkward column names. It does not use BIRD validation rows.

Train mix:

- train: `datasets/sql/train/bird_identifier_copy_token1536_87_v1.jsonl`
- train: `datasets/sql/train/bird_schema_grounded_token1024_120_v1.jsonl`
- train: `datasets/sql/train/spider_train_100_v1.jsonl`
- manifest: `experiments/sql/qwen35_0_8b__exp006_identifier_copy_one_shot_sft.json`

Because the prompt changed, compare exp006 carefully: adapter-vs-base under the same prompt
is clean, while exp005-vs-exp006 combines prompt and data changes.

Local exp006 result:

- BIRD base under exp006 prompt: `0/25`.
- BIRD adapter: `2/25`; schema failures dropped versus exp005, but syntax failures rose.
- Spider adapter: `18/25`, matching the exp003/exp004/exp005 guardrail.

Read this as real adapter signal over base, but not yet a new best over exp003. The next
one-shot step should reduce malformed long SQL after identifier copying, likely by adding
short, validation-shaped arithmetic examples rather than more generic identifier-copy rows.

## Exp007 TRL SFTTrainer Backend Gate

Exp007 is a tooling-control experiment. It keeps the exp006 prompt and train mix fixed, but
switches the SFT backend from the repo custom `transformers.Trainer` path to TRL
`SFTTrainer`.

Train mix:

- train: `datasets/sql/train/bird_identifier_copy_token1536_87_v1.jsonl`
- train: `datasets/sql/train/bird_schema_grounded_token1024_120_v1.jsonl`
- train: `datasets/sql/train/spider_train_100_v1.jsonl`
- manifest: `experiments/sql/qwen35_0_8b__exp007_trl_sft_identifier_copy.json`

Backend settings:

- `trainer.backend`: `trl_sft_trainer`
- packing disabled
- prompt/completion dataset format
- completion-only loss enabled so prompt tokens are masked

Success bar:

- BIRD near exp006 adapter `2/25`.
- Spider near exp006 adapter `18/25`.
- No generated-format regression.
- Runtime is not materially worse than exp006.

Local exp007 result:

- BIRD adapter: `3/25`, the best fixed-BIRD result so far.
- Spider adapter: `17/25`, one point below the exp006 guardrail.
- Train runtime: about `1346s`, slower than exp006's about `1026s` on the same train mix.

Read this as a useful but mixed trainer migration: TRL improved BIRD, likely through slightly
different completion masking/collation mechanics, but did not preserve the Spider guardrail
and was slower on this dataset. Before moving to bitsandbytes, decide whether BIRD gain is
worth the Spider/runtime tradeoff or whether exp008 should use TRL plus a small Spider
stabilizer.

## Exp008 TRL Packing Runtime Trial

Exp008 keeps the exp007 data recipe fixed and tests whether TRL sequence packing can make
the same one-shot SFT loop faster without sacrificing fixed-eval quality.

Train mix:

- train: `datasets/sql/train/bird_identifier_copy_token1536_87_v1.jsonl`
- train: `datasets/sql/train/bird_schema_grounded_token1024_120_v1.jsonl`
- train: `datasets/sql/train/spider_train_100_v1.jsonl`
- manifest: `experiments/sql/qwen35_0_8b__exp008_trl_packing_identifier_copy.json`

Backend settings:

- `trainer.backend`: `trl_sft_trainer`
- `packing`: `true`
- `packing_strategy`: `bfd`
- `max_length`: `1024`
- `bf16`: `true`
- `tf32`: `false`
- `gradient_checkpointing`: `false`

Implementation notes:

- TRL rejects packing when it receives the Qwen `AutoProcessor`, because it treats the run
  as vision-language training. The TRL backend now passes the inner tokenizer to
  `SFTTrainer` while preserving the existing processor/tokenizer save path.
- `tf32=true` failed on the local GPU/runtime, so the checked-in exp008 manifest explicitly
  disables it.
- TRL warned that BFD packing enables padding-free training without a supported flash
  attention implementation. This makes exp008 a useful speed/quality measurement, not a
  recipe to promote blindly.

Local exp008 result:

- Packed train sequences: `194` from `307` source rows.
- Train runtime: about `675s`, faster than exp007's about `1346s`.
- Train loss: about `0.3568`.
- BIRD adapter: `0/25`; failures were schema `12`, syntax `7`, execution `1`, row-count
  `1`, row-value `4`.
- Spider adapter: `14/25`; failures were schema `4`, row-count `3`, row-value `4`.

Read this as a rejected fast recipe. Packing materially improved runtime, but it failed both
quality gates: BIRD dropped below exp007's `3/25`, and Spider dropped below the `18/25`
guardrail. Do not stack Liger or bitsandbytes on this exact recipe. The next tooling step
should either add a supported flash-attention implementation for packed TRL or return to
unpacked TRL and improve the data/prompt recipe.

## Exp009 Packed TRL Flash-Attention Gate

Exp009 is the direct follow-up to exp008. It keeps the exp008 prompt, train mix, LoRA
config, packing config, and fixed eval plan unchanged, but requests a TRL-supported
attention implementation for BFD packing.

Train mix:

- train: `datasets/sql/train/bird_identifier_copy_token1536_87_v1.jsonl`
- train: `datasets/sql/train/bird_schema_grounded_token1024_120_v1.jsonl`
- train: `datasets/sql/train/spider_train_100_v1.jsonl`
- manifest: `experiments/sql/qwen35_0_8b__exp009_trl_packing_flash_attention_identifier_copy.json`

Backend settings:

- `trainer.backend`: `trl_sft_trainer`
- `attn_implementation`: `kernels-community/flash-attn2`
- `packing`: `true`
- `packing_strategy`: `bfd`
- `max_length`: `1024`
- `bf16`: `true`
- `tf32`: `true`
- `gradient_checkpointing`: `false`

Implementation notes:

- `trainer.attn_implementation` now flows through the manifest, MLflow trainer config, train
  model load, and eval model load.
- The `training` dependency group includes `kernels>=0.14.0`, which lets Transformers load
  `kernels-community/flash-attn2`.
- On the current local GPU, `NVIDIA GeForce RTX 2080 Ti` / compute capability `7.5`,
  `kernels-community/flash-attn2` loads but fails on first forward pass with
  `FlashAttention only supports Ampere GPUs or newer`.

Local exp009 status:

- Manifest validation passed.
- Dry-run SFT passed.
- Unit tests passed.
- Full train/eval is blocked on this machine because the supported flash-attention backend
  requires Ampere-or-newer hardware.

Cloud exp009 result:

- Instance GPU: `NVIDIA RTX A6000`, compute capability `8.6`.
- Flash-attention preflight passed with `kernels-community/flash-attn2`.
- TRL packed training ran without the unsupported packed-attention warning seen in exp008.
- Packed train sequences: `194` from `307` source rows.
- Train runtime: about `464s`, faster than exp008's about `675s` and exp007's about
  `1346s`.
- Train loss: about `0.3557`.
- BIRD adapter: `2/25`; failures were schema `8`, syntax `12`, row-count `1`, row-value
  `2`.
- Spider adapter: `14/25`; failures were schema `4`, syntax `1`, row-count `2`, row-value
  `4`.

Read this as a technically valid packed-TRL run, but still a rejected quality recipe. The
FlashAttention path removes the packing safety warning and improves runtime, but it does not
recover exp007 BIRD `3/25` or the Spider `18/25` guardrail. Do not proceed to Liger or
bitsandbytes on this packed recipe until the quality issue is understood.

## Exp010-Exp017 Local Unpacked Overnight Queue

The next local queue returns to unpacked one-shot SFT on the RTX 2080 Ti. It avoids
FlashAttention, packing, Liger, bitsandbytes, repair, and eval-time retries. The goal is to
separate data-mix quality from runtime-tooling effects.

Queue runner:

```bash
uv run python scripts/run_sql_experiment_queue.py --mlflow \
  experiments/sql/qwen35_0_8b__exp010_trl_schema_spider100_unpacked.json \
  experiments/sql/qwen35_0_8b__exp011_trl_bird100_spider100_unpacked.json \
  experiments/sql/qwen35_0_8b__exp012_trl_bird250_spider100_unpacked.json \
  experiments/sql/qwen35_0_8b__exp013_trl_schema_spider250_unpacked.json \
  experiments/sql/qwen35_0_8b__exp014_trl_identifier_schema_spider250_unpacked.json \
  experiments/sql/qwen35_0_8b__exp015_trl_identifier_schema_spider100_lr1e4_unpacked.json \
  experiments/sql/qwen35_0_8b__exp016_transformers_schema_spider100.json \
  experiments/sql/qwen35_0_8b__exp017_transformers_bird100_spider100.json
```

Each experiment runs:

- manifest validation
- SFT train
- fixed BIRD 25 adapter eval
- BIRD failure analysis
- fixed Spider 25 adapter eval
- Spider failure analysis

Experiment intent:

- exp010: TRL, BIRD schema-grounded 120 + Spider 100.
- exp011: TRL, real BIRD 100 + Spider 100.
- exp012: TRL, real BIRD 250 + Spider 100.
- exp013: TRL, BIRD schema-grounded 120 + Spider 250.
- exp014: TRL, identifier-copy 87 + BIRD schema-grounded 120 + Spider 250.
- exp015: TRL, exp007 train mix with lower learning rate `1e-4`.
- exp016: custom `transformers.Trainer`, exp010 train mix.
- exp017: custom `transformers.Trainer`, exp011 train mix.

Success target:

- Any BIRD result above exp007 `3/25`.
- Spider at or above the exp006 guardrail `18/25`.
- If no run beats BIRD, prefer the run with the fewest syntax/schema failures for the next
  data-building step.

Local queue result:

| Experiment | Train rows | Runtime | BIRD | Spider | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| exp010 | 220 | `498s` | `1/25` | `20/25` | Good Spider, weak BIRD. |
| exp011 | 200 | `465s` | `1/25` | `19/25` | Real BIRD 100 did not help BIRD. |
| exp012 | 350 | `923s` | `0/25` | `16/25` | More real BIRD regressed both. |
| exp013 | 370 | `899s` | `2/25` | `20/25` | Best balanced no-identifier run. |
| exp014 | 457 | `1078s` | `3/25` | `19/25` | Best queue run; matches exp007 BIRD and beats Spider. |
| exp015 | 307 | `1051s` | `2/25` | `18/25` | Lower LR preserves Spider guardrail but loses BIRD. |
| exp016 | 220 | `430s` | `0/25` | `20/25` | Custom trainer is fast/good Spider, bad BIRD. |
| exp017 | 200 | `421s` | `1/25` | `20/25` | Custom trainer real BIRD still weak. |

No run beat exp007's BIRD `3/25`, but exp014 is a better balanced run than exp007:
it matches BIRD `3/25` and improves Spider from `17/25` to `19/25`. The queue also shows
that simply adding more real BIRD rows is not enough; exp012 regressed to BIRD `0/25`.
The next one-shot data step should start from exp014 and target BIRD syntax/schema failures
without giving up the Spider-250 guardrail.

## BIRD Dataset Gap From PremSQL Reference

The PremSQL reference keeps `db_path` as a required row field and builds schema text from the
live SQLite database at prompt time. This repo now preserves `db_path` on imported train rows
too, while keeping it optional for older checked-in train data. That matters because BIRD
work needs database-derived enrichment later: value previews, executable filtering, table
stats, and prompt regeneration.

The old first-N BIRD slices were heavily biased:

- `bird_train_100_v1`: 100 rows from only `movie_platform`
- `bird_train_250_v1`: 250 rows from only 3 databases
- `bird_validation_25_v1`: 25 rows from only `california_schools`

The new stratified files are:

- `datasets/sql/train/bird_train_stratified_500_v1.jsonl`: 500 rows across all 69 BIRD train DBs
- `datasets/sql/eval/bird_validation_stratified_25_v1.jsonl`: quick BIRD check across all 11 validation DBs
- `datasets/sql/eval/bird_validation_stratified_110_v1.jsonl`: broader BIRD check with 10 rows per validation DB

Generate a stratified BIRD train slice:

```bash
uv run python -m sqlbench_lab.cli sql import-benchmark \
  --benchmark bird \
  --split train \
  --artifact train \
  --limit 500 \
  --selection stratified \
  --output datasets/sql/train/bird_train_stratified_500_v1.jsonl
```

Generate the broader stratified BIRD eval slice:

```bash
uv run python -m sqlbench_lab.cli sql import-benchmark \
  --benchmark bird \
  --split validation \
  --artifact eval \
  --limit 110 \
  --selection stratified \
  --output datasets/sql/eval/bird_validation_stratified_110_v1.jsonl
```

## Exp018-Exp019 PremSQL-Style BIRD Follow-Up

Exp018 isolates the prompt change. It starts from exp014's best balanced recipe
(`3/25` BIRD, `19/25` Spider) and changes only the prompt style to `premsql_text`.

```bash
uv run python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp018_trl_premsql_prompt_identifier_schema_spider250.json \
  --mlflow
```

After training, run both old and stratified BIRD checks:

```bash
uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp018_trl_premsql_prompt_identifier_schema_spider250.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_validation_25_v1.jsonl \
  --mlflow

uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp018_trl_premsql_prompt_identifier_schema_spider250.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_validation_stratified_110_v1.jsonl \
  --mlflow
```

Exp019 then adds stratified real BIRD train coverage on top of the exp014-style recipe:
identifier-copy rows, schema-grounded rows, stratified 500-row BIRD train, and Spider 250.

```bash
uv run python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp019_trl_premsql_prompt_stratified_bird_spider250.json \
  --mlflow
```

Read exp019 as a data-coverage test, not a clean prompt-only comparison. The main pass
condition is improvement over exp014 on the old fixed BIRD 25 without falling below the
Spider guardrail, plus a non-broken result on the stratified 110-row BIRD slice.

Exp018 local result:

| Eval | Result | Failure counts | Read |
| --- | ---: | --- | --- |
| Old fixed BIRD 25 | `2/25` | schema `10`, syntax `6`, row-count `1`, row-value `6` | Regresses from exp014 `3/25`. |
| Spider 25 | `18/25` | schema `2`, row-count `1`, row-value `4` | One point below exp014 but still at guardrail. |
| Stratified BIRD 110 | `13/110` | execution `4`, schema `34`, syntax `11`, row-count `20`, row-value `28` | Broad BIRD remains weak; schema failures dominate. |

Training metrics:

- train rows: `457`
- runtime: `1776s`
- train loss: `0.2933`
- mean token accuracy at final trainer log: about `0.9638`

Read this as a rejected prompt-only recipe. The PremSQL-style prompt made the prompt longer
and did not improve BIRD. Keep the stratified BIRD eval files, but do not treat
`premsql_text` as the default next direction unless a data-coverage run proves otherwise.
The cleaner next follow-up is exp020: canonical prompt plus stratified BIRD coverage, so
prompt and data effects stay separable.

```bash
uv run python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp020_trl_canonical_stratified_bird_spider250.json \
  --mlflow
```

## Exp021 Single-DB BIRD Schema-Linking Lab

The broad BIRD failures show that the model is not reliably schema-linking unseen BIRD
schemas. Exp021 narrows the problem to one train-split BIRD database so we can debug the
skill before scaling. This is not a BIRD benchmark score.

Lab database:

- BIRD train DB: `superstore`
- reason: multiple regional fact tables, `people` and `product` joins, FK relationships,
  and many awkward identifiers such as `Customer ID`, `Order ID`, `Order Date`,
  `Ship Mode`, `Product ID`, `Product Name`, and `Sub-Category`

Generate the lab datasets:

```bash
uv run python -m sqlbench_lab.cli sql generate-bird-lab \
  --db-id superstore \
  --train-output datasets/sql/train/bird_superstore_schema_lab_train_v1.jsonl \
  --eval-output datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl
```

Generated artifacts:

- train: `datasets/sql/train/bird_superstore_schema_lab_train_v1.jsonl`
- dev eval: `datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl`
- train rows: `40`
- dev rows: `40`
- SQL overlap between train targets and dev gold SQL: `0`

The lab covers:

- exact quoted identifier copying
- value filters
- two-table product joins
- two-table customer joins
- three-table joins
- quoted identifier arithmetic
- computed `ORDER BY`
- date extraction
- grouped aggregates
- `HAVING`

Train exp021:

```bash
uv run python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp021_trl_superstore_schema_lab.json \
  --mlflow
```

Evaluate the lab adapter:

```bash
uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp021_trl_superstore_schema_lab.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --mlflow
```

Pass condition: high execution accuracy on the heldout `superstore` lab dev set. If this
fails, broad BIRD scaling is premature. If this passes, expand the same generator pattern
to 5-10 BIRD train DBs before returning to the stratified BIRD validation slice.

Exp021 local result:

- train rows: `40`
- train runtime: `476s`
- train loss: `0.0436`
- heldout lab dev: `36/40`
- failure counts: schema `4`

The four failures are all the same structural miss in the heldout computed-order pattern:
the model selected `T2.\`Ship Mode\`` from a joined `people` or `product` alias even though
`Ship Mode` belongs to the regional fact table. This is the first useful lab signal: the
model can learn most of the single-DB schema-linking curriculum, but the next generator pass
needs direct fact-table computed-order rows in train, not only joined computed-order rows.
Do not scale to broad BIRD from this exact curriculum until that failure family is covered.

## Exp022 Superstore Computed-Order Curriculum Fix

Exp022 keeps the Exp021 heldout dev set fixed and changes only the training curriculum.

Intent:

- fix the four Exp021 computed-order schema errors
- teach direct fact-table computed ordering before adding more DBs
- preserve the same-DB dev comparison against `bird_superstore_schema_lab_dev_v1.jsonl`

Generate train_v2:

```bash
uv run python -m sqlbench_lab.cli sql generate-bird-lab \
  --db-id superstore \
  --curriculum-version v2 \
  --train-output datasets/sql/train/bird_superstore_schema_lab_train_v2.jsonl \
  --eval-output /tmp/bird_superstore_schema_lab_dev_v2_check.jsonl
```

Generated train_v2 adds direct fact-table computed-order rows:

- `Ship Mode` ordered by sales per ordered unit
- `Customer ID` ordered by profit per sales dollar

The fixed comparison eval remains:

- `datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl`

Audit leakage:

```bash
uv run python -m sqlbench_lab.cli sql audit-leakage \
  --train-dataset datasets/sql/train/bird_superstore_schema_lab_train_v2.jsonl \
  --eval-dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl
```

Train exp022:

```bash
uv run python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp022_trl_superstore_computed_order_v2.json \
  --mlflow
```

Evaluate exp022:

```bash
uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp022_trl_superstore_computed_order_v2.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --mlflow
```

Pass condition: at least `39/40` on the fixed superstore dev set and no remaining
computed-order schema errors for `Ship Mode`.

Exp022 local result:

- train rows: `48`
- train runtime: `499s`
- train loss: `0.0423`
- fixed heldout lab dev: `40/40`
- failure counts: none

The added direct fact-table computed-order rows closed the Exp021 failure family. The four
previous `Ship Mode` computed-order cases now predict `Ship Mode` from the regional fact
table instead of a joined alias. This is still same-DB generalization, not unseen-DB
generalization, but the generator pattern is now strong enough to reuse when adding
`regional_sales`.

## Exp023 Two-DB BIRD Train-Lab

Exp023 adds `regional_sales` as the second BIRD train-lab DB. This is still train-DB dev,
not unseen-DB generalization.

Regional sales lab database:

- BIRD train DB: `regional_sales`
- reason: table names and columns with spaces, one fact table (`Sales Orders`), multiple
  dimensions (`Customers`, `Products`, `Regions`, `Store Locations`, `Sales Team`), text
  currency fields with commas, and region joins through store locations

Generate the regional_sales lab:

```bash
uv run python -m sqlbench_lab.cli sql generate-bird-lab \
  --db-id regional_sales \
  --train-output datasets/sql/train/bird_regional_sales_schema_lab_train_v1.jsonl \
  --eval-output datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl
```

Generated artifacts:

- train: `datasets/sql/train/bird_regional_sales_schema_lab_train_v1.jsonl`
- dev eval: `datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl`
- train rows: `40`
- dev rows: `40`
- SQL overlap between train targets and dev gold SQL: `0`

The lab covers:

- exact quoted identifier copying
- value filters over `Sales Channel`
- product joins
- customer joins
- multi-table customer/product/region joins
- quoted identifier arithmetic over `Order Quantity` and text `Unit Price`
- direct fact-table computed `ORDER BY`
- text date suffix filtering
- grouped aggregates
- `HAVING`

Audit the two train-DB dev lanes together:

```bash
uv run python -m sqlbench_lab.cli sql audit-leakage \
  --train-dataset datasets/sql/train/bird_superstore_schema_lab_train_v2.jsonl \
  --train-dataset datasets/sql/train/bird_regional_sales_schema_lab_train_v1.jsonl \
  --eval-dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --eval-dataset datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl
```

Train exp023:

```bash
uv run python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp023_trl_superstore_regional_sales.json \
  --mlflow
```

Evaluate train-DB dev lanes separately:

```bash
uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp023_trl_superstore_regional_sales.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --mlflow

uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp023_trl_superstore_regional_sales.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl \
  --mlflow
```

Pass condition: maintain high `superstore` dev accuracy while learning `regional_sales`
dev. If `superstore` regresses materially, the two-DB training recipe is not stable enough
to expand.

Exp023 local result:

- train rows: `88`
- train runtime: `744s`
- train loss: `0.0431`
- superstore fixed heldout lab dev: `40/40`
- regional_sales heldout lab dev: `37/40`
- regional_sales failure counts: row-value mismatch `3`

The two-DB recipe preserved the Exp022 `superstore` result, so adding `regional_sales` did
not cause same-DB regression. The three `regional_sales` misses are all computed-order
unit-price cases. The model generated the correct joins and grouping, but used
`AVG(T1.\`Unit Price\`)` instead of
`AVG(CAST(REPLACE(T1.\`Unit Price\`, ',', '') AS REAL))`. SQLite executes both, so these
are clean wrong-result failures rather than schema or syntax errors. The next curriculum
fix should add more text-number normalization examples before expanding to `sales`.

## Exp024 Regional Sales Text-Number Normalization Fix

Exp024 keeps both Exp023 dev sets fixed and changes only the `regional_sales` training
curriculum.

Intent:

- fix the three Exp023 `regional_sales` unit-price normalization failures
- teach `CAST(REPLACE(..., ',', '') AS REAL)` for comma-formatted text numeric fields
- preserve the `superstore` `40/40` result while improving `regional_sales`

Generate regional_sales train_v2:

```bash
uv run python -m sqlbench_lab.cli sql generate-bird-lab \
  --db-id regional_sales \
  --curriculum-version v2 \
  --train-output datasets/sql/train/bird_regional_sales_schema_lab_train_v2.jsonl \
  --eval-output /tmp/bird_regional_sales_schema_lab_dev_v2_check.jsonl
```

Generated train_v2 adds text-number normalization rows:

- average normalized `Unit Cost`
- sum normalized `Unit Price`
- average extended normalized `Unit Price`

The fixed comparison eval remains:

- `datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl`
- `datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl`

Audit leakage:

```bash
uv run python -m sqlbench_lab.cli sql audit-leakage \
  --train-dataset datasets/sql/train/bird_superstore_schema_lab_train_v2.jsonl \
  --train-dataset datasets/sql/train/bird_regional_sales_schema_lab_train_v2.jsonl \
  --eval-dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --eval-dataset datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl
```

Train exp024:

```bash
uv run python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp024_trl_regional_sales_normalization_v2.json \
  --mlflow
```

Evaluate exp024:

```bash
uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp024_trl_regional_sales_normalization_v2.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl \
  --mlflow

uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp024_trl_regional_sales_normalization_v2.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --mlflow
```

Pass condition: close the three `regional_sales` row-value mismatches while keeping
`superstore` at or near `40/40`.

Exp024 local result:

- train rows: `100`
- train runtime: `823s`
- train loss: `0.0392`
- superstore fixed heldout lab dev: `40/40`
- regional_sales fixed heldout lab dev: `33/40`
- regional_sales failure counts: schema `4`, row-value mismatch `3`

Decision: reject Exp024 as a curriculum regression. The extra text-number normalization
rows preserved `superstore`, but made `regional_sales` worse than Exp023 (`37/40`).
The original three unit-price failures still missed comma normalization, now generating
`AVG(CAST(T1.\`Unit Price\` AS REAL))` instead of
`AVG(CAST(REPLACE(T1.\`Unit Price\`, ',', '') AS REAL))`. The new four failures are exact
identifier regressions: the model generated `T1.OrderQuantity` instead of quoted
`T1.\`Order Quantity\``. This suggests the v2 additions biased the adapter toward unquoted
identifier shortcuts while still failing to teach the full normalization expression.

Next step: do not expand to `sales` from Exp024. Prefer either a prompt/schema rendering
fix that explicitly marks text numeric columns, or a smaller isolated normalization lab
that also reinforces quoted identifiers in the same rows.

## Exp025 Regional Sales Normalization Micro-Lab

Exp025 returns to the Exp023 stable base and adds a separate train-only micro-lab instead
of using the broader rejected Exp024 `regional_sales` v2 curriculum.

Intent:

- preserve the Exp023 `superstore` `40/40` and `regional_sales` `37/40` baseline
- teach the full expression `CAST(REPLACE(T1.\`Unit Price\`, ',', '') AS REAL)`
- reinforce `T1.\`Order Quantity\`` in the same rows so normalization does not cause
  unquoted identifier regressions

Generate the micro-lab:

```bash
uv run python -m sqlbench_lab.cli sql generate-bird-regional-sales-normalization-lab \
  --train-output datasets/sql/train/bird_regional_sales_normalization_micro_v1.jsonl
```

Train inputs:

- `datasets/sql/train/bird_superstore_schema_lab_train_v2.jsonl`
- `datasets/sql/train/bird_regional_sales_schema_lab_train_v1.jsonl`
- `datasets/sql/train/bird_regional_sales_normalization_micro_v1.jsonl`

Fixed comparison evals:

- `datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl`
- `datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl`

Audit leakage:

```bash
uv run python -m sqlbench_lab.cli sql audit-leakage \
  --train-dataset datasets/sql/train/bird_superstore_schema_lab_train_v2.jsonl \
  --train-dataset datasets/sql/train/bird_regional_sales_schema_lab_train_v1.jsonl \
  --train-dataset datasets/sql/train/bird_regional_sales_normalization_micro_v1.jsonl \
  --eval-dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --eval-dataset datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl
```

Train exp025:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp025_trl_regional_sales_normalization_micro.json \
  --mlflow
```

Evaluate exp025:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp025_trl_regional_sales_normalization_micro.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1.jsonl \
  --mlflow

uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp025_trl_regional_sales_normalization_micro.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --mlflow
```

Pass condition: improve `regional_sales` above Exp023 `37/40`, ideally to `40/40`, while
keeping `superstore` at `40/40` and avoiding unquoted identifier schema failures.

Exp025 local result:

- train rows: `104`
- train runtime: `1217s`
- train loss: `0.0360`
- superstore fixed heldout lab dev: `40/40`
- regional_sales fixed heldout lab dev: `37/40`
- regional_sales failure counts: row-value mismatch `3`

Decision: neutral, not a fix. Exp025 preserved the Exp023 two-DB result and avoided the
Exp024 schema regression, but it did not improve `regional_sales`. The same three
computed-order unit-price cases still generated `AVG(T1.\`Unit Price\`)` instead of
`AVG(CAST(REPLACE(T1.\`Unit Price\`, ',', '') AS REAL))`.

Practical learning: a small side micro-lab can prevent blast-radius regressions, but it
may not override a memorized shorter expression when the heldout target asks for the same
`ORDER BY AVG(unit price) DESC LIMIT 1` shape. The next experiment should make the target
shape explicit in the base regional_sales train rows or change the prompt/schema rendering
to mark text numeric columns before decoding.

## Exp026 Regional Sales Column Value Notes

Exp026 tests prompt grounding rather than more adjacent examples. The failure is not SQL
structure; it is hidden value semantics. `Unit Price` is declared `TEXT`, and the model
needs evidence that numeric aggregation should remove commas before casting.

Contract change:

- SQL train/eval/repair rows may include optional `column_value_notes`.
- Prompt rendering includes these notes in both `canonical_chat` and `premsql_text`.
- Existing datasets remain valid when the field is absent.

Generate note-bearing regional_sales train/dev:

```bash
uv run python -m sqlbench_lab.cli sql generate-bird-lab \
  --db-id regional_sales \
  --curriculum-version v1 \
  --include-column-value-notes \
  --train-output datasets/sql/train/bird_regional_sales_schema_lab_train_v1_column_notes.jsonl \
  --eval-output datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1_column_notes.jsonl
```

Train inputs:

- `datasets/sql/train/bird_superstore_schema_lab_train_v2.jsonl`
- `datasets/sql/train/bird_regional_sales_schema_lab_train_v1_column_notes.jsonl`

Fixed comparison evals:

- `datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl`
- `datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1_column_notes.jsonl`

Audit leakage:

```bash
uv run python -m sqlbench_lab.cli sql audit-leakage \
  --train-dataset datasets/sql/train/bird_superstore_schema_lab_train_v2.jsonl \
  --train-dataset datasets/sql/train/bird_regional_sales_schema_lab_train_v1_column_notes.jsonl \
  --eval-dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --eval-dataset datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1_column_notes.jsonl
```

Train exp026:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql run-sft \
  --manifest experiments/sql/qwen35_0_8b__exp026_trl_regional_sales_column_notes.json \
  --mlflow
```

Evaluate exp026:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp026_trl_regional_sales_column_notes.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_regional_sales_schema_lab_dev_v1_column_notes.jsonl \
  --mlflow

uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp026_trl_regional_sales_column_notes.json \
  --model adapter \
  --dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl \
  --mlflow
```

Pass condition: improve `regional_sales` above Exp023/Exp025 `37/40`, ideally to `40/40`,
while preserving `superstore` at `40/40`.

Exp026 local result:

- train rows: `88`
- train runtime: `815s`
- train loss: `0.0357`
- superstore fixed heldout lab dev: `40/40`
- regional_sales note-bearing heldout lab dev: `37/40`
- regional_sales failure counts: row-value mismatch `3`

Decision: neutral, not a fix. Adding passive column value notes to both training and eval
prompts preserved the stable result, but did not change the three unit-price computed-order
failures. The adapter still generated `AVG(T1.\`Unit Price\`)` instead of
`AVG(CAST(REPLACE(T1.\`Unit Price\`, ',', '') AS REAL))`.

Practical learning: value notes are necessary for principled grounding, but this SFT setup
did not learn to apply them to an already-learned shorter decode path. The next attempt
should make the correction part of the exact target shape, for example with direct
same-shape contrast rows or a stronger renderer rule that ties text numeric notes to
numeric aggregation.

## BIRD DB-Level Expansion Protocol

When expanding beyond `superstore`, treat the database ID as the scientific split unit. The
question is not only whether the adapter memorized one generated lab; the question is when
performance starts transferring to DBs whose schemas were never used in training.

Use three eval lanes for every expansion:

- train-DB dev: generated heldout questions on DBs used for training
- unseen-DB dev: generated heldout questions on DBs never used for training
- benchmark dev: official validation imports only for measurement, never for training rows

Rules:

- Never train on BIRD validation rows or hidden benchmark rows.
- Never mix a DB into both train and unseen-DB dev.
- Same-DB dev may share `db_id` with train, but it must have no exact task, question, or SQL overlap.
- Unseen-DB dev must have no exact task, question, SQL, or `db_id` overlap.
- Record train-DB dev and unseen-DB dev separately in MLflow; do not collapse them into one score.
- A rising train-DB dev score with flat unseen-DB dev score is overfitting/schema memorization, not generalization.

Audit same-DB dev leakage:

```bash
uv run python -m sqlbench_lab.cli sql audit-leakage \
  --train-dataset datasets/sql/train/bird_superstore_schema_lab_train_v1.jsonl \
  --eval-dataset datasets/sql/eval/bird_superstore_schema_lab_dev_v1.jsonl
```

Audit unseen-DB dev leakage:

```bash
uv run python -m sqlbench_lab.cli sql audit-leakage \
  --train-dataset datasets/sql/train/<train_db_lab>.jsonl \
  --eval-dataset datasets/sql/eval/<unseen_db_lab_dev>.jsonl \
  --require-db-disjoint
```

Expansion order:

1. Fix the Exp021 computed-order failure inside `superstore`.
2. Add 2-3 more train DB labs with the same pattern families.
3. Reserve at least 2 BIRD train-split DBs as generated unseen-DB dev labs.
4. Train on the train DB labs only.
5. Evaluate both train-DB dev and unseen-DB dev after every run.
6. Only return to stratified BIRD validation after unseen-DB dev improves.

Initial partition:

- train-lab DBs: `superstore`, then `regional_sales`, `sales`, `bike_share_1`
- unseen-DB dev reserve: `restaurant`, `airline`
- do not move a reserve DB into train without first choosing a new reserve DB and recording
  the change before seeing its eval result

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
