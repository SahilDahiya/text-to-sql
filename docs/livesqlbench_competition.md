# LiveSQLBench Competition Plan

## Objective

Build the strongest allowed model for LiveSQLBench through iterative supervised
fine-tuning. The local repository is a pipeline, not a copy of the benchmark's
protected evaluation environment.

## Data Boundary

Training inputs must be one of:

- public LiveSQLBench development/training rows;
- locally generated rows whose labels are verified against an allowed database;
- public SQL datasets used only when their licensing and split provenance are recorded.

LiveSQLBench ground truth, hidden test cases, protected external knowledge, and
official test databases never enter training artifacts. The package may live outside
git and is referenced by absolute or explicitly configured paths.

## Canonical Loop

1. Assemble a versioned, allowed training mixture.
2. Import only explicit, execution-verified targets from the external task package and validate every row against the v2 train schema.
3. Validate the local development dataset.
4. Train one direct-SQL ISFT continuation.
5. Run one-shot local evaluation.
6. Record the experiment and decision in Linear issue `TAP-532`.
7. Use the official LiveSQLBench runner separately when explicitly requested.

## Gate Policy

The old 12-case storefront `dev_v2` and `eval_v1` suites are deleted. They were
useful for debugging alias binding but are not competition gates.

The first loop reports:

- training-set provenance and row counts;
- one-shot development score;
- per-case execution results;
- official LiveSQLBench runner output, when run.

A local score is never reported as an official benchmark score.

## Current State

The repository deliberately contains no benchmark GT/test-case package and no
competition manifest until the permitted LiveSQLBench development package is present.
The package itself is prompt/environment input. Because its public task payloads may
have empty `sol_sql` and `test_cases`, the next artifact is a separate verified-target
manifest that points to manually or independently execution-verified SQL labels. The
resulting v2 train/eval artifacts and Qwen manifest must pass validation before any
GPU run.

## Code Boundaries

* `livesqlbench_adapter.py` reads task metadata and refuses protected/public target fields.
* `verify-targets` executes each pending target against its declared database before import.
* `models.py`, `loaders.py`, and the v2 JSON schemas enforce provenance and execution verification.
* `evaluator.py` has explicit SQLite and PostgreSQL backends; unsupported dialects fail.
* `eval_runner.py` is one-shot only and emits per-case execution results.

## Verified Target Manifest

The pending target manifest is JSONL with one row per task. It must supply the
target SQL, task split, `order_sensitive`, `numeric_tolerance`, and
`verification.status: pending`. Run `verify-targets`
against the allowed database environment first. Only its output, with
`execution_verified`, may be passed to `livesqlbench-import`; missing fields are
errors rather than inferred defaults.

## Official Runner Boundary

`sqlbench_lab.livesqlbench_submission` validates public task inputs, records the
pinned official CLI commit, prepares the official task tree, and can invoke Harbor.
This lane is measurement-only. It is not used to manufacture training labels from
protected benchmark content.
