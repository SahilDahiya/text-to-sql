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
2. Validate every row against the train schema.
3. Validate local development and unseen-database gates separately.
4. Run exact leakage checks before training.
5. Train one direct-SQL ISFT continuation from the best gated adapter.
6. Run one-shot local evaluation and classify failures.
7. Add only verified, diverse failure curricula to the next mixture.
8. Record the experiment and decision in Linear issue `TAP-532`.
9. Use the official LiveSQLBench runner only after the local promotion gate passes.

## Gate Policy

The old 12-case storefront `dev_v2` and `eval_v1` suites are deleted. They were
useful for debugging alias binding but are not competition gates.

Every competition candidate must report, separately:

- training-set provenance and row counts;
- exact leakage audit results;
- one-shot development score;
- database-disjoint score;
- task-family slices for joins, aggregation, filtering, nesting, and execution errors;
- official LiveSQLBench runner output, when run.

Promotion requires improvement on the target local gate without regression on the
frozen guardrail gates. A local score is never reported as an official benchmark score.

## Current State

The repository deliberately contains no benchmark GT/test-case package and no
competition manifest until the permitted LiveSQLBench development package is present.
The next artifact is a manifest that points to that package's allowed training rows,
local holdout rows, and the Qwen student model. It must pass validation and leakage
checks before any GPU run.

## Official Runner Boundary

`sqlbench_lab.livesqlbench_submission` validates public task inputs, records the
pinned official CLI commit, prepares the official task tree, and can invoke Harbor.
This lane is measurement-only. It is not used to manufacture training labels from
protected benchmark content.
