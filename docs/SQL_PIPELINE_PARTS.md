# SQL Pipeline Parts

read_when: you are discussing or changing the SQL pipeline roadmap

This document lists the major parts of the SQL training and LiveSQLBench pipeline.
We will discuss and implement these one by one.

## 1. Project Contracts

Docs and schemas define what counts as a SQL training row, eval row, experiment manifest,
result row, and LiveSQLBench run.

The goal is to prevent ad hoc files and unclear experiment state.

## 2. Data Ingestion

Load source data into one canonical format.

Likely sources:

- Spider-style public data
- BIRD-style public data
- LiveSQLBench-compatible public/dev data where allowed
- synthetic teacher-generated rows
- execution-repair rows

## 3. Data Validation

Strict validation before training or eval.

Checks should include:

- required fields
- SQL dialect
- duplicate row IDs
- duplicate task IDs
- train/eval leakage
- hidden benchmark contamination
- schema validity

## 4. Prompt Rendering

Convert canonical rows into model messages.

Base shape:

```text
system instruction
question
SQL dialect
schema text
optional knowledge
optional previous SQL/error
assistant target SQL
```

## 5. Baseline Evaluation

Run the base `qwen3.5:0.8b` model before training.

Baseline targets:

- SQL smoke set
- local execution-based dev set
- LiveSQLBench Base-Lite once the official runner is wired

## 6. Training Data Build

Build and version the actual training datasets.

First data lanes:

- direct SQL SFT rows
- execution-repair rows
- tool-use trajectory rows

## 7. Fine-Tuning

Start with LoRA SFT.

Input:

`qwen3.5:0.8b + train rows`

Output:

`adapter/checkpoint + train summary`

## 8. Local SQL Execution Eval

Run generated SQL against local databases.

Primary score:

- execution or test-case pass

Secondary signals:

- exact match
- syntax validity
- error type
- latency

## 9. Error And Repair Loop

Capture failed predictions and turn them into repair data.

Useful fields:

- original question
- schema
- generated SQL
- execution error or wrong-result note
- corrected SQL

## 10. Agent Runtime

LiveSQLBench competition work needs an agent-capable runtime.

Controlled tools:

- schema lookup
- knowledge lookup
- SQL execution
- retry/repair loop
- final SQL submit

## 11. LiveSQLBench Adapter

Explicit adapter for LiveSQLBench tasks and official runner integration.

Responsibilities:

- load benchmark tasks
- prepare database connections
- expose schema and knowledge fields
- invoke official runner
- preserve official output format

## 12. Official Evaluation

Use official LiveSQLBench tooling for any benchmark claim.

Local evaluation is useful, but it is only a development signal.

## 13. Result Reporting

Write inspectable artifacts.

Artifacts should include:

- per-case JSONL
- summary JSON
- failure taxonomy
- comparison reports
- review CSVs where useful

## 14. Experiment Manifests

One manifest per experiment.

It should freeze:

- model
- data
- method
- eval sets
- output paths
- benchmark mode
- git commit when available

## 15. Promotion And Decision Log

Every experiment ends with a decision:

- keep
- defer
- discard

The decision should state why the trained model or adapter is better, worse, or inconclusive
relative to the base model.
