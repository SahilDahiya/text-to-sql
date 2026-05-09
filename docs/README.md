# Docs

read_when: you are starting work in this repo or need the current repo map

Start here when working in this repo.

## Reference Repo

Use `/home/dahiy/repos/tapasya.mobile/llm` as the reference for:

- strict docs-first planning
- manifest-style experiment tracking
- dataset split policy
- CLI layout
- tests around contracts

Do not copy its Nietzsche passage task into this repo. This repo's new lane is SQL training
and LiveSQLBench competition work.

## Current Docs

- `SQL_PIPELINE_PARTS.md` lists the pipeline components we will discuss and implement one by one.
- `SQL_TRAINING_PIPELINE.md` defines the `qwen3.5:0.8b` SQL training path.
- `LIVESQLBENCH_COMPETITION.md` defines the competition posture and constraints.

## Current Code Contracts

- `schemas/sql_train_example_v1.schema.json`
- `schemas/sql_repair_example_v1.schema.json`
- `schemas/sql_eval_case_v1.schema.json`
- `schemas/sql_sft_experiment_v1.schema.json`
