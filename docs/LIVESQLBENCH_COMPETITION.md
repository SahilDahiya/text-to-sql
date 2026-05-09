# LiveSQLBench Competition

read_when: you are changing LiveSQLBench adapters, eval runners, or benchmark claims

## Position

LiveSQLBench is the competition target, but not the first implementation target.

For `qwen3.5:0.8b`, direct raw SQL generation is a useful measurement lane, not the most
serious competition lane. The serious path is agent mode with retrieval and execution feedback.

## First Target

Start with:

- Base-Lite
- Base-Lite-SQLite if it makes local iteration faster

Do not start with Large-v1. Its context scale is the wrong first milestone for a 0.8B student.

## Modes

### Model Base Mode

One prompt, one SQL answer.

Use this for:

- baseline measurement
- SFT progress tracking
- regression checks

### Agent Mode

Controlled tools:

- schema lookup
- optional knowledge lookup
- SQL execution
- final SQL submission

Use this for:

- competition-oriented evaluation
- execution-repair loops
- testing whether the small student can be useful as a policy inside a scaffold

## Official Scoring Policy

Any leaderboard or competition claim must use official LiveSQLBench tooling.

Local smoke tests are allowed, but they are development signals only.

## Open Design Questions

- Should the first local eval dialect be SQLite for speed or PostgreSQL for benchmark fidelity?
- Should the first training data come from Spider/BIRD-style public data, synthetic teacher rows, or both?
- Should `qwen3.5:0.8b` be trained first as a direct SQL generator or immediately as a tool-use policy?
