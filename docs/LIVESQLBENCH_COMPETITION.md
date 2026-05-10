# LiveSQLBench Competition

read_when: you are changing LiveSQLBench adapters, eval runners, or benchmark claims

## Position

LiveSQLBench is the competition target, but not the first implementation target.

For `qwen3.5:0.8b`, direct raw SQL generation is a useful measurement lane, not the most
serious competition lane. The serious path is agent mode with retrieval and execution feedback.

As of May 10, 2026, the public LiveSQLBench site separates direct `Model Base` evaluation
from tool-using `Agent` evaluation. It also covers more than plain SELECT queries: SELECT
queries are scored by execution-result comparison, while management SQLs are checked by test
cases. That means this repo's SQLite SELECT eval is a necessary warm-up, not a complete
competition readiness claim.

## First Target

Start with:

- Base-Lite
- Base-Lite-SQLite if it makes local iteration faster

Do not start with Large-v1. Its context scale is the wrong first milestone for a 0.8B student.

## One-Shot Readiness Gate

Before agent work, the one-shot model must show a clean local improvement path:

- one generation per question
- no repair loop during scoring
- fixed Spider eval JSONL, base vs adapter
- fixed BIRD eval JSONL, base vs adapter
- MLflow runs tagged by experiment, dataset, dataset family, model variant, and git commit
- failure buckets logged for diagnosis

This is not a LiveSQLBench score. It is the local gate that tells us whether the student is
learning schema-grounded SQL at all.

## Competition Readiness Gate

A model can be considered ready for LiveSQLBench integration only when the repo has:

- a direct SQL generation interface that can emit one final SQL string reliably
- PostgreSQL dialect support or a documented SQLite-only local adapter boundary
- hierarchical knowledge/context injection support for LiveSQLBench-style HKB fields
- management SQL handling policy, even if the first run only targets SELECT tasks
- official-runner isolation so local Spider/BIRD metrics are never mixed with official scores

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
