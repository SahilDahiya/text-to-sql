# Play Poker SQL Workspace

This repo is being set up as a small, explicit workspace for training and evaluating a
`qwen3.5:0.8b` SQL student, with LiveSQLBench competition work as the long-term target.

The reference repo for structure and discipline is:

- `/home/dahiy/repos/tapasya.mobile/llm`

We will borrow its style:

- `uv` first
- docs before broad implementation
- manifest-driven experiments
- protected eval data separated from train data
- official benchmark tooling for official benchmark claims
- no silent fallback behavior

## First Scope

The first scope is intentionally small:

1. make this a valid `uv` Python project
2. document the SQL training lane
3. document the LiveSQLBench competition lane
4. add schemas and strict loaders next
5. add training/eval code after the contracts are stable

## Current Target

Student:

- `qwen3.5:0.8b`

Training direction:

- LoRA supervised fine-tuning first
- execution-repair data second
- tool-use trajectory training third

Competition direction:

- start with LiveSQLBench Base-Lite or Base-Lite-SQLite
- use official LiveSQLBench tooling for competition claims
- treat agent mode as the serious path for a 0.8B student

