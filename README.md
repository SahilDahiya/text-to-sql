# LiveSQLBench Competition Workspace

This repository is the training and evaluation control surface for competing on
LiveSQLBench with iterative supervised fine-tuning.

## Current Direction

- Train only on permitted public or open-development SQL examples.
- Keep LiveSQLBench ground truth, test cases, and protected evaluation data outside
  this repository and outside training.
- Use one-shot generation as the first measurement lane.
- Track local execution scores, unseen-database gates, and official LiveSQLBench
  results as separate measurements.
- Use the official LiveSQLBench runner for any competition score claim.

The former storefront, BIRD/Spider lab, cloud deployment, serving, web-console,
experiment-ledger, and generated HTML-docs surfaces are intentionally removed.
Research notes remain in `blog/`; operating instructions live in `docs/`.

## Commands

```bash
uv run python -m sqlbench_lab.cli sql validate-train --dataset <train.jsonl>
uv run python -m sqlbench_lab.cli sql validate-eval --dataset <dev.jsonl>
uv run python -m sqlbench_lab.cli sql audit-leakage \
  --train-dataset <train.jsonl> \
  --eval-dataset <dev.jsonl> \
  --require-db-disjoint
uv run python -m sqlbench_lab.cli sql validate-manifest --manifest <manifest.json>
uv run --group training python -m sqlbench_lab.cli sql run-sft --manifest <manifest.json>
uv run --group training python -m sqlbench_lab.cli sql eval \
  --manifest <manifest.json> --model adapter --dataset <dev.jsonl>
```

The official runner commands are intentionally separate and are not part of the
ISFT loop. See `docs/livesqlbench_competition.md`.
