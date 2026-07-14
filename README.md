# LiveSQLBench Competition Workspace

This repository is the training and evaluation control surface for competing on
LiveSQLBench with iterative supervised fine-tuning.

## Current Direction

- Train only on permitted public or open-development SQL examples.
- Keep LiveSQLBench ground truth, test cases, and protected evaluation data outside
  this repository and outside training.
- Use one-shot generation as the first measurement lane.
- Track local execution scores and official LiveSQLBench results as separate
  measurements.
- Use the official LiveSQLBench runner for any competition score claim.

The former storefront, BIRD/Spider lab, cloud deployment, serving, web-console,
experiment-ledger, and generated HTML-docs surfaces are intentionally removed.
Research notes remain in `blog/`; operating instructions live in `docs/`.

## Commands

Importing a LiveSQLBench package requires a separate JSONL target manifest. The
package's public task payloads are prompts and database environments; empty
`sol_sql` or `test_cases` fields are never treated as training labels.

```bash
uv run python -m sqlbench_lab.cli sql verify-targets \
  --package-root /path/to/base-lite-269 \
  --target-manifest /path/to/pending-targets.jsonl \
  --source-revision base-lite-269 \
  --verified-output /path/to/verified-targets.jsonl \
  --verified-by local-postgres-run --verified-at 2026-07-14T00:00:00Z
uv run python -m sqlbench_lab.cli sql livesqlbench-import \
  --package-root /path/to/base-lite-269 \
  --target-manifest /path/to/verified-targets.jsonl \
  --source-revision base-lite-269 \
  --train-output artifacts/private/livesqlbench/train.v2.jsonl \
  --eval-output artifacts/private/livesqlbench/dev.v2.jsonl
```

```bash
uv run python -m sqlbench_lab.cli sql validate-train --dataset <train.jsonl>
uv run python -m sqlbench_lab.cli sql validate-eval --dataset <dev.jsonl>
uv run python -m sqlbench_lab.cli sql audit-mixture --dataset <train.jsonl>
uv run python -m sqlbench_lab.cli sql build-review-packet \
  --iteration iter-001 --phase artifacts --manifest <manifest.json> \
  --output reviews/iter-001-artifacts.md --conversation conversation.md
uv run python -m sqlbench_lab.cli sql record-review \
  --packet reviews/iter-001-artifacts.json --reviewer human \
  --decision approve --output reviews/iter-001-artifacts-review.json
uv run python -m sqlbench_lab.cli sql validate-manifest --manifest <manifest.json>
uv run --group training python -m sqlbench_lab.cli sql run-sft \
  --manifest <manifest.json> --review reviews/iter-001-artifacts-review.json
uv run --group training python -m sqlbench_lab.cli sql eval \
  --manifest <manifest.json> --model adapter --dataset <dev.jsonl>
uv run python -m sqlbench_lab.cli sql build-review-packet \
  --iteration iter-001 --phase evaluation --manifest <manifest.json> \
  --result <eval-result.json> --output reviews/iter-001-evaluation.md
```

The official runner commands are intentionally separate and are not part of the
ISFT loop. See `docs/livesqlbench_competition.md`.

The review packet is the human-in-the-loop boundary. It contains the manifest,
dataset evidence, per-case SQL and execution results, and the optional coding-agent
conversation. A reviewer can approve, reject, or request extra review before
training proceeds.
