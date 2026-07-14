# Repository Guidelines

read_when: you are starting work in this repo or need repo-specific coding rules

## Start Here
- Read `README.md` and `docs/livesqlbench_competition.md` before changing the pipeline.
- `/home/dahiy/repos/tapasya.mobile/llm` is the reference repo for workflow style, not for task semantics.
- Do not copy the Nietzsche passage task into this repo. This repo is for SQL training and LiveSQLBench work.

## Working Style
- Keep changes small and reviewable.
- Use ASCII by default.
- Python 3.11+ with 4-space indentation.
- Use `uv` as the default Python workflow.
- Prefer one root package and one canonical codepath.

## Testing
- For bug reports, start with a reproducing test.
- Default to strict TDD for durable feature development.
- Use full red-to-green cycles: failing test first, minimal fix, then refactor.
- Run relevant tests when available and say what you did not run.

## Engineering Direction
- No fallback behavior, ever.
- Fail hard on failure conditions.
- No backward compatibility guarantees.
- Forward-only development: do not carry legacy baggage.
- Keep SQL training, SQL evaluation, and LiveSQLBench adapters explicit.
- Do not train on hidden or protected benchmark data.
- Do not report local approximate scores as official LiveSQLBench scores.

## Research Direction
- Keep experiment-specific plans in repo docs and Linear, not in this file.
- Use research to drive concrete pipeline changes only when it maps to an artifact,
  command, dataset contract, eval gate, or failure analysis.
- Preserve the active measurement boundary: one-shot generation, repair, reranking,
  candidate selection, and agentic workflows must be tracked as separate lanes.
- When adopting a paper pattern, record what the paper does, what this repo does
  today, the smallest next implementation, and the eval gate that decides whether
  it worked.

## Hard-Cut Product Policy
- Optimize for one canonical current-state implementation.
- Prefer fail-fast diagnostics and explicit recovery steps.
- Invalid durable state must never be written.
- Write paths must enforce the full canonical invariants before persistence.
- Read-path validation is defense in depth, not the first line of enforcement.
- Do not add migration shims, compatibility bridges, fallback paths, or dual behavior unless the user explicitly asks for them.
- Do not add automatic migration.
- Do not add silent fallbacks.
- If temporary compatibility code is introduced, the same diff must state why it exists and the exact deletion criteria.

## Docs
- Markdown is the only durable documentation surface.
- Keep the competition contract in `docs/livesqlbench_competition.md`.
- Do not add generated HTML, a docs builder, browser docs, or documentation-only web UI.

## Experiment Learning Ledger
- Keep Linear learning issue `TAP-532` current as the practical fine-tuning learning ledger.
- After every meaningful training/eval experiment, add a learning record. Do this even when the experiment fails or regresses.
- Prefer adding a comment to an existing learning child issue when the result reinforces an existing lesson. Create a new child issue under `TAP-532` when the experiment teaches a distinct new lesson.
- Learning records must be practical, not decorative. Include:
  - experiment ID and changed variable
  - what stayed fixed
  - train/eval datasets and whether DB-level holdout applies
  - key metrics and failure counts
  - decision: promote, reject, or investigate
  - the practical lesson a future engineer should remember
- Keep wording consistent with the existing learning child issues:
  - `## Experiment` or `## Experiments`
  - `## What changed`
  - `## Result`
  - `## Practical learning`
  - `## Rule captured`
- If an experiment changes the SQL pipeline direction, update the relevant repo docs and
  the relevant Linear learning issue.
- Do not record local approximate scores as official benchmark claims. Mark lab, local, same-DB, unseen-DB, and official benchmark results distinctly.
- For regressions, state the regression mechanism plainly and do not expand from the regressed checkpoint.

## Git
- Safe by default: `git status`, `git diff`, `git log`.
- No destructive operations unless explicitly requested.
- Do not delete or rename unexpected files without stopping first.

## Decision Rules
- Fix root cause, not symptoms.
- If unsure, read code and docs before asking.
- Protect the SQL data and evaluation contracts before adding training code.
- Prefer official benchmark tooling for benchmark claims.
