# Repository Guidelines

read_when: you are starting work in this repo or need repo-specific coding rules

## Start Here

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


## Git
- Safe by default: `git status`, `git diff`, `git log`.
- No destructive operations unless explicitly requested.
- Do not delete or rename unexpected files without stopping first.

## Decision Rules
- Fix root cause, not symptoms.
- If unsure, read code and docs before asking.
- Protect the SQL data and evaluation contracts before adding training code.
- Prefer official benchmark tooling for benchmark claims.
