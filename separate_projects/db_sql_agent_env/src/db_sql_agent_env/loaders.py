"""Standalone dataset loaders."""

from __future__ import annotations

from pathlib import Path

from .jsonl import read_jsonl_objects
from .models import SQLEvalCase


def load_eval_cases(path: str | Path) -> list[SQLEvalCase]:
    resolved_path = Path(path)
    rows = read_jsonl_objects(resolved_path)
    if not rows:
        raise ValueError(f"SQL eval dataset must contain at least one row: {path}")

    cases: list[SQLEvalCase] = []
    seen_case_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        row = dict(row)
        row["db_path"] = _resolve_db_path(row.get("db_path"), dataset_path=resolved_path)
        case = SQLEvalCase.from_dict(row)
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id in SQL eval dataset row {index}: {case.case_id}")
        seen_case_ids.add(case.case_id)
        cases.append(case)
    return cases


def _resolve_db_path(value: object, *, dataset_path: Path) -> str:
    if value is None or not str(value).strip():
        raise ValueError("SQL eval row must include db_path")
    raw_path = Path(str(value))
    if raw_path.is_absolute():
        if not raw_path.exists():
            raise FileNotFoundError(f"db_path does not exist: {raw_path}")
        return str(raw_path)

    candidates = [Path.cwd() / raw_path]
    dataset_parent = dataset_path.resolve().parent
    candidates.extend(parent / raw_path for parent in [dataset_parent, *dataset_parent.parents])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"db_path does not exist: {raw_path}")
