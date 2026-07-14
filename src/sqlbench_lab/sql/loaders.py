"""Strict loaders for forward-only SQL pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from jsonschema import Draft202012Validator

from sqlbench_lab.paths import WORKSPACE_ROOT
from sqlbench_lab.shared import read_jsonl_objects

from .models import SQLEvalCase, SQLTrainExample

T = TypeVar("T", SQLEvalCase, SQLTrainExample)


def load_sql_train_examples(path: str | Path) -> list[SQLTrainExample]:
    return _load_rows(
        path=path,
        schema_name="sql_train_example_v2.schema.json",
        row_type=SQLTrainExample,
        id_field="row_id",
        artifact_name="SQL train dataset",
    )


def load_sql_eval_cases(path: str | Path) -> list[SQLEvalCase]:
    return _load_rows(
        path=path,
        schema_name="sql_eval_case_v2.schema.json",
        row_type=SQLEvalCase,
        id_field="case_id",
        artifact_name="SQL eval dataset",
    )


def _load_rows(
    *,
    path: str | Path,
    schema_name: str,
    row_type: type[T],
    id_field: str,
    artifact_name: str,
) -> list[T]:
    resolved_path = _resolve_workspace_path(path)
    rows = read_jsonl_objects(resolved_path)
    if not rows:
        raise ValueError(f"{artifact_name} must contain at least one row")

    validator = _schema_validator(schema_name)
    parsed_rows: list[T] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _validate_row(validator, row, artifact_name=f"{artifact_name} row {index}")
        row_id = str(row[id_field])
        if row_id in seen_ids:
            raise ValueError(f"duplicate {id_field} in {artifact_name}: {row_id}")
        seen_ids.add(row_id)
        parsed_rows.append(row_type.from_dict(row))
    return parsed_rows


def _schema_validator(schema_name: str) -> Draft202012Validator:
    schema_path = WORKSPACE_ROOT / "schemas" / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _validate_row(
    validator: Draft202012Validator,
    row: dict[str, Any],
    *,
    artifact_name: str,
) -> None:
    errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
    if not errors:
        return
    first_error = errors[0]
    error_path = ".".join(str(item) for item in first_error.absolute_path) or "<root>"
    raise ValueError(f"{artifact_name} schema validation failed at {error_path}: {first_error.message}")


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return WORKSPACE_ROOT / candidate
