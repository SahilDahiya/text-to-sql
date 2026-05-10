"""Collect SQL repair examples from eval failures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT

from .eval_analysis import classify_sql_eval_failure
from .loaders import load_sql_eval_cases, load_sql_repair_examples
from .models import SQLEvalCase

STRONG_REPAIR_FAILURE_TYPES = frozenset(
    {
        "empty_prediction",
        "prediction_execution_error",
        "prediction_schema_error",
        "prediction_syntax_error",
    }
)


@dataclass(frozen=True)
class SQLRepairCollectionSummary:
    result_path: str
    eval_dataset: str
    output_path: str
    collected_count: int
    skipped_count: int
    failure_counts: dict[str, int]


def collect_sql_repair_data(
    *,
    result_path: str | Path,
    eval_dataset: str | Path,
    output_path: str | Path,
    failure_types: set[str] | None = None,
    strong_only: bool = False,
) -> SQLRepairCollectionSummary:
    """Convert failed eval records into SQL repair JSONL rows."""

    resolved_result_path = _resolve_workspace_path(result_path)
    resolved_output_path = _resolve_workspace_path(output_path)
    eval_cases = load_sql_eval_cases(eval_dataset)
    cases_by_id = {case.case_id: case for case in eval_cases}
    result_payload = json.loads(resolved_result_path.read_text(encoding="utf-8"))
    records = result_payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"eval result records must be a list: {resolved_result_path}")

    allowed_failure_types = _allowed_failure_types(failure_types=failure_types, strong_only=strong_only)
    rows: list[dict[str, Any]] = []
    skipped_count = 0
    failure_counts: dict[str, int] = {}
    for record in records:
        if bool(record.get("passed")):
            skipped_count += 1
            continue
        failure_type = classify_sql_eval_failure(record)
        if allowed_failure_types is not None and failure_type not in allowed_failure_types:
            skipped_count += 1
            continue
        case_id = str(record.get("case_id", ""))
        if case_id not in cases_by_id:
            raise ValueError(f"eval result case_id not found in eval dataset: {case_id}")
        case = cases_by_id[case_id]
        rows.append(
            _repair_row(
                case=case,
                record=record,
                failure_type=failure_type,
                result_path=result_path,
                eval_dataset=eval_dataset,
                model_variant=str(result_payload.get("model_variant", "unknown")),
            )
        )
        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

    if not rows:
        raise ValueError("no repair rows collected from eval result")

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    load_sql_repair_examples(resolved_output_path)
    return SQLRepairCollectionSummary(
        result_path=str(result_path),
        eval_dataset=str(eval_dataset),
        output_path=str(output_path),
        collected_count=len(rows),
        skipped_count=skipped_count,
        failure_counts=dict(sorted(failure_counts.items())),
    )


def _repair_row(
    *,
    case: SQLEvalCase,
    record: dict[str, Any],
    failure_type: str,
    result_path: str | Path,
    eval_dataset: str | Path,
    model_variant: str,
) -> dict[str, Any]:
    previous_sql = str(record.get("predicted_sql", "")).strip() or "<empty>"
    return {
        "schema_version": "sql_repair_example:v1",
        "row_id": _repair_row_id(case=case, model_variant=model_variant, failure_type=failure_type),
        "source_benchmark": case.source_benchmark,
        "source_split": case.source_split,
        "task_id": case.task_id,
        "db_id": case.db_id,
        "dialect": case.dialect,
        "question": case.question,
        "schema_text": case.schema_text,
        "knowledge_text": case.knowledge_text,
        "previous_sql": previous_sql,
        "execution_error": _repair_observation(record=record, failure_type=failure_type),
        "target_sql": case.gold_sql,
        "task_type": case.task_type,
        "provenance": {
            "created_by": "sqlbench_lab.collect_repair_data",
            "teacher_model": None,
            "source_path": f"{result_path}::{eval_dataset}",
        },
        "tags": [
            case.source_benchmark,
            "repair",
            failure_type,
            f"model_variant_{model_variant}",
            "strong_repair_candidate" if failure_type in STRONG_REPAIR_FAILURE_TYPES else "weak_repair_candidate",
        ],
    }


def _repair_observation(*, record: dict[str, Any], failure_type: str) -> str:
    prediction_error = _optional_string(record.get("prediction_error"))
    gold_error = _optional_string(record.get("gold_error"))
    if prediction_error is not None:
        return f"Execution error ({failure_type}): {prediction_error}"
    if gold_error is not None:
        return f"Gold SQL execution error ({failure_type}): {gold_error}"

    predicted_rows = _rows(record.get("predicted_rows"))
    gold_rows = _rows(record.get("gold_rows"))
    return (
        f"Result mismatch ({failure_type}): predicted {len(predicted_rows)} row(s), "
        f"gold returned {len(gold_rows)} row(s). "
        f"Predicted preview: {json.dumps(predicted_rows[:3], ensure_ascii=True)}. "
        f"Gold preview: {json.dumps(gold_rows[:3], ensure_ascii=True)}."
    )


def _allowed_failure_types(*, failure_types: set[str] | None, strong_only: bool) -> set[str] | None:
    if failure_types is not None:
        return failure_types
    if strong_only:
        return set(STRONG_REPAIR_FAILURE_TYPES)
    return None


def _repair_row_id(*, case: SQLEvalCase, model_variant: str, failure_type: str) -> str:
    return _safe_id(f"{case.source_benchmark}_{case.case_id}_{model_variant}_{failure_type}_repair")


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)


def _rows(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return WORKSPACE_ROOT / candidate
