"""Failure classification and observations for SQL environment steps."""

from __future__ import annotations

import json
from typing import Any


def classify_failure(record: dict[str, Any]) -> str | None:
    prediction_error = _optional_text(record.get("prediction_error"))
    gold_error = _optional_text(record.get("gold_error"))
    predicted_sql = str(record.get("predicted_sql", ""))
    predicted_rows = _rows(record.get("predicted_rows"))
    gold_rows = _rows(record.get("gold_rows"))
    gold_available = bool(record.get("gold_available"))

    if gold_error is not None:
        return "gold_execution_error"
    if not predicted_sql.strip():
        return "empty_prediction"
    if prediction_error is not None:
        lowered = prediction_error.lower()
        if "syntax error" in lowered or "incomplete input" in lowered:
            return "prediction_syntax_error"
        if "no such column" in lowered or "no such table" in lowered:
            return "prediction_schema_error"
        return "prediction_execution_error"
    if not gold_available:
        return None
    if len(predicted_rows) != len(gold_rows):
        return "row_count_mismatch"
    if record.get("passed"):
        return None
    return "row_value_mismatch"


def failure_observation(record: dict[str, Any], *, failure_type: str | None = None) -> str | None:
    resolved_failure_type = failure_type or classify_failure(record)
    if resolved_failure_type is None:
        return None
    prediction_error = _optional_text(record.get("prediction_error"))
    gold_error = _optional_text(record.get("gold_error"))
    if prediction_error is not None:
        return f"Execution error ({resolved_failure_type}): {prediction_error}"
    if gold_error is not None:
        return f"Gold SQL execution error ({resolved_failure_type}): {gold_error}"

    predicted_rows = _rows(record.get("predicted_rows"))
    gold_rows = _rows(record.get("gold_rows"))
    return (
        f"Result mismatch ({resolved_failure_type}): predicted {len(predicted_rows)} row(s), "
        f"gold returned {len(gold_rows)} row(s). "
        f"Predicted preview: {json.dumps(predicted_rows[:3], ensure_ascii=True)}. "
        f"Gold preview: {json.dumps(gold_rows[:3], ensure_ascii=True)}."
    )


def _rows(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
