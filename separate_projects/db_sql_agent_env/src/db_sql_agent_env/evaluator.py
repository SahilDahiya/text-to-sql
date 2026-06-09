"""Read-only SQLite execution and result comparison."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .models import SQLEvalCase


@dataclass(frozen=True)
class SQLEvaluationResult:
    passed: bool
    prediction_error: str | None
    gold_error: str | None
    predicted_rows: list[tuple[Any, ...]]
    gold_rows: list[tuple[Any, ...]]


def evaluate_case(case: SQLEvalCase, *, sql: str) -> SQLEvaluationResult:
    if case.dialect != "sqlite":
        raise ValueError("standalone SQL agent env currently supports sqlite only")

    db_path = _resolve_db_path(case.db_path)
    predicted_rows, prediction_error = _execute_sql_readonly(db_path, sql)
    if case.gold_sql is None:
        return SQLEvaluationResult(
            passed=False,
            prediction_error=prediction_error,
            gold_error=None,
            predicted_rows=predicted_rows,
            gold_rows=[],
        )

    gold_rows, gold_error = _execute_sql_readonly(db_path, case.gold_sql)
    passed = (
        prediction_error is None
        and gold_error is None
        and rows_equivalent(
            predicted_rows,
            gold_rows,
            order_sensitive=case.order_sensitive,
            numeric_tolerance=case.numeric_tolerance,
        )
    )
    return SQLEvaluationResult(
        passed=passed,
        prediction_error=prediction_error,
        gold_error=gold_error,
        predicted_rows=predicted_rows,
        gold_rows=gold_rows,
    )


def rows_equivalent(
    predicted_rows: list[tuple[Any, ...]],
    gold_rows: list[tuple[Any, ...]],
    *,
    order_sensitive: bool,
    numeric_tolerance: float,
) -> bool:
    if len(predicted_rows) != len(gold_rows):
        return False
    if order_sensitive:
        return all(
            _row_equal(predicted, gold, numeric_tolerance=numeric_tolerance)
            for predicted, gold in zip(predicted_rows, gold_rows)
        )

    unmatched_gold = list(gold_rows)
    for predicted in predicted_rows:
        match_index = _find_matching_row(predicted, unmatched_gold, numeric_tolerance=numeric_tolerance)
        if match_index is None:
            return False
        unmatched_gold.pop(match_index)
    return not unmatched_gold


def _resolve_db_path(db_path: str) -> Path:
    path = Path(db_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"db_path does not exist: {path}")
    return path


def _execute_sql_readonly(db_path: Path, sql: str) -> tuple[list[tuple[Any, ...]], str | None]:
    try:
        uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.execute(sql)
            return [tuple(row) for row in cursor.fetchall()], None
    except sqlite3.Error as exc:
        return [], str(exc)


def _find_matching_row(
    predicted: tuple[Any, ...],
    candidates: list[tuple[Any, ...]],
    *,
    numeric_tolerance: float,
) -> int | None:
    for index, candidate in enumerate(candidates):
        if _row_equal(predicted, candidate, numeric_tolerance=numeric_tolerance):
            return index
    return None


def _row_equal(left: tuple[Any, ...], right: tuple[Any, ...], *, numeric_tolerance: float) -> bool:
    if len(left) != len(right):
        return False
    return all(
        _value_equal(left_value, right_value, numeric_tolerance=numeric_tolerance)
        for left_value, right_value in zip(left, right)
    )


def _value_equal(left: Any, right: Any, *, numeric_tolerance: float) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=numeric_tolerance, abs_tol=numeric_tolerance)
    return left == right
