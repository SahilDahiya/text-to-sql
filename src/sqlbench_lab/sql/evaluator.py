"""Explicit SQLite/PostgreSQL execution and result-equivalence evaluation."""

from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import SQLEvalCase


@dataclass(frozen=True)
class SQLEvaluationResult:
    passed: bool
    prediction_error: str | None
    gold_error: str | None
    predicted_rows: list[tuple[Any, ...]]
    gold_rows: list[tuple[Any, ...]]
    predicted_columns: tuple[str, ...]
    gold_columns: tuple[str, ...]
    execution_ms: float


def evaluate_sql_case(
    case: SQLEvalCase,
    *,
    predicted_sql: str,
    postgres_connect: Callable[..., Any] | None = None,
) -> SQLEvaluationResult:
    """Evaluate SQL against the case's declared backend with isolated executions."""

    if case.dialect == "sqlite":
        executor = _sqlite_executor(case.db_path)
    elif case.dialect == "postgresql":
        executor = _postgres_executor(case.db_path, postgres_connect=postgres_connect)
    else:
        raise ValueError(f"unsupported SQL dialect: {case.dialect}")

    started = time.perf_counter()
    predicted_rows, predicted_columns, prediction_error = executor(predicted_sql)
    gold_rows, gold_columns, gold_error = executor(case.gold_sql)
    elapsed_ms = (time.perf_counter() - started) * 1000
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
        predicted_columns=predicted_columns,
        gold_columns=gold_columns,
        execution_ms=elapsed_ms,
    )


def evaluate_sqlite_case(
    case: SQLEvalCase,
    *,
    predicted_sql: str,
) -> SQLEvaluationResult:
    """Evaluate an explicitly SQLite case."""

    if case.dialect != "sqlite":
        raise ValueError("evaluate_sqlite_case requires a sqlite case")
    return evaluate_sql_case(case, predicted_sql=predicted_sql)


def evaluate_postgresql_case(
    case: SQLEvalCase,
    *,
    predicted_sql: str,
    postgres_connect: Callable[..., Any] | None = None,
) -> SQLEvaluationResult:
    """Evaluate an explicitly PostgreSQL case using its external env file."""

    if case.dialect != "postgresql":
        raise ValueError("evaluate_postgresql_case requires a postgresql case")
    return evaluate_sql_case(case, predicted_sql=predicted_sql, postgres_connect=postgres_connect)


def rows_equivalent(
    predicted_rows: list[tuple[Any, ...]],
    gold_rows: list[tuple[Any, ...]],
    *,
    order_sensitive: bool,
    numeric_tolerance: float,
) -> bool:
    """Compare SQL result rows with optional order sensitivity and numeric tolerance."""

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


def _sqlite_executor(db_path: str) -> Callable[[str], tuple[list[tuple[Any, ...]], tuple[str, ...], str | None]]:
    resolved = Path(db_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {resolved}")

    def execute(sql: str) -> tuple[list[tuple[Any, ...]], tuple[str, ...], str | None]:
        if not sql.strip():
            return [], (), "empty SQL prediction"
        try:
            with sqlite3.connect(resolved) as conn:
                cursor = conn.execute(sql)
                columns = tuple(description[0] for description in cursor.description or ())
                rows = [tuple(row) for row in cursor.fetchall()] if cursor.description else []
                conn.rollback()
                return rows, columns, None
        except sqlite3.Error as exc:
            return [], (), str(exc)

    return execute


def _postgres_executor(
    env_path: str,
    *,
    postgres_connect: Callable[..., Any] | None,
) -> Callable[[str], tuple[list[tuple[Any, ...]], tuple[str, ...], str | None]]:
    resolved = Path(env_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PostgreSQL environment file does not exist: {resolved}")
    connection_values = _read_env_file(resolved)
    required = {"PGHOST", "PGPORT", "PGUSER", "PGDATABASE"}
    missing = sorted(required - connection_values.keys())
    if missing:
        raise ValueError(f"PostgreSQL environment file is missing: {', '.join(missing)}")
    if postgres_connect is None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("PostgreSQL evaluation requires the psycopg dependency") from exc
        connector = psycopg.connect
        database_error = psycopg.Error
    else:
        connector = postgres_connect
        database_error = Exception

    def execute(sql: str) -> tuple[list[tuple[Any, ...]], tuple[str, ...], str | None]:
        if not sql.strip():
            return [], (), "empty SQL prediction"
        connection = connector(
            host=connection_values["PGHOST"],
            port=int(connection_values["PGPORT"]),
            user=connection_values["PGUSER"],
            dbname=connection_values["PGDATABASE"],
            password=connection_values.get("PGPASSWORD"),
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                columns = tuple(description[0] for description in cursor.description or ())
                rows = [tuple(row) for row in cursor.fetchall()] if cursor.description else []
            connection.rollback()
            return rows, columns, None
        except database_error as exc:
            connection.rollback()
            return [], (), str(exc)
        finally:
            connection.close()

    return execute


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    pattern = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.match(line)
        if match is None:
            raise ValueError(f"unsupported PostgreSQL env syntax at {path}:{line_number}")
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[match.group(1)] = value
    return values


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
