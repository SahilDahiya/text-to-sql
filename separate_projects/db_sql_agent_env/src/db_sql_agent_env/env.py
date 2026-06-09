"""Standalone model-free SQL environment step."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .analysis import classify_failure, failure_observation
from .evaluator import evaluate_case
from .loaders import load_eval_cases
from .models import SQLEvalCase


@dataclass(frozen=True)
class SQLEnvAction:
    type: str
    sql: str


@dataclass(frozen=True)
class SQLEnvValidation:
    syntax_ok: bool
    schema_ok: bool
    error: str | None
    failure_type: str | None
    observation: str | None


@dataclass(frozen=True)
class SQLEnvExecution:
    ran: bool
    row_count: int | None
    preview: list[Any]


@dataclass(frozen=True)
class SQLEnvEvaluation:
    gold_available: bool
    passed: bool
    gold_error: str | None
    gold_row_count: int
    gold_preview: list[Any]


@dataclass(frozen=True)
class SQLEnvStep:
    case_id: str
    task_id: str
    db_id: str
    dialect: str
    attempt: int
    done: bool
    reward: float
    action: SQLEnvAction
    validation: SQLEnvValidation
    execution: SQLEnvExecution
    evaluation: SQLEnvEvaluation


def run_env_step(
    *,
    dataset_path: str | Path,
    case_id: str,
    sql: str,
    attempt: int = 1,
    preview_rows: int = 3,
    output_path: str | Path | None = None,
) -> SQLEnvStep:
    if attempt < 1:
        raise ValueError("attempt must be at least 1")
    if preview_rows < 0:
        raise ValueError("preview_rows must be >= 0")
    if not sql.strip():
        raise ValueError("sql must not be empty")

    cases = load_eval_cases(dataset_path)
    case = _find_case(cases, case_id=case_id)
    result = evaluate_case(case, sql=sql)
    gold_available = case.gold_sql is not None
    record = {
        "case_id": case.case_id,
        "task_id": case.task_id,
        "predicted_sql": sql,
        "passed": result.passed,
        "prediction_error": result.prediction_error,
        "gold_error": result.gold_error,
        "predicted_rows": result.predicted_rows,
        "gold_rows": result.gold_rows,
        "gold_available": gold_available,
    }
    failure_type = classify_failure(record)
    observation = failure_observation(record, failure_type=failure_type)
    done = result.passed if gold_available else result.prediction_error is None
    step = SQLEnvStep(
        case_id=case.case_id,
        task_id=case.task_id,
        db_id=case.db_id,
        dialect=case.dialect,
        attempt=attempt,
        done=done,
        reward=1.0 if result.passed else 0.0,
        action=SQLEnvAction(type="sql", sql=sql),
        validation=SQLEnvValidation(
            syntax_ok=failure_type != "prediction_syntax_error",
            schema_ok=failure_type != "prediction_schema_error",
            error=result.prediction_error,
            failure_type=failure_type,
            observation=observation,
        ),
        execution=SQLEnvExecution(
            ran=result.prediction_error is None,
            row_count=len(result.predicted_rows) if result.prediction_error is None else None,
            preview=_preview(result.predicted_rows, preview_rows),
        ),
        evaluation=SQLEnvEvaluation(
            gold_available=gold_available,
            passed=result.passed,
            gold_error=result.gold_error,
            gold_row_count=len(result.gold_rows),
            gold_preview=_preview(result.gold_rows, preview_rows),
        ),
    )
    if output_path is not None:
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(
            json.dumps(asdict(step), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return step


def _find_case(cases: list[SQLEvalCase], *, case_id: str) -> SQLEvalCase:
    matches = [case for case in cases if case.case_id == case_id]
    if not matches:
        raise ValueError(f"case_id not found in eval dataset: {case_id}")
    if len(matches) > 1:
        raise ValueError(f"duplicate case_id in eval dataset: {case_id}")
    return matches[0]


def _preview(rows: list[tuple[Any, ...]], preview_rows: int) -> list[Any]:
    return [list(row) for row in rows[:preview_rows]]
