"""SQL evaluation run type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SQLCaseEvalRecord:
    case_id: str
    task_id: str
    model_variant: str
    predicted_sql: str
    passed: bool
    prediction_error: str | None
    gold_error: str | None
    predicted_rows: list[tuple[Any, ...]]
    gold_rows: list[tuple[Any, ...]]


@dataclass(frozen=True)
class SQLEvalRunSummary:
    experiment_id: str
    base_model: str
    model_variant: str
    adapter_dir: str | None
    eval_dataset: str
    result_path: str
    case_count: int
    passed_count: int
    pass_rate: float
    records: list[SQLCaseEvalRecord]


@dataclass(frozen=True)
class SQLRepairAttemptRecord:
    attempt_index: int
    input_sql: str
    input_failure_type: str
    observation: str
    repaired_sql: str
    result: SQLCaseEvalRecord


@dataclass(frozen=True)
class SQLRepairEvalCaseRecord:
    case_id: str
    task_id: str
    model_variant: str
    first_result: SQLCaseEvalRecord
    final_result: SQLCaseEvalRecord
    repair_attempts: list[SQLRepairAttemptRecord]


@dataclass(frozen=True)
class SQLRepairEvalRunSummary:
    experiment_id: str
    base_model: str
    model_variant: str
    adapter_dir: str | None
    eval_dataset: str
    result_path: str
    case_count: int
    first_passed_count: int
    first_pass_rate: float
    final_passed_count: int
    final_pass_rate: float
    repair_attempt_count: int
    repair_success_count: int
    repair_failure_types: tuple[str, ...]
    max_repair_attempts: int
    records: list[SQLRepairEvalCaseRecord]
