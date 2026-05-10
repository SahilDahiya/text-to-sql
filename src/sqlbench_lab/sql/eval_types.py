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

