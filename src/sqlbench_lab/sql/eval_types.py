"""One-shot SQL evaluation result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SQLCaseEvalRecord:
    case_id: str
    task_id: str
    db_id: str
    task_family: str
    curriculum_tier: int
    model_variant: str
    predicted_sql: str
    passed: bool
    primary_failure_type: str
    prediction_error: str | None
    gold_error: str | None
    predicted_rows: list[tuple[Any, ...]]
    gold_rows: list[tuple[Any, ...]]
    predicted_columns: tuple[str, ...]
    gold_columns: tuple[str, ...]
    execution_ms: float


@dataclass(frozen=True)
class SQLEvalRunSummary:
    schema_version: str
    experiment_id: str
    base_model: str
    model_variant: str
    adapter_dir: str | None
    eval_dataset: str
    dataset_fingerprint: str
    eval_db_ids: tuple[str, ...]
    db_disjoint_verified: bool
    scorer_version: str
    generation_config: dict[str, Any]
    result_path: str
    case_count: int
    passed_count: int
    pass_rate: float
    records: list[SQLCaseEvalRecord]
