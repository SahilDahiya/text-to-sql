"""SQL pipeline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SQLProvenance:
    created_by: str
    teacher_model: str | None
    source_path: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLProvenance":
        return cls(
            created_by=str(payload["created_by"]),
            teacher_model=payload["teacher_model"] if payload["teacher_model"] is not None else None,
            source_path=str(payload["source_path"]),
        )


@dataclass(frozen=True)
class SQLTrainExample:
    schema_version: str
    row_id: str
    source_benchmark: str
    source_split: str
    task_id: str
    db_id: str
    dialect: str
    question: str
    schema_text: str
    knowledge_text: str | None
    target_sql: str
    task_type: str
    provenance: SQLProvenance
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLTrainExample":
        return cls(
            schema_version=str(payload["schema_version"]),
            row_id=str(payload["row_id"]),
            source_benchmark=str(payload["source_benchmark"]),
            source_split=str(payload["source_split"]),
            task_id=str(payload["task_id"]),
            db_id=str(payload["db_id"]),
            dialect=str(payload["dialect"]),
            question=str(payload["question"]),
            schema_text=str(payload["schema_text"]),
            knowledge_text=_optional_string(payload.get("knowledge_text")),
            target_sql=str(payload["target_sql"]),
            task_type=str(payload["task_type"]),
            provenance=SQLProvenance.from_dict(payload["provenance"]),
            tags=tuple(str(item) for item in payload["tags"]),
        )


@dataclass(frozen=True)
class SQLRepairExample:
    schema_version: str
    row_id: str
    source_benchmark: str
    source_split: str
    task_id: str
    db_id: str
    dialect: str
    question: str
    schema_text: str
    knowledge_text: str | None
    previous_sql: str
    execution_error: str
    target_sql: str
    task_type: str
    provenance: SQLProvenance
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLRepairExample":
        return cls(
            schema_version=str(payload["schema_version"]),
            row_id=str(payload["row_id"]),
            source_benchmark=str(payload["source_benchmark"]),
            source_split=str(payload["source_split"]),
            task_id=str(payload["task_id"]),
            db_id=str(payload["db_id"]),
            dialect=str(payload["dialect"]),
            question=str(payload["question"]),
            schema_text=str(payload["schema_text"]),
            knowledge_text=_optional_string(payload.get("knowledge_text")),
            previous_sql=str(payload["previous_sql"]),
            execution_error=str(payload["execution_error"]),
            target_sql=str(payload["target_sql"]),
            task_type=str(payload["task_type"]),
            provenance=SQLProvenance.from_dict(payload["provenance"]),
            tags=tuple(str(item) for item in payload["tags"]),
        )


@dataclass(frozen=True)
class SQLEvalCase:
    schema_version: str
    case_id: str
    source_benchmark: str
    source_split: str
    task_id: str
    fixture_id: str
    db_id: str
    dialect: str
    question: str
    schema_text: str
    knowledge_text: str | None
    gold_sql: str
    task_type: str
    order_sensitive: bool
    numeric_tolerance: float
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLEvalCase":
        return cls(
            schema_version=str(payload["schema_version"]),
            case_id=str(payload["case_id"]),
            source_benchmark=str(payload["source_benchmark"]),
            source_split=str(payload["source_split"]),
            task_id=str(payload["task_id"]),
            fixture_id=str(payload["fixture_id"]),
            db_id=str(payload["db_id"]),
            dialect=str(payload["dialect"]),
            question=str(payload["question"]),
            schema_text=str(payload["schema_text"]),
            knowledge_text=_optional_string(payload.get("knowledge_text")),
            gold_sql=str(payload["gold_sql"]),
            task_type=str(payload["task_type"]),
            order_sensitive=bool(payload["order_sensitive"]),
            numeric_tolerance=float(payload["numeric_tolerance"]),
            tags=tuple(str(item) for item in payload["tags"]),
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)

