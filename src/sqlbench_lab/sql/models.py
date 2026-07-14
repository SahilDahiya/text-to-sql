"""Forward-only SQL pipeline models for the LiveSQLBench lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SQLProvenance:
    source_package: str
    source_revision: str
    source_task_path: str
    created_by: str
    target_source: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLProvenance":
        return cls(
            source_package=str(payload["source_package"]),
            source_revision=str(payload["source_revision"]),
            source_task_path=str(payload["source_task_path"]),
            created_by=str(payload["created_by"]),
            target_source=str(payload["target_source"]),
        )


@dataclass(frozen=True)
class SQLVerification:
    status: str
    verified_by: str
    verification_id: str
    verified_at: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLVerification":
        return cls(
            status=str(payload["status"]),
            verified_by=str(payload["verified_by"]),
            verification_id=str(payload["verification_id"]),
            verified_at=str(payload["verified_at"]),
        )


@dataclass(frozen=True)
class SQLTrainExample:
    schema_version: str
    row_id: str
    source_benchmark: str
    source_split: str
    task_id: str
    db_id: str
    db_path: str
    dialect: str
    question: str
    schema_text: str
    knowledge_text: str | None
    target_sql: str
    task_type: str
    provenance: SQLProvenance
    verification: SQLVerification

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLTrainExample":
        return cls(
            schema_version=str(payload["schema_version"]),
            row_id=str(payload["row_id"]),
            source_benchmark=str(payload["source_benchmark"]),
            source_split=str(payload["source_split"]),
            task_id=str(payload["task_id"]),
            db_id=str(payload["db_id"]),
            db_path=str(payload["db_path"]),
            dialect=str(payload["dialect"]),
            question=str(payload["question"]),
            schema_text=str(payload["schema_text"]),
            knowledge_text=_optional_string(payload.get("knowledge_text")),
            target_sql=str(payload["target_sql"]),
            task_type=str(payload["task_type"]),
            provenance=SQLProvenance.from_dict(payload["provenance"]),
            verification=SQLVerification.from_dict(payload["verification"]),
        )


@dataclass(frozen=True)
class SQLEvalCase:
    schema_version: str
    case_id: str
    source_benchmark: str
    source_split: str
    task_id: str
    db_id: str
    db_path: str
    dialect: str
    question: str
    schema_text: str
    knowledge_text: str | None
    gold_sql: str
    task_type: str
    verification: SQLVerification
    order_sensitive: bool
    numeric_tolerance: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLEvalCase":
        return cls(
            schema_version=str(payload["schema_version"]),
            case_id=str(payload["case_id"]),
            source_benchmark=str(payload["source_benchmark"]),
            source_split=str(payload["source_split"]),
            task_id=str(payload["task_id"]),
            db_id=str(payload["db_id"]),
            db_path=str(payload["db_path"]),
            dialect=str(payload["dialect"]),
            question=str(payload["question"]),
            schema_text=str(payload["schema_text"]),
            knowledge_text=_optional_string(payload.get("knowledge_text")),
            gold_sql=str(payload["gold_sql"]),
            task_type=str(payload["task_type"]),
            verification=SQLVerification.from_dict(payload["verification"]),
            order_sensitive=bool(payload["order_sensitive"]),
            numeric_tolerance=float(payload["numeric_tolerance"]),
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
