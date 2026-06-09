"""Standalone SQL environment models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SQLEvalCase:
    case_id: str
    task_id: str
    db_id: str
    db_path: str
    dialect: str
    question: str
    schema_text: str
    gold_sql: str | None
    order_sensitive: bool
    numeric_tolerance: float
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SQLEvalCase":
        db_path = payload.get("db_path")
        if db_path is None or not str(db_path).strip():
            raise ValueError(f"eval case {payload.get('case_id', '<unknown>')} must include db_path")
        return cls(
            case_id=str(payload["case_id"]),
            task_id=str(payload.get("task_id", payload["case_id"])),
            db_id=str(payload["db_id"]),
            db_path=str(db_path),
            dialect=str(payload.get("dialect", "sqlite")),
            question=str(payload.get("question", "")),
            schema_text=str(payload.get("schema_text", "")),
            gold_sql=_optional_string(payload.get("gold_sql")),
            order_sensitive=bool(payload.get("order_sensitive", False)),
            numeric_tolerance=float(payload.get("numeric_tolerance", 0.000001)),
            tags=tuple(str(item) for item in payload.get("tags", [])),
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None
