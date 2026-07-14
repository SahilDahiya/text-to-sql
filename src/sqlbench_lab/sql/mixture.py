"""Deterministic mixture audits for the LiveSQLBench ISFT lane."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .loaders import load_sql_train_examples
from .models import SQLTrainExample


@dataclass(frozen=True)
class SQLMixtureAuditSummary:
    dataset_paths: tuple[str, ...]
    row_count: int
    duplicate_questions: tuple[str, ...]
    duplicate_sql: tuple[str, ...]
    fingerprint: str

    @property
    def passed(self) -> bool:
        return not self.duplicate_questions and not self.duplicate_sql


def audit_sql_mixture(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    output_path: str | Path | None = None,
) -> SQLMixtureAuditSummary:
    if not paths:
        raise ValueError("at least one train dataset is required")
    rows = [row for path in paths for row in load_sql_train_examples(path)]
    _assert_training_invariants(rows)
    summary = SQLMixtureAuditSummary(
        dataset_paths=tuple(str(Path(path)) for path in paths),
        row_count=len(rows),
        duplicate_questions=_duplicates(rows, lambda row: _normalize(row.question)),
        duplicate_sql=_duplicates(rows, lambda row: _normalize_sql(row.target_sql)),
        fingerprint=_fingerprint(rows),
    )
    if output_path is not None:
        resolved = Path(output_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    if not summary.passed:
        problems = []
        if summary.duplicate_questions:
            problems.append(f"duplicate questions: {', '.join(summary.duplicate_questions[:10])}")
        if summary.duplicate_sql:
            problems.append(f"duplicate SQL: {', '.join(summary.duplicate_sql[:10])}")
        raise ValueError("SQL mixture audit failed: " + "; ".join(problems))
    return summary


def _assert_training_invariants(rows: list[SQLTrainExample]) -> None:
    for row in rows:
        if row.verification.status != "execution_verified":
            raise ValueError(f"train row is not execution_verified: {row.row_id}")
        if not row.db_path.strip():
            raise ValueError(f"train row is missing database runtime path: {row.row_id}")


def _duplicates(rows: list[SQLTrainExample], key: Any) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(key(row))
        counts[value] = counts.get(value, 0) + 1
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _normalize_sql(value: str) -> str:
    return _normalize(value).rstrip(";")


def _fingerprint(rows: list[SQLTrainExample]) -> str:
    payload = [
        {
            "row_id": row.row_id,
            "db_id": row.db_id,
            "target_sql": _normalize_sql(row.target_sql),
            "question": _normalize(row.question),
        }
        for row in sorted(rows, key=lambda item: item.row_id)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
