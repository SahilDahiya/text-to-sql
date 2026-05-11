"""Leakage audits for SQL train/eval artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .loaders import load_sql_eval_cases, load_sql_train_examples


@dataclass(frozen=True)
class SQLLeakageAuditSummary:
    train_paths: tuple[str, ...]
    eval_paths: tuple[str, ...]
    train_row_count: int
    eval_case_count: int
    train_db_ids: tuple[str, ...]
    eval_db_ids: tuple[str, ...]
    overlapping_db_ids: tuple[str, ...]
    overlapping_task_ids: tuple[str, ...]
    overlapping_questions: tuple[str, ...]
    overlapping_sql: tuple[str, ...]
    require_db_disjoint: bool

    @property
    def blocking_leak_count(self) -> int:
        count = (
            len(self.overlapping_task_ids)
            + len(self.overlapping_questions)
            + len(self.overlapping_sql)
        )
        if self.require_db_disjoint:
            count += len(self.overlapping_db_ids)
        return count

    @property
    def passed(self) -> bool:
        return self.blocking_leak_count == 0


def audit_sql_dataset_leakage(
    *,
    train_paths: list[str | Path] | tuple[str | Path, ...],
    eval_paths: list[str | Path] | tuple[str | Path, ...],
    require_db_disjoint: bool = False,
) -> SQLLeakageAuditSummary:
    """Audit exact train/eval leakage across SQL artifacts."""

    if not train_paths:
        raise ValueError("at least one train dataset is required")
    if not eval_paths:
        raise ValueError("at least one eval dataset is required")

    train_rows = [
        row
        for path in train_paths
        for row in load_sql_train_examples(path)
    ]
    eval_cases = [
        case
        for path in eval_paths
        for case in load_sql_eval_cases(path)
    ]

    train_task_ids = {row.task_id for row in train_rows}
    eval_task_ids = {case.task_id for case in eval_cases}
    train_questions = {_normalize_text(row.question) for row in train_rows}
    eval_questions = {_normalize_text(case.question) for case in eval_cases}
    train_sql = {_normalize_sql(row.target_sql) for row in train_rows}
    eval_sql = {_normalize_sql(case.gold_sql) for case in eval_cases}
    train_db_ids = {row.db_id for row in train_rows}
    eval_db_ids = {case.db_id for case in eval_cases}

    return SQLLeakageAuditSummary(
        train_paths=tuple(str(path) for path in train_paths),
        eval_paths=tuple(str(path) for path in eval_paths),
        train_row_count=len(train_rows),
        eval_case_count=len(eval_cases),
        train_db_ids=tuple(sorted(train_db_ids)),
        eval_db_ids=tuple(sorted(eval_db_ids)),
        overlapping_db_ids=tuple(sorted(train_db_ids & eval_db_ids)),
        overlapping_task_ids=tuple(sorted(train_task_ids & eval_task_ids)),
        overlapping_questions=tuple(sorted(train_questions & eval_questions)),
        overlapping_sql=tuple(sorted(train_sql & eval_sql)),
        require_db_disjoint=require_db_disjoint,
    )


def assert_no_sql_dataset_leakage(
    *,
    train_paths: list[str | Path] | tuple[str | Path, ...],
    eval_paths: list[str | Path] | tuple[str | Path, ...],
    require_db_disjoint: bool = False,
) -> SQLLeakageAuditSummary:
    summary = audit_sql_dataset_leakage(
        train_paths=train_paths,
        eval_paths=eval_paths,
        require_db_disjoint=require_db_disjoint,
    )
    if summary.passed:
        return summary

    problems: list[str] = []
    if summary.overlapping_task_ids:
        problems.append(f"task_id overlap: {', '.join(summary.overlapping_task_ids[:10])}")
    if summary.overlapping_questions:
        problems.append(f"question overlap: {', '.join(summary.overlapping_questions[:10])}")
    if summary.overlapping_sql:
        problems.append(f"SQL overlap: {', '.join(summary.overlapping_sql[:10])}")
    if require_db_disjoint and summary.overlapping_db_ids:
        problems.append(f"db_id overlap: {', '.join(summary.overlapping_db_ids[:10])}")
    raise ValueError("SQL dataset leakage audit failed: " + "; ".join(problems))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _normalize_sql(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().rstrip(";")).casefold()
