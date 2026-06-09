"""Import DB-specific seed train/eval datasets into the standalone project."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .jsonl import read_jsonl_objects


@dataclass(frozen=True)
class SeedDatasetSummary:
    output_dir: str
    train_source_path: str
    eval_source_path: str
    train_path: str
    eval_path: str
    train_row_count: int
    eval_row_count: int
    train_db_ids: list[str]
    eval_db_ids: list[str]
    tag_counts: dict[str, int]
    overlapping_questions: list[str]
    overlapping_sql: list[str]


def import_seed_dataset(
    *,
    train_path: str | Path,
    eval_path: str | Path,
    output_dir: str | Path,
    allow_overlap: bool = False,
) -> SeedDatasetSummary:
    """Copy seed train/eval JSONL files and write a contamination summary."""

    resolved_train_path = Path(train_path)
    resolved_eval_path = Path(eval_path)
    resolved_output_dir = Path(output_dir)
    if not resolved_train_path.exists():
        raise FileNotFoundError(f"train dataset does not exist: {resolved_train_path}")
    if not resolved_eval_path.exists():
        raise FileNotFoundError(f"eval dataset does not exist: {resolved_eval_path}")

    train_rows = read_jsonl_objects(resolved_train_path)
    eval_rows = read_jsonl_objects(resolved_eval_path)
    if not train_rows:
        raise ValueError(f"train dataset must contain at least one row: {resolved_train_path}")
    if not eval_rows:
        raise ValueError(f"eval dataset must contain at least one row: {resolved_eval_path}")

    overlapping_questions = _overlaps(
        (_normalized_text(row.get("question")) for row in train_rows),
        (_normalized_text(row.get("question")) for row in eval_rows),
    )
    overlapping_sql = _overlaps(
        (_normalized_sql(row.get("target_sql")) for row in train_rows),
        (_normalized_sql(row.get("gold_sql")) for row in eval_rows),
    )
    if not allow_overlap and (overlapping_questions or overlapping_sql):
        details = []
        if overlapping_questions:
            details.append(f"question overlap={len(overlapping_questions)}")
        if overlapping_sql:
            details.append(f"sql overlap={len(overlapping_sql)}")
        raise ValueError(f"seed train/eval overlap detected: {', '.join(details)}")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    copied_train_path = resolved_output_dir / "train.jsonl"
    copied_eval_path = resolved_output_dir / "eval.jsonl"
    shutil.copyfile(resolved_train_path, copied_train_path)
    shutil.copyfile(resolved_eval_path, copied_eval_path)

    summary = SeedDatasetSummary(
        output_dir=str(resolved_output_dir),
        train_source_path=str(resolved_train_path),
        eval_source_path=str(resolved_eval_path),
        train_path=str(copied_train_path),
        eval_path=str(copied_eval_path),
        train_row_count=len(train_rows),
        eval_row_count=len(eval_rows),
        train_db_ids=_sorted_values(row.get("db_id") for row in train_rows),
        eval_db_ids=_sorted_values(row.get("db_id") for row in eval_rows),
        tag_counts=_tag_counts([*train_rows, *eval_rows]),
        overlapping_questions=overlapping_questions,
        overlapping_sql=overlapping_sql,
    )
    summary_path = resolved_output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def _overlaps(left_values: Iterable[str | None], right_values: Iterable[str | None]) -> list[str]:
    left = {value for value in left_values if isinstance(value, str) and value}
    right = {value for value in right_values if isinstance(value, str) and value}
    return sorted(left & right)


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().lower().split())


def _normalized_sql(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().rstrip(";").lower().split())


def _sorted_values(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values if value is not None})


def _tag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        tags = row.get("tags", [])
        if not isinstance(tags, list):
            continue
        for tag in tags:
            tag_text = str(tag)
            counts[tag_text] = counts.get(tag_text, 0) + 1
    return dict(sorted(counts.items()))
