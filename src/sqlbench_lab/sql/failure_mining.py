"""Verified failure mining for the next direct-SQL ISFT mixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval_analysis import classify_sql_eval_failure, sql_eval_failure_observation
from .loaders import load_sql_correction_examples, load_sql_eval_cases, load_sql_train_examples


@dataclass(frozen=True)
class SQLCorrectionCollectionSummary:
    result_path: str
    eval_dataset: str
    review_path: str
    output_path: str
    collected_count: int
    skipped_count: int
    failure_counts: dict[str, int]


def collect_verified_sql_corrections(
    *,
    result_path: str | Path,
    eval_dataset: str | Path,
    review_path: str | Path,
    output_path: str | Path,
) -> SQLCorrectionCollectionSummary:
    result_file = Path(result_path).resolve()
    eval_cases = load_sql_eval_cases(eval_dataset)
    cases_by_id = {case.case_id: case for case in eval_cases}
    result_payload = json.loads(result_file.read_text(encoding="utf-8"))
    if not isinstance(result_payload, dict):
        raise ValueError(f"eval result must be a JSON object: {result_file}")
    records = result_payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"eval result records must be a list: {result_file}")
    scorer_version = _required(result_payload, "scorer_version", result_file)
    reviews = _load_reviews(review_path)
    rows: list[dict[str, Any]] = []
    skipped_count = 0
    failure_counts: dict[str, int] = {}
    for record in records:
        if bool(record.get("passed")):
            skipped_count += 1
            continue
        case_id = str(record.get("case_id", ""))
        review = reviews.get(case_id)
        if review is None:
            raise ValueError(f"missing failure review for failed case_id: {case_id}")
        case = cases_by_id.get(case_id)
        if case is None:
            raise ValueError(f"eval result case_id not found in eval dataset: {case_id}")
        _assert_review_is_trainable(review, case_id)
        failure_type = classify_sql_eval_failure(record)
        rows.append(_correction_row(case, record, review, failure_type, scorer_version, eval_dataset))
        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
    if not rows:
        raise ValueError("no independently reviewed corrections collected")

    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_name(f".{output_file.name}.tmp")
    temp_file.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    try:
        load_sql_correction_examples(temp_file)
        temp_file.replace(output_file)
    except Exception:
        temp_file.unlink(missing_ok=True)
        raise
    return SQLCorrectionCollectionSummary(
        result_path=str(result_file),
        eval_dataset=str(Path(eval_dataset).resolve()),
        review_path=str(Path(review_path).resolve()),
        output_path=str(output_file),
        collected_count=len(rows),
        skipped_count=skipped_count,
        failure_counts=dict(sorted(failure_counts.items())),
    )


def build_next_sql_train_mixture(
    *,
    base_train_datasets: list[str | Path] | tuple[str | Path, ...],
    correction_dataset: str | Path,
    output_path: str | Path,
    max_corrections: int,
) -> Path:
    if max_corrections < 0:
        raise ValueError("max_corrections must be non-negative")
    base_rows = [row for path in base_train_datasets for row in load_sql_train_examples(path)]
    corrections = load_sql_correction_examples(correction_dataset)
    if len(corrections) > max_corrections:
        raise ValueError(f"correction budget exceeded: {len(corrections)} > {max_corrections}")
    base_ids = {row.row_id for row in base_rows}
    correction_ids = {row.row_id for row in corrections}
    if base_ids & correction_ids:
        raise ValueError("correction IDs overlap the base mixture")
    output_rows = [*[_train_payload(row) for row in base_rows], *[_correction_train_payload(row) for row in corrections]]
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_name(f".{output_file.name}.tmp")
    temp_file.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    try:
        load_sql_train_examples(temp_file)
        temp_file.replace(output_file)
    except Exception:
        temp_file.unlink(missing_ok=True)
        raise
    return output_file


def _load_reviews(path: str | Path) -> dict[str, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"failure review file does not exist: {resolved}")
    reviews: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        review = json.loads(line)
        if not isinstance(review, dict):
            raise ValueError(f"failure review row must be an object: {resolved}:{line_number}")
        case_id = _required(review, "case_id", resolved)
        if case_id in reviews:
            raise ValueError(f"duplicate failure review case_id: {case_id}")
        reviews[case_id] = review
    return reviews


def _assert_review_is_trainable(review: dict[str, Any], case_id: str) -> None:
    if review.get("scorer_verdict") != "correct" or review.get("prediction_verdict") != "incorrect":
        raise ValueError(f"review is not a model-error/scorer-correct pair: {case_id}")
    if review.get("evidence_source") != "allowed_eval_case":
        raise ValueError(f"review does not cite an allowed eval case: {case_id}")
    for field in ("review_id", "reviewer", "verification_id", "reviewed_at"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"review is missing {field}: {case_id}")


def _correction_row(
    case: Any,
    record: dict[str, Any],
    review: dict[str, Any],
    failure_type: str,
    scorer_version: str,
    eval_dataset: str | Path,
) -> dict[str, Any]:
    return {
        "schema_version": "sql_correction_example:v1",
        "row_id": f"correction::{case.case_id}::{review['review_id']}",
        "source_benchmark": "livesqlbench",
        "source_split": case.source_split,
        "task_id": case.task_id,
        "db_id": case.db_id,
        "db_path": case.db_path,
        "dialect": case.dialect,
        "question": case.question,
        "schema_text": case.schema_text,
        "knowledge_text": case.knowledge_text,
        "column_value_notes": list(case.column_value_notes),
        "schema_linking_notes": list(case.schema_linking_notes),
        "previous_sql": str(record.get("predicted_sql", "")).strip() or "<empty>",
        "failure_type": failure_type,
        "execution_observation": sql_eval_failure_observation(record, failure_type=failure_type),
        "target_sql": case.gold_sql,
        "task_type": case.task_type,
        "metadata": {
            "task_family": case.metadata.task_family,
            "difficulty": case.metadata.difficulty,
            "curriculum_tier": case.metadata.curriculum_tier,
            "sql_shape": case.metadata.sql_shape,
            "grounding_requirement": case.metadata.grounding_requirement,
            "shortcut_status": case.metadata.shortcut_status,
            "tags": sorted(set([*case.metadata.tags, "correction", failure_type])),
        },
        "provenance": {
            "source_package": "livesqlbench-eval-correction",
            "source_revision": scorer_version,
            "source_task_path": f"{Path(eval_dataset).resolve()}::{case.case_id}",
            "created_by": "sqlbench_lab.failure_mining",
            "teacher_model": None,
            "target_source": "allowed_eval_gold",
        },
        "verification": {
            "status": "execution_verified",
            "verified_by": review["reviewer"],
            "verification_id": review["verification_id"],
            "verified_at": review["reviewed_at"],
        },
        "review_id": review["review_id"],
    }


def _train_payload(row: Any) -> dict[str, Any]:
    payload = {
        "schema_version": "sql_train_example:v2",
        "row_id": row.row_id,
        "source_benchmark": row.source_benchmark,
        "source_split": row.source_split,
        "task_id": row.task_id,
        "db_id": row.db_id,
        "db_path": row.db_path,
        "dialect": row.dialect,
        "question": row.question,
        "schema_text": row.schema_text,
        "knowledge_text": row.knowledge_text,
        "column_value_notes": list(row.column_value_notes),
        "schema_linking_notes": list(row.schema_linking_notes),
        "target_sql": row.target_sql,
        "task_type": row.task_type,
        "metadata": _metadata_payload(row.metadata),
        "provenance": _provenance_payload(row.provenance),
        "verification": _verification_payload(row.verification),
    }
    return payload


def _correction_train_payload(row: Any) -> dict[str, Any]:
    payload = _train_payload(row)
    payload["row_id"] = row.row_id.replace("correction::", "train::correction::", 1)
    payload["provenance"]["created_by"] = "sqlbench_lab.failure_mining"
    return payload


def _metadata_payload(metadata: Any) -> dict[str, Any]:
    return {
        "task_family": metadata.task_family,
        "difficulty": metadata.difficulty,
        "curriculum_tier": metadata.curriculum_tier,
        "sql_shape": metadata.sql_shape,
        "grounding_requirement": metadata.grounding_requirement,
        "shortcut_status": metadata.shortcut_status,
        "tags": list(metadata.tags),
    }


def _provenance_payload(provenance: Any) -> dict[str, Any]:
    return {
        "source_package": provenance.source_package,
        "source_revision": provenance.source_revision,
        "source_task_path": provenance.source_task_path,
        "created_by": provenance.created_by,
        "teacher_model": provenance.teacher_model,
        "target_source": provenance.target_source,
    }


def _verification_payload(verification: Any) -> dict[str, Any]:
    return {
        "status": verification.status,
        "verified_by": verification.verified_by,
        "verification_id": verification.verification_id,
        "verified_at": verification.verified_at,
    }


def _required(payload: dict[str, Any], field: str, path: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string at {path}")
    return value.strip()
