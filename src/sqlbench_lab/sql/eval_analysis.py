"""Failure analysis for SQL eval result files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SQLEvalFailureAnalysis:
    case_id: str
    task_id: str
    failure_type: str
    predicted_sql: str
    prediction_error: str | None
    gold_error: str | None
    predicted_row_count: int
    gold_row_count: int
    predicted_preview: list[Any]
    gold_preview: list[Any]


@dataclass(frozen=True)
class SQLEvalTagSlice:
    tag: str
    case_count: int
    passed_count: int
    failed_count: int
    pass_rate: float
    failure_counts: dict[str, int]


@dataclass(frozen=True)
class SQLEvalAnalysisSummary:
    result_path: str
    analysis_path: str
    experiment_id: str
    model_variant: str
    eval_dataset: str
    case_count: int
    passed_count: int
    failed_count: int
    pass_rate: float
    failure_counts: dict[str, int]
    tag_slices: list[SQLEvalTagSlice]
    failures: list[SQLEvalFailureAnalysis]


def analyze_sql_eval_result(
    result_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> SQLEvalAnalysisSummary:
    """Analyze a SQL eval result JSON and write a compact failure summary."""

    resolved_result_path = Path(result_path)
    payload = json.loads(resolved_result_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"eval result records must be a list: {resolved_result_path}")

    failures = [_analyze_failure(record) for record in records if not bool(record.get("passed"))]
    failure_counts: dict[str, int] = {}
    for failure in failures:
        failure_counts[failure.failure_type] = failure_counts.get(failure.failure_type, 0) + 1
    tag_slices = _analyze_tag_slices(
        result_path=resolved_result_path,
        eval_dataset=_optional_text(payload.get("eval_dataset")),
        records=records,
    )

    resolved_output_path = Path(output_path) if output_path is not None else _default_analysis_path(resolved_result_path)
    summary = SQLEvalAnalysisSummary(
        result_path=str(resolved_result_path),
        analysis_path=str(resolved_output_path),
        experiment_id=str(payload.get("experiment_id", "")),
        model_variant=str(payload.get("model_variant", "")),
        eval_dataset=str(payload.get("eval_dataset", "")),
        case_count=int(payload.get("case_count", len(records))),
        passed_count=int(payload.get("passed_count", 0)),
        failed_count=len(failures),
        pass_rate=float(payload.get("pass_rate", 0.0)),
        failure_counts=dict(sorted(failure_counts.items())),
        tag_slices=tag_slices,
        failures=failures,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def classify_sql_eval_failure(record: dict[str, Any]) -> str:
    """Classify a failed eval record into a coarse repair/eval bucket."""

    predicted_rows = _rows(record.get("predicted_rows"))
    gold_rows = _rows(record.get("gold_rows"))
    return _failure_type(
        predicted_sql=str(record.get("predicted_sql", "")),
        prediction_error=_optional_text(record.get("prediction_error")),
        gold_error=_optional_text(record.get("gold_error")),
        predicted_row_count=len(predicted_rows),
        gold_row_count=len(gold_rows),
    )


def sql_eval_failure_observation(record: dict[str, Any], *, failure_type: str | None = None) -> str:
    """Build the repair-loop observation for a failed eval record."""

    resolved_failure_type = failure_type or classify_sql_eval_failure(record)
    prediction_error = _optional_text(record.get("prediction_error"))
    gold_error = _optional_text(record.get("gold_error"))
    if prediction_error is not None:
        return f"Execution error ({resolved_failure_type}): {prediction_error}"
    if gold_error is not None:
        return f"Gold SQL execution error ({resolved_failure_type}): {gold_error}"

    predicted_rows = _rows(record.get("predicted_rows"))
    gold_rows = _rows(record.get("gold_rows"))
    return (
        f"Result mismatch ({resolved_failure_type}): predicted {len(predicted_rows)} row(s), "
        f"gold returned {len(gold_rows)} row(s). "
        f"Predicted preview: {json.dumps(predicted_rows[:3], ensure_ascii=True)}. "
        f"Gold preview: {json.dumps(gold_rows[:3], ensure_ascii=True)}."
    )


def _analyze_failure(record: dict[str, Any]) -> SQLEvalFailureAnalysis:
    predicted_rows = _rows(record.get("predicted_rows"))
    gold_rows = _rows(record.get("gold_rows"))
    prediction_error = _optional_text(record.get("prediction_error"))
    gold_error = _optional_text(record.get("gold_error"))
    predicted_sql = str(record.get("predicted_sql", ""))
    return SQLEvalFailureAnalysis(
        case_id=str(record.get("case_id", "")),
        task_id=str(record.get("task_id", "")),
        failure_type=classify_sql_eval_failure(record),
        predicted_sql=predicted_sql,
        prediction_error=prediction_error,
        gold_error=gold_error,
        predicted_row_count=len(predicted_rows),
        gold_row_count=len(gold_rows),
        predicted_preview=predicted_rows[:3],
        gold_preview=gold_rows[:3],
    )


def _failure_type(
    *,
    predicted_sql: str,
    prediction_error: str | None,
    gold_error: str | None,
    predicted_row_count: int,
    gold_row_count: int,
) -> str:
    if gold_error is not None:
        return "gold_execution_error"
    if not predicted_sql.strip():
        return "empty_prediction"
    if prediction_error is not None:
        lowered = prediction_error.lower()
        if "syntax error" in lowered or "incomplete input" in lowered:
            return "prediction_syntax_error"
        if "no such column" in lowered or "no such table" in lowered:
            return "prediction_schema_error"
        return "prediction_execution_error"
    if predicted_row_count != gold_row_count:
        return "row_count_mismatch"
    return "row_value_mismatch"


def _rows(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _analyze_tag_slices(
    *,
    result_path: Path,
    eval_dataset: str | None,
    records: list[dict[str, Any]],
) -> list[SQLEvalTagSlice]:
    if eval_dataset is None or not eval_dataset.strip():
        return []

    eval_path = _resolve_eval_dataset_path(result_path=result_path, eval_dataset=eval_dataset)
    case_tags = _load_eval_case_tags(eval_path)
    missing_case_ids = sorted(
        {
            str(record.get("case_id", ""))
            for record in records
            if str(record.get("case_id", "")) not in case_tags
        }
    )
    if missing_case_ids:
        raise ValueError(
            f"eval result references case_id values missing from {eval_path}: "
            f"{', '.join(missing_case_ids[:10])}"
        )

    slice_counts: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id", ""))
        passed = bool(record.get("passed"))
        failure_type = None if passed else classify_sql_eval_failure(record)
        for tag in case_tags[case_id]:
            tag_counts = slice_counts.setdefault(
                tag,
                {"case_count": 0, "passed_count": 0, "failure_counts": {}},
            )
            tag_counts["case_count"] += 1
            if passed:
                tag_counts["passed_count"] += 1
            else:
                failures = tag_counts["failure_counts"]
                failures[failure_type] = failures.get(failure_type, 0) + 1

    slices = []
    for tag, counts in sorted(slice_counts.items()):
        case_count = int(counts["case_count"])
        passed_count = int(counts["passed_count"])
        failed_count = case_count - passed_count
        slices.append(
            SQLEvalTagSlice(
                tag=tag,
                case_count=case_count,
                passed_count=passed_count,
                failed_count=failed_count,
                pass_rate=passed_count / case_count if case_count else 0.0,
                failure_counts=dict(sorted(counts["failure_counts"].items())),
            )
        )
    return slices


def _resolve_eval_dataset_path(*, result_path: Path, eval_dataset: str) -> Path:
    raw_path = Path(eval_dataset)
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [raw_path, result_path.parent / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"eval dataset does not exist for analysis: {eval_dataset}")


def _load_eval_case_tags(eval_path: Path) -> dict[str, list[str]]:
    case_tags: dict[str, list[str]] = {}
    with eval_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id", ""))
            if not case_id:
                raise ValueError(f"eval case missing case_id at {eval_path}:{line_number}")
            tags = row.get("tags", [])
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError(f"eval case tags must be a list of strings at {eval_path}:{line_number}")
            if case_id in case_tags:
                raise ValueError(f"duplicate eval case_id in {eval_path}: {case_id}")
            case_tags[case_id] = sorted(set(tags))
    return case_tags


def _default_analysis_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}.analysis.json")
