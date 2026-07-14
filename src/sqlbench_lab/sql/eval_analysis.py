"""Deterministic failure analysis for one-shot SQL eval result files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT


@dataclass(frozen=True)
class SQLEvalFailureAnalysis:
    case_id: str
    task_id: str
    db_id: str
    task_family: str
    curriculum_tier: int
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

    @property
    def failure_count(self) -> int:
        return self.failed_count


def analyze_sql_eval_result(
    result_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> SQLEvalAnalysisSummary:
    resolved_result_path = Path(result_path).resolve()
    payload = json.loads(resolved_result_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"eval result records must be a list: {resolved_result_path}")
    failures = [_analyze_failure(record) for record in records if not bool(record.get("passed"))]
    failure_counts: dict[str, int] = {}
    for failure in failures:
        failure_counts[failure.failure_type] = failure_counts.get(failure.failure_type, 0) + 1
    eval_dataset = str(payload.get("eval_dataset", ""))
    tag_slices = _analyze_tag_slices(records, eval_dataset=eval_dataset, result_path=resolved_result_path)
    resolved_output_path = Path(output_path).resolve() if output_path else _default_analysis_path(resolved_result_path)
    summary = SQLEvalAnalysisSummary(
        result_path=str(resolved_result_path),
        analysis_path=str(resolved_output_path),
        experiment_id=str(payload.get("experiment_id", "")),
        model_variant=str(payload.get("model_variant", "")),
        eval_dataset=eval_dataset,
        case_count=int(payload.get("case_count", len(records))),
        passed_count=sum(1 for record in records if bool(record.get("passed"))),
        failed_count=len(failures),
        pass_rate=float(payload.get("pass_rate", 0.0)),
        failure_counts=dict(sorted(failure_counts.items())),
        tag_slices=tag_slices,
        failures=failures,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return summary


def classify_sql_eval_failure(record: dict[str, Any]) -> str:
    """Classify one failed record using deterministic execution evidence."""

    existing = str(record.get("primary_failure_type", "")).strip()
    if existing and existing != "passed":
        return existing
    predicted_rows = _rows(record.get("predicted_rows"))
    gold_rows = _rows(record.get("gold_rows"))
    predicted_sql = str(record.get("predicted_sql", ""))
    prediction_error = _optional_text(record.get("prediction_error"))
    gold_error = _optional_text(record.get("gold_error"))
    if bool(record.get("passed")):
        return "passed"
    if gold_error is not None:
        return "gold_execution_error"
    if not predicted_sql.strip():
        return "empty_prediction"
    if prediction_error is not None:
        lowered = prediction_error.casefold()
        if "timeout" in lowered or "statement timeout" in lowered:
            return "prediction_timeout"
        if "syntax error" in lowered or "incomplete input" in lowered:
            return "prediction_syntax_error"
        if "no such column" in lowered or "no such table" in lowered or "undefinedcolumn" in lowered or "undefinedtable" in lowered:
            return "prediction_schema_error"
        return "prediction_execution_error"
    if len(predicted_rows) != len(gold_rows):
        return "row_count_mismatch"
    return "row_value_mismatch"


def sql_eval_failure_observation(record: dict[str, Any], *, failure_type: str | None = None) -> str:
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
        f"gold returned {len(gold_rows)} row(s). Predicted preview: "
        f"{json.dumps(predicted_rows[:3], ensure_ascii=True, default=str)}. Gold preview: "
        f"{json.dumps(gold_rows[:3], ensure_ascii=True, default=str)}."
    )


def _analyze_failure(record: dict[str, Any]) -> SQLEvalFailureAnalysis:
    predicted_rows = _rows(record.get("predicted_rows"))
    gold_rows = _rows(record.get("gold_rows"))
    return SQLEvalFailureAnalysis(
        case_id=str(record.get("case_id", "")),
        task_id=str(record.get("task_id", "")),
        db_id=str(record.get("db_id", "")),
        task_family=str(record.get("task_family", "")),
        curriculum_tier=int(record.get("curriculum_tier", 0)),
        failure_type=classify_sql_eval_failure(record),
        predicted_sql=str(record.get("predicted_sql", "")),
        prediction_error=_optional_text(record.get("prediction_error")),
        gold_error=_optional_text(record.get("gold_error")),
        predicted_row_count=len(predicted_rows),
        gold_row_count=len(gold_rows),
        predicted_preview=predicted_rows[:3],
        gold_preview=gold_rows[:3],
    )


def _analyze_tag_slices(records: list[dict[str, Any]], *, eval_dataset: str, result_path: Path) -> list[SQLEvalTagSlice]:
    case_metadata = _load_eval_case_metadata(eval_dataset, result_path=result_path)
    slice_counts: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id", ""))
        metadata = case_metadata.get(case_id, {})
        tags = set(str(tag) for tag in metadata.get("tags", []))
        tags.update({f"db:{record.get('db_id', metadata.get('db_id', ''))}", f"family:{record.get('task_family', metadata.get('task_family', ''))}", f"tier:{record.get('curriculum_tier', metadata.get('curriculum_tier', ''))}"})
        for tag in sorted(tags):
            counts = slice_counts.setdefault(tag, {"case_count": 0, "passed_count": 0, "failure_counts": {}})
            counts["case_count"] += 1
            if bool(record.get("passed")):
                counts["passed_count"] += 1
            else:
                failure_type = classify_sql_eval_failure(record)
                failures = counts["failure_counts"]
                failures[failure_type] = failures.get(failure_type, 0) + 1
    return [
        SQLEvalTagSlice(
            tag=tag,
            case_count=int(counts["case_count"]),
            passed_count=int(counts["passed_count"]),
            failed_count=int(counts["case_count"] - counts["passed_count"]),
            pass_rate=float(counts["passed_count"] / counts["case_count"]),
            failure_counts=dict(sorted(counts["failure_counts"].items())),
        )
        for tag, counts in sorted(slice_counts.items())
    ]


def _load_eval_case_metadata(eval_dataset: str, *, result_path: Path) -> dict[str, dict[str, Any]]:
    if not eval_dataset.strip():
        return {}
    raw = Path(eval_dataset)
    candidates = [raw] if raw.is_absolute() else [WORKSPACE_ROOT / raw, result_path.parent / raw]
    eval_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if eval_path is None:
        raise FileNotFoundError(f"eval dataset does not exist for analysis: {eval_dataset}")
    metadata: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in metadata:
            raise ValueError(f"invalid or duplicate eval case_id at {eval_path}:{line_number}")
        row_metadata = row.get("metadata", {})
        metadata[case_id] = {
            "db_id": row.get("db_id", ""),
            "task_family": row_metadata.get("task_family", ""),
            "curriculum_tier": row_metadata.get("curriculum_tier", 0),
            "tags": row_metadata.get("tags", []),
        }
    return metadata


def _rows(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _default_analysis_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}.analysis.json")
