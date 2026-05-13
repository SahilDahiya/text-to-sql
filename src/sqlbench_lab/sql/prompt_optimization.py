"""Prompt optimization candidate tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT

from .eval_analysis import classify_sql_eval_failure
from .loaders import load_sql_eval_cases

PROMPT_OPTIMIZERS = {"mipro_v2", "gepa", "manual"}
PROMPT_CANDIDATE_DECISIONS = {"pending", "selected", "rejected"}


@dataclass(frozen=True)
class SQLPromptCandidateSummary:
    """One tracked prompt-optimization candidate."""

    schema_version: str
    experiment_id: str
    optimizer: str
    candidate_id: str
    decision: str
    prompt_file: str
    prompt_sha256: str
    prompt_char_count: int
    prompt_dev_dataset: str
    prompt_dev_case_count: int
    fresh_gate_dataset: str | None
    fresh_gate_case_count: int | None
    source_manifest: str | None
    model_variant: str | None
    eval_result: str | None
    eval_dataset_role: str | None
    eval_case_count: int | None
    eval_passed_count: int | None
    eval_pass_rate: float | None
    failure_counts: dict[str, int]
    notes: str | None
    output_path: str


def record_sql_prompt_candidate(
    *,
    experiment_id: str,
    optimizer: str,
    candidate_id: str,
    prompt_file: str | Path,
    prompt_dev_dataset: str | Path,
    fresh_gate_dataset: str | Path | None = None,
    source_manifest: str | Path | None = None,
    model_variant: str | None = None,
    eval_result: str | Path | None = None,
    analysis: str | Path | None = None,
    decision: str = "pending",
    notes: str | None = None,
    output_path: str | Path | None = None,
    log_mlflow: bool | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
) -> SQLPromptCandidateSummary:
    """Write and optionally log one prompt candidate."""

    if optimizer not in PROMPT_OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {sorted(PROMPT_OPTIMIZERS)}")
    if decision not in PROMPT_CANDIDATE_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(PROMPT_CANDIDATE_DECISIONS)}")
    if not experiment_id.strip():
        raise ValueError("experiment_id is required")
    if not candidate_id.strip():
        raise ValueError("candidate_id is required")

    resolved_prompt_file = _resolve_workspace_path(prompt_file)
    if not resolved_prompt_file.exists():
        raise ValueError(f"prompt_file does not exist: {resolved_prompt_file}")
    prompt_text = resolved_prompt_file.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise ValueError(f"prompt_file must not be empty: {resolved_prompt_file}")

    resolved_prompt_dev_dataset = _resolve_workspace_path(prompt_dev_dataset)
    prompt_dev_cases = load_sql_eval_cases(resolved_prompt_dev_dataset)
    resolved_fresh_gate_dataset = (
        _resolve_workspace_path(fresh_gate_dataset)
        if fresh_gate_dataset is not None
        else None
    )
    fresh_gate_case_count = (
        len(load_sql_eval_cases(resolved_fresh_gate_dataset))
        if resolved_fresh_gate_dataset is not None
        else None
    )

    resolved_source_manifest = (
        _resolve_workspace_path(source_manifest)
        if source_manifest is not None
        else None
    )
    if resolved_source_manifest is not None and not resolved_source_manifest.exists():
        raise ValueError(f"source_manifest does not exist: {resolved_source_manifest}")

    resolved_eval_result = _resolve_workspace_path(eval_result) if eval_result is not None else None
    eval_payload = _load_eval_result(resolved_eval_result) if resolved_eval_result is not None else None
    eval_dataset_role = _eval_result_dataset_role(
        eval_payload=eval_payload,
        prompt_dev_dataset=resolved_prompt_dev_dataset,
        fresh_gate_dataset=resolved_fresh_gate_dataset,
    )
    resolved_analysis = _resolve_workspace_path(analysis) if analysis is not None else None
    if resolved_analysis is not None and not resolved_analysis.exists():
        raise ValueError(f"analysis does not exist: {resolved_analysis}")
    failure_counts = _failure_counts(eval_payload=eval_payload, analysis_path=resolved_analysis)

    resolved_output_path = (
        _resolve_workspace_path(output_path)
        if output_path is not None
        else WORKSPACE_ROOT / "results" / "sql" / experiment_id / "prompt_candidates" / f"{candidate_id}.json"
    )
    summary = SQLPromptCandidateSummary(
        schema_version="sql_prompt_candidate:v1",
        experiment_id=experiment_id,
        optimizer=optimizer,
        candidate_id=candidate_id,
        decision=decision,
        prompt_file=str(_display_path(resolved_prompt_file)),
        prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        prompt_char_count=len(prompt_text),
        prompt_dev_dataset=str(_display_path(resolved_prompt_dev_dataset)),
        prompt_dev_case_count=len(prompt_dev_cases),
        fresh_gate_dataset=(
            str(_display_path(resolved_fresh_gate_dataset))
            if resolved_fresh_gate_dataset is not None
            else None
        ),
        fresh_gate_case_count=fresh_gate_case_count,
        source_manifest=(
            str(_display_path(resolved_source_manifest))
            if resolved_source_manifest is not None
            else None
        ),
        model_variant=model_variant,
        eval_result=(
            str(_display_path(resolved_eval_result))
            if resolved_eval_result is not None
            else None
        ),
        eval_dataset_role=eval_dataset_role,
        eval_case_count=_optional_int(eval_payload, "case_count"),
        eval_passed_count=_optional_int(eval_payload, "passed_count"),
        eval_pass_rate=_optional_float(eval_payload, "pass_rate"),
        failure_counts=failure_counts,
        notes=notes,
        output_path=str(_display_path(resolved_output_path)),
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _maybe_log_mlflow_prompt_candidate(
        summary=summary,
        candidate_path=resolved_output_path,
        prompt_file=resolved_prompt_file,
        eval_result=resolved_eval_result,
        analysis=resolved_analysis,
        explicit=log_mlflow,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )
    return summary


def _load_eval_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"eval_result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"eval_result must be a JSON object: {path}")
    records = payload.get("records")
    if records is not None and not isinstance(records, list):
        raise ValueError(f"eval_result records must be a list: {path}")
    return payload


def _eval_result_dataset_role(
    *,
    eval_payload: dict[str, Any] | None,
    prompt_dev_dataset: Path,
    fresh_gate_dataset: Path | None,
) -> str | None:
    if eval_payload is None or eval_payload.get("eval_dataset") is None:
        return None
    eval_dataset = _resolve_workspace_path(str(eval_payload["eval_dataset"]))
    if eval_dataset == prompt_dev_dataset:
        return "prompt_dev"
    if fresh_gate_dataset is not None and eval_dataset == fresh_gate_dataset:
        return "fresh_gate"
    raise ValueError(
        "eval_result dataset must match prompt_dev_dataset or fresh_gate_dataset: "
        f"{eval_dataset}"
    )


def _failure_counts(*, eval_payload: dict[str, Any] | None, analysis_path: Path | None) -> dict[str, int]:
    if analysis_path is not None:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"analysis must be a JSON object: {analysis_path}")
        raw_counts = payload.get("failure_counts", {})
        if not isinstance(raw_counts, dict):
            raise ValueError(f"analysis failure_counts must be an object: {analysis_path}")
        return {str(key): int(value) for key, value in sorted(raw_counts.items())}
    if eval_payload is None:
        return {}
    records = eval_payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("eval_result records must be a list")
    counts: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict) or bool(record.get("passed")):
            continue
        failure_type = classify_sql_eval_failure(record)
        counts[failure_type] = counts.get(failure_type, 0) + 1
    return dict(sorted(counts.items()))


def _optional_int(payload: dict[str, Any] | None, key: str) -> int | None:
    if payload is None or payload.get(key) is None:
        return None
    return int(payload[key])


def _optional_float(payload: dict[str, Any] | None, key: str) -> float | None:
    if payload is None or payload.get(key) is None:
        return None
    return float(payload[key])


def _maybe_log_mlflow_prompt_candidate(
    *,
    summary: SQLPromptCandidateSummary,
    candidate_path: Path,
    prompt_file: Path,
    eval_result: Path | None,
    analysis: Path | None,
    explicit: bool | None,
    tracking_uri: str | None,
    experiment_name: str | None,
) -> None:
    from sqlbench_lab.observability import log_sql_prompt_candidate_run, mlflow_enabled

    if not mlflow_enabled(explicit):
        return
    log_sql_prompt_candidate_run(
        summary=summary,
        candidate_path=candidate_path,
        prompt_file=prompt_file,
        eval_result=eval_result,
        analysis=analysis,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return WORKSPACE_ROOT / candidate


def _display_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        return path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return path
