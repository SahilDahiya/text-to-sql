"""SQL SFT experiment manifest loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from sqlbench_lab.paths import WORKSPACE_ROOT


@dataclass(frozen=True)
class SQLStudentConfig:
    model_family: str
    base_model: str
    adapter_name: str


@dataclass(frozen=True)
class SQLTrainingMethodConfig:
    method: str
    loss_target: str
    stage: str
    notes: str | None


@dataclass(frozen=True)
class SQLTrainInputsConfig:
    train_datasets: tuple[str, ...]
    validation_datasets: tuple[str, ...]


@dataclass(frozen=True)
class SQLEvalPlanConfig:
    smoke_dataset: str
    baseline_results: str
    post_train_results: str


@dataclass(frozen=True)
class SQLOutputPathsConfig:
    experiment_root: str
    adapter_dir: str
    train_summary_json: str
    eval_summary_json: str


@dataclass(frozen=True)
class SQLSFTExperimentManifest:
    schema_version: str
    experiment_id: str
    student: SQLStudentConfig
    training_method: SQLTrainingMethodConfig
    train_inputs: SQLTrainInputsConfig
    eval_plan: SQLEvalPlanConfig
    output_paths: SQLOutputPathsConfig

    def resolve_workspace_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return WORKSPACE_ROOT / path


def load_sql_sft_manifest(path: str | Path) -> SQLSFTExperimentManifest:
    """Load and validate one SQL SFT experiment manifest."""

    resolved_path = _resolve_workspace_path(path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SQL SFT manifest must be a JSON object")
    _validate_manifest(payload)
    return SQLSFTExperimentManifest(
        schema_version=str(payload["schema_version"]),
        experiment_id=str(payload["experiment_id"]),
        student=SQLStudentConfig(**payload["student"]),
        training_method=SQLTrainingMethodConfig(**payload["training_method"]),
        train_inputs=SQLTrainInputsConfig(
            train_datasets=tuple(str(item) for item in payload["train_inputs"]["train_datasets"]),
            validation_datasets=tuple(
                str(item) for item in payload["train_inputs"]["validation_datasets"]
            ),
        ),
        eval_plan=SQLEvalPlanConfig(**payload["eval_plan"]),
        output_paths=SQLOutputPathsConfig(**payload["output_paths"]),
    )


def _validate_manifest(payload: dict[str, Any]) -> None:
    schema_path = WORKSPACE_ROOT / "schemas" / "sql_sft_experiment_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if not errors:
        return
    first_error = errors[0]
    error_path = ".".join(str(item) for item in first_error.absolute_path) or "<root>"
    raise ValueError(f"SQL SFT manifest schema validation failed at {error_path}: {first_error.message}")


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return WORKSPACE_ROOT / candidate

