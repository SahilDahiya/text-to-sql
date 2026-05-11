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
class SQLTrainerConfig:
    backend: str
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    logging_steps: int


@dataclass(frozen=True)
class SQLLoRAConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]


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
    trainer: SQLTrainerConfig
    lora: SQLLoRAConfig
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
        trainer=_load_trainer_config(payload.get("trainer", {})),
        lora=_load_lora_config(payload.get("lora", _default_lora_payload())),
        eval_plan=SQLEvalPlanConfig(**payload["eval_plan"]),
        output_paths=SQLOutputPathsConfig(**payload["output_paths"]),
    )


def _default_trainer_payload() -> dict[str, int | float | str]:
    return {
        "backend": "transformers_trainer",
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "logging_steps": 1,
    }


def _load_trainer_config(payload: dict[str, Any]) -> SQLTrainerConfig:
    merged = {**_default_trainer_payload(), **payload}
    return SQLTrainerConfig(
        backend=str(merged["backend"]),
        num_train_epochs=float(merged["num_train_epochs"]),
        per_device_train_batch_size=int(merged["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(merged["gradient_accumulation_steps"]),
        learning_rate=float(merged["learning_rate"]),
        logging_steps=int(merged["logging_steps"]),
    )


def _default_lora_payload() -> dict[str, Any]:
    return {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }


def _load_lora_config(payload: dict[str, Any]) -> SQLLoRAConfig:
    return SQLLoRAConfig(
        r=int(payload["r"]),
        lora_alpha=int(payload["lora_alpha"]),
        lora_dropout=float(payload["lora_dropout"]),
        target_modules=tuple(str(item) for item in payload["target_modules"]),
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
