"""Strict v2 SQL SFT experiment manifest loading."""

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
    initial_adapter_dir: str | None = None


@dataclass(frozen=True)
class SQLTrainingMethodConfig:
    method: str
    loss_target: str
    stage: str
    notes: str | None


@dataclass(frozen=True)
class SQLPromptConfig:
    style: str


@dataclass(frozen=True)
class SQLMixtureConfig:
    dataset_id: str
    source_package: str
    source_revision: str
    fingerprint: str


@dataclass(frozen=True)
class SQLTrainInputsConfig:
    train_datasets: tuple[str, ...]


@dataclass(frozen=True)
class SQLTrainerConfig:
    backend: str
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    logging_steps: int
    attn_implementation: str | None
    packing: bool
    packing_strategy: str
    max_length: int | None
    bf16: bool | None
    tf32: bool | None
    gradient_checkpointing: bool
    save_strategy: str
    save_steps: int | None
    save_total_limit: int | None
    auto_resume_from_checkpoint: bool


@dataclass(frozen=True)
class SQLQuantizationConfig:
    mode: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str
    device_map: str | None
    prepare_model_for_kbit_training: bool


@dataclass(frozen=True)
class SQLLoRAConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class SQLEvalPlanConfig:
    target_dataset: str
    baseline_results: str
    post_train_results: str
    scorer_version: str
    max_new_tokens: int


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
    prompt: SQLPromptConfig
    mixture: SQLMixtureConfig
    train_inputs: SQLTrainInputsConfig
    trainer: SQLTrainerConfig
    quantization: SQLQuantizationConfig
    lora: SQLLoRAConfig
    eval_plan: SQLEvalPlanConfig
    output_paths: SQLOutputPathsConfig

    def resolve_workspace_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return WORKSPACE_ROOT / path

def load_sql_sft_manifest(path: str | Path) -> SQLSFTExperimentManifest:
    """Load and validate one v2 manifest."""

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
        prompt=SQLPromptConfig(**payload["prompt"]),
        mixture=SQLMixtureConfig(**payload["mixture"]),
        train_inputs=SQLTrainInputsConfig(
            train_datasets=tuple(str(item) for item in payload["train_inputs"]["train_datasets"]),
        ),
        trainer=SQLTrainerConfig(**payload["trainer"]),
        quantization=SQLQuantizationConfig(**payload["quantization"]),
        lora=SQLLoRAConfig(
            r=int(payload["lora"]["r"]),
            lora_alpha=int(payload["lora"]["lora_alpha"]),
            lora_dropout=float(payload["lora"]["lora_dropout"]),
            bias=str(payload["lora"]["bias"]),
            target_modules=tuple(str(item) for item in payload["lora"]["target_modules"]),
        ),
        eval_plan=SQLEvalPlanConfig(
            target_dataset=str(payload["eval_plan"]["target_dataset"]),
            baseline_results=str(payload["eval_plan"]["baseline_results"]),
            post_train_results=str(payload["eval_plan"]["post_train_results"]),
            scorer_version=str(payload["eval_plan"]["scorer_version"]),
            max_new_tokens=int(payload["eval_plan"]["max_new_tokens"]),
        ),
        output_paths=SQLOutputPathsConfig(**payload["output_paths"]),
    )


def _validate_manifest(payload: dict[str, Any]) -> None:
    schema_path = WORKSPACE_ROOT / "schemas" / "sql_sft_experiment_v2.schema.json"
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
