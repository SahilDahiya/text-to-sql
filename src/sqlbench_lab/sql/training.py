"""Minimal LoRA SFT runner for SQL experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .loaders import load_sql_eval_cases, load_sql_train_examples
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .rendering import build_train_messages
from .training_types import SQLSFTTrainingSummary

IGNORE_INDEX = -100
SUPPORTED_METHOD = "lora_sft"
SUPPORTED_LOSS_TARGET = "assistant_sql_only"
SUPPORTED_STAGE = "direct_sql_sft"


def run_sql_sft(
    manifest_path: str | Path,
    *,
    dry_run: bool = False,
    log_mlflow: bool | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
) -> SQLSFTTrainingSummary:
    """Run one minimal manifest-driven SQL LoRA SFT experiment."""

    manifest = load_sql_sft_manifest(manifest_path)
    _validate_supported_manifest(manifest)
    resolved_manifest_path = _resolve_manifest_path(manifest_path)
    train_rows = []
    train_dataset_counts: dict[str, int] = {}
    for dataset_path in manifest.train_inputs.train_datasets:
        dataset_rows = load_sql_train_examples(dataset_path)
        train_dataset_counts[dataset_path] = len(dataset_rows)
        train_rows.extend(dataset_rows)
    if not train_rows:
        raise ValueError("SQL SFT training requires at least one train row")

    rendered_messages = [build_train_messages(row) for row in train_rows]
    experiment_root = manifest.resolve_workspace_path(manifest.output_paths.experiment_root)
    adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
    train_summary_path = manifest.resolve_workspace_path(manifest.output_paths.train_summary_json)
    smoke_eval_case_count = len(load_sql_eval_cases(manifest.eval_plan.smoke_dataset))
    experiment_root.mkdir(parents=True, exist_ok=True)
    train_summary_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        dry_run_summary_path = _dry_run_summary_path(train_summary_path)
        summary = SQLSFTTrainingSummary(
            experiment_id=manifest.experiment_id,
            base_model=manifest.student.base_model,
            adapter_dir=str(adapter_dir),
            train_row_count=len(rendered_messages),
            dry_run=True,
        )
        _write_summary(dry_run_summary_path, summary)
        _maybe_log_mlflow(
            manifest=manifest,
            manifest_path=resolved_manifest_path,
            summary=summary,
            summary_path=dry_run_summary_path,
            train_dataset_counts=train_dataset_counts,
            smoke_eval_case_count=smoke_eval_case_count,
            training_config=_training_config(),
            lora_config=_lora_config(),
            explicit=log_mlflow,
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment,
        )
        return summary

    torch, transformers, peft = _import_training_stack()

    tokenizer = _load_tokenizer_like(transformers, manifest.student.base_model)
    _ensure_pad_token(tokenizer)

    encoded_examples = [
        tokenize_sql_sft_messages(tokenizer=tokenizer, messages=messages)
        for messages in rendered_messages
    ]

    model = _load_trainable_model(transformers, manifest.student.base_model, torch_module=torch)
    lora_config = _lora_config()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
        ),
    )

    training_config = _training_config()
    training_args = transformers.TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        logging_steps=training_config["logging_steps"],
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=_SQLSFTDataset(encoded_examples),
        data_collator=_SQLSFTDataCollator(pad_token_id=_pad_token_id(tokenizer), torch_module=torch),
    )
    train_output = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    trainable_parameters, total_parameters = _parameter_counts(model)
    training_metrics = _training_metrics(train_output)
    summary = SQLSFTTrainingSummary(
        experiment_id=manifest.experiment_id,
        base_model=manifest.student.base_model,
        adapter_dir=str(adapter_dir),
        train_row_count=len(rendered_messages),
        dry_run=False,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        training_metrics=training_metrics,
    )
    _write_summary(train_summary_path, summary)
    _maybe_log_mlflow(
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        summary=summary,
        summary_path=train_summary_path,
        train_dataset_counts=train_dataset_counts,
        smoke_eval_case_count=smoke_eval_case_count,
        training_config=training_config,
        lora_config=lora_config,
        explicit=log_mlflow,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )
    return summary


def tokenize_sql_sft_messages(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, list[int]]:
    """Tokenize one rendered SQL SFT sample and mask prompt tokens."""

    if len(messages) != 3 or messages[-1]["role"] != "assistant":
        raise ValueError("SQL SFT messages must be system/user/assistant")
    prompt_text = render_sql_sft_prompt(messages)
    target_text = messages[-1]["content"]
    prompt_ids = _tokenize_text(tokenizer, prompt_text)
    target_ids = _tokenize_text(tokenizer, target_text + _eos_text(tokenizer))
    input_ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def render_sql_sft_prompt(messages: list[dict[str, str]]) -> str:
    """Render the repo-owned base-model SFT prompt format."""

    if len(messages) != 3:
        raise ValueError("SQL SFT messages must contain system, user, and assistant messages")
    system_message, user_message, assistant_message = messages
    if system_message["role"] != "system":
        raise ValueError("first SQL SFT message must have role=system")
    if user_message["role"] != "user":
        raise ValueError("second SQL SFT message must have role=user")
    if assistant_message["role"] != "assistant":
        raise ValueError("third SQL SFT message must have role=assistant")
    return (
        f"<|system|>\n{system_message['content'].strip()}\n"
        f"<|user|>\n{user_message['content'].strip()}\n"
        "<|assistant|>\n"
    )


def _validate_supported_manifest(manifest: SQLSFTExperimentManifest) -> None:
    if manifest.training_method.method != SUPPORTED_METHOD:
        raise ValueError(f"SQL SFT runner only supports method={SUPPORTED_METHOD!r}")
    if manifest.training_method.loss_target != SUPPORTED_LOSS_TARGET:
        raise ValueError(f"SQL SFT runner only supports loss_target={SUPPORTED_LOSS_TARGET!r}")
    if manifest.training_method.stage != SUPPORTED_STAGE:
        raise ValueError(f"SQL SFT runner only supports stage={SUPPORTED_STAGE!r}")
    if manifest.train_inputs.validation_datasets:
        raise ValueError("SQL SFT runner does not support validation_datasets yet")


def _training_config() -> dict[str, int | float]:
    return {
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "logging_steps": 1,
    }


def _lora_config() -> dict[str, Any]:
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


def _training_metrics(train_output: Any) -> dict[str, float]:
    raw_metrics = getattr(train_output, "metrics", {})
    if not isinstance(raw_metrics, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            metrics[str(key)] = float(value)
    return metrics


def _maybe_log_mlflow(
    *,
    manifest: SQLSFTExperimentManifest,
    manifest_path: Path,
    summary: SQLSFTTrainingSummary,
    summary_path: Path,
    train_dataset_counts: dict[str, int],
    smoke_eval_case_count: int,
    training_config: dict[str, Any],
    lora_config: dict[str, Any],
    explicit: bool | None,
    tracking_uri: str | None,
    experiment_name: str | None,
) -> None:
    from sqlbench_lab.observability import log_sql_sft_run, mlflow_enabled

    if not mlflow_enabled(explicit):
        return
    log_sql_sft_run(
        manifest=manifest,
        manifest_path=manifest_path,
        summary=summary,
        summary_path=summary_path,
        train_dataset_counts=train_dataset_counts,
        smoke_eval_case_count=smoke_eval_case_count,
        training_config=training_config,
        lora_config=lora_config,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )


def _resolve_manifest_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _dry_run_summary_path(train_summary_path: Path) -> Path:
    return train_summary_path.with_name(
        f"{train_summary_path.stem}.dry_run{train_summary_path.suffix}"
    )


def _import_training_stack() -> tuple[Any, Any, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "SQL SFT training requires torch, transformers, and peft. "
            "Install the training extras before running without --dry-run."
        ) from exc
    return torch, transformers, peft


def _load_tokenizer_like(transformers: Any, base_model: str) -> Any:
    if _is_qwen35_model(base_model):
        return transformers.AutoProcessor.from_pretrained(base_model)
    return transformers.AutoTokenizer.from_pretrained(base_model)


def _load_trainable_model(transformers: Any, base_model: str, *, torch_module: Any) -> Any:
    kwargs = {"torch_dtype": _default_torch_dtype(torch_module)}
    if _is_qwen35_model(base_model):
        return transformers.AutoModelForImageTextToText.from_pretrained(base_model, **kwargs)
    return transformers.AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def _is_qwen35_model(base_model: str) -> bool:
    return base_model.startswith("Qwen/Qwen3.5-")


def _ensure_pad_token(tokenizer_like: Any) -> None:
    tokenizer = _inner_tokenizer(tokenizer_like)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("tokenizer must expose either pad_token_id or eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token


def _pad_token_id(tokenizer_like: Any) -> int:
    tokenizer = _inner_tokenizer(tokenizer_like)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must expose pad_token_id after setup")
    return int(pad_token_id)


def _inner_tokenizer(tokenizer_like: Any) -> Any:
    return getattr(tokenizer_like, "tokenizer", tokenizer_like)


def _default_torch_dtype(torch_module: Any) -> Any:
    if hasattr(torch_module, "bfloat16"):
        return torch_module.bfloat16
    return None


def _parameter_counts(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def _write_summary(path: Path, summary: SQLSFTTrainingSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _extract_token_ids(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, dict) and "input_ids" in value:
        return _extract_token_ids(value["input_ids"])
    if hasattr(value, "input_ids"):
        return _extract_token_ids(value.input_ids)
    raise ValueError("unable to extract token ids from tokenizer output")


def _tokenize_text(tokenizer_like: Any, text: str) -> list[int]:
    tokenizer = _inner_tokenizer(tokenizer_like)
    return _extract_token_ids(_encode_target_text(tokenizer, text))


def _eos_text(tokenizer: Any) -> str:
    eos_token = getattr(_inner_tokenizer(tokenizer), "eos_token", None)
    return eos_token if isinstance(eos_token, str) else ""


def _encode_target_text(tokenizer_like: Any, text: str) -> Any:
    try:
        return tokenizer_like(text, add_special_tokens=False)["input_ids"]
    except TypeError:
        return tokenizer_like(text=text, add_special_tokens=False)["input_ids"]


class _SQLSFTDataset:
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class _SQLSFTDataCollator:
    def __init__(self, *, pad_token_id: int, torch_module: Any) -> None:
        self.pad_token_id = pad_token_id
        self.torch = torch_module

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        labels = []
        attention_mask = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad_length)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad_length)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention_mask, dtype=self.torch.long),
        }
