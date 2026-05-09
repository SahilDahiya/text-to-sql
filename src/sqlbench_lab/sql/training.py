"""Minimal LoRA SFT runner for SQL experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .loaders import load_sql_train_examples
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .rendering import build_train_messages

IGNORE_INDEX = -100
SUPPORTED_METHOD = "lora_sft"
SUPPORTED_LOSS_TARGET = "assistant_sql_only"
SUPPORTED_STAGE = "direct_sql_sft"


@dataclass(frozen=True)
class SQLSFTTrainingSummary:
    experiment_id: str
    base_model: str
    adapter_dir: str
    train_row_count: int
    dry_run: bool
    trainable_parameters: int | None = None
    total_parameters: int | None = None


def run_sql_sft(manifest_path: str | Path, *, dry_run: bool = False) -> SQLSFTTrainingSummary:
    """Run one minimal manifest-driven SQL LoRA SFT experiment."""

    manifest = load_sql_sft_manifest(manifest_path)
    _validate_supported_manifest(manifest)
    train_rows = [
        row
        for dataset_path in manifest.train_inputs.train_datasets
        for row in load_sql_train_examples(dataset_path)
    ]
    if not train_rows:
        raise ValueError("SQL SFT training requires at least one train row")

    rendered_messages = [build_train_messages(row) for row in train_rows]
    experiment_root = manifest.resolve_workspace_path(manifest.output_paths.experiment_root)
    adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
    train_summary_path = manifest.resolve_workspace_path(manifest.output_paths.train_summary_json)
    experiment_root.mkdir(parents=True, exist_ok=True)
    train_summary_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        summary = SQLSFTTrainingSummary(
            experiment_id=manifest.experiment_id,
            base_model=manifest.student.base_model,
            adapter_dir=str(adapter_dir),
            train_row_count=len(rendered_messages),
            dry_run=True,
        )
        _write_summary(train_summary_path, summary)
        return summary

    torch, transformers, peft = _import_training_stack()

    tokenizer = _load_tokenizer_like(transformers, manifest.student.base_model)
    _ensure_pad_token(tokenizer)

    encoded_examples = [
        tokenize_sql_sft_messages(tokenizer=tokenizer, messages=messages)
        for messages in rendered_messages
    ]

    model = _load_trainable_model(transformers, manifest.student.base_model, torch_module=torch)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    training_args = transformers.TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
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
    trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    trainable_parameters, total_parameters = _parameter_counts(model)
    summary = SQLSFTTrainingSummary(
        experiment_id=manifest.experiment_id,
        base_model=manifest.student.base_model,
        adapter_dir=str(adapter_dir),
        train_row_count=len(rendered_messages),
        dry_run=False,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
    )
    _write_summary(train_summary_path, summary)
    return summary


def tokenize_sql_sft_messages(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, list[int]]:
    """Tokenize one rendered SQL SFT chat sample and mask prompt tokens."""

    if len(messages) != 3 or messages[-1]["role"] != "assistant":
        raise ValueError("SQL SFT messages must be system/user/assistant")
    prompt_messages = messages[:-1]
    target_text = messages[-1]["content"]
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer must support apply_chat_template for SQL SFT")

    prompt_ids = _extract_token_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    target_ids = _extract_token_ids(
        _encode_target_text(tokenizer, target_text + _eos_text(tokenizer))
    )
    input_ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def _validate_supported_manifest(manifest: SQLSFTExperimentManifest) -> None:
    if manifest.training_method.method != SUPPORTED_METHOD:
        raise ValueError(f"SQL SFT runner only supports method={SUPPORTED_METHOD!r}")
    if manifest.training_method.loss_target != SUPPORTED_LOSS_TARGET:
        raise ValueError(f"SQL SFT runner only supports loss_target={SUPPORTED_LOSS_TARGET!r}")
    if manifest.training_method.stage != SUPPORTED_STAGE:
        raise ValueError(f"SQL SFT runner only supports stage={SUPPORTED_STAGE!r}")
    if manifest.train_inputs.validation_datasets:
        raise ValueError("SQL SFT runner does not support validation_datasets yet")


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
