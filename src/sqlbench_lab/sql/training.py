"""Minimal LoRA SFT runner for SQL experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .loaders import load_sql_train_examples
from .manifest import SQLLoRAConfig, SQLQuantizationConfig, SQLSFTExperimentManifest, SQLTrainerConfig, load_sql_sft_manifest
from .rendering import build_train_messages
from .review import require_approved_review
from .training_types import SQLSFTTrainingSummary

IGNORE_INDEX = -100
LORA_SFT_METHOD = "lora_sft"
QLORA_SFT_METHOD = "qlora_sft"
SUPPORTED_METHODS = {LORA_SFT_METHOD, QLORA_SFT_METHOD}
NO_QUANTIZATION_MODE = "none"
BITSANDBYTES_4BIT_QUANTIZATION_MODE = "bitsandbytes_4bit"


def run_sql_sft(
    manifest_path: str | Path,
    *,
    dry_run: bool = False,
    review_path: str | Path | None = None,
) -> SQLSFTTrainingSummary:
    """Run one minimal manifest-driven SQL LoRA SFT experiment."""

    manifest = load_sql_sft_manifest(manifest_path)
    if review_path is not None:
        require_approved_review(review_path, manifest_path=manifest_path)
    _validate_supported_manifest(manifest)
    train_rows = []
    for dataset_path in manifest.train_datasets:
        dataset_rows = load_sql_train_examples(dataset_path)
        train_rows.extend(dataset_rows)
    if not train_rows:
        raise ValueError("SQL SFT training requires at least one train row")

    rendered_messages = [
        build_train_messages(row)
        for row in train_rows
    ]
    experiment_root = manifest.resolve_workspace_path(manifest.output_paths.experiment_root)
    adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
    train_summary_path = manifest.resolve_workspace_path(manifest.output_paths.train_summary_json)
    experiment_root.mkdir(parents=True, exist_ok=True)
    train_summary_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        dry_run_summary_path = train_summary_path.with_name(
            f"{train_summary_path.stem}.dry_run{train_summary_path.suffix}"
        )
        summary = SQLSFTTrainingSummary(
            experiment_id=manifest.experiment_id,
            base_model=manifest.base_model,
            adapter_dir=str(adapter_dir),
            train_row_count=len(rendered_messages),
            dry_run=True,
        )
        _write_summary(dry_run_summary_path, summary)
        return summary

    torch, transformers, peft = _import_training_stack()

    tokenizer = transformers.AutoTokenizer.from_pretrained(manifest.base_model)
    _ensure_pad_token(tokenizer)

    model = _load_trainable_model(
        transformers,
        manifest.base_model,
        torch_module=torch,
        attn_implementation=manifest.trainer.attn_implementation,
        quantization_config=manifest.quantization,
    )
    model = _prepare_model_for_quantized_training(peft, model, manifest.quantization)
    model, initial_adapter_loaded = _load_initial_trainable_adapter(
        peft,
        model,
        manifest.resolve_workspace_path(manifest.initial_adapter_dir)
        if manifest.initial_adapter_dir is not None
        else None,
    )

    model, train_output = _train_with_transformers_trainer(
        torch_module=torch,
        transformers=transformers,
        peft=peft,
        tokenizer=tokenizer,
        model=model,
        rendered_messages=rendered_messages,
        adapter_dir=adapter_dir,
        training_config=manifest.trainer,
        lora_config=manifest.lora,
        initial_adapter_loaded=initial_adapter_loaded,
    )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    trainable_parameters, total_parameters = _parameter_counts(model)
    training_metrics = _training_metrics(train_output)
    summary = SQLSFTTrainingSummary(
        experiment_id=manifest.experiment_id,
        base_model=manifest.base_model,
        adapter_dir=str(adapter_dir),
        train_row_count=len(rendered_messages),
        dry_run=False,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        training_metrics=training_metrics,
    )
    _write_summary(train_summary_path, summary)
    return summary


def _train_with_transformers_trainer(
    *,
    torch_module: Any,
    transformers: Any,
    peft: Any,
    tokenizer: Any,
    model: Any,
    rendered_messages: list[list[dict[str, str]]],
    adapter_dir: Path,
    training_config: SQLTrainerConfig,
    lora_config: SQLLoRAConfig,
    initial_adapter_loaded: bool,
) -> tuple[Any, Any]:
    encoded_examples = [
        tokenize_sql_sft_messages(tokenizer=tokenizer, messages=messages)
        for messages in rendered_messages
    ]
    if not initial_adapter_loaded:
        model = peft.get_peft_model(model, _peft_lora_config(peft, lora_config))
    training_args_kwargs = {
        "output_dir": str(adapter_dir),
        "num_train_epochs": training_config.num_train_epochs,
        "per_device_train_batch_size": training_config.per_device_train_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "learning_rate": training_config.learning_rate,
        "logging_steps": training_config.logging_steps,
        "save_strategy": training_config.save_strategy,
        "report_to": [],
        "remove_unused_columns": False,
    }
    if training_config.save_steps is not None:
        training_args_kwargs["save_steps"] = training_config.save_steps
    if training_config.save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = training_config.save_total_limit
    training_args = transformers.TrainingArguments(**training_args_kwargs)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=_SQLSFTDataset(encoded_examples),
        data_collator=_SQLSFTDataCollator(pad_token_id=_pad_token_id(tokenizer), torch_module=torch_module),
    )
    train_output = trainer.train()
    return model, train_output


def tokenize_sql_sft_messages(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, list[int]]:
    """Tokenize one rendered SQL SFT sample and mask prompt tokens."""

    if len(messages) != 3 or messages[-1]["role"] != "assistant":
        raise ValueError("SQL SFT messages must be system/user/assistant")
    prompt_text = render_sql_sft_prompt(messages)
    target_text = messages[-1]["content"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
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
    if manifest.method not in SUPPORTED_METHODS:
        raise ValueError(f"SQL SFT runner only supports method in {sorted(SUPPORTED_METHODS)!r}")
    if manifest.method == QLORA_SFT_METHOD:
        if manifest.quantization.mode != BITSANDBYTES_4BIT_QUANTIZATION_MODE:
            raise ValueError("method='qlora_sft' requires quantization.mode='bitsandbytes_4bit'")
        if not manifest.quantization.prepare_model_for_kbit_training:
            raise ValueError("method='qlora_sft' requires prepare_model_for_kbit_training=true")
    if manifest.method == LORA_SFT_METHOD and manifest.quantization.mode != NO_QUANTIZATION_MODE:
        raise ValueError("method='lora_sft' requires quantization.mode='none'")


def _peft_lora_config(peft: Any, lora_config: SQLLoRAConfig) -> Any:
    return peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=list(lora_config.target_modules),
    )


def _load_initial_trainable_adapter(
    peft: Any,
    model: Any,
    adapter_dir: Path | None,
) -> tuple[Any, bool]:
    if adapter_dir is None:
        return model, False
    if not adapter_dir.exists():
        raise ValueError(f"initial_adapter_dir does not exist: {adapter_dir}")
    missing_files = [
        filename
        for filename in ("adapter_config.json", "adapter_model.safetensors")
        if not (adapter_dir / filename).exists()
    ]
    if missing_files:
        raise ValueError(
            "initial_adapter_dir is missing required PEFT adapter file(s): "
            + ", ".join(missing_files)
        )
    return peft.PeftModel.from_pretrained(model, adapter_dir, is_trainable=True), True


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


def _load_trainable_model(
    transformers: Any,
    base_model: str,
    *,
    torch_module: Any,
    attn_implementation: str | None = None,
    quantization_config: SQLQuantizationConfig | None = None,
) -> Any:
    kwargs = {"torch_dtype": torch_module.bfloat16 if hasattr(torch_module, "bfloat16") else None}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if quantization_config is not None and quantization_config.mode != NO_QUANTIZATION_MODE:
        kwargs.update(_quantized_model_load_kwargs(transformers, torch_module, quantization_config))
    return transformers.AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def _prepare_model_for_quantized_training(
    peft: Any,
    model: Any,
    quantization_config: SQLQuantizationConfig,
) -> Any:
    if not quantization_config.prepare_model_for_kbit_training:
        return model
    return peft.prepare_model_for_kbit_training(model)


def _quantized_model_load_kwargs(
    transformers: Any,
    torch_module: Any,
    quantization_config: SQLQuantizationConfig,
) -> dict[str, Any]:
    if quantization_config.mode != BITSANDBYTES_4BIT_QUANTIZATION_MODE:
        raise ValueError(f"unsupported quantization mode: {quantization_config.mode}")
    bitsandbytes_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quantization_config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_torch_dtype_from_name(
            torch_module,
            quantization_config.bnb_4bit_compute_dtype,
        ),
    )
    kwargs: dict[str, Any] = {"quantization_config": bitsandbytes_config}
    if quantization_config.device_map is not None:
        kwargs["device_map"] = quantization_config.device_map
    return kwargs


def _torch_dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"unsupported torch dtype name: {dtype_name}")


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("tokenizer must expose either pad_token_id or eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token


def _pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must expose pad_token_id after setup")
    return int(pad_token_id)


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
