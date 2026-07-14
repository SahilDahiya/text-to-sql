"""Manifest-driven one-shot SQL model evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from sqlbench_lab.paths import WORKSPACE_ROOT

from .eval_types import SQLCaseEvalRecord, SQLEvalRunSummary
from .evaluator import evaluate_sql_case
from .loaders import load_sql_eval_cases
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .models import SQLEvalCase
from .rendering import SQL_SYSTEM_PROMPT, build_eval_messages
from .training import (
    _ensure_pad_token,
    _import_training_stack,
    _inner_tokenizer,
    _load_tokenizer_like,
    _load_trainable_model,
    render_sql_sft_prompt,
)

SQL_MODEL_VARIANTS = {"base", "adapter"}


def run_sql_eval(
    manifest_path: str | Path,
    *,
    model_variant: str,
    eval_dataset: str | Path | None = None,
    max_new_tokens: int | None = None,
    system_prompt: str | None = None,
    result_label: str | None = None,
    predictor: Callable[[SQLEvalCase], str] | None = None,
) -> SQLEvalRunSummary:
    """Run exactly one deterministic, one-shot generation pass."""

    if model_variant not in SQL_MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(SQL_MODEL_VARIANTS)}")
    manifest = load_sql_sft_manifest(manifest_path)
    eval_dataset_path = str(eval_dataset) if eval_dataset is not None else manifest.eval_plan.target_dataset
    cases = load_sql_eval_cases(eval_dataset_path)
    resolved_max_new_tokens = (
        manifest.eval_plan.max_new_tokens if max_new_tokens is None else max_new_tokens
    )
    if resolved_max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    result_path = _eval_result_path(manifest, model_variant, eval_dataset=eval_dataset, result_label=result_label)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
    predict_sql = (
        predictor
        if predictor is not None
        else _build_hf_predictor(
            manifest=manifest,
            model_variant=model_variant,
            adapter_dir=adapter_dir,
            max_new_tokens=resolved_max_new_tokens,
            system_prompt=SQL_SYSTEM_PROMPT if system_prompt is None else system_prompt,
        )
    )
    records = [_evaluate_case(case, model_variant=model_variant, predict_sql=predict_sql) for case in cases]
    passed_count = sum(1 for record in records if record.passed)
    summary = SQLEvalRunSummary(
        schema_version="sql_eval_run:v2",
        experiment_id=manifest.experiment_id,
        base_model=manifest.student.base_model,
        model_variant=model_variant,
        adapter_dir=str(adapter_dir) if model_variant == "adapter" else None,
        eval_dataset=eval_dataset_path,
        dataset_fingerprint=_file_fingerprint(eval_dataset_path),
        eval_db_ids=tuple(sorted({case.db_id for case in cases})),
        scorer_version=manifest.eval_plan.scorer_version,
        generation_config={"max_new_tokens": resolved_max_new_tokens, "do_sample": False},
        result_path=str(result_path),
        case_count=len(records),
        passed_count=passed_count,
        pass_rate=passed_count / len(records),
        records=records,
    )
    result_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return summary


def extract_generated_sql(text: str) -> str:
    """Normalize model text into the single SQL statement passed to evaluation."""

    stripped = text.strip()
    stripped = _strip_code_fence(stripped)
    if "<|assistant|>" in stripped:
        stripped = stripped.split("<|assistant|>", maxsplit=1)[-1].strip()
    match = re.search(r";", stripped)
    if match:
        return stripped[: match.end()].strip()
    return stripped


def _build_hf_predictor(
    *,
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    adapter_dir: Path,
    max_new_tokens: int,
    system_prompt: str,
) -> Callable[[SQLEvalCase], str]:
    predict_messages = _build_hf_message_predictor(
        manifest=manifest,
        model_variant=model_variant,
        adapter_dir=adapter_dir,
        max_new_tokens=max_new_tokens,
    )

    def predict(case: SQLEvalCase) -> str:
        return predict_messages(
            build_eval_messages(
                case,
                prompt_style=manifest.prompt.style,
                system_prompt=system_prompt,
            )
        )

    return predict


def _build_hf_message_predictor(
    *,
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    adapter_dir: Path,
    max_new_tokens: int,
) -> Callable[[list[dict[str, str]]], str]:
    torch, transformers, peft = _import_training_stack()
    tokenizer_like = _load_tokenizer_like(transformers, manifest.student.base_model)
    _ensure_pad_token(tokenizer_like)
    tokenizer = _inner_tokenizer(tokenizer_like)
    model = _load_trainable_model(
        transformers,
        manifest.student.base_model,
        torch_module=torch,
        attn_implementation=manifest.trainer.attn_implementation,
    )
    if model_variant == "adapter":
        if not adapter_dir.is_dir():
            raise ValueError(f"adapter_dir does not exist: {adapter_dir}")
        model = peft.PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    if hasattr(model, "cuda") and torch.cuda.is_available():
        model = model.cuda()

    def predict(messages: list[dict[str, str]]) -> str:
        prompt = _render_generation_prompt(
            tokenizer,
            [*messages, {"role": "assistant", "content": ""}],
            model_variant=model_variant,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        input_length = int(encoded["input_ids"].shape[-1])
        generated_ids = output_ids[0][input_length:]
        return extract_generated_sql(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return predict


def _render_generation_prompt(tokenizer: object, messages: list[dict[str, str]], *, model_variant: str) -> str:
    if model_variant == "base" and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True))
    return render_sql_sft_prompt(messages)


def _evaluate_case(
    case: SQLEvalCase,
    *,
    model_variant: str,
    predict_sql: Callable[[SQLEvalCase], str],
) -> SQLCaseEvalRecord:
    predicted_sql = predict_sql(case)
    result = evaluate_sql_case(case, predicted_sql=predicted_sql)
    return SQLCaseEvalRecord(
        case_id=case.case_id,
        task_id=case.task_id,
        db_id=case.db_id,
        model_variant=model_variant,
        predicted_sql=predicted_sql,
        passed=result.passed,
        primary_failure_type="passed" if result.passed else "failed",
        prediction_error=result.prediction_error,
        gold_error=result.gold_error,
        predicted_rows=result.predicted_rows,
        gold_rows=result.gold_rows,
        predicted_columns=result.predicted_columns,
        gold_columns=result.gold_columns,
        execution_ms=result.execution_ms,
    )


def _eval_result_path(
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    *,
    eval_dataset: str | Path | None,
    result_label: str | None,
) -> Path:
    if eval_dataset is None:
        if model_variant == "base":
            return manifest.resolve_workspace_path(manifest.eval_plan.baseline_results)
        return manifest.resolve_workspace_path(manifest.eval_plan.post_train_results)
    dataset_stem = Path(eval_dataset).stem
    label = f"__{result_label}" if result_label else ""
    return manifest.resolve_workspace_path(manifest.output_paths.experiment_root) / "eval" / f"{model_variant}__{dataset_stem}{label}.json"


def _file_fingerprint(path: str | Path) -> str:
    raw_path = Path(path).expanduser()
    resolved = raw_path if raw_path.is_absolute() else WORKSPACE_ROOT / raw_path
    resolved = resolved.resolve()
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _strip_code_fence(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return text
