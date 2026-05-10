"""Manifest-driven SQL model evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .eval_types import SQLCaseEvalRecord, SQLEvalRunSummary
from .evaluator import evaluate_sqlite_case
from .loaders import load_sql_eval_cases
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .models import SQLEvalCase
from .rendering import build_eval_messages
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
    max_new_tokens: int = 128,
    predictor: Callable[[SQLEvalCase], str] | None = None,
    log_mlflow: bool | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
) -> SQLEvalRunSummary:
    """Run the manifest smoke eval with result-equivalence scoring."""

    if model_variant not in SQL_MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(SQL_MODEL_VARIANTS)}")

    manifest = load_sql_sft_manifest(manifest_path)
    eval_dataset_path = str(eval_dataset) if eval_dataset is not None else manifest.eval_plan.smoke_dataset
    cases = load_sql_eval_cases(eval_dataset_path)
    if not cases:
        raise ValueError("SQL eval requires at least one eval case")

    result_path = _eval_result_path(manifest, model_variant, eval_dataset=eval_dataset)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
    predict_sql = predictor or _build_hf_predictor(
        manifest=manifest,
        model_variant=model_variant,
        adapter_dir=adapter_dir,
        max_new_tokens=max_new_tokens,
    )

    records = [_evaluate_case(case, model_variant=model_variant, predict_sql=predict_sql) for case in cases]
    passed_count = sum(1 for record in records if record.passed)
    summary = SQLEvalRunSummary(
        experiment_id=manifest.experiment_id,
        base_model=manifest.student.base_model,
        model_variant=model_variant,
        adapter_dir=str(adapter_dir) if model_variant == "adapter" else None,
        eval_dataset=eval_dataset_path,
        result_path=str(result_path),
        case_count=len(records),
        passed_count=passed_count,
        pass_rate=passed_count / len(records),
        records=records,
    )
    _write_eval_summary(result_path, summary)
    _maybe_log_mlflow_eval(
        manifest=manifest,
        manifest_path=_resolve_manifest_path(manifest_path),
        summary=summary,
        result_path=result_path,
        explicit=log_mlflow,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )
    return summary


def extract_generated_sql(text: str) -> str:
    """Normalize model text into the SQL string passed to execution eval."""

    stripped = text.strip()
    stripped = _strip_code_fence(stripped)
    assistant_marker = "<|assistant|>"
    if assistant_marker in stripped:
        stripped = stripped.split(assistant_marker, maxsplit=1)[-1].strip()
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
) -> Callable[[SQLEvalCase], str]:
    torch, transformers, peft = _import_training_stack()
    tokenizer_like = _load_tokenizer_like(transformers, manifest.student.base_model)
    _ensure_pad_token(tokenizer_like)
    tokenizer = _inner_tokenizer(tokenizer_like)
    model = _load_trainable_model(transformers, manifest.student.base_model, torch_module=torch)
    if model_variant == "adapter":
        if not adapter_dir.exists():
            raise ValueError(f"adapter_dir does not exist: {adapter_dir}")
        model = peft.PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    if hasattr(model, "cuda") and torch.cuda.is_available():
        model = model.cuda()

    def predict(case: SQLEvalCase) -> str:
        messages = build_eval_messages(case)
        prompt = render_sql_sft_prompt([*messages, {"role": "assistant", "content": ""}])
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
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
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return extract_generated_sql(generated_text)

    return predict


def _evaluate_case(
    case: SQLEvalCase,
    *,
    model_variant: str,
    predict_sql: Callable[[SQLEvalCase], str],
) -> SQLCaseEvalRecord:
    predicted_sql = predict_sql(case)
    result = evaluate_sqlite_case(case, predicted_sql=predicted_sql)
    return SQLCaseEvalRecord(
        case_id=case.case_id,
        task_id=case.task_id,
        model_variant=model_variant,
        predicted_sql=predicted_sql,
        passed=result.passed,
        prediction_error=result.prediction_error,
        gold_error=result.gold_error,
        predicted_rows=result.predicted_rows,
        gold_rows=result.gold_rows,
    )


def _eval_result_path(
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    *,
    eval_dataset: str | Path | None,
) -> Path:
    if eval_dataset is not None:
        dataset_stem = Path(eval_dataset).stem
        return _workspace_results_root() / manifest.experiment_id / f"{model_variant}__{dataset_stem}.json"
    if model_variant == "base":
        return manifest.resolve_workspace_path(manifest.eval_plan.baseline_results)
    if model_variant == "adapter":
        return manifest.resolve_workspace_path(manifest.eval_plan.post_train_results)
    raise ValueError(f"unsupported model_variant: {model_variant}")


def _workspace_results_root() -> Path:
    from sqlbench_lab.paths import WORKSPACE_ROOT

    return WORKSPACE_ROOT / "results" / "sql"


def _write_eval_summary(path: Path, summary: SQLEvalRunSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _maybe_log_mlflow_eval(
    *,
    manifest: SQLSFTExperimentManifest,
    manifest_path: Path,
    summary: SQLEvalRunSummary,
    result_path: Path,
    explicit: bool | None,
    tracking_uri: str | None,
    experiment_name: str | None,
) -> None:
    from sqlbench_lab.observability import log_sql_eval_run, mlflow_enabled

    if not mlflow_enabled(explicit):
        return
    log_sql_eval_run(
        manifest=manifest,
        manifest_path=manifest_path,
        summary=summary,
        result_path=result_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )


def _resolve_manifest_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    match = re.match(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
