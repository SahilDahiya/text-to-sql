"""Manifest-driven SQL model evaluation."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .eval_analysis import classify_sql_eval_failure, sql_eval_failure_observation
from .eval_types import (
    SQLCandidateEvalRecord,
    SQLCandidatePoolCaseRecord,
    SQLCandidatePoolEvalRunSummary,
    SQLCaseEvalRecord,
    SQLEvalRunSummary,
    SQLRepairAttemptRecord,
    SQLRepairEvalCaseRecord,
    SQLRepairEvalRunSummary,
)
from .evaluator import evaluate_sqlite_case
from .loaders import load_sql_eval_cases
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .models import SQLEvalCase
from .rendering import SQL_SYSTEM_PROMPT, build_eval_messages, build_repair_eval_messages
from .repair_collection import STRONG_REPAIR_FAILURE_TYPES
from .training import (
    _ensure_pad_token,
    _import_training_stack,
    _inner_tokenizer,
    _load_tokenizer_like,
    _load_trainable_model,
    render_sql_sft_prompt,
)

SQL_MODEL_VARIANTS = {"base", "adapter"}
SQLRepairPredictor = Callable[[SQLEvalCase, str, str], str]
SQLCandidatePoolPredictor = Callable[[SQLEvalCase], list[str]]


def run_sql_eval(
    manifest_path: str | Path,
    *,
    model_variant: str,
    eval_dataset: str | Path | None = None,
    max_new_tokens: int = 128,
    system_prompt: str | None = None,
    result_label: str | None = None,
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

    result_path = _eval_result_path(
        manifest,
        model_variant,
        eval_dataset=eval_dataset,
        result_label=result_label,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
    predict_sql = predictor or _build_hf_predictor(
        manifest=manifest,
        model_variant=model_variant,
        adapter_dir=adapter_dir,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
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


def run_sql_candidate_pool_eval(
    manifest_path: str | Path,
    *,
    model_variant: str,
    eval_dataset: str | Path | None = None,
    candidate_count: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    system_prompt: str | None = None,
    result_label: str | None = None,
    predictor: SQLCandidatePoolPredictor | None = None,
    log_mlflow: bool | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
) -> SQLCandidatePoolEvalRunSummary:
    """Run SQL eval with N generated candidates per case and non-gold selection."""

    if model_variant not in SQL_MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(SQL_MODEL_VARIANTS)}")
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be > 0 and <= 1")

    manifest = load_sql_sft_manifest(manifest_path)
    eval_dataset_path = str(eval_dataset) if eval_dataset is not None else manifest.eval_plan.smoke_dataset
    cases = load_sql_eval_cases(eval_dataset_path)
    if not cases:
        raise ValueError("SQL candidate-pool eval requires at least one eval case")

    result_path = _candidate_pool_eval_result_path(
        manifest,
        model_variant,
        eval_dataset_path=eval_dataset_path,
        result_label=result_label,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
    predict_candidates = predictor or _build_hf_candidate_pool_predictor(
        manifest=manifest,
        model_variant=model_variant,
        adapter_dir=adapter_dir,
        candidate_count=candidate_count,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )

    records = [
        _evaluate_candidate_pool_case(
            case,
            model_variant=model_variant,
            candidate_count=candidate_count,
            predict_candidates=predict_candidates,
        )
        for case in cases
    ]
    first_passed_count = sum(1 for record in records if record.first_candidate_passed)
    pass_at_n_count = sum(1 for record in records if record.any_candidate_passed)
    selected_passed_count = sum(1 for record in records if record.selected_candidate_passed)
    summary = SQLCandidatePoolEvalRunSummary(
        experiment_id=manifest.experiment_id,
        base_model=manifest.student.base_model,
        model_variant=model_variant,
        adapter_dir=str(adapter_dir) if model_variant == "adapter" else None,
        eval_dataset=eval_dataset_path,
        result_path=str(result_path),
        case_count=len(records),
        candidate_count=candidate_count,
        first_passed_count=first_passed_count,
        pass_at_n_count=pass_at_n_count,
        selected_passed_count=selected_passed_count,
        first_pass_rate=first_passed_count / len(records),
        pass_at_n_rate=pass_at_n_count / len(records),
        selected_pass_rate=selected_passed_count / len(records),
        selector="valid_nonempty_shortest",
        records=records,
    )
    _write_candidate_pool_eval_summary(result_path, summary)
    _maybe_log_mlflow_candidate_pool_eval(
        manifest=manifest,
        manifest_path=_resolve_manifest_path(manifest_path),
        summary=summary,
        result_path=result_path,
        explicit=log_mlflow,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )
    return summary


def run_sql_eval_with_repair(
    manifest_path: str | Path,
    *,
    model_variant: str,
    eval_dataset: str | Path | None = None,
    max_new_tokens: int = 128,
    max_repair_attempts: int = 1,
    repair_failure_types: set[str] | None = None,
    predictor: Callable[[SQLEvalCase], str] | None = None,
    repair_predictor: SQLRepairPredictor | None = None,
) -> SQLRepairEvalRunSummary:
    """Run SQL eval with execution-guided repair attempts after failed first pass."""

    if model_variant not in SQL_MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(SQL_MODEL_VARIANTS)}")
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be >= 0")

    manifest = load_sql_sft_manifest(manifest_path)
    eval_dataset_path = str(eval_dataset) if eval_dataset is not None else manifest.eval_plan.smoke_dataset
    cases = load_sql_eval_cases(eval_dataset_path)
    if not cases:
        raise ValueError("SQL eval requires at least one eval case")

    result_path = _repair_eval_result_path(manifest, model_variant, eval_dataset_path=eval_dataset_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
    predict_sql, predict_repair_sql = _resolve_repair_predictors(
        manifest=manifest,
        model_variant=model_variant,
        adapter_dir=adapter_dir,
        max_new_tokens=max_new_tokens,
        predictor=predictor,
        repair_predictor=repair_predictor,
    )
    allowed_failure_types = (
        set(STRONG_REPAIR_FAILURE_TYPES)
        if repair_failure_types is None
        else set(repair_failure_types)
    )

    records = [
        _evaluate_case_with_repair(
            case,
            model_variant=model_variant,
            predict_sql=predict_sql,
            repair_predict_sql=predict_repair_sql,
            allowed_failure_types=allowed_failure_types,
            max_repair_attempts=max_repair_attempts,
        )
        for case in cases
    ]
    first_passed_count = sum(1 for record in records if record.first_result.passed)
    final_passed_count = sum(1 for record in records if record.final_result.passed)
    repair_attempt_count = sum(len(record.repair_attempts) for record in records)
    repair_success_count = sum(
        1
        for record in records
        if not record.first_result.passed and record.final_result.passed
    )
    summary = SQLRepairEvalRunSummary(
        experiment_id=manifest.experiment_id,
        base_model=manifest.student.base_model,
        model_variant=model_variant,
        adapter_dir=str(adapter_dir) if model_variant == "adapter" else None,
        eval_dataset=eval_dataset_path,
        result_path=str(result_path),
        case_count=len(records),
        first_passed_count=first_passed_count,
        first_pass_rate=first_passed_count / len(records),
        final_passed_count=final_passed_count,
        final_pass_rate=final_passed_count / len(records),
        repair_attempt_count=repair_attempt_count,
        repair_success_count=repair_success_count,
        repair_failure_types=tuple(sorted(allowed_failure_types)),
        max_repair_attempts=max_repair_attempts,
        records=records,
    )
    _write_repair_eval_summary(result_path, summary)
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
    system_prompt: str | None = None,
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
                system_prompt=system_prompt or SQL_SYSTEM_PROMPT,
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
        if not adapter_dir.exists():
            raise ValueError(f"adapter_dir does not exist: {adapter_dir}")
        model = peft.PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    if hasattr(model, "cuda") and torch.cuda.is_available():
        model = model.cuda()

    def predict(messages: list[dict[str, str]]) -> str:
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


def _build_hf_candidate_pool_predictor(
    *,
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    adapter_dir: Path,
    candidate_count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str | None,
) -> SQLCandidatePoolPredictor:
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
        if not adapter_dir.exists():
            raise ValueError(f"adapter_dir does not exist: {adapter_dir}")
        model = peft.PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    if hasattr(model, "cuda") and torch.cuda.is_available():
        model = model.cuda()

    def predict(case: SQLEvalCase) -> list[str]:
        messages = build_eval_messages(
            case,
            prompt_style=manifest.prompt.style,
            system_prompt=system_prompt or SQL_SYSTEM_PROMPT,
        )
        prompt = render_sql_sft_prompt([*messages, {"role": "assistant", "content": ""}])
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        input_length = int(encoded["input_ids"].shape[-1])
        candidates: list[str] = []
        with torch.no_grad():
            for candidate_index in range(candidate_count):
                candidate_generation_kwargs = dict(generation_kwargs)
                if candidate_index == 0 or temperature == 0:
                    candidate_generation_kwargs["do_sample"] = False
                else:
                    candidate_generation_kwargs["do_sample"] = True
                    candidate_generation_kwargs["temperature"] = temperature
                    candidate_generation_kwargs["top_p"] = top_p
                output_ids = model.generate(**encoded, **candidate_generation_kwargs)
                candidates.append(
                    extract_generated_sql(tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True))
                )
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return candidates

    return predict


def _resolve_repair_predictors(
    *,
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    adapter_dir: Path,
    max_new_tokens: int,
    predictor: Callable[[SQLEvalCase], str] | None,
    repair_predictor: SQLRepairPredictor | None,
) -> tuple[Callable[[SQLEvalCase], str], SQLRepairPredictor]:
    if predictor is not None and repair_predictor is not None:
        return predictor, repair_predictor

    predict_messages = _build_hf_message_predictor(
        manifest=manifest,
        model_variant=model_variant,
        adapter_dir=adapter_dir,
        max_new_tokens=max_new_tokens,
    )

    resolved_predictor = predictor or (
        lambda case: predict_messages(build_eval_messages(case, prompt_style=manifest.prompt.style))
    )
    resolved_repair_predictor = repair_predictor or (
        lambda case, previous_sql, observation: predict_messages(
            build_repair_eval_messages(
                case,
                previous_sql=previous_sql,
                execution_observation=observation,
                prompt_style=manifest.prompt.style,
            )
        )
    )
    return resolved_predictor, resolved_repair_predictor


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


def _evaluate_candidate_pool_case(
    case: SQLEvalCase,
    *,
    model_variant: str,
    candidate_count: int,
    predict_candidates: SQLCandidatePoolPredictor,
) -> SQLCandidatePoolCaseRecord:
    predicted_sqls = predict_candidates(case)
    if len(predicted_sqls) != candidate_count:
        raise ValueError(
            f"candidate predictor returned {len(predicted_sqls)} candidates; expected {candidate_count}"
        )

    candidate_records: list[SQLCandidateEvalRecord] = []
    gold_rows: list[tuple[object, ...]] = []
    gold_error: str | None = None
    for index, predicted_sql in enumerate(predicted_sqls, start=1):
        result = evaluate_sqlite_case(case, predicted_sql=predicted_sql)
        if index == 1:
            gold_rows = result.gold_rows
            gold_error = result.gold_error
        candidate_records.append(
            SQLCandidateEvalRecord(
                candidate_index=index,
                predicted_sql=predicted_sql,
                passed=result.passed,
                prediction_error=result.prediction_error,
                predicted_rows=result.predicted_rows,
                result_signature=_result_signature(result.predicted_rows, result.prediction_error),
            )
        )
    selected_candidate = _select_candidate(candidate_records)
    return SQLCandidatePoolCaseRecord(
        case_id=case.case_id,
        task_id=case.task_id,
        model_variant=model_variant,
        selected_candidate_index=selected_candidate.candidate_index if selected_candidate else None,
        first_candidate_passed=candidate_records[0].passed,
        any_candidate_passed=any(candidate.passed for candidate in candidate_records),
        selected_candidate_passed=selected_candidate.passed if selected_candidate else False,
        gold_error=gold_error,
        gold_rows=gold_rows,
        candidates=candidate_records,
    )


def _select_candidate(candidates: list[SQLCandidateEvalRecord]) -> SQLCandidateEvalRecord | None:
    valid_candidates = [candidate for candidate in candidates if candidate.prediction_error is None]
    if not valid_candidates:
        return candidates[0] if candidates else None
    non_empty_candidates = [candidate for candidate in valid_candidates if candidate.predicted_rows]
    selection_pool = non_empty_candidates or valid_candidates
    return min(selection_pool, key=lambda candidate: (len(candidate.predicted_sql), candidate.candidate_index))


def _result_signature(rows: list[tuple[object, ...]], error: str | None) -> str:
    payload = json.dumps(
        {"error": error, "rows": rows},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _evaluate_case_with_repair(
    case: SQLEvalCase,
    *,
    model_variant: str,
    predict_sql: Callable[[SQLEvalCase], str],
    repair_predict_sql: SQLRepairPredictor,
    allowed_failure_types: set[str],
    max_repair_attempts: int,
) -> SQLRepairEvalCaseRecord:
    first_sql = predict_sql(case)
    first_result = _evaluate_predicted_sql(case, model_variant=model_variant, predicted_sql=first_sql)
    current_result = first_result
    repair_attempts: list[SQLRepairAttemptRecord] = []
    for attempt_index in range(1, max_repair_attempts + 1):
        if current_result.passed:
            break
        current_failure_type = _failure_type_for_record(current_result)
        if current_failure_type not in allowed_failure_types:
            break
        observation = sql_eval_failure_observation(
            _record_payload(current_result),
            failure_type=current_failure_type,
        )
        repaired_sql = repair_predict_sql(case, current_result.predicted_sql, observation)
        repaired_result = _evaluate_predicted_sql(
            case,
            model_variant=model_variant,
            predicted_sql=repaired_sql,
        )
        repair_attempts.append(
            SQLRepairAttemptRecord(
                attempt_index=attempt_index,
                input_sql=current_result.predicted_sql,
                input_failure_type=current_failure_type,
                observation=observation,
                repaired_sql=repaired_sql,
                result=repaired_result,
            )
        )
        current_result = repaired_result
    return SQLRepairEvalCaseRecord(
        case_id=case.case_id,
        task_id=case.task_id,
        model_variant=model_variant,
        first_result=first_result,
        final_result=current_result,
        repair_attempts=repair_attempts,
    )


def _evaluate_predicted_sql(
    case: SQLEvalCase,
    *,
    model_variant: str,
    predicted_sql: str,
) -> SQLCaseEvalRecord:
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


def _failure_type_for_record(record: SQLCaseEvalRecord) -> str:
    return classify_sql_eval_failure(_record_payload(record))


def _record_payload(record: SQLCaseEvalRecord) -> dict[str, object]:
    return asdict(record)


def _eval_result_path(
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    *,
    eval_dataset: str | Path | None,
    result_label: str | None = None,
) -> Path:
    if eval_dataset is not None:
        dataset_stem = Path(eval_dataset).stem
        label = f"__{_safe_result_label(result_label)}" if result_label else ""
        return _workspace_results_root() / manifest.experiment_id / f"{model_variant}__{dataset_stem}{label}.json"
    if model_variant == "base":
        return manifest.resolve_workspace_path(manifest.eval_plan.baseline_results)
    if model_variant == "adapter":
        return manifest.resolve_workspace_path(manifest.eval_plan.post_train_results)
    raise ValueError(f"unsupported model_variant: {model_variant}")


def _repair_eval_result_path(
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    *,
    eval_dataset_path: str | Path,
) -> Path:
    dataset_stem = Path(eval_dataset_path).stem
    return _workspace_results_root() / manifest.experiment_id / f"repair__{model_variant}__{dataset_stem}.json"


def _candidate_pool_eval_result_path(
    manifest: SQLSFTExperimentManifest,
    model_variant: str,
    *,
    eval_dataset_path: str | Path,
    result_label: str | None,
) -> Path:
    dataset_stem = Path(eval_dataset_path).stem
    label = f"__{_safe_result_label(result_label)}" if result_label else ""
    return _workspace_results_root() / manifest.experiment_id / f"candidates__{model_variant}__{dataset_stem}{label}.json"


def _workspace_results_root() -> Path:
    from sqlbench_lab.paths import WORKSPACE_ROOT

    return WORKSPACE_ROOT / "results" / "sql"


def _safe_result_label(value: str) -> str:
    label = value.strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    if not label:
        raise ValueError("result_label must not be empty")
    return label


def _write_eval_summary(path: Path, summary: SQLEvalRunSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_candidate_pool_eval_summary(path: Path, summary: SQLCandidatePoolEvalRunSummary) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_repair_eval_summary(path: Path, summary: SQLRepairEvalRunSummary) -> None:
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


def _maybe_log_mlflow_candidate_pool_eval(
    *,
    manifest: SQLSFTExperimentManifest,
    manifest_path: Path,
    summary: SQLCandidatePoolEvalRunSummary,
    result_path: Path,
    explicit: bool | None,
    tracking_uri: str | None,
    experiment_name: str | None,
) -> None:
    from sqlbench_lab.observability import log_sql_candidate_pool_eval_run, mlflow_enabled

    if not mlflow_enabled(explicit):
        return
    log_sql_candidate_pool_eval_run(
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
