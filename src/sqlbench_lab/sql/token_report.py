"""Token-length reporting for SQL train and eval prompts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .loaders import load_sql_eval_cases, load_sql_train_examples
from .manifest import load_sql_sft_manifest
from .rendering import SQL_SYSTEM_PROMPT, build_eval_messages, build_train_messages
from .training import _ensure_pad_token, _inner_tokenizer, _load_tokenizer_like, render_sql_sft_prompt


@dataclass(frozen=True)
class SQLTokenLengthDatasetReport:
    dataset_path: str
    artifact: str
    row_count: int
    min_tokens: int
    p50_tokens: int
    p90_tokens: int
    p95_tokens: int
    max_tokens: int
    mean_tokens: float


@dataclass(frozen=True)
class SQLTokenLengthReport:
    experiment_id: str
    base_model: str
    result_path: str | None
    dataset_reports: list[SQLTokenLengthDatasetReport]


def report_sql_prompt_lengths(
    manifest_path: str | Path,
    *,
    eval_dataset: str | Path | None = None,
    include_train: bool = True,
    include_eval: bool = True,
    output_path: str | Path | None = None,
    tokenizer_like: Any | None = None,
) -> SQLTokenLengthReport:
    """Measure rendered SQL prompt token lengths for manifest train/eval inputs."""

    if not include_train and not include_eval:
        raise ValueError("at least one of include_train/include_eval must be true")

    manifest = load_sql_sft_manifest(manifest_path)
    tokenizer = _resolve_tokenizer(manifest.student.base_model, tokenizer_like=tokenizer_like)
    reports: list[SQLTokenLengthDatasetReport] = []
    if include_train:
        for dataset_path in manifest.train_inputs.train_datasets:
            reports.append(
                _report_train_dataset(
                    dataset_path,
                    prompt_style=manifest.prompt.style,
                    tokenizer=tokenizer,
                )
            )
    if include_eval:
        resolved_eval_dataset = str(eval_dataset) if eval_dataset is not None else manifest.eval_plan.smoke_dataset
        reports.append(
            _report_eval_dataset(
                resolved_eval_dataset,
                prompt_style=manifest.prompt.style,
                tokenizer=tokenizer,
            )
        )

    resolved_output = _resolve_output_path(output_path)
    report = SQLTokenLengthReport(
        experiment_id=manifest.experiment_id,
        base_model=manifest.student.base_model,
        result_path=str(resolved_output) if resolved_output is not None else None,
        dataset_reports=reports,
    )
    if resolved_output is not None:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def _resolve_tokenizer(base_model: str, *, tokenizer_like: Any | None) -> Any:
    if tokenizer_like is None:
        import transformers

        tokenizer_like = _load_tokenizer_like(transformers, base_model)
        _ensure_pad_token(tokenizer_like)
    return _inner_tokenizer(tokenizer_like)


def _report_train_dataset(
    dataset_path: str,
    *,
    prompt_style: str,
    tokenizer: Any,
) -> SQLTokenLengthDatasetReport:
    rows = load_sql_train_examples(dataset_path)
    lengths = [
        _token_count(
            tokenizer,
            render_sql_sft_prompt(build_train_messages(row, prompt_style=prompt_style)),
        )
        for row in rows
    ]
    return _dataset_report(dataset_path=dataset_path, artifact="train", lengths=lengths)


def _report_eval_dataset(
    dataset_path: str,
    *,
    prompt_style: str,
    tokenizer: Any,
) -> SQLTokenLengthDatasetReport:
    cases = load_sql_eval_cases(dataset_path)
    lengths = [
        _token_count(
            tokenizer,
            render_sql_sft_prompt(
                [
                    *build_eval_messages(
                        case,
                        prompt_style=prompt_style,
                        system_prompt=SQL_SYSTEM_PROMPT,
                    ),
                    {"role": "assistant", "content": ""},
                ]
            ),
        )
        for case in cases
    ]
    return _dataset_report(dataset_path=dataset_path, artifact="eval", lengths=lengths)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _dataset_report(
    *,
    dataset_path: str,
    artifact: str,
    lengths: list[int],
) -> SQLTokenLengthDatasetReport:
    if not lengths:
        raise ValueError(f"token report requires at least one row: {dataset_path}")
    sorted_lengths = sorted(lengths)
    return SQLTokenLengthDatasetReport(
        dataset_path=dataset_path,
        artifact=artifact,
        row_count=len(sorted_lengths),
        min_tokens=sorted_lengths[0],
        p50_tokens=_percentile(sorted_lengths, 50),
        p90_tokens=_percentile(sorted_lengths, 90),
        p95_tokens=_percentile(sorted_lengths, 95),
        max_tokens=sorted_lengths[-1],
        mean_tokens=sum(sorted_lengths) / len(sorted_lengths),
    )


def _percentile(sorted_values: list[int], percentile: int) -> int:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = math.ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(rank, len(sorted_values) - 1))]


def _resolve_output_path(output_path: str | Path | None) -> Path | None:
    if output_path is None:
        return None
    path = Path(output_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path
