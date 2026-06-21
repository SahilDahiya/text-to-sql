"""SQL model serving helpers for OpenAI-compatible runtimes."""

from __future__ import annotations

import json
import shlex
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Any

from .eval_runner import extract_generated_sql
from .loaders import load_sql_eval_cases
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .models import SQLEvalCase
from .rendering import SQL_SYSTEM_PROMPT, build_eval_messages
from .training import render_sql_sft_prompt


@dataclass(frozen=True)
class VLLMServeCommand:
    command: tuple[str, ...]

    @property
    def shell_command(self) -> str:
        return shlex.join(self.command)


@dataclass(frozen=True)
class OpenAILoadTestRequestRecord:
    request_index: int
    case_id: str
    success: bool
    latency_seconds: float
    generated_char_count: int
    error: str | None


@dataclass(frozen=True)
class OpenAILoadTestSummary:
    manifest_path: str
    experiment_id: str
    model_variant: str
    eval_dataset: str
    openai_base_url: str
    openai_model: str
    request_count: int
    concurrency: int
    max_new_tokens: int
    success_count: int
    failure_count: int
    timeout_count: int
    min_latency_seconds: float | None
    p50_latency_seconds: float | None
    p95_latency_seconds: float | None
    max_latency_seconds: float | None
    generated_char_count_min: int | None
    generated_char_count_p50: int | None
    generated_char_count_p95: int | None
    generated_char_count_max: int | None
    generated_char_count_mean: float | None
    requests_per_second: float
    result_path: str | None
    records: tuple[OpenAILoadTestRequestRecord, ...]


OpenAICompletionTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class OpenAICompletionClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_new_tokens: int,
        timeout_seconds: float,
        api_key: str | None = None,
        temperature: float = 0.0,
        transport: OpenAICompletionTransport | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = _non_empty(model, "model")
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.timeout_seconds = _positive_float(timeout_seconds, "timeout_seconds")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.temperature = temperature
        self.api_key = api_key
        self._transport = transport or _post_json

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        response = self._transport(
            f"{self.base_url}/v1/completions",
            _headers(self.api_key),
            payload,
            self.timeout_seconds,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI completion response missing non-empty choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("OpenAI completion choice must be an object")
        text = first_choice.get("text")
        if not isinstance(text, str):
            raise ValueError("OpenAI completion choice missing text")
        return text


def build_vllm_serve_command(
    manifest_path: str | Path,
    *,
    model_variant: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_model_len: int = 1536,
    gpu_memory_utilization: float | None = None,
    enforce_eager: bool = False,
    flashinfer_autotune: bool | None = None,
    attention_backend: str | None = None,
    served_model_name: str | None = None,
    lora_name: str | None = None,
    language_model_only: bool = True,
) -> VLLMServeCommand:
    if model_variant not in {"base", "adapter"}:
        raise ValueError("model_variant must be base or adapter")
    manifest = load_sql_sft_manifest(manifest_path)
    _positive_int(port, "port")
    _positive_int(max_model_len, "max_model_len")
    command = [
        "vllm",
        "serve",
        manifest.student.base_model,
        "--host",
        _non_empty(host, "host"),
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
    ]
    if gpu_memory_utilization is not None:
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be > 0 and <= 1")
        command.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if enforce_eager:
        command.append("--enforce-eager")
    if flashinfer_autotune is not None:
        command.append("--enable-flashinfer-autotune" if flashinfer_autotune else "--no-enable-flashinfer-autotune")
    if attention_backend is not None:
        command.extend(["--attention-backend", _non_empty(attention_backend, "attention_backend")])
    if language_model_only:
        command.append("--language-model-only")
    if served_model_name:
        command.extend(["--served-model-name", served_model_name])
    if model_variant == "adapter":
        adapter_dir = manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)
        if not adapter_dir.exists():
            raise ValueError(f"adapter_dir does not exist: {adapter_dir}")
        module_name = lora_name or manifest.student.adapter_name
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(manifest.lora.r),
                "--lora-modules",
                f"{module_name}={adapter_dir}",
            ]
        )
    return VLLMServeCommand(tuple(command))


def build_openai_completion_predictor(
    manifest_path: str | Path,
    *,
    model_variant: str,
    base_url: str,
    model: str,
    max_new_tokens: int = 128,
    timeout_seconds: float = 60.0,
    api_key: str | None = None,
    system_prompt: str | None = None,
    transport: OpenAICompletionTransport | None = None,
) -> Callable[[SQLEvalCase], str]:
    if model_variant not in {"base", "adapter"}:
        raise ValueError("model_variant must be base or adapter")
    manifest = load_sql_sft_manifest(manifest_path)
    client = OpenAICompletionClient(
        base_url=base_url,
        model=model,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        temperature=0.0,
        transport=transport,
    )

    def predict(case: SQLEvalCase) -> str:
        messages = build_eval_messages(
            case,
            prompt_style=manifest.prompt.style,
            system_prompt=system_prompt or SQL_SYSTEM_PROMPT,
        )
        prompt = render_sql_sft_prompt([*messages, {"role": "assistant", "content": ""}])
        return extract_generated_sql(client.complete(prompt))

    return predict


def run_openai_completion_load_test(
    manifest_path: str | Path,
    *,
    model_variant: str,
    eval_dataset: str | Path | None,
    base_url: str,
    model: str,
    request_count: int,
    concurrency: int,
    max_new_tokens: int = 128,
    timeout_seconds: float = 60.0,
    api_key: str | None = None,
    system_prompt: str | None = None,
    output_path: str | Path | None = None,
    transport: OpenAICompletionTransport | None = None,
) -> OpenAILoadTestSummary:
    if model_variant not in {"base", "adapter"}:
        raise ValueError("model_variant must be base or adapter")
    _positive_int(request_count, "request_count")
    _positive_int(concurrency, "concurrency")
    manifest = load_sql_sft_manifest(manifest_path)
    eval_dataset_path = str(eval_dataset) if eval_dataset is not None else manifest.eval_plan.smoke_dataset
    cases = load_sql_eval_cases(eval_dataset_path)
    if not cases:
        raise ValueError("OpenAI load test requires at least one eval case")
    client = OpenAICompletionClient(
        base_url=base_url,
        model=model,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        temperature=0.0,
        transport=transport,
    )
    prompts = [
        _render_case_prompt(
            manifest,
            cases[index % len(cases)],
            system_prompt=system_prompt,
        )
        for index in range(request_count)
    ]
    started_at = time.perf_counter()
    records: list[OpenAILoadTestRequestRecord] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_run_load_request, index, cases[index % len(cases)], prompts[index], client): index
            for index in range(request_count)
        }
        for future in as_completed(futures):
            records.append(future.result())
    elapsed = time.perf_counter() - started_at
    records.sort(key=lambda record: record.request_index)
    latencies = [record.latency_seconds for record in records if record.success]
    generated_char_counts = [record.generated_char_count for record in records if record.success]
    success_count = sum(1 for record in records if record.success)
    timeout_count = sum(
        1
        for record in records
        if not record.success and record.error is not None and "timeout" in record.error.lower()
    )
    resolved_output_path = Path(output_path) if output_path is not None else None
    summary = OpenAILoadTestSummary(
        manifest_path=str(manifest_path),
        experiment_id=manifest.experiment_id,
        model_variant=model_variant,
        eval_dataset=eval_dataset_path,
        openai_base_url=_normalize_base_url(base_url),
        openai_model=model,
        request_count=request_count,
        concurrency=concurrency,
        max_new_tokens=max_new_tokens,
        success_count=success_count,
        failure_count=request_count - success_count,
        timeout_count=timeout_count,
        min_latency_seconds=min(latencies) if latencies else None,
        p50_latency_seconds=_percentile(latencies, 0.50),
        p95_latency_seconds=_percentile(latencies, 0.95),
        max_latency_seconds=max(latencies) if latencies else None,
        generated_char_count_min=min(generated_char_counts) if generated_char_counts else None,
        generated_char_count_p50=_percentile_int(generated_char_counts, 0.50),
        generated_char_count_p95=_percentile_int(generated_char_counts, 0.95),
        generated_char_count_max=max(generated_char_counts) if generated_char_counts else None,
        generated_char_count_mean=(
            sum(generated_char_counts) / len(generated_char_counts) if generated_char_counts else None
        ),
        requests_per_second=request_count / elapsed if elapsed > 0 else 0.0,
        result_path=str(resolved_output_path) if resolved_output_path is not None else None,
        records=tuple(records),
    )
    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(
            json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _render_case_prompt(
    manifest: SQLSFTExperimentManifest,
    case: SQLEvalCase,
    *,
    system_prompt: str | None,
) -> str:
    messages = build_eval_messages(
        case,
        prompt_style=manifest.prompt.style,
        system_prompt=system_prompt or SQL_SYSTEM_PROMPT,
    )
    return render_sql_sft_prompt([*messages, {"role": "assistant", "content": ""}])


def _run_load_request(
    request_index: int,
    case: SQLEvalCase,
    prompt: str,
    client: OpenAICompletionClient,
) -> OpenAILoadTestRequestRecord:
    started_at = time.perf_counter()
    try:
        text = client.complete(prompt)
    except Exception as exc:  # pragma: no cover - exception message is the testable payload.
        latency = time.perf_counter() - started_at
        return OpenAILoadTestRequestRecord(
            request_index=request_index,
            case_id=case.case_id,
            success=False,
            latency_seconds=latency,
            generated_char_count=0,
            error=str(exc),
        )
    latency = time.perf_counter() - started_at
    return OpenAILoadTestRequestRecord(
        request_index=request_index,
        case_id=case.case_id,
        success=True,
        latency_seconds=latency,
        generated_char_count=len(text),
        error=None,
    )


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI completion request failed status={exc.code}: {body}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI completion response must be a JSON object")
    return parsed


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _normalize_base_url(value: str) -> str:
    normalized = _non_empty(value, "base_url").rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("base_url must start with http:// or https://")
    return normalized


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


def _positive_int(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _positive_float(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]


def _percentile_int(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]
