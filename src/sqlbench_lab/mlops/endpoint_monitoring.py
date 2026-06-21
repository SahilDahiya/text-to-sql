"""Dev endpoint monitoring contract for SQL adapter serving tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlbench_lab.mlops.dev_endpoint import SQLAdapterDevEndpointPlan
from sqlbench_lab.mlops.run_contract import DEV_ENVIRONMENT, ENDPOINT_EVAL_GATE, SQLAdapterRunContract

DEV_ENDPOINT_MONITORING_SCHEMA_VERSION = "sql_adapter_dev_endpoint_monitoring:v1"


@dataclass(frozen=True)
class SQLAdapterDevEndpointMonitoringRecord:
    schema_version: str
    environment: str
    endpoint_id: str
    openai_model: str
    adapter_name: str
    request_count: int
    success_count: int
    failure_count: int
    timeout_count: int
    concurrency: int | None
    requests_per_second: float | None
    p50_latency_seconds: float | None
    p95_latency_seconds: float | None
    p99_latency_seconds: float | None
    max_latency_seconds: float | None
    generated_char_count_min: int | None
    generated_char_count_p50: int | None
    generated_char_count_p95: int | None
    generated_char_count_max: int | None
    generated_char_count_mean: float | None
    endpoint_eval_passed_count: int | None
    endpoint_eval_case_count: int | None
    endpoint_eval_pass_rate: float | None
    empty_sql_count: int | None
    syntax_failure_count: int
    schema_failure_count: int
    runtime_failure_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dev_endpoint_monitoring_record(
    contract: SQLAdapterRunContract,
    endpoint_plan: SQLAdapterDevEndpointPlan,
    *,
    timeout_count: int | None = None,
    p99_latency_seconds: float | None = None,
    empty_sql_count: int | None = None,
) -> SQLAdapterDevEndpointMonitoringRecord:
    """Summarize endpoint eval and load-test evidence for dev monitoring."""

    if contract.environment.environment != DEV_ENVIRONMENT:
        raise ValueError(f"dev endpoint monitoring only supports environment={DEV_ENVIRONMENT!r}")
    endpoint_gates = [gate for gate in contract.eval_gates if gate.gate_type == ENDPOINT_EVAL_GATE]
    endpoint_gate = endpoint_gates[-1] if endpoint_gates else None
    load = contract.load_tests[-1] if contract.load_tests else None
    failure_counts = endpoint_gate.failure_counts if endpoint_gate is not None else {}
    resolved_timeout_count = load.timeout_count if timeout_count is None and load is not None else (timeout_count or 0)
    return SQLAdapterDevEndpointMonitoringRecord(
        schema_version=DEV_ENDPOINT_MONITORING_SCHEMA_VERSION,
        environment=DEV_ENVIRONMENT,
        endpoint_id=endpoint_plan.endpoint_id,
        openai_model=endpoint_plan.openai_model,
        adapter_name=contract.inputs.adapter_name,
        request_count=load.request_count if load is not None else 0,
        success_count=load.success_count if load is not None else 0,
        failure_count=load.failure_count if load is not None else 0,
        timeout_count=_non_negative_int(resolved_timeout_count, "timeout_count"),
        concurrency=load.concurrency if load is not None else None,
        requests_per_second=load.requests_per_second if load is not None else None,
        p50_latency_seconds=load.p50_latency_seconds if load is not None else None,
        p95_latency_seconds=load.p95_latency_seconds if load is not None else None,
        p99_latency_seconds=p99_latency_seconds,
        max_latency_seconds=load.max_latency_seconds if load is not None else None,
        generated_char_count_min=load.generated_char_count_min if load is not None else None,
        generated_char_count_p50=load.generated_char_count_p50 if load is not None else None,
        generated_char_count_p95=load.generated_char_count_p95 if load is not None else None,
        generated_char_count_max=load.generated_char_count_max if load is not None else None,
        generated_char_count_mean=load.generated_char_count_mean if load is not None else None,
        endpoint_eval_passed_count=endpoint_gate.passed_count if endpoint_gate is not None else None,
        endpoint_eval_case_count=endpoint_gate.case_count if endpoint_gate is not None else None,
        endpoint_eval_pass_rate=endpoint_gate.pass_rate if endpoint_gate is not None else None,
        empty_sql_count=empty_sql_count,
        syntax_failure_count=int(failure_counts.get("prediction_syntax_error", 0)),
        schema_failure_count=int(failure_counts.get("prediction_schema_error", 0)),
        runtime_failure_count=int(failure_counts.get("prediction_runtime_error", 0)),
    )


def _non_negative_int(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
