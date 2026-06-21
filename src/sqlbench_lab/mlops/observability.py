"""Dev observability contract for SQL adapter MLOps runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlbench_lab.mlops.gcs_sync import SQLAdapterGCSSyncPlan
from sqlbench_lab.mlops.promotion_registry import SQLAdapterDevPromotionRegistryPlan
from sqlbench_lab.mlops.run_contract import DEV_ENVIRONMENT, SQLAdapterPromotionDecision, SQLAdapterRunContract

DEV_OBSERVABILITY_SCHEMA_VERSION = "sql_adapter_dev_observability:v1"


@dataclass(frozen=True)
class SQLAdapterEvalObservation:
    label: str
    gate_type: str
    case_count: int
    passed_count: int
    pass_rate: float
    failed_count: int
    failure_counts: dict[str, int]


@dataclass(frozen=True)
class SQLAdapterDevObservabilityRecord:
    schema_version: str
    environment: str
    experiment_id: str
    adapter_name: str
    base_model: str
    run_id: str
    git_sha: str | None
    container_image_uri: str | None
    train_row_count: int | None
    train_runtime_seconds: float | None
    evals: tuple[SQLAdapterEvalObservation, ...]
    decision: str
    failed_gates: tuple[str, ...]
    passed_gates: tuple[str, ...]
    gcs_prefix: str
    registry_current_pointer_uri: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dev_observability_record(
    contract: SQLAdapterRunContract,
    decision: SQLAdapterPromotionDecision,
    gcs_plan: SQLAdapterGCSSyncPlan,
    *,
    git_sha: str | None = None,
    container_image_uri: str | None = None,
    registry_plan: SQLAdapterDevPromotionRegistryPlan | None = None,
) -> SQLAdapterDevObservabilityRecord:
    """Build a compact run record for dev inspection and dashboards."""

    if contract.environment.environment != DEV_ENVIRONMENT:
        raise ValueError(f"dev observability only supports environment={DEV_ENVIRONMENT!r}")
    evals = tuple(
        SQLAdapterEvalObservation(
            label=gate.label,
            gate_type=gate.gate_type,
            case_count=gate.case_count,
            passed_count=gate.passed_count,
            pass_rate=gate.pass_rate,
            failed_count=gate.case_count - gate.passed_count,
            failure_counts=gate.failure_counts,
        )
        for gate in contract.eval_gates
    )
    return SQLAdapterDevObservabilityRecord(
        schema_version=DEV_OBSERVABILITY_SCHEMA_VERSION,
        environment=DEV_ENVIRONMENT,
        experiment_id=contract.inputs.experiment_id,
        adapter_name=contract.inputs.adapter_name,
        base_model=contract.inputs.base_model,
        run_id=gcs_plan.run_id,
        git_sha=git_sha,
        container_image_uri=container_image_uri,
        train_row_count=contract.train.train_row_count if contract.train is not None else None,
        train_runtime_seconds=contract.train.train_runtime_seconds if contract.train is not None else None,
        evals=evals,
        decision=decision.decision,
        failed_gates=decision.failed_gates,
        passed_gates=decision.passed_gates,
        gcs_prefix=gcs_plan.prefix,
        registry_current_pointer_uri=registry_plan.current_pointer_uri if registry_plan is not None else None,
    )
