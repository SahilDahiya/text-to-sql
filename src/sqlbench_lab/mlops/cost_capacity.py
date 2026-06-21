"""Dev cost and capacity monitoring contract for SQL adapter runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from sqlbench_lab.mlops.dev_endpoint import SQLAdapterDevEndpointPlan
from sqlbench_lab.mlops.run_contract import DEV_ENVIRONMENT, SQLAdapterRunContract
from sqlbench_lab.mlops.vertex_job import SQLAdapterVertexTrainingJobPlan

DEV_COST_CAPACITY_SCHEMA_VERSION = "sql_adapter_dev_cost_capacity:v1"


@dataclass(frozen=True)
class SQLAdapterDevCostCapacityRecord:
    schema_version: str
    environment: str
    experiment_id: str
    training_machine_type: str | None
    training_accelerator_type: str | None
    training_accelerator_count: int | None
    training_runtime_hours: float | None
    training_hourly_cost_usd: float | None
    training_estimated_cost_usd: float | None
    endpoint_machine_type: str | None
    endpoint_accelerator_type: str | None
    endpoint_accelerator_count: int | None
    endpoint_min_replica_count: int | None
    endpoint_max_replica_count: int | None
    endpoint_uptime_hours: float | None
    endpoint_hourly_cost_usd: float | None
    endpoint_estimated_cost_usd: float | None
    request_count: int
    peak_concurrency: int | None
    requests_per_second: float | None
    total_estimated_cost_usd: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dev_cost_capacity_record(
    contract: SQLAdapterRunContract,
    *,
    vertex_plan: SQLAdapterVertexTrainingJobPlan | None = None,
    endpoint_plan: SQLAdapterDevEndpointPlan | None = None,
    endpoint_uptime_hours: float | None = None,
    training_hourly_cost_usd: float | None = None,
    endpoint_hourly_cost_usd: float | None = None,
) -> SQLAdapterDevCostCapacityRecord:
    """Build dev cost/capacity signals with caller-supplied cost rates."""

    if contract.environment.environment != DEV_ENVIRONMENT:
        raise ValueError(f"dev cost/capacity only supports environment={DEV_ENVIRONMENT!r}")
    training_hours = (
        contract.train.train_runtime_seconds / 3600.0
        if contract.train is not None and contract.train.train_runtime_seconds is not None
        else None
    )
    load = contract.load_tests[-1] if contract.load_tests else None
    training_cost = _cost(training_hours, training_hourly_cost_usd, vertex_plan.machine.accelerator_count if vertex_plan else 1)
    endpoint_cost = _cost(
        endpoint_uptime_hours,
        endpoint_hourly_cost_usd,
        endpoint_plan.max_replica_count if endpoint_plan is not None else 1,
    )
    total_cost = _sum_costs(training_cost, endpoint_cost)
    return SQLAdapterDevCostCapacityRecord(
        schema_version=DEV_COST_CAPACITY_SCHEMA_VERSION,
        environment=DEV_ENVIRONMENT,
        experiment_id=contract.inputs.experiment_id,
        training_machine_type=vertex_plan.machine.machine_type if vertex_plan is not None else None,
        training_accelerator_type=vertex_plan.machine.accelerator_type if vertex_plan is not None else None,
        training_accelerator_count=vertex_plan.machine.accelerator_count if vertex_plan is not None else None,
        training_runtime_hours=training_hours,
        training_hourly_cost_usd=training_hourly_cost_usd,
        training_estimated_cost_usd=training_cost,
        endpoint_machine_type=endpoint_plan.machine_type if endpoint_plan is not None else None,
        endpoint_accelerator_type=endpoint_plan.accelerator_type if endpoint_plan is not None else None,
        endpoint_accelerator_count=endpoint_plan.accelerator_count if endpoint_plan is not None else None,
        endpoint_min_replica_count=endpoint_plan.min_replica_count if endpoint_plan is not None else None,
        endpoint_max_replica_count=endpoint_plan.max_replica_count if endpoint_plan is not None else None,
        endpoint_uptime_hours=endpoint_uptime_hours,
        endpoint_hourly_cost_usd=endpoint_hourly_cost_usd,
        endpoint_estimated_cost_usd=endpoint_cost,
        request_count=load.request_count if load is not None else 0,
        peak_concurrency=load.concurrency if load is not None else None,
        requests_per_second=load.requests_per_second if load is not None else None,
        total_estimated_cost_usd=total_cost,
    )


def _cost(hours: float | None, hourly_cost: float | None, multiplier: int) -> float | None:
    if hours is None or hourly_cost is None:
        return None
    if hours < 0 or hourly_cost < 0:
        raise ValueError("cost inputs must be non-negative")
    amount = Decimal(str(hours)) * Decimal(str(hourly_cost)) * Decimal(str(multiplier))
    return float(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _sum_costs(*costs: float | None) -> float | None:
    present = [Decimal(str(cost)) for cost in costs if cost is not None]
    if not present:
        return None
    return float(sum(present).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
