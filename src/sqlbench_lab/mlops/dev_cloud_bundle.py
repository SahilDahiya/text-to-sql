"""Combined dev cloud artifact bundle for SQL adapter MLOps runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.mlops.cost_capacity import SQLAdapterDevCostCapacityRecord, build_dev_cost_capacity_record
from sqlbench_lab.mlops.dev_endpoint import SQLAdapterDevEndpointPlan, build_dev_gcp_vllm_endpoint_plan
from sqlbench_lab.mlops.endpoint_monitoring import (
    SQLAdapterDevEndpointMonitoringRecord,
    build_dev_endpoint_monitoring_record,
)
from sqlbench_lab.mlops.gcs_sync import SQLAdapterGCSSyncPlan
from sqlbench_lab.mlops.observability import SQLAdapterDevObservabilityRecord, build_dev_observability_record
from sqlbench_lab.mlops.offline_dev_flow import (
    SQLAdapterOfflineFlowPlan,
    build_offline_flow_gcs_sync_plan,
    build_offline_run_contract,
    decide_offline_flow_promotion,
)
from sqlbench_lab.mlops.promotion_registry import SQLAdapterDevPromotionRegistryPlan, build_dev_promotion_registry_plan
from sqlbench_lab.mlops.run_contract import SQLAdapterPromotionDecision, SQLAdapterRunContract
from sqlbench_lab.mlops.vertex_job import SQLAdapterVertexTrainingJobPlan, build_dev_vertex_training_job_plan

DEV_CLOUD_BUNDLE_SCHEMA_VERSION = "sql_adapter_dev_cloud_bundle:v1"


@dataclass(frozen=True)
class SQLAdapterDevCloudBundle:
    schema_version: str
    run_contract: SQLAdapterRunContract
    promotion_decision: SQLAdapterPromotionDecision
    gcs_sync_plan: SQLAdapterGCSSyncPlan
    vertex_training_job_plan: SQLAdapterVertexTrainingJobPlan
    dev_endpoint_plan: SQLAdapterDevEndpointPlan
    promotion_registry_plan: SQLAdapterDevPromotionRegistryPlan
    dev_observability_record: SQLAdapterDevObservabilityRecord
    endpoint_monitoring_record: SQLAdapterDevEndpointMonitoringRecord
    cost_capacity_record: SQLAdapterDevCostCapacityRecord

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dev_cloud_bundle_from_offline_plan(
    plan: SQLAdapterOfflineFlowPlan,
    *,
    run_id: str,
    gcp_project: str,
    gcp_region: str,
    training_image_uri: str,
    serving_image_uri: str,
    dev_db_id: str,
    git_sha: str | None = None,
    endpoint_uptime_hours: float | None = None,
    training_hourly_cost_usd: float | None = None,
    endpoint_hourly_cost_usd: float | None = None,
) -> SQLAdapterDevCloudBundle:
    """Build all dev cloud/hardening artifacts from one offline flow plan."""

    contract = build_offline_run_contract(plan)
    decision = decide_offline_flow_promotion(plan)
    gcs_sync_plan = build_offline_flow_gcs_sync_plan(plan, run_id=run_id)
    vertex_training_job_plan = build_dev_vertex_training_job_plan(
        contract,
        gcs_sync_plan,
        project_id=gcp_project,
        region=gcp_region,
        image_uri=training_image_uri,
    )
    dev_endpoint_plan = build_dev_gcp_vllm_endpoint_plan(
        contract,
        gcs_sync_plan,
        project_id=gcp_project,
        region=gcp_region,
        image_uri=serving_image_uri,
    )
    promotion_registry_plan = build_dev_promotion_registry_plan(
        contract,
        gcs_sync_plan,
        decision,
        db_id=dev_db_id,
    )
    dev_observability_record = build_dev_observability_record(
        contract,
        decision,
        gcs_sync_plan,
        git_sha=git_sha,
        container_image_uri=training_image_uri,
        registry_plan=promotion_registry_plan,
    )
    endpoint_monitoring_record = build_dev_endpoint_monitoring_record(contract, dev_endpoint_plan)
    cost_capacity_record = build_dev_cost_capacity_record(
        contract,
        vertex_plan=vertex_training_job_plan,
        endpoint_plan=dev_endpoint_plan,
        endpoint_uptime_hours=endpoint_uptime_hours,
        training_hourly_cost_usd=training_hourly_cost_usd,
        endpoint_hourly_cost_usd=endpoint_hourly_cost_usd,
    )
    return SQLAdapterDevCloudBundle(
        schema_version=DEV_CLOUD_BUNDLE_SCHEMA_VERSION,
        run_contract=contract,
        promotion_decision=decision,
        gcs_sync_plan=gcs_sync_plan,
        vertex_training_job_plan=vertex_training_job_plan,
        dev_endpoint_plan=dev_endpoint_plan,
        promotion_registry_plan=promotion_registry_plan,
        dev_observability_record=dev_observability_record,
        endpoint_monitoring_record=endpoint_monitoring_record,
        cost_capacity_record=cost_capacity_record,
    )


def write_dev_cloud_bundle(
    bundle: SQLAdapterDevCloudBundle,
    *,
    output_path: str | Path,
    vertex_config_output_path: str | Path | None = None,
) -> None:
    """Write the bundle JSON and optional Vertex CustomJobSpec JSON."""

    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(bundle.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    if vertex_config_output_path is not None:
        resolved_vertex_config = Path(vertex_config_output_path)
        resolved_vertex_config.parent.mkdir(parents=True, exist_ok=True)
        resolved_vertex_config.write_text(
            json.dumps(bundle.vertex_training_job_plan.to_custom_job_spec(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
