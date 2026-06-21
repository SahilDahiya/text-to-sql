"""Dev adapter promotion registry contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlbench_lab.mlops.gcs_sync import SQLAdapterGCSSyncPlan
from sqlbench_lab.mlops.run_contract import (
    DEV_ENVIRONMENT,
    PROMOTE_DECISION,
    SQLAdapterPromotionDecision,
    SQLAdapterRunContract,
)

DEV_PROMOTION_REGISTRY_SCHEMA_VERSION = "sql_adapter_dev_promotion_registry:v1"


@dataclass(frozen=True)
class SQLAdapterDevPromotionRegistryPlan:
    schema_version: str
    environment: str
    db_id: str
    adapter_version: str
    adapter_uri: str
    metadata_uri: str
    current_pointer_uri: str
    rollback_pointer_uri: str
    run_contract_uri: str
    decision_uri: str
    eligible_for_current: bool
    decision: str
    reasons: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dev_promotion_registry_plan(
    contract: SQLAdapterRunContract,
    gcs_plan: SQLAdapterGCSSyncPlan,
    decision: SQLAdapterPromotionDecision,
    *,
    db_id: str,
) -> SQLAdapterDevPromotionRegistryPlan:
    """Build the dev registry pointer plan without mutating GCS state."""

    if contract.environment.environment != DEV_ENVIRONMENT:
        raise ValueError(f"dev promotion registry only supports environment={DEV_ENVIRONMENT!r}")
    if gcs_plan.environment != DEV_ENVIRONMENT:
        raise ValueError(f"dev promotion registry only supports GCS plan environment={DEV_ENVIRONMENT!r}")
    resolved_db_id = _safe_path_part(db_id, "db_id")
    adapter_version = f"{contract.inputs.adapter_name}__{gcs_plan.run_id}"
    registry_root = _join_gcs_uri(contract.environment.model_bucket, "promoted", resolved_db_id, DEV_ENVIRONMENT)
    metadata_uri = _join_gcs_uri(contract.environment.model_bucket, "adapters", adapter_version, "metadata.json")
    return SQLAdapterDevPromotionRegistryPlan(
        schema_version=DEV_PROMOTION_REGISTRY_SCHEMA_VERSION,
        environment=DEV_ENVIRONMENT,
        db_id=resolved_db_id,
        adapter_version=adapter_version,
        adapter_uri=gcs_plan.adapter_uri,
        metadata_uri=metadata_uri,
        current_pointer_uri=_join_gcs_uri(registry_root, "current.json"),
        rollback_pointer_uri=_join_gcs_uri(registry_root, "rollback.json"),
        run_contract_uri=gcs_plan.run_contract_uri,
        decision_uri=gcs_plan.decision_uri,
        eligible_for_current=decision.decision == PROMOTE_DECISION,
        decision=decision.decision,
        reasons=decision.reasons,
    )


def _join_gcs_uri(prefix: str, *parts: str) -> str:
    if not prefix.startswith("gs://"):
        raise ValueError(f"GCS URI prefix must start with gs://: {prefix}")
    stripped_prefix = prefix.rstrip("/")
    stripped_parts = [part.strip("/") for part in parts if part.strip("/")]
    return "/".join((stripped_prefix, *stripped_parts))


def _safe_path_part(value: str, name: str) -> str:
    resolved = value.strip().strip("/")
    if not resolved:
        raise ValueError(f"{name} must be non-empty")
    if "/" in resolved:
        raise ValueError(f"{name} must be a single path segment")
    return resolved
