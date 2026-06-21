"""Temporary dev GCP vLLM endpoint contract for SQL adapter serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlbench_lab.mlops.gcs_sync import SQLAdapterGCSSyncPlan
from sqlbench_lab.mlops.run_contract import DEV_ENVIRONMENT, SQLAdapterRunContract

DEV_ENDPOINT_PLAN_SCHEMA_VERSION = "sql_adapter_dev_vllm_endpoint:v1"


@dataclass(frozen=True)
class SQLAdapterDevEndpointPlan:
    schema_version: str
    environment: str
    project_id: str
    region: str
    endpoint_id: str
    serving_target: str
    image_uri: str
    service_account: str
    base_model: str
    adapter_name: str
    adapter_uri: str
    openai_model: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    min_replica_count: int
    max_replica_count: int
    max_model_len: int
    max_num_seqs: int
    max_lora_rank: int
    gpu_memory_utilization: float
    environment_variables: dict[str, str]
    startup_args: tuple[str, ...]
    health_path: str = "/health"
    completions_path: str = "/v1/completions"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dev_gcp_vllm_endpoint_plan(
    contract: SQLAdapterRunContract,
    gcs_plan: SQLAdapterGCSSyncPlan,
    *,
    project_id: str,
    region: str,
    image_uri: str,
    endpoint_id: str = "sql-adapter-dev-vllm",
    serving_target: str = "gcp_temporary_gpu_endpoint",
    machine_type: str = "g2-standard-4",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    max_model_len: int = 4096,
    max_num_seqs: int = 64,
    max_lora_rank: int = 16,
    gpu_memory_utilization: float = 0.75,
) -> SQLAdapterDevEndpointPlan:
    """Build the temporary dev endpoint plan without provisioning infrastructure."""

    if contract.environment.environment != DEV_ENVIRONMENT:
        raise ValueError(f"dev endpoint only supports environment={DEV_ENVIRONMENT!r}")
    if gcs_plan.environment != DEV_ENVIRONMENT:
        raise ValueError(f"dev endpoint only supports GCS plan environment={DEV_ENVIRONMENT!r}")
    if min_replica_count <= 0 or max_replica_count < min_replica_count:
        raise ValueError("replica counts must be positive and max_replica_count >= min_replica_count")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    resolved_project = _non_empty(project_id, "project_id")
    resolved_max_model_len = _positive_int(max_model_len, "max_model_len")
    resolved_max_num_seqs = _positive_int(max_num_seqs, "max_num_seqs")
    resolved_max_lora_rank = _positive_int(max_lora_rank, "max_lora_rank")
    openai_model = f"{contract.inputs.experiment_id}-dev"
    environment_variables = {
        "SQLBENCH_BASE_MODEL": contract.inputs.base_model,
        "SQLBENCH_OPENAI_MODEL": openai_model,
        "SQLBENCH_ADAPTER_NAME": contract.inputs.adapter_name,
        "SQLBENCH_ADAPTER_URI": gcs_plan.adapter_uri,
        "SQLBENCH_MAX_MODEL_LEN": str(resolved_max_model_len),
        "SQLBENCH_MAX_NUM_SEQS": str(resolved_max_num_seqs),
        "SQLBENCH_MAX_LORA_RANK": str(resolved_max_lora_rank),
        "SQLBENCH_GPU_MEMORY_UTILIZATION": f"{gpu_memory_utilization:.2f}",
    }
    startup_args = (
        "--model",
        contract.inputs.base_model,
        "--served-model-name",
        openai_model,
        "--enable-lora",
        "--lora-modules",
        f"{contract.inputs.adapter_name}={gcs_plan.adapter_uri}",
        "--max-model-len",
        str(resolved_max_model_len),
        "--max-num-seqs",
        str(resolved_max_num_seqs),
        "--max-lora-rank",
        str(resolved_max_lora_rank),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.2f}",
    )
    return SQLAdapterDevEndpointPlan(
        schema_version=DEV_ENDPOINT_PLAN_SCHEMA_VERSION,
        environment=DEV_ENVIRONMENT,
        project_id=resolved_project,
        region=_non_empty(region, "region"),
        endpoint_id=_non_empty(endpoint_id, "endpoint_id"),
        serving_target=_non_empty(serving_target, "serving_target"),
        image_uri=_non_empty(image_uri, "image_uri"),
        service_account=_gcp_service_account_email(contract.environment.serving_service_account, resolved_project),
        base_model=contract.inputs.base_model,
        adapter_name=contract.inputs.adapter_name,
        adapter_uri=gcs_plan.adapter_uri,
        openai_model=openai_model,
        machine_type=_non_empty(machine_type, "machine_type"),
        accelerator_type=_non_empty(accelerator_type, "accelerator_type"),
        accelerator_count=_positive_int(accelerator_count, "accelerator_count"),
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        max_model_len=resolved_max_model_len,
        max_num_seqs=resolved_max_num_seqs,
        max_lora_rank=resolved_max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        environment_variables=environment_variables,
        startup_args=startup_args,
    )


def _non_empty(value: str, name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty")
    return resolved


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _gcp_service_account_email(value: str, project_id: str) -> str:
    resolved = _non_empty(value, "service_account")
    if "@" in resolved:
        return resolved
    return f"{resolved}@{_non_empty(project_id, 'project_id')}.iam.gserviceaccount.com"
