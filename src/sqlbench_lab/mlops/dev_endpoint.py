"""Temporary dev GCP vLLM endpoint contract for SQL adapter serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlbench_lab.mlops.gcs_sync import SQLAdapterGCSSyncPlan
from sqlbench_lab.mlops.run_contract import DEV_ENVIRONMENT, SQLAdapterRunContract

DEV_ENDPOINT_PLAN_SCHEMA_VERSION = "sql_adapter_dev_vllm_endpoint:v1"
DEV_ENDPOINT_SERVING_TARGET_GCE_GPU_VM = "gce_gpu_vm"
DEV_ENDPOINT_SERVING_TARGET_LOCAL_GPU_DOCKER = "local_gpu_docker"
DEV_ENDPOINT_SERVING_TARGET_CLOUD_RUN_GPU = "cloud_run_gpu"
DEV_ENDPOINT_SUPPORTED_SERVING_TARGETS = (
    DEV_ENDPOINT_SERVING_TARGET_GCE_GPU_VM,
    DEV_ENDPOINT_SERVING_TARGET_LOCAL_GPU_DOCKER,
    "gke_gpu_node_pool",
    "vertex_custom_gpu_endpoint",
)
DEV_ENDPOINT_REJECTED_SERVING_TARGETS = (DEV_ENDPOINT_SERVING_TARGET_CLOUD_RUN_GPU,)


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
    base_model_uri: str | None
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
    requires_gpu_driver_control: bool
    preflight_checks: tuple[str, ...]
    runtime_requirements: tuple[str, ...]
    rejected_serving_targets: tuple[str, ...]
    deployment_notes: tuple[str, ...]
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
    serving_target: str = DEV_ENDPOINT_SERVING_TARGET_GCE_GPU_VM,
    machine_type: str = "g2-standard-4",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    max_model_len: int = 4096,
    max_num_seqs: int = 64,
    max_lora_rank: int = 16,
    gpu_memory_utilization: float = 0.75,
    base_model_uri: str | None = None,
    vllm_extra_args: str | None = None,
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
    resolved_serving_target = _serving_target(serving_target)
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
    resolved_base_model_uri = _optional_gcs_uri(base_model_uri, "base_model_uri")
    if resolved_base_model_uri is not None:
        environment_variables["SQLBENCH_BASE_MODEL_URI"] = resolved_base_model_uri
    resolved_vllm_extra_args = _optional_non_empty(vllm_extra_args, "vllm_extra_args")
    if resolved_vllm_extra_args is not None:
        environment_variables["SQLBENCH_VLLM_EXTRA_ARGS"] = resolved_vllm_extra_args
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
        serving_target=resolved_serving_target,
        image_uri=_non_empty(image_uri, "image_uri"),
        service_account=_gcp_service_account_email(contract.environment.serving_service_account, resolved_project),
        base_model=contract.inputs.base_model,
        base_model_uri=resolved_base_model_uri,
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
        requires_gpu_driver_control=True,
        preflight_checks=_preflight_checks(
            resolved_serving_target,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
        ),
        runtime_requirements=(
            "current dev image requires a CUDA 13.0-compatible NVIDIA driver",
            "base model must be available from SQLBENCH_BASE_MODEL_URI or Hugging Face before vLLM starts",
            "adapter GCS prefix must contain adapter_config.json and adapter_model.safetensors",
        ),
        rejected_serving_targets=DEV_ENDPOINT_REJECTED_SERVING_TARGETS,
        deployment_notes=(
            "Cloud Run GPU is rejected for this dev image after the 2026-06-21 L4 attempt reported driver 12020.",
            "local_gpu_docker is the prod-like rehearsal target: same image, same env contract, local NVIDIA runtime, and local GCP ADC mount.",
            "Use a target with explicit NVIDIA driver control, such as a GCE GPU VM, GKE GPU node pool, or Vertex custom GPU endpoint.",
            "Promotion still requires endpoint eval and load-test artifacts from the live OpenAI-compatible endpoint.",
        ),
        environment_variables=environment_variables,
        startup_args=startup_args,
    )


def build_docker_local_vllm_run_command(
    plan: SQLAdapterDevEndpointPlan,
    *,
    host_port: int = 8000,
    container_name: str = "sqlbench-vllm-local",
    gcloud_config_dir: str = "$HOME/.config/gcloud",
    local_base_model_dir: str | None = None,
    local_adapter_dir: str | None = None,
) -> tuple[str, ...]:
    """Build a local GPU Docker command that rehearses the dev serving endpoint."""

    if plan.serving_target != DEV_ENDPOINT_SERVING_TARGET_LOCAL_GPU_DOCKER:
        raise ValueError("local Docker run command requires serving_target='local_gpu_docker'")
    resolved_host_port = _positive_int(host_port, "host_port")
    resolved_container_name = _non_empty(container_name, "container_name")
    resolved_gcloud_config_dir = _non_empty(gcloud_config_dir, "gcloud_config_dir")
    env = dict(plan.environment_variables)
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--name",
        resolved_container_name,
        "-p",
        f"{resolved_host_port}:8000",
        "-v",
        f"{resolved_gcloud_config_dir}:/root/.config/gcloud:ro",
    ]
    if local_base_model_dir is not None:
        resolved_base_model_dir = _non_empty(local_base_model_dir, "local_base_model_dir")
        command.extend(("-v", f"{resolved_base_model_dir}:/mnt/sqlbench/base_model:ro"))
        env["SQLBENCH_BASE_MODEL_URI"] = "file:///mnt/sqlbench/base_model"
    if local_adapter_dir is not None:
        resolved_adapter_dir = _non_empty(local_adapter_dir, "local_adapter_dir")
        command.extend(("-v", f"{resolved_adapter_dir}:/mnt/sqlbench/adapter:ro"))
        env["SQLBENCH_ADAPTER_URI"] = "file:///mnt/sqlbench/adapter"
    command.extend(("-e", f"GOOGLE_CLOUD_PROJECT={plan.project_id}"))
    command.extend(("-e", f"CLOUDSDK_CORE_PROJECT={plan.project_id}"))
    for key, value in env.items():
        command.extend(("-e", f"{key}={value}"))
    command.append(plan.image_uri)
    return tuple(command)


def build_gcloud_gce_vllm_vm_create_command(
    plan: SQLAdapterDevEndpointPlan,
    *,
    zone: str,
    instance_name: str = "sqlbench-vllm-dev-l4",
    image_family: str = "common-cu129-ubuntu-2404-nvidia-580",
    image_project: str = "deeplearning-platform-release",
    boot_disk_size_gb: int = 120,
    boot_disk_type: str = "pd-balanced",
) -> tuple[str, ...]:
    """Build the GCE VM create command for a driver-controlled dev serving endpoint."""

    if plan.serving_target != DEV_ENDPOINT_SERVING_TARGET_GCE_GPU_VM:
        raise ValueError("GCE VM create command requires serving_target='gce_gpu_vm'")
    resolved_zone = _non_empty(zone, "zone")
    resolved_instance_name = _non_empty(instance_name, "instance_name")
    resolved_image_family = _non_empty(image_family, "image_family")
    resolved_image_project = _non_empty(image_project, "image_project")
    resolved_boot_disk_type = _non_empty(boot_disk_type, "boot_disk_type")
    resolved_boot_disk_size_gb = _positive_int(boot_disk_size_gb, "boot_disk_size_gb")
    return (
        "gcloud",
        "compute",
        "instances",
        "create",
        resolved_instance_name,
        f"--project={plan.project_id}",
        f"--zone={resolved_zone}",
        f"--machine-type={plan.machine_type}",
        f"--accelerator=type={_gce_accelerator_type(plan.accelerator_type)},count={plan.accelerator_count}",
        "--maintenance-policy=TERMINATE",
        f"--image-family={resolved_image_family}",
        f"--image-project={resolved_image_project}",
        f"--boot-disk-size={resolved_boot_disk_size_gb}GB",
        f"--boot-disk-type={resolved_boot_disk_type}",
        f"--service-account={plan.service_account}",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        "--labels=purpose=sqlbench-dev-serving,owner=codex,delete_after=short",
        "--metadata=enable-oslogin=TRUE",
    )


def _gce_accelerator_type(value: str) -> str:
    resolved = _non_empty(value, "accelerator_type")
    if resolved == "NVIDIA_L4":
        return "nvidia-l4"
    return resolved.lower().replace("_", "-")


def _local_gpu_docker_preflight_checks() -> tuple[str, ...]:
    return (
        "local nvidia-smi must report a driver compatible with the image CUDA runtime",
        "docker info must list the nvidia runtime",
        "docker must be authenticated to Artifact Registry for the serving image",
        "local GCP ADC must be mounted read-only so the container can read GCS model and adapter prefixes",
    )


def _gce_gpu_vm_preflight_checks(*, accelerator_type: str, accelerator_count: int) -> tuple[str, ...]:
    return (
        "compute.googleapis.com/gpus_all_regions quota must be at least accelerator_count",
        f"{accelerator_type} regional quota in the selected region must be at least {accelerator_count}",
        "the serving service account must have Artifact Registry read and GCS object read permissions",
        "the selected zone must list the requested accelerator type",
    )


def _preflight_checks(
    serving_target: str,
    *,
    accelerator_type: str,
    accelerator_count: int,
) -> tuple[str, ...]:
    if serving_target == DEV_ENDPOINT_SERVING_TARGET_LOCAL_GPU_DOCKER:
        return _local_gpu_docker_preflight_checks()
    if serving_target == DEV_ENDPOINT_SERVING_TARGET_GCE_GPU_VM:
        return _gce_gpu_vm_preflight_checks(
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
        )
    return (
        "serving target must provide a GPU driver compatible with the image CUDA runtime",
        "the serving runtime must have Artifact Registry read and GCS object read permissions",
    )


def _serving_target(value: str) -> str:
    resolved = _non_empty(value, "serving_target")
    if resolved in DEV_ENDPOINT_REJECTED_SERVING_TARGETS:
        raise ValueError(
            "Cloud Run GPU is rejected for the current dev vLLM image because the managed L4 driver "
            "reported 12020 while the image requires a CUDA 13.0-compatible NVIDIA driver"
        )
    if resolved not in DEV_ENDPOINT_SUPPORTED_SERVING_TARGETS:
        supported = ", ".join(DEV_ENDPOINT_SUPPORTED_SERVING_TARGETS)
        raise ValueError(f"unsupported dev serving target {resolved!r}; supported: {supported}")
    return resolved


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


def _optional_gcs_uri(value: str | None, name: str) -> str | None:
    if value is None or not value.strip():
        return None
    resolved = value.strip().rstrip("/") + "/"
    if not resolved.startswith("gs://"):
        raise ValueError(f"{name} must start with gs://")
    return resolved


def _optional_non_empty(value: str | None, name: str) -> str | None:
    if value is None or not value.strip():
        return None
    return _non_empty(value, name)
