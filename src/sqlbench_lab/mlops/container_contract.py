"""Dev container contract for SQL adapter CLI execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DEV_CLI_IMAGE_NAME = "sqlbench-lab-dev-cli"
DEV_CLI_IMAGE_TAG = "dev"
DEV_CONTAINER_DOCKERFILE = "docker/sqlbench-dev-cli.Dockerfile"
DEV_VLLM_IMAGE_NAME = "sqlbench-vllm"
DEV_VLLM_IMAGE_TAG = "dev"
DEV_VLLM_BASE_IMAGE = "vllm/vllm-openai:v0.22.1"
DEV_VLLM_TORCH_CUDA_RUNTIME = "torch 2.11.0+cu130 / CUDA 13.0"
DEV_VLLM_DOCKERFILE = "docker/sqlbench-vllm.Dockerfile"
DEV_VLLM_ENTRYPOINT = "docker/sqlbench-vllm-entrypoint.py"


@dataclass(frozen=True)
class SQLAdapterDevContainerContract:
    image_name: str
    image_tag: str
    dockerfile_path: str
    build_context: str
    entrypoint: tuple[str, ...]
    default_dependency_groups: tuple[str, ...]
    optional_dependency_groups: tuple[str, ...]
    supported_command_summaries: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SQLAdapterDevServingContainerContract:
    image_name: str
    image_tag: str
    base_image: str
    torch_cuda_runtime: str
    dockerfile_path: str
    entrypoint_path: str
    build_context: str
    required_environment_variables: tuple[str, ...]
    optional_environment_variables: tuple[str, ...]
    exposed_port: int
    startup_summary: str
    runtime_compatibility_notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def dev_cli_container_contract() -> SQLAdapterDevContainerContract:
    """Return the canonical dev CLI container contract."""

    return SQLAdapterDevContainerContract(
        image_name=DEV_CLI_IMAGE_NAME,
        image_tag=DEV_CLI_IMAGE_TAG,
        dockerfile_path=DEV_CONTAINER_DOCKERFILE,
        build_context=".",
        entrypoint=("python", "-m", "sqlbench_lab.cli"),
        default_dependency_groups=(),
        optional_dependency_groups=("mlops", "training", "serving"),
        supported_command_summaries=(
            "sql validate-manifest",
            "sql run-sft --dry-run",
            "sql analyze-eval",
            "sql eval --openai-base-url",
            "sql openai-load-test",
            "docs build",
        ),
    )


def dev_vllm_serving_container_contract() -> SQLAdapterDevServingContainerContract:
    """Return the canonical dev vLLM serving container contract."""

    return SQLAdapterDevServingContainerContract(
        image_name=DEV_VLLM_IMAGE_NAME,
        image_tag=DEV_VLLM_IMAGE_TAG,
        base_image=DEV_VLLM_BASE_IMAGE,
        torch_cuda_runtime=DEV_VLLM_TORCH_CUDA_RUNTIME,
        dockerfile_path=DEV_VLLM_DOCKERFILE,
        entrypoint_path=DEV_VLLM_ENTRYPOINT,
        build_context=".",
        required_environment_variables=(
            "SQLBENCH_BASE_MODEL",
            "SQLBENCH_BASE_SERVED_MODEL",
            "SQLBENCH_OPENAI_MODEL",
            "SQLBENCH_ADAPTER_NAME",
            "SQLBENCH_ADAPTER_URI",
        ),
        optional_environment_variables=(
            "SQLBENCH_HOST",
            "SQLBENCH_PORT",
            "SQLBENCH_MAX_MODEL_LEN",
            "SQLBENCH_MAX_NUM_SEQS",
            "SQLBENCH_GPU_MEMORY_UTILIZATION",
            "SQLBENCH_MAX_LORA_RANK",
            "SQLBENCH_BASE_MODEL_URI",
            "SQLBENCH_VLLM_EXTRA_ARGS",
        ),
        exposed_port=8000,
        startup_summary="download GCS adapter prefix, then exec vllm serve with --enable-lora and a LoRA request model",
        runtime_compatibility_notes=(
            "Cloud Run L4 rejected the current image on 2026-06-21 because its managed driver reported CUDA driver 12020.",
            "The current dev image uses torch 2.11.0+cu130 / CUDA 13.0 and requires a GPU target with a compatible NVIDIA driver.",
            "Use Vertex, GKE, or GCE where the driver can satisfy the image, or build a separate CUDA 12.x serving stack and rerun endpoint eval plus load gates.",
        ),
    )


def build_dev_cli_docker_build_command(
    *,
    tag: str = f"{DEV_CLI_IMAGE_NAME}:{DEV_CLI_IMAGE_TAG}",
    install_groups: tuple[str, ...] = ("mlops",),
) -> tuple[str, ...]:
    contract = dev_cli_container_contract()
    return (
        "docker",
        "build",
        "-f",
        contract.dockerfile_path,
        "--build-arg",
        f"INSTALL_GROUPS={' '.join(install_groups)}",
        "-t",
        tag,
        contract.build_context,
    )


def build_dev_cli_docker_run_command(
    *,
    tag: str = f"{DEV_CLI_IMAGE_NAME}:{DEV_CLI_IMAGE_TAG}",
    cli_args: tuple[str, ...],
) -> tuple[str, ...]:
    if not cli_args:
        raise ValueError("cli_args must be non-empty")
    return ("docker", "run", "--rm", tag, *cli_args)


def build_dev_vllm_docker_build_command(
    *,
    tag: str = f"{DEV_VLLM_IMAGE_NAME}:{DEV_VLLM_IMAGE_TAG}",
) -> tuple[str, ...]:
    contract = dev_vllm_serving_container_contract()
    return (
        "docker",
        "build",
        "-f",
        contract.dockerfile_path,
        "-t",
        tag,
        contract.build_context,
    )
