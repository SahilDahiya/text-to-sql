"""Dev container contract for SQL adapter CLI execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DEV_CLI_IMAGE_NAME = "sqlbench-lab-dev-cli"
DEV_CLI_IMAGE_TAG = "dev"
DEV_CONTAINER_DOCKERFILE = "docker/sqlbench-dev-cli.Dockerfile"


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
