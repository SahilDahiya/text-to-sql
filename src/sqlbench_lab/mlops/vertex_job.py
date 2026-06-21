"""Dev Vertex AI custom-job contract for SQL adapter training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlbench_lab.mlops.gcs_sync import SQLAdapterGCSArtifactKind, SQLAdapterGCSSyncPlan
from sqlbench_lab.mlops.run_contract import (
    DEV_ENVIRONMENT,
    SQLAdapterRunContract,
)

DEV_VERTEX_JOB_SCHEMA_VERSION = "sql_adapter_vertex_training_job:v1"


@dataclass(frozen=True)
class SQLAdapterVertexMachineSpec:
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    replica_count: int = 1


@dataclass(frozen=True)
class SQLAdapterVertexTrainingJobPlan:
    schema_version: str
    environment: str
    project_id: str
    region: str
    display_name: str
    experiment_id: str
    image_uri: str
    service_account: str
    machine: SQLAdapterVertexMachineSpec
    command: tuple[str, ...]
    args: tuple[str, ...]
    manifest_uri: str
    container_manifest_path: str
    dry_run: bool
    output_prefix_uri: str
    run_contract_uri: str
    labels: dict[str, str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_custom_job_spec(self) -> dict[str, Any]:
        return {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": self.machine.machine_type,
                        "acceleratorType": self.machine.accelerator_type,
                        "acceleratorCount": self.machine.accelerator_count,
                    },
                    "replicaCount": self.machine.replica_count,
                    "containerSpec": {
                        "imageUri": self.image_uri,
                        "command": list(self.command),
                        "args": list(self.args),
                    },
                }
            ],
            "serviceAccount": self.service_account,
        }


def build_dev_vertex_training_job_plan(
    contract: SQLAdapterRunContract,
    gcs_plan: SQLAdapterGCSSyncPlan,
    *,
    project_id: str,
    region: str,
    image_uri: str,
    machine_type: str = "g2-standard-4",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    replica_count: int = 1,
    container_manifest_path: str | None = None,
    dry_run: bool = False,
) -> SQLAdapterVertexTrainingJobPlan:
    """Build the dev Vertex custom-job plan without submitting it."""

    _validate_dev_inputs(contract, gcs_plan)
    resolved_project = _non_empty(project_id, "project_id")
    resolved_region = _non_empty(region, "region")
    resolved_image = _non_empty(image_uri, "image_uri")
    manifest_uri = _artifact_uri(gcs_plan, SQLAdapterGCSArtifactKind.MANIFEST)
    resolved_container_manifest_path = _non_empty(
        container_manifest_path or contract.inputs.manifest_path,
        "container_manifest_path",
    )
    display_name = f"sql-adapter-dev-{contract.inputs.experiment_id}-{gcs_plan.run_id}"
    args = [
        "sql",
        "run-sft",
        "--manifest",
        resolved_container_manifest_path,
    ]
    if dry_run:
        args.append("--dry-run")
    return SQLAdapterVertexTrainingJobPlan(
        schema_version=DEV_VERTEX_JOB_SCHEMA_VERSION,
        environment=DEV_ENVIRONMENT,
        project_id=resolved_project,
        region=resolved_region,
        display_name=display_name,
        experiment_id=contract.inputs.experiment_id,
        image_uri=resolved_image,
        service_account=_gcp_service_account_email(contract.environment.training_service_account, resolved_project),
        machine=SQLAdapterVertexMachineSpec(
            machine_type=_non_empty(machine_type, "machine_type"),
            accelerator_type=_non_empty(accelerator_type, "accelerator_type"),
            accelerator_count=_positive_int(accelerator_count, "accelerator_count"),
            replica_count=_positive_int(replica_count, "replica_count"),
        ),
        command=("python", "-m", "sqlbench_lab.cli"),
        args=tuple(args),
        manifest_uri=manifest_uri,
        container_manifest_path=resolved_container_manifest_path,
        dry_run=dry_run,
        output_prefix_uri=gcs_plan.prefix,
        run_contract_uri=gcs_plan.run_contract_uri,
        labels={
            "environment": DEV_ENVIRONMENT,
            "experiment": _label_value(contract.inputs.experiment_id),
            "adapter": _label_value(contract.inputs.adapter_name),
        },
    )


def build_gcloud_vertex_custom_job_command(
    plan: SQLAdapterVertexTrainingJobPlan,
    *,
    config_path: str,
) -> tuple[str, ...]:
    """Build the gcloud command that submits a rendered CustomJobSpec YAML file."""

    return (
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--project={plan.project_id}",
        f"--region={plan.region}",
        f"--display-name={plan.display_name}",
        f"--config={_non_empty(config_path, 'config_path')}",
    )


def _validate_dev_inputs(contract: SQLAdapterRunContract, gcs_plan: SQLAdapterGCSSyncPlan) -> None:
    if contract.environment.environment != DEV_ENVIRONMENT:
        raise ValueError(f"Vertex dev job only supports environment={DEV_ENVIRONMENT!r}")
    if gcs_plan.environment != DEV_ENVIRONMENT:
        raise ValueError(f"Vertex dev job only supports GCS plan environment={DEV_ENVIRONMENT!r}")
    if contract.inputs.experiment_id != gcs_plan.experiment_id:
        raise ValueError("contract experiment_id must match gcs_plan experiment_id")


def _artifact_uri(gcs_plan: SQLAdapterGCSSyncPlan, kind: SQLAdapterGCSArtifactKind) -> str:
    matches = [artifact.gcs_uri for artifact in gcs_plan.artifacts if artifact.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind} artifact in GCS plan")
    return matches[0]


def _non_empty(value: str, name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty")
    return resolved


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _label_value(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)[:63].strip("_-") or "sql"


def _gcp_service_account_email(value: str, project_id: str) -> str:
    resolved = _non_empty(value, "service_account")
    if "@" in resolved:
        return resolved
    return f"{resolved}@{_non_empty(project_id, 'project_id')}.iam.gserviceaccount.com"
