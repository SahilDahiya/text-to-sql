"""Materialize dev cloud bundle artifacts locally and in GCS."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlbench_lab.mlops.dev_cloud_bundle import SQLAdapterDevCloudBundle
from sqlbench_lab.mlops.gcs_sync import SQLAdapterGCSArtifact, SQLAdapterGCSArtifactKind

DEV_CLOUD_PUBLISH_SCHEMA_VERSION = "sql_adapter_dev_cloud_publish:v1"
DEV_CLOUD_ARTIFACT_MANIFEST_SCHEMA_VERSION = "sql_adapter_dev_artifact_manifest:v1"


class CommandRunner(Protocol):
    def __call__(self, command: tuple[str, ...]) -> None: ...


@dataclass(frozen=True)
class SQLAdapterDevCloudPublishedArtifact:
    local_path: str
    gcs_uri: str
    is_directory: bool
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SQLAdapterDevCloudPublishRecord:
    schema_version: str
    environment: str
    run_id: str
    experiment_id: str
    local_dir: str
    published_to_gcs: bool
    current_pointer_updated: bool
    uploaded_uris: tuple[str, ...]
    generated_files: tuple[str, ...]
    artifact_manifest_path: str
    artifacts: tuple[SQLAdapterDevCloudPublishedArtifact, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def materialize_dev_cloud_bundle(
    bundle: SQLAdapterDevCloudBundle,
    *,
    local_dir: str | Path,
    publish_gcs: bool = False,
    update_current_pointer: bool = False,
    runner: CommandRunner | None = None,
) -> SQLAdapterDevCloudPublishRecord:
    """Write generated dev artifacts and optionally publish them to GCS."""

    resolved_local_dir = Path(local_dir)
    resolved_local_dir.mkdir(parents=True, exist_ok=True)
    generated_files = _write_generated_files(bundle, resolved_local_dir)
    if update_current_pointer and not bundle.promotion_registry_plan.eligible_for_current:
        raise ValueError("refusing to update dev current pointer for a non-promoted run")
    upload_specs_without_manifest = _build_upload_specs(bundle, generated_files)
    pointer_upload_specs = (
        (
            (str(generated_files["current_pointer"]), bundle.promotion_registry_plan.current_pointer_uri, False),
            (str(generated_files["rollback_pointer"]), bundle.promotion_registry_plan.rollback_pointer_uri, False),
        )
        if update_current_pointer
        else ()
    )
    published_artifacts = _build_published_artifacts((*upload_specs_without_manifest, *pointer_upload_specs))
    artifact_manifest_path = resolved_local_dir / "artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        {
            "schema_version": DEV_CLOUD_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "environment": bundle.run_contract.environment.environment,
            "run_id": bundle.gcs_sync_plan.run_id,
            "experiment_id": bundle.run_contract.inputs.experiment_id,
            "artifacts": [asdict(artifact) for artifact in published_artifacts],
        },
    )
    generated_files["artifact_manifest"] = artifact_manifest_path
    upload_specs = (
        *upload_specs_without_manifest,
        (str(artifact_manifest_path), f"{bundle.gcs_sync_plan.prefix}/artifact_manifest.json", False),
    )
    uploaded_uris: list[str] = []
    command_runner = runner or _run_command

    if publish_gcs:
        for local_path, gcs_uri, is_dir in upload_specs:
            if is_dir:
                command_runner(("gsutil", "-m", "rsync", "-r", local_path, gcs_uri))
            else:
                command_runner(("gsutil", "cp", local_path, gcs_uri))
            uploaded_uris.append(gcs_uri)

    if update_current_pointer:
        current_pointer = generated_files["current_pointer"]
        rollback_pointer = generated_files["rollback_pointer"]
        if publish_gcs:
            command_runner(("gsutil", "cp", str(current_pointer), bundle.promotion_registry_plan.current_pointer_uri))
            command_runner(("gsutil", "cp", str(rollback_pointer), bundle.promotion_registry_plan.rollback_pointer_uri))
            uploaded_uris.extend(
                (
                    bundle.promotion_registry_plan.current_pointer_uri,
                    bundle.promotion_registry_plan.rollback_pointer_uri,
                )
            )

    record = SQLAdapterDevCloudPublishRecord(
        schema_version=DEV_CLOUD_PUBLISH_SCHEMA_VERSION,
        environment=bundle.run_contract.environment.environment,
        run_id=bundle.gcs_sync_plan.run_id,
        experiment_id=bundle.run_contract.inputs.experiment_id,
        local_dir=str(resolved_local_dir),
        published_to_gcs=publish_gcs,
        current_pointer_updated=update_current_pointer,
        uploaded_uris=tuple(uploaded_uris),
        generated_files=tuple(str(path) for path in generated_files.values()),
        artifact_manifest_path=str(artifact_manifest_path),
        artifacts=published_artifacts,
    )
    publish_record_path = resolved_local_dir / "publish_record.json"
    publish_record_path.write_text(json.dumps(record.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return record


def _write_generated_files(bundle: SQLAdapterDevCloudBundle, local_dir: Path) -> dict[str, Path]:
    files = {
        "cloud_bundle": local_dir / "cloud_bundle.json",
        "run_contract": local_dir / "run_contract.json",
        "decision": local_dir / "decision.json",
        "promotion_metadata": local_dir / "promotion_metadata.json",
        "observability": local_dir / "observability.json",
        "endpoint_monitoring": local_dir / "endpoint_monitoring.json",
        "cost_capacity": local_dir / "cost_capacity.json",
        "current_pointer": local_dir / "current_pointer.json",
        "rollback_pointer": local_dir / "rollback_pointer.json",
    }
    _write_json(files["cloud_bundle"], bundle.to_json_dict())
    _write_json(files["run_contract"], bundle.run_contract.to_json_dict())
    _write_json(files["decision"], bundle.promotion_decision.to_json_dict())
    _write_json(files["promotion_metadata"], bundle.promotion_registry_plan.to_json_dict())
    _write_json(files["observability"], bundle.dev_observability_record.to_json_dict())
    _write_json(files["endpoint_monitoring"], bundle.endpoint_monitoring_record.to_json_dict())
    _write_json(files["cost_capacity"], bundle.cost_capacity_record.to_json_dict())
    pointer = {
        "schema_version": "sql_adapter_dev_pointer:v1",
        "environment": bundle.promotion_registry_plan.environment,
        "db_id": bundle.promotion_registry_plan.db_id,
        "adapter_version": bundle.promotion_registry_plan.adapter_version,
        "adapter_uri": bundle.promotion_registry_plan.adapter_uri,
        "metadata_uri": bundle.promotion_registry_plan.metadata_uri,
        "run_contract_uri": bundle.promotion_registry_plan.run_contract_uri,
        "decision_uri": bundle.promotion_registry_plan.decision_uri,
    }
    _write_json(files["current_pointer"], pointer)
    _write_json(files["rollback_pointer"], pointer)
    return files


def _build_upload_specs(
    bundle: SQLAdapterDevCloudBundle,
    generated_files: dict[str, Path],
) -> tuple[tuple[str, str, bool], ...]:
    specs: list[tuple[str, str, bool]] = []
    generated_artifacts = {
        SQLAdapterGCSArtifactKind.RUN_CONTRACT: generated_files["run_contract"],
        SQLAdapterGCSArtifactKind.PROMOTION_DECISION: generated_files["decision"],
    }
    for artifact in bundle.gcs_sync_plan.artifacts:
        specs.append(_artifact_upload_spec(artifact, generated_artifacts))
    specs.extend(
        (
            (str(generated_files["cloud_bundle"]), f"{bundle.gcs_sync_plan.prefix}/cloud_bundle.json", False),
            (str(generated_files["promotion_metadata"]), bundle.promotion_registry_plan.metadata_uri, False),
            (str(generated_files["observability"]), f"{bundle.gcs_sync_plan.prefix}/observability.json", False),
            (
                str(generated_files["endpoint_monitoring"]),
                f"{bundle.gcs_sync_plan.prefix}/endpoint_monitoring.json",
                False,
            ),
            (str(generated_files["cost_capacity"]), f"{bundle.gcs_sync_plan.prefix}/cost_capacity.json", False),
        )
    )
    return tuple(specs)


def _artifact_upload_spec(
    artifact: SQLAdapterGCSArtifact,
    generated_artifacts: dict[SQLAdapterGCSArtifactKind, Path],
) -> tuple[str, str, bool]:
    if artifact.kind in generated_artifacts:
        return str(generated_artifacts[artifact.kind]), artifact.gcs_uri, False
    is_dir = artifact.kind == SQLAdapterGCSArtifactKind.ADAPTER_DIR
    local_path = Path(artifact.local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"required artifact does not exist: {local_path}")
    return str(local_path), artifact.gcs_uri, is_dir


def _build_published_artifacts(
    upload_specs: tuple[tuple[str, str, bool], ...],
) -> tuple[SQLAdapterDevCloudPublishedArtifact, ...]:
    artifacts: list[SQLAdapterDevCloudPublishedArtifact] = []
    for local_path, gcs_uri, is_dir in upload_specs:
        resolved_path = Path(local_path)
        if is_dir:
            size_bytes, digest = _directory_digest(resolved_path)
        else:
            if not resolved_path.is_file():
                raise FileNotFoundError(f"required artifact file does not exist: {resolved_path}")
            size_bytes = resolved_path.stat().st_size
            digest = _file_sha256(resolved_path)
        artifacts.append(
            SQLAdapterDevCloudPublishedArtifact(
                local_path=str(resolved_path),
                gcs_uri=gcs_uri,
                is_directory=is_dir,
                size_bytes=size_bytes,
                sha256=digest,
            )
        )
    return tuple(artifacts)


def _directory_digest(path: Path) -> tuple[int, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"required artifact directory does not exist: {path}")
    total_size = 0
    hasher = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative_path = child.relative_to(path).as_posix()
        child_size = child.stat().st_size
        child_digest = _file_sha256(child)
        total_size += child_size
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(child_size).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(child_digest.encode("ascii"))
        hasher.update(b"\n")
    return total_size, hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_command(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True)
