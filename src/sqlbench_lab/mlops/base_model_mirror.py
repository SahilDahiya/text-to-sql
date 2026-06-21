"""Dev base-model mirror contract for serving cold-start hardening."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlbench_lab.mlops.run_contract import DEV_MODEL_BUCKET

DEV_BASE_MODEL_MIRROR_SCHEMA_VERSION = "sql_adapter_dev_base_model_mirror:v1"


@dataclass(frozen=True)
class SQLAdapterDevBaseModelMirrorPlan:
    schema_version: str
    base_model: str
    revision: str
    local_dir: str
    gcs_uri: str
    download_command: tuple[str, ...]
    upload_command: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dev_base_model_mirror_plan(
    *,
    base_model: str,
    revision: str,
    local_dir: str,
    gcs_uri: str | None = None,
) -> SQLAdapterDevBaseModelMirrorPlan:
    """Build the explicit dev base-model mirror commands."""

    resolved_base_model = _non_empty(base_model, "base_model")
    resolved_revision = _non_empty(revision, "revision")
    resolved_local_dir = _non_empty(local_dir, "local_dir").rstrip("/")
    resolved_gcs_uri = (gcs_uri or _default_gcs_uri(resolved_base_model, resolved_revision)).rstrip("/") + "/"
    if not resolved_gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    return SQLAdapterDevBaseModelMirrorPlan(
        schema_version=DEV_BASE_MODEL_MIRROR_SCHEMA_VERSION,
        base_model=resolved_base_model,
        revision=resolved_revision,
        local_dir=resolved_local_dir,
        gcs_uri=resolved_gcs_uri,
        download_command=(
            "hf",
            "download",
            resolved_base_model,
            "--revision",
            resolved_revision,
            "--local-dir",
            resolved_local_dir,
        ),
        upload_command=("gsutil", "-m", "rsync", "-r", resolved_local_dir, resolved_gcs_uri),
    )


def _default_gcs_uri(base_model: str, revision: str) -> str:
    return f"{DEV_MODEL_BUCKET}/base-models/{_safe_path_part(base_model)}/{_safe_path_part(revision)}/"


def _safe_path_part(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    resolved = "".join(safe).strip("_")
    if not resolved:
        raise ValueError("safe path part must be non-empty")
    return resolved


def _non_empty(value: str, name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty")
    return resolved
