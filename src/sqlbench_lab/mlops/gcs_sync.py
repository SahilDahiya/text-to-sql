"""Dev GCS sync plan for SQL adapter MLOps artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlbench_lab.mlops.run_contract import (
    DEV_ENVIRONMENT,
    SQLAdapterPromotionDecision,
    SQLAdapterRunContract,
)

DEV_GCS_SYNC_PLAN_SCHEMA_VERSION = "sql_adapter_gcs_sync_plan:v1"


class SQLAdapterGCSArtifactKind(StrEnum):
    ADAPTER_DIR = "adapter_dir"
    MANIFEST = "manifest"
    TRAIN_SUMMARY = "train_summary"
    EVAL_RESULT = "eval_result"
    EVAL_ANALYSIS = "eval_analysis"
    LOAD_TEST = "load_test"
    RUN_CONTRACT = "run_contract"
    PROMOTION_DECISION = "promotion_decision"


@dataclass(frozen=True)
class SQLAdapterGCSArtifact:
    kind: SQLAdapterGCSArtifactKind
    label: str
    local_path: str
    gcs_uri: str
    required: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SQLAdapterGCSSyncPlan:
    schema_version: str
    environment: str
    experiment_id: str
    run_id: str
    prefix: str
    adapter_uri: str
    run_contract_uri: str
    decision_uri: str
    artifacts: tuple[SQLAdapterGCSArtifact, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dev_gcs_sync_plan(
    contract: SQLAdapterRunContract,
    decision: SQLAdapterPromotionDecision,
    *,
    run_id: str,
) -> SQLAdapterGCSSyncPlan:
    """Build a deterministic dev GCS sync plan without uploading anything."""

    _validate_dev_contract(contract)
    resolved_run_id = _non_empty(run_id, "run_id")
    prefix = _join_gcs_uri(
        contract.environment.artifact_bucket,
        contract.environment.run_artifact_prefix,
        contract.inputs.experiment_id,
        resolved_run_id,
    )
    adapter_uri = _join_gcs_uri(
        contract.environment.model_bucket,
        "adapters",
        contract.inputs.adapter_name,
    ) + "/"
    run_contract_uri = _join_gcs_uri(prefix, "run_contract.json")
    decision_uri = _join_gcs_uri(prefix, "decision.json")
    artifacts = [
        SQLAdapterGCSArtifact(
            kind=SQLAdapterGCSArtifactKind.ADAPTER_DIR,
            label="adapter",
            local_path=contract.inputs.adapter_dir,
            gcs_uri=adapter_uri,
        ),
        SQLAdapterGCSArtifact(
            kind=SQLAdapterGCSArtifactKind.MANIFEST,
            label="manifest",
            local_path=contract.inputs.manifest_path,
            gcs_uri=_join_gcs_uri(prefix, "manifest.json"),
        )
    ]
    if contract.train is not None:
        artifacts.append(
            SQLAdapterGCSArtifact(
                kind=SQLAdapterGCSArtifactKind.TRAIN_SUMMARY,
                label="train",
                local_path=contract.train.path,
                gcs_uri=_join_gcs_uri(prefix, "train", Path(contract.train.path).name),
            )
        )
    for gate in contract.eval_gates:
        artifacts.append(
            SQLAdapterGCSArtifact(
                kind=SQLAdapterGCSArtifactKind.EVAL_RESULT,
                label=gate.label,
                local_path=gate.result_path,
                gcs_uri=_join_gcs_uri(prefix, "eval", gate.label, Path(gate.result_path).name),
            )
        )
        if gate.analysis_path is not None:
            artifacts.append(
                SQLAdapterGCSArtifact(
                    kind=SQLAdapterGCSArtifactKind.EVAL_ANALYSIS,
                    label=gate.label,
                    local_path=gate.analysis_path,
                    gcs_uri=_join_gcs_uri(prefix, "eval", gate.label, Path(gate.analysis_path).name),
                )
            )
    for load in contract.load_tests:
        artifacts.append(
            SQLAdapterGCSArtifact(
                kind=SQLAdapterGCSArtifactKind.LOAD_TEST,
                label=load.label,
                local_path=load.path,
                gcs_uri=_join_gcs_uri(prefix, "load_tests", Path(load.path).name),
            )
        )
    artifacts.extend(
        (
            SQLAdapterGCSArtifact(
                kind=SQLAdapterGCSArtifactKind.RUN_CONTRACT,
                label="run_contract",
                local_path="run_contract.json",
                gcs_uri=run_contract_uri,
            ),
            SQLAdapterGCSArtifact(
                kind=SQLAdapterGCSArtifactKind.PROMOTION_DECISION,
                label=decision.decision,
                local_path="decision.json",
                gcs_uri=decision_uri,
            ),
        )
    )
    return SQLAdapterGCSSyncPlan(
        schema_version=DEV_GCS_SYNC_PLAN_SCHEMA_VERSION,
        environment=contract.environment.environment,
        experiment_id=contract.inputs.experiment_id,
        run_id=resolved_run_id,
        prefix=prefix,
        adapter_uri=adapter_uri,
        run_contract_uri=run_contract_uri,
        decision_uri=decision_uri,
        artifacts=tuple(artifacts),
    )


def _validate_dev_contract(contract: SQLAdapterRunContract) -> None:
    if contract.environment.environment != DEV_ENVIRONMENT:
        raise ValueError(f"dev GCS sync only supports environment={DEV_ENVIRONMENT!r}")
    if not contract.environment.artifact_bucket.startswith("gs://"):
        raise ValueError("dev artifact bucket must be a gs:// URI")
    if not contract.environment.model_bucket.startswith("gs://"):
        raise ValueError("dev model bucket must be a gs:// URI")


def _join_gcs_uri(prefix: str, *parts: str) -> str:
    if not prefix.startswith("gs://"):
        raise ValueError(f"GCS URI prefix must start with gs://: {prefix}")
    stripped_prefix = prefix.rstrip("/")
    stripped_parts = [part.strip("/") for part in parts if part.strip("/")]
    return "/".join((stripped_prefix, *stripped_parts))


def _non_empty(value: str, name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty")
    return resolved
