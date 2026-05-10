"""MLflow logging for SQL training experiments."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT

DEFAULT_MLFLOW_EXPERIMENT = "sqlbench-lab"
MLFLOW_ENABLED_ENV = "SQLBENCH_MLFLOW"
MLFLOW_TRACKING_URI_ENV = "SQLBENCH_MLFLOW_TRACKING_URI"
MLFLOW_EXPERIMENT_ENV = "SQLBENCH_MLFLOW_EXPERIMENT"


def mlflow_enabled(explicit: bool | None = None) -> bool:
    """Return whether MLflow logging was explicitly requested."""

    if explicit is not None:
        return explicit
    return os.environ.get(MLFLOW_ENABLED_ENV, "").lower() in {"1", "true", "yes", "on"}


def log_sql_sft_run(
    *,
    manifest: Any,
    manifest_path: Path,
    summary: Any,
    summary_path: Path,
    train_dataset_counts: dict[str, int],
    smoke_eval_case_count: int,
    training_config: dict[str, Any],
    lora_config: dict[str, Any],
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Log one SQL SFT run to MLflow."""

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow logging requires the observability dependency group. "
            "Run with `uv run --group training --group observability ...`."
        ) from exc

    mlflow.set_tracking_uri(_resolve_tracking_uri(tracking_uri))
    mlflow.set_experiment(
        experiment_name
        or os.environ.get(MLFLOW_EXPERIMENT_ENV)
        or DEFAULT_MLFLOW_EXPERIMENT
    )
    with mlflow.start_run(run_name=manifest.experiment_id):
        mlflow.set_tags(
            {
                "sqlbench.experiment_id": manifest.experiment_id,
                "sqlbench.stage": manifest.training_method.stage,
                "sqlbench.method": manifest.training_method.method,
                "sqlbench.loss_target": manifest.training_method.loss_target,
                "sqlbench.base_model": manifest.student.base_model,
                "sqlbench.adapter_name": manifest.student.adapter_name,
                "sqlbench.git_commit": _git_commit(),
            }
        )
        mlflow.log_params(
            {
                "student.model_family": manifest.student.model_family,
                "student.base_model": manifest.student.base_model,
                "student.adapter_name": manifest.student.adapter_name,
                "train.rows_total": summary.train_row_count,
                "eval.smoke_cases": smoke_eval_case_count,
                "output.adapter_dir": summary.adapter_dir,
                **_prefix_params("train", training_config),
                **_prefix_params("lora", lora_config),
                **_prefix_params("train_dataset_rows", train_dataset_counts),
            }
        )
        mlflow.log_metrics(_numeric_metrics(asdict(summary), prefix="summary"))
        if summary.training_metrics:
            mlflow.log_metrics(_numeric_metrics(summary.training_metrics, prefix="trainer"))
        mlflow.log_artifact(str(manifest_path))
        if summary_path.exists():
            mlflow.log_artifact(str(summary_path))
        adapter_config = Path(summary.adapter_dir) / "adapter_config.json"
        if adapter_config.exists():
            mlflow.log_artifact(str(adapter_config), artifact_path="adapter")


def _resolve_tracking_uri(tracking_uri: str | None) -> str:
    if tracking_uri:
        return tracking_uri
    if os.environ.get(MLFLOW_TRACKING_URI_ENV):
        return str(os.environ[MLFLOW_TRACKING_URI_ENV])
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return str(os.environ["MLFLOW_TRACKING_URI"])
    return f"sqlite:///{WORKSPACE_ROOT / 'mlflow.db'}"


def _prefix_params(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{_safe_key(key)}": _param_value(value) for key, value in values.items()}


def _param_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _safe_key(value: Any) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def _numeric_metrics(values: dict[str, Any], *, prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            metrics[f"{prefix}.{key}"] = float(value)
    return metrics


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=WORKSPACE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"
