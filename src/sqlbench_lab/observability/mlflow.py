"""MLflow logging for SQL experiments."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict
from pathlib import Path
import re
import shutil
import sys
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


def mlflow_tracking_uri(tracking_uri: str | None = None) -> str:
    """Resolve the effective MLflow tracking URI for this workspace."""

    return _resolve_tracking_uri(tracking_uri)


def launch_mlflow_ui(
    *,
    backend_store_uri: str | None = None,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> int:
    """Launch a blocking local MLflow UI process."""

    mlflow_executable = shutil.which("mlflow")
    command = [mlflow_executable] if mlflow_executable else [sys.executable, "-m", "mlflow"]
    command.extend(
        [
            "ui",
            "--backend-store-uri",
            _resolve_tracking_uri(backend_store_uri),
            "--host",
            host,
            "--port",
            str(port),
        ]
    )
    return subprocess.run(command, cwd=WORKSPACE_ROOT, check=False).returncode


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
    train_dataset_names = [_dataset_name(path) for path in train_dataset_counts]
    with mlflow.start_run(run_name=f"{_experiment_label(manifest.experiment_id)}/train"):
        mlflow.set_tags(
            {
                "sqlbench.experiment_id": manifest.experiment_id,
                "sqlbench.run_kind": "train",
                "sqlbench.dataset_name": ",".join(train_dataset_names),
                "sqlbench.dataset_family": ",".join(sorted({_dataset_family(name) for name in train_dataset_names})),
                "sqlbench.model_variant": "adapter",
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
                "mlflow.run_name": f"{_experiment_label(manifest.experiment_id)}/train",
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


def log_sql_eval_run(
    *,
    manifest: Any,
    manifest_path: Path,
    summary: Any,
    result_path: Path,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Log one SQL eval run to MLflow."""

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
    dataset_name = _dataset_name(summary.eval_dataset)
    with mlflow.start_run(
        run_name=f"{_experiment_label(manifest.experiment_id)}/eval/{dataset_name}/{summary.model_variant}"
    ):
        mlflow.set_tags(
            {
                "sqlbench.experiment_id": manifest.experiment_id,
                "sqlbench.run_kind": "eval",
                "sqlbench.dataset_name": dataset_name,
                "sqlbench.dataset_family": _dataset_family(dataset_name),
                "sqlbench.model_variant": summary.model_variant,
                "sqlbench.stage": manifest.training_method.stage,
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
                "eval.dataset": summary.eval_dataset,
                "eval.dataset_name": dataset_name,
                "eval.case_count": summary.case_count,
                "eval.model_variant": summary.model_variant,
                "eval.adapter_dir": summary.adapter_dir or "",
                "mlflow.run_name": (
                    f"{_experiment_label(manifest.experiment_id)}/eval/{dataset_name}/{summary.model_variant}"
                ),
            }
        )
        mlflow.log_metrics(
            {
                "eval.pass_rate": float(summary.pass_rate),
                "eval.passed_count": float(summary.passed_count),
                "eval.case_count": float(summary.case_count),
                **_eval_failure_metrics(summary.records),
            }
        )
        for record in summary.records:
            mlflow.log_metric(
                f"eval.case.{_safe_key(record.case_id)}.passed",
                1.0 if record.passed else 0.0,
            )
        mlflow.log_artifact(str(manifest_path))
        mlflow.log_artifact(str(result_path), artifact_path="eval")


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


def _experiment_label(experiment_id: str) -> str:
    match = re.search(r"exp\d+", experiment_id)
    if match:
        return match.group(0)
    return _safe_key(experiment_id)


def _dataset_name(dataset_path: str | Path) -> str:
    stem = Path(dataset_path).stem
    if stem.startswith("sql_smoke"):
        return "smoke"
    return _safe_key(stem)


def _dataset_family(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    if "bird" in lowered:
        return "bird"
    if "spider" in lowered:
        return "spider"
    if "smoke" in lowered:
        return "smoke"
    return lowered.split("_", maxsplit=1)[0]


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
        if isinstance(value, (int, float)):
            metrics[f"{prefix}.{key}"] = float(value)
    return metrics


def _eval_failure_metrics(records: list[Any]) -> dict[str, float]:
    from sqlbench_lab.sql.eval_analysis import classify_sql_eval_failure

    failure_counts: dict[str, int] = {}
    failed_count = 0
    for record in records:
        if bool(getattr(record, "passed", False)):
            continue
        failed_count += 1
        failure_type = classify_sql_eval_failure(asdict(record))
        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
    metrics = {"eval.failed_count": float(failed_count)}
    metrics.update(
        {
            f"eval.failure.{_safe_key(failure_type)}": float(count)
            for failure_type, count in failure_counts.items()
        }
    )
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
