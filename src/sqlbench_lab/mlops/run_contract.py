"""Run contract for production-shaped SQL adapter experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.sql.manifest import load_sql_sft_manifest

DEV_ENVIRONMENT = "dev"
SUPPORTED_ENVIRONMENTS = (DEV_ENVIRONMENT,)

DEV_ARTIFACT_BUCKET = "gs://mistri-sqlbench-dev-artifacts"
DEV_DATASET_BUCKET = "gs://mistri-sqlbench-dev-datasets"
DEV_MODEL_BUCKET = "gs://mistri-sqlbench-dev-models"
DEV_PIPELINE_SERVICE_ACCOUNT = "sqlbench-dev-pipeline-sa"
DEV_TRAINING_SERVICE_ACCOUNT = "sqlbench-dev-train-sa"
DEV_SERVING_SERVICE_ACCOUNT = "sqlbench-dev-serving-sa"
DEV_RUN_ARTIFACT_PREFIX = "sql-adapter-runs/dev"

OFFLINE_EVAL_GATE = "offline_eval"
ENDPOINT_EVAL_GATE = "endpoint_eval"
LOAD_TEST_GATE = "load_test"

PROMOTE_DECISION = "promote"
REJECT_DECISION = "reject"
INVESTIGATE_DECISION = "investigate"
DECISIONS = {PROMOTE_DECISION, REJECT_DECISION, INVESTIGATE_DECISION}


@dataclass(frozen=True)
class SQLAdapterEnvironmentConfig:
    environment: str
    artifact_bucket: str
    dataset_bucket: str
    model_bucket: str
    pipeline_service_account: str
    training_service_account: str
    serving_service_account: str
    run_artifact_prefix: str


@dataclass(frozen=True)
class SQLAdapterRunInputs:
    environment: str
    experiment_id: str
    manifest_path: str
    base_model: str
    adapter_name: str
    adapter_method: str
    train_datasets: tuple[str, ...]
    output_root: str
    adapter_dir: str


@dataclass(frozen=True)
class SQLAdapterTrainSummary:
    path: str
    train_row_count: int
    dry_run: bool
    trainable_parameters: int | None
    total_parameters: int | None
    train_loss: float | None
    train_runtime_seconds: float | None


@dataclass(frozen=True)
class SQLAdapterEvalGateConfig:
    label: str
    result_path: str
    gate_type: str = OFFLINE_EVAL_GATE
    protected: bool = False
    required: bool = True
    min_passed_count: int | None = None
    min_pass_rate: float | None = None


@dataclass(frozen=True)
class SQLAdapterEvalGateSummary:
    label: str
    gate_type: str
    result_path: str
    analysis_path: str | None
    eval_dataset: str
    case_count: int
    passed_count: int
    pass_rate: float
    protected: bool
    required: bool
    min_passed_count: int | None
    min_pass_rate: float | None
    failure_counts: dict[str, int]
    failed_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class SQLAdapterLoadTestSummary:
    label: str
    path: str
    request_count: int
    concurrency: int
    success_count: int
    failure_count: int
    timeout_count: int
    requests_per_second: float
    p50_latency_seconds: float | None
    p95_latency_seconds: float | None
    max_latency_seconds: float | None
    generated_char_count_min: int | None
    generated_char_count_p50: int | None
    generated_char_count_p95: int | None
    generated_char_count_max: int | None
    generated_char_count_mean: float | None
    required: bool = True
    min_success_rate: float = 1.0


@dataclass(frozen=True)
class SQLAdapterRunContract:
    environment: SQLAdapterEnvironmentConfig
    inputs: SQLAdapterRunInputs
    train: SQLAdapterTrainSummary | None
    eval_gates: tuple[SQLAdapterEvalGateSummary, ...]
    load_tests: tuple[SQLAdapterLoadTestSummary, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SQLAdapterPromotionPolicy:
    require_train: bool = True
    require_endpoint_eval: bool = False
    require_load_test: bool = False


@dataclass(frozen=True)
class SQLAdapterPromotionDecision:
    decision: str
    reasons: tuple[str, ...]
    failed_gates: tuple[str, ...]
    passed_gates: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sql_adapter_run_contract(
    *,
    manifest_path: str | Path,
    environment: str = DEV_ENVIRONMENT,
    environment_config: SQLAdapterEnvironmentConfig | None = None,
    train_summary_path: str | Path | None = None,
    eval_gates: tuple[SQLAdapterEvalGateConfig, ...] = (),
    load_test_paths: tuple[str | Path, ...] = (),
) -> SQLAdapterRunContract:
    """Build a durable run contract from existing experiment artifacts."""

    resolved_manifest_path = Path(manifest_path)
    resolved_environment_config = _resolve_environment_config(environment, environment_config)
    manifest = load_sql_sft_manifest(resolved_manifest_path)
    inputs = SQLAdapterRunInputs(
        environment=resolved_environment_config.environment,
        experiment_id=manifest.experiment_id,
        manifest_path=str(resolved_manifest_path),
        base_model=manifest.student.base_model,
        adapter_name=manifest.student.adapter_name,
        adapter_method=manifest.training_method.method,
        train_datasets=tuple(manifest.train_inputs.train_datasets),
        output_root=manifest.output_paths.experiment_root,
        adapter_dir=manifest.output_paths.adapter_dir,
    )
    train = _load_train_summary(train_summary_path) if train_summary_path is not None else None
    return SQLAdapterRunContract(
        environment=resolved_environment_config,
        inputs=inputs,
        train=train,
        eval_gates=tuple(_load_eval_gate(config) for config in eval_gates),
        load_tests=tuple(_load_load_test(path) for path in load_test_paths),
    )


def decide_sql_adapter_promotion(
    contract: SQLAdapterRunContract,
    *,
    policy: SQLAdapterPromotionPolicy = SQLAdapterPromotionPolicy(),
) -> SQLAdapterPromotionDecision:
    """Decide whether a run is promotable from explicit gates."""

    reasons: list[str] = []
    failed_gates: list[str] = []
    passed_gates: list[str] = []
    missing_required_gate = False

    if policy.require_train:
        if contract.train is None:
            missing_required_gate = True
            failed_gates.append("train")
            reasons.append("missing required train summary")
        elif contract.train.dry_run:
            failed_gates.append("train")
            reasons.append("train summary is dry_run=true")
        else:
            passed_gates.append("train")

    required_endpoint_count = sum(
        1 for gate in contract.eval_gates if gate.gate_type == ENDPOINT_EVAL_GATE and gate.required
    )
    if policy.require_endpoint_eval and required_endpoint_count == 0:
        missing_required_gate = True
        failed_gates.append(ENDPOINT_EVAL_GATE)
        reasons.append("missing required endpoint eval gate")

    if policy.require_load_test and not any(load.required for load in contract.load_tests):
        missing_required_gate = True
        failed_gates.append(LOAD_TEST_GATE)
        reasons.append("missing required load-test gate")

    for gate in contract.eval_gates:
        if not gate.required:
            continue
        gate_reasons = _eval_gate_failures(gate)
        if gate_reasons:
            failed_gates.append(gate.label)
            reasons.extend(gate_reasons)
        else:
            passed_gates.append(gate.label)

    for load in contract.load_tests:
        if not load.required:
            continue
        success_rate = load.success_count / load.request_count if load.request_count else 0.0
        if success_rate < load.min_success_rate:
            failed_gates.append(load.label)
            reasons.append(
                f"{load.label} success_rate {success_rate:.4f} below required {load.min_success_rate:.4f}"
            )
        else:
            passed_gates.append(load.label)

    if missing_required_gate:
        return SQLAdapterPromotionDecision(
            decision=INVESTIGATE_DECISION,
            reasons=tuple(reasons),
            failed_gates=tuple(failed_gates),
            passed_gates=tuple(passed_gates),
        )
    if failed_gates:
        return SQLAdapterPromotionDecision(
            decision=REJECT_DECISION,
            reasons=tuple(reasons),
            failed_gates=tuple(failed_gates),
            passed_gates=tuple(passed_gates),
        )
    return SQLAdapterPromotionDecision(
        decision=PROMOTE_DECISION,
        reasons=("all required gates passed",),
        failed_gates=(),
        passed_gates=tuple(passed_gates),
    )


def _resolve_environment_config(
    environment: str,
    environment_config: SQLAdapterEnvironmentConfig | None,
) -> SQLAdapterEnvironmentConfig:
    if environment not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(SUPPORTED_ENVIRONMENTS)
        raise ValueError(f"unsupported SQL adapter environment {environment!r}; supported: {supported}")
    if environment_config is None:
        return _dev_environment_config()
    if environment_config.environment != environment:
        raise ValueError("environment_config.environment must match the requested environment")
    return environment_config


def _dev_environment_config() -> SQLAdapterEnvironmentConfig:
    return SQLAdapterEnvironmentConfig(
        environment=DEV_ENVIRONMENT,
        artifact_bucket=DEV_ARTIFACT_BUCKET,
        dataset_bucket=DEV_DATASET_BUCKET,
        model_bucket=DEV_MODEL_BUCKET,
        pipeline_service_account=DEV_PIPELINE_SERVICE_ACCOUNT,
        training_service_account=DEV_TRAINING_SERVICE_ACCOUNT,
        serving_service_account=DEV_SERVING_SERVICE_ACCOUNT,
        run_artifact_prefix=DEV_RUN_ARTIFACT_PREFIX,
    )


def _load_train_summary(path: str | Path) -> SQLAdapterTrainSummary:
    resolved_path = Path(path)
    payload = _load_json_object(resolved_path)
    metrics = _mapping(payload.get("training_metrics"))
    return SQLAdapterTrainSummary(
        path=str(resolved_path),
        train_row_count=_required_int(payload, "train_row_count"),
        dry_run=bool(payload.get("dry_run")),
        trainable_parameters=_optional_int(payload.get("trainable_parameters")),
        total_parameters=_optional_int(payload.get("total_parameters")),
        train_loss=_optional_float(metrics.get("train_loss")),
        train_runtime_seconds=_optional_float(metrics.get("train_runtime")),
    )


def _load_eval_gate(config: SQLAdapterEvalGateConfig) -> SQLAdapterEvalGateSummary:
    result_path = Path(config.result_path)
    payload = _load_json_object(result_path)
    analysis_path = _matching_analysis_path(result_path)
    analysis = _load_json_object(analysis_path) if analysis_path is not None else {}
    records = _records(payload.get("records"))
    failed_case_ids = tuple(str(record.get("case_id", "")) for record in records if not bool(record.get("passed")))
    return SQLAdapterEvalGateSummary(
        label=config.label,
        gate_type=config.gate_type,
        result_path=str(result_path),
        analysis_path=str(analysis_path) if analysis_path is not None else None,
        eval_dataset=str(payload.get("eval_dataset", "")),
        case_count=_required_int(payload, "case_count"),
        passed_count=_required_int(payload, "passed_count"),
        pass_rate=_required_float(payload, "pass_rate"),
        protected=config.protected,
        required=config.required,
        min_passed_count=config.min_passed_count,
        min_pass_rate=config.min_pass_rate,
        failure_counts={key: int(value) for key, value in _mapping(analysis.get("failure_counts")).items()},
        failed_case_ids=failed_case_ids,
    )


def _load_load_test(path: str | Path) -> SQLAdapterLoadTestSummary:
    resolved_path = Path(path)
    payload = _load_json_object(resolved_path)
    records = _records(payload.get("records"))
    generated_char_counts = _successful_generated_char_counts(records)
    return SQLAdapterLoadTestSummary(
        label=resolved_path.stem,
        path=str(resolved_path),
        request_count=_required_int(payload, "request_count"),
        concurrency=_required_int(payload, "concurrency"),
        success_count=_required_int(payload, "success_count"),
        failure_count=_required_int(payload, "failure_count"),
        timeout_count=_timeout_count(records),
        requests_per_second=_required_float(payload, "requests_per_second"),
        p50_latency_seconds=_optional_float(payload.get("p50_latency_seconds")),
        p95_latency_seconds=_optional_float(payload.get("p95_latency_seconds")),
        max_latency_seconds=_optional_float(payload.get("max_latency_seconds")),
        generated_char_count_min=min(generated_char_counts) if generated_char_counts else None,
        generated_char_count_p50=_percentile_int(generated_char_counts, 0.50),
        generated_char_count_p95=_percentile_int(generated_char_counts, 0.95),
        generated_char_count_max=max(generated_char_counts) if generated_char_counts else None,
        generated_char_count_mean=(
            sum(generated_char_counts) / len(generated_char_counts) if generated_char_counts else None
        ),
    )


def _eval_gate_failures(gate: SQLAdapterEvalGateSummary) -> tuple[str, ...]:
    failures: list[str] = []
    if gate.min_passed_count is not None and gate.passed_count < gate.min_passed_count:
        failures.append(f"{gate.label} passed_count {gate.passed_count} below required {gate.min_passed_count}")
    if gate.min_pass_rate is not None and gate.pass_rate < gate.min_pass_rate:
        failures.append(f"{gate.label} pass_rate {gate.pass_rate:.4f} below required {gate.min_pass_rate:.4f}")
    return tuple(failures)


def _matching_analysis_path(result_path: Path) -> Path | None:
    analysis_path = result_path.with_name(f"{result_path.stem}.analysis.json")
    return analysis_path if analysis_path.exists() else None


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("records must be a list")
    records: list[dict[str, Any]] = []
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("records entries must be objects")
        records.append(record)
    return records


def _successful_generated_char_counts(records: list[dict[str, Any]]) -> list[int]:
    counts: list[int] = []
    for record in records:
        if not bool(record.get("success")):
            continue
        raw_value = record.get("generated_char_count")
        if raw_value is None:
            continue
        count = int(raw_value)
        if count < 0:
            raise ValueError("generated_char_count must be non-negative")
        counts.append(count)
    return counts


def _timeout_count(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        if bool(record.get("success")):
            continue
        error = record.get("error")
        if isinstance(error, str) and "timeout" in error.lower():
            count += 1
    return count


def _percentile_int(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected JSON object value")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("optional integer value must be an integer or null")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError("optional float value must be numeric or null")
    return float(value)
