"""Local/dev SQL adapter offline flow planning."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlbench_lab.mlops.gcs_sync import SQLAdapterGCSSyncPlan, build_dev_gcs_sync_plan
from sqlbench_lab.mlops.run_contract import (
    DEV_ENVIRONMENT,
    ENDPOINT_EVAL_GATE,
    LOAD_TEST_GATE,
    OFFLINE_EVAL_GATE,
    SQLAdapterEvalGateConfig,
    SQLAdapterPromotionDecision,
    SQLAdapterPromotionPolicy,
    SQLAdapterRunContract,
    build_sql_adapter_run_contract,
    decide_sql_adapter_promotion,
)

EXP056_EXPERIMENT_ID = "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010"
EXP056_MANIFEST_PATH = f"experiments/sql/{EXP056_EXPERIMENT_ID}.json"
EXP056_TRAIN_SUMMARY_PATH = f"artifacts/sql/{EXP056_EXPERIMENT_ID}/train_summary.json"
EXP056_RESULT_ROOT = f"results/sql/{EXP056_EXPERIMENT_ID}"
EXP056_DEV_RESULT_PATH = f"{EXP056_RESULT_ROOT}/adapter__storefront_sales_lab_dev_v2__dev_v2.json"
EXP056_EVAL_RESULT_PATH = f"{EXP056_RESULT_ROOT}/adapter__storefront_sales_lab_eval_v1__eval_v1.json"
EXP056_CHALLENGE_RESULT_PATH = f"{EXP056_RESULT_ROOT}/adapter__storefront_sales_lab_challenge_v1__challenge_v1.json"


@dataclass(frozen=True)
class SQLAdapterOfflineEvalSpec:
    label: str
    result_path: str
    gate_type: str = OFFLINE_EVAL_GATE
    protected: bool = False
    required: bool = True
    min_passed_count: int | None = None
    min_pass_rate: float | None = None

    def to_gate_config(self) -> SQLAdapterEvalGateConfig:
        return SQLAdapterEvalGateConfig(
            label=self.label,
            result_path=self.result_path,
            gate_type=self.gate_type,
            protected=self.protected,
            required=self.required,
            min_passed_count=self.min_passed_count,
            min_pass_rate=self.min_pass_rate,
        )


@dataclass(frozen=True)
class SQLAdapterOfflineFlowPlan:
    environment: str
    manifest_path: str
    train_summary_path: str
    eval_specs: tuple[SQLAdapterOfflineEvalSpec, ...]
    endpoint_eval_spec: SQLAdapterOfflineEvalSpec | None = None
    require_endpoint_eval: bool = False
    load_test_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.environment != DEV_ENVIRONMENT:
            raise ValueError(f"offline SQL adapter flow only supports environment={DEV_ENVIRONMENT!r}")
        if not self.eval_specs:
            raise ValueError("offline SQL adapter flow requires at least one eval spec")

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CLICommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def default_exp056_offline_flow_plan() -> SQLAdapterOfflineFlowPlan:
    """Build the default replay plan for the promoted Exp056 adapter."""

    return SQLAdapterOfflineFlowPlan(
        environment=DEV_ENVIRONMENT,
        manifest_path=EXP056_MANIFEST_PATH,
        train_summary_path=EXP056_TRAIN_SUMMARY_PATH,
        eval_specs=(
            SQLAdapterOfflineEvalSpec(
                label="dev_v2",
                result_path=EXP056_DEV_RESULT_PATH,
                min_passed_count=11,
            ),
            SQLAdapterOfflineEvalSpec(
                label="eval_v1",
                result_path=EXP056_EVAL_RESULT_PATH,
                protected=True,
                min_passed_count=12,
            ),
            SQLAdapterOfflineEvalSpec(
                label="challenge_v1",
                result_path=EXP056_CHALLENGE_RESULT_PATH,
                min_passed_count=22,
            ),
        ),
    )


def build_offline_flow_plan(
    *,
    manifest_path: str,
    train_summary_path: str,
    dev_result_path: str,
    eval_result_path: str,
    challenge_result_path: str,
    endpoint_eval_result_path: str | None = None,
    load_test_paths: tuple[str, ...] = (),
    environment: str = DEV_ENVIRONMENT,
    dev_min_passed_count: int = 11,
    eval_min_passed_count: int = 12,
    challenge_min_passed_count: int = 22,
    endpoint_min_passed_count: int | None = None,
) -> SQLAdapterOfflineFlowPlan:
    """Build a dev offline flow plan from explicit artifact paths."""

    return SQLAdapterOfflineFlowPlan(
        environment=environment,
        manifest_path=manifest_path,
        train_summary_path=train_summary_path,
        eval_specs=(
            SQLAdapterOfflineEvalSpec(
                label="dev_v2",
                result_path=dev_result_path,
                min_passed_count=dev_min_passed_count,
            ),
            SQLAdapterOfflineEvalSpec(
                label="eval_v1",
                result_path=eval_result_path,
                protected=True,
                min_passed_count=eval_min_passed_count,
            ),
            SQLAdapterOfflineEvalSpec(
                label="challenge_v1",
                result_path=challenge_result_path,
                min_passed_count=challenge_min_passed_count,
            ),
        ),
        endpoint_eval_spec=(
            SQLAdapterOfflineEvalSpec(
                label="endpoint_eval",
                result_path=endpoint_eval_result_path,
                gate_type=ENDPOINT_EVAL_GATE,
                min_passed_count=endpoint_min_passed_count,
            )
            if endpoint_eval_result_path is not None
            else None
        ),
        require_endpoint_eval=endpoint_eval_result_path is not None or endpoint_min_passed_count is not None,
        load_test_paths=load_test_paths,
    )


def validate_offline_flow_plan(plan: SQLAdapterOfflineFlowPlan) -> None:
    """Fail fast if any required local replay artifact is missing."""

    required_paths = [plan.manifest_path, plan.train_summary_path]
    required_paths.extend(spec.result_path for spec in plan.eval_specs)
    if plan.endpoint_eval_spec is not None:
        required_paths.append(plan.endpoint_eval_spec.result_path)
    required_paths.extend(plan.load_test_paths)
    missing_paths = [path for path in required_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"missing offline flow artifact(s): {', '.join(missing_paths)}")


def build_validate_manifest_command(manifest_path: str, *, python_executable: str = sys.executable) -> tuple[str, ...]:
    return _repo_cli_command(
        "sql",
        "validate-manifest",
        "--manifest",
        manifest_path,
        python_executable=python_executable,
    )


def build_train_command(
    manifest_path: str,
    *,
    dry_run: bool,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    command = [
        *_repo_cli_command(
            "sql",
            "run-sft",
            "--manifest",
            manifest_path,
            python_executable=python_executable,
        )
    ]
    if dry_run:
        command.append("--dry-run")
    return tuple(command)


def build_analyze_eval_command(
    result_path: str,
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    return _repo_cli_command(
        "sql",
        "analyze-eval",
        "--result",
        result_path,
        python_executable=python_executable,
    )


def build_endpoint_eval_command(
    manifest_path: str,
    *,
    dataset_path: str,
    openai_base_url: str,
    openai_model: str,
    result_label: str,
    max_new_tokens: int = 128,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    return _repo_cli_command(
        "sql",
        "eval",
        "--manifest",
        manifest_path,
        "--model",
        "adapter",
        "--dataset",
        dataset_path,
        "--max-new-tokens",
        str(max_new_tokens),
        "--result-label",
        result_label,
        "--openai-base-url",
        openai_base_url,
        "--openai-model",
        openai_model,
        python_executable=python_executable,
    )


def build_load_test_command(
    manifest_path: str,
    *,
    dataset_path: str,
    openai_base_url: str,
    openai_model: str,
    output_path: str,
    request_count: int,
    concurrency: int,
    max_new_tokens: int = 128,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    return _repo_cli_command(
        "sql",
        "openai-load-test",
        "--manifest",
        manifest_path,
        "--model",
        "adapter",
        "--dataset",
        dataset_path,
        "--openai-base-url",
        openai_base_url,
        "--openai-model",
        openai_model,
        "--requests",
        str(request_count),
        "--concurrency",
        str(concurrency),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output",
        output_path,
        python_executable=python_executable,
    )


def run_repo_cli_command(command: tuple[str, ...]) -> CLICommandResult:
    """Run a repo CLI command and fail hard on non-zero exit."""

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result = CLICommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "repo CLI command failed "
            f"returncode={completed.returncode} command={json.dumps(command)} "
            f"stderr={completed.stderr.strip()}"
        )
    return result


def build_offline_run_contract(plan: SQLAdapterOfflineFlowPlan) -> SQLAdapterRunContract:
    """Build the run contract for the offline flow's local artifacts."""

    eval_gates = tuple(spec.to_gate_config() for spec in plan.eval_specs)
    if plan.endpoint_eval_spec is not None:
        eval_gates = (*eval_gates, plan.endpoint_eval_spec.to_gate_config())
    return build_sql_adapter_run_contract(
        manifest_path=plan.manifest_path,
        environment=plan.environment,
        train_summary_path=plan.train_summary_path,
        eval_gates=eval_gates,
        load_test_paths=plan.load_test_paths,
    )


def decide_offline_flow_promotion(plan: SQLAdapterOfflineFlowPlan) -> SQLAdapterPromotionDecision:
    """Decide whether the offline flow artifacts are dev-promotable."""

    contract = build_offline_run_contract(plan)
    return decide_sql_adapter_promotion(
        contract,
        policy=SQLAdapterPromotionPolicy(
            require_train=True,
            require_endpoint_eval=plan.require_endpoint_eval,
            require_load_test=bool(plan.load_test_paths),
        ),
    )


def build_offline_flow_gcs_sync_plan(
    plan: SQLAdapterOfflineFlowPlan,
    *,
    run_id: str,
) -> SQLAdapterGCSSyncPlan:
    """Build the dev GCS sync manifest for the offline flow artifacts."""

    contract = build_offline_run_contract(plan)
    decision = decide_offline_flow_promotion(plan)
    return build_dev_gcs_sync_plan(contract, decision, run_id=run_id)


def _repo_cli_command(
    *args: str,
    python_executable: str,
) -> tuple[str, ...]:
    return (python_executable, "-m", "sqlbench_lab.cli", *args)
