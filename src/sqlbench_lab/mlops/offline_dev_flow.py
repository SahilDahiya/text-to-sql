"""Local/dev SQL adapter offline flow planning."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlbench_lab.mlops.run_contract import (
    DEV_ENVIRONMENT,
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
    protected: bool = False
    required: bool = True
    min_passed_count: int | None = None
    min_pass_rate: float | None = None

    def to_gate_config(self) -> SQLAdapterEvalGateConfig:
        return SQLAdapterEvalGateConfig(
            label=self.label,
            result_path=self.result_path,
            gate_type=OFFLINE_EVAL_GATE,
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
    environment: str = DEV_ENVIRONMENT,
    dev_min_passed_count: int = 11,
    eval_min_passed_count: int = 12,
    challenge_min_passed_count: int = 22,
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
    )


def validate_offline_flow_plan(plan: SQLAdapterOfflineFlowPlan) -> None:
    """Fail fast if any required local replay artifact is missing."""

    required_paths = [plan.manifest_path, plan.train_summary_path]
    required_paths.extend(spec.result_path for spec in plan.eval_specs)
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

    return build_sql_adapter_run_contract(
        manifest_path=plan.manifest_path,
        environment=plan.environment,
        train_summary_path=plan.train_summary_path,
        eval_gates=tuple(spec.to_gate_config() for spec in plan.eval_specs),
    )


def decide_offline_flow_promotion(plan: SQLAdapterOfflineFlowPlan) -> SQLAdapterPromotionDecision:
    """Decide whether the offline flow artifacts are dev-promotable."""

    contract = build_offline_run_contract(plan)
    return decide_sql_adapter_promotion(
        contract,
        policy=SQLAdapterPromotionPolicy(require_train=True),
    )


def _repo_cli_command(
    *args: str,
    python_executable: str,
) -> tuple[str, ...]:
    return (python_executable, "-m", "sqlbench_lab.cli", *args)
