"""Metaflow entrypoint for the local/dev SQL adapter offline loop."""

from __future__ import annotations

try:
    from metaflow import FlowSpec, Parameter, step
except ImportError as exc:  # pragma: no cover - exercised by the runtime command path.
    raise SystemExit(
        "Metaflow is required for this flow. Run with: "
        "uv run --group mlops python flows/sql_adapter_offline_dev_flow.py run"
    ) from exc

from sqlbench_lab.mlops import (
    DEV_ENVIRONMENT,
    EXP056_CHALLENGE_RESULT_PATH,
    EXP056_DEV_RESULT_PATH,
    EXP056_EVAL_RESULT_PATH,
    EXP056_MANIFEST_PATH,
    EXP056_TRAIN_SUMMARY_PATH,
    build_analyze_eval_command,
    build_offline_flow_plan,
    build_offline_run_contract,
    build_train_command,
    build_validate_manifest_command,
    decide_offline_flow_promotion,
    run_repo_cli_command,
    validate_offline_flow_plan,
)


class SQLAdapterOfflineDevFlow(FlowSpec):
    environment = Parameter("environment", default=DEV_ENVIRONMENT)
    manifest = Parameter("manifest", default=EXP056_MANIFEST_PATH)
    train_summary = Parameter("train-summary", default=EXP056_TRAIN_SUMMARY_PATH)
    dev_result = Parameter("dev-result", default=EXP056_DEV_RESULT_PATH)
    eval_result = Parameter("eval-result", default=EXP056_EVAL_RESULT_PATH)
    challenge_result = Parameter("challenge-result", default=EXP056_CHALLENGE_RESULT_PATH)
    dev_min_passed = Parameter("dev-min-passed", default=11)
    eval_min_passed = Parameter("eval-min-passed", default=12)
    challenge_min_passed = Parameter("challenge-min-passed", default=22)
    run_dry_train = Parameter("run-dry-train", default=False)

    @step
    def start(self):
        self.plan = build_offline_flow_plan(
            environment=self.environment,
            manifest_path=self.manifest,
            train_summary_path=self.train_summary,
            dev_result_path=self.dev_result,
            eval_result_path=self.eval_result,
            challenge_result_path=self.challenge_result,
            dev_min_passed_count=int(self.dev_min_passed),
            eval_min_passed_count=int(self.eval_min_passed),
            challenge_min_passed_count=int(self.challenge_min_passed),
        ).to_json_dict()
        self.next(self.validate_inputs)

    @step
    def validate_inputs(self):
        plan = build_offline_flow_plan(
            environment=str(self.plan["environment"]),
            manifest_path=str(self.plan["manifest_path"]),
            train_summary_path=str(self.plan["train_summary_path"]),
            dev_result_path=str(self.plan["eval_specs"][0]["result_path"]),
            eval_result_path=str(self.plan["eval_specs"][1]["result_path"]),
            challenge_result_path=str(self.plan["eval_specs"][2]["result_path"]),
            dev_min_passed_count=int(self.plan["eval_specs"][0]["min_passed_count"]),
            eval_min_passed_count=int(self.plan["eval_specs"][1]["min_passed_count"]),
            challenge_min_passed_count=int(self.plan["eval_specs"][2]["min_passed_count"]),
        )
        validate_offline_flow_plan(plan)
        self.validate_manifest_command = build_validate_manifest_command(plan.manifest_path)
        self.validate_manifest_result = run_repo_cli_command(self.validate_manifest_command).to_json_dict()
        self.next(self.train_adapter)

    @step
    def train_adapter(self):
        if self.run_dry_train:
            self.train_command = build_train_command(str(self.plan["manifest_path"]), dry_run=True)
            self.train_command_result = run_repo_cli_command(self.train_command).to_json_dict()
        else:
            self.train_command = None
            self.train_command_result = {
                "mode": "replay_existing_train_summary",
                "train_summary_path": self.plan["train_summary_path"],
            }
        self.next(self.eval_dev)

    @step
    def eval_dev(self):
        self.dev_analysis_command = build_analyze_eval_command(str(self.plan["eval_specs"][0]["result_path"]))
        self.dev_analysis_result = run_repo_cli_command(self.dev_analysis_command).to_json_dict()
        self.next(self.eval_eval)

    @step
    def eval_eval(self):
        self.eval_analysis_command = build_analyze_eval_command(str(self.plan["eval_specs"][1]["result_path"]))
        self.eval_analysis_result = run_repo_cli_command(self.eval_analysis_command).to_json_dict()
        self.next(self.eval_challenge)

    @step
    def eval_challenge(self):
        self.challenge_analysis_command = build_analyze_eval_command(str(self.plan["eval_specs"][2]["result_path"]))
        self.challenge_analysis_result = run_repo_cli_command(self.challenge_analysis_command).to_json_dict()
        self.next(self.decide_dev_promote_or_reject)

    @step
    def decide_dev_promote_or_reject(self):
        plan = build_offline_flow_plan(
            environment=str(self.plan["environment"]),
            manifest_path=str(self.plan["manifest_path"]),
            train_summary_path=str(self.plan["train_summary_path"]),
            dev_result_path=str(self.plan["eval_specs"][0]["result_path"]),
            eval_result_path=str(self.plan["eval_specs"][1]["result_path"]),
            challenge_result_path=str(self.plan["eval_specs"][2]["result_path"]),
            dev_min_passed_count=int(self.plan["eval_specs"][0]["min_passed_count"]),
            eval_min_passed_count=int(self.plan["eval_specs"][1]["min_passed_count"]),
            challenge_min_passed_count=int(self.plan["eval_specs"][2]["min_passed_count"]),
        )
        self.run_contract = build_offline_run_contract(plan).to_json_dict()
        self.promotion_decision = decide_offline_flow_promotion(plan).to_json_dict()
        self.next(self.end)

    @step
    def end(self):
        print(self.promotion_decision)


if __name__ == "__main__":
    SQLAdapterOfflineDevFlow()
