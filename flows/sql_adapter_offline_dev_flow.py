"""Metaflow entrypoint for the local/dev SQL adapter offline loop."""

from __future__ import annotations

try:
    from metaflow import FlowSpec, Parameter, current, step
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
    build_endpoint_eval_command,
    build_load_test_command,
    build_offline_flow_gcs_sync_plan,
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
    endpoint_eval_result = Parameter("endpoint-eval-result", default="")
    load_test_result = Parameter("load-test-result", default="")
    dev_min_passed = Parameter("dev-min-passed", default=11)
    eval_min_passed = Parameter("eval-min-passed", default=12)
    challenge_min_passed = Parameter("challenge-min-passed", default=22)
    endpoint_min_passed = Parameter("endpoint-min-passed", default=0)
    run_dry_train = Parameter("run-dry-train", default=False)
    run_endpoint_eval = Parameter("run-endpoint-eval", default=False)
    run_load_test = Parameter("run-load-test", default=False)
    endpoint_dataset = Parameter("endpoint-dataset", default="datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl")
    openai_base_url = Parameter("openai-base-url", default="")
    openai_model = Parameter("openai-model", default="")
    endpoint_result_label = Parameter("endpoint-result-label", default="dev_endpoint_eval")
    load_test_output = Parameter("load-test-output", default="")
    load_test_requests = Parameter("load-test-requests", default=8)
    load_test_concurrency = Parameter("load-test-concurrency", default=4)

    @step
    def start(self):
        endpoint_eval_result = str(self.endpoint_eval_result).strip() or None
        load_test_paths = (str(self.load_test_result).strip(),) if str(self.load_test_result).strip() else ()
        endpoint_min_passed_count = int(self.endpoint_min_passed) if int(self.endpoint_min_passed) > 0 else None
        self.plan = build_offline_flow_plan(
            environment=self.environment,
            manifest_path=self.manifest,
            train_summary_path=self.train_summary,
            dev_result_path=self.dev_result,
            eval_result_path=self.eval_result,
            challenge_result_path=self.challenge_result,
            endpoint_eval_result_path=endpoint_eval_result,
            load_test_paths=load_test_paths,
            dev_min_passed_count=int(self.dev_min_passed),
            eval_min_passed_count=int(self.eval_min_passed),
            challenge_min_passed_count=int(self.challenge_min_passed),
            endpoint_min_passed_count=endpoint_min_passed_count,
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
            endpoint_eval_result_path=(
                str(self.plan["endpoint_eval_spec"]["result_path"])
                if self.plan["endpoint_eval_spec"] is not None and not self.run_endpoint_eval
                else None
            ),
            load_test_paths=() if self.run_load_test else tuple(str(path) for path in self.plan["load_test_paths"]),
            dev_min_passed_count=int(self.plan["eval_specs"][0]["min_passed_count"]),
            eval_min_passed_count=int(self.plan["eval_specs"][1]["min_passed_count"]),
            challenge_min_passed_count=int(self.plan["eval_specs"][2]["min_passed_count"]),
            endpoint_min_passed_count=(
                int(self.plan["endpoint_eval_spec"]["min_passed_count"])
                if self.plan["endpoint_eval_spec"] is not None
                and self.plan["endpoint_eval_spec"]["min_passed_count"] is not None
                else None
            ),
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
        self.next(self.start_temp_dev_endpoint)

    @step
    def start_temp_dev_endpoint(self):
        if self.run_endpoint_eval or self.run_load_test:
            if not str(self.openai_base_url).strip() or not str(self.openai_model).strip():
                raise ValueError("--openai-base-url and --openai-model are required for endpoint execution")
            self.temp_endpoint = {
                "mode": "external_openai_compatible_endpoint",
                "openai_base_url": self.openai_base_url,
                "openai_model": self.openai_model,
            }
        else:
            self.temp_endpoint = {"mode": "replay_existing_endpoint_artifacts"}
        self.next(self.wait_for_health)

    @step
    def wait_for_health(self):
        self.endpoint_health = {
            "mode": self.temp_endpoint["mode"],
            "checked": bool(self.run_endpoint_eval or self.run_load_test),
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
        self.next(self.endpoint_eval)

    @step
    def endpoint_eval(self):
        if self.run_endpoint_eval:
            if not str(self.endpoint_eval_result).strip():
                raise ValueError("--endpoint-eval-result is required when --run-endpoint-eval is true")
            self.endpoint_eval_command = build_endpoint_eval_command(
                str(self.plan["manifest_path"]),
                dataset_path=str(self.endpoint_dataset),
                openai_base_url=str(self.openai_base_url),
                openai_model=str(self.openai_model),
                result_label=str(self.endpoint_result_label),
            )
            self.endpoint_eval_result_record = run_repo_cli_command(self.endpoint_eval_command).to_json_dict()
        elif self.plan["endpoint_eval_spec"] is not None:
            self.endpoint_eval_command = build_analyze_eval_command(str(self.plan["endpoint_eval_spec"]["result_path"]))
            self.endpoint_eval_result_record = run_repo_cli_command(self.endpoint_eval_command).to_json_dict()
        else:
            self.endpoint_eval_command = None
            self.endpoint_eval_result_record = {"mode": "endpoint_eval_gate_disabled"}
        self.next(self.load_test)

    @step
    def load_test(self):
        if self.run_load_test:
            output_path = str(self.load_test_output).strip() or str(self.load_test_result).strip()
            if not output_path:
                raise ValueError("--load-test-output or --load-test-result is required when --run-load-test is true")
            self.load_test_command = build_load_test_command(
                str(self.plan["manifest_path"]),
                dataset_path=str(self.endpoint_dataset),
                openai_base_url=str(self.openai_base_url),
                openai_model=str(self.openai_model),
                output_path=output_path,
                request_count=int(self.load_test_requests),
                concurrency=int(self.load_test_concurrency),
            )
            self.load_test_result_record = run_repo_cli_command(self.load_test_command).to_json_dict()
        elif self.plan["load_test_paths"]:
            self.load_test_command = None
            self.load_test_result_record = {
                "mode": "replay_existing_load_test",
                "load_test_paths": self.plan["load_test_paths"],
            }
        else:
            self.load_test_command = None
            self.load_test_result_record = {"mode": "load_test_gate_disabled"}
        self.next(self.stop_temp_dev_endpoint)

    @step
    def stop_temp_dev_endpoint(self):
        self.temp_endpoint_stop = {
            "mode": self.temp_endpoint["mode"],
            "stopped": False,
            "reason": "flow uses external or replay endpoint; no local server process was started",
        }
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
            endpoint_eval_result_path=(
                str(self.plan["endpoint_eval_spec"]["result_path"])
                if self.plan["endpoint_eval_spec"] is not None
                else None
            ),
            load_test_paths=tuple(str(path) for path in self.plan["load_test_paths"]),
            dev_min_passed_count=int(self.plan["eval_specs"][0]["min_passed_count"]),
            eval_min_passed_count=int(self.plan["eval_specs"][1]["min_passed_count"]),
            challenge_min_passed_count=int(self.plan["eval_specs"][2]["min_passed_count"]),
            endpoint_min_passed_count=(
                int(self.plan["endpoint_eval_spec"]["min_passed_count"])
                if self.plan["endpoint_eval_spec"] is not None
                and self.plan["endpoint_eval_spec"]["min_passed_count"] is not None
                else None
            ),
        )
        self.run_contract = build_offline_run_contract(plan).to_json_dict()
        self.promotion_decision = decide_offline_flow_promotion(plan).to_json_dict()
        self.gcs_sync_plan = build_offline_flow_gcs_sync_plan(plan, run_id=str(current.run_id)).to_json_dict()
        self.next(self.end)

    @step
    def end(self):
        print(self.promotion_decision)


if __name__ == "__main__":
    SQLAdapterOfflineDevFlow()
