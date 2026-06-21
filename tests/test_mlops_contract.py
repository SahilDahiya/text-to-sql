from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sqlbench_lab.mlops import (
    DEV_ENVIRONMENT,
    ENDPOINT_EVAL_GATE,
    PROMOTE_DECISION,
    REJECT_DECISION,
    SQLAdapterEnvironmentConfig,
    SQLAdapterEvalGateConfig,
    SQLAdapterPromotionPolicy,
    build_sql_adapter_run_contract,
    decide_sql_adapter_promotion,
)


class SQLAdapterMLOpsContractTests(unittest.TestCase):
    def test_exp056_shape_promotes_when_required_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id="qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12)
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12)
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24)

            contract = build_sql_adapter_run_contract(
                manifest_path=manifest,
                train_summary_path=train,
                eval_gates=(
                    SQLAdapterEvalGateConfig("dev_v2", str(dev), min_passed_count=11),
                    SQLAdapterEvalGateConfig("eval_v1", str(protected_eval), protected=True, min_passed_count=12),
                    SQLAdapterEvalGateConfig("challenge_v1", str(challenge), min_passed_count=22),
                ),
            )
            decision = decide_sql_adapter_promotion(contract)

            self.assertEqual(decision.decision, PROMOTE_DECISION)
            self.assertEqual(contract.inputs.environment, DEV_ENVIRONMENT)
            self.assertEqual(contract.environment.artifact_bucket, "gs://mistri-sqlbench-dev-artifacts")
            self.assertEqual(contract.inputs.adapter_method, "lora_sft")
            self.assertEqual(contract.train.train_row_count if contract.train else None, 200)
            self.assertIn("eval_v1", decision.passed_gates)
            self.assertEqual(contract.to_json_dict()["inputs"]["experiment_id"], manifest.stem)
            self.assertEqual(contract.to_json_dict()["environment"]["environment"], DEV_ENVIRONMENT)

    def test_contract_fails_for_non_dev_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")

            with self.assertRaisesRegex(ValueError, "unsupported SQL adapter environment"):
                build_sql_adapter_run_contract(manifest_path=manifest, environment="prod")

    def test_contract_fails_when_environment_config_does_not_match_requested_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            config = SQLAdapterEnvironmentConfig(
                environment="staging",
                artifact_bucket="gs://mistri-sqlbench-dev-artifacts",
                dataset_bucket="gs://mistri-sqlbench-dev-datasets",
                model_bucket="gs://mistri-sqlbench-dev-models",
                pipeline_service_account="sqlbench-dev-pipeline-sa",
                training_service_account="sqlbench-dev-train-sa",
                serving_service_account="sqlbench-dev-serving-sa",
                run_artifact_prefix="sql-adapter-runs/dev",
            )

            with self.assertRaisesRegex(ValueError, "environment_config.environment must match"):
                build_sql_adapter_run_contract(
                    manifest_path=manifest,
                    environment=DEV_ENVIRONMENT,
                    environment_config=config,
                )

    def test_exp062_shape_rejects_when_protected_eval_regresses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp062_storefront_v5_lora_r16_a32_d010")
            train = _write_train_summary(
                root,
                manifest_id="qwen35_0_8b__exp062_storefront_v5_lora_r16_a32_d010",
                rows=236,
            )
            dev = _write_eval_result(root, label="dev_v2", passed=12, total=12)
            protected_eval = _write_eval_result(
                root,
                label="eval_v1",
                passed=11,
                total=12,
                failed_case_ids=("storefront_sales_lab_eval_003",),
                failure_counts={"prediction_schema_error": 1},
            )
            challenge = _write_eval_result(root, label="challenge_v2", passed=12, total=15)

            contract = build_sql_adapter_run_contract(
                manifest_path=manifest,
                train_summary_path=train,
                eval_gates=(
                    SQLAdapterEvalGateConfig("dev_v2", str(dev), min_passed_count=12),
                    SQLAdapterEvalGateConfig("eval_v1", str(protected_eval), protected=True, min_passed_count=12),
                    SQLAdapterEvalGateConfig("challenge_v2", str(challenge), min_passed_count=12),
                ),
            )
            decision = decide_sql_adapter_promotion(contract)

            self.assertEqual(decision.decision, REJECT_DECISION)
            self.assertIn("eval_v1", decision.failed_gates)
            self.assertIn("eval_v1 passed_count 11 below required 12", decision.reasons)
            eval_gate = next(gate for gate in contract.eval_gates if gate.label == "eval_v1")
            self.assertEqual(eval_gate.failure_counts, {"prediction_schema_error": 1})
            self.assertEqual(eval_gate.failed_case_ids, ("storefront_sales_lab_eval_003",))

    def test_exp049_shape_rejects_when_endpoint_quality_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(
                root,
                "qwen35_0_8b__exp049_storefront_v3_qlora_r16_a32_d010",
                method="qlora_sft",
            )
            train = _write_train_summary(
                root,
                manifest_id="qwen35_0_8b__exp049_storefront_v3_qlora_r16_a32_d010",
                rows=130,
            )
            protected_eval = _write_eval_result(root, label="eval", passed=9, total=12)
            endpoint_eval = _write_eval_result(root, label="vllm_eval", passed=9, total=12)
            load = _write_load_test(root, success=32, total=32, concurrency=8)

            contract = build_sql_adapter_run_contract(
                manifest_path=manifest,
                train_summary_path=train,
                eval_gates=(
                    SQLAdapterEvalGateConfig("eval", str(protected_eval), protected=True, min_passed_count=10),
                    SQLAdapterEvalGateConfig(
                        "vllm_eval",
                        str(endpoint_eval),
                        gate_type=ENDPOINT_EVAL_GATE,
                        min_passed_count=10,
                    ),
                ),
                load_test_paths=(load,),
            )
            decision = decide_sql_adapter_promotion(
                contract,
                policy=SQLAdapterPromotionPolicy(require_endpoint_eval=True, require_load_test=True),
            )

            self.assertEqual(decision.decision, REJECT_DECISION)
            self.assertEqual(contract.inputs.adapter_method, "qlora_sft")
            load_summary = contract.load_tests[0]
            self.assertEqual(load_summary.timeout_count, 0)
            self.assertEqual(load_summary.generated_char_count_min, 100)
            self.assertEqual(load_summary.generated_char_count_p50, 116)
            self.assertEqual(load_summary.generated_char_count_p95, 129)
            self.assertEqual(load_summary.generated_char_count_max, 131)
            self.assertEqual(load_summary.generated_char_count_mean, 115.5)
            self.assertIn("eval", decision.failed_gates)
            self.assertIn("vllm_eval", decision.failed_gates)
            self.assertIn("vllm_stress_c8_r32", decision.passed_gates)


def _write_manifest(root: Path, experiment_id: str, *, method: str = "lora_sft") -> Path:
    path = root / f"{experiment_id}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "sql_sft_experiment:v1",
                "experiment_id": experiment_id,
                "student": {
                    "model_family": "qwen35",
                    "base_model": "Qwen/Qwen3.5-0.8B-Base",
                    "adapter_name": f"{experiment_id}_adapter",
                },
                "training_method": {
                    "method": method,
                    "loss_target": "assistant_sql_only",
                    "stage": "direct_sql_sft",
                    "notes": None,
                },
                "prompt": {"style": "canonical_chat"},
                "train_inputs": {
                    "train_datasets": ["datasets/sql/train/storefront_sales_lab_train_v4.jsonl"],
                    "validation_datasets": [],
                },
                "eval_plan": {
                    "smoke_dataset": "datasets/sql/eval/storefront_sales_lab_dev_v2.jsonl",
                    "baseline_results": f"results/sql/{experiment_id}/base.json",
                    "post_train_results": f"results/sql/{experiment_id}/adapter.json",
                },
                "output_paths": {
                    "experiment_root": f"artifacts/sql/{experiment_id}",
                    "adapter_dir": f"artifacts/sql/{experiment_id}/adapter",
                    "train_summary_json": f"artifacts/sql/{experiment_id}/train_summary.json",
                    "eval_summary_json": f"artifacts/sql/{experiment_id}/eval_summary.json",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_train_summary(root: Path, *, manifest_id: str, rows: int = 200) -> Path:
    path = root / "train_summary.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": manifest_id,
                "base_model": "Qwen/Qwen3.5-0.8B-Base",
                "adapter_dir": f"artifacts/sql/{manifest_id}/adapter",
                "train_row_count": rows,
                "dry_run": False,
                "trainable_parameters": 6389760,
                "total_parameters": 859375680,
                "training_metrics": {"train_loss": 0.07, "train_runtime": 1234.0},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_eval_result(
    root: Path,
    *,
    label: str,
    passed: int,
    total: int,
    failed_case_ids: tuple[str, ...] = (),
    failure_counts: dict[str, int] | None = None,
) -> Path:
    path = root / f"{label}.json"
    records = []
    for index in range(total):
        failed_case_id = failed_case_ids[index - passed] if index >= passed and index - passed < len(failed_case_ids) else None
        case_id = failed_case_id or f"{label}_{index + 1:03d}"
        records.append(
            {
                "case_id": case_id,
                "task_id": case_id,
                "model_variant": "adapter",
                "predicted_sql": "SELECT 1",
                "passed": index < passed,
                "prediction_error": None if index < passed else "no such column: T3.item_id",
                "gold_error": None,
                "predicted_rows": [[1]] if index < passed else [],
                "gold_rows": [[1]],
            }
        )
    path.write_text(
        json.dumps(
            {
                "experiment_id": "exp",
                "base_model": "Qwen/Qwen3.5-0.8B-Base",
                "model_variant": "adapter",
                "adapter_dir": "artifacts/sql/exp/adapter",
                "eval_dataset": f"datasets/sql/eval/{label}.jsonl",
                "result_path": str(path),
                "case_count": total,
                "passed_count": passed,
                "pass_rate": passed / total,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    if failure_counts is not None:
        path.with_name(f"{path.stem}.analysis.json").write_text(
            json.dumps(
                {
                    "result_path": str(path),
                    "analysis_path": str(path.with_name(f"{path.stem}.analysis.json")),
                    "experiment_id": "exp",
                    "model_variant": "adapter",
                    "eval_dataset": f"datasets/sql/eval/{label}.jsonl",
                    "case_count": total,
                    "passed_count": passed,
                    "failed_count": total - passed,
                    "pass_rate": passed / total,
                    "failure_counts": failure_counts,
                    "tag_slices": [],
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )
    return path


def _write_load_test(root: Path, *, success: int, total: int, concurrency: int) -> Path:
    path = root / f"vllm_stress_c{concurrency}_r{total}.json"
    records = [
        {
            "request_index": index,
            "case_id": f"eval_{index + 1:03d}",
            "success": index < success,
            "latency_seconds": 1.0 + index,
            "generated_char_count": 100 + index if index < success else 0,
            "error": None if index < success else "request failed",
        }
        for index in range(total)
    ]
    path.write_text(
        json.dumps(
            {
                "manifest_path": "experiments/sql/exp.json",
                "experiment_id": "exp",
                "model_variant": "adapter",
                "eval_dataset": "datasets/sql/eval/eval.jsonl",
                "openai_base_url": "http://127.0.0.1:8001",
                "openai_model": "storefront-sql",
                "request_count": total,
                "concurrency": concurrency,
                "max_new_tokens": 128,
                "success_count": success,
                "failure_count": total - success,
                "min_latency_seconds": 1.0,
                "p50_latency_seconds": 10.0,
                "p95_latency_seconds": 20.0,
                "max_latency_seconds": 22.0,
                "requests_per_second": 0.5,
                "result_path": str(path),
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
