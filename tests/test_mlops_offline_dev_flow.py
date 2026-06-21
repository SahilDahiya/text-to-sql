from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sqlbench_lab.mlops import (
    DEV_ENVIRONMENT,
    INVESTIGATE_DECISION,
    PROMOTE_DECISION,
    REJECT_DECISION,
    build_dev_cost_capacity_record,
    build_dev_endpoint_monitoring_record,
    build_dev_gcp_vllm_endpoint_plan,
    build_dev_observability_record,
    build_dev_promotion_registry_plan,
    build_dev_vertex_training_job_plan,
    build_endpoint_eval_command,
    build_load_test_command,
    build_offline_flow_gcs_sync_plan,
    build_offline_flow_plan,
    build_offline_run_contract,
    build_train_command,
    build_validate_manifest_command,
    decide_offline_flow_promotion,
    default_exp056_offline_flow_plan,
    validate_offline_flow_plan,
)


class SQLAdapterOfflineDevFlowTests(unittest.TestCase):
    def test_default_exp056_plan_is_dev_only_and_protects_eval_gate(self) -> None:
        plan = default_exp056_offline_flow_plan()

        self.assertEqual(plan.environment, DEV_ENVIRONMENT)
        self.assertEqual(plan.manifest_path, "experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json")
        self.assertEqual(tuple(spec.label for spec in plan.eval_specs), ("dev_v2", "eval_v1", "challenge_v1"))
        self.assertFalse(plan.eval_specs[0].protected)
        self.assertTrue(plan.eval_specs[1].protected)
        self.assertEqual(plan.eval_specs[1].min_passed_count, 12)

    def test_cli_commands_wrap_repo_cli(self) -> None:
        self.assertEqual(
            build_validate_manifest_command("experiments/sql/exp.json", python_executable="python"),
            (
                "python",
                "-m",
                "sqlbench_lab.cli",
                "sql",
                "validate-manifest",
                "--manifest",
                "experiments/sql/exp.json",
            ),
        )
        self.assertEqual(
            build_train_command("experiments/sql/exp.json", dry_run=True, python_executable="python"),
            (
                "python",
                "-m",
                "sqlbench_lab.cli",
                "sql",
                "run-sft",
                "--manifest",
                "experiments/sql/exp.json",
                "--dry-run",
            ),
        )
        self.assertEqual(
            build_endpoint_eval_command(
                "experiments/sql/exp.json",
                dataset_path="datasets/sql/eval/eval.jsonl",
                openai_base_url="http://127.0.0.1:8000",
                openai_model="storefront-sql",
                result_label="endpoint_eval",
                python_executable="python",
            ),
            (
                "python",
                "-m",
                "sqlbench_lab.cli",
                "sql",
                "eval",
                "--manifest",
                "experiments/sql/exp.json",
                "--model",
                "adapter",
                "--dataset",
                "datasets/sql/eval/eval.jsonl",
                "--max-new-tokens",
                "128",
                "--result-label",
                "endpoint_eval",
                "--openai-base-url",
                "http://127.0.0.1:8000",
                "--openai-model",
                "storefront-sql",
            ),
        )
        self.assertEqual(
            build_load_test_command(
                "experiments/sql/exp.json",
                dataset_path="datasets/sql/eval/eval.jsonl",
                openai_base_url="http://127.0.0.1:8000",
                openai_model="storefront-sql",
                output_path="artifacts/sql/exp/load.json",
                request_count=32,
                concurrency=8,
                python_executable="python",
            ),
            (
                "python",
                "-m",
                "sqlbench_lab.cli",
                "sql",
                "openai-load-test",
                "--manifest",
                "experiments/sql/exp.json",
                "--model",
                "adapter",
                "--dataset",
                "datasets/sql/eval/eval.jsonl",
                "--openai-base-url",
                "http://127.0.0.1:8000",
                "--openai-model",
                "storefront-sql",
                "--requests",
                "32",
                "--concurrency",
                "8",
                "--max-new-tokens",
                "128",
                "--output",
                "artifacts/sql/exp/load.json",
            ),
        )

    def test_plan_fails_fast_when_artifacts_are_missing(self) -> None:
        plan = build_offline_flow_plan(
            manifest_path="missing_manifest.json",
            train_summary_path="missing_train_summary.json",
            dev_result_path="missing_dev.json",
            eval_result_path="missing_eval.json",
            challenge_result_path="missing_challenge.json",
        )

        with self.assertRaisesRegex(FileNotFoundError, "missing offline flow artifact"):
            validate_offline_flow_plan(plan)

    def test_temp_replay_plan_promotes_from_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12)
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12)
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24)
            plan = build_offline_flow_plan(
                manifest_path=str(manifest),
                train_summary_path=str(train),
                dev_result_path=str(dev),
                eval_result_path=str(protected_eval),
                challenge_result_path=str(challenge),
            )

            validate_offline_flow_plan(plan)
            decision = decide_offline_flow_promotion(plan)

            self.assertEqual(decision.decision, PROMOTE_DECISION)
            self.assertEqual(decision.failed_gates, ())
            self.assertIn("eval_v1", decision.passed_gates)

    def test_temp_replay_plan_investigates_when_endpoint_gate_required_but_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12)
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12)
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24)
            plan = build_offline_flow_plan(
                manifest_path=str(manifest),
                train_summary_path=str(train),
                dev_result_path=str(dev),
                eval_result_path=str(protected_eval),
                challenge_result_path=str(challenge),
                endpoint_min_passed_count=12,
            )

            validate_offline_flow_plan(plan)
            decision = decide_offline_flow_promotion(plan)

            self.assertEqual(decision.decision, INVESTIGATE_DECISION)
            self.assertIn("endpoint_eval", decision.failed_gates)
            self.assertIn("missing required endpoint eval gate", decision.reasons)

    def test_temp_replay_plan_promotes_with_endpoint_and_load_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12)
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12)
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24)
            endpoint = _write_eval_result(root, label="endpoint_eval", passed=10, total=12)
            load = _write_load_test(root, label="vllm_stress_c8_r32", success=32, total=32, concurrency=8)
            plan = build_offline_flow_plan(
                manifest_path=str(manifest),
                train_summary_path=str(train),
                dev_result_path=str(dev),
                eval_result_path=str(protected_eval),
                challenge_result_path=str(challenge),
                endpoint_eval_result_path=str(endpoint),
                endpoint_min_passed_count=10,
                load_test_paths=(str(load),),
            )

            validate_offline_flow_plan(plan)
            contract = build_offline_run_contract(plan)
            decision = decide_offline_flow_promotion(plan)

            self.assertEqual(decision.decision, PROMOTE_DECISION)
            self.assertIn("endpoint_eval", decision.passed_gates)
            self.assertIn("vllm_stress_c8_r32", decision.passed_gates)
            self.assertEqual(contract.eval_gates[-1].gate_type, "endpoint_eval")
            self.assertEqual(contract.load_tests[0].success_count, 32)

    def test_temp_replay_plan_builds_dev_cloud_and_hardening_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12)
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12)
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24)
            endpoint = _write_eval_result(root, label="endpoint_eval", passed=10, total=12)
            load = _write_load_test(root, label="vllm_stress_c8_r32", success=32, total=32, concurrency=8)
            plan = build_offline_flow_plan(
                manifest_path=str(manifest),
                train_summary_path=str(train),
                dev_result_path=str(dev),
                eval_result_path=str(protected_eval),
                challenge_result_path=str(challenge),
                endpoint_eval_result_path=str(endpoint),
                endpoint_min_passed_count=10,
                load_test_paths=(str(load),),
            )

            contract = build_offline_run_contract(plan)
            decision = decide_offline_flow_promotion(plan)
            gcs_plan = build_offline_flow_gcs_sync_plan(plan, run_id="metaflow-run-123")
            vertex = build_dev_vertex_training_job_plan(
                contract,
                gcs_plan,
                project_id="mistri-467901",
                region="us-central1",
                image_uri="train:dev",
            )
            endpoint_plan = build_dev_gcp_vllm_endpoint_plan(
                contract,
                gcs_plan,
                project_id="mistri-467901",
                region="us-central1",
                image_uri="vllm:dev",
            )
            registry = build_dev_promotion_registry_plan(
                contract,
                gcs_plan,
                decision,
                db_id="storefront_sales_lab",
            )
            observability = build_dev_observability_record(contract, decision, gcs_plan, registry_plan=registry)
            monitoring = build_dev_endpoint_monitoring_record(contract, endpoint_plan)
            capacity = build_dev_cost_capacity_record(
                contract,
                vertex_plan=vertex,
                endpoint_plan=endpoint_plan,
                endpoint_uptime_hours=1.0,
                training_hourly_cost_usd=1.0,
                endpoint_hourly_cost_usd=1.0,
            )

            self.assertEqual(vertex.environment, DEV_ENVIRONMENT)
            self.assertEqual(endpoint_plan.served_model_name, f"{manifest.stem}-dev")
            self.assertEqual(endpoint_plan.openai_model, f"{manifest.stem}_adapter")
            self.assertTrue(registry.eligible_for_current)
            self.assertEqual(observability.decision, PROMOTE_DECISION)
            self.assertEqual(monitoring.request_count, 32)
            self.assertEqual(capacity.peak_concurrency, 8)

    def test_temp_replay_plan_rejects_when_endpoint_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12)
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12)
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24)
            endpoint = _write_eval_result(root, label="endpoint_eval", passed=9, total=12)
            load = _write_load_test(root, label="vllm_stress_c8_r32", success=32, total=32, concurrency=8)
            plan = build_offline_flow_plan(
                manifest_path=str(manifest),
                train_summary_path=str(train),
                dev_result_path=str(dev),
                eval_result_path=str(protected_eval),
                challenge_result_path=str(challenge),
                endpoint_eval_result_path=str(endpoint),
                endpoint_min_passed_count=10,
                load_test_paths=(str(load),),
            )

            decision = decide_offline_flow_promotion(plan)

            self.assertEqual(decision.decision, REJECT_DECISION)
            self.assertIn("endpoint_eval", decision.failed_gates)
            self.assertIn("endpoint_eval passed_count 9 below required 10", decision.reasons)

    def test_temp_replay_plan_rejects_when_load_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12)
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12)
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24)
            endpoint = _write_eval_result(root, label="endpoint_eval", passed=10, total=12)
            load = _write_load_test(root, label="vllm_stress_c8_r32", success=31, total=32, concurrency=8)
            plan = build_offline_flow_plan(
                manifest_path=str(manifest),
                train_summary_path=str(train),
                dev_result_path=str(dev),
                eval_result_path=str(protected_eval),
                challenge_result_path=str(challenge),
                endpoint_eval_result_path=str(endpoint),
                endpoint_min_passed_count=10,
                load_test_paths=(str(load),),
            )

            decision = decide_offline_flow_promotion(plan)

            self.assertEqual(decision.decision, REJECT_DECISION)
            self.assertIn("vllm_stress_c8_r32", decision.failed_gates)
            self.assertIn("vllm_stress_c8_r32 success_rate 0.9688 below required 1.0000", decision.reasons)

    def test_temp_replay_plan_builds_dev_gcs_sync_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12)
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12)
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24)
            plan = build_offline_flow_plan(
                manifest_path=str(manifest),
                train_summary_path=str(train),
                dev_result_path=str(dev),
                eval_result_path=str(protected_eval),
                challenge_result_path=str(challenge),
            )

            sync_plan = build_offline_flow_gcs_sync_plan(plan, run_id="metaflow-run-456")

            self.assertEqual(sync_plan.run_id, "metaflow-run-456")
            self.assertEqual(sync_plan.experiment_id, manifest.stem)
            self.assertTrue(sync_plan.prefix.endswith(f"/{manifest.stem}/metaflow-run-456"))
            self.assertTrue(sync_plan.decision_uri.endswith("/decision.json"))


def _write_manifest(root: Path, experiment_id: str) -> Path:
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
                    "method": "lora_sft",
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


def _write_train_summary(root: Path, *, manifest_id: str) -> Path:
    path = root / "train_summary.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": manifest_id,
                "base_model": "Qwen/Qwen3.5-0.8B-Base",
                "adapter_dir": f"artifacts/sql/{manifest_id}/adapter",
                "train_row_count": 200,
                "dry_run": False,
                "trainable_parameters": 6389760,
                "total_parameters": 859375680,
                "training_metrics": {"train_loss": 0.07, "train_runtime": 1234.0},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_eval_result(root: Path, *, label: str, passed: int, total: int) -> Path:
    path = root / f"{label}.json"
    records = [
        {
            "case_id": f"{label}_{index + 1:03d}",
            "task_id": f"{label}_{index + 1:03d}",
            "model_variant": "adapter",
            "predicted_sql": "SELECT 1",
            "passed": index < passed,
            "prediction_error": None if index < passed else "no such column: T3.item_id",
            "gold_error": None,
            "predicted_rows": [[1]] if index < passed else [],
            "gold_rows": [[1]],
        }
        for index in range(total)
    ]
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
    return path


def _write_load_test(root: Path, *, label: str, success: int, total: int, concurrency: int) -> Path:
    path = root / f"{label}.json"
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
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
