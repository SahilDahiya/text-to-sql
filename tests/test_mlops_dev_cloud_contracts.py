from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from sqlbench_lab.cli import main
from sqlbench_lab.mlops import (
    DEV_CLOUD_BUNDLE_SCHEMA_VERSION,
    DEV_COST_CAPACITY_SCHEMA_VERSION,
    DEV_ENDPOINT_MONITORING_SCHEMA_VERSION,
    DEV_ENDPOINT_PLAN_SCHEMA_VERSION,
    DEV_OBSERVABILITY_SCHEMA_VERSION,
    DEV_PROMOTION_REGISTRY_SCHEMA_VERSION,
    DEV_VERTEX_JOB_SCHEMA_VERSION,
    build_dev_cost_capacity_record,
    build_dev_endpoint_monitoring_record,
    build_dev_gcp_vllm_endpoint_plan,
    build_dev_gcs_sync_plan,
    build_dev_observability_record,
    build_dev_promotion_registry_plan,
    build_dev_vertex_training_job_plan,
    build_gcloud_vertex_custom_job_command,
    build_sql_adapter_run_contract,
    decide_sql_adapter_promotion,
)
from sqlbench_lab.mlops.run_contract import ENDPOINT_EVAL_GATE, SQLAdapterEvalGateConfig


class SQLAdapterDevCloudContractTests(unittest.TestCase):
    def test_vertex_training_job_plan_uses_dev_container_and_gcs_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, decision, gcs_plan = _promoted_contract(Path(tmp_dir))

            plan = build_dev_vertex_training_job_plan(
                contract,
                gcs_plan,
                project_id="mistri-467901",
                region="us-central1",
                image_uri="us-central1-docker.pkg.dev/mistri-467901/sqlbench/sqlbench-lab-dev-cli:dev",
            )
            command = build_gcloud_vertex_custom_job_command(plan, config_path="artifacts/dev/vertex_job.yaml")

            self.assertEqual(plan.schema_version, DEV_VERTEX_JOB_SCHEMA_VERSION)
            self.assertEqual(plan.machine.machine_type, "g2-standard-4")
            self.assertEqual(plan.machine.accelerator_type, "NVIDIA_L4")
            self.assertEqual(plan.command, ("python", "-m", "sqlbench_lab.cli"))
            self.assertEqual(plan.args[:3], ("sql", "run-sft", "--manifest"))
            self.assertTrue(plan.manifest_uri.endswith("/manifest.json"))
            self.assertEqual(plan.to_custom_job_spec()["workerPoolSpecs"][0]["containerSpec"]["args"], list(plan.args))
            self.assertEqual(
                command,
                (
                    "gcloud",
                    "ai",
                    "custom-jobs",
                    "create",
                    "--project=mistri-467901",
                    "--region=us-central1",
                    f"--display-name={plan.display_name}",
                    "--config=artifacts/dev/vertex_job.yaml",
                ),
            )

    def test_dev_gcp_vllm_endpoint_plan_names_adapter_and_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, _decision, gcs_plan = _promoted_contract(Path(tmp_dir))

            plan = build_dev_gcp_vllm_endpoint_plan(
                contract,
                gcs_plan,
                project_id="mistri-467901",
                region="us-central1",
                image_uri="us-central1-docker.pkg.dev/mistri-467901/sqlbench/sqlbench-vllm:dev",
            )

            self.assertEqual(plan.schema_version, DEV_ENDPOINT_PLAN_SCHEMA_VERSION)
            self.assertEqual(plan.service_account, "sqlbench-dev-serving-sa")
            self.assertEqual(plan.openai_model, f"{contract.inputs.experiment_id}-dev")
            self.assertIn("--enable-lora", plan.startup_args)
            self.assertIn(f"{contract.inputs.adapter_name}={gcs_plan.adapter_uri}", plan.startup_args)
            self.assertEqual(plan.max_replica_count, 1)
            self.assertEqual(plan.gpu_memory_utilization, 0.75)

    def test_dev_promotion_registry_plan_marks_promoted_adapter_current_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, decision, gcs_plan = _promoted_contract(Path(tmp_dir))

            plan = build_dev_promotion_registry_plan(contract, gcs_plan, decision, db_id="storefront_sales_lab")

            self.assertEqual(plan.schema_version, DEV_PROMOTION_REGISTRY_SCHEMA_VERSION)
            self.assertTrue(plan.eligible_for_current)
            self.assertEqual(plan.decision, "promote")
            self.assertTrue(plan.current_pointer_uri.endswith("/promoted/storefront_sales_lab/dev/current.json"))
            self.assertTrue(plan.rollback_pointer_uri.endswith("/promoted/storefront_sales_lab/dev/rollback.json"))
            self.assertIn(gcs_plan.run_id, plan.adapter_version)

    def test_dev_observability_record_summarizes_train_eval_and_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, decision, gcs_plan = _promoted_contract(Path(tmp_dir))
            registry = build_dev_promotion_registry_plan(contract, gcs_plan, decision, db_id="storefront_sales_lab")

            record = build_dev_observability_record(
                contract,
                decision,
                gcs_plan,
                git_sha="abc123",
                container_image_uri="sqlbench-lab-dev-cli:dev",
                registry_plan=registry,
            )

            self.assertEqual(record.schema_version, DEV_OBSERVABILITY_SCHEMA_VERSION)
            self.assertEqual(record.git_sha, "abc123")
            self.assertEqual(record.train_row_count, 200)
            self.assertEqual(record.decision, "promote")
            self.assertEqual(record.registry_current_pointer_uri, registry.current_pointer_uri)
            self.assertEqual(tuple(item.label for item in record.evals), ("dev_v2", "eval_v1", "challenge_v1", "endpoint_eval"))
            endpoint_eval = record.evals[-1]
            self.assertEqual(endpoint_eval.failed_count, 2)
            self.assertEqual(endpoint_eval.failure_counts, {"prediction_schema_error": 2})

    def test_dev_endpoint_monitoring_record_tracks_quality_latency_and_failure_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, _decision, gcs_plan = _promoted_contract(Path(tmp_dir))
            endpoint = build_dev_gcp_vllm_endpoint_plan(
                contract,
                gcs_plan,
                project_id="mistri-467901",
                region="us-central1",
                image_uri="vllm:dev",
            )

            record = build_dev_endpoint_monitoring_record(contract, endpoint, timeout_count=1, p99_latency_seconds=21.5)

            self.assertEqual(record.schema_version, DEV_ENDPOINT_MONITORING_SCHEMA_VERSION)
            self.assertEqual(record.request_count, 32)
            self.assertEqual(record.success_count, 32)
            self.assertEqual(record.failure_count, 0)
            self.assertEqual(record.timeout_count, 1)
            self.assertEqual(record.p95_latency_seconds, 20.0)
            self.assertEqual(record.p99_latency_seconds, 21.5)
            self.assertEqual(record.endpoint_eval_passed_count, 10)
            self.assertEqual(record.schema_failure_count, 2)

    def test_dev_cost_capacity_record_tracks_gpu_capacity_and_caller_supplied_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, _decision, gcs_plan = _promoted_contract(Path(tmp_dir))
            vertex = build_dev_vertex_training_job_plan(
                contract,
                gcs_plan,
                project_id="mistri-467901",
                region="us-central1",
                image_uri="train:dev",
            )
            endpoint = build_dev_gcp_vllm_endpoint_plan(
                contract,
                gcs_plan,
                project_id="mistri-467901",
                region="us-central1",
                image_uri="vllm:dev",
                max_replica_count=2,
            )

            record = build_dev_cost_capacity_record(
                contract,
                vertex_plan=vertex,
                endpoint_plan=endpoint,
                endpoint_uptime_hours=3.0,
                training_hourly_cost_usd=1.00,
                endpoint_hourly_cost_usd=0.50,
            )

            self.assertEqual(record.schema_version, DEV_COST_CAPACITY_SCHEMA_VERSION)
            self.assertEqual(record.training_machine_type, "g2-standard-4")
            self.assertEqual(record.endpoint_max_replica_count, 2)
            self.assertAlmostEqual(record.training_runtime_hours or 0.0, 0.5)
            self.assertEqual(record.training_estimated_cost_usd, 0.5)
            self.assertEqual(record.endpoint_estimated_cost_usd, 3.0)
            self.assertEqual(record.total_estimated_cost_usd, 3.5)
            self.assertEqual(record.request_count, 32)
            self.assertEqual(record.peak_concurrency, 8)

    def test_cli_writes_dev_cloud_plan_bundle_and_vertex_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12, failure_counts={})
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12, failure_counts={})
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24, failure_counts={})
            endpoint = _write_eval_result(
                root,
                label="endpoint_eval",
                passed=10,
                total=12,
                failure_counts={"prediction_schema_error": 2},
            )
            load = _write_load_test(root, label="vllm_stress_c8_r32", success=32, total=32, concurrency=8)
            output = root / "bundle.json"
            vertex_config = root / "vertex_custom_job.json"
            stdout = StringIO()

            with redirect_stdout(stdout):
                status = main(
                    [
                        "mlops",
                        "dev-cloud-plan",
                        "--manifest",
                        str(manifest),
                        "--train-summary",
                        str(train),
                        "--dev-result",
                        str(dev),
                        "--eval-result",
                        str(protected_eval),
                        "--challenge-result",
                        str(challenge),
                        "--endpoint-eval-result",
                        str(endpoint),
                        "--endpoint-min-passed",
                        "10",
                        "--load-test-result",
                        str(load),
                        "--run-id",
                        "cli-run-123",
                        "--output",
                        str(output),
                        "--vertex-config-output",
                        str(vertex_config),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertIn("wrote dev cloud plan", stdout.getvalue())
            bundle = json.loads(output.read_text(encoding="utf-8"))
            config = json.loads(vertex_config.read_text(encoding="utf-8"))
            self.assertEqual(bundle["schema_version"], DEV_CLOUD_BUNDLE_SCHEMA_VERSION)
            self.assertEqual(bundle["promotion_decision"]["decision"], "promote")
            self.assertEqual(bundle["gcs_sync_plan"]["run_id"], "cli-run-123")
            self.assertEqual(bundle["vertex_training_job_plan"]["schema_version"], DEV_VERTEX_JOB_SCHEMA_VERSION)
            self.assertEqual(bundle["dev_endpoint_plan"]["schema_version"], DEV_ENDPOINT_PLAN_SCHEMA_VERSION)
            self.assertEqual(config["workerPoolSpecs"][0]["containerSpec"]["args"][:3], ["sql", "run-sft", "--manifest"])


def _promoted_contract(root: Path):
    manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
    train = _write_train_summary(root, manifest_id=manifest.stem)
    dev = _write_eval_result(root, label="dev_v2", passed=11, total=12, failure_counts={})
    protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12, failure_counts={})
    challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24, failure_counts={})
    endpoint = _write_eval_result(
        root,
        label="endpoint_eval",
        passed=10,
        total=12,
        failure_counts={"prediction_schema_error": 2},
    )
    load = _write_load_test(root, label="vllm_stress_c8_r32", success=32, total=32, concurrency=8)
    contract = build_sql_adapter_run_contract(
        manifest_path=manifest,
        train_summary_path=train,
        eval_gates=(
            SQLAdapterEvalGateConfig("dev_v2", str(dev), min_passed_count=11),
            SQLAdapterEvalGateConfig("eval_v1", str(protected_eval), protected=True, min_passed_count=12),
            SQLAdapterEvalGateConfig("challenge_v1", str(challenge), min_passed_count=22),
            SQLAdapterEvalGateConfig(
                "endpoint_eval",
                str(endpoint),
                gate_type=ENDPOINT_EVAL_GATE,
                min_passed_count=10,
            ),
        ),
        load_test_paths=(str(load),),
    )
    decision = decide_sql_adapter_promotion(contract)
    gcs_plan = build_dev_gcs_sync_plan(contract, decision, run_id="metaflow-run-123")
    return contract, decision, gcs_plan


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
                "training_metrics": {"train_loss": 0.07, "train_runtime": 1800.0},
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
    failure_counts: dict[str, int],
) -> Path:
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
