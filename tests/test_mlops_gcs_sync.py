from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sqlbench_lab.mlops.gcs_sync import (
    DEV_GCS_SYNC_PLAN_SCHEMA_VERSION,
    SQLAdapterGCSArtifactKind,
    build_dev_gcs_sync_plan,
)
from sqlbench_lab.mlops.run_contract import (
    DEV_ARTIFACT_BUCKET,
    DEV_ENVIRONMENT,
    DEV_MODEL_BUCKET,
    SQLAdapterEnvironmentConfig,
    SQLAdapterEvalGateConfig,
    build_sql_adapter_run_contract,
    decide_sql_adapter_promotion,
)


class SQLAdapterGCSSyncTests(unittest.TestCase):
    def test_dev_sync_plan_maps_run_artifacts_to_dev_gcs_uris(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            dev = _write_eval_result(root, label="dev_v2", passed=11, total=12, failure_counts={})
            protected_eval = _write_eval_result(root, label="eval_v1", passed=12, total=12, failure_counts={})
            challenge = _write_eval_result(root, label="challenge_v1", passed=22, total=24, failure_counts={})
            endpoint = _write_eval_result(root, label="endpoint_eval", passed=10, total=12, failure_counts={})
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
                        gate_type="endpoint_eval",
                        min_passed_count=10,
                    ),
                ),
                load_test_paths=(str(load),),
            )
            decision = decide_sql_adapter_promotion(contract)
            plan = build_dev_gcs_sync_plan(contract, decision, run_id="metaflow-run-123")

            self.assertEqual(plan.schema_version, DEV_GCS_SYNC_PLAN_SCHEMA_VERSION)
            self.assertEqual(plan.environment, DEV_ENVIRONMENT)
            self.assertEqual(plan.run_id, "metaflow-run-123")
            self.assertEqual(plan.experiment_id, manifest.stem)
            self.assertTrue(plan.prefix.startswith(f"{DEV_ARTIFACT_BUCKET}/sql-adapter-runs/dev/{manifest.stem}/"))
            self.assertTrue(plan.decision_uri.endswith("/decision.json"))
            self.assertTrue(plan.run_contract_uri.endswith("/run_contract.json"))
            self.assertEqual(plan.adapter_uri, f"{DEV_MODEL_BUCKET}/adapters/{manifest.stem}_adapter/")

            artifact_by_kind = {artifact.kind: artifact for artifact in plan.artifacts}
            self.assertEqual(artifact_by_kind[SQLAdapterGCSArtifactKind.MANIFEST].local_path, str(manifest))
            self.assertTrue(artifact_by_kind[SQLAdapterGCSArtifactKind.MANIFEST].gcs_uri.endswith("/manifest.json"))
            self.assertEqual(artifact_by_kind[SQLAdapterGCSArtifactKind.TRAIN_SUMMARY].local_path, str(train))
            self.assertTrue(artifact_by_kind[SQLAdapterGCSArtifactKind.LOAD_TEST].gcs_uri.endswith("/load_tests/vllm_stress_c8_r32.json"))
            self.assertEqual(
                artifact_by_kind[SQLAdapterGCSArtifactKind.PROMOTION_DECISION].gcs_uri,
                plan.decision_uri,
            )

            analysis_artifacts = [
                artifact for artifact in plan.artifacts if artifact.kind == SQLAdapterGCSArtifactKind.EVAL_ANALYSIS
            ]
            self.assertEqual(len(analysis_artifacts), 4)
            self.assertTrue(all(artifact.required for artifact in plan.artifacts))

    def test_dev_sync_plan_rejects_non_dev_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = _write_manifest(root, "qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010")
            train = _write_train_summary(root, manifest_id=manifest.stem)
            environment_config = SQLAdapterEnvironmentConfig(
                environment="staging",
                artifact_bucket="gs://mistri-sqlbench-staging-artifacts",
                dataset_bucket="gs://mistri-sqlbench-staging-datasets",
                model_bucket="gs://mistri-sqlbench-staging-models",
                pipeline_service_account="sqlbench-staging-pipeline-sa",
                training_service_account="sqlbench-staging-train-sa",
                serving_service_account="sqlbench-staging-serving-sa",
                run_artifact_prefix="sql-adapter-runs/staging",
            )
            dev_contract = build_sql_adapter_run_contract(
                manifest_path=manifest,
                train_summary_path=train,
            )
            contract = replace(dev_contract, environment=environment_config)
            decision = decide_sql_adapter_promotion(contract)

            with self.assertRaisesRegex(ValueError, "dev GCS sync only supports environment"):
                build_dev_gcs_sync_plan(contract, decision, run_id="metaflow-run-123")


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
