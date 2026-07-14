from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sqlbench_lab.sql import (
    audit_sql_mixture,
    build_livesqlbench_artifacts,
    build_next_sql_train_mixture,
    collect_verified_sql_corrections,
    compare_sql_promotion,
    evaluate_postgresql_case,
    evaluate_sqlite_case,
    load_sql_correction_examples,
    load_sql_eval_cases,
    load_sql_sft_manifest,
    load_sql_train_examples,
    run_sql_sft,
    verify_livesqlbench_targets,
)


def _metadata(*, family: str = "filtering", tier: int = 2) -> dict[str, object]:
    return {
        "task_family": family,
        "difficulty": "simple",
        "curriculum_tier": tier,
        "sql_shape": "select_filter",
        "grounding_requirement": "requires_schema_and_database",
        "shortcut_status": "requires_schema_and_database",
        "tags": [family, f"tier_{tier}"],
    }


def _verification(identifier: str = "verify-1") -> dict[str, str]:
    return {
        "status": "execution_verified",
        "verified_by": "test",
        "verification_id": identifier,
        "verified_at": "2026-07-14T00:00:00Z",
    }


def _provenance(task_id: str) -> dict[str, object]:
    return {
        "source_package": "test-package",
        "source_revision": "test-revision",
        "source_task_path": task_id,
        "created_by": "test",
        "teacher_model": None,
        "target_source": "manual_verified",
    }


def _train_row(db_path: Path, task_id: str = "train-1") -> dict[str, object]:
    return {
        "schema_version": "sql_train_example:v2",
        "row_id": f"row::{task_id}",
        "source_benchmark": "livesqlbench",
        "source_split": "train",
        "task_id": task_id,
        "db_id": f"db-{task_id}",
        "db_path": str(db_path),
        "dialect": "sqlite",
        "question": f"question {task_id}",
        "schema_text": "CREATE TABLE items (id INTEGER, value TEXT);",
        "knowledge_text": None,
        "column_value_notes": [],
        "schema_linking_notes": [],
        "target_sql": "SELECT id FROM items ORDER BY id",
        "task_type": "select",
        "metadata": _metadata(),
        "provenance": _provenance(task_id),
        "verification": _verification(task_id),
    }


def _eval_row(db_path: Path, task_id: str = "eval-1") -> dict[str, object]:
    row = _train_row(db_path, task_id)
    return {
        "schema_version": "sql_eval_case:v2",
        "case_id": f"case::{task_id}",
        "source_benchmark": row["source_benchmark"],
        "source_split": "dev",
        "task_id": task_id,
        "fixture_id": task_id,
        "db_id": row["db_id"],
        "db_path": row["db_path"],
        "dialect": row["dialect"],
        "question": row["question"],
        "schema_text": row["schema_text"],
        "knowledge_text": row["knowledge_text"],
        "column_value_notes": row["column_value_notes"],
        "schema_linking_notes": row["schema_linking_notes"],
        "gold_sql": row["target_sql"],
        "task_type": row["task_type"],
        "metadata": row["metadata"],
        "verification": row["verification"],
        "order_sensitive": True,
        "numeric_tolerance": 0.0,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _make_sqlite_db(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items (id INTEGER, value TEXT)")
        connection.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b")])
    return path


def _make_task(package: Path, name: str, database: str) -> None:
    task = package / name
    (task / "tests").mkdir(parents=True)
    (task / "environment" / "documents").mkdir(parents=True)
    (task / "environment" / "db_assets").mkdir(parents=True)
    (task / "task.toml").write_text(
        '[metadata]\ntags = ["postgresql", "livesqlbench", "query"]\ndifficulty = "hard"\ncategory = "text-to-sql"\n',
        encoding="utf-8",
    )
    (task / "tests" / "task_payload.json").write_text(
        json.dumps({"instance_id": name, "selected_database": database, "query": "List item IDs", "sol_sql": [], "test_cases": [], "category": "Query"}),
        encoding="utf-8",
    )
    (task / "environment" / "db_assets" / f"{database}_schema.txt").write_text("CREATE TABLE items (id integer);", encoding="utf-8")
    (task / "environment" / "documents" / "db_env.sh").write_text(
        'export PGHOST="postgresql"\nexport PGPORT="5432"\nexport PGUSER="root"\nexport PGDATABASE="demo"\n',
        encoding="utf-8",
    )


def test_livesqlbench_adapter_requires_explicit_verified_targets(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _make_task(package, "livesqlbench-demo-1", "demo")
    _make_task(package, "livesqlbench-demo-2", "demo2")
    target_manifest = _write_jsonl(
        tmp_path / "targets.jsonl",
        [
            {
                "task_id": "livesqlbench-demo-1",
                "split": "train",
                "target_sql": "SELECT id FROM items",
                "target_source": "manual_verified",
                "verification": _verification("train-verification"),
                "difficulty": "hard",
                "task_type": "select",
                "task_family": "filtering",
                "curriculum_tier": 2,
                "sql_shape": "select_filter",
                "grounding_requirement": "requires_schema_and_database",
                "shortcut_status": "requires_schema_and_database",
                "order_sensitive": False,
                "numeric_tolerance": 0.0,
                "tags": ["filtering"],
            },
            {
                "task_id": "livesqlbench-demo-2",
                "split": "dev",
                "target_sql": "SELECT id FROM items",
                "target_source": "manual_verified",
                "verification": _verification("eval-verification"),
                "difficulty": "hard",
                "task_type": "select",
                "task_family": "joins",
                "curriculum_tier": 3,
                "sql_shape": "select_join",
                "grounding_requirement": "requires_schema_and_database",
                "shortcut_status": "requires_schema_and_database",
                "order_sensitive": False,
                "numeric_tolerance": 0.0,
                "tags": ["joins"],
            },
        ],
    )
    summary = build_livesqlbench_artifacts(
        package_root=package,
        target_manifest=target_manifest,
        source_revision="base-lite-test",
        train_output=tmp_path / "train.jsonl",
        eval_output=tmp_path / "eval.jsonl",
    )
    assert summary.train_row_count == 1
    assert summary.eval_case_count == 1
    assert load_sql_train_examples(summary.train_output)[0].verification.status == "execution_verified"
    assert load_sql_eval_cases(summary.eval_output)[0].dialect == "postgresql"


def test_target_verification_executes_pending_targets_before_import(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _make_task(package, "livesqlbench-demo-1", "demo")
    pending = _write_jsonl(
        tmp_path / "pending.jsonl",
        [{
            "task_id": "livesqlbench-demo-1",
            "split": "train",
            "target_sql": "SELECT id FROM items",
            "target_source": "manual_verified",
            "verification": {"status": "pending"},
            "difficulty": "hard",
            "task_type": "select",
            "task_family": "filtering",
            "curriculum_tier": 2,
            "sql_shape": "select_filter",
            "grounding_requirement": "requires_schema_and_database",
            "shortcut_status": "requires_schema_and_database",
            "order_sensitive": False,
            "numeric_tolerance": 0.0,
            "tags": ["filtering"],
        }],
    )

    class Cursor:
        description = [("id",)]

        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, _sql: str) -> None:
            return None

        def fetchall(self) -> list[tuple[int]]:
            return [(1,)]

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            return None

    verified = verify_livesqlbench_targets(
        package_root=package,
        target_manifest=pending,
        source_revision="base-lite-test",
        verified_output=tmp_path / "verified.jsonl",
        verified_by="test-postgres",
        verified_at="2026-07-14T00:00:00Z",
        postgres_connect=lambda **_kwargs: Connection(),
    )
    assert verified.target_count == 1
    output = json.loads((tmp_path / "verified.jsonl").read_text(encoding="utf-8"))
    assert output["verification"]["status"] == "execution_verified"


def test_target_manifest_does_not_infer_scoring_metadata(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _make_task(package, "livesqlbench-demo-1", "demo")
    target_manifest = _write_jsonl(
        tmp_path / "targets.jsonl",
        [{
            "task_id": "livesqlbench-demo-1",
            "split": "train",
            "target_sql": "SELECT id FROM items",
            "target_source": "manual_verified",
            "verification": _verification("verification"),
            "task_family": "filtering",
            "curriculum_tier": 2,
            "sql_shape": "select_filter",
            "grounding_requirement": "requires_schema_and_database",
            "shortcut_status": "requires_schema_and_database",
            "tags": ["filtering"],
        }],
    )
    with pytest.raises(ValueError, match="difficulty"):
        build_livesqlbench_artifacts(
            package_root=package,
            target_manifest=target_manifest,
            source_revision="base-lite-test",
            train_output=tmp_path / "train.jsonl",
            eval_output=tmp_path / "eval.jsonl",
        )


def test_mixture_audit_and_sqlite_result_equivalence(tmp_path: Path) -> None:
    database = _make_sqlite_db(tmp_path / "items.sqlite")
    train_path = _write_jsonl(tmp_path / "train.jsonl", [_train_row(database)])
    eval_path = _write_jsonl(tmp_path / "eval.jsonl", [_eval_row(database)])
    audit = audit_sql_mixture([train_path])
    assert audit.row_count == 1
    case = load_sql_eval_cases(eval_path)[0]
    result = evaluate_sqlite_case(case, predicted_sql="SELECT id FROM items ORDER BY id")
    assert result.passed is True
    assert result.predicted_columns == ("id",)


def test_postgresql_backend_uses_declared_env_and_result_equivalence(tmp_path: Path) -> None:
    env_file = tmp_path / "db_env.sh"
    env_file.write_text('export PGHOST="localhost"\nexport PGPORT="5432"\nexport PGUSER="root"\nexport PGDATABASE="demo"\n', encoding="utf-8")
    payload = _eval_row(env_file, "postgres-case")
    payload["dialect"] = "postgresql"
    payload["db_path"] = str(env_file)
    case = load_sql_eval_cases(_write_jsonl(tmp_path / "eval.jsonl", [payload]))[0]

    class Cursor:
        description = [("id",)]

        def __init__(self) -> None:
            self.sql = ""

        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, sql: str) -> None:
            self.sql = sql

        def fetchall(self) -> list[tuple[int]]:
            return [(1,), (2,)]

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            return None

    calls: list[dict[str, object]] = []

    def connect(**kwargs: object) -> Connection:
        calls.append(kwargs)
        return Connection()

    result = evaluate_postgresql_case(case, predicted_sql="SELECT id FROM items", postgres_connect=connect)
    assert result.passed is True
    assert calls[0]["host"] == "localhost"


def test_postgresql_connection_failure_is_not_recorded_as_sql_failure(tmp_path: Path) -> None:
    env_file = tmp_path / "db_env.sh"
    env_file.write_text('PGHOST="localhost"\nPGPORT="5432"\nPGUSER="root"\nPGDATABASE="demo"\n', encoding="utf-8")
    payload = _eval_row(env_file, "postgres-failure")
    payload["dialect"] = "postgresql"
    payload["db_path"] = str(env_file)
    case = load_sql_eval_cases(_write_jsonl(tmp_path / "eval.jsonl", [payload]))[0]

    def connect(**_kwargs: object) -> object:
        raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="database unavailable"):
        evaluate_postgresql_case(case, predicted_sql="SELECT id FROM items", postgres_connect=connect)


def test_failure_mining_requires_four_way_review_and_respects_budget(tmp_path: Path) -> None:
    database = _make_sqlite_db(tmp_path / "items.sqlite")
    eval_path = _write_jsonl(tmp_path / "eval.jsonl", [_eval_row(database)])
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"scorer_version": "v1", "records": [{"case_id": "case::eval-1", "task_id": "eval-1", "predicted_sql": "SELECT missing FROM items", "passed": False, "prediction_error": "no such column: missing", "gold_error": None, "predicted_rows": [], "gold_rows": []}]}) + "\n", encoding="utf-8")
    review_path = _write_jsonl(
        tmp_path / "review.jsonl",
        [{"case_id": "case::eval-1", "review_id": "review-1", "scorer_verdict": "correct", "prediction_verdict": "incorrect", "evidence_source": "allowed_eval_case", "reviewer": "human", "verification_id": "verify-eval", "reviewed_at": "2026-07-14T00:00:00Z"}],
    )
    correction_path = tmp_path / "corrections.jsonl"
    summary = collect_verified_sql_corrections(result_path=result_path, eval_dataset=eval_path, review_path=review_path, output_path=correction_path)
    assert summary.collected_count == 1
    assert load_sql_correction_examples(correction_path)[0].failure_type == "prediction_schema_error"
    base_path = _write_jsonl(tmp_path / "base.jsonl", [_train_row(database, "base")])
    output_path = build_next_sql_train_mixture(base_train_datasets=[base_path], correction_dataset=correction_path, output_path=tmp_path / "next.jsonl", max_corrections=1)
    assert len(load_sql_train_examples(output_path)) == 2
    with pytest.raises(ValueError, match="correction budget exceeded"):
        build_next_sql_train_mixture(base_train_datasets=[base_path], correction_dataset=correction_path, output_path=tmp_path / "too-small.jsonl", max_corrections=0)


def test_failure_mining_does_not_skip_unreviewed_failures(tmp_path: Path) -> None:
    database = _make_sqlite_db(tmp_path / "items.sqlite")
    eval_path = _write_jsonl(tmp_path / "eval.jsonl", [_eval_row(database)])
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"scorer_version": "v1", "records": [{"case_id": "case::eval-1", "passed": False}]}) + "\n", encoding="utf-8")
    review_path = _write_jsonl(tmp_path / "review.jsonl", [])
    with pytest.raises(ValueError, match="missing failure review"):
        collect_verified_sql_corrections(
            result_path=result_path,
            eval_dataset=eval_path,
            review_path=review_path,
            output_path=tmp_path / "corrections.jsonl",
        )


def test_promotion_rejects_guardrail_regression(tmp_path: Path) -> None:
    def result(path: Path, dataset: str, rate: float) -> Path:
        path.write_text(json.dumps({"eval_dataset": dataset, "dataset_fingerprint": dataset, "eval_db_ids": [dataset], "db_disjoint_verified": True, "scorer_version": "v1", "generation_config": {"do_sample": False}, "case_count": 2, "pass_rate": rate, "records": [{"case_id": "a"}, {"case_id": "b"}]}) + "\n", encoding="utf-8")
        return path

    decision = compare_sql_promotion(
        base_target=result(tmp_path / "base-target.json", "target", 0.5),
        candidate_target=result(tmp_path / "candidate-target.json", "target", 1.0),
        base_guardrails=[result(tmp_path / "base-guard.json", "guard", 1.0)],
        candidate_guardrails=[result(tmp_path / "candidate-guard.json", "guard", 0.5)],
        target_min_improvement=0.1,
        max_guardrail_regression=0.0,
    )
    assert decision.decision == "reject"


def test_v2_manifest_and_dry_run_enforce_mixture_fingerprint_and_holdout(tmp_path: Path) -> None:
    train_db = _make_sqlite_db(tmp_path / "train.sqlite")
    eval_db = _make_sqlite_db(tmp_path / "eval.sqlite")
    train_path = _write_jsonl(tmp_path / "train.jsonl", [_train_row(train_db, "train")])
    eval_payload = _eval_row(eval_db, "eval")
    eval_payload["gold_sql"] = "SELECT id FROM items"
    eval_path = _write_jsonl(tmp_path / "eval.jsonl", [eval_payload])
    fingerprint = audit_sql_mixture([train_path]).fingerprint
    manifest_payload = {
        "schema_version": "sql_sft_experiment:v2",
        "experiment_id": "exp-test-v2",
        "student": {"model_family": "qwen", "base_model": "Qwen/Qwen2.5-1.5B", "adapter_name": "adapter"},
        "training_method": {"method": "lora_sft", "loss_target": "assistant_sql_only", "stage": "direct_sql_sft", "notes": None},
        "prompt": {"style": "canonical_chat"},
        "mixture": {"dataset_id": "mixture-test", "source_package": "test-package", "source_revision": "test-revision", "fingerprint": fingerprint},
        "train_inputs": {"train_datasets": [str(train_path)]},
        "eval_plan": {"target_dataset": str(eval_path), "guardrail_datasets": [str(eval_path)], "baseline_results": str(tmp_path / "base.json"), "post_train_results": str(tmp_path / "post.json"), "scorer_version": "sql-eval-v2", "require_db_disjoint": True, "target_min_improvement": 0.01, "max_guardrail_regression": 0.0, "max_new_tokens": 128},
        "trainer": {"backend": "transformers_trainer", "num_train_epochs": 1.0, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1, "learning_rate": 0.0002, "logging_steps": 1, "attn_implementation": None, "packing": False, "packing_strategy": "bfd", "max_length": None, "bf16": None, "tf32": None, "gradient_checkpointing": False, "save_strategy": "no", "save_steps": None, "save_total_limit": None, "auto_resume_from_checkpoint": False},
        "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "target_modules": ["q_proj"]},
        "quantization": {"mode": "none", "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16", "device_map": None, "prepare_model_for_kbit_training": False},
        "output_paths": {"experiment_root": str(tmp_path / "experiment"), "adapter_dir": str(tmp_path / "adapter"), "train_summary_json": str(tmp_path / "train-summary.json"), "eval_summary_json": str(tmp_path / "eval-summary.json")},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    assert load_sql_sft_manifest(manifest_path).mixture.fingerprint == fingerprint
    summary = run_sql_sft(manifest_path, dry_run=True)
    assert summary.train_row_count == 1
