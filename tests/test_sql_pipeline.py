from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sqlbench_lab.sql import (
    build_repair_messages,
    build_sqlite_fixture,
    build_train_messages,
    evaluate_sqlite_case,
    load_sql_eval_cases,
    load_sql_repair_examples,
    load_sql_sft_manifest,
    load_sql_train_examples,
    run_sql_sft,
    tokenize_sql_sft_messages,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class _FakeSQLTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return [index + 1 for index, _ in enumerate(rendered)]

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": [1000 + index for index, _ in enumerate(text)]}


class SQLPipelineTests(unittest.TestCase):
    def test_load_sql_eval_cases_reads_smoke_dataset(self) -> None:
        cases = load_sql_eval_cases("datasets/sql/smoke/sql_smoke_v1.jsonl")

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].dialect, "sqlite")
        self.assertFalse(cases[0].order_sensitive)
        self.assertEqual(cases[0].numeric_tolerance, 0.000001)

    def test_load_sql_train_examples_reads_seed_dataset(self) -> None:
        rows = load_sql_train_examples("datasets/sql/train/qwen35_0_8b_direct_sql_seed_v1.jsonl")

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0].dialect, "sqlite")
        self.assertEqual(rows[0].target_sql.split()[0], "SELECT")

    def test_load_sql_sft_manifest_validates_seed_experiment(self) -> None:
        manifest = load_sql_sft_manifest("experiments/sql/qwen35_0_8b__exp001_sql_sft.json")

        self.assertEqual(manifest.experiment_id, "qwen35_0_8b__exp001_sql_sft")
        self.assertEqual(manifest.student.base_model, "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(manifest.training_method.stage, "direct_sql_sft")
        self.assertEqual(
            manifest.train_inputs.train_datasets,
            ("datasets/sql/train/qwen35_0_8b_direct_sql_seed_v1.jsonl",),
        )

    def test_run_sql_sft_dry_run_writes_training_summary(self) -> None:
        summary_path = WORKSPACE_ROOT / "artifacts/sql/qwen35_0_8b__exp001_sql_sft/train_summary.json"
        if summary_path.exists():
            summary_path.unlink()

        summary = run_sql_sft("experiments/sql/qwen35_0_8b__exp001_sql_sft.json", dry_run=True)

        self.assertTrue(summary.dry_run)
        self.assertEqual(summary.train_row_count, 5)
        self.assertTrue(summary_path.exists())
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment_id"], "qwen35_0_8b__exp001_sql_sft")
        self.assertTrue(payload["dry_run"])

    def test_tokenize_sql_sft_messages_masks_prompt_tokens(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "SELECT 1;"},
        ]

        encoded = tokenize_sql_sft_messages(tokenizer=_FakeSQLTokenizer(), messages=messages)

        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))
        self.assertIn(-100, encoded["labels"])
        self.assertEqual(encoded["labels"][-1], encoded["input_ids"][-1])

    def test_cli_validates_seed_manifest(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sqlbench_lab.cli",
                "sql",
                "validate-manifest",
                "--manifest",
                "experiments/sql/qwen35_0_8b__exp001_sql_sft.json",
            ],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("qwen35_0_8b__exp001_sql_sft", result.stdout)
        self.assertIn("5 train row(s)", result.stdout)
        self.assertIn("2 smoke case(s)", result.stdout)

    def test_cli_run_sft_dry_run(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sqlbench_lab.cli",
                "sql",
                "run-sft",
                "--manifest",
                "experiments/sql/qwen35_0_8b__exp001_sql_sft.json",
                "--dry-run",
            ],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("dry_run=True", result.stdout)
        self.assertIn("train_rows=5", result.stdout)

    def test_sqlite_fixture_builder_creates_company_small_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "company_small.sqlite"

            build_sqlite_fixture("company_small", db_path)

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("SELECT name FROM departments ORDER BY id").fetchall()

        self.assertEqual(rows, [("Engineering",), ("Sales",), ("Operations",)])

    def test_evaluate_sqlite_case_uses_order_insensitive_result_equivalence(self) -> None:
        case = load_sql_eval_cases("datasets/sql/smoke/sql_smoke_v1.jsonl")[0]

        result = evaluate_sqlite_case(
            case,
            predicted_sql=(
                "SELECT e.name FROM departments d "
                "JOIN employees e ON e.department_id = d.id "
                "WHERE d.name = 'Engineering' ORDER BY e.name DESC"
            ),
        )

        self.assertTrue(result.passed)
        self.assertIsNone(result.prediction_error)
        self.assertEqual(set(result.predicted_rows), {("Ava",), ("Ben",)})

    def test_evaluate_sqlite_case_uses_numeric_tolerance(self) -> None:
        case = load_sql_eval_cases("datasets/sql/smoke/sql_smoke_v1.jsonl")[1]

        result = evaluate_sqlite_case(
            case,
            predicted_sql=(
                "SELECT d.name, AVG(e.salary) + 0.0000001 AS average_salary "
                "FROM departments d JOIN employees e ON e.department_id = d.id "
                "GROUP BY d.name"
            ),
        )

        self.assertTrue(result.passed)

    def test_train_and_repair_loaders_validate_and_render_rows(self) -> None:
        train_row = {
            "schema_version": "sql_train_example:v1",
            "row_id": "train_001",
            "source_benchmark": "smoke",
            "source_split": "train",
            "task_id": "company_small_engineering_names",
            "db_id": "company_small",
            "dialect": "sqlite",
            "question": "List Engineering employees.",
            "schema_text": "CREATE TABLE employees (name TEXT);",
            "knowledge_text": None,
            "target_sql": "SELECT name FROM employees;",
            "task_type": "select",
            "provenance": {
                "created_by": "human",
                "teacher_model": None,
                "source_path": "tests",
            },
            "tags": ["smoke"],
        }
        repair_row = {
            **train_row,
            "schema_version": "sql_repair_example:v1",
            "row_id": "repair_001",
            "previous_sql": "SELECT missing FROM employees;",
            "execution_error": "no such column: missing",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.jsonl"
            repair_path = Path(tmp_dir) / "repair.jsonl"
            train_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            repair_path.write_text(json.dumps(repair_row) + "\n", encoding="utf-8")

            train_examples = load_sql_train_examples(train_path)
            repair_examples = load_sql_repair_examples(repair_path)

        train_messages = build_train_messages(train_examples[0])
        repair_messages = build_repair_messages(repair_examples[0])

        self.assertEqual(train_messages[-1]["content"], "SELECT name FROM employees;")
        self.assertIn("Previous SQL:", repair_messages[1]["content"])
        self.assertIn("no such column: missing", repair_messages[1]["content"])

    def test_load_sql_train_examples_rejects_duplicate_row_ids(self) -> None:
        row = {
            "schema_version": "sql_train_example:v1",
            "row_id": "duplicate",
            "source_benchmark": "smoke",
            "source_split": "train",
            "task_id": "task",
            "db_id": "company_small",
            "dialect": "sqlite",
            "question": "List employees.",
            "schema_text": "CREATE TABLE employees (name TEXT);",
            "knowledge_text": None,
            "target_sql": "SELECT name FROM employees;",
            "task_type": "select",
            "provenance": {
                "created_by": "human",
                "teacher_model": None,
                "source_path": "tests",
            },
            "tags": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate row_id"):
                load_sql_train_examples(path)


if __name__ == "__main__":
    unittest.main()
