from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sqlbench_lab.sql import (
    analyze_sql_eval_result,
    assert_no_sql_dataset_leakage,
    attach_sql_schema_linking,
    attach_sqlite_profile_metadata,
    audit_sql_dataset_leakage,
    build_eval_messages,
    build_repair_messages,
    build_repair_eval_messages,
    build_sqlite_fixture,
    build_train_messages,
    collect_sql_repair_data,
    evaluate_sqlite_case,
    extract_generated_sql,
    filter_sql_train_by_token_budget,
    generate_bird_regional_sales_normalization_micro_lab,
    generate_bird_regional_sales_schema_lab,
    generate_bird_regional_sales_unit_price_contrast_lab,
    generate_bird_superstore_schema_lab,
    import_sql_benchmark,
    load_sql_eval_cases,
    load_sql_repair_examples,
    load_sql_sft_manifest,
    load_sql_train_examples,
    record_sql_prompt_candidate,
    report_sql_prompt_lengths,
    run_sql_candidate_pool_eval,
    run_sql_eval,
    run_sql_eval_with_repair,
    run_sql_sft,
    tokenize_sql_sft_messages,
)
from sqlbench_lab.sql.benchmark_import import _select_raw_rows
from sqlbench_lab.sql.benchmark_import import _filter_raw_rows_by_db_id
from sqlbench_lab.sql.benchmark_import import _exclude_raw_rows_by_db_id
from sqlbench_lab.sql.eval_runner import _render_generation_prompt


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


class _FakeChatTemplateTokenizer:
    chat_template = "fake"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if tokenize:
            raise AssertionError("generation prompt rendering should request text")
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "|assistant:"
        return rendered


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
        self.assertIsNone(rows[0].db_path)
        self.assertEqual(rows[0].target_sql.split()[0], "SELECT")

    def test_record_sql_prompt_candidate_writes_artifact_with_failure_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            eval_dataset = root / "prompt_dev.jsonl"
            eval_dataset.write_text(
                json.dumps(
                    {
                        "schema_version": "sql_eval_case:v1",
                        "case_id": "case_1",
                        "source_benchmark": "smoke",
                        "source_split": "smoke",
                        "task_id": "task_1",
                        "fixture_id": "smoke:tiny",
                        "db_id": "tiny",
                        "db_path": None,
                        "dialect": "sqlite",
                        "question": "How many rows?",
                        "schema_text": "CREATE TABLE items(id INTEGER)",
                        "knowledge_text": None,
                        "column_value_notes": [],
                        "gold_sql": "SELECT COUNT(*) FROM items",
                        "task_type": "select",
                        "order_sensitive": False,
                        "numeric_tolerance": 0.000001,
                        "tags": ["smoke"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prompt_file = root / "candidate_prompt.txt"
            prompt_file.write_text("Return SQLite only.\n", encoding="utf-8")
            eval_result = root / "result.json"
            eval_result.write_text(
                json.dumps(
                    {
                        "experiment_id": "exp_test",
                        "model_variant": "adapter",
                        "eval_dataset": str(eval_dataset),
                        "case_count": 1,
                        "passed_count": 0,
                        "pass_rate": 0.0,
                        "records": [
                            {
                                "case_id": "case_1",
                                "task_id": "task_1",
                                "model_variant": "adapter",
                                "predicted_sql": "SELECT missing FROM items",
                                "passed": False,
                                "prediction_error": "no such column: missing",
                                "gold_error": None,
                                "predicted_rows": [],
                                "gold_rows": [[1]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "candidate.json"

            summary = record_sql_prompt_candidate(
                experiment_id="exp032_test",
                optimizer="mipro_v2",
                candidate_id="c000",
                prompt_file=prompt_file,
                prompt_dev_dataset=eval_dataset,
                model_variant="adapter",
                eval_result=eval_result,
                output_path=output_path,
            )

            self.assertEqual(summary.prompt_dev_case_count, 1)
            self.assertEqual(summary.eval_dataset_role, "prompt_dev")
            self.assertEqual(summary.eval_pass_rate, 0.0)
            self.assertEqual(summary.failure_counts, {"prediction_schema_error": 1})
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "sql_prompt_candidate:v1")
            self.assertEqual(payload["candidate_id"], "c000")
            self.assertEqual(payload["eval_dataset_role"], "prompt_dev")

    def test_attach_sql_schema_linking_adds_train_notes_from_gold_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "train.jsonl"
            output_path = root / "train_linked.jsonl"
            row = {
                "schema_version": "sql_train_example:v1",
                "row_id": "row_1",
                "source_benchmark": "smoke",
                "source_split": "smoke",
                "task_id": "task_1",
                "db_id": "tiny",
                "db_path": None,
                "dialect": "sqlite",
                "question": "How many employees are in Engineering?",
                "schema_text": (
                    "CREATE TABLE employees(id INTEGER, name TEXT, department_id INTEGER)\n"
                    "CREATE TABLE departments(id INTEGER, name TEXT)"
                ),
                "knowledge_text": None,
                "column_value_notes": ["departments.name sample values: Engineering."],
                "target_sql": (
                    "SELECT COUNT(*) FROM employees "
                    "JOIN departments ON departments.id = employees.department_id "
                    "WHERE departments.name = 'Engineering'"
                ),
                "task_type": "select",
                "provenance": {
                    "created_by": "test",
                    "teacher_model": None,
                    "source_path": "inline",
                },
                "tags": ["smoke"],
            }
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = attach_sql_schema_linking(
                input_path=input_path,
                output_path=output_path,
                artifact="train",
                mode="gold_sql",
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            notes = payload["schema_linking_notes"]
            self.assertEqual(summary.row_count, 1)
            self.assertIn("Result shape: count.", notes)
            self.assertTrue(any("employees" in note for note in notes))
            self.assertTrue(any("departments.name" in note for note in notes))

    def test_attach_sql_schema_linking_rejects_gold_sql_eval_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval schema linking must not use gold_sql mode"):
            attach_sql_schema_linking(
                input_path="datasets/sql/smoke/sql_smoke_v1.jsonl",
                output_path="/tmp/unused.jsonl",
                artifact="eval",
                mode="gold_sql",
            )

    def test_import_benchmark_writes_train_db_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            bird_root = cache_root / "premai-io__birdbench"
            train_root = bird_root / "train"
            db_dir = train_root / "train_databases" / "tiny_db"
            db_dir.mkdir(parents=True)
            expected_db_path = str(db_dir / "tiny_db.sqlite")
            with sqlite3.connect(db_dir / "tiny_db.sqlite") as conn:
                conn.execute("CREATE TABLE employees (name TEXT)")
            (train_root / "train.json").write_text(
                json.dumps(
                    [
                        {
                            "db_id": "tiny_db",
                            "question": "List names.",
                            "evidence": "names are in employees.name",
                            "SQL": "SELECT name FROM employees",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_path = Path(tmp_dir) / "bird_train.jsonl"

            summary = import_sql_benchmark(
                benchmark="bird",
                split="train",
                artifact="train",
                output_path=output_path,
                cache_root=cache_root,
            )
            rows = load_sql_train_examples(output_path)

        self.assertEqual(summary.row_count, 1)
        self.assertEqual(summary.selection, "first")
        self.assertEqual(rows[0].db_path, expected_db_path)

    def test_generate_bird_superstore_schema_lab_writes_heldout_dev_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "bird" / "train"
            _write_superstore_lab_fixture(dataset_root)
            train_path = Path(tmp_dir) / "train.jsonl"
            eval_path = Path(tmp_dir) / "eval.jsonl"

            summary = generate_bird_superstore_schema_lab(
                train_output_path=train_path,
                eval_output_path=eval_path,
                dataset_root=dataset_root,
            )
            train_rows = load_sql_train_examples(train_path)
            eval_rows = load_sql_eval_cases(eval_path)

        self.assertEqual(summary.db_id, "superstore")
        self.assertEqual(len(train_rows), 40)
        self.assertEqual(len(eval_rows), 40)
        self.assertEqual({row.db_id for row in train_rows}, {"superstore"})
        self.assertEqual(
            {tag for row in train_rows for tag in row.tags if tag.startswith("region_")},
            {"region_central", "region_east", "region_south", "region_west"},
        )
        self.assertEqual(
            {row.target_sql for row in train_rows} & {case.gold_sql for case in eval_rows},
            set(),
        )

    def test_generate_bird_superstore_schema_lab_v2_adds_direct_fact_computed_order_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "bird" / "train"
            _write_superstore_lab_fixture(dataset_root)
            train_path = Path(tmp_dir) / "train_v2.jsonl"
            eval_path = Path(tmp_dir) / "eval_v2.jsonl"

            summary = generate_bird_superstore_schema_lab(
                train_output_path=train_path,
                eval_output_path=eval_path,
                dataset_root=dataset_root,
                curriculum_version="v2",
            )
            train_rows = load_sql_train_examples(train_path)
            eval_rows = load_sql_eval_cases(eval_path)

        self.assertEqual(summary.train_row_count, 48)
        self.assertEqual(summary.eval_row_count, 40)
        self.assertEqual(len(train_rows), 48)
        self.assertEqual(len(eval_rows), 40)
        self.assertEqual(
            sum("computed_order_by_direct_fact" in row.tags for row in train_rows),
            8,
        )
        self.assertEqual(
            {row.target_sql for row in train_rows} & {case.gold_sql for case in eval_rows},
            set(),
        )

    def test_generate_bird_regional_sales_schema_lab_writes_heldout_dev_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "bird" / "train"
            _write_regional_sales_lab_fixture(dataset_root)
            train_path = Path(tmp_dir) / "regional_sales_train.jsonl"
            eval_path = Path(tmp_dir) / "regional_sales_eval.jsonl"

            summary = generate_bird_regional_sales_schema_lab(
                train_output_path=train_path,
                eval_output_path=eval_path,
                dataset_root=dataset_root,
            )
            train_rows = load_sql_train_examples(train_path)
            eval_rows = load_sql_eval_cases(eval_path)

        self.assertEqual(summary.db_id, "regional_sales")
        self.assertEqual(len(train_rows), 40)
        self.assertEqual(len(eval_rows), 40)
        self.assertEqual({row.db_id for row in train_rows}, {"regional_sales"})
        self.assertEqual(
            {tag for row in train_rows for tag in row.tags if tag.startswith("region_")},
            {"region_midwest", "region_northeast", "region_south", "region_west"},
        )
        self.assertEqual(
            {row.target_sql for row in train_rows} & {case.gold_sql for case in eval_rows},
            set(),
        )

    def test_generate_bird_regional_sales_schema_lab_v2_adds_text_number_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "bird" / "train"
            _write_regional_sales_lab_fixture(dataset_root)
            train_path = Path(tmp_dir) / "regional_sales_train_v2.jsonl"
            eval_path = Path(tmp_dir) / "regional_sales_eval_v2.jsonl"

            summary = generate_bird_regional_sales_schema_lab(
                train_output_path=train_path,
                eval_output_path=eval_path,
                dataset_root=dataset_root,
                curriculum_version="v2",
            )
            train_rows = load_sql_train_examples(train_path)
            eval_rows = load_sql_eval_cases(eval_path)

        self.assertEqual(summary.train_row_count, 52)
        self.assertEqual(summary.eval_row_count, 40)
        self.assertEqual(len(train_rows), 52)
        self.assertEqual(len(eval_rows), 40)
        self.assertEqual(
            sum("text_number_normalization" in row.tags for row in train_rows),
            12,
        )
        self.assertEqual(
            {row.target_sql for row in train_rows} & {case.gold_sql for case in eval_rows},
            set(),
        )

    def test_generate_bird_regional_sales_schema_lab_v3_changes_train_computed_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "bird" / "train"
            _write_regional_sales_lab_fixture(dataset_root)
            train_path = Path(tmp_dir) / "regional_sales_train_v3.jsonl"
            eval_path = Path(tmp_dir) / "regional_sales_eval_v3.jsonl"

            summary = generate_bird_regional_sales_schema_lab(
                train_output_path=train_path,
                eval_output_path=eval_path,
                dataset_root=dataset_root,
                curriculum_version="v3",
            )
            train_rows = load_sql_train_examples(train_path)
            eval_rows = load_sql_eval_cases(eval_path)

        self.assertEqual(summary.train_row_count, 40)
        self.assertEqual(summary.eval_row_count, 40)
        train_computed_rows = [row for row in train_rows if "computed_order_by_direct_fact" in row.tags]
        eval_computed_cases = [case for case in eval_rows if "computed_order_by_direct_fact" in case.tags]
        self.assertEqual(len(train_computed_rows), 4)
        self.assertEqual(len(eval_computed_cases), 4)
        self.assertTrue(
            all("AVG(CAST(REPLACE(T1.`Unit Price`, ',', '') AS REAL))" in row.target_sql for row in train_computed_rows)
        )
        self.assertTrue(all("T1.CurrencyCode = 'USD'" in row.target_sql for row in train_computed_rows))
        self.assertTrue(all("T1.CurrencyCode = 'USD'" not in case.gold_sql for case in eval_computed_cases))
        self.assertEqual(
            {row.target_sql for row in train_rows} & {case.gold_sql for case in eval_rows},
            set(),
        )

    def test_generate_bird_regional_sales_schema_lab_can_add_column_value_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "bird" / "train"
            _write_regional_sales_lab_fixture(dataset_root)
            train_path = Path(tmp_dir) / "regional_sales_train_notes.jsonl"
            eval_path = Path(tmp_dir) / "regional_sales_eval_notes.jsonl"

            summary = generate_bird_regional_sales_schema_lab(
                train_output_path=train_path,
                eval_output_path=eval_path,
                dataset_root=dataset_root,
                include_column_value_notes=True,
            )
            train_rows = load_sql_train_examples(train_path)
            eval_rows = load_sql_eval_cases(eval_path)

        self.assertEqual(summary.db_id, "regional_sales")
        self.assertEqual(len(train_rows), 40)
        self.assertEqual(len(eval_rows), 40)
        self.assertTrue(all(row.column_value_notes for row in train_rows))
        self.assertTrue(all(case.column_value_notes for case in eval_rows))
        self.assertIn("Unit Price", train_rows[0].column_value_notes[0])
        self.assertIn('"1,000.00"', train_rows[0].column_value_notes[0])
        self.assertIn("may include commas", train_rows[0].column_value_notes[0])
        self.assertIn("CAST(REPLACE(T1.`Unit Price`, ',', '') AS REAL)", train_rows[0].column_value_notes[0])
        self.assertTrue(any("T1.`Order Quantity`, not OrderQuantity" in note for note in train_rows[0].column_value_notes))
        self.assertTrue(any("T1.`Sales Channel`, not SalesChannel" in note for note in train_rows[0].column_value_notes))

    def test_generate_bird_regional_sales_normalization_micro_lab_writes_train_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "bird" / "train"
            _write_regional_sales_lab_fixture(dataset_root)
            train_path = Path(tmp_dir) / "regional_sales_normalization_micro_train.jsonl"
            eval_path = Path(tmp_dir) / "regional_sales_eval.jsonl"

            summary = generate_bird_regional_sales_normalization_micro_lab(
                train_output_path=train_path,
                dataset_root=dataset_root,
            )
            generate_bird_regional_sales_schema_lab(
                train_output_path=Path(tmp_dir) / "regional_sales_train.jsonl",
                eval_output_path=eval_path,
                dataset_root=dataset_root,
            )
            train_rows = load_sql_train_examples(train_path)
            eval_rows = load_sql_eval_cases(eval_path)

        self.assertEqual(summary.db_id, "regional_sales")
        self.assertEqual(summary.train_row_count, 16)
        self.assertEqual(len(train_rows), 16)
        self.assertEqual({row.db_id for row in train_rows}, {"regional_sales"})
        self.assertEqual(
            sum("normalization_micro_lab" in row.tags for row in train_rows),
            16,
        )
        self.assertTrue(
            all("CAST(REPLACE(T1.`Unit Price`, ',', '') AS REAL)" in row.target_sql for row in train_rows)
        )
        self.assertTrue(any("T1.`Order Quantity`" in row.target_sql for row in train_rows))
        self.assertEqual(
            {row.target_sql for row in train_rows} & {case.gold_sql for case in eval_rows},
            set(),
        )

    def test_generate_bird_regional_sales_unit_price_contrast_lab_writes_train_only_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "bird" / "train"
            _write_regional_sales_lab_fixture(dataset_root)
            train_path = Path(tmp_dir) / "regional_sales_unit_price_train.jsonl"
            eval_path = Path(tmp_dir) / "regional_sales_eval.jsonl"

            summary = generate_bird_regional_sales_unit_price_contrast_lab(
                train_output_path=train_path,
                dataset_root=dataset_root,
            )
            generate_bird_regional_sales_schema_lab(
                train_output_path=Path(tmp_dir) / "regional_sales_train.jsonl",
                eval_output_path=eval_path,
                dataset_root=dataset_root,
            )
            train_rows = load_sql_train_examples(train_path)
            eval_rows = load_sql_eval_cases(eval_path)

        self.assertEqual(summary.db_id, "regional_sales")
        self.assertEqual(summary.train_row_count, 8)
        self.assertEqual(len(train_rows), 8)
        self.assertEqual({row.db_id for row in train_rows}, {"regional_sales"})
        self.assertEqual(sum("unit_price_target_shape" in row.tags for row in train_rows), 8)
        self.assertTrue(
            all("AVG(CAST(REPLACE(T1.`Unit Price`, ',', '') AS REAL))" in row.target_sql for row in train_rows)
        )
        self.assertEqual(sum(" ASC LIMIT 1" in row.target_sql for row in train_rows), 4)
        self.assertEqual(sum("T1.CurrencyCode = 'USD'" in row.target_sql for row in train_rows), 4)
        self.assertEqual(
            {row.target_sql for row in train_rows} & {case.gold_sql for case in eval_rows},
            set(),
        )

    def test_sql_leakage_audit_allows_same_db_dev_without_exact_overlap(self) -> None:
        train_row = _train_row(
            row_id="train_001",
            task_id="task_train",
            db_id="shared_db",
            question="List names.",
            target_sql="SELECT name FROM people",
        )
        eval_row = _eval_row(
            case_id="eval_001",
            task_id="task_eval",
            db_id="shared_db",
            question="Count names.",
            gold_sql="SELECT COUNT(*) FROM people",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.jsonl"
            eval_path = Path(tmp_dir) / "eval.jsonl"
            train_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            eval_path.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

            summary = audit_sql_dataset_leakage(
                train_paths=[train_path],
                eval_paths=[eval_path],
            )

        self.assertTrue(summary.passed)
        self.assertEqual(summary.overlapping_db_ids, ("shared_db",))

    def test_sql_leakage_audit_rejects_question_and_sql_overlap(self) -> None:
        train_row = _train_row(
            row_id="train_001",
            task_id="task_train",
            db_id="train_db",
            question="List names.",
            target_sql="SELECT name FROM people;",
        )
        eval_row = _eval_row(
            case_id="eval_001",
            task_id="task_eval",
            db_id="eval_db",
            question="  list   names. ",
            gold_sql="select name from people",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.jsonl"
            eval_path = Path(tmp_dir) / "eval.jsonl"
            train_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            eval_path.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "question overlap.*SQL overlap"):
                assert_no_sql_dataset_leakage(
                    train_paths=[train_path],
                    eval_paths=[eval_path],
                )

    def test_sql_leakage_audit_rejects_db_overlap_for_unseen_db_eval(self) -> None:
        train_row = _train_row(
            row_id="train_001",
            task_id="task_train",
            db_id="shared_db",
            question="List names.",
            target_sql="SELECT name FROM people",
        )
        eval_row = _eval_row(
            case_id="eval_001",
            task_id="task_eval",
            db_id="shared_db",
            question="Count names.",
            gold_sql="SELECT COUNT(*) FROM people",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.jsonl"
            eval_path = Path(tmp_dir) / "eval.jsonl"
            train_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            eval_path.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "db_id overlap: shared_db"):
                assert_no_sql_dataset_leakage(
                    train_paths=[train_path],
                    eval_paths=[eval_path],
                    require_db_disjoint=True,
                )

    def test_stratified_selection_round_robins_by_database(self) -> None:
        raw_rows = [
            {"db_id": "a", "question": "a1"},
            {"db_id": "a", "question": "a2"},
            {"db_id": "a", "question": "a3"},
            {"db_id": "b", "question": "b1"},
            {"db_id": "b", "question": "b2"},
            {"db_id": "c", "question": "c1"},
        ]

        selected = _select_raw_rows(raw_rows, limit=5, selection="stratified")

        self.assertEqual([row["question"] for _, row in selected], ["a1", "b1", "c1", "a2", "b2"])

    def test_filter_raw_rows_by_db_id_restricts_import_rows(self) -> None:
        raw_rows = [
            {"db_id": "restaurant", "question": "r1"},
            {"db_id": "restaurant", "question": "r2"},
            {"db_id": "airline", "question": "a1"},
            {"db_id": "sales", "question": "s1"},
        ]

        filtered = _filter_raw_rows_by_db_id(raw_rows, db_ids=("restaurant", "airline"))

        self.assertEqual([row["question"] for row in filtered], ["r1", "r2", "a1"])

    def test_filter_raw_rows_by_db_id_rejects_empty_match(self) -> None:
        raw_rows = [{"db_id": "restaurant", "question": "r1"}]

        with self.assertRaisesRegex(ValueError, "no benchmark rows found"):
            _filter_raw_rows_by_db_id(raw_rows, db_ids=("missing_db",))

    def test_exclude_raw_rows_by_db_id_removes_unseen_holdout_dbs(self) -> None:
        raw_rows = [
            {"db_id": "restaurant", "question": "r1"},
            {"db_id": "airline", "question": "a1"},
            {"db_id": "sales", "question": "s1"},
            {"db_id": "bike_share_1", "question": "b1"},
        ]

        remaining = _exclude_raw_rows_by_db_id(
            raw_rows,
            exclude_db_ids=("restaurant", "airline"),
        )

        self.assertEqual([row["question"] for row in remaining], ["s1", "b1"])

    def test_exclude_raw_rows_by_db_id_rejects_empty_remaining_set(self) -> None:
        raw_rows = [{"db_id": "restaurant", "question": "r1"}]

        with self.assertRaisesRegex(ValueError, "exclusion removed every benchmark row"):
            _exclude_raw_rows_by_db_id(raw_rows, exclude_db_ids=("restaurant",))

    def test_import_benchmark_preserves_source_task_ids_after_db_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "cache"
            bird_root = cache_root / "premai-io__birdbench"
            train_root = bird_root / "train"
            for db_id in ("skip_db", "target_db"):
                db_dir = train_root / "train_databases" / db_id
                db_dir.mkdir(parents=True)
                with sqlite3.connect(db_dir / f"{db_id}.sqlite") as conn:
                    conn.execute("CREATE TABLE employees (name TEXT)")
            (train_root / "train.json").write_text(
                json.dumps(
                    [
                        {
                            "db_id": "skip_db",
                            "question": "Skip names.",
                            "SQL": "SELECT name FROM employees",
                        },
                        {
                            "db_id": "target_db",
                            "question": "Target names.",
                            "SQL": "SELECT name FROM employees",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            output_path = Path(tmp_dir) / "bird_train.jsonl"

            import_sql_benchmark(
                benchmark="bird",
                split="train",
                artifact="train",
                output_path=output_path,
                cache_root=cache_root,
                db_ids=("target_db",),
            )
            rows = load_sql_train_examples(output_path)

        self.assertEqual(rows[0].row_id, "bird_train_00002")
        self.assertEqual(rows[0].task_id, "bird_train_00002")

    def test_filter_sql_train_by_token_budget_keeps_rows_with_rendered_prompt_under_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "train.jsonl"
            output_path = Path(tmp_dir) / "train_filtered.jsonl"
            rows = [
                _train_row(
                    row_id="short",
                    task_id="short",
                    db_id="db",
                    question="List names.",
                    target_sql="SELECT name FROM people",
                ),
                _train_row(
                    row_id="long",
                    task_id="long",
                    db_id="db",
                    question="List names. " * 200,
                    target_sql="SELECT name FROM people",
                ),
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            summary = filter_sql_train_by_token_budget(
                input_path=input_path,
                output_path=output_path,
                base_model="fake",
                prompt_style="canonical_chat",
                max_tokens=500,
                tokenizer_like=_FakeSQLTokenizer(),
            )
            filtered = load_sql_train_examples(output_path)

        self.assertEqual(summary.input_row_count, 2)
        self.assertEqual(summary.kept_row_count, 1)
        self.assertEqual(summary.rejected_row_count, 1)
        self.assertEqual(filtered[0].row_id, "short")

    def test_load_sql_sft_manifest_validates_seed_experiment(self) -> None:
        manifest = load_sql_sft_manifest("experiments/sql/qwen35_0_8b__exp001_sql_sft.json")

        self.assertEqual(manifest.experiment_id, "qwen35_0_8b__exp001_sql_sft")
        self.assertEqual(manifest.student.base_model, "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(manifest.training_method.stage, "direct_sql_sft")
        self.assertEqual(manifest.trainer.backend, "transformers_trainer")
        self.assertIsNone(manifest.trainer.attn_implementation)
        self.assertFalse(manifest.trainer.packing)
        self.assertEqual(manifest.trainer.packing_strategy, "bfd")
        self.assertIsNone(manifest.trainer.max_length)
        self.assertIsNone(manifest.trainer.bf16)
        self.assertIsNone(manifest.trainer.tf32)
        self.assertFalse(manifest.trainer.gradient_checkpointing)
        self.assertEqual(manifest.trainer.save_strategy, "no")
        self.assertIsNone(manifest.trainer.save_steps)
        self.assertIsNone(manifest.trainer.save_total_limit)
        self.assertFalse(manifest.trainer.auto_resume_from_checkpoint)
        self.assertEqual(
            manifest.train_inputs.train_datasets,
            ("datasets/sql/train/qwen35_0_8b_direct_sql_seed_v1.jsonl",),
        )

    def test_load_sql_sft_manifest_allows_qwen_coder_model_family(self) -> None:
        manifest = load_sql_sft_manifest(
            "experiments/sql/qwen25_coder_1_5b__exp037_trl_bird_full_no_unseen_gates.json"
        )

        self.assertEqual(manifest.student.model_family, "qwen25-coder")
        self.assertEqual(manifest.student.base_model, "Qwen/Qwen2.5-Coder-1.5B")

    def test_load_sql_sft_manifest_reads_trl_backend(self) -> None:
        manifest = load_sql_sft_manifest(
            "experiments/sql/qwen35_0_8b__exp007_trl_sft_identifier_copy.json"
        )

        self.assertEqual(manifest.trainer.backend, "trl_sft_trainer")
        self.assertEqual(manifest.trainer.logging_steps, 25)

    def test_load_sql_sft_manifest_reads_trl_packing_options(self) -> None:
        manifest = load_sql_sft_manifest(
            "experiments/sql/qwen35_0_8b__exp008_trl_packing_identifier_copy.json"
        )

        self.assertEqual(manifest.trainer.backend, "trl_sft_trainer")
        self.assertIsNone(manifest.trainer.attn_implementation)
        self.assertTrue(manifest.trainer.packing)
        self.assertEqual(manifest.trainer.packing_strategy, "bfd")
        self.assertEqual(manifest.trainer.max_length, 1024)
        self.assertTrue(manifest.trainer.bf16)
        self.assertFalse(manifest.trainer.tf32)
        self.assertFalse(manifest.trainer.gradient_checkpointing)

    def test_load_sql_sft_manifest_reads_attention_implementation(self) -> None:
        manifest = load_sql_sft_manifest(
            "experiments/sql/qwen35_0_8b__exp009_trl_packing_flash_attention_identifier_copy.json"
        )

        self.assertEqual(manifest.trainer.attn_implementation, "kernels-community/flash-attn2")

    def test_load_sql_sft_manifest_reads_checkpoint_options(self) -> None:
        manifest = load_sql_sft_manifest(
            "experiments/sql/qwen35_0_8b__exp036_trl_bird_full_no_unseen_gates.json"
        )

        self.assertEqual(manifest.trainer.save_strategy, "steps")
        self.assertEqual(manifest.trainer.save_steps, 100)
        self.assertEqual(manifest.trainer.save_total_limit, 2)
        self.assertTrue(manifest.trainer.auto_resume_from_checkpoint)

    def test_load_sql_sft_manifest_defaults_prompt_style(self) -> None:
        manifest = load_sql_sft_manifest("experiments/sql/qwen35_0_8b__exp001_sql_sft.json")

        self.assertEqual(manifest.prompt.style, "canonical_chat")

    def test_run_sql_sft_dry_run_writes_training_summary(self) -> None:
        summary_path = WORKSPACE_ROOT / "artifacts/sql/qwen35_0_8b__exp001_sql_sft/train_summary.json"
        dry_summary_path = WORKSPACE_ROOT / "artifacts/sql/qwen35_0_8b__exp001_sql_sft/train_summary.dry_run.json"
        if summary_path.exists():
            summary_path.unlink()
        if dry_summary_path.exists():
            dry_summary_path.unlink()

        summary = run_sql_sft("experiments/sql/qwen35_0_8b__exp001_sql_sft.json", dry_run=True)

        self.assertTrue(summary.dry_run)
        self.assertEqual(summary.train_row_count, 5)
        self.assertFalse(summary_path.exists())
        self.assertTrue(dry_summary_path.exists())
        payload = json.loads(dry_summary_path.read_text(encoding="utf-8"))
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

    def test_cli_validates_repair_dataset(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sqlbench_lab.cli",
                "sql",
                "validate-repair",
                "--dataset",
                "datasets/sql/repair/bird_dev_repair_seed_v1.jsonl",
            ],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("validated SQL repair dataset with 2 row(s)", result.stdout)

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

    def test_run_sql_eval_scores_with_result_equivalence(self) -> None:
        result_path = WORKSPACE_ROOT / "results/sql/qwen35_0_8b__exp001_sql_sft/post_train_smoke.json"
        if result_path.exists():
            result_path.unlink()

        summary = run_sql_eval(
            "experiments/sql/qwen35_0_8b__exp001_sql_sft.json",
            model_variant="adapter",
            predictor=lambda case: case.gold_sql,
        )

        self.assertEqual(summary.case_count, 2)
        self.assertEqual(summary.passed_count, 2)
        self.assertEqual(summary.pass_rate, 1.0)
        self.assertTrue(result_path.exists())
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model_variant"], "adapter")
        self.assertEqual(payload["passed_count"], 2)

    def test_run_sql_eval_result_label_keeps_prompt_candidate_outputs_separate(self) -> None:
        result_path = (
            WORKSPACE_ROOT
            / "results/sql/qwen35_0_8b__exp001_sql_sft/adapter__sql_smoke_v1__c001.json"
        )
        if result_path.exists():
            result_path.unlink()

        summary = run_sql_eval(
            "experiments/sql/qwen35_0_8b__exp001_sql_sft.json",
            model_variant="adapter",
            eval_dataset="datasets/sql/smoke/sql_smoke_v1.jsonl",
            result_label="c001",
            predictor=lambda case: case.gold_sql,
        )

        self.assertEqual(summary.passed_count, 2)
        self.assertTrue(result_path.exists())

    def test_report_sql_prompt_lengths_writes_dataset_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "token_report.json"

            summary = report_sql_prompt_lengths(
                "experiments/sql/qwen35_0_8b__exp001_sql_sft.json",
                output_path=output_path,
                tokenizer_like=_FakeSQLTokenizer(),
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(summary.experiment_id, "qwen35_0_8b__exp001_sql_sft")
            self.assertEqual(len(summary.dataset_reports), 2)
            self.assertEqual(summary.dataset_reports[0].artifact, "train")
            self.assertEqual(summary.dataset_reports[0].row_count, 5)
            self.assertGreaterEqual(
                summary.dataset_reports[0].max_tokens,
                summary.dataset_reports[0].p95_tokens,
            )

    def test_run_sql_candidate_pool_eval_reports_pass_at_n_and_selection(self) -> None:
        result_path = (
            WORKSPACE_ROOT
            / "results/sql/qwen35_0_8b__exp001_sql_sft/"
            "candidates__adapter__sql_smoke_v1__unit_pool.json"
        )
        if result_path.exists():
            result_path.unlink()

        def candidate_predictor(case):
            if case.case_id.endswith("001"):
                return ["SELECT missing FROM employees;", case.gold_sql, "SELECT name FROM employees WHERE 0;"]
            return [case.gold_sql, "SELECT missing FROM departments;", "SELECT name FROM employees WHERE 0;"]

        summary = run_sql_candidate_pool_eval(
            "experiments/sql/qwen35_0_8b__exp001_sql_sft.json",
            model_variant="adapter",
            eval_dataset="datasets/sql/smoke/sql_smoke_v1.jsonl",
            candidate_count=3,
            result_label="unit_pool",
            predictor=candidate_predictor,
        )

        self.assertEqual(summary.case_count, 2)
        self.assertEqual(summary.first_passed_count, 1)
        self.assertEqual(summary.pass_at_n_count, 2)
        self.assertEqual(summary.selected_passed_count, 2)
        self.assertEqual(summary.records[0].selected_candidate_index, 2)
        self.assertEqual(summary.records[0].candidates[0].prediction_error, "no such column: missing")
        self.assertTrue(summary.records[0].candidates[1].result_signature)
        self.assertTrue(result_path.exists())

    def test_run_sql_eval_with_repair_preserves_first_pass_and_final_scores(self) -> None:
        cases = load_sql_eval_cases("datasets/sql/smoke/sql_smoke_v1.jsonl")

        def first_predictor(case):
            if case.case_id == cases[0].case_id:
                return "SELECT missing FROM employees;"
            return case.gold_sql

        def repair_predictor(case, previous_sql, observation):
            self.assertEqual(previous_sql, "SELECT missing FROM employees;")
            self.assertIn("no such column: missing", observation)
            return case.gold_sql

        summary = run_sql_eval_with_repair(
            "experiments/sql/qwen35_0_8b__exp001_sql_sft.json",
            model_variant="adapter",
            predictor=first_predictor,
            repair_predictor=repair_predictor,
        )

        self.assertEqual(summary.case_count, 2)
        self.assertEqual(summary.first_passed_count, 1)
        self.assertEqual(summary.final_passed_count, 2)
        self.assertEqual(summary.repair_attempt_count, 1)
        self.assertEqual(summary.repair_success_count, 1)
        failed_then_repaired = [
            record for record in summary.records if not record.first_result.passed
        ][0]
        self.assertEqual(failed_then_repaired.first_result.predicted_sql, "SELECT missing FROM employees;")
        self.assertEqual(failed_then_repaired.repair_attempts[0].input_failure_type, "prediction_schema_error")
        self.assertTrue(Path(summary.result_path).exists())

    def test_run_sql_eval_with_repair_respects_configured_failure_types(self) -> None:
        cases = load_sql_eval_cases("datasets/sql/smoke/sql_smoke_v1.jsonl")

        def first_predictor(case):
            if case.case_id == cases[0].case_id:
                return "SELECT name FROM employees;"
            return case.gold_sql

        def repair_predictor(case, previous_sql, observation):
            raise AssertionError("row-count mismatches were not configured for repair")

        summary = run_sql_eval_with_repair(
            "experiments/sql/qwen35_0_8b__exp001_sql_sft.json",
            model_variant="adapter",
            predictor=first_predictor,
            repair_predictor=repair_predictor,
            repair_failure_types={"prediction_schema_error"},
        )

        self.assertEqual(summary.first_passed_count, 1)
        self.assertEqual(summary.final_passed_count, 1)
        self.assertEqual(summary.repair_attempt_count, 0)

    def test_build_repair_eval_messages_include_observation_without_target(self) -> None:
        case = load_sql_eval_cases("datasets/sql/smoke/sql_smoke_v1.jsonl")[0]

        messages = build_repair_eval_messages(
            case,
            previous_sql="SELECT missing FROM employees;",
            execution_observation="Execution error: no such column: missing",
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Previous SQL:", messages[1]["content"])
        self.assertIn("Execution Observation:", messages[1]["content"])
        self.assertNotIn(case.gold_sql, messages[1]["content"])

    def test_analyze_sql_eval_result_classifies_failures(self) -> None:
        result_payload = {
            "experiment_id": "exp_test",
            "model_variant": "adapter",
            "eval_dataset": "datasets/sql/eval/test.jsonl",
            "case_count": 4,
            "passed_count": 1,
            "pass_rate": 0.25,
            "records": [
                {
                    "case_id": "passed",
                    "task_id": "passed",
                    "predicted_sql": "SELECT 1;",
                    "passed": True,
                    "prediction_error": None,
                    "gold_error": None,
                    "predicted_rows": [[1]],
                    "gold_rows": [[1]],
                },
                {
                    "case_id": "schema_error",
                    "task_id": "schema_error",
                    "predicted_sql": "SELECT missing FROM employees;",
                    "passed": False,
                    "prediction_error": "no such column: missing",
                    "gold_error": None,
                    "predicted_rows": [],
                    "gold_rows": [["Ava"]],
                },
                {
                    "case_id": "syntax_error",
                    "task_id": "syntax_error",
                    "predicted_sql": "SELECT FROM employees;",
                    "passed": False,
                    "prediction_error": "near \"FROM\": syntax error",
                    "gold_error": None,
                    "predicted_rows": [],
                    "gold_rows": [["Ava"]],
                },
                {
                    "case_id": "wrong_count",
                    "task_id": "wrong_count",
                    "predicted_sql": "SELECT name FROM employees;",
                    "passed": False,
                    "prediction_error": None,
                    "gold_error": None,
                    "predicted_rows": [["Ava"], ["Ben"]],
                    "gold_rows": [["Ava"]],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "eval.json"
            result_path.write_text(json.dumps(result_payload), encoding="utf-8")

            summary = analyze_sql_eval_result(result_path)
            self.assertTrue(Path(summary.analysis_path).exists())

        self.assertEqual(summary.failed_count, 3)
        self.assertEqual(summary.failure_counts["prediction_schema_error"], 1)
        self.assertEqual(summary.failure_counts["prediction_syntax_error"], 1)
        self.assertEqual(summary.failure_counts["row_count_mismatch"], 1)

    def test_collect_sql_repair_data_writes_valid_repair_rows(self) -> None:
        eval_rows = [
            {
                "schema_version": "sql_eval_case:v1",
                "case_id": "case_schema",
                "source_benchmark": "smoke",
                "source_split": "dev",
                "task_id": "case_schema",
                "fixture_id": "company_small",
                "db_id": "company_small",
                "db_path": None,
                "dialect": "sqlite",
                "question": "List employee names.",
                "schema_text": "CREATE TABLE employees (name TEXT);",
                "knowledge_text": None,
                "gold_sql": "SELECT name FROM employees;",
                "task_type": "select",
                "order_sensitive": False,
                "numeric_tolerance": 0.000001,
                "tags": ["smoke"],
            },
            {
                "schema_version": "sql_eval_case:v1",
                "case_id": "case_wrong_count",
                "source_benchmark": "smoke",
                "source_split": "dev",
                "task_id": "case_wrong_count",
                "fixture_id": "company_small",
                "db_id": "company_small",
                "db_path": None,
                "dialect": "sqlite",
                "question": "List one employee name.",
                "schema_text": "CREATE TABLE employees (name TEXT);",
                "knowledge_text": None,
                "gold_sql": "SELECT name FROM employees LIMIT 1;",
                "task_type": "select",
                "order_sensitive": False,
                "numeric_tolerance": 0.000001,
                "tags": ["smoke"],
            },
        ]
        result_payload = {
            "experiment_id": "exp_test",
            "base_model": "Qwen/Qwen3.5-0.8B-Base",
            "model_variant": "adapter",
            "eval_dataset": "eval.jsonl",
            "case_count": 2,
            "passed_count": 0,
            "pass_rate": 0.0,
            "records": [
                {
                    "case_id": "case_schema",
                    "task_id": "case_schema",
                    "predicted_sql": "SELECT missing FROM employees;",
                    "passed": False,
                    "prediction_error": "no such column: missing",
                    "gold_error": None,
                    "predicted_rows": [],
                    "gold_rows": [["Ava"]],
                },
                {
                    "case_id": "case_wrong_count",
                    "task_id": "case_wrong_count",
                    "predicted_sql": "SELECT name FROM employees;",
                    "passed": False,
                    "prediction_error": None,
                    "gold_error": None,
                    "predicted_rows": [["Ava"], ["Ben"]],
                    "gold_rows": [["Ava"]],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_path = Path(tmp_dir) / "eval.jsonl"
            result_path = Path(tmp_dir) / "result.json"
            output_path = Path(tmp_dir) / "repair.jsonl"
            eval_path.write_text(
                "".join(json.dumps(row) + "\n" for row in eval_rows),
                encoding="utf-8",
            )
            result_path.write_text(json.dumps(result_payload), encoding="utf-8")

            summary = collect_sql_repair_data(
                result_path=result_path,
                eval_dataset=eval_path,
                output_path=output_path,
            )
            repair_rows = load_sql_repair_examples(output_path)

        self.assertEqual(summary.collected_count, 2)
        self.assertEqual(summary.failure_counts["prediction_schema_error"], 1)
        self.assertEqual(summary.failure_counts["row_count_mismatch"], 1)
        self.assertEqual(repair_rows[0].previous_sql, "SELECT missing FROM employees;")
        self.assertIn("Execution error", repair_rows[0].execution_error)
        self.assertIn("Result mismatch", repair_rows[1].execution_error)

    def test_collect_sql_repair_data_strong_only_filters_wrong_result_rows(self) -> None:
        eval_row = {
            "schema_version": "sql_eval_case:v1",
            "case_id": "case_wrong_count",
            "source_benchmark": "smoke",
            "source_split": "dev",
            "task_id": "case_wrong_count",
            "fixture_id": "company_small",
            "db_id": "company_small",
            "db_path": None,
            "dialect": "sqlite",
            "question": "List one employee name.",
            "schema_text": "CREATE TABLE employees (name TEXT);",
            "knowledge_text": None,
            "gold_sql": "SELECT name FROM employees LIMIT 1;",
            "task_type": "select",
            "order_sensitive": False,
            "numeric_tolerance": 0.000001,
            "tags": ["smoke"],
        }
        result_payload = {
            "experiment_id": "exp_test",
            "base_model": "Qwen/Qwen3.5-0.8B-Base",
            "model_variant": "adapter",
            "eval_dataset": "eval.jsonl",
            "case_count": 1,
            "passed_count": 0,
            "pass_rate": 0.0,
            "records": [
                {
                    "case_id": "case_wrong_count",
                    "task_id": "case_wrong_count",
                    "predicted_sql": "SELECT name FROM employees;",
                    "passed": False,
                    "prediction_error": None,
                    "gold_error": None,
                    "predicted_rows": [["Ava"], ["Ben"]],
                    "gold_rows": [["Ava"]],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_path = Path(tmp_dir) / "eval.jsonl"
            result_path = Path(tmp_dir) / "result.json"
            output_path = Path(tmp_dir) / "repair.jsonl"
            eval_path.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")
            result_path.write_text(json.dumps(result_payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no repair rows collected"):
                collect_sql_repair_data(
                    result_path=result_path,
                    eval_dataset=eval_path,
                    output_path=output_path,
                    strong_only=True,
                )

    def test_extract_generated_sql_strips_code_fence_and_trailing_text(self) -> None:
        generated = "```sql\nSELECT name FROM employees;\n```\nextra"

        self.assertEqual(extract_generated_sql(generated), "SELECT name FROM employees;")

    def test_base_eval_uses_native_chat_template_when_available(self) -> None:
        messages = [
            {"role": "system", "content": "Return SQL only."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": ""},
        ]

        rendered = _render_generation_prompt(
            _FakeChatTemplateTokenizer(),
            messages,
            model_variant="base",
        )

        self.assertEqual(rendered, "system:Return SQL only.|user:Question|assistant:")

    def test_adapter_eval_keeps_repo_sft_prompt_even_with_chat_template(self) -> None:
        messages = [
            {"role": "system", "content": "Return SQL only."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": ""},
        ]

        rendered = _render_generation_prompt(
            _FakeChatTemplateTokenizer(),
            messages,
            model_variant="adapter",
        )

        self.assertIn("<|system|>", rendered)
        self.assertIn("<|user|>", rendered)
        self.assertTrue(rendered.rstrip().endswith("<|assistant|>"))

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
        eval_messages = build_eval_messages(train_examples[0], system_prompt="Custom system")

        self.assertEqual(train_messages[-1]["content"], "SELECT name FROM employees;")
        self.assertIn("Use table and column names exactly", train_messages[0]["content"])
        self.assertIn("quote the full identifier with backticks", train_messages[0]["content"])
        self.assertEqual(eval_messages[0]["content"], "Custom system")
        self.assertIn("Previous SQL:", repair_messages[1]["content"])
        self.assertIn("no such column: missing", repair_messages[1]["content"])

    def test_premsql_prompt_style_labels_evidence_and_sql_slot(self) -> None:
        train_row = {
            "schema_version": "sql_train_example:v1",
            "row_id": "train_001",
            "source_benchmark": "bird",
            "source_split": "train",
            "task_id": "bird_task",
            "db_id": "tiny_db",
            "db_path": "external/sql/benchmarks/premai-io__birdbench/train/train_databases/tiny_db/tiny_db.sqlite",
            "dialect": "sqlite",
            "question": "Which county has the highest rate?",
            "schema_text": "CREATE TABLE frpm (`County Name` TEXT, `Free Meal Count (K-12)` INTEGER);",
            "knowledge_text": "rate = free meals / enrollment",
            "target_sql": "SELECT `County Name` FROM frpm;",
            "task_type": "select",
            "provenance": {
                "created_by": "benchmark",
                "teacher_model": None,
                "source_path": "tests",
            },
            "tags": ["bird"],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.jsonl"
            train_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            row = load_sql_train_examples(train_path)[0]

        messages = build_train_messages(row, prompt_style="premsql_text")

        self.assertIn("# Additional Knowledge:", messages[1]["content"])
        self.assertIn("# Database and Table Schema:", messages[1]["content"])
        self.assertTrue(messages[1]["content"].rstrip().endswith("# SQL:"))

    def test_prompt_rendering_includes_column_value_notes(self) -> None:
        train_row = {
            "schema_version": "sql_train_example:v1",
            "row_id": "train_001",
            "source_benchmark": "bird",
            "source_split": "train",
            "task_id": "bird_task",
            "db_id": "regional_sales",
            "db_path": "external/sql/benchmarks/premai-io__birdbench/train/train_databases/regional_sales/regional_sales.sqlite",
            "dialect": "sqlite",
            "question": "Which sales channel has the highest average unit price?",
            "schema_text": "CREATE TABLE `Sales Orders` (`Sales Channel` TEXT, `Unit Price` TEXT);",
            "knowledge_text": None,
            "column_value_notes": [
                "`Sales Orders`.`Unit Price`: TEXT numeric values may include commas, e.g. \"1,000.00\"; "
                "for numeric aggregation use CAST(REPLACE(T1.`Unit Price`, ',', '') AS REAL)."
            ],
            "target_sql": "SELECT `Sales Channel` FROM `Sales Orders`;",
            "task_type": "select",
            "provenance": {
                "created_by": "benchmark",
                "teacher_model": None,
                "source_path": "tests",
            },
            "tags": ["bird"],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.jsonl"
            train_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            row = load_sql_train_examples(train_path)[0]

        canonical_messages = build_train_messages(row)
        premsql_messages = build_train_messages(row, prompt_style="premsql_text")

        self.assertIn("Column value notes:", canonical_messages[1]["content"])
        self.assertIn("may include commas", canonical_messages[1]["content"])
        self.assertIn("# Column Value Notes:", premsql_messages[1]["content"])
        self.assertIn("CAST(REPLACE(T1.`Unit Price`, ',', '') AS REAL)", premsql_messages[1]["content"])

    def test_attach_sqlite_profile_metadata_adds_compact_train_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "sales.sqlite"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE Customers (
                        CustomerID INTEGER PRIMARY KEY,
                        Name TEXT
                    );
                    CREATE TABLE Sales (
                        SalesID INTEGER PRIMARY KEY,
                        CustomerID INTEGER NOT NULL,
                        Quantity INTEGER NOT NULL,
                        FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
                    );
                    INSERT INTO Customers VALUES (80, 'Acme'), (81, 'Globex');
                    INSERT INTO Sales VALUES (1, 80, 3), (2, 80, 4), (3, 81, 1);
                    """
                )
            train_row = {
                "schema_version": "sql_train_example:v1",
                "row_id": "train_001",
                "source_benchmark": "bird",
                "source_split": "train",
                "task_id": "bird_task",
                "db_id": "sales",
                "db_path": str(db_path),
                "dialect": "sqlite",
                "question": "How many sales ids are there for customer id 80?",
                "schema_text": (
                    "CREATE TABLE Customers (CustomerID INTEGER PRIMARY KEY, Name TEXT);"
                    "CREATE TABLE Sales (SalesID INTEGER PRIMARY KEY, CustomerID INTEGER);"
                ),
                "knowledge_text": None,
                "target_sql": "SELECT COUNT(SalesID) FROM Sales WHERE CustomerID = 80",
                "task_type": "select",
                "provenance": {
                    "created_by": "benchmark",
                    "teacher_model": None,
                    "source_path": "tests",
                },
                "tags": ["bird"],
            }
            input_path = Path(tmp_dir) / "train.jsonl"
            output_path = Path(tmp_dir) / "train_profile_notes.jsonl"
            input_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")

            summary = attach_sqlite_profile_metadata(
                input_path=input_path,
                output_path=output_path,
                artifact="train",
                max_column_notes=4,
            )
            rows = load_sql_train_examples(output_path)

        self.assertEqual(summary.row_count, 1)
        self.assertEqual(summary.db_count, 1)
        self.assertEqual(len(rows[0].column_value_notes), 4)
        joined_notes = "\n".join(rows[0].column_value_notes)
        self.assertIn("`Sales`.`CustomerID`", joined_notes)
        self.assertIn("joins `Customers`.`CustomerID`", joined_notes)
        self.assertIn("'80'", joined_notes)

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


def _train_row(
    *,
    row_id: str,
    task_id: str,
    db_id: str,
    question: str,
    target_sql: str,
) -> dict[str, object]:
    return {
        "schema_version": "sql_train_example:v1",
        "row_id": row_id,
        "source_benchmark": "synthetic",
        "source_split": "train",
        "task_id": task_id,
        "db_id": db_id,
        "dialect": "sqlite",
        "question": question,
        "schema_text": "CREATE TABLE people (name TEXT);",
        "knowledge_text": None,
        "target_sql": target_sql,
        "task_type": "select",
        "provenance": {
            "created_by": "test",
            "teacher_model": None,
            "source_path": "tests",
        },
        "tags": ["test"],
    }


def _eval_row(
    *,
    case_id: str,
    task_id: str,
    db_id: str,
    question: str,
    gold_sql: str,
) -> dict[str, object]:
    return {
        "schema_version": "sql_eval_case:v1",
        "case_id": case_id,
        "source_benchmark": "synthetic",
        "source_split": "dev",
        "task_id": task_id,
        "fixture_id": f"test:{db_id}",
        "db_id": db_id,
        "dialect": "sqlite",
        "question": question,
        "schema_text": "CREATE TABLE people (name TEXT);",
        "knowledge_text": None,
        "gold_sql": gold_sql,
        "task_type": "select",
        "order_sensitive": False,
        "numeric_tolerance": 0.000001,
        "tags": ["test"],
    }


def _write_superstore_lab_fixture(dataset_root: Path) -> None:
    db_dir = dataset_root / "train_databases" / "superstore"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "superstore.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE people (
                `Customer ID` TEXT,
                `Customer Name` TEXT,
                Segment TEXT,
                Country TEXT,
                City TEXT,
                State TEXT,
                `Postal Code` INTEGER,
                Region TEXT,
                PRIMARY KEY (`Customer ID`, Region)
            );
            CREATE TABLE product (
                `Product ID` TEXT,
                `Product Name` TEXT,
                Category TEXT,
                `Sub-Category` TEXT,
                Region TEXT,
                PRIMARY KEY (`Product ID`, Region)
            );
            CREATE TABLE central_superstore (
                `Row ID` INTEGER PRIMARY KEY,
                `Order ID` TEXT,
                `Order Date` DATE,
                `Ship Date` DATE,
                `Ship Mode` TEXT,
                `Customer ID` TEXT,
                Region TEXT,
                `Product ID` TEXT,
                Sales REAL,
                Quantity INTEGER,
                Discount REAL,
                Profit REAL
            );
            CREATE TABLE east_superstore AS SELECT * FROM central_superstore WHERE 0;
            CREATE TABLE south_superstore AS SELECT * FROM central_superstore WHERE 0;
            CREATE TABLE west_superstore AS SELECT * FROM central_superstore WHERE 0;
            """
        )
        for region, table_name, base in [
            ("Central", "central_superstore", 100),
            ("East", "east_superstore", 200),
            ("South", "south_superstore", 300),
            ("West", "west_superstore", 400),
        ]:
            conn.executemany(
                "INSERT INTO people VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (f"C{base}A", f"Alex {region}", "Consumer", "United States", "Austin", "Texas", 10000, region),
                    (f"C{base}B", f"Blair {region}", "Corporate", "United States", "Boston", "Massachusetts", 20000, region),
                ],
            )
            conn.executemany(
                "INSERT INTO product VALUES (?, ?, ?, ?, ?)",
                [
                    (f"P{base}A", f"Desk {region}", "Furniture", "Tables", region),
                    (f"P{base}B", f"Paper {region}", "Office Supplies", "Paper", region),
                ],
            )
            conn.executemany(
                f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (base + 1, f"O{base}A", "2013-01-01", "2013-01-03", "First Class", f"C{base}A", region, f"P{base}A", 100.0, 2, 0.0, 10.0),
                    (base + 2, f"O{base}B", "2014-01-01", "2014-01-05", "Standard Class", f"C{base}B", region, f"P{base}B", 50.0, 5, 0.1, 5.0),
                ],
            )


def _write_regional_sales_lab_fixture(dataset_root: Path) -> None:
    db_dir = dataset_root / "train_databases" / "regional_sales"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "regional_sales.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE Customers (
                CustomerID INTEGER PRIMARY KEY,
                `Customer Names` TEXT
            );
            CREATE TABLE Products (
                ProductID INTEGER PRIMARY KEY,
                `Product Name` TEXT
            );
            CREATE TABLE Regions (
                StateCode TEXT PRIMARY KEY,
                State TEXT,
                Region TEXT
            );
            CREATE TABLE `Sales Team` (
                SalesTeamID INTEGER PRIMARY KEY,
                `Sales Team` TEXT,
                Region TEXT
            );
            CREATE TABLE `Store Locations` (
                StoreID INTEGER PRIMARY KEY,
                `City Name` TEXT,
                County TEXT,
                StateCode TEXT,
                State TEXT,
                Type TEXT,
                Latitude REAL,
                Longitude REAL,
                AreaCode INTEGER,
                Population INTEGER,
                `Household Income` INTEGER,
                `Median Income` INTEGER,
                `Land Area` INTEGER,
                `Water Area` INTEGER,
                `Time Zone` TEXT
            );
            CREATE TABLE `Sales Orders` (
                OrderNumber TEXT PRIMARY KEY,
                `Sales Channel` TEXT,
                WarehouseCode TEXT,
                ProcuredDate TEXT,
                OrderDate TEXT,
                ShipDate TEXT,
                DeliveryDate TEXT,
                CurrencyCode TEXT,
                _SalesTeamID INTEGER,
                _CustomerID INTEGER,
                _StoreID INTEGER,
                _ProductID INTEGER,
                `Order Quantity` INTEGER,
                `Discount Applied` REAL,
                `Unit Price` TEXT,
                `Unit Cost` TEXT
            );
            """
        )
        customers = [
            (1, "Acme Corp"),
            (2, "Bravo Ltd"),
            (3, "Coda Inc"),
            (4, "Delta Group"),
            (5, "Echo LLC"),
            (6, "Foxtrot Co"),
            (7, "Gamma Shop"),
            (8, "Helio Partners"),
        ]
        products = [
            (1, "Cookware"),
            (2, "Photo Frames"),
            (3, "Table Lamps"),
            (4, "Bean Bags"),
            (5, "Wall Coverings"),
            (6, "Outdoor Furniture"),
            (7, "Candles"),
            (8, "Clocks"),
        ]
        conn.executemany("INSERT INTO Customers VALUES (?, ?)", customers)
        conn.executemany("INSERT INTO Products VALUES (?, ?)", products)
        for index, region in enumerate(("Midwest", "Northeast", "South", "West"), start=1):
            state_code = f"S{index}"
            conn.execute(
                "INSERT INTO Regions VALUES (?, ?, ?)",
                (state_code, f"State {index}", region),
            )
            conn.execute(
                "INSERT INTO `Sales Team` VALUES (?, ?, ?)",
                (index, f"Seller {region}", region),
            )
            conn.execute(
                "INSERT INTO `Store Locations` VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    index,
                    f"City {index}",
                    f"County {index}",
                    state_code,
                    f"State {index}",
                    "City",
                    1.0,
                    2.0,
                    100 + index,
                    1000 + index,
                    2000 + index,
                    3000 + index,
                    4000 + index,
                    5000 + index,
                    "UTC",
                ),
            )
            base = index * 100
            orders = [
                (f"SO - {base}A", "Distributor", f"WARE-{base}A", "1/1/18", "5/1/18", "5/3/18", "5/4/18", "USD", index, 1, index, 1, 2, 0.0, "1,000.00", "600.00"),
                (f"SO - {base}B", "In-Store", f"WARE-{base}B", "1/1/19", "6/1/19", "6/3/19", "6/4/19", "USD", index, 2, index, 2, 4, 0.1, "2,000.00", "800.00"),
                (f"SO - {base}C", "Online", f"WARE-{base}C", "1/1/20", "7/1/20", "7/3/20", "7/4/20", "USD", index, 3, index, 3, 6, 0.2, "3,000.00", "900.00"),
            ]
            conn.executemany(
                "INSERT INTO `Sales Orders` VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                orders,
            )


if __name__ == "__main__":
    unittest.main()
