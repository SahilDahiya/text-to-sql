from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from db_sql_agent_env import import_seed_dataset, inspect_sqlite_schema, run_env_step


def test_env_step_returns_success_feedback(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)

    step = run_env_step(
        dataset_path=dataset_path,
        case_id="case_001",
        sql="SELECT name FROM employees ORDER BY name;",
        preview_rows=1,
    )

    assert step.done
    assert step.reward == 1.0
    assert step.validation.error is None
    assert step.validation.failure_type is None
    assert step.execution.ran
    assert step.execution.row_count == 2
    assert step.execution.preview == [["Ava"]]
    assert step.evaluation.gold_available
    assert step.evaluation.passed


def test_env_step_returns_schema_error_observation(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)

    step = run_env_step(
        dataset_path=dataset_path,
        case_id="case_001",
        sql="SELECT missing FROM employees;",
    )

    assert not step.done
    assert step.reward == 0.0
    assert step.validation.syntax_ok
    assert not step.validation.schema_ok
    assert step.validation.failure_type == "prediction_schema_error"
    assert "no such column: missing" in (step.validation.error or "")
    assert "Execution error" in (step.validation.observation or "")
    assert not step.execution.ran


def test_env_step_rejects_write_sql_without_mutating_database(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)

    step = run_env_step(
        dataset_path=dataset_path,
        case_id="case_001",
        sql="DELETE FROM employees;",
    )

    assert not step.done
    assert step.validation.failure_type == "prediction_execution_error"
    assert "readonly" in (step.validation.error or "").lower()
    assert not step.execution.ran

    check_step = run_env_step(
        dataset_path=dataset_path,
        case_id="case_001",
        sql="SELECT COUNT(*) FROM employees;",
    )
    assert check_step.execution.preview == [[2]]


def test_cli_prints_json_feedback(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "db_sql_agent_env.cli",
            "env-step",
            "--dataset",
            str(dataset_path),
            "--case-id",
            "case_001",
            "--sql",
            "SELECT missing FROM employees;",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["case_id"] == "case_001"
    assert payload["validation"]["failure_type"] == "prediction_schema_error"
    assert not payload["execution"]["ran"]


def test_loader_resolves_db_path_against_dataset_parent_chain(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    db_dir = repo_root / "datasets" / "sql" / "dbs" / "company"
    eval_dir = repo_root / "datasets" / "sql" / "eval"
    db_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    db_path = db_dir / "company.sqlite"
    _write_db(db_path)
    dataset_path = eval_dir / "eval.jsonl"
    row = _eval_row("datasets/sql/dbs/company/company.sqlite")
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    work_dir = tmp_path / "not_repo"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    step = run_env_step(
        dataset_path=dataset_path,
        case_id="case_001",
        sql="SELECT name FROM employees ORDER BY name;",
    )

    assert step.done


def test_inspect_sqlite_schema_returns_tables_columns_foreign_keys_and_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "company.sqlite"
    _write_db(db_path)

    profile = inspect_sqlite_schema(db_path)

    assert profile.dialect == "sqlite"
    assert profile.table_count == 2
    table_by_name = {table.name: table for table in profile.tables}
    assert table_by_name["departments"].row_count == 2
    employees = table_by_name["employees"]
    assert employees.row_count == 2
    assert [column.name for column in employees.columns] == ["id", "name", "department_id"]
    assert employees.columns[0].primary_key_position == 1
    assert employees.foreign_keys[0].column == "department_id"
    assert employees.foreign_keys[0].references_table == "departments"
    assert employees.foreign_keys[0].references_column == "id"
    assert employees.indexes[0].unique
    assert employees.indexes[0].columns[0].name == "name"


def test_inspect_sqlite_schema_writes_json_output(tmp_path: Path) -> None:
    db_path = tmp_path / "company.sqlite"
    output_path = tmp_path / "schema.json"
    _write_db(db_path)

    inspect_sqlite_schema(db_path, output_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["table_count"] == 2
    assert payload["tables"][0]["name"] == "departments"


def test_cli_inspect_schema_prints_json_profile(tmp_path: Path) -> None:
    db_path = tmp_path / "company.sqlite"
    _write_db(db_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "db_sql_agent_env.cli",
            "inspect-schema",
            "--db",
            str(db_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["dialect"] == "sqlite"
    assert payload["table_count"] == 2
    assert {table["name"] for table in payload["tables"]} == {"departments", "employees"}


def test_import_seed_dataset_copies_files_and_writes_summary(tmp_path: Path) -> None:
    train_path = tmp_path / "source_train.jsonl"
    eval_path = tmp_path / "source_eval.jsonl"
    output_dir = tmp_path / "seed"
    _write_seed_jsonl(train_path, [_train_row("train_001", "How many employees?", "SELECT COUNT(*) FROM employees;")])
    _write_seed_jsonl(eval_path, [_eval_seed_row("eval_001", "List employee names.", "SELECT name FROM employees;")])

    summary = import_seed_dataset(train_path=train_path, eval_path=eval_path, output_dir=output_dir)

    assert summary.train_row_count == 1
    assert summary.eval_row_count == 1
    assert summary.train_db_ids == ["company"]
    assert summary.eval_db_ids == ["company"]
    assert summary.tag_counts == {"count": 1, "names": 1}
    assert summary.overlapping_questions == []
    assert summary.overlapping_sql == []
    assert (output_dir / "train.jsonl").read_text(encoding="utf-8") == train_path.read_text(encoding="utf-8")
    assert (output_dir / "eval.jsonl").read_text(encoding="utf-8") == eval_path.read_text(encoding="utf-8")
    payload = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert payload["train_row_count"] == 1


def test_import_seed_dataset_rejects_exact_question_or_sql_overlap(tmp_path: Path) -> None:
    train_path = tmp_path / "source_train.jsonl"
    eval_path = tmp_path / "source_eval.jsonl"
    _write_seed_jsonl(train_path, [_train_row("train_001", "List employee names.", "SELECT name FROM employees;")])
    _write_seed_jsonl(eval_path, [_eval_seed_row("eval_001", "List employee names.", "SELECT name FROM employees;")])

    with pytest.raises(ValueError, match="seed train/eval overlap detected"):
        import_seed_dataset(train_path=train_path, eval_path=eval_path, output_dir=tmp_path / "seed")


def test_import_seed_dataset_can_record_allowed_overlap(tmp_path: Path) -> None:
    train_path = tmp_path / "source_train.jsonl"
    eval_path = tmp_path / "source_eval.jsonl"
    _write_seed_jsonl(train_path, [_train_row("train_001", "List employee names.", "SELECT name FROM employees;")])
    _write_seed_jsonl(eval_path, [_eval_seed_row("eval_001", "List employee names.", "SELECT name FROM employees;")])

    summary = import_seed_dataset(
        train_path=train_path,
        eval_path=eval_path,
        output_dir=tmp_path / "seed",
        allow_overlap=True,
    )

    assert summary.overlapping_questions == ["list employee names."]
    assert summary.overlapping_sql == ["select name from employees"]


def test_cli_import_seed_prints_summary(tmp_path: Path) -> None:
    train_path = tmp_path / "source_train.jsonl"
    eval_path = tmp_path / "source_eval.jsonl"
    output_dir = tmp_path / "seed"
    _write_seed_jsonl(train_path, [_train_row("train_001", "How many employees?", "SELECT COUNT(*) FROM employees;")])
    _write_seed_jsonl(eval_path, [_eval_seed_row("eval_001", "List employee names.", "SELECT name FROM employees;")])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "db_sql_agent_env.cli",
            "import-seed",
            "--train",
            str(train_path),
            "--eval",
            str(eval_path),
            "--output",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["train_row_count"] == 1
    assert payload["eval_row_count"] == 1
    assert (output_dir / "dataset_summary.json").exists()


def _write_dataset(tmp_path: Path) -> Path:
    db_path = tmp_path / "company.sqlite"
    _write_db(db_path)
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(json.dumps(_eval_row(str(db_path))) + "\n", encoding="utf-8")
    return dataset_path


def _write_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );

            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER NOT NULL,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            );
            CREATE UNIQUE INDEX idx_employees_name ON employees(name);

            INSERT INTO departments (id, name) VALUES (1, 'Engineering'), (2, 'Sales');
            INSERT INTO employees (id, name, department_id) VALUES (1, 'Ava', 1), (2, 'Ben', 2);
            """
        )


def _eval_row(db_path: str) -> dict[str, object]:
    return {
        "case_id": "case_001",
        "task_id": "case_001",
        "db_id": "company",
        "db_path": db_path,
        "dialect": "sqlite",
        "question": "List employee names.",
        "schema_text": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department_id INTEGER NOT NULL);",
        "gold_sql": "SELECT name FROM employees ORDER BY name;",
        "order_sensitive": True,
        "numeric_tolerance": 0.000001,
        "tags": ["smoke"],
    }


def _write_seed_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _train_row(row_id: str, question: str, target_sql: str) -> dict[str, object]:
    return {
        "row_id": row_id,
        "task_id": row_id,
        "db_id": "company",
        "question": question,
        "target_sql": target_sql,
        "tags": ["count"],
    }


def _eval_seed_row(case_id: str, question: str, gold_sql: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "task_id": case_id,
        "db_id": "company",
        "question": question,
        "gold_sql": gold_sql,
        "tags": ["names"],
    }
