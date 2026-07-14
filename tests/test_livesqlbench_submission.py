from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sqlbench_lab.livesqlbench_submission import (
    LiveSQLBenchSubmissionConfig,
    build_prepare_command,
    build_run_command,
    build_submission_plan,
    run_prepare,
    run_submission,
    write_submission_plan,
)


def _config(tmp_path: Path) -> LiveSQLBenchSubmissionConfig:
    cli_repo = tmp_path / "LiveSQLBench-CLI"
    (cli_repo / "adapters" / "livesqlbench").mkdir(parents=True)
    (cli_repo / "adapters" / "livesqlbench" / "run_adapter.py").write_text("# test\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    data_jsonl = data_root / "tasks.jsonl"
    data_jsonl.write_text(json.dumps({"instance_id": "task-1", "selected_database": "demo"}) + "\n", encoding="utf-8")
    eval_src_dir = tmp_path / "evaluation" / "src"
    eval_src_dir.mkdir(parents=True)
    db_dump_root = tmp_path / "dumps"
    db_dump_root.mkdir()
    return LiveSQLBenchSubmissionConfig(
        cli_repo=cli_repo,
        data_root=data_root,
        data_jsonl=data_jsonl,
        eval_src_dir=eval_src_dir,
        db_dump_root=db_dump_root,
        output_dir=tmp_path / "generated",
        agent_image="test-agent:latest",
        agent="codex",
        model="local/qwen",
        trials=2,
        limit=1,
        force=True,
    )


def test_build_commands_keep_preparation_and_official_run_separate(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert build_prepare_command(config) == (
        "python3",
        "adapters/livesqlbench/run_adapter.py",
        "--data-root",
        str(config.data_root),
        "--data-jsonl",
        str(config.data_jsonl),
        "--eval-src-dir",
        str(config.eval_src_dir),
        "--db-dump-root",
        str(config.db_dump_root),
        "--output-dir",
        str(config.output_dir),
        "--agent-image",
        "test-agent:latest",
        "--limit",
        "1",
        "--force",
    )
    assert build_run_command(config) == (
        "uv",
        "run",
        "harbor",
        "run",
        "-p",
        str(config.output_dir),
        "-a",
        "codex",
        "-n",
        "2",
        "-m",
        "local/qwen",
    )


def test_submission_plan_records_official_commit_and_rejects_no_gt_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with patch(
        "sqlbench_lab.livesqlbench_submission.subprocess.run",
        return_value=type("Completed", (), {"stdout": "abc123\n"})(),
    ) as run:
        plan = build_submission_plan(config)

    assert plan.official_cli_commit == "abc123"
    assert plan.workflow == "official_livesqlbench_cli"
    run.assert_called_once_with(
        ("git", "-C", str(config.cli_repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )

    config.data_jsonl.write_text(
        json.dumps({"instance_id": "task-1", "test_cases": [{"expected": "secret"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protected LiveSQLBench content"):
        build_submission_plan(config)


def test_public_placeholder_fields_are_allowed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.data_jsonl.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "sol_sql": [],
                "test_cases": [],
                "external_knowledge": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with patch(
        "sqlbench_lab.livesqlbench_submission.subprocess.run",
        return_value=type("Completed", (), {"stdout": "abc123\n"})(),
    ):
        plan = build_submission_plan(config)

    assert plan.official_cli_commit == "abc123"


def test_run_submission_executes_prepare_then_harbor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with patch(
        "sqlbench_lab.livesqlbench_submission.subprocess.run",
        return_value=type("Completed", (), {"stdout": "abc123\n"})(),
    ):
        plan = build_submission_plan(config)

    calls: list[tuple[tuple[str, ...], Path]] = []
    run_submission(plan, runner=lambda command, cwd: calls.append((command, cwd)))

    assert calls == [
        (plan.prepare_command, config.cli_repo),
        (plan.run_command, config.cli_repo),
    ]


def test_run_prepare_executes_only_the_official_adapter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with patch(
        "sqlbench_lab.livesqlbench_submission.subprocess.run",
        return_value=type("Completed", (), {"stdout": "abc123\n"})(),
    ):
        plan = build_submission_plan(config)

    calls: list[tuple[tuple[str, ...], Path]] = []
    run_prepare(plan, runner=lambda command, cwd: calls.append((command, cwd)))

    assert calls == [(plan.prepare_command, config.cli_repo)]


def test_write_submission_plan_is_json_audit_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with patch(
        "sqlbench_lab.livesqlbench_submission.subprocess.run",
        return_value=type("Completed", (), {"stdout": "abc123\n"})(),
    ):
        plan = build_submission_plan(config)

    output = write_submission_plan(plan, tmp_path / "artifacts" / "submission.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "livesqlbench_submission:v1"
    assert payload["official_cli_commit"] == "abc123"
    assert payload["prepare_command"][0:2] == ["python3", "adapters/livesqlbench/run_adapter.py"]
