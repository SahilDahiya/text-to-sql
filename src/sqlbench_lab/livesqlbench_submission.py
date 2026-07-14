"""Explicit LiveSQLBench website-score submission workflow."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


SUBMISSION_SCHEMA_VERSION = "livesqlbench_submission:v1"
PROTECTED_FIELDS = frozenset({"sol_sql", "test_cases", "external_knowledge"})
RUN_ADAPTER_RELATIVE_PATH = Path("adapters/livesqlbench/run_adapter.py")


CommandRunner = Callable[[tuple[str, ...], Path], None]


@dataclass(frozen=True)
class LiveSQLBenchSubmissionConfig:
    """Inputs needed to run the public official LiveSQLBench CLI lane."""

    cli_repo: Path
    data_root: Path
    data_jsonl: Path
    eval_src_dir: Path
    db_dump_root: Path
    output_dir: Path
    agent_image: str = "livesqlbench-main-openhands:latest"
    agent: str = "codex"
    model: str | None = None
    trials: int = 1
    limit: int | None = None
    force: bool = False


@dataclass(frozen=True)
class LiveSQLBenchSubmissionPlan:
    """Auditable commands and provenance for one official-runner attempt."""

    schema_version: str
    workflow: str
    official_cli_repo: str
    official_cli_commit: str
    data_root: str
    data_jsonl: str
    eval_src_dir: str
    db_dump_root: str
    output_dir: str
    agent_image: str
    agent: str
    model: str | None
    trials: int
    limit: int | None
    force: bool
    prepare_command: tuple[str, ...]
    run_command: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_submission_plan(config: LiveSQLBenchSubmissionConfig) -> LiveSQLBenchSubmissionPlan:
    """Validate public inputs and build the official preparation/run commands."""

    _validate_config(config)
    _validate_public_dataset(config.data_jsonl)
    commit = _read_git_commit(config.cli_repo)
    prepare_command = build_prepare_command(config)
    run_command = build_run_command(config)
    return LiveSQLBenchSubmissionPlan(
        schema_version=SUBMISSION_SCHEMA_VERSION,
        workflow="official_livesqlbench_cli",
        official_cli_repo=str(config.cli_repo),
        official_cli_commit=commit,
        data_root=str(config.data_root),
        data_jsonl=str(config.data_jsonl),
        eval_src_dir=str(config.eval_src_dir),
        db_dump_root=str(config.db_dump_root),
        output_dir=str(config.output_dir),
        agent_image=config.agent_image,
        agent=config.agent,
        model=config.model,
        trials=config.trials,
        limit=config.limit,
        force=config.force,
        prepare_command=prepare_command,
        run_command=run_command,
    )


def build_prepare_command(config: LiveSQLBenchSubmissionConfig) -> tuple[str, ...]:
    """Build the pinned public adapter command without executing it."""

    command = [
        "python3",
        str(RUN_ADAPTER_RELATIVE_PATH),
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
        config.agent_image,
    ]
    if config.limit is not None:
        command.extend(("--limit", str(config.limit)))
    if config.force:
        command.append("--force")
    return tuple(command)


def build_run_command(config: LiveSQLBenchSubmissionConfig) -> tuple[str, ...]:
    """Build the official Harbor agent command for website-score evaluation."""

    command = [
        "uv",
        "run",
        "harbor",
        "run",
        "-p",
        str(config.output_dir),
        "-a",
        config.agent,
        "-n",
        str(config.trials),
    ]
    if config.model:
        command.extend(("-m", config.model))
    return tuple(command)


def write_submission_plan(plan: LiveSQLBenchSubmissionPlan, output_path: str | Path) -> Path:
    """Persist a plan only after all input and public-data checks pass."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_submission(
    plan: LiveSQLBenchSubmissionPlan,
    *,
    runner: CommandRunner | None = None,
) -> None:
    """Run task preparation followed by the official Harbor evaluation."""

    command_runner = runner or _run_command
    cli_repo = Path(plan.official_cli_repo)
    command_runner(plan.prepare_command, cli_repo)
    command_runner(plan.run_command, cli_repo)


def run_prepare(
    plan: LiveSQLBenchSubmissionPlan,
    *,
    runner: CommandRunner | None = None,
) -> None:
    """Run only the official adapter task-generation command."""

    command_runner = runner or _run_command
    command_runner(plan.prepare_command, Path(plan.official_cli_repo))


def _validate_config(config: LiveSQLBenchSubmissionConfig) -> None:
    if not config.cli_repo.is_dir():
        raise FileNotFoundError(f"LiveSQLBench-CLI repo not found: {config.cli_repo}")
    run_adapter = config.cli_repo / RUN_ADAPTER_RELATIVE_PATH
    if not run_adapter.is_file():
        raise FileNotFoundError(f"official LiveSQLBench adapter not found: {run_adapter}")
    for label, path in (
        ("data_root", config.data_root),
        ("eval_src_dir", config.eval_src_dir),
        ("db_dump_root", config.db_dump_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not config.data_jsonl.is_file():
        raise FileNotFoundError(f"data_jsonl not found: {config.data_jsonl}")
    if not config.agent_image.strip():
        raise ValueError("agent_image must not be empty")
    if not config.agent.strip():
        raise ValueError("agent must not be empty")
    if config.trials < 1:
        raise ValueError("trials must be at least 1")
    if config.limit is not None and config.limit < 1:
        raise ValueError("limit must be at least 1 when provided")


def _validate_public_dataset(path: Path) -> None:
    seen_ids: set[str] = set()
    record_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid public task JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"public task record at {path}:{line_number} must be an object")
            protected = sorted(
                field
                for field in PROTECTED_FIELDS.intersection(record)
                if _has_protected_content(record[field])
            )
            if protected:
                raise ValueError(
                    "protected LiveSQLBench content is forbidden in the website-score input: "
                    + ", ".join(protected)
                )
            instance_id = record.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValueError(f"public task record at {path}:{line_number} needs a non-empty instance_id")
            if instance_id in seen_ids:
                raise ValueError(f"duplicate public task instance_id at {path}:{line_number}: {instance_id}")
            seen_ids.add(instance_id)
            record_count += 1
    if record_count == 0:
        raise ValueError(f"public task JSONL is empty: {path}")


def _has_protected_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _read_git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise ValueError(f"could not read official LiveSQLBench-CLI commit: {repo}")
    return commit


def _run_command(command: tuple[str, ...], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)
