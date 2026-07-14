"""Import allowed LiveSQLBench open-development tasks into SQLBench v2 artifacts."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluator import evaluate_sql_case
from .loaders import load_sql_eval_cases, load_sql_train_examples
from .models import SQLEvalCase
from .mixture import audit_sql_mixture

ALLOWED_TARGET_SOURCES = {"manual_verified", "independent_verified", "allowed_eval_gold"}
ALLOWED_SPLITS = {"train", "dev", "eval", "challenge"}
ALLOWED_TASK_FAMILIES = {
    "schema_grounding",
    "filtering",
    "joins",
    "date_value",
    "aggregation",
    "nesting",
    "management",
    "other",
}
ALLOWED_GROUNDING = {"requires_schema", "requires_database", "requires_schema_and_database"}


@dataclass(frozen=True)
class LiveSQLBenchTask:
    task_id: str
    task_path: str
    db_id: str
    dialect: str
    question: str
    schema_text: str
    knowledge_text: str | None
    db_path: str
    task_type: str
    difficulty: str
    category: str
    tags: tuple[str, ...]
    public_payload_has_targets: bool


@dataclass(frozen=True)
class LiveSQLBenchImportSummary:
    package_root: str
    source_revision: str
    discovered_task_count: int
    train_row_count: int
    eval_case_count: int
    train_output: str
    eval_output: str
    fingerprint: str


@dataclass(frozen=True)
class LiveSQLBenchVerificationSummary:
    package_root: str
    source_revision: str
    target_count: int
    verified_output: str


def discover_livesqlbench_tasks(package_root: str | Path) -> list[LiveSQLBenchTask]:
    """Read public task metadata without consuming protected target fields."""

    root = Path(package_root).resolve()
    if not root.is_dir():
        raise ValueError(f"LiveSQLBench package root does not exist: {root}")

    tasks = []
    for task_path in sorted(path for path in root.iterdir() if path.is_dir()):
        task_file = task_path / "task.toml"
        payload_file = task_path / "tests" / "task_payload.json"
        if not task_file.exists() and not payload_file.exists():
            continue
        if not task_file.exists() or not payload_file.exists():
            raise ValueError(f"incomplete LiveSQLBench task directory: {task_path}")
        tasks.append(_read_task(task_path, task_file, payload_file))

    if not tasks:
        raise ValueError(f"no LiveSQLBench tasks found under {root}")
    task_ids = [task.task_id for task in tasks]
    duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate LiveSQLBench task IDs: {', '.join(duplicates)}")
    return tasks


def build_livesqlbench_artifacts(
    *,
    package_root: str | Path,
    target_manifest: str | Path,
    source_revision: str,
    train_output: str | Path,
    eval_output: str | Path,
) -> LiveSQLBenchImportSummary:
    """Materialize only explicitly supplied, execution-verified targets."""

    if not source_revision.strip():
        raise ValueError("source_revision is required")
    tasks = discover_livesqlbench_tasks(package_root)
    task_by_id = {task.task_id: task for task in tasks}
    target_specs = _read_target_manifest(target_manifest)
    unknown_task_ids = sorted(set(target_specs) - set(task_by_id))
    if unknown_task_ids:
        raise ValueError(f"target manifest references unknown task IDs: {', '.join(unknown_task_ids[:10])}")

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for task_id, spec in sorted(target_specs.items()):
        task = task_by_id[task_id]
        row = _materialize_row(task, spec, source_revision=source_revision)
        if spec["split"] == "train":
            train_rows.append(row)
        else:
            eval_rows.append(_as_eval_case(row, spec))

    if not train_rows:
        raise ValueError("target manifest produced no train rows")
    if not eval_rows:
        raise ValueError("target manifest produced no eval cases")

    train_path = _write_jsonl_after_validation(train_output, train_rows, kind="train")
    eval_path = _write_jsonl_after_validation(eval_output, eval_rows, kind="eval")
    fingerprint = audit_sql_mixture([train_path]).fingerprint
    return LiveSQLBenchImportSummary(
        package_root=str(Path(package_root).resolve()),
        source_revision=source_revision,
        discovered_task_count=len(tasks),
        train_row_count=len(train_rows),
        eval_case_count=len(eval_rows),
        train_output=str(train_path),
        eval_output=str(eval_path),
        fingerprint=fingerprint,
    )


def verify_livesqlbench_targets(
    *,
    package_root: str | Path,
    target_manifest: str | Path,
    source_revision: str,
    verified_output: str | Path,
    verified_by: str,
    verified_at: str,
    postgres_connect: Any | None = None,
) -> LiveSQLBenchVerificationSummary:
    """Execute every target and write a manifest whose status is execution_verified."""

    if not verified_by.strip() or not verified_at.strip():
        raise ValueError("verified_by and verified_at are required")
    tasks = {task.task_id: task for task in discover_livesqlbench_tasks(package_root)}
    specs = _read_target_manifest(target_manifest, allow_pending=True)
    output_rows: list[dict[str, Any]] = []
    for task_id, original_spec in sorted(specs.items()):
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"target manifest references unknown task ID: {task_id}")
        spec = dict(original_spec)
        spec["verification"] = {
            "status": "execution_verified",
            "verified_by": verified_by,
            "verification_id": f"{source_revision}:{task_id}",
            "verified_at": verified_at,
        }
        row = _materialize_row(task, spec, source_revision=source_revision)
        case = SQLEvalCase.from_dict(_as_eval_case(row, spec))
        result = evaluate_sql_case(case, predicted_sql=case.gold_sql, postgres_connect=postgres_connect)
        if not result.passed:
            raise ValueError(
                f"target verification failed for {task_id}: "
                f"prediction_error={result.prediction_error!r} gold_error={result.gold_error!r}"
            )
        output_rows.append({**original_spec, "verification": spec["verification"]})

    output_path = Path(verified_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    try:
        _read_target_manifest(temp_path)
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return LiveSQLBenchVerificationSummary(
        package_root=str(Path(package_root).resolve()),
        source_revision=source_revision,
        target_count=len(output_rows),
        verified_output=str(output_path),
    )


def _read_task(task_path: Path, task_file: Path, payload_file: Path) -> LiveSQLBenchTask:
    task_config = tomllib.loads(task_file.read_text(encoding="utf-8"))
    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task payload must be an object: {payload_file}")
    if payload.get("sol_sql") or payload.get("test_cases"):
        raise ValueError(
            "LiveSQLBench task contains public target/test fields; refusing to ingest them: "
            f"{payload_file}"
        )

    db_id = _required_text(payload, "selected_database", payload_file)
    task_id = _required_text(payload, "instance_id", payload_file)
    question = _required_text(payload, "query", payload_file)
    metadata = task_config.get("metadata", {})
    tags = tuple(str(tag).casefold() for tag in metadata.get("tags", []))
    dialect = _dialect_from_tags(tags, task_path)
    schema_path = _single_file(task_path / "environment" / "db_assets", f"{db_id}_schema.txt")
    env_path = task_path / "environment" / "documents" / "db_env.sh"
    if dialect == "postgresql" and not env_path.exists():
        raise ValueError(f"PostgreSQL task is missing db_env.sh: {task_path}")
    db_path = str(env_path.resolve()) if dialect == "postgresql" else str(_sqlite_path(task_path).resolve())
    knowledge_text = _read_knowledge(task_path / "environment" / "db_assets", db_id)
    difficulty = _normalize_difficulty(metadata.get("difficulty") or payload.get("difficulty_tier"))
    category = str(payload.get("category") or metadata.get("category") or "unknown").casefold()
    return LiveSQLBenchTask(
        task_id=task_id,
        task_path=str(task_path.resolve()),
        db_id=db_id,
        dialect=dialect,
        question=question,
        schema_text=schema_path.read_text(encoding="utf-8"),
        knowledge_text=knowledge_text,
        db_path=db_path,
        task_type="management" if category == "management" else "select",
        difficulty=difficulty,
        category=category,
        tags=tags,
        public_payload_has_targets=False,
    )


def _read_target_manifest(path: str | Path, *, allow_pending: bool = False) -> dict[str, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"target manifest does not exist: {resolved}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"target manifest row must be an object: {resolved}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError("target manifest must contain at least one row")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = _required_text(row, "task_id", resolved)
        if task_id in result:
            raise ValueError(f"duplicate target manifest task_id: {task_id}")
        _validate_target_spec(row, resolved, allow_pending=allow_pending)
        result[task_id] = row
    return result


def _validate_target_spec(spec: dict[str, Any], path: Path, *, allow_pending: bool = False) -> None:
    split = _required_text(spec, "split", path)
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported target split {split!r} in {path}")
    target_sql = _required_text(spec, "target_sql", path)
    if not target_sql.strip():
        raise ValueError(f"target_sql cannot be empty in {path}")
    target_source = _required_text(spec, "target_source", path)
    if target_source not in ALLOWED_TARGET_SOURCES:
        raise ValueError(f"unsupported target_source {target_source!r} in {path}")
    if not isinstance(spec.get("verification"), dict):
        raise ValueError(f"verification object is required in {path}")
    verification = spec["verification"]
    if verification.get("status") not in ({"execution_verified", "pending"} if allow_pending else {"execution_verified"}):
        raise ValueError(f"target is not execution_verified in {path}")
    if verification.get("status") == "execution_verified":
        for field in ("verified_by", "verification_id", "verified_at"):
            _required_text(verification, field, path)
    family = _required_text(spec, "task_family", path)
    if family not in ALLOWED_TASK_FAMILIES:
        raise ValueError(f"unsupported task_family {family!r} in {path}")
    tier = spec.get("curriculum_tier")
    if not isinstance(tier, int) or not 1 <= tier <= 6:
        raise ValueError(f"curriculum_tier must be an integer from 1 to 6 in {path}")
    grounding = _required_text(spec, "grounding_requirement", path)
    shortcut = _required_text(spec, "shortcut_status", path)
    if grounding not in ALLOWED_GROUNDING or shortcut not in ALLOWED_GROUNDING:
        raise ValueError(f"unsupported grounding metadata in {path}")
    if shortcut not in {"requires_database", "requires_schema_and_database"}:
        raise ValueError(f"target must require database grounding in {path}")
    if not isinstance(spec.get("tags"), list) or not all(isinstance(tag, str) and tag for tag in spec["tags"]):
        raise ValueError(f"tags must be a non-empty list of strings in {path}")


def _materialize_row(task: LiveSQLBenchTask, spec: dict[str, Any], *, source_revision: str) -> dict[str, Any]:
    metadata = {
        "task_family": spec["task_family"],
        "difficulty": _normalize_difficulty(spec.get("difficulty", task.difficulty)),
        "curriculum_tier": spec["curriculum_tier"],
        "sql_shape": spec["sql_shape"],
        "grounding_requirement": spec["grounding_requirement"],
        "shortcut_status": spec["shortcut_status"],
        "tags": sorted(set([*task.tags, *spec["tags"]])),
    }
    verification = dict(spec["verification"])
    provenance = {
        "source_package": "livesqlbench-open-development",
        "source_revision": source_revision,
        "source_task_path": task.task_path,
        "created_by": "sqlbench_lab.livesqlbench_adapter",
        "teacher_model": spec.get("teacher_model"),
        "target_source": spec["target_source"],
    }
    return {
        "schema_version": "sql_train_example:v2",
        "row_id": f"livesqlbench::{task.task_id}",
        "source_benchmark": "livesqlbench",
        "source_split": spec["split"],
        "task_id": task.task_id,
        "db_id": task.db_id,
        "db_path": task.db_path,
        "dialect": task.dialect,
        "question": task.question,
        "schema_text": task.schema_text,
        "knowledge_text": task.knowledge_text,
        "column_value_notes": [],
        "schema_linking_notes": [],
        "target_sql": spec["target_sql"],
        "task_type": spec.get("task_type", task.task_type),
        "metadata": metadata,
        "provenance": provenance,
        "verification": verification,
    }


def _as_eval_case(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "sql_eval_case:v2",
        "case_id": f"livesqlbench::{row['task_id']}",
        "source_benchmark": row["source_benchmark"],
        "source_split": row["source_split"],
        "task_id": row["task_id"],
        "fixture_id": row["task_id"],
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
        "order_sensitive": bool(spec.get("order_sensitive", False)),
        "numeric_tolerance": float(spec.get("numeric_tolerance", 0.0)),
    }


def _write_jsonl_after_validation(path: str | Path, rows: list[dict[str, Any]], *, kind: str) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_name(f".{resolved.name}.tmp")
    temp_path.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    try:
        if kind == "train":
            load_sql_train_examples(temp_path)
        elif kind == "eval":
            load_sql_eval_cases(temp_path)
        else:
            raise ValueError(f"unsupported artifact kind: {kind}")
        temp_path.replace(resolved)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return resolved


def _required_text(payload: dict[str, Any], field: str, path: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string in {path}")
    return value.strip()


def _dialect_from_tags(tags: tuple[str, ...], task_path: Path) -> str:
    dialects = {tag for tag in tags if tag in {"postgresql", "sqlite"}}
    if len(dialects) != 1:
        raise ValueError(f"task must declare exactly one supported dialect in task.toml: {task_path}")
    return next(iter(dialects))


def _single_file(directory: Path, name: str) -> Path:
    exact = directory / name
    if exact.exists():
        return exact
    matches = sorted(directory.glob("*_schema.txt"))
    if len(matches) != 1:
        raise ValueError(f"could not identify one schema file under {directory}")
    return matches[0]


def _sqlite_path(task_path: Path) -> Path:
    matches = sorted((task_path / "environment").rglob("*.sqlite"))
    if len(matches) != 1:
        raise ValueError(f"SQLite task must expose exactly one sqlite database: {task_path}")
    return matches[0]


def _read_knowledge(db_assets: Path, db_id: str) -> str | None:
    paths = sorted([*db_assets.glob(f"{db_id}_column_meaning*.json"), *db_assets.glob(f"{db_id}_kb*.jsonl")])
    if not paths:
        return None
    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


def _normalize_difficulty(value: Any) -> str:
    normalized = str(value or "unknown").casefold()
    if normalized in {"simple", "easy"}:
        return "simple"
    if normalized in {"medium", "moderate"}:
        return "medium"
    if normalized in {"hard", "difficult"}:
        return "hard"
    return "unknown"
