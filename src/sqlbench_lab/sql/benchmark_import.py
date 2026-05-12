"""Import real text-to-SQL benchmark rows from Hugging Face snapshots."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT

BENCHMARK_REPOS = {
    "bird": "premai-io/birdbench",
    "spider": "premai-io/spider",
}
DEFAULT_CACHE_ROOT = WORKSPACE_ROOT / "external" / "sql" / "benchmarks"


@dataclass(frozen=True)
class SQLBenchmarkImportSummary:
    benchmark: str
    split: str
    artifact: str
    source_repo: str
    source_root: str
    output_path: str
    row_count: int
    selection: str


def import_sql_benchmark(
    *,
    benchmark: str,
    split: str,
    artifact: str,
    output_path: str | Path,
    limit: int | None = None,
    selection: str = "first",
    db_ids: tuple[str, ...] = (),
    cache_root: str | Path | None = None,
    force_download: bool = False,
) -> SQLBenchmarkImportSummary:
    """Download/cache a PremSQL benchmark and export rows in this repo's JSONL format."""

    if benchmark not in BENCHMARK_REPOS:
        raise ValueError(f"benchmark must be one of {sorted(BENCHMARK_REPOS)}")
    if artifact not in {"train", "eval"}:
        raise ValueError("artifact must be either 'train' or 'eval'")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if selection not in {"first", "stratified"}:
        raise ValueError("selection must be either 'first' or 'stratified'")

    source_root = _ensure_snapshot(
        benchmark=benchmark,
        cache_root=Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT,
        force_download=force_download,
    )
    raw_rows, dataset_root, db_folder_name, json_path = _load_raw_rows(benchmark, source_root, split)
    indexed_rows = list(enumerate(raw_rows, start=1))
    filtered_rows = _filter_indexed_raw_rows_by_db_id(indexed_rows, db_ids=db_ids)
    selected_rows = _select_indexed_raw_rows(filtered_rows, limit=limit, selection=selection)
    rows = [
        _convert_raw_row(
            row,
            benchmark=benchmark,
            split=split,
            artifact=artifact,
            ordinal=ordinal,
            dataset_root=dataset_root,
            db_folder_name=db_folder_name,
            source_path=json_path,
        )
        for ordinal, row in selected_rows
    ]
    resolved_output = _resolve_workspace_path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return SQLBenchmarkImportSummary(
        benchmark=benchmark,
        split=split,
        artifact=artifact,
        source_repo=BENCHMARK_REPOS[benchmark],
        source_root=str(source_root),
        output_path=str(_display_path(resolved_output)),
        row_count=len(rows),
        selection=selection,
    )


def _filter_raw_rows_by_db_id(
    raw_rows: list[dict[str, Any]],
    *,
    db_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [row for _, row in _filter_indexed_raw_rows_by_db_id(list(enumerate(raw_rows, start=1)), db_ids=db_ids)]


def _filter_indexed_raw_rows_by_db_id(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    *,
    db_ids: tuple[str, ...],
) -> list[tuple[int, dict[str, Any]]]:
    if not db_ids:
        return indexed_rows
    normalized = {db_id.strip() for db_id in db_ids if db_id.strip()}
    if not normalized:
        return indexed_rows
    filtered = [item for item in indexed_rows if str(item[1]["db_id"]) in normalized]
    if not filtered:
        raise ValueError(f"no benchmark rows found for db_id filter: {', '.join(sorted(normalized))}")
    return filtered


def _select_raw_rows(
    raw_rows: list[dict[str, Any]],
    *,
    limit: int | None,
    selection: str,
) -> list[tuple[int, dict[str, Any]]]:
    return _select_indexed_raw_rows(
        list(enumerate(raw_rows, start=1)),
        limit=limit,
        selection=selection,
    )


def _select_indexed_raw_rows(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    *,
    limit: int | None,
    selection: str,
) -> list[tuple[int, dict[str, Any]]]:
    if limit is None:
        return indexed_rows
    if selection == "first":
        return indexed_rows[:limit]
    if selection != "stratified":
        raise ValueError("selection must be either 'first' or 'stratified'")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    group_order: list[str] = []
    for item in indexed_rows:
        _, row = item
        db_id = str(row["db_id"])
        if db_id not in grouped:
            grouped[db_id] = []
            group_order.append(db_id)
        grouped[db_id].append(item)

    selected: list[tuple[int, dict[str, Any]]] = []
    while len(selected) < limit:
        added_this_round = False
        for db_id in group_order:
            rows = grouped[db_id]
            if rows:
                selected.append(rows.pop(0))
                added_this_round = True
                if len(selected) >= limit:
                    break
        if not added_this_round:
            break
    return selected


def _ensure_snapshot(*, benchmark: str, cache_root: Path, force_download: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Benchmark import requires huggingface_hub. Run with the training dependency group."
        ) from exc

    repo_id = BENCHMARK_REPOS[benchmark]
    local_dir = cache_root / repo_id.replace("/", "__")
    if local_dir.exists() and not force_download:
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        force_download=force_download,
    )
    return local_dir


def _load_raw_rows(
    benchmark: str,
    source_root: Path,
    split: str,
) -> tuple[list[dict[str, Any]], Path, str, Path]:
    if benchmark == "bird":
        if split not in {"train", "validation"}:
            raise ValueError("BIRD split must be train or validation")
        dataset_root = source_root / split
        json_path = dataset_root / ("train.json" if split == "train" else "validation.json")
        db_folder_name = "train_databases" if split == "train" else "dev_databases"
    elif benchmark == "spider":
        if split not in {"train", "validation"}:
            raise ValueError("Spider split must be train or validation")
        dataset_root = source_root
        json_path = source_root / ("train.json" if split == "train" else "validation.json")
        db_folder_name = "database"
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON array in {json_path}")
    return [row for row in rows if isinstance(row, dict)], dataset_root, db_folder_name, json_path


def _convert_raw_row(
    row: dict[str, Any],
    *,
    benchmark: str,
    split: str,
    artifact: str,
    ordinal: int,
    dataset_root: Path,
    db_folder_name: str,
    source_path: Path,
) -> dict[str, Any]:
    db_id = str(row["db_id"])
    db_path = dataset_root / db_folder_name / db_id / f"{db_id}.sqlite"
    schema_text = _schema_text(db_path)
    question = str(row["question"])
    sql = str(row["SQL"] if benchmark == "bird" else row["query"])
    knowledge = row.get("evidence") if benchmark == "bird" else None
    source_split = "dev" if split == "validation" else split
    common = {
        "source_benchmark": benchmark,
        "source_split": source_split,
        "task_id": f"{benchmark}_{split}_{ordinal:05d}",
        "db_id": db_id,
        "dialect": "sqlite",
        "question": question,
        "schema_text": schema_text,
        "knowledge_text": str(knowledge) if knowledge else None,
        "task_type": _task_type(sql),
        "tags": _tags(row, benchmark=benchmark, split=split),
    }
    if artifact == "train":
        return {
            "schema_version": "sql_train_example:v1",
            "row_id": f"{benchmark}_{split}_{ordinal:05d}",
            **common,
            "db_path": str(_display_path(db_path)),
            "target_sql": sql,
            "provenance": {
                "created_by": "benchmark",
                "teacher_model": None,
                "source_path": str(_display_path(source_path)),
            },
        }
    return {
        "schema_version": "sql_eval_case:v1",
        "case_id": f"{benchmark}_{split}_{ordinal:05d}",
        "fixture_id": f"{benchmark}:{db_id}",
        "db_path": str(_display_path(db_path)),
        **common,
        "gold_sql": sql,
        "order_sensitive": False,
        "numeric_tolerance": 0.000001,
    }


def _schema_text(db_path: Path) -> str:
    if not db_path.exists():
        raise ValueError(f"SQLite database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name"
        ).fetchall()
    statements = [str(row[0]) for row in rows if row[0]]
    if not statements:
        raise ValueError(f"SQLite database has no table schema: {db_path}")
    return "\n".join(statements)


def _task_type(sql: str) -> str:
    first = sql.strip().split(maxsplit=1)[0].lower() if sql.strip() else ""
    if first == "select":
        return "select"
    if first in {"alter", "create", "delete", "drop", "insert", "update"}:
        return "management"
    return "unknown"


def _tags(row: dict[str, Any], *, benchmark: str, split: str) -> list[str]:
    tags = [benchmark, f"split_{split}"]
    for key in ("difficulty", "difficulty_tier"):
        value = row.get(key)
        if isinstance(value, str) and value:
            tags.append(f"difficulty_{value.lower()}")
    return tags


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return WORKSPACE_ROOT / candidate


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return path
