"""Attach compact SQLite profile metadata to SQL datasets."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT
from sqlbench_lab.shared import read_jsonl_objects


TEXT_NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
DATE_LIKE_RE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
SAFE_SQLITE_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class SQLProfileMetadataSummary:
    input_path: str
    output_path: str
    artifact: str
    row_count: int
    db_count: int
    total_note_count: int


@dataclass(frozen=True)
class ColumnProfile:
    table_name: str
    column_name: str
    declared_type: str
    row_count: int
    non_null_count: int
    distinct_count: int
    sample_values: tuple[str, ...]
    min_value: str | None
    max_value: str | None
    is_primary_key: bool
    foreign_key_target: tuple[str, str] | None
    exact_profile: bool
    sampled_non_null_count: int


@dataclass(frozen=True)
class DatabaseProfile:
    columns: tuple[ColumnProfile, ...]


def attach_sqlite_profile_metadata(
    *,
    input_path: str | Path,
    output_path: str | Path,
    artifact: str,
    max_column_notes: int = 12,
    max_sample_values: int = 3,
    max_exact_profile_rows: int = 50_000,
    profile_sample_rows: int = 10_000,
) -> SQLProfileMetadataSummary:
    """Attach deterministic SQLite profile notes to train or eval JSONL rows."""

    if artifact not in {"train", "eval"}:
        raise ValueError("artifact must be either 'train' or 'eval'")
    if max_column_notes < 1:
        raise ValueError("max_column_notes must be at least 1")
    if max_sample_values < 1:
        raise ValueError("max_sample_values must be at least 1")
    if max_exact_profile_rows < 1:
        raise ValueError("max_exact_profile_rows must be at least 1")
    if profile_sample_rows < 1:
        raise ValueError("profile_sample_rows must be at least 1")

    resolved_input = _resolve_workspace_path(input_path)
    resolved_output = _resolve_workspace_path(output_path)
    rows = read_jsonl_objects(resolved_input)
    if not rows:
        raise ValueError("profile metadata input dataset must contain at least one row")

    profile_cache: dict[Path, DatabaseProfile] = {}
    enriched_rows: list[dict[str, Any]] = []
    db_paths: set[Path] = set()
    total_note_count = 0
    for row in rows:
        _validate_artifact_row(row, artifact=artifact)
        db_path = _row_db_path(row)
        db_paths.add(db_path)
        if db_path not in profile_cache:
            profile_cache[db_path] = _profile_sqlite_database(
                db_path,
                max_sample_values=max_sample_values,
                max_exact_profile_rows=max_exact_profile_rows,
                profile_sample_rows=profile_sample_rows,
            )
        notes = _notes_for_row(
            row,
            profile=profile_cache[db_path],
            max_column_notes=max_column_notes,
        )
        enriched = dict(row)
        enriched["column_value_notes"] = notes
        enriched_rows.append(enriched)
        total_note_count += len(notes)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        for row in enriched_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return SQLProfileMetadataSummary(
        input_path=str(_display_path(resolved_input)),
        output_path=str(_display_path(resolved_output)),
        artifact=artifact,
        row_count=len(enriched_rows),
        db_count=len(db_paths),
        total_note_count=total_note_count,
    )


def _validate_artifact_row(row: dict[str, Any], *, artifact: str) -> None:
    expected = "sql_train_example:v1" if artifact == "train" else "sql_eval_case:v1"
    if row.get("schema_version") != expected:
        raise ValueError(f"profile metadata expected {expected}, got {row.get('schema_version')!r}")
    if str(row.get("dialect", "")) != "sqlite":
        raise ValueError("profile metadata only supports sqlite rows")


def _row_db_path(row: dict[str, Any]) -> Path:
    raw_path = row.get("db_path")
    if not raw_path:
        raise ValueError(f"profile metadata row has no db_path: {row.get('task_id', '<unknown>')}")
    db_path = _resolve_workspace_path(str(raw_path))
    if not db_path.exists():
        raise ValueError(f"SQLite database not found: {db_path}")
    return db_path


def _profile_sqlite_database(
    db_path: Path,
    *,
    max_sample_values: int,
    max_exact_profile_rows: int,
    profile_sample_rows: int,
) -> DatabaseProfile:
    with sqlite3.connect(db_path) as conn:
        table_names = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name != 'sqlite_sequence' ORDER BY name"
            ).fetchall()
        ]
        if not table_names:
            raise ValueError(f"SQLite database has no tables: {db_path}")
        foreign_keys = _foreign_keys_by_column(conn, table_names=table_names)
        columns: list[ColumnProfile] = []
        for table_name in table_names:
            table_info = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
            row_count = int(
                conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}").fetchone()[0]
            )
            for column in table_info:
                column_name = str(column[1])
                columns.append(
                    _profile_column(
                        conn,
                        table_name=table_name,
                        column_name=column_name,
                        declared_type=str(column[2] or "UNKNOWN").upper(),
                        row_count=row_count,
                        is_primary_key=bool(column[5]),
                        foreign_key_target=foreign_keys.get((table_name, column_name)),
                        max_sample_values=max_sample_values,
                        max_exact_profile_rows=max_exact_profile_rows,
                        profile_sample_rows=profile_sample_rows,
                    )
                )
    return DatabaseProfile(columns=tuple(columns))


def _foreign_keys_by_column(
    conn: sqlite3.Connection,
    *,
    table_names: list[str],
) -> dict[tuple[str, str], tuple[str, str]]:
    foreign_keys: dict[tuple[str, str], tuple[str, str]] = {}
    for table_name in table_names:
        rows = conn.execute(f"PRAGMA foreign_key_list({_quote_identifier(table_name)})").fetchall()
        for row in rows:
            source_column = str(row[3])
            target_table = str(row[2])
            target_column = str(row[4])
            foreign_keys[(table_name, source_column)] = (target_table, target_column)
    return foreign_keys


def _profile_column(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    declared_type: str,
    row_count: int,
    is_primary_key: bool,
    foreign_key_target: tuple[str, str] | None,
    max_sample_values: int,
    max_exact_profile_rows: int,
    profile_sample_rows: int,
) -> ColumnProfile:
    quoted_table = _quote_identifier(table_name)
    quoted_column = _quote_identifier(column_name)
    if row_count > max_exact_profile_rows:
        return _sampled_column_profile(
            conn,
            table_name=table_name,
            column_name=column_name,
            declared_type=declared_type,
            row_count=row_count,
            is_primary_key=is_primary_key,
            foreign_key_target=foreign_key_target,
            max_sample_values=max_sample_values,
            profile_sample_rows=profile_sample_rows,
        )
    non_null_count = int(
        conn.execute(f"SELECT COUNT({quoted_column}) FROM {quoted_table}").fetchone()[0]
    )
    distinct_count = int(
        conn.execute(f"SELECT COUNT(DISTINCT {quoted_column}) FROM {quoted_table}").fetchone()[0]
    )
    sample_rows = conn.execute(
        f"""
        SELECT {quoted_column}
        FROM {quoted_table}
        WHERE {quoted_column} IS NOT NULL
        GROUP BY {quoted_column}
        ORDER BY COUNT(*) DESC, {quoted_column}
        LIMIT ?
        """,
        (max_sample_values,),
    ).fetchall()
    min_max = conn.execute(
        f"SELECT MIN({quoted_column}), MAX({quoted_column}) FROM {quoted_table} "
        f"WHERE {quoted_column} IS NOT NULL"
    ).fetchone()
    return ColumnProfile(
        table_name=table_name,
        column_name=column_name,
        declared_type=declared_type,
        row_count=row_count,
        non_null_count=non_null_count,
        distinct_count=distinct_count,
        sample_values=tuple(str(row[0]) for row in sample_rows if row[0] is not None),
        min_value=str(min_max[0]) if min_max and min_max[0] is not None else None,
        max_value=str(min_max[1]) if min_max and min_max[1] is not None else None,
        is_primary_key=is_primary_key,
        foreign_key_target=foreign_key_target,
        exact_profile=True,
        sampled_non_null_count=non_null_count,
    )


def _sampled_column_profile(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    declared_type: str,
    row_count: int,
    is_primary_key: bool,
    foreign_key_target: tuple[str, str] | None,
    max_sample_values: int,
    profile_sample_rows: int,
) -> ColumnProfile:
    quoted_table = _quote_identifier(table_name)
    quoted_column = _quote_identifier(column_name)
    rows = conn.execute(
        f"""
        SELECT {quoted_column}
        FROM {quoted_table}
        WHERE {quoted_column} IS NOT NULL
        LIMIT ?
        """,
        (profile_sample_rows,),
    ).fetchall()
    values = [str(row[0]) for row in rows if row[0] is not None]
    value_counts: dict[str, int] = {}
    for value in values:
        value_counts[value] = value_counts.get(value, 0) + 1
    sample_values = tuple(
        value
        for value, _ in sorted(value_counts.items(), key=lambda item: (-item[1], item[0]))[
            :max_sample_values
        ]
    )
    min_value = min(values) if values else None
    max_value = max(values) if values else None
    return ColumnProfile(
        table_name=table_name,
        column_name=column_name,
        declared_type=declared_type,
        row_count=row_count,
        non_null_count=-1,
        distinct_count=len(value_counts),
        sample_values=sample_values,
        min_value=min_value,
        max_value=max_value,
        is_primary_key=is_primary_key,
        foreign_key_target=foreign_key_target,
        exact_profile=False,
        sampled_non_null_count=len(values),
    )


def _notes_for_row(
    row: dict[str, Any],
    *,
    profile: DatabaseProfile,
    max_column_notes: int,
) -> list[str]:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("db_id", "question", "knowledge_text")
    ).lower()
    scored_columns = sorted(
        profile.columns,
        key=lambda column: (-_column_relevance(column, text), column.table_name, column.column_name),
    )
    selected = [
        column
        for column in scored_columns
        if _column_relevance(column, text) > 0 or column.is_primary_key or column.foreign_key_target
    ][:max_column_notes]
    if not selected:
        selected = list(scored_columns[:max_column_notes])
    return [_column_note(column) for column in selected]


def _column_relevance(column: ColumnProfile, text: str) -> int:
    score = 0
    table_tokens = _identifier_tokens(column.table_name)
    column_tokens = _identifier_tokens(column.column_name)
    if _contains_identifier_phrase(text, column.column_name):
        score += 8
    if _contains_identifier_phrase(text, column.table_name):
        score += 3
    score += 2 * sum(1 for token in column_tokens if token in text)
    score += sum(1 for token in table_tokens if token in text)
    if column.is_primary_key:
        score += 1
    if column.foreign_key_target:
        score += 2
    if any(str(value).lower() in text for value in column.sample_values if len(str(value)) >= 3):
        score += 6
    if column.min_value and len(column.min_value) >= 3 and column.min_value.lower() in text:
        score += 3
    if column.max_value and len(column.max_value) >= 3 and column.max_value.lower() in text:
        score += 3
    return score


def _column_note(column: ColumnProfile) -> str:
    ref = f"`{column.table_name}`.`{column.column_name}`"
    if column.exact_profile:
        coverage = f"{column.non_null_count}/{column.row_count} non-NULL"
        distinct = f"{column.distinct_count} distinct"
    else:
        coverage = (
            f"sampled {column.sampled_non_null_count} non-NULL values "
            f"from {column.row_count} rows"
        )
        distinct = f"{column.distinct_count} distinct values in sample"
    parts = [
        f"{ref}: declared {column.declared_type}",
        coverage,
        distinct,
    ]
    if column.is_primary_key:
        parts.append("primary key")
    if column.foreign_key_target:
        target_table, target_column = column.foreign_key_target
        parts.append(f"joins `{target_table}`.`{target_column}`")
    if column.min_value is not None and column.max_value is not None:
        parts.append(f"range {column.min_value} to {column.max_value}")
    if column.sample_values:
        parts.append("sample values: " + ", ".join(_format_sample(value) for value in column.sample_values))
    if _requires_quoting(column.column_name) or _requires_quoting(column.table_name):
        parts.append("quote this identifier with backticks")
    if _looks_text_numeric(column):
        parts.append("text numeric values may need CAST(REPLACE(..., ',', '') AS REAL)")
    if _looks_date_like(column):
        parts.append("date-like values should be matched in their stored string format")
    return "; ".join(parts) + "."


def _looks_text_numeric(column: ColumnProfile) -> bool:
    return "TEXT" in column.declared_type and bool(column.sample_values) and all(
        TEXT_NUMERIC_RE.fullmatch(value) for value in column.sample_values
    )


def _looks_date_like(column: ColumnProfile) -> bool:
    values = column.sample_values
    return bool(values) and all(DATE_LIKE_RE.search(value) for value in values)


def _identifier_tokens(identifier: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", identifier.lower())
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(part for part in re.findall(r"[a-z]+|[0-9]+", token) if part != token)
    return tuple(dict.fromkeys(expanded))


def _contains_identifier_phrase(text: str, identifier: str) -> bool:
    normalized_identifier = " ".join(_identifier_tokens(identifier))
    return bool(normalized_identifier) and normalized_identifier in text


def _format_sample(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _requires_quoting(identifier: str) -> bool:
    return SAFE_SQLITE_IDENTIFIER_RE.fullmatch(identifier) is None


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


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
