"""Attach schema-linking notes to SQL train/eval rows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT
from sqlbench_lab.shared import read_jsonl_objects

SCHEMA_LINKING_MODES = {"gold_sql", "question"}


@dataclass(frozen=True)
class SQLSchemaLinkingSummary:
    """Summary of a schema-linking annotation pass."""

    artifact: str
    mode: str
    input_path: str
    output_path: str
    row_count: int
    total_note_count: int


@dataclass(frozen=True)
class SchemaColumn:
    table: str
    name: str


def attach_sql_schema_linking(
    *,
    input_path: str | Path,
    output_path: str | Path,
    artifact: str,
    mode: str,
    max_tables: int = 6,
    max_columns: int = 16,
) -> SQLSchemaLinkingSummary:
    """Attach deterministic schema-linking notes to SQL train/eval JSONL."""

    if artifact not in {"train", "eval"}:
        raise ValueError("artifact must be train or eval")
    if mode not in SCHEMA_LINKING_MODES:
        raise ValueError(f"mode must be one of {sorted(SCHEMA_LINKING_MODES)}")
    if artifact == "eval" and mode == "gold_sql":
        raise ValueError("eval schema linking must not use gold_sql mode")
    if max_tables < 1:
        raise ValueError("max_tables must be at least 1")
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1")

    resolved_input = _resolve_workspace_path(input_path)
    resolved_output = _resolve_workspace_path(output_path)
    rows = read_jsonl_objects(resolved_input)
    output_rows = [
        _attach_notes(row, artifact=artifact, mode=mode, max_tables=max_tables, max_columns=max_columns)
        for row in rows
    ]
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return SQLSchemaLinkingSummary(
        artifact=artifact,
        mode=mode,
        input_path=str(_display_path(resolved_input)),
        output_path=str(_display_path(resolved_output)),
        row_count=len(output_rows),
        total_note_count=sum(len(row.get("schema_linking_notes", [])) for row in output_rows),
    )


def _attach_notes(
    row: dict[str, Any],
    *,
    artifact: str,
    mode: str,
    max_tables: int,
    max_columns: int,
) -> dict[str, Any]:
    schema = _parse_schema(str(row["schema_text"]))
    sql = _sql_for_row(row, artifact=artifact, mode=mode)
    text = _question_text(row)
    result_shape = _result_shape(sql=sql, text=text)
    tables = _rank_tables(schema=schema, sql=sql, text=text, mode=mode)[:max_tables]
    columns = _rank_columns(schema=schema, sql=sql, text=text, tables=tables, mode=mode)[:max_columns]
    notes = [f"Result shape: {result_shape}."]
    if tables:
        notes.append("Relevant tables: " + ", ".join(tables) + ".")
    if columns:
        notes.append("Relevant columns: " + ", ".join(f"{column.table}.{column.name}" for column in columns) + ".")
    output = dict(row)
    output["schema_linking_notes"] = notes
    output["tags"] = [*list(row.get("tags", [])), f"schema_linking_{mode}"]
    return output


def _parse_schema(schema_text: str) -> dict[str, list[SchemaColumn]]:
    schema: dict[str, list[SchemaColumn]] = {}
    for match in re.finditer(
        r"CREATE\s+TABLE\s+(?P<table>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|\w+)\s*\((?P<body>.*?)\)\s*(?=CREATE|$)",
        schema_text + "\nCREATE",
        flags=re.IGNORECASE | re.DOTALL,
    ):
        table = _clean_identifier(match.group("table"))
        columns: list[SchemaColumn] = []
        for raw_line in re.split(r",|\n", match.group("body")):
            stripped = raw_line.strip().rstrip(",")
            if not stripped or stripped.upper().startswith(
                ("PRIMARY ", "FOREIGN ", "UNIQUE ", "CHECK ", "CONSTRAINT ", "ON ")
            ):
                continue
            column_match = re.match(r"(?P<column>`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|\w+)", stripped)
            if column_match is None:
                continue
            column = _clean_identifier(column_match.group("column"))
            columns.append(SchemaColumn(table=table, name=column))
        schema[table] = columns
    return schema


def _sql_for_row(row: dict[str, Any], *, artifact: str, mode: str) -> str:
    if mode == "question":
        return ""
    if artifact == "train":
        return str(row["target_sql"])
    return str(row["gold_sql"])


def _question_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("question", "")),
        str(row.get("knowledge_text") or ""),
        " ".join(str(note) for note in row.get("column_value_notes", [])),
    ]
    return " ".join(part for part in parts if part.strip())


def _rank_tables(*, schema: dict[str, list[SchemaColumn]], sql: str, text: str, mode: str) -> list[str]:
    sql_norm = _normalize(sql)
    text_norm = _normalize(text)
    scored: list[tuple[int, str]] = []
    for table in schema:
        score = 0
        table_norm = _normalize(table)
        if mode == "gold_sql" and _contains_identifier(sql_norm, table_norm):
            score += 100
        score += _token_overlap_score(table_norm, text_norm)
        for column in schema[table]:
            score += _token_overlap_score(_normalize(column.name), text_norm)
        if score > 0:
            scored.append((-score, table))
    return [table for _, table in sorted(scored)]


def _rank_columns(
    *,
    schema: dict[str, list[SchemaColumn]],
    sql: str,
    text: str,
    tables: list[str],
    mode: str,
) -> list[SchemaColumn]:
    sql_norm = _normalize(sql)
    text_norm = _normalize(text)
    table_rank = {table: index for index, table in enumerate(tables)}
    scored: list[tuple[int, int, str, str, SchemaColumn]] = []
    for table, columns in schema.items():
        for column in columns:
            column_norm = _normalize(column.name)
            score = 0
            if mode == "gold_sql" and _contains_identifier(sql_norm, column_norm):
                score += 100
            score += _token_overlap_score(column_norm, text_norm) * 5
            if table in table_rank:
                score += 2
            if score > 0:
                scored.append((-score, table_rank.get(table, 999), table, column.name, column))
    return [column for *_unused, column in sorted(scored)]


def _result_shape(*, sql: str, text: str) -> str:
    combined = f"{sql} {text}".lower()
    if re.search(r"\bcount\s*\(|\bhow many\b|\bnumber of\b", combined):
        return "count"
    if re.search(r"\bsum\s*\(|\btotal\b", combined):
        return "sum"
    if re.search(r"\bavg\s*\(|\baverage\b|\bmean\b", combined):
        return "average"
    if re.search(r"\bmin\s*\(|\bminimum\b|\blowest\b|\bleast\b", combined):
        return "minimum"
    if re.search(r"\bmax\s*\(|\bmaximum\b|\bhighest\b|\bmost\b", combined):
        return "maximum"
    if re.search(r"\bgroup\s+by\b|\bper\b|\bfor each\b", combined):
        return "grouped"
    if re.search(r"\border\s+by\b|\bsort\b|\bordered\b", combined):
        return "ordered_list"
    return "select"


def _contains_identifier(haystack: str, identifier: str) -> bool:
    if not identifier:
        return False
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(identifier)}(?![a-z0-9_])", haystack))


def _token_overlap_score(identifier: str, text: str) -> int:
    tokens = [token for token in identifier.split() if len(token) > 1]
    return sum(1 for token in tokens if _contains_identifier(text, token))


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _clean_identifier(value: str) -> str:
    stripped = value.strip()
    if stripped[:1] in {'"', "`", "["} and stripped[-1:] in {'"', "`", "]"}:
        return stripped[1:-1]
    return stripped


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
