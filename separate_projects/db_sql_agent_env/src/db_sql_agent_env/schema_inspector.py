"""Read-only SQLite schema inspection for DB-specific agent cold starts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote


@dataclass(frozen=True)
class SQLiteColumnProfile:
    name: str
    type: str
    not_null: bool
    default_value: str | None
    primary_key_position: int


@dataclass(frozen=True)
class SQLiteForeignKeyProfile:
    column: str
    references_table: str
    references_column: str
    on_update: str
    on_delete: str


@dataclass(frozen=True)
class SQLiteIndexColumnProfile:
    name: str
    sequence: int


@dataclass(frozen=True)
class SQLiteIndexProfile:
    name: str
    unique: bool
    origin: str
    columns: list[SQLiteIndexColumnProfile]


@dataclass(frozen=True)
class SQLiteTableProfile:
    name: str
    row_count: int
    create_sql: str
    columns: list[SQLiteColumnProfile]
    foreign_keys: list[SQLiteForeignKeyProfile]
    indexes: list[SQLiteIndexProfile]


@dataclass(frozen=True)
class SQLiteSchemaProfile:
    dialect: str
    db_path: str
    table_count: int
    tables: list[SQLiteTableProfile]


def inspect_sqlite_schema(
    db_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> SQLiteSchemaProfile:
    """Inspect SQLite schema metadata using a read-only connection."""

    resolved_db_path = _resolve_db_path(db_path)
    with _connect_readonly(resolved_db_path) as conn:
        table_rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        tables = [_inspect_table(conn, name=str(name), create_sql=str(create_sql or "")) for name, create_sql in table_rows]
    profile = SQLiteSchemaProfile(
        dialect="sqlite",
        db_path=str(resolved_db_path),
        table_count=len(tables),
        tables=tables,
    )
    if output_path is not None:
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(
            json.dumps(asdict(profile), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return profile


def _inspect_table(sqlite_conn: sqlite3.Connection, *, name: str, create_sql: str) -> SQLiteTableProfile:
    row_count = int(sqlite_conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(name)}").fetchone()[0])
    columns = [
        SQLiteColumnProfile(
            name=str(row[1]),
            type=str(row[2]),
            not_null=bool(row[3]),
            default_value=None if row[4] is None else str(row[4]),
            primary_key_position=int(row[5]),
        )
        for row in sqlite_conn.execute(f"PRAGMA table_info({_quote_pragma_string(name)})").fetchall()
    ]
    foreign_keys = [
        SQLiteForeignKeyProfile(
            column=str(row[3]),
            references_table=str(row[2]),
            references_column=str(row[4]),
            on_update=str(row[5]),
            on_delete=str(row[6]),
        )
        for row in sqlite_conn.execute(f"PRAGMA foreign_key_list({_quote_pragma_string(name)})").fetchall()
    ]
    indexes = [
        _inspect_index(sqlite_conn, index_name=str(row[1]), unique=bool(row[2]), origin=str(row[3]))
        for row in sqlite_conn.execute(f"PRAGMA index_list({_quote_pragma_string(name)})").fetchall()
    ]
    return SQLiteTableProfile(
        name=name,
        row_count=row_count,
        create_sql=create_sql,
        columns=columns,
        foreign_keys=foreign_keys,
        indexes=indexes,
    )


def _inspect_index(
    sqlite_conn: sqlite3.Connection,
    *,
    index_name: str,
    unique: bool,
    origin: str,
) -> SQLiteIndexProfile:
    columns = [
        SQLiteIndexColumnProfile(
            name=str(row[2]),
            sequence=int(row[0]),
        )
        for row in sqlite_conn.execute(f"PRAGMA index_info({_quote_pragma_string(index_name)})").fetchall()
    ]
    return SQLiteIndexProfile(
        name=index_name,
        unique=unique,
        origin=origin,
        columns=columns,
    )


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _resolve_db_path(db_path: str | Path) -> Path:
    path = Path(db_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"db_path does not exist: {path}")
    return path.resolve()


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_pragma_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
