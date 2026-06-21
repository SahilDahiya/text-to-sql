"""Local natural-language SQL query service for the storefront adapter."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sqlbench_lab.sql.eval_runner import extract_generated_sql
from sqlbench_lab.sql.loaders import load_sql_eval_cases
from sqlbench_lab.sql.manifest import load_sql_sft_manifest
from sqlbench_lab.sql.rendering import SQL_SYSTEM_PROMPT, build_eval_messages
from sqlbench_lab.sql.serving import OpenAICompletionClient, OpenAICompletionTransport
from sqlbench_lab.sql.training import render_sql_sft_prompt


@dataclass(frozen=True)
class SQLAskAppConfig:
    manifest_path: str | Path
    schema_source_path: str | Path
    db_path: str | Path
    openai_base_url: str
    openai_model: str
    row_limit: int = 100
    timeout_seconds: float = 60.0
    max_new_tokens: int = 128
    system_prompt: str = SQL_SYSTEM_PROMPT


@dataclass(frozen=True)
class SQLExecutionResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool


@dataclass(frozen=True)
class SQLAskQueryResult:
    question: str
    generated_sql: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    row_limit: int
    latency_seconds: float
    error: str | None
    openai_base_url: str
    openai_model: str
    db_path: str

    @property
    def row_count(self) -> int:
        return len(self.rows)


class SQLAskQueryService:
    """Generate SQL through the local endpoint and execute it read-only."""

    def __init__(
        self,
        config: SQLAskAppConfig,
        *,
        transport: OpenAICompletionTransport | None = None,
    ) -> None:
        self.config = _validated_config(config)
        self._manifest = load_sql_sft_manifest(self.config.manifest_path)
        schema_cases = load_sql_eval_cases(self.config.schema_source_path)
        if not schema_cases:
            raise ValueError(f"schema_source_path has no eval cases: {self.config.schema_source_path}")
        self._schema_template = schema_cases[0]
        self._client = OpenAICompletionClient(
            base_url=str(self.config.openai_base_url),
            model=str(self.config.openai_model),
            max_new_tokens=self.config.max_new_tokens,
            timeout_seconds=self.config.timeout_seconds,
            temperature=0.0,
            transport=transport,
        )

    def ask(self, question: str) -> SQLAskQueryResult:
        resolved_question = _non_empty(question, "question")
        started = time.perf_counter()
        generated_sql = ""
        try:
            generated_sql = self._generate_sql(resolved_question)
            execution = execute_readonly_select(
                self.config.db_path,
                generated_sql,
                row_limit=self.config.row_limit,
            )
            error = None
        except Exception as exc:
            execution = SQLExecutionResult(columns=(), rows=(), truncated=False)
            error = str(exc)
        latency = time.perf_counter() - started
        return SQLAskQueryResult(
            question=resolved_question,
            generated_sql=generated_sql,
            columns=execution.columns,
            rows=execution.rows,
            truncated=execution.truncated,
            row_limit=self.config.row_limit,
            latency_seconds=latency,
            error=error,
            openai_base_url=str(self.config.openai_base_url),
            openai_model=str(self.config.openai_model),
            db_path=str(self.config.db_path),
        )

    def _generate_sql(self, question: str) -> str:
        case = replace(
            self._schema_template,
            case_id="manual_storefront_query",
            source_split="manual",
            task_id="manual_storefront_query",
            db_path=str(self.config.db_path),
            question=question,
            gold_sql="SELECT 1",
            tags=(*self._schema_template.tags, "manual_query"),
        )
        messages = build_eval_messages(
            case,
            prompt_style=self._manifest.prompt.style,
            system_prompt=self.config.system_prompt,
        )
        prompt = render_sql_sft_prompt([*messages, {"role": "assistant", "content": ""}])
        generated_sql = extract_generated_sql(self._client.complete(prompt))
        if not generated_sql.strip():
            raise ValueError("model returned empty SQL")
        return generated_sql


def execute_readonly_select(
    db_path: str | Path,
    sql: str,
    *,
    row_limit: int,
) -> SQLExecutionResult:
    """Execute one SELECT/WITH SQLite statement using a read-only connection."""

    resolved_db_path = _existing_file(db_path, "db_path")
    resolved_row_limit = _positive_int(row_limit, "row_limit")
    resolved_sql = _validated_select_sql(sql)
    uri = f"file:{quote(str(resolved_db_path.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(resolved_sql)
        columns = tuple(description[0] for description in cursor.description or ())
        raw_rows = cursor.fetchmany(resolved_row_limit + 1)
    truncated = len(raw_rows) > resolved_row_limit
    rows = tuple(tuple(row) for row in raw_rows[:resolved_row_limit])
    return SQLExecutionResult(columns=columns, rows=rows, truncated=truncated)


def _validated_select_sql(sql: str) -> str:
    resolved_sql = _non_empty(sql, "sql").strip()
    without_trailing_semicolon = resolved_sql[:-1].strip() if resolved_sql.endswith(";") else resolved_sql
    if ";" in without_trailing_semicolon:
        raise ValueError("query app executes exactly one SQL statement")
    first_token = re.match(r"^\s*([A-Za-z]+)", resolved_sql)
    if first_token is None or first_token.group(1).upper() not in {"SELECT", "WITH"}:
        raise ValueError("query app only SELECT or WITH statements")
    return resolved_sql


def _validated_config(config: SQLAskAppConfig) -> SQLAskAppConfig:
    row_limit = _positive_int(config.row_limit, "row_limit")
    max_new_tokens = _positive_int(config.max_new_tokens, "max_new_tokens")
    timeout_seconds = _positive_float(config.timeout_seconds, "timeout_seconds")
    return replace(
        config,
        manifest_path=_existing_file(config.manifest_path, "manifest_path"),
        schema_source_path=_existing_file(config.schema_source_path, "schema_source_path"),
        db_path=_existing_file(config.db_path, "db_path"),
        openai_base_url=_non_empty(config.openai_base_url, "openai_base_url"),
        openai_model=_non_empty(config.openai_model, "openai_model"),
        row_limit=row_limit,
        timeout_seconds=timeout_seconds,
        max_new_tokens=max_new_tokens,
        system_prompt=_non_empty(config.system_prompt, "system_prompt"),
    )


def _existing_file(path: str | Path, name: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _positive_int(value: int, name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _positive_float(value: float, name: str) -> float:
    resolved = float(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _non_empty(value: str, name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    return resolved
