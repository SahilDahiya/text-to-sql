"""Prompt rendering for SQL SFT rows."""

from __future__ import annotations

from .models import SQLEvalCase, SQLRepairExample, SQLTrainExample

SQL_SYSTEM_PROMPT = (
    "You are a precise text-to-SQL model. Return only the final SQL statement. "
    "Use the declared SQL dialect and stay grounded in the provided schema. "
    "Use table and column names exactly as they appear in the schema. "
    "For SQLite identifiers containing spaces, punctuation, parentheses, percent signs, "
    "hyphens, or question marks, quote the full identifier with backticks."
)
PREMSQL_PROMPT_STYLE = "premsql_text"
CANONICAL_PROMPT_STYLE = "canonical_chat"
SUPPORTED_PROMPT_STYLES = {CANONICAL_PROMPT_STYLE, PREMSQL_PROMPT_STYLE}


def build_train_messages(
    example: SQLTrainExample,
    *,
    prompt_style: str = CANONICAL_PROMPT_STYLE,
    system_prompt: str = SQL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _base_user_content(example, prompt_style=prompt_style)},
        {"role": "assistant", "content": example.target_sql},
    ]


def build_repair_messages(
    example: SQLRepairExample,
    *,
    prompt_style: str = CANONICAL_PROMPT_STYLE,
    system_prompt: str = SQL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    user_content = "\n\n".join(
        [
            _base_user_content(example, prompt_style=prompt_style),
            f"Previous SQL:\n{example.previous_sql}",
            f"Execution Error:\n{example.execution_error}",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example.target_sql},
    ]


def build_eval_messages(
    case: SQLEvalCase,
    *,
    prompt_style: str = CANONICAL_PROMPT_STYLE,
    system_prompt: str = SQL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _base_user_content(case, prompt_style=prompt_style)},
    ]


def build_repair_eval_messages(
    case: SQLEvalCase,
    *,
    previous_sql: str,
    execution_observation: str,
    prompt_style: str = CANONICAL_PROMPT_STYLE,
    system_prompt: str = SQL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    user_content = "\n\n".join(
        [
            _base_user_content(case, prompt_style=prompt_style),
            f"Previous SQL:\n{previous_sql}",
            f"Execution Observation:\n{execution_observation}",
            "Return corrected SQL only.",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _base_user_content(
    example: SQLEvalCase | SQLTrainExample | SQLRepairExample,
    *,
    prompt_style: str = CANONICAL_PROMPT_STYLE,
) -> str:
    if prompt_style == CANONICAL_PROMPT_STYLE:
        return _canonical_user_content(example)
    if prompt_style == PREMSQL_PROMPT_STYLE:
        return _premsql_user_content(example)
    raise ValueError(f"unsupported SQL prompt_style: {prompt_style}")


def _canonical_user_content(example: SQLEvalCase | SQLTrainExample | SQLRepairExample) -> str:
    sections = [
        f"Dialect:\n{example.dialect}",
        f"Database ID:\n{example.db_id}",
        f"Schema:\n{example.schema_text}",
    ]
    if example.knowledge_text:
        sections.append(f"Knowledge:\n{example.knowledge_text}")
    if example.column_value_notes:
        sections.append("Column value notes:\n" + "\n".join(f"- {note}" for note in example.column_value_notes))
    if example.schema_linking_notes:
        sections.append("Schema linking notes:\n" + "\n".join(f"- {note}" for note in example.schema_linking_notes))
    sections.append(f"Question:\n{example.question}")
    return "\n\n".join(sections)


def _premsql_user_content(example: SQLEvalCase | SQLTrainExample | SQLRepairExample) -> str:
    additional_knowledge = (
        f"# Additional Knowledge:\n{example.knowledge_text}\n"
        if example.knowledge_text
        else ""
    )
    column_value_notes = (
        "# Column Value Notes:\n"
        + "\n".join(f"- {note}" for note in example.column_value_notes)
        + "\n\n"
        if example.column_value_notes
        else ""
    )
    schema_linking_notes = (
        "# Schema Linking Notes:\n"
        + "\n".join(f"- {note}" for note in example.schema_linking_notes)
        + "\n\n"
        if example.schema_linking_notes
        else ""
    )
    return (
        "# Follow these instruction:\n"
        "You will be given schemas of tables of a database. Your job is to write correct\n"
        "error free SQL query based on the question asked. Please make sure:\n\n"
        "1. Do not add ``` at start / end of the query. It should be a single line query in a single line.\n"
        "2. Make sure the column names are correct and exist in the table.\n"
        "3. For column names with spaces, punctuation, percent signs, hyphens, or question marks, "
        "quote the full identifier with backticks.\n"
        "4. Think step by step internally and always check schema, question, evidence, and column names "
        "before writing the query.\n\n"
        f"# SQL Dialect:\n{example.dialect}\n\n"
        f"# Database ID:\n{example.db_id}\n\n"
        f"# Database and Table Schema:\n{example.schema_text}\n\n"
        f"{additional_knowledge}"
        f"{column_value_notes}"
        f"{schema_linking_notes}"
        f"# Question: {example.question}\n\n"
        "# SQL:"
    )
