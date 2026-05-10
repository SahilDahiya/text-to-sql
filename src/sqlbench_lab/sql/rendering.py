"""Prompt rendering for SQL SFT rows."""

from __future__ import annotations

from .models import SQLEvalCase, SQLRepairExample, SQLTrainExample

SQL_SYSTEM_PROMPT = (
    "You are a precise text-to-SQL model. Return only the final SQL statement. "
    "Use the declared SQL dialect and stay grounded in the provided schema."
)


def build_train_messages(example: SQLTrainExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": _base_user_content(example)},
        {"role": "assistant", "content": example.target_sql},
    ]


def build_repair_messages(example: SQLRepairExample) -> list[dict[str, str]]:
    user_content = "\n\n".join(
        [
            _base_user_content(example),
            f"Previous SQL:\n{example.previous_sql}",
            f"Execution Error:\n{example.execution_error}",
        ]
    )
    return [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example.target_sql},
    ]


def build_eval_messages(case: SQLEvalCase) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": _base_user_content(case)},
    ]


def build_repair_eval_messages(
    case: SQLEvalCase,
    *,
    previous_sql: str,
    execution_observation: str,
) -> list[dict[str, str]]:
    user_content = "\n\n".join(
        [
            _base_user_content(case),
            f"Previous SQL:\n{previous_sql}",
            f"Execution Observation:\n{execution_observation}",
            "Return corrected SQL only.",
        ]
    )
    return [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _base_user_content(example: SQLEvalCase | SQLTrainExample | SQLRepairExample) -> str:
    sections = [
        f"Dialect:\n{example.dialect}",
        f"Database ID:\n{example.db_id}",
        f"Schema:\n{example.schema_text}",
    ]
    if example.knowledge_text:
        sections.append(f"Knowledge:\n{example.knowledge_text}")
    sections.append(f"Question:\n{example.question}")
    return "\n\n".join(sections)
