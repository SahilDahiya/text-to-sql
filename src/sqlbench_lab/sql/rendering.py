"""Prompt rendering for SQL SFT rows."""

from __future__ import annotations

from .models import SQLEvalCase, SQLTrainExample

SQL_SYSTEM_PROMPT = (
    "You are a precise text-to-SQL model. Return only the final SQL statement. "
    "Use the declared SQL dialect and stay grounded in the provided schema. "
    "Use table and column names exactly as they appear in the schema. "
    "For SQLite identifiers containing spaces, punctuation, parentheses, percent signs, "
    "hyphens, or question marks, quote the full identifier with backticks."
)


def build_train_messages(
    example: SQLTrainExample,
    *,
    system_prompt: str = SQL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _canonical_user_content(example)},
        {"role": "assistant", "content": example.target_sql},
    ]


def build_eval_messages(
    case: SQLEvalCase,
    *,
    system_prompt: str = SQL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _canonical_user_content(case)},
    ]


def render_sql_sft_prompt(messages: list[dict[str, str]]) -> str:
    """Render the repo-owned base-model SFT and generation prompt format."""

    if len(messages) != 3:
        raise ValueError("SQL SFT messages must contain system, user, and assistant messages")
    system_message, user_message, assistant_message = messages
    if system_message["role"] != "system":
        raise ValueError("first SQL SFT message must have role=system")
    if user_message["role"] != "user":
        raise ValueError("second SQL SFT message must have role=user")
    if assistant_message["role"] != "assistant":
        raise ValueError("third SQL SFT message must have role=assistant")
    return (
        f"<|system|>\n{system_message['content'].strip()}\n"
        f"<|user|>\n{user_message['content'].strip()}\n"
        "<|assistant|>\n"
    )


def _canonical_user_content(example: SQLEvalCase | SQLTrainExample) -> str:
    sections = [
        f"Dialect:\n{example.dialect}",
        f"Database ID:\n{example.db_id}",
        f"Schema:\n{example.schema_text}",
    ]
    if example.knowledge_text:
        sections.append(f"Knowledge:\n{example.knowledge_text}")
    sections.append(f"Question:\n{example.question}")
    return "\n\n".join(sections)
