from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlbench_lab.pipeline import _verify_target, render_prompt


def test_prompt_is_shared_shape() -> None:
    example = {
        "dialect": "sqlite",
        "db_id": "demo",
        "schema_text": "CREATE TABLE items (id INTEGER);",
        "knowledge_text": "No derived knowledge.",
        "question": "List item IDs.",
    }
    prompt = render_prompt(example)
    assert prompt.count("<|system|>") == 1
    assert prompt.count("<|user|>") == 1
    assert prompt.endswith("<|assistant|>\n")


def test_target_requires_read_only_sql_and_is_deterministic(tmp_path: Path) -> None:
    database = tmp_path / "items.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (id INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])
    _verify_target(database, "SELECT id FROM items ORDER BY id")

    with pytest.raises(ValueError, match="read-only"):
        _verify_target(database, "UPDATE items SET id = 3")
