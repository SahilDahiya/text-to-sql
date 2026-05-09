"""Generated SQLite fixtures for smoke evaluation."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def build_sqlite_fixture(fixture_id: str, db_path: str | Path) -> Path:
    """Build a named SQLite fixture database."""

    resolved_path = Path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if fixture_id == "company_small":
        _build_company_small(resolved_path)
        return resolved_path
    raise ValueError(f"unknown SQLite fixture_id: {fixture_id}")


def _build_company_small(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );

            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER NOT NULL,
                salary INTEGER NOT NULL,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            );

            INSERT INTO departments (id, name) VALUES
                (1, 'Engineering'),
                (2, 'Sales'),
                (3, 'Operations');

            INSERT INTO employees (id, name, department_id, salary) VALUES
                (1, 'Ava', 1, 120000),
                (2, 'Ben', 1, 110000),
                (3, 'Cy', 2, 90000),
                (4, 'Dee', 3, 80000);
            """
        )

