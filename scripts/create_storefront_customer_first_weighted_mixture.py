"""Build a train_v4 mix with a 2x-weighted customer-first curriculum."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_v4.jsonl"
CURRICULUM_PATH = ROOT / "datasets/sql/train/storefront_customer_first_alias_curriculum_v1.jsonl"
OUTPUT_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_exp070_weighted_customer_first_v1.jsonl"
EVAL_PATHS = tuple(sorted((ROOT / "datasets/sql/eval").glob("storefront_sales_lab_*.jsonl")))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing required dataset: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split()).rstrip(";")


def _execute_checked(row: dict) -> None:
    db_path = ROOT / str(row["db_path"])
    sql = str(row["target_sql"]).strip()
    if not sql.casefold().startswith("select"):
        raise ValueError(f"weighted rows must be direct SELECT statements: {row['row_id']}")
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.execute(sql).fetchall()


def main() -> None:
    base_rows = _read_jsonl(BASE_PATH)
    curriculum_rows = _read_jsonl(CURRICULUM_PATH)
    if len(base_rows) != 200:
        raise ValueError(f"expected 200 train_v4 rows, got {len(base_rows)}")
    if len(curriculum_rows) != 8:
        raise ValueError(f"expected eight curriculum rows, got {len(curriculum_rows)}")

    eval_questions = set()
    eval_sql = set()
    for path in EVAL_PATHS:
        for row in _read_jsonl(path):
            eval_questions.add(_normalize(str(row["question"])))
            eval_sql.add(_normalize(str(row["gold_sql"])))

    rows = []
    seen_pairs = set()
    seen_questions = set()
    for source_row in base_rows:
        row = dict(source_row)
        question = _normalize(str(row["question"]))
        sql = _normalize(str(row["target_sql"]))
        pair = (question, sql)
        if pair in seen_pairs:
            continue
        if question in seen_questions:
            raise ValueError(f"question maps to multiple SQL statements: {row['question']}")
        row["tags"] = [*row.get("tags", []), "exp070_weighted_customer_first_v1"]
        _execute_checked(row)
        rows.append(row)
        seen_pairs.add(pair)
        seen_questions.add(question)

    for repeat in range(1, 3):
        for source_row in curriculum_rows:
            row = dict(source_row)
            row["row_id"] = f"{source_row['row_id']}_repeat_{repeat:02d}"
            row["task_id"] = f"{source_row['task_id']}_repeat_{repeat:02d}"
            row["tags"] = [
                *row.get("tags", []),
                "exp070_weighted_customer_first_v1",
                f"weighted_repeat_{repeat}",
            ]
            question = _normalize(str(row["question"]))
            sql = _normalize(str(row["target_sql"]))
            if question in eval_questions or sql in eval_sql:
                raise ValueError(f"curriculum overlaps frozen eval: {row['row_id']}")
            _execute_checked(row)
            rows.append(row)

    if len(rows) != 214:
        raise ValueError(f"expected 214 rows after exact base deduplication, got {len(rows)}")
    OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"rows={len(rows)} base_unique=198 curriculum_repeats=2 output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
