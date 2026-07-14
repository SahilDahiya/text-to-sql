"""Build the full verified train mix plus the customer-first curriculum."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_v4.jsonl"
CURRICULUM_PATH = ROOT / "datasets/sql/train/storefront_customer_first_alias_curriculum_v1.jsonl"
OUTPUT_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_exp069_customer_first_mixture_v1.jsonl"
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
        raise ValueError(f"mixture rows must be direct SELECT statements: {row['row_id']}")
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
    seen_questions = set()
    seen_pairs = set()
    for source_rows in (base_rows, curriculum_rows):
        for source_row in source_rows:
            row = dict(source_row)
            row["tags"] = [*row.get("tags", []), "exp069_customer_first_mixture_v1"]
            question = _normalize(str(row["question"]))
            sql = _normalize(str(row["target_sql"]))
            pair = (question, sql)
            if pair in seen_pairs:
                continue
            if question in seen_questions:
                raise ValueError(f"question maps to multiple SQL statements: {row['question']}")
            if row["row_id"].startswith("storefront_customer_first_") and (
                question in eval_questions or sql in eval_sql
            ):
                raise ValueError(f"curriculum overlaps frozen eval: {row['row_id']}")
            _execute_checked(row)
            rows.append(row)
            seen_questions.add(question)
            seen_pairs.add(pair)

    if len(rows) != 206:
        raise ValueError(f"expected 206 mixture rows after exact deduplication, got {len(rows)}")
    OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"rows={len(rows)} base={len(base_rows)} curriculum={len(curriculum_rows)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
