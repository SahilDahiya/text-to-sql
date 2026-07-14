"""Build a small execution-checked alias-binding contrast set."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_v4.jsonl"
OUTPUT_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_alias_binding_contrast_v1.jsonl"
TRAIN_PATHS = (
    BASE_PATH,
    ROOT / "datasets/sql/train/sql_isft_blacksmith_pilot_v1.jsonl",
)
EVAL_PATHS = tuple(sorted((ROOT / "datasets/sql/eval").glob("storefront_sales_lab_*.jsonl")))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing required dataset: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.casefold().split()).rstrip(";")


def _execute_checked(row: dict) -> None:
    db_path = ROOT / str(row["db_path"])
    sql = str(row["target_sql"]).strip()
    if not sql.casefold().startswith("select"):
        raise ValueError(f"contrast rows must be direct SELECT statements: {row['row_id']}")
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.execute(sql).fetchall()


def _contrast_rows(base: dict) -> list[dict]:
    common = {
        "schema_version": "sql_train_example:v1",
        "source_benchmark": "synthetic",
        "source_split": "train",
        "db_id": base["db_id"],
        "db_path": base["db_path"],
        "dialect": base["dialect"],
        "schema_text": base["schema_text"],
        "column_value_notes": base["column_value_notes"],
        "task_type": "select",
        "provenance": {
            "created_by": "scripts/create_storefront_alias_binding_contrast.py",
            "teacher_model": None,
            "source_path": "scripts/create_storefront_alias_binding_contrast.py",
        },
    }
    specifications = [
        (
            "Which carrier has the longest mean transit duration between shipment and delivery?",
            "SELECT S.carrier, ROUND(AVG(julianday(S.delivery_date) - julianday(S.shipped_date)), 2) AS avg_days "
            "FROM shipments AS S GROUP BY S.carrier ORDER BY avg_days DESC, S.carrier ASC LIMIT 1",
            "Every carrier, shipped_date, and delivery_date reference comes from the shipments alias S; transit duration is julianday(S.delivery_date) - julianday(S.shipped_date).",
            ("alias_binding_contrast_v1", "shipment_alias_binding", "grouped_ranking"),
        ),
        (
            "List each carrier with its average transit duration, ordered by carrier name.",
            "SELECT S.carrier, ROUND(AVG(julianday(S.delivery_date) - julianday(S.shipped_date)), 2) AS avg_days "
            "FROM shipments AS S GROUP BY S.carrier ORDER BY S.carrier ASC",
            "Every carrier, shipped_date, and delivery_date reference comes from the shipments alias S; transit duration is julianday(S.delivery_date) - julianday(S.shipped_date).",
            ("alias_binding_contrast_v1", "shipment_alias_binding", "grouped_aggregation"),
        ),
        (
            "For completed orders, which carrier has the shortest mean transit duration?",
            "SELECT S.carrier, ROUND(AVG(julianday(S.delivery_date) - julianday(S.shipped_date)), 2) AS avg_days "
            "FROM shipments AS S INNER JOIN orders AS O ON S.order_id = O.order_id "
            "WHERE O.status = 'completed' GROUP BY S.carrier ORDER BY avg_days ASC, S.carrier ASC LIMIT 1",
            "S is shipments and O is orders; use S for carrier and shipment dates, and O for order status.",
            ("alias_binding_contrast_v1", "shipment_alias_binding", "join_path"),
        ),
        (
            "What fraction of completed Footwear item rows had a matching return record?",
            "SELECT ROUND(CAST(COUNT(R.return_id) AS REAL) / COUNT(I.item_id), 3) "
            "FROM products AS P INNER JOIN order_items AS I ON P.product_id = I.product_id "
            "INNER JOIN orders AS O ON I.order_id = O.order_id "
            "LEFT JOIN returns AS R ON I.order_id = R.order_id AND I.product_id = R.product_id "
            "WHERE P.category = 'Footwear' AND O.status = 'completed'",
            "P is products, I is order_items, O is orders, and R is returns; the denominator is COUNT(I.item_id) and the numerator is COUNT(R.return_id).",
            ("alias_binding_contrast_v1", "ratio_denominator_binding", "return_ratio"),
        ),
        (
            "What proportion of completed web-channel item rows had a matching return record?",
            "SELECT ROUND(CAST(COUNT(R.return_id) AS REAL) / COUNT(I.item_id), 3) "
            "FROM orders AS O INNER JOIN order_items AS I ON O.order_id = I.order_id "
            "LEFT JOIN returns AS R ON I.order_id = R.order_id AND I.product_id = R.product_id "
            "WHERE O.channel = 'web' AND O.status = 'completed'",
            "O is orders, I is order_items, and R is returns; the denominator is COUNT(I.item_id) and the numerator is COUNT(R.return_id).",
            ("alias_binding_contrast_v1", "ratio_denominator_binding", "return_ratio"),
        ),
        (
            "What proportion of completed Bags item rows from the marketplace channel had a matching return record?",
            "SELECT ROUND(CAST(COUNT(R.return_id) AS REAL) / COUNT(I.item_id), 3) "
            "FROM products AS P INNER JOIN order_items AS I ON P.product_id = I.product_id "
            "INNER JOIN orders AS O ON I.order_id = O.order_id "
            "LEFT JOIN returns AS R ON I.order_id = R.order_id AND I.product_id = R.product_id "
            "WHERE P.category = 'Bags' AND O.channel = 'marketplace' AND O.status = 'completed'",
            "P is products, I is order_items, O is orders, and R is returns; the denominator is COUNT(I.item_id) and the numerator is COUNT(R.return_id).",
            ("alias_binding_contrast_v1", "ratio_denominator_binding", "return_ratio"),
        ),
    ]
    rows = []
    for index, (question, target_sql, knowledge_text, tags) in enumerate(specifications, start=1):
        row = {
            **common,
            "row_id": f"storefront_sales_lab_alias_binding_contrast_v1_{index:03d}",
            "task_id": f"storefront_sales_lab_alias_binding_contrast_v1_{index:03d}",
            "question": question,
            "knowledge_text": knowledge_text,
            "target_sql": target_sql,
            "tags": list(tags),
        }
        rows.append(row)
    return rows


def main() -> None:
    base_rows = _read_jsonl(BASE_PATH)
    if not base_rows:
        raise ValueError(f"missing base dataset: {BASE_PATH}")
    rows = _contrast_rows(base_rows[0])
    if len(rows) != 6:
        raise ValueError(f"expected six contrast rows, got {len(rows)}")

    existing_questions = set()
    existing_sql = set()
    for path in TRAIN_PATHS:
        for row in _read_jsonl(path):
            existing_questions.add(_normalize_question(str(row["question"])))
            existing_sql.add(_normalize_sql(str(row["target_sql"])))
    for path in EVAL_PATHS:
        for row in _read_jsonl(path):
            existing_questions.add(_normalize_question(str(row["question"])))
            existing_sql.add(_normalize_sql(str(row["gold_sql"])))

    questions = set()
    sql_statements = set()
    for row in rows:
        question = _normalize_question(row["question"])
        sql = _normalize_sql(row["target_sql"])
        if question in existing_questions or question in questions:
            raise ValueError(f"contrast question overlaps an existing question: {row['question']}")
        if sql in existing_sql or sql in sql_statements:
            raise ValueError(f"contrast SQL overlaps an existing statement: {row['row_id']}")
        _execute_checked(row)
        questions.add(question)
        sql_statements.add(sql)

    OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"rows={len(rows)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
