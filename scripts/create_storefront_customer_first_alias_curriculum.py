"""Build an execution-checked customer-first alias curriculum."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_v4.jsonl"
OUTPUT_PATH = ROOT / "datasets/sql/train/storefront_customer_first_alias_curriculum_v1.jsonl"
TRAIN_PATHS = (
    BASE_PATH,
    ROOT / "datasets/sql/train/sql_isft_blacksmith_pilot_v1.jsonl",
    ROOT / "datasets/sql/train/storefront_sales_lab_alias_binding_contrast_v1.jsonl",
)
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
        raise ValueError(f"curriculum rows must be direct SELECT statements: {row['row_id']}")
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.execute(sql).fetchall()


def _curriculum_rows(base: dict) -> list[dict]:
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
            "created_by": "scripts/create_storefront_customer_first_alias_curriculum.py",
            "teacher_model": None,
            "source_path": "scripts/create_storefront_customer_first_alias_curriculum.py",
        },
    }
    ratio_knowledge = (
        "Customer-first binding: T1 is customers, T2 is orders, T3 is order_items, and T4 is returns. "
        "For item-row return ratios, count matching returns with COUNT(T4.return_id) and count item rows with COUNT(T3.item_id)."
    )
    item_knowledge = (
        "Customer-first binding: T1 is customers, T2 is orders, and T3 is order_items. "
        "Item-row counts and quantities come from T3; customer filters come from T1 and order status comes from T2."
    )
    specifications = [
        (
            "What fraction of completed store-order item rows from Northeast customers had a matching return row?",
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T3.item_id), 3) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T3.order_id = T4.order_id AND T3.product_id = T4.product_id "
            "WHERE T1.region = 'Northeast' AND T2.channel = 'store' AND T2.status = 'completed'",
            ratio_knowledge,
            ("customer_first_alias_curriculum_v1", "alias_binding", "ratio_denominator", "return_ratio"),
        ),
        (
            "What proportion of completed web-order item rows from West customers had a matching return row?",
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T3.item_id), 3) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T3.order_id = T4.order_id AND T3.product_id = T4.product_id "
            "WHERE T1.region = 'West' AND T2.channel = 'web' AND T2.status = 'completed'",
            ratio_knowledge,
            ("customer_first_alias_curriculum_v1", "alias_binding", "ratio_denominator", "return_ratio"),
        ),
        (
            "For South customers with completed marketplace orders, what percentage of item rows had a matching return row?",
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T3.item_id), 3) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T3.order_id = T4.order_id AND T3.product_id = T4.product_id "
            "WHERE T1.region = 'South' AND T2.channel = 'marketplace' AND T2.status = 'completed'",
            ratio_knowledge,
            ("customer_first_alias_curriculum_v1", "alias_binding", "ratio_denominator", "return_ratio"),
        ),
        (
            "What fraction of completed-order item rows for Gold loyalty customers from Northeast had a matching return row?",
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T3.item_id), 3) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T3.order_id = T4.order_id AND T3.product_id = T4.product_id "
            "WHERE T1.loyalty_tier = 'Gold' AND T1.region = 'Northeast' AND T2.status = 'completed'",
            ratio_knowledge,
            ("customer_first_alias_curriculum_v1", "alias_binding", "ratio_denominator", "return_ratio"),
        ),
        (
            "How many completed-order item rows were purchased by Midwest customers?",
            "SELECT COUNT(T3.item_id) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "WHERE T1.region = 'Midwest' AND T2.status = 'completed'",
            item_knowledge,
            ("customer_first_alias_curriculum_v1", "alias_binding", "item_row_denominator"),
        ),
        (
            "What was the average quantity per completed-order item row for South customers?",
            "SELECT ROUND(AVG(T3.quantity), 2) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "WHERE T1.region = 'South' AND T2.status = 'completed'",
            item_knowledge,
            ("customer_first_alias_curriculum_v1", "alias_binding", "item_measure"),
        ),
        (
            "What total number of completed-order units did West customers purchase?",
            "SELECT SUM(T3.quantity) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "WHERE T1.region = 'West' AND T2.status = 'completed'",
            item_knowledge,
            ("customer_first_alias_curriculum_v1", "alias_binding", "item_measure"),
        ),
        (
            "Which customer region had the most completed-order item rows?",
            "SELECT T1.region, COUNT(T3.item_id) AS item_rows FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "WHERE T2.status = 'completed' GROUP BY T1.region "
            "ORDER BY item_rows DESC, T1.region ASC LIMIT 1",
            item_knowledge,
            ("customer_first_alias_curriculum_v1", "alias_binding", "grouped_item_rows"),
        ),
    ]
    rows = []
    for index, (question, target_sql, knowledge_text, tags) in enumerate(specifications, start=1):
        rows.append(
            {
                **common,
                "row_id": f"storefront_customer_first_alias_curriculum_v1_{index:03d}",
                "task_id": f"storefront_customer_first_alias_curriculum_v1_{index:03d}",
                "question": question,
                "knowledge_text": knowledge_text,
                "target_sql": target_sql,
                "tags": list(tags),
            }
        )
    return rows


def main() -> None:
    base_rows = _read_jsonl(BASE_PATH)
    rows = _curriculum_rows(base_rows[0])
    if len(rows) != 8:
        raise ValueError(f"expected eight curriculum rows, got {len(rows)}")

    existing_questions = set()
    existing_sql = set()
    for path in (*TRAIN_PATHS, *EVAL_PATHS):
        for row in _read_jsonl(path):
            existing_questions.add(_normalize(str(row["question"])))
            existing_sql.add(_normalize(str(row.get("target_sql", row.get("gold_sql")))))

    questions = set()
    sql_statements = set()
    for row in rows:
        question = _normalize(row["question"])
        sql = _normalize(row["target_sql"])
        if question in existing_questions or question in questions:
            raise ValueError(f"curriculum question overlaps existing data: {row['question']}")
        if sql in existing_sql or sql in sql_statements:
            raise ValueError(f"curriculum SQL overlaps existing data: {row['row_id']}")
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
