"""Create the execution-checked Exp065 continuation dataset."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.sql.leakage import audit_sql_dataset_leakage

from create_storefront_sql_lab import COLUMN_VALUE_NOTES, DB_PATH, ROOT, SCHEMA_TEXT


DB_ID = "storefront_sales_lab"
BASE_TRAIN_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_v4.jsonl"
OUTPUT_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_exp065_v1.jsonl"
EVAL_PATHS = (
    ROOT / "datasets/sql/eval/storefront_sales_lab_dev_v2.jsonl",
    ROOT / "datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl",
    ROOT / "datasets/sql/eval/storefront_sales_lab_challenge_v2.jsonl",
)


@dataclass(frozen=True)
class Task:
    question: str
    sql: str
    knowledge: str
    tags: tuple[str, ...]
    order_sensitive: bool = False


def main() -> int:
    base_rows = _read_jsonl(BASE_TRAIN_PATH)
    supplement = _supplement_tasks()
    _validate_tasks(base_rows, supplement)
    rows = base_rows + [_train_row(index, task) for index, task in enumerate(supplement, start=1)]
    _write_jsonl(OUTPUT_PATH, rows)
    print(f"output={OUTPUT_PATH.relative_to(ROOT)} rows={len(rows)} supplement={len(supplement)}")
    return 0


def _supplement_tasks() -> list[Task]:
    return [
        Task(
            question="Which active Bags products appeared in completed web orders?",
            sql=(
                "SELECT DISTINCT T1.product_name FROM products AS T1 "
                "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
                "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
                "WHERE T1.active = 1 AND T1.category = 'Bags' AND T3.status = 'completed' "
                "AND T3.channel = 'web' ORDER BY T1.product_name"
            ),
            knowledge="active and category belong to products; channel and status belong to orders; DISTINCT removes repeated item rows.",
            tags=("single_db_lab", "exp065", "alias_ownership", "distinct", "join_path"),
            order_sensitive=True,
        ),
        Task(
            question="List products with no completed store sales, ordered by product name.",
            sql=(
                "SELECT T1.product_name FROM products AS T1 WHERE NOT EXISTS ("
                "SELECT 1 FROM order_items AS T2 INNER JOIN orders AS T3 "
                "ON T2.order_id = T3.order_id WHERE T2.product_id = T1.product_id "
                "AND T3.status = 'completed' AND T3.channel = 'store'"
                ") "
                "ORDER BY T1.product_name"
            ),
            knowledge="use a correlated NOT EXISTS query to exclude products with completed store sales.",
            tags=("single_db_lab", "exp065", "anti_join", "not_exists"),
            order_sensitive=True,
        ),
        Task(
            question="List customers without a completed marketplace purchase, ordered by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 LEFT JOIN orders AS T2 "
                "ON T1.customer_id = T2.customer_id AND T2.status = 'completed' "
                "AND T2.channel = 'marketplace' WHERE T2.order_id IS NULL ORDER BY T1.customer_name"
            ),
            knowledge="completed marketplace predicates belong in the orders LEFT JOIN ON clause before checking for NULL.",
            tags=("single_db_lab", "exp065", "anti_join", "left_join_predicate", "customer_order"),
            order_sensitive=True,
        ),
        Task(
            question="List order IDs delivered in fewer than 3 days, ordered by order ID.",
            sql=(
                "SELECT T1.order_id FROM shipments AS T1 "
                "WHERE julianday(T1.delivery_date) - julianday(T1.shipped_date) < 3 "
                "ORDER BY T1.order_id"
            ),
            knowledge="shipment duration uses shipments.delivery_date minus shipments.shipped_date.",
            tags=("single_db_lab", "exp065", "date_math", "column_ownership"),
            order_sensitive=True,
        ),
        Task(
            question="List order IDs delivered in at least 5 days, ordered by order ID.",
            sql=(
                "SELECT T1.order_id FROM shipments AS T1 "
                "WHERE julianday(T1.delivery_date) - julianday(T1.shipped_date) >= 5 "
                "ORDER BY T1.order_id"
            ),
            knowledge="at least 5 days means duration >= 5 and both date columns come from shipments.",
            tags=("single_db_lab", "exp065", "date_math", "boundary_semantics"),
            order_sensitive=True,
        ),
        Task(
            question="How many products have more than 5 units sold in completed orders?",
            sql=(
                "SELECT COUNT(*) FROM (SELECT T1.product_id, SUM(T1.quantity) AS units "
                "FROM order_items AS T1 INNER JOIN orders AS T2 ON T1.order_id = T2.order_id "
                "WHERE T2.status = 'completed' GROUP BY T1.product_id HAVING units > 5)"
            ),
            knowledge="quantity belongs to order_items; aggregate the completed rows before applying HAVING.",
            tags=("single_db_lab", "exp065", "having", "column_ownership"),
        ),
        Task(
            question="How many products have at most 2 units sold in completed orders?",
            sql=(
                "SELECT COUNT(*) FROM (SELECT T1.product_id, SUM(T1.quantity) AS units "
                "FROM order_items AS T1 INNER JOIN orders AS T2 ON T1.order_id = T2.order_id "
                "WHERE T2.status = 'completed' GROUP BY T1.product_id HAVING units <= 2)"
            ),
            knowledge="at most 2 means HAVING units <= 2 after summing order_items.quantity by product.",
            tags=("single_db_lab", "exp065", "having", "boundary_semantics"),
        ),
        Task(
            question="How many active Apparel products have never been returned?",
            sql=(
                "SELECT COUNT(*) FROM products AS T1 WHERE T1.active = 1 "
                "AND T1.category = 'Apparel' AND NOT EXISTS ("
                "SELECT 1 FROM returns AS T2 WHERE T2.product_id = T1.product_id)"
            ),
            knowledge="filter active Apparel products and exclude any product with a matching return using NOT EXISTS.",
            tags=("single_db_lab", "exp065", "anti_join", "not_exists", "return_product_join"),
        ),
        Task(
            question="List completed orders with no support ticket, ordered by order ID.",
            sql=(
                "SELECT T1.order_id FROM orders AS T1 LEFT JOIN support_tickets AS T2 "
                "ON T1.order_id = T2.order_id WHERE T1.status = 'completed' "
                "AND T2.ticket_id IS NULL ORDER BY T1.order_id"
            ),
            knowledge="status belongs to orders; the NULL check belongs to support_tickets.",
            tags=("single_db_lab", "exp065", "anti_join", "column_ownership"),
            order_sensitive=True,
        ),
        Task(
            question="List customers with no support ticket attached to a completed order, ordered by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 LEFT JOIN support_tickets AS T2 "
                "ON T1.customer_id = T2.customer_id AND T2.order_id IN ("
                "SELECT T3.order_id FROM orders AS T3 WHERE T3.status = 'completed'"
                ") WHERE T2.ticket_id IS NULL ORDER BY T1.customer_name"
            ),
            knowledge="the completed-order restriction is inside the support-ticket LEFT JOIN condition.",
            tags=("single_db_lab", "exp065", "anti_join", "subquery", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="How many completed order item rows were shipped by USPS?",
            sql=(
                "SELECT COUNT(*) FROM order_items AS T1 INNER JOIN orders AS T2 "
                "ON T1.order_id = T2.order_id INNER JOIN shipments AS T3 "
                "ON T2.order_id = T3.order_id WHERE T2.status = 'completed' AND T3.carrier = 'USPS'"
            ),
            knowledge="item rows come from order_items; completed status comes from orders; carrier comes from shipments.",
            tags=("single_db_lab", "exp065", "alias_ownership", "multi_join"),
        ),
        Task(
            question="Which active products have no return record, ordered by product name?",
            sql=(
                "SELECT T1.product_name FROM products AS T1 LEFT JOIN returns AS T2 "
                "ON T1.product_id = T2.product_id WHERE T1.active = 1 "
                "AND T2.return_id IS NULL ORDER BY T1.product_name"
            ),
            knowledge="active belongs to products; preserve products with LEFT JOIN and test returns.return_id for NULL.",
            tags=("single_db_lab", "exp065", "anti_join", "left_join_predicate", "product_return"),
            order_sensitive=True,
        ),
    ]


def _validate_tasks(base_rows: list[dict[str, Any]], tasks: list[Task]) -> None:
    if len(tasks) != 12:
        raise ValueError(f"expected 12 Exp065 supplement rows, got {len(tasks)}")
    train_questions = {str(row["question"]) for row in base_rows}
    train_sql = {str(row["target_sql"]) for row in base_rows}
    supplement_questions = [task.question for task in tasks]
    supplement_sql = [task.sql for task in tasks]
    if len(set(supplement_questions)) != len(supplement_questions):
        raise ValueError("Exp065 supplement contains duplicate questions")
    if len(set(supplement_sql)) != len(supplement_sql):
        raise ValueError("Exp065 supplement contains duplicate SQL")
    if train_questions.intersection(supplement_questions):
        raise ValueError("Exp065 supplement repeats a train question")
    if train_sql.intersection(supplement_sql):
        raise ValueError("Exp065 supplement repeats train SQL")

    connection = sqlite3.connect(DB_PATH)
    try:
        for task in tasks:
            connection.execute(task.sql).fetchall()
    finally:
        connection.close()

    for eval_path in EVAL_PATHS:
        eval_rows = _read_jsonl(eval_path)
        eval_questions = {str(row["question"]) for row in eval_rows}
        eval_sql = {str(row.get("gold_sql", "")) for row in eval_rows}
        if eval_questions.intersection(supplement_questions):
            raise ValueError(f"Exp065 question overlap with {eval_path}")
        if eval_sql.intersection(supplement_sql):
            raise ValueError(f"Exp065 SQL overlap with {eval_path}")

    summary = audit_sql_dataset_leakage(train_paths=(BASE_TRAIN_PATH,), eval_paths=(EVAL_PATHS[1],))
    if summary.overlapping_questions or summary.overlapping_sql:
        raise ValueError("base train/eval leakage audit failed")


def _train_row(index: int, task: Task) -> dict[str, Any]:
    row_id = f"{DB_ID}_train_exp065_{index:03d}"
    return {
        "schema_version": "sql_train_example:v1",
        "row_id": row_id,
        "source_benchmark": "synthetic",
        "source_split": "train",
        "task_id": row_id,
        "db_id": DB_ID,
        "db_path": str(DB_PATH.relative_to(ROOT)),
        "dialect": "sqlite",
        "question": task.question,
        "schema_text": SCHEMA_TEXT,
        "knowledge_text": task.knowledge,
        "column_value_notes": COLUMN_VALUE_NOTES,
        "target_sql": task.sql,
        "task_type": "select",
        "provenance": {
            "created_by": "scripts/create_storefront_sql_train_v6.py",
            "teacher_model": None,
            "source_path": "scripts/create_storefront_sql_train_v6.py",
        },
        "tags": list(task.tags),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
