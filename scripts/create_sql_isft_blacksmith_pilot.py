"""Build a bounded, execution-checked multi-database ISFT pilot."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "datasets/sql/train/storefront_sales_lab_train_v4.jsonl"
PUBLIC_PATH = ROOT / "datasets/sql/train/spider_blacksmith_pilot_public_v1.jsonl"
OUTPUT_PATH = ROOT / "datasets/sql/train/sql_isft_blacksmith_pilot_v1.jsonl"
RAW_EVAL_PATH = ROOT / "datasets/sql/eval/spider_blacksmith_pilot_unseen_raw_v1.jsonl"
EVAL_PATH = ROOT / "datasets/sql/eval/spider_blacksmith_pilot_unseen_v1.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _execute_checked(row: dict, sql_key: str) -> None:
    db_path = ROOT / str(row["db_path"])
    if not db_path.exists():
        raise ValueError(f"missing database fixture for {row['task_id']}: {db_path}")
    sql = str(row[sql_key]).strip()
    if not sql.lower().startswith("select"):
        raise ValueError(f"pilot only supports direct SELECT rows: {row['task_id']}")
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.execute(sql).fetchall()


def main() -> None:
    replay_rows = _read_jsonl(REPLAY_PATH)
    public_rows = _read_jsonl(PUBLIC_PATH)
    raw_eval_rows = _read_jsonl(RAW_EVAL_PATH)
    if len(replay_rows) != 200:
        raise ValueError(f"expected 200 Exp056 replay rows, got {len(replay_rows)}")
    if len(public_rows) != 160:
        raise ValueError(f"expected 160 public Spider rows, got {len(public_rows)}")
    if len(raw_eval_rows) != 40:
        raise ValueError(f"expected 40 raw Spider eval rows, got {len(raw_eval_rows)}")

    rows = []
    seen_pairs: set[tuple[str, str]] = set()
    question_sql: dict[str, str] = {}
    for row in replay_rows + public_rows:
        question = str(row["question"]).strip()
        sql = str(row["target_sql"]).strip()
        previous_sql = question_sql.setdefault(question, sql)
        if previous_sql != sql:
            raise ValueError(f"question maps to multiple SQL statements: {question}")
        pair = (question, sql)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rows.append(row)
    if len(rows) != 358:
        raise ValueError(f"expected 358 rows after exact duplicate removal, got {len(rows)}")
    for row in rows:
        _execute_checked(row, "target_sql")
        row.setdefault("tags", []).append("isft_blacksmith_pilot_v1")

    train_questions = {" ".join(str(row["question"]).casefold().split()) for row in rows}
    train_sql = {" ".join(str(row["target_sql"]).casefold().split()).rstrip(";") for row in rows}
    eval_rows = []
    seen_eval_pairs: set[tuple[str, str]] = set()
    for row in raw_eval_rows:
        question = " ".join(str(row["question"]).casefold().split())
        sql = " ".join(str(row["gold_sql"]).casefold().split()).rstrip(";")
        if question in train_questions or sql in train_sql:
            continue
        if (question, sql) in seen_eval_pairs:
            continue
        seen_eval_pairs.add((question, sql))
        _execute_checked(row, "gold_sql")
        row.setdefault("tags", []).append("isft_blacksmith_pilot_unseen_v1")
        eval_rows.append(row)
    if len(eval_rows) != 33:
        raise ValueError(f"expected 33 clean unseen eval rows, got {len(eval_rows)}")

    OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    EVAL_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in eval_rows),
        encoding="utf-8",
    )
    print(
        f"rows={len(rows)} replay={len(replay_rows)} public={len(public_rows)} "
        f"eval_rows={len(eval_rows)} databases={dict(Counter(row['db_id'] for row in rows))}"
    )


if __name__ == "__main__":
    main()
