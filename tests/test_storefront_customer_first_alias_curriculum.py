from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from sqlbench_lab.sql import load_sql_train_examples


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = WORKSPACE_ROOT / "datasets/sql/train/storefront_customer_first_alias_curriculum_v1.jsonl"


class StorefrontCustomerFirstAliasCurriculumTest(unittest.TestCase):
    def test_rows_keep_customer_first_alias_roles(self) -> None:
        rows = load_sql_train_examples(DATASET_PATH)

        self.assertEqual(len(rows), 8)
        self.assertTrue(
            all("customer_first_alias_curriculum_v1" in row.tags for row in rows)
        )
        ratio_rows = [row for row in rows if "ratio_denominator" in row.tags]
        self.assertEqual(len(ratio_rows), 4)
        for row in ratio_rows:
            self.assertIn("COUNT(T4.return_id)", row.target_sql)
            self.assertIn("COUNT(T3.item_id)", row.target_sql)
            self.assertNotIn("COUNT(T2.item_id)", row.target_sql)
            self.assertIn("T1 is customers", row.knowledge_text)
        for row in rows:
            db_path = WORKSPACE_ROOT / str(row.db_path)
            with sqlite3.connect(db_path) as connection:
                connection.execute("PRAGMA query_only = ON")
                connection.execute(row.target_sql).fetchall()


if __name__ == "__main__":
    unittest.main()
