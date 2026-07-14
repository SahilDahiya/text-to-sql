from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from sqlbench_lab.sql import load_sql_train_examples


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = WORKSPACE_ROOT / "datasets/sql/train/storefront_sales_lab_alias_binding_contrast_v1.jsonl"


class StorefrontAliasBindingContrastTest(unittest.TestCase):
    def test_contrast_rows_bind_aliases_and_execute(self) -> None:
        rows = load_sql_train_examples(DATASET_PATH)

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {"shipment_alias_binding", "ratio_denominator_binding"},
            {tag for row in rows for tag in row.tags if tag.endswith("_binding")},
        )
        for row in rows:
            self.assertIn("alias_binding_contrast_v1", row.tags)
            self.assertIsNotNone(row.knowledge_text)
            db_path = WORKSPACE_ROOT / str(row.db_path)
            with sqlite3.connect(db_path) as connection:
                connection.execute("PRAGMA query_only = ON")
                connection.execute(row.target_sql).fetchall()

        shipment_rows = [row for row in rows if "shipment_alias_binding" in row.tags]
        ratio_rows = [row for row in rows if "ratio_denominator_binding" in row.tags]
        self.assertEqual(len(shipment_rows), 3)
        self.assertEqual(len(ratio_rows), 3)
        self.assertTrue(
            all(
                "shipments alias S" in row.knowledge_text or "S is shipments" in row.knowledge_text
                for row in shipment_rows
            )
        )
        self.assertTrue(all("COUNT(I.item_id)" in row.knowledge_text for row in ratio_rows))
        self.assertTrue(all("COUNT(R.return_id)" in row.knowledge_text for row in ratio_rows))


if __name__ == "__main__":
    unittest.main()
