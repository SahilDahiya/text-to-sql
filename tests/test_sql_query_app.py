from __future__ import annotations

import unittest
from pathlib import Path

from sqlbench_lab.webapp.sql_query import (
    SQLAskAppConfig,
    SQLAskQueryService,
    execute_readonly_select,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = WORKSPACE_ROOT / "experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json"
SCHEMA_SOURCE = WORKSPACE_ROOT / "datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl"
DB_PATH = WORKSPACE_ROOT / "datasets/sql/dbs/storefront_sales_lab/storefront_sales_lab.sqlite"


class SQLQueryAppTests(unittest.TestCase):
    def test_execute_readonly_select_rejects_non_select_statements(self) -> None:
        with self.assertRaisesRegex(ValueError, "only SELECT or WITH statements"):
            execute_readonly_select(DB_PATH, "DELETE FROM customers", row_limit=20)

    def test_execute_readonly_select_rejects_multiple_statements(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one SQL statement"):
            execute_readonly_select(DB_PATH, "SELECT 1; SELECT 2", row_limit=20)

    def test_execute_readonly_select_returns_columns_rows_and_truncation(self) -> None:
        result = execute_readonly_select(
            DB_PATH,
            "SELECT customer_name FROM customers ORDER BY customer_name",
            row_limit=2,
        )

        self.assertEqual(result.columns, ("customer_name",))
        self.assertEqual(len(result.rows), 2)
        self.assertTrue(result.truncated)

    def test_query_service_reuses_eval_prompt_and_runs_generated_sql(self) -> None:
        calls: list[dict[str, object]] = []

        def transport(url, headers, payload, timeout_seconds):
            calls.append(
                {
                    "url": url,
                    "headers": headers,
                    "payload": payload,
                    "timeout_seconds": timeout_seconds,
                }
            )
            return {"choices": [{"text": "SELECT COUNT(*) AS customer_count FROM customers;"}]}

        service = SQLAskQueryService(
            SQLAskAppConfig(
                manifest_path=MANIFEST,
                schema_source_path=SCHEMA_SOURCE,
                db_path=DB_PATH,
                openai_base_url="http://127.0.0.1:8000",
                openai_model="qwen35_0_8b_storefront_v4_lora_r16_a32_d010_exp056",
                row_limit=20,
                timeout_seconds=5.0,
                max_new_tokens=64,
            ),
            transport=transport,
        )

        result = service.ask("How many customers are in the database?")

        self.assertEqual(result.generated_sql, "SELECT COUNT(*) AS customer_count FROM customers;")
        self.assertEqual(result.columns, ("customer_count",))
        self.assertEqual(result.rows, ((12,),))
        self.assertIsNone(result.error)
        self.assertIn("<|system|>", str(calls[0]["payload"]))
        self.assertIn("CREATE TABLE customers", str(calls[0]["payload"]))
        self.assertIn("How many customers are in the database?", str(calls[0]["payload"]))

    def test_fastapi_htmx_route_renders_generated_sql_and_rows(self) -> None:
        from fastapi.testclient import TestClient
        from sqlbench_lab.webapp.app import create_app

        def transport(url, headers, payload, timeout_seconds):
            return {"choices": [{"text": "SELECT customer_name FROM customers ORDER BY customer_name LIMIT 1;"}]}

        app = create_app(
            config=SQLAskAppConfig(
                manifest_path=MANIFEST,
                schema_source_path=SCHEMA_SOURCE,
                db_path=DB_PATH,
                openai_base_url="http://127.0.0.1:8000",
                openai_model="qwen35_0_8b_storefront_v4_lora_r16_a32_d010_exp056",
                row_limit=20,
                timeout_seconds=5.0,
                max_new_tokens=64,
            ),
            transport=transport,
        )
        client = TestClient(app)

        index_response = client.get("/")
        query_response = client.post("/query", data={"question": "Show one customer"})

        self.assertEqual(index_response.status_code, 200)
        self.assertIn("hx-post=\"/query\"", index_response.text)
        self.assertEqual(query_response.status_code, 200)
        self.assertIn("SELECT customer_name FROM customers", query_response.text)
        self.assertIn("<table", query_response.text)


if __name__ == "__main__":
    unittest.main()
