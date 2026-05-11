"""Train-split BIRD curriculum labs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT

SUPERSTORE_DB_ID = "superstore"
REGIONAL_SALES_DB_ID = "regional_sales"
SUPERSTORE_TABLES = (
    ("central_superstore", "Central"),
    ("east_superstore", "East"),
    ("south_superstore", "South"),
    ("west_superstore", "West"),
)
REGIONAL_SALES_REGIONS = ("Midwest", "Northeast", "South", "West")
DEFAULT_BIRD_TRAIN_DB_ROOT = (
    WORKSPACE_ROOT / "external" / "sql" / "benchmarks" / "premai-io__birdbench" / "train"
)


@dataclass(frozen=True)
class BIRDSchemaLabSummary:
    db_id: str
    train_output_path: str
    eval_output_path: str
    train_row_count: int
    eval_row_count: int


def generate_bird_superstore_schema_lab(
    *,
    train_output_path: str | Path,
    eval_output_path: str | Path,
    dataset_root: str | Path | None = None,
    curriculum_version: str = "v1",
) -> BIRDSchemaLabSummary:
    """Generate a train-split-only BIRD schema-linking lab for the superstore DB."""

    if curriculum_version not in {"v1", "v2"}:
        raise ValueError("curriculum_version must be v1 or v2")
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_BIRD_TRAIN_DB_ROOT
    db_path = root / "train_databases" / SUPERSTORE_DB_ID / f"{SUPERSTORE_DB_ID}.sqlite"
    if not db_path.exists():
        raise ValueError(f"BIRD train database not found: {db_path}")

    schema_text = _schema_text(db_path)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        for table_index, (table_name, region) in enumerate(SUPERSTORE_TABLES):
            facts = _superstore_facts(conn, table_name=table_name, region=region)
            train_rows.extend(
                _rows_for_split(
                    facts=facts,
                    schema_text=schema_text,
                    db_path=db_path,
                    split_name="train",
                    split_offset=table_index * 100,
                    artifact="train",
                    curriculum_version=curriculum_version,
                )
            )
            eval_rows.extend(
                _rows_for_split(
                    facts=facts,
                    schema_text=schema_text,
                    db_path=db_path,
                    split_name="dev",
                    split_offset=table_index * 100,
                    artifact="eval",
                    curriculum_version=curriculum_version,
                )
            )

    train_output = _resolve_workspace_path(train_output_path)
    eval_output = _resolve_workspace_path(eval_output_path)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_output, train_rows)
    _write_jsonl(eval_output, eval_rows)
    return BIRDSchemaLabSummary(
        db_id=SUPERSTORE_DB_ID,
        train_output_path=str(_display_path(train_output)),
        eval_output_path=str(_display_path(eval_output)),
        train_row_count=len(train_rows),
        eval_row_count=len(eval_rows),
    )


def generate_bird_regional_sales_schema_lab(
    *,
    train_output_path: str | Path,
    eval_output_path: str | Path,
    dataset_root: str | Path | None = None,
) -> BIRDSchemaLabSummary:
    """Generate a train-split-only BIRD schema-linking lab for regional_sales."""

    root = Path(dataset_root) if dataset_root is not None else DEFAULT_BIRD_TRAIN_DB_ROOT
    db_path = root / "train_databases" / REGIONAL_SALES_DB_ID / f"{REGIONAL_SALES_DB_ID}.sqlite"
    if not db_path.exists():
        raise ValueError(f"BIRD train database not found: {db_path}")

    schema_text = _schema_text(db_path)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        for region_index, region in enumerate(REGIONAL_SALES_REGIONS):
            facts = _regional_sales_facts(conn, region=region)
            train_rows.extend(
                _regional_sales_rows_for_split(
                    facts=facts,
                    schema_text=schema_text,
                    db_path=db_path,
                    split_name="train",
                    split_offset=region_index * 100,
                    artifact="train",
                )
            )
            eval_rows.extend(
                _regional_sales_rows_for_split(
                    facts=facts,
                    schema_text=schema_text,
                    db_path=db_path,
                    split_name="dev",
                    split_offset=region_index * 100,
                    artifact="eval",
                )
            )

    train_output = _resolve_workspace_path(train_output_path)
    eval_output = _resolve_workspace_path(eval_output_path)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_output, train_rows)
    _write_jsonl(eval_output, eval_rows)
    return BIRDSchemaLabSummary(
        db_id=REGIONAL_SALES_DB_ID,
        train_output_path=str(_display_path(train_output)),
        eval_output_path=str(_display_path(eval_output)),
        train_row_count=len(train_rows),
        eval_row_count=len(eval_rows),
    )


def _rows_for_split(
    *,
    facts: dict[str, Any],
    schema_text: str,
    db_path: Path,
    split_name: str,
    split_offset: int,
    artifact: str,
    curriculum_version: str,
) -> list[dict[str, Any]]:
    start = 0 if split_name == "train" else 1
    selected = {
        key: values[start % len(values)]
        for key, values in facts.items()
        if isinstance(values, list)
    }
    rows = _superstore_curriculum_rows(
        table_name=facts["table_name"],
        region=facts["region"],
        values=selected,
        include_direct_fact_computed_order=artifact == "train" and curriculum_version == "v2",
    )
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        ordinal = split_offset + index
        common = {
            "source_benchmark": "bird",
            "source_split": "train" if artifact == "train" else "dev",
            "task_id": f"bird_superstore_schema_lab_{split_name}_{ordinal:04d}",
            "db_id": SUPERSTORE_DB_ID,
            "dialect": "sqlite",
            "question": row["question"],
            "schema_text": schema_text,
            "knowledge_text": row["knowledge_text"],
            "task_type": "select",
            "tags": [
                "bird",
                "schema_linking_lab",
                "split_train_db_only",
                f"curriculum_{curriculum_version}",
                f"lab_{split_name}",
                f"region_{facts['region'].lower()}",
                row["pattern"],
            ],
        }
        if artifact == "train":
            output.append(
                {
                    "schema_version": "sql_train_example:v1",
                    "row_id": f"bird_superstore_schema_lab_{split_name}_{ordinal:04d}",
                    **common,
                    "db_path": str(_display_path(db_path)),
                    "target_sql": row["sql"],
                    "provenance": {
                        "created_by": "curriculum_generator",
                        "teacher_model": None,
                        "source_path": str(_display_path(db_path)),
                    },
                }
            )
        else:
            output.append(
                {
                    "schema_version": "sql_eval_case:v1",
                    "case_id": f"bird_superstore_schema_lab_{split_name}_{ordinal:04d}",
                    "fixture_id": f"bird:{SUPERSTORE_DB_ID}",
                    "db_path": str(_display_path(db_path)),
                    **common,
                    "gold_sql": row["sql"],
                    "order_sensitive": False,
                    "numeric_tolerance": 0.000001,
                }
            )
    return output


def _regional_sales_rows_for_split(
    *,
    facts: dict[str, Any],
    schema_text: str,
    db_path: Path,
    split_name: str,
    split_offset: int,
    artifact: str,
) -> list[dict[str, Any]]:
    start = 0 if split_name == "train" else 1
    selected = {
        key: values[start % len(values)]
        for key, values in facts.items()
        if isinstance(values, list)
    }
    rows = _regional_sales_curriculum_rows(region=facts["region"], values=selected)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        ordinal = split_offset + index
        common = {
            "source_benchmark": "bird",
            "source_split": "train" if artifact == "train" else "dev",
            "task_id": f"bird_regional_sales_schema_lab_{split_name}_{ordinal:04d}",
            "db_id": REGIONAL_SALES_DB_ID,
            "dialect": "sqlite",
            "question": row["question"],
            "schema_text": schema_text,
            "knowledge_text": row["knowledge_text"],
            "task_type": "select",
            "tags": [
                "bird",
                "schema_linking_lab",
                "split_train_db_only",
                "curriculum_v1",
                f"lab_{split_name}",
                f"region_{facts['region'].lower()}",
                row["pattern"],
            ],
        }
        if artifact == "train":
            output.append(
                {
                    "schema_version": "sql_train_example:v1",
                    "row_id": f"bird_regional_sales_schema_lab_{split_name}_{ordinal:04d}",
                    **common,
                    "db_path": str(_display_path(db_path)),
                    "target_sql": row["sql"],
                    "provenance": {
                        "created_by": "curriculum_generator",
                        "teacher_model": None,
                        "source_path": str(_display_path(db_path)),
                    },
                }
            )
        else:
            output.append(
                {
                    "schema_version": "sql_eval_case:v1",
                    "case_id": f"bird_regional_sales_schema_lab_{split_name}_{ordinal:04d}",
                    "fixture_id": f"bird:{REGIONAL_SALES_DB_ID}",
                    "db_path": str(_display_path(db_path)),
                    **common,
                    "gold_sql": row["sql"],
                    "order_sensitive": False,
                    "numeric_tolerance": 0.000001,
                }
            )
    return output


def _superstore_curriculum_rows(
    *,
    table_name: str,
    region: str,
    values: dict[str, Any],
    include_direct_fact_computed_order: bool = False,
) -> list[dict[str, str]]:
    table_ref = f"`{table_name}`"
    rows = [
        {
            "pattern": "exact_identifier_copy",
            "question": (
                f"How many {region} superstore rows have a non-empty exact column "
                f"'{values['copy_column']}'?"
            ),
            "knowledge_text": f"Use the exact column `{values['copy_column']}`; quote identifiers that contain spaces.",
            "sql": f"SELECT COUNT(*) FROM {table_ref} WHERE `{values['copy_column']}` IS NOT NULL",
        },
        {
            "pattern": "value_filter",
            "question": (
                f"How many {region} superstore orders used ship mode "
                f"'{values['ship_mode']}'?"
            ),
            "knowledge_text": f"Ship mode maps to the exact column `Ship Mode`; {region} rows are in `{table_name}`.",
            "sql": (
                f"SELECT COUNT(DISTINCT `Order ID`) FROM {table_ref} "
                f"WHERE `Ship Mode` = '{_sql_string(values['ship_mode'])}'"
            ),
        },
        {
            "pattern": "product_join",
            "question": (
                f"List the product names in {region} order "
                f"{values['order_id']}."
            ),
            "knowledge_text": "Product names are in `product`.`Product Name`; join on `Product ID` and Region.",
            "sql": (
                f"SELECT DISTINCT T2.`Product Name` FROM {table_ref} AS T1 "
                "INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` "
                "AND T1.Region = T2.Region "
                f"WHERE T1.`Order ID` = '{_sql_string(values['order_id'])}'"
            ),
        },
        {
            "pattern": "customer_join",
            "question": (
                f"How many distinct {region} orders were made by customer "
                f"{values['customer_name']}?"
            ),
            "knowledge_text": "Customer names are in `people`.`Customer Name`; join on `Customer ID` and Region.",
            "sql": (
                f"SELECT COUNT(DISTINCT T1.`Order ID`) FROM {table_ref} AS T1 "
                "INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` "
                "AND T1.Region = T2.Region "
                f"WHERE T2.`Customer Name` = '{_sql_string(values['customer_name'])}'"
            ),
        },
        {
            "pattern": "three_table_join",
            "question": (
                f"How many distinct {region} orders from customer {values['customer_name']} "
                f"include products in category '{values['category']}'?"
            ),
            "knowledge_text": "Use people for customer names and product for categories; join through the regional superstore table.",
            "sql": (
                f"SELECT COUNT(DISTINCT T1.`Order ID`) FROM {table_ref} AS T1 "
                "INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` "
                "AND T1.Region = T2.Region "
                "INNER JOIN product AS T3 ON T1.`Product ID` = T3.`Product ID` "
                "AND T1.Region = T3.Region "
                f"WHERE T2.`Customer Name` = '{_sql_string(values['customer_name'])}' "
                f"AND T3.Category = '{_sql_string(values['category'])}'"
            ),
        },
        {
            "pattern": "quoted_identifier_arithmetic",
            "question": (
                f"For {region} row id {values['row_id']}, what is sales per unit quantity?"
            ),
            "knowledge_text": "sales per unit quantity = `Sales` / `Quantity`; row id maps to exact column `Row ID`.",
            "sql": (
                f"SELECT CAST(`Sales` AS REAL) / `Quantity` FROM {table_ref} "
                f"WHERE `Row ID` = {int(values['row_id'])}"
            ),
        },
        {
            "pattern": "computed_order_by",
            **_computed_order_row(table_name=table_name, region=region, variant=values["computed_variant"]),
        },
        {
            "pattern": "date_function",
            "question": (
                f"How many distinct {region} orders were placed in year "
                f"{values['order_year']}?"
            ),
            "knowledge_text": "Order year is extracted from exact column `Order Date` with STRFTIME('%Y', `Order Date`).",
            "sql": (
                f"SELECT COUNT(DISTINCT `Order ID`) FROM {table_ref} "
                f"WHERE STRFTIME('%Y', `Order Date`) = '{_sql_string(values['order_year'])}'"
            ),
        },
        {
            "pattern": "grouped_aggregate",
            "question": (
                f"What is the total quantity by {values['group_column']} for {region} orders? "
                f"Return {values['group_column']} and total quantity."
            ),
            "knowledge_text": f"Group by exact column `{values['group_column']}` and sum Quantity.",
            "sql": (
                f"SELECT `{values['group_column']}`, SUM(Quantity) FROM {table_ref} "
                f"GROUP BY `{values['group_column']}`"
            ),
        },
        {
            "pattern": "having",
            "question": (
                f"List {region} customers whose order count is greater than "
                f"{int(values['order_count_threshold'])}."
            ),
            "knowledge_text": "Customer names are in people; count distinct `Order ID` and filter with HAVING.",
            "sql": (
                "SELECT T2.`Customer Name` FROM "
                f"{table_ref} AS T1 INNER JOIN people AS T2 "
                "ON T1.`Customer ID` = T2.`Customer ID` AND T1.Region = T2.Region "
                "GROUP BY T2.`Customer Name` "
                f"HAVING COUNT(DISTINCT T1.`Order ID`) > {int(values['order_count_threshold'])}"
            ),
        },
    ]
    if include_direct_fact_computed_order:
        rows.extend(_direct_fact_computed_order_rows(table_name=table_name, region=region))
    return rows


def _regional_sales_curriculum_rows(*, region: str, values: dict[str, Any]) -> list[dict[str, str]]:
    sales_orders = "`Sales Orders`"
    region_join = (
        f"FROM {sales_orders} AS T1 "
        "INNER JOIN `Store Locations` AS T2 ON T1._StoreID = T2.StoreID "
        "INNER JOIN Regions AS T3 ON T2.StateCode = T3.StateCode"
    )
    rows = [
        {
            "pattern": "exact_identifier_copy",
            "question": (
                f"How many {region} regional sales orders have a non-empty exact column "
                f"'{values['copy_column']}'?"
            ),
            "knowledge_text": (
                f"Use exact column `{values['copy_column']}` from `Sales Orders`; "
                "quote identifiers that contain spaces."
            ),
            "sql": (
                f"SELECT COUNT(*) {region_join} "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                f"AND T1.`{values['copy_column']}` IS NOT NULL"
            ),
        },
        {
            "pattern": "value_filter",
            "question": (
                f"How many {region} regional sales orders used sales channel "
                f"'{values['sales_channel']}'?"
            ),
            "knowledge_text": "Sales channel maps to exact column `Sales Channel` on `Sales Orders`.",
            "sql": (
                f"SELECT COUNT(DISTINCT T1.OrderNumber) {region_join} "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                f"AND T1.`Sales Channel` = '{_sql_string(values['sales_channel'])}'"
            ),
        },
        {
            "pattern": "product_join",
            "question": f"Which product was sold in regional sales order {values['order_number']}?",
            "knowledge_text": "Product names are in `Products`.`Product Name`; join through `_ProductID`.",
            "sql": (
                f"SELECT DISTINCT T4.`Product Name` {region_join} "
                "INNER JOIN Products AS T4 ON T1._ProductID = T4.ProductID "
                f"WHERE T1.OrderNumber = '{_sql_string(values['order_number'])}'"
            ),
        },
        {
            "pattern": "customer_join",
            "question": (
                f"How many distinct {region} orders were made by customer "
                f"{values['customer_name']}?"
            ),
            "knowledge_text": "Customer names are in `Customers`.`Customer Names`; join through `_CustomerID`.",
            "sql": (
                f"SELECT COUNT(DISTINCT T1.OrderNumber) {region_join} "
                "INNER JOIN Customers AS T4 ON T1._CustomerID = T4.CustomerID "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                f"AND T4.`Customer Names` = '{_sql_string(values['customer_name'])}'"
            ),
        },
        {
            "pattern": "three_table_join",
            "question": (
                f"How many distinct {region} orders from customer {values['customer_name']} "
                f"include product '{values['product_name']}'?"
            ),
            "knowledge_text": (
                "Use Customers for customer names and Products for product names; "
                "join through the Sales Orders fact table."
            ),
            "sql": (
                f"SELECT COUNT(DISTINCT T1.OrderNumber) {region_join} "
                "INNER JOIN Customers AS T4 ON T1._CustomerID = T4.CustomerID "
                "INNER JOIN Products AS T5 ON T1._ProductID = T5.ProductID "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                f"AND T4.`Customer Names` = '{_sql_string(values['customer_name'])}' "
                f"AND T5.`Product Name` = '{_sql_string(values['product_name'])}'"
            ),
        },
        {
            "pattern": "quoted_identifier_arithmetic",
            "question": (
                f"For regional sales order {values['order_number']}, what is quantity times unit price?"
            ),
            "knowledge_text": (
                "quantity times unit price = `Order Quantity` * `Unit Price`; "
                "`Unit Price` is text with commas and must be normalized before CAST."
            ),
            "sql": (
                "SELECT T1.`Order Quantity` * CAST(REPLACE(T1.`Unit Price`, ',', '') AS REAL) "
                f"FROM {sales_orders} AS T1 "
                f"WHERE T1.OrderNumber = '{_sql_string(values['order_number'])}'"
            ),
        },
        {
            "pattern": "computed_order_by_direct_fact",
            **_regional_sales_computed_order_row(region=region, variant=values["computed_variant"]),
        },
        {
            "pattern": "date_function",
            "question": f"How many distinct {region} orders were placed in 20{values['order_year_suffix']}?",
            "knowledge_text": (
                "`OrderDate` is stored as m/d/yy text; match the year suffix in exact column `OrderDate`."
            ),
            "sql": (
                f"SELECT COUNT(DISTINCT T1.OrderNumber) {region_join} "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                f"AND T1.OrderDate LIKE '%/{_sql_string(values['order_year_suffix'])}'"
            ),
        },
        {
            "pattern": "grouped_aggregate",
            "question": (
                f"What is the total order quantity by {values['group_column']} for {region} orders? "
                f"Return {values['group_column']} and total order quantity."
            ),
            "knowledge_text": f"Group by exact column `{values['group_column']}` and sum `Order Quantity`.",
            "sql": (
                f"SELECT T1.`{values['group_column']}`, SUM(T1.`Order Quantity`) {region_join} "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                f"GROUP BY T1.`{values['group_column']}`"
            ),
        },
        {
            "pattern": "having",
            "question": (
                f"List {region} product names whose distinct order count is greater than "
                f"{int(values['order_count_threshold'])}."
            ),
            "knowledge_text": "Product names are in Products; count distinct OrderNumber and filter with HAVING.",
            "sql": (
                f"SELECT T4.`Product Name` {region_join} "
                "INNER JOIN Products AS T4 ON T1._ProductID = T4.ProductID "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                "GROUP BY T4.`Product Name` "
                f"HAVING COUNT(DISTINCT T1.OrderNumber) > {int(values['order_count_threshold'])}"
            ),
        },
    ]
    return rows


def _superstore_facts(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    region: str,
) -> dict[str, Any]:
    table_ref = _quote_identifier(table_name)
    order_rows = conn.execute(
        f"SELECT `Order ID`, `Row ID`, `Ship Mode`, STRFTIME('%Y', `Order Date`) "
        f"FROM {table_ref} WHERE `Order ID` IS NOT NULL ORDER BY `Row ID` LIMIT 6"
    ).fetchall()
    ship_mode_rows = conn.execute(
        f"SELECT DISTINCT `Ship Mode` FROM {table_ref} "
        "WHERE `Ship Mode` IS NOT NULL ORDER BY `Ship Mode` LIMIT 6"
    ).fetchall()
    year_rows = conn.execute(
        f"SELECT DISTINCT STRFTIME('%Y', `Order Date`) FROM {table_ref} "
        "WHERE `Order Date` IS NOT NULL ORDER BY STRFTIME('%Y', `Order Date`) LIMIT 6"
    ).fetchall()
    customer_rows = conn.execute(
        f"SELECT DISTINCT T2.`Customer Name` FROM {table_ref} AS T1 "
        "INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` "
        "AND T1.Region = T2.Region "
        "WHERE T2.`Customer Name` IS NOT NULL ORDER BY T2.`Customer Name` LIMIT 6"
    ).fetchall()
    category_rows = conn.execute(
        f"SELECT DISTINCT T2.Category FROM {table_ref} AS T1 "
        "INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` "
        "AND T1.Region = T2.Region "
        "WHERE T2.Category IS NOT NULL ORDER BY T2.Category LIMIT 6"
    ).fetchall()
    ship_modes = [str(row[0]) for row in ship_mode_rows if row[0] is not None]
    years = [str(row[0]) for row in year_rows if row[0] is not None]
    return {
        "table_name": table_name,
        "region": region,
        "order_id": [str(row[0]) for row in order_rows],
        "row_id": [int(row[1]) for row in order_rows],
        "ship_mode": ship_modes,
        "order_year": years,
        "customer_name": [str(row[0]) for row in customer_rows],
        "category": [str(row[0]) for row in category_rows],
        "copy_column": ["Customer ID", "Order ID"],
        "group_column": ["Ship Mode", "Customer ID"],
        "computed_variant": ["subcategory_profit_per_sales", "ship_mode_profit_per_quantity"],
        "order_count_threshold": [1, 2, 3],
    }


def _regional_sales_facts(conn: sqlite3.Connection, *, region: str) -> dict[str, Any]:
    order_rows = conn.execute(
        """
        SELECT T1.OrderNumber, T1.OrderDate
        FROM `Sales Orders` AS T1
        INNER JOIN `Store Locations` AS T2 ON T1._StoreID = T2.StoreID
        INNER JOIN Regions AS T3 ON T2.StateCode = T3.StateCode
        WHERE T3.Region = ?
        ORDER BY T1.OrderNumber
        LIMIT 6
        """,
        (region,),
    ).fetchall()
    sales_channel_rows = conn.execute(
        """
        SELECT DISTINCT T1.`Sales Channel`
        FROM `Sales Orders` AS T1
        INNER JOIN `Store Locations` AS T2 ON T1._StoreID = T2.StoreID
        INNER JOIN Regions AS T3 ON T2.StateCode = T3.StateCode
        WHERE T3.Region = ? AND T1.`Sales Channel` IS NOT NULL
        ORDER BY T1.`Sales Channel`
        LIMIT 6
        """,
        (region,),
    ).fetchall()
    customer_rows = conn.execute(
        """
        SELECT DISTINCT T4.`Customer Names`
        FROM `Sales Orders` AS T1
        INNER JOIN `Store Locations` AS T2 ON T1._StoreID = T2.StoreID
        INNER JOIN Regions AS T3 ON T2.StateCode = T3.StateCode
        INNER JOIN Customers AS T4 ON T1._CustomerID = T4.CustomerID
        WHERE T3.Region = ? AND T4.`Customer Names` IS NOT NULL
        ORDER BY T4.`Customer Names`
        LIMIT 6
        """,
        (region,),
    ).fetchall()
    product_rows = conn.execute(
        """
        SELECT DISTINCT T4.`Product Name`
        FROM `Sales Orders` AS T1
        INNER JOIN `Store Locations` AS T2 ON T1._StoreID = T2.StoreID
        INNER JOIN Regions AS T3 ON T2.StateCode = T3.StateCode
        INNER JOIN Products AS T4 ON T1._ProductID = T4.ProductID
        WHERE T3.Region = ? AND T4.`Product Name` IS NOT NULL
        ORDER BY T4.`Product Name`
        LIMIT 6
        """,
        (region,),
    ).fetchall()
    year_rows = conn.execute(
        """
        SELECT DISTINCT SUBSTR(T1.OrderDate, LENGTH(T1.OrderDate) - 1, 2)
        FROM `Sales Orders` AS T1
        INNER JOIN `Store Locations` AS T2 ON T1._StoreID = T2.StoreID
        INNER JOIN Regions AS T3 ON T2.StateCode = T3.StateCode
        WHERE T3.Region = ? AND T1.OrderDate IS NOT NULL
        ORDER BY SUBSTR(T1.OrderDate, LENGTH(T1.OrderDate) - 1, 2)
        LIMIT 6
        """,
        (region,),
    ).fetchall()
    facts = {
        "region": region,
        "order_number": [str(row[0]) for row in order_rows],
        "order_year_suffix": [str(row[0]) for row in year_rows],
        "sales_channel": [str(row[0]) for row in sales_channel_rows],
        "customer_name": [str(row[0]) for row in customer_rows],
        "product_name": [str(row[0]) for row in product_rows],
        "copy_column": ["Sales Channel", "Order Quantity"],
        "group_column": ["Sales Channel", "CurrencyCode"],
        "computed_variant": ["sales_channel_avg_quantity", "sales_channel_avg_unit_price"],
        "order_count_threshold": [1, 2, 3],
    }
    _require_regional_sales_values(facts)
    return facts


def _computed_order_row(*, table_name: str, region: str, variant: str) -> dict[str, str]:
    table_ref = f"`{table_name}`"
    if variant == "subcategory_profit_per_sales":
        return {
            "question": f"Which {region} product sub-category has the highest profit per sales dollar?",
            "knowledge_text": "profit per sales dollar = SUM(Profit) / SUM(Sales); sub-category is `Product`.`Sub-Category`.",
            "sql": (
                "SELECT T2.`Sub-Category` FROM "
                f"{table_ref} AS T1 INNER JOIN product AS T2 "
                "ON T1.`Product ID` = T2.`Product ID` AND T1.Region = T2.Region "
                "GROUP BY T2.`Sub-Category` "
                "ORDER BY CAST(SUM(T1.Profit) AS REAL) / SUM(T1.Sales) DESC LIMIT 1"
            ),
        }
    if variant == "ship_mode_profit_per_quantity":
        return {
            "question": f"Which {region} ship mode has the highest profit per ordered unit?",
            "knowledge_text": "profit per ordered unit = SUM(Profit) / SUM(Quantity); ship mode is exact column `Ship Mode`.",
            "sql": (
                f"SELECT `Ship Mode` FROM {table_ref} "
                "GROUP BY `Ship Mode` "
                "ORDER BY CAST(SUM(Profit) AS REAL) / SUM(Quantity) DESC LIMIT 1"
            ),
        }
    raise ValueError(f"unsupported superstore computed order variant: {variant}")


def _regional_sales_computed_order_row(*, region: str, variant: str) -> dict[str, str]:
    region_join = (
        "FROM `Sales Orders` AS T1 "
        "INNER JOIN `Store Locations` AS T2 ON T1._StoreID = T2.StoreID "
        "INNER JOIN Regions AS T3 ON T2.StateCode = T3.StateCode"
    )
    if variant == "sales_channel_avg_quantity":
        return {
            "question": f"Which {region} sales channel has the highest average order quantity?",
            "knowledge_text": (
                "average order quantity = AVG(`Order Quantity`); "
                "`Sales Channel` is on the Sales Orders fact table."
            ),
            "sql": (
                f"SELECT T1.`Sales Channel` {region_join} "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                "GROUP BY T1.`Sales Channel` "
                "ORDER BY AVG(T1.`Order Quantity`) DESC LIMIT 1"
            ),
        }
    if variant == "sales_channel_avg_unit_price":
        return {
            "question": f"Which {region} sales channel has the highest average unit price?",
            "knowledge_text": (
                "average unit price = AVG(`Unit Price`); `Unit Price` is text with commas, "
                "so normalize it before CAST."
            ),
            "sql": (
                f"SELECT T1.`Sales Channel` {region_join} "
                f"WHERE T3.Region = '{_sql_string(region)}' "
                "GROUP BY T1.`Sales Channel` "
                "ORDER BY AVG(CAST(REPLACE(T1.`Unit Price`, ',', '') AS REAL)) DESC LIMIT 1"
            ),
        }
    raise ValueError(f"unsupported regional_sales computed order variant: {variant}")


def _direct_fact_computed_order_rows(*, table_name: str, region: str) -> list[dict[str, str]]:
    table_ref = f"`{table_name}`"
    return [
        {
            "pattern": "computed_order_by_direct_fact",
            "question": f"Which {region} ship mode has the highest sales per ordered unit?",
            "knowledge_text": (
                "sales per ordered unit = SUM(Sales) / SUM(Quantity); "
                "`Ship Mode` is on the regional fact table, not a joined table."
            ),
            "sql": (
                f"SELECT `Ship Mode` FROM {table_ref} "
                "GROUP BY `Ship Mode` "
                "ORDER BY CAST(SUM(Sales) AS REAL) / SUM(Quantity) DESC LIMIT 1"
            ),
        },
        {
            "pattern": "computed_order_by_direct_fact",
            "question": f"Which {region} customer id has the highest profit per sales dollar?",
            "knowledge_text": (
                "profit per sales dollar = SUM(Profit) / SUM(Sales); "
                "`Customer ID` is on the regional fact table, not a joined table."
            ),
            "sql": (
                f"SELECT `Customer ID` FROM {table_ref} "
                "GROUP BY `Customer ID` "
                "ORDER BY CAST(SUM(Profit) AS REAL) / SUM(Sales) DESC LIMIT 1"
            ),
        },
    ]


def _require_regional_sales_values(facts: dict[str, Any]) -> None:
    for key, value in facts.items():
        if isinstance(value, list) and not value:
            raise ValueError(f"regional_sales lab has no values for {key} in region {facts['region']}")


def _schema_text(db_path: Path) -> str:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name"
        ).fetchall()
    statements = [str(row[0]) for row in rows if row[0]]
    if not statements:
        raise ValueError(f"SQLite database has no table schema: {db_path}")
    return "\n".join(statements)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    return str(value).replace("'", "''")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return WORKSPACE_ROOT / candidate


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return path
