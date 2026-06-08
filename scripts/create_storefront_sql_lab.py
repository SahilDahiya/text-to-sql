"""Create a self-contained storefront SQL SFT lab."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DB_ID = "storefront_sales_lab"
DB_DIR = ROOT / "datasets" / "sql" / "dbs" / DB_ID
DB_PATH = DB_DIR / f"{DB_ID}.sqlite"
TRAIN_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_v1.jsonl"
TRAIN_V2_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_v2.jsonl"
TRAIN_V3_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_v3.jsonl"
TRAIN_EXP051_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_exp051_support_v1.jsonl"
TRAIN_EXP052_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_exp052_date_v1.jsonl"
TRAIN_EXP053_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_exp053_return_ratio_v1.jsonl"
TRAIN_EXP054_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_exp054_having_v1.jsonl"
TRAIN_EXP055_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_exp055_antijoin_v1.jsonl"
TRAIN_V4_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_v4.jsonl"
TRAIN_EXP059_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_exp059_alias_contrast_v1.jsonl"
TRAIN_EXP060_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_exp060_boundary_contrast_v1.jsonl"
TRAIN_EXP061_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_exp061_antijoin_contrast_v1.jsonl"
TRAIN_V5_PATH = ROOT / "datasets" / "sql" / "train" / f"{DB_ID}_train_v5.jsonl"
DEV_PATH = ROOT / "datasets" / "sql" / "eval" / f"{DB_ID}_dev_v1.jsonl"
DEV_V2_PATH = ROOT / "datasets" / "sql" / "eval" / f"{DB_ID}_dev_v2.jsonl"
EVAL_PATH = ROOT / "datasets" / "sql" / "eval" / f"{DB_ID}_eval_v1.jsonl"
CHALLENGE_PATH = ROOT / "datasets" / "sql" / "eval" / f"{DB_ID}_challenge_v1.jsonl"
CHALLENGE_V2_PATH = ROOT / "datasets" / "sql" / "eval" / f"{DB_ID}_challenge_v2.jsonl"

SCHEMA_TEXT = """CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    region TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    loyalty_tier TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    active INTEGER NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    channel TEXT NOT NULL,
    status TEXT NOT NULL,
    discount_pct REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    carrier TEXT NOT NULL,
    shipped_date TEXT NOT NULL,
    delivery_date TEXT NOT NULL,
    shipping_cost REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
CREATE TABLE returns (
    return_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    return_date TEXT NOT NULL,
    reason TEXT NOT NULL,
    refund_amount REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE TABLE support_tickets (
    ticket_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_id INTEGER,
    opened_date TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    resolved INTEGER NOT NULL,
    satisfaction_score INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);"""

COLUMN_VALUE_NOTES = [
    "orders.status values are 'completed', 'cancelled', or 'pending'; revenue tasks use completed orders only unless stated otherwise.",
    "orders.discount_pct is a percent value, so discounted item revenue is quantity * unit_price * (1 - discount_pct / 100.0).",
    "customers.region values are Northeast, Midwest, South, and West.",
    "support_tickets.resolved uses 1 for resolved and 0 for unresolved.",
    "products.active uses 1 for active and 0 for inactive.",
]


@dataclass(frozen=True)
class Task:
    question: str
    sql: str
    knowledge: str
    tags: tuple[str, ...]
    order_sensitive: bool = False
    numeric_tolerance: float = 0.001


def main() -> int:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEV_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    _write_database(DB_PATH)
    splits = _tasks()
    _write_train_jsonl(TRAIN_PATH, splits["train"], split_name="train")
    _write_train_jsonl(TRAIN_V2_PATH, splits["train_v2"], split_name="train_v2")
    _write_train_jsonl(TRAIN_V3_PATH, splits["train_v3"], split_name="train_v3")
    _write_train_jsonl(TRAIN_EXP051_PATH, splits["train_exp051"], split_name="train_exp051")
    _write_train_jsonl(TRAIN_EXP052_PATH, splits["train_exp052"], split_name="train_exp052")
    _write_train_jsonl(TRAIN_EXP053_PATH, splits["train_exp053"], split_name="train_exp053")
    _write_train_jsonl(TRAIN_EXP054_PATH, splits["train_exp054"], split_name="train_exp054")
    _write_train_jsonl(TRAIN_EXP055_PATH, splits["train_exp055"], split_name="train_exp055")
    _write_train_jsonl(TRAIN_V4_PATH, splits["train_v4"], split_name="train_v4")
    _write_train_jsonl(TRAIN_EXP059_PATH, splits["train_exp059"], split_name="train_exp059")
    _write_train_jsonl(TRAIN_EXP060_PATH, splits["train_exp060"], split_name="train_exp060")
    _write_train_jsonl(TRAIN_EXP061_PATH, splits["train_exp061"], split_name="train_exp061")
    _write_train_jsonl(TRAIN_V5_PATH, splits["train_v5"], split_name="train_v5")
    _write_eval_jsonl(DEV_PATH, splits["dev"], split_name="dev")
    _write_eval_jsonl(DEV_V2_PATH, splits["dev_v2"], split_name="dev_v2")
    _write_eval_jsonl(EVAL_PATH, splits["eval"], split_name="eval")
    _write_eval_jsonl(CHALLENGE_PATH, splits["challenge"], split_name="challenge")
    _write_eval_jsonl(CHALLENGE_V2_PATH, splits["challenge_v2"], split_name="challenge_v2")
    print(f"database={DB_PATH.relative_to(ROOT)}")
    print(f"train={TRAIN_PATH.relative_to(ROOT)} rows={len(splits['train'])}")
    print(f"train_v2={TRAIN_V2_PATH.relative_to(ROOT)} rows={len(splits['train_v2'])}")
    print(f"train_v3={TRAIN_V3_PATH.relative_to(ROOT)} rows={len(splits['train_v3'])}")
    print(f"train_exp051={TRAIN_EXP051_PATH.relative_to(ROOT)} rows={len(splits['train_exp051'])}")
    print(f"train_exp052={TRAIN_EXP052_PATH.relative_to(ROOT)} rows={len(splits['train_exp052'])}")
    print(f"train_exp053={TRAIN_EXP053_PATH.relative_to(ROOT)} rows={len(splits['train_exp053'])}")
    print(f"train_exp054={TRAIN_EXP054_PATH.relative_to(ROOT)} rows={len(splits['train_exp054'])}")
    print(f"train_exp055={TRAIN_EXP055_PATH.relative_to(ROOT)} rows={len(splits['train_exp055'])}")
    print(f"train_v4={TRAIN_V4_PATH.relative_to(ROOT)} rows={len(splits['train_v4'])}")
    print(f"train_exp059={TRAIN_EXP059_PATH.relative_to(ROOT)} rows={len(splits['train_exp059'])}")
    print(f"train_exp060={TRAIN_EXP060_PATH.relative_to(ROOT)} rows={len(splits['train_exp060'])}")
    print(f"train_exp061={TRAIN_EXP061_PATH.relative_to(ROOT)} rows={len(splits['train_exp061'])}")
    print(f"train_v5={TRAIN_V5_PATH.relative_to(ROOT)} rows={len(splits['train_v5'])}")
    print(f"dev={DEV_PATH.relative_to(ROOT)} rows={len(splits['dev'])}")
    print(f"dev_v2={DEV_V2_PATH.relative_to(ROOT)} rows={len(splits['dev_v2'])}")
    print(f"eval={EVAL_PATH.relative_to(ROOT)} rows={len(splits['eval'])}")
    print(f"challenge={CHALLENGE_PATH.relative_to(ROOT)} rows={len(splits['challenge'])}")
    print(f"challenge_v2={CHALLENGE_V2_PATH.relative_to(ROOT)} rows={len(splits['challenge_v2'])}")
    return 0


def _write_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA_TEXT)
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Avery Johnson", "Northeast", "2023-01-08", "Gold"),
                (2, "Blake Smith", "Midwest", "2023-02-14", "Silver"),
                (3, "Casey Rivera", "South", "2023-03-03", "Bronze"),
                (4, "Dakota Lee", "West", "2023-04-19", "Gold"),
                (5, "Emerson Chen", "Northeast", "2023-05-22", "Silver"),
                (6, "Finley Patel", "Midwest", "2023-06-11", "Gold"),
                (7, "Gray Morgan", "South", "2023-07-07", "Silver"),
                (8, "Harper Kim", "West", "2023-08-16", "Bronze"),
                (9, "Indigo Brooks", "Northeast", "2024-01-12", "Bronze"),
                (10, "Jordan Quinn", "Midwest", "2024-02-01", "Silver"),
                (11, "Kai Turner", "South", "2024-02-18", "Gold"),
                (12, "Logan Wright", "West", "2024-03-05", "Silver"),
            ],
        )
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Nimbus Jacket", "Apparel", 120.0, 1),
                (2, "Trail Boots", "Footwear", 150.0, 1),
                (3, "City Sneakers", "Footwear", 95.0, 1),
                (4, "Canvas Tote", "Accessories", 35.0, 1),
                (5, "Rain Shell", "Apparel", 180.0, 1),
                (6, "Wool Beanie", "Accessories", 28.0, 1),
                (7, "Travel Backpack", "Bags", 140.0, 1),
                (8, "Gym Duffel", "Bags", 85.0, 1),
                (9, "Legacy Sandal", "Footwear", 60.0, 0),
                (10, "Fleece Hoodie", "Apparel", 90.0, 1),
            ],
        )
        orders = [
            (1001, 1, "2024-01-05", "web", "completed", 10.0),
            (1002, 2, "2024-01-08", "store", "completed", 0.0),
            (1003, 3, "2024-01-12", "web", "cancelled", 0.0),
            (1004, 4, "2024-01-18", "marketplace", "completed", 5.0),
            (1005, 5, "2024-02-02", "web", "completed", 0.0),
            (1006, 6, "2024-02-09", "store", "completed", 15.0),
            (1007, 7, "2024-02-14", "web", "pending", 0.0),
            (1008, 8, "2024-02-20", "marketplace", "completed", 0.0),
            (1009, 9, "2024-03-04", "web", "completed", 20.0),
            (1010, 10, "2024-03-09", "store", "completed", 0.0),
            (1011, 11, "2024-03-15", "web", "completed", 10.0),
            (1012, 12, "2024-03-18", "marketplace", "completed", 0.0),
            (1013, 1, "2024-04-02", "web", "completed", 0.0),
            (1014, 2, "2024-04-08", "store", "cancelled", 0.0),
            (1015, 3, "2024-04-12", "web", "completed", 5.0),
            (1016, 4, "2024-04-20", "marketplace", "completed", 0.0),
            (1017, 5, "2024-05-01", "web", "completed", 12.0),
            (1018, 6, "2024-05-07", "store", "completed", 0.0),
            (1019, 7, "2024-05-11", "web", "completed", 0.0),
            (1020, 8, "2024-05-19", "marketplace", "pending", 0.0),
            (1021, 9, "2024-06-03", "web", "completed", 8.0),
            (1022, 10, "2024-06-10", "store", "completed", 0.0),
            (1023, 11, "2024-06-18", "web", "completed", 15.0),
            (1024, 12, "2024-06-24", "marketplace", "completed", 0.0),
        ]
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)
        items = [
            (1, 1001, 1, 1, 120.0), (2, 1001, 4, 2, 35.0),
            (3, 1002, 2, 1, 150.0), (4, 1002, 6, 1, 28.0),
            (5, 1003, 3, 2, 95.0),
            (6, 1004, 5, 1, 180.0), (7, 1004, 7, 1, 140.0),
            (8, 1005, 10, 2, 90.0), (9, 1005, 4, 1, 35.0),
            (10, 1006, 2, 2, 150.0), (11, 1006, 8, 1, 85.0),
            (12, 1007, 6, 3, 28.0),
            (13, 1008, 7, 1, 140.0), (14, 1008, 3, 1, 95.0),
            (15, 1009, 1, 1, 120.0), (16, 1009, 6, 2, 28.0),
            (17, 1010, 8, 2, 85.0), (18, 1010, 4, 3, 35.0),
            (19, 1011, 5, 1, 180.0), (20, 1011, 10, 1, 90.0),
            (21, 1012, 2, 1, 150.0), (22, 1012, 7, 1, 140.0),
            (23, 1013, 3, 2, 95.0), (24, 1013, 4, 2, 35.0),
            (25, 1014, 1, 1, 120.0),
            (26, 1015, 10, 1, 90.0), (27, 1015, 6, 4, 28.0),
            (28, 1016, 5, 1, 180.0), (29, 1016, 8, 2, 85.0),
            (30, 1017, 1, 1, 120.0), (31, 1017, 7, 1, 140.0),
            (32, 1018, 2, 1, 150.0), (33, 1018, 10, 2, 90.0),
            (34, 1019, 3, 1, 95.0), (35, 1019, 4, 2, 35.0),
            (36, 1020, 8, 1, 85.0),
            (37, 1021, 5, 1, 180.0), (38, 1021, 6, 2, 28.0),
            (39, 1022, 7, 1, 140.0), (40, 1022, 4, 1, 35.0),
            (41, 1023, 2, 2, 150.0), (42, 1023, 1, 1, 120.0),
            (43, 1024, 10, 1, 90.0), (44, 1024, 8, 1, 85.0),
        ]
        connection.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items)
        shipments = [
            (1, 1001, "UPS", "2024-01-06", "2024-01-09", 9.5),
            (2, 1002, "FedEx", "2024-01-09", "2024-01-12", 11.0),
            (3, 1004, "UPS", "2024-01-19", "2024-01-24", 14.0),
            (4, 1005, "USPS", "2024-02-03", "2024-02-07", 7.5),
            (5, 1006, "FedEx", "2024-02-10", "2024-02-13", 13.0),
            (6, 1008, "UPS", "2024-02-21", "2024-02-26", 12.0),
            (7, 1009, "USPS", "2024-03-05", "2024-03-09", 8.0),
            (8, 1010, "FedEx", "2024-03-10", "2024-03-13", 9.0),
            (9, 1011, "UPS", "2024-03-16", "2024-03-20", 10.5),
            (10, 1012, "DHL", "2024-03-19", "2024-03-25", 16.0),
            (11, 1013, "USPS", "2024-04-03", "2024-04-06", 7.0),
            (12, 1015, "USPS", "2024-04-13", "2024-04-17", 7.5),
            (13, 1016, "UPS", "2024-04-21", "2024-04-25", 13.5),
            (14, 1017, "FedEx", "2024-05-02", "2024-05-06", 12.5),
            (15, 1018, "FedEx", "2024-05-08", "2024-05-11", 10.0),
            (16, 1019, "USPS", "2024-05-12", "2024-05-16", 6.5),
            (17, 1021, "UPS", "2024-06-04", "2024-06-10", 15.0),
            (18, 1022, "DHL", "2024-06-11", "2024-06-16", 14.5),
            (19, 1023, "UPS", "2024-06-19", "2024-06-24", 13.0),
            (20, 1024, "DHL", "2024-06-25", "2024-06-30", 15.5),
        ]
        connection.executemany("INSERT INTO shipments VALUES (?, ?, ?, ?, ?, ?)", shipments)
        connection.executemany(
            "INSERT INTO returns VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, 1004, 7, "2024-02-01", "damaged", 140.0),
                (2, 1006, 8, "2024-02-18", "late_delivery", 85.0),
                (3, 1011, 10, "2024-03-29", "wrong_size", 90.0),
                (4, 1017, 1, "2024-05-12", "changed_mind", 120.0),
                (5, 1023, 2, "2024-07-02", "damaged", 150.0),
            ],
        )
        connection.executemany(
            "INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 1001, "2024-01-10", "shipping", 1, 5),
                (2, 4, 1004, "2024-01-26", "return", 1, 3),
                (3, 6, 1006, "2024-02-15", "shipping", 1, 2),
                (4, 8, 1008, "2024-02-28", "product", 0, None),
                (5, 11, 1011, "2024-03-25", "return", 1, 4),
                (6, 3, 1015, "2024-04-19", "product", 1, 5),
                (7, 5, 1017, "2024-05-08", "return", 0, None),
                (8, 9, 1021, "2024-06-12", "shipping", 1, 3),
                (9, 11, 1023, "2024-06-28", "return", 0, None),
                (10, 12, 1024, "2024-07-01", "shipping", 1, 4),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _tasks() -> dict[str, list[Task]]:
    train_tasks = _train_tasks()
    train_v2_tasks = train_tasks + _train_v2_supplemental_tasks()
    train_v3_tasks = train_v2_tasks + _train_v3_supplemental_tasks()
    exp051_tasks = _train_exp051_support_tasks()
    exp052_tasks = _train_exp052_date_tasks()
    exp053_tasks = _train_exp053_return_ratio_tasks()
    exp054_tasks = _train_exp054_having_tasks()
    exp055_tasks = _train_exp055_antijoin_tasks()
    train_v4_tasks = train_v3_tasks + exp051_tasks + exp052_tasks + exp053_tasks + exp054_tasks + exp055_tasks
    exp059_tasks = _train_exp059_alias_contrast_tasks()
    exp060_tasks = _train_exp060_boundary_contrast_tasks()
    exp061_tasks = _train_exp061_antijoin_contrast_tasks()
    return {
        "train": train_tasks,
        "train_v2": train_v2_tasks,
        "train_v3": train_v3_tasks,
        "train_exp051": train_v3_tasks + exp051_tasks,
        "train_exp052": train_v3_tasks + exp052_tasks,
        "train_exp053": train_v3_tasks + exp053_tasks,
        "train_exp054": train_v3_tasks + exp054_tasks,
        "train_exp055": train_v3_tasks + exp055_tasks,
        "train_v4": train_v4_tasks,
        "train_exp059": train_v4_tasks + exp059_tasks,
        "train_exp060": train_v4_tasks + exp060_tasks,
        "train_exp061": train_v4_tasks + exp061_tasks,
        "train_v5": train_v4_tasks + exp059_tasks + exp060_tasks + exp061_tasks,
        "dev": _dev_tasks(),
        "dev_v2": _dev_v2_tasks(),
        "eval": _eval_tasks(),
        "challenge": _challenge_tasks(),
        "challenge_v2": _challenge_v2_tasks(),
    }


def _train_tasks() -> list[Task]:
    return [
        _revenue_by_region("Northeast"),
        _revenue_by_region("Midwest"),
        _revenue_by_region("West"),
        _category_units("Apparel"),
        _category_units("Footwear"),
        _category_units("Bags"),
        _orders_by_channel("web"),
        _orders_by_channel("store"),
        _orders_by_channel("marketplace"),
        _top_customer_by_revenue("2024-01-01", "2024-03-31"),
        _top_customer_by_revenue("2024-04-01", "2024-06-30"),
        _returned_refunds_by_reason("damaged"),
        _returned_refunds_by_reason("late_delivery"),
        _shipment_average_days("UPS"),
        _shipment_average_days("FedEx"),
        _unresolved_ticket_count("return"),
        _unresolved_ticket_count("product"),
        _loyalty_order_count("Gold"),
        _loyalty_order_count("Silver"),
        _active_products_by_category("Accessories"),
        _active_products_by_category("Apparel"),
        _monthly_completed_orders("2024-01"),
        _monthly_completed_orders("2024-03"),
        _monthly_completed_orders("2024-05"),
        _net_revenue_by_channel("web"),
        _net_revenue_by_channel("marketplace"),
        _product_revenue("Nimbus Jacket"),
        _product_revenue("Trail Boots"),
        _product_revenue("Fleece Hoodie"),
        _customers_without_returns("Northeast"),
        _customers_without_returns("Midwest"),
        _shipping_cost_by_region("West"),
        _shipping_cost_by_region("South"),
        _average_ticket_score("shipping"),
        _average_ticket_score("return"),
        _cancelled_or_pending_count("cancelled"),
        _cancelled_or_pending_count("pending"),
        _largest_order_in_region("South"),
        _largest_order_in_region("West"),
        _orders_with_more_than_units(2),
    ]


def _train_v2_supplemental_tasks() -> list[Task]:
    return [
        _discounted_revenue_between("2024-01-01", "2024-02-29"),
        _discounted_revenue_between("2024-05-01", "2024-06-30"),
        _completed_quantity_by_discount_floor(5.0),
        _completed_quantity_by_discount_floor(12.0),
        _discounted_order_ids(8.0),
        _average_discount_by_channel("web"),
        _average_discount_by_channel("marketplace"),
        _line_count_by_order_status("completed"),
        _line_count_by_order_status("pending"),
        _quantity_by_region_and_category("South", "Accessories"),
        _quantity_by_region_and_category("West", "Bags"),
        _quantity_by_region_and_category("Northeast", "Footwear"),
        _category_revenue_for_region("Apparel", "South"),
        _category_revenue_for_region("Bags", "Midwest"),
        _product_quantity_for_channel("Travel Backpack", "marketplace"),
        _product_quantity_for_channel("Canvas Tote", "store"),
        _products_with_completed_sales_count("Footwear"),
        _products_with_completed_sales_count("Accessories"),
        _active_products_without_completed_sales_count("Apparel"),
        _active_products_without_completed_sales_count("Footwear"),
        _carrier_delivery_after_date("UPS", "2024-04-01"),
        _carrier_delivery_after_date("DHL", "2024-03-01"),
        _shipping_cost_for_carrier("USPS"),
        _shipping_cost_for_region_and_carrier("West", "DHL"),
        _orders_delivered_after_days(5),
        _returns_after_slow_shipment_count(3),
        _refund_total_for_region_and_reason("West", "damaged"),
        _refund_total_for_region_and_reason("South", "wrong_size"),
        _refund_count_by_tier_and_reason("Silver", "changed_mind"),
        _refund_count_by_tier_and_reason("Gold", "damaged"),
        _returned_products_for_region("South"),
        _customers_with_returns_in_region("West"),
        _ticket_count_by_region_and_resolution("West", 0),
        _ticket_count_by_region_and_resolution("South", 1),
        _unresolved_ticket_customers_by_issue("return"),
        _unresolved_ticket_customers_by_issue("product"),
        _ticket_count_for_order_status("completed"),
        _avg_ticket_score_by_region("Northeast"),
        _orders_with_support_issue_type("shipping"),
        _completed_order_count_by_region_channel("Midwest", "store"),
        _completed_order_count_by_region_channel("West", "marketplace"),
        _repeat_customer_count_by_region("Northeast"),
        _customers_with_at_least_completed_units("South", 3),
        _orders_with_more_than_products(1),
        _order_unit_total_over(3),
        _month_revenue_by_category("2024-06", "Apparel"),
        _month_return_count("2024-02"),
        _quarter_revenue_for_tier("Gold", "2024-04-01", "2024-06-30"),
        _new_customer_completed_orders("2024-01-01"),
        _product_names_for_return_reason("damaged"),
        _refund_total_for_carrier("UPS"),
        _return_rate_for_region("West"),
        _category_return_count("Bags"),
        _product_names_never_ordered(),
        _top_region_by_completed_units(),
        _carrier_count_by_region("West"),
        _unresolved_ticket_order_ids(),
    ]


def _train_v3_supplemental_tasks() -> list[Task]:
    return [
        _category_revenue_rank_for_channel("web"),
        _category_revenue_rank_for_channel("store"),
        _category_revenue_rank_for_channel("marketplace"),
        _category_revenue_rank_for_region("Northeast"),
        _category_revenue_rank_for_region("Midwest"),
        _category_revenue_rank_for_region("South"),
        _category_revenue_rank_for_month("2024-02"),
        _category_revenue_rank_for_month("2024-05"),
        _return_item_share_for_category("Apparel"),
        _return_item_share_for_category("Bags"),
        _return_item_share_for_category("Accessories"),
        _return_item_share_for_channel("web"),
        _return_item_share_for_channel("store"),
        _carrier_fastest_delivery(),
        _carrier_slowest_delivery_after_date("2024-03-01"),
        _carrier_delivery_rank_for_region("West"),
        _active_product_no_completed_sales_names_by_category("Apparel"),
        _active_product_no_completed_sales_names_by_category("Footwear"),
        _active_product_no_completed_sales_names_by_category("Bags"),
        _active_product_no_completed_sales_count_by_channel("web"),
        _active_product_no_completed_sales_count_by_channel("store"),
        _repeat_customer_names(),
        _repeat_customer_count_by_tier("Gold"),
        _repeat_customer_count_by_tier("Silver"),
        _top_completed_order_product_by_channel("web"),
        _top_completed_order_product_by_channel("store"),
        _top_completed_order_product_by_channel("marketplace"),
        _refund_total_for_slow_shipments(3),
        _refund_total_for_slow_shipments(5),
        _refund_total_for_slow_shipments_by_reason("damaged", 3),
        _refund_total_for_slow_shipments_by_reason("late_delivery", 3),
        _refund_total_for_slow_shipments_by_carrier("UPS", 3),
        _refund_total_for_slow_shipments_by_carrier("FedEx", 2),
    ]


def _dev_tasks() -> list[Task]:
    return [
        _revenue_by_region("South"),
        _category_units("Accessories"),
        _top_customer_by_revenue("2024-02-01", "2024-05-31"),
        _shipment_average_days("USPS"),
        _loyalty_order_count("Bronze"),
        _monthly_completed_orders("2024-06"),
        _net_revenue_by_channel("store"),
        _product_revenue("Travel Backpack"),
        _shipping_cost_by_region("Northeast"),
        _average_ticket_score("product"),
        _largest_order_in_region("Northeast"),
        _orders_with_more_than_units(3),
    ]


def _eval_tasks() -> list[Task]:
    return [
        _revenue_after_date("2024-04-01"),
        _category_revenue_rank(),
        _return_rate_by_category("Footwear"),
        _carrier_slowest_delivery(),
        _customer_return_refunds("Gold"),
        _unresolved_ticket_customers(),
        _discounted_order_count(10.0),
        _active_product_no_sales(),
        _region_refund_total("West"),
        _highest_quantity_order_product("2024-06-01"),
        _repeat_customer_count(),
        _late_delivery_refunds(),
    ]


def _revenue_by_region(region: str) -> Task:
    return Task(
        question=f"What is the discounted completed-order revenue for customers in the {region} region?",
        sql=(
            "SELECT ROUND(SUM(T3.quantity * T3.unit_price * (1 - T2.discount_pct / 100.0)), 2) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed'"
        ),
        knowledge="completed-order revenue uses orders.status = 'completed' and applies discount_pct to every item.",
        tags=("single_db_lab", "revenue", "join"),
    )


def _category_units(category: str) -> Task:
    return Task(
        question=f"How many completed-order units were sold in the {category} category?",
        sql=(
            "SELECT SUM(T2.quantity) FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.category = '{category}' AND T3.status = 'completed'"
        ),
        knowledge="units sold means SUM(order_items.quantity) for completed orders.",
        tags=("single_db_lab", "aggregation", "category"),
    )


def _orders_by_channel(channel: str) -> Task:
    return Task(
        question=f"How many completed orders came through the {channel} channel?",
        sql=f"SELECT COUNT(*) FROM orders WHERE channel = '{channel}' AND status = 'completed'",
        knowledge="completed orders have orders.status = 'completed'.",
        tags=("single_db_lab", "filter", "count"),
    )


def _top_customer_by_revenue(start: str, end: str) -> Task:
    return Task(
        question=f"Which customer had the highest discounted completed-order revenue from {start} through {end}?",
        sql=(
            "SELECT T1.customer_name, ROUND(SUM(T3.quantity * T3.unit_price * (1 - T2.discount_pct / 100.0)), 2) AS revenue "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T2.status = 'completed' AND T2.order_date BETWEEN '{start}' AND '{end}' "
            "GROUP BY T1.customer_id, T1.customer_name ORDER BY revenue DESC, T1.customer_name ASC LIMIT 1"
        ),
        knowledge="discounted completed-order revenue applies discount_pct and excludes non-completed orders.",
        tags=("single_db_lab", "ranking", "date_filter"),
        order_sensitive=True,
    )


def _returned_refunds_by_reason(reason: str) -> Task:
    return Task(
        question=f"What total refund amount is recorded for returns with reason {reason}?",
        sql=f"SELECT ROUND(SUM(refund_amount), 2) FROM returns WHERE reason = '{reason}'",
        knowledge="refund amount refers to returns.refund_amount.",
        tags=("single_db_lab", "returns", "aggregation"),
    )


def _shipment_average_days(carrier: str) -> Task:
    return Task(
        question=f"What is the average delivery time in days for {carrier} shipments?",
        sql=(
            "SELECT ROUND(AVG(julianday(delivery_date) - julianday(shipped_date)), 2) "
            f"FROM shipments WHERE carrier = '{carrier}'"
        ),
        knowledge="delivery time in days is julianday(delivery_date) - julianday(shipped_date).",
        tags=("single_db_lab", "date_math", "shipment"),
    )


def _unresolved_ticket_count(issue_type: str) -> Task:
    return Task(
        question=f"How many unresolved support tickets have issue type {issue_type}?",
        sql=f"SELECT COUNT(*) FROM support_tickets WHERE issue_type = '{issue_type}' AND resolved = 0",
        knowledge="unresolved means support_tickets.resolved = 0.",
        tags=("single_db_lab", "support", "count"),
    )


def _loyalty_order_count(tier: str) -> Task:
    return Task(
        question=f"How many completed orders were placed by {tier} loyalty customers?",
        sql=(
            "SELECT COUNT(*) FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.loyalty_tier = '{tier}' AND T2.status = 'completed'"
        ),
        knowledge="completed orders have orders.status = 'completed'.",
        tags=("single_db_lab", "loyalty", "join"),
    )


def _active_products_by_category(category: str) -> Task:
    return Task(
        question=f"List active product names in the {category} category sorted alphabetically.",
        sql=f"SELECT product_name FROM products WHERE category = '{category}' AND active = 1 ORDER BY product_name ASC",
        knowledge="active products have products.active = 1.",
        tags=("single_db_lab", "products", "order_by"),
        order_sensitive=True,
    )


def _monthly_completed_orders(month: str) -> Task:
    return Task(
        question=f"How many completed orders were placed in {month}?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = 'completed' AND substr(order_date, 1, 7) = '{month}'",
        knowledge="month filtering uses the first seven characters of order_date in YYYY-MM format.",
        tags=("single_db_lab", "date_filter", "count"),
    )


def _net_revenue_by_channel(channel: str) -> Task:
    return Task(
        question=f"What is net revenue for the {channel} channel after subtracting refunds?",
        sql=(
            "SELECT ROUND(SUM(T2.quantity * T2.unit_price * (1 - T1.discount_pct / 100.0)) - "
            "COALESCE((SELECT SUM(T4.refund_amount) FROM returns AS T4 INNER JOIN orders AS T5 ON T4.order_id = T5.order_id "
            f"WHERE T5.channel = '{channel}' AND T5.status = 'completed'), 0), 2) "
            "FROM orders AS T1 INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.channel = '{channel}' AND T1.status = 'completed'"
        ),
        knowledge="net revenue means discounted completed-order item revenue minus refunds for completed orders in the same channel.",
        tags=("single_db_lab", "net_revenue", "subquery"),
    )


def _product_revenue(product_name: str) -> Task:
    return Task(
        question=f"What discounted completed-order revenue came from {product_name}?",
        sql=(
            "SELECT ROUND(SUM(T2.quantity * T2.unit_price * (1 - T3.discount_pct / 100.0)), 2) "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.product_name = '{product_name}' AND T3.status = 'completed'"
        ),
        knowledge="product revenue applies the order discount and excludes non-completed orders.",
        tags=("single_db_lab", "product", "revenue"),
    )


def _customers_without_returns(region: str) -> Task:
    return Task(
        question=f"How many {region} customers have completed orders but no returns?",
        sql=(
            "SELECT COUNT(DISTINCT T1.customer_id) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "LEFT JOIN returns AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed' AND T3.return_id IS NULL"
        ),
        knowledge="no returns means no matching returns row for the customer's completed order.",
        tags=("single_db_lab", "anti_join", "returns"),
    )


def _shipping_cost_by_region(region: str) -> Task:
    return Task(
        question=f"What total shipping cost was paid for completed orders from {region} customers?",
        sql=(
            "SELECT ROUND(SUM(T3.shipping_cost), 2) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN shipments AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed'"
        ),
        knowledge="shipping cost refers to shipments.shipping_cost for completed orders.",
        tags=("single_db_lab", "shipment", "join"),
    )


def _average_ticket_score(issue_type: str) -> Task:
    return Task(
        question=f"What is the average satisfaction score for resolved {issue_type} support tickets?",
        sql=(
            "SELECT ROUND(AVG(satisfaction_score), 2) FROM support_tickets "
            f"WHERE issue_type = '{issue_type}' AND resolved = 1"
        ),
        knowledge="resolved tickets have resolved = 1; unresolved tickets have no satisfaction score.",
        tags=("single_db_lab", "support", "average"),
    )


def _cancelled_or_pending_count(status: str) -> Task:
    return Task(
        question=f"How many orders have status {status}?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = '{status}'",
        knowledge="order status is stored in orders.status.",
        tags=("single_db_lab", "status", "count"),
    )


def _largest_order_in_region(region: str) -> Task:
    return Task(
        question=f"Which completed order from the {region} region had the largest discounted item total?",
        sql=(
            "SELECT T2.order_id, ROUND(SUM(T3.quantity * T3.unit_price * (1 - T2.discount_pct / 100.0)), 2) AS order_total "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed' "
            "GROUP BY T2.order_id ORDER BY order_total DESC, T2.order_id ASC LIMIT 1"
        ),
        knowledge="order total applies discount_pct and includes completed orders only.",
        tags=("single_db_lab", "ranking", "region"),
        order_sensitive=True,
    )


def _orders_with_more_than_units(min_units: int) -> Task:
    return Task(
        question=f"How many completed orders contain more than {min_units} total item units?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.order_id, SUM(T2.quantity) AS units "
            "FROM orders AS T1 INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            "WHERE T1.status = 'completed' GROUP BY T1.order_id "
            f"HAVING units > {min_units})"
        ),
        knowledge="total item units means SUM(order_items.quantity) per completed order.",
        tags=("single_db_lab", "having", "count"),
    )


def _discounted_revenue_between(start: str, end: str) -> Task:
    return Task(
        question=f"What discounted revenue came from completed orders dated from {start} through {end}?",
        sql=(
            "SELECT ROUND(SUM(T2.quantity * T2.unit_price * (1 - T1.discount_pct / 100.0)), 2) "
            "FROM orders AS T1 INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.status = 'completed' AND T1.order_date BETWEEN '{start}' AND '{end}'"
        ),
        knowledge="orders.discount_pct is stored on orders; item quantity and unit_price are stored on order_items.",
        tags=("single_db_lab", "targeted_v2", "column_ownership", "revenue"),
    )


def _completed_quantity_by_discount_floor(min_discount: float) -> Task:
    return Task(
        question=f"How many item units were in completed orders with discounts of at least {min_discount:g} percent?",
        sql=(
            "SELECT SUM(T2.quantity) FROM orders AS T1 "
            "INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.status = 'completed' AND T1.discount_pct >= {min_discount:g}"
        ),
        knowledge="discount_pct belongs to orders; quantity belongs to order_items.",
        tags=("single_db_lab", "targeted_v2", "column_ownership", "join_path"),
    )


def _discounted_order_ids(min_discount: float) -> Task:
    return Task(
        question=f"List completed order IDs with a discount greater than {min_discount:g} percent in ascending order.",
        sql=(
            "SELECT order_id FROM orders "
            f"WHERE status = 'completed' AND discount_pct > {min_discount:g} ORDER BY order_id ASC"
        ),
        knowledge="discount_pct belongs to orders and is expressed as a percentage.",
        tags=("single_db_lab", "targeted_v2", "discount", "order_by"),
        order_sensitive=True,
    )


def _average_discount_by_channel(channel: str) -> Task:
    return Task(
        question=f"What is the average discount percent for completed {channel} orders?",
        sql=(
            "SELECT ROUND(AVG(discount_pct), 2) FROM orders "
            f"WHERE channel = '{channel}' AND status = 'completed'"
        ),
        knowledge="discount_pct is stored on orders, not on order_items.",
        tags=("single_db_lab", "targeted_v2", "discount", "aggregation"),
    )


def _line_count_by_order_status(status: str) -> Task:
    return Task(
        question=f"How many order item rows belong to {status} orders?",
        sql=(
            "SELECT COUNT(*) FROM orders AS T1 "
            "INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.status = '{status}'"
        ),
        knowledge="order status belongs to orders; item rows are in order_items.",
        tags=("single_db_lab", "targeted_v2", "column_ownership", "join_path"),
    )


def _quantity_by_region_and_category(region: str, category: str) -> Task:
    return Task(
        question=f"How many completed-order units did {region} customers buy from the {category} category?",
        sql=(
            "SELECT SUM(T3.quantity) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "INNER JOIN products AS T4 ON T3.product_id = T4.product_id "
            f"WHERE T1.region = '{region}' AND T4.category = '{category}' AND T2.status = 'completed'"
        ),
        knowledge="region is on customers, category is on products, and quantity is on order_items.",
        tags=("single_db_lab", "targeted_v2", "column_ownership", "join_path"),
    )


def _category_revenue_for_region(category: str, region: str) -> Task:
    return Task(
        question=f"What discounted completed-order revenue came from {category} products bought by {region} customers?",
        sql=(
            "SELECT ROUND(SUM(T3.quantity * T3.unit_price * (1 - T2.discount_pct / 100.0)), 2) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "INNER JOIN products AS T4 ON T3.product_id = T4.product_id "
            f"WHERE T4.category = '{category}' AND T1.region = '{region}' AND T2.status = 'completed'"
        ),
        knowledge="category comes from products; discounted revenue uses order_items and orders.discount_pct.",
        tags=("single_db_lab", "targeted_v2", "revenue", "join_path"),
    )


def _product_quantity_for_channel(product_name: str, channel: str) -> Task:
    return Task(
        question=f"How many completed-order units of {product_name} were sold through the {channel} channel?",
        sql=(
            "SELECT SUM(T2.quantity) FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.product_name = '{product_name}' AND T3.channel = '{channel}' AND T3.status = 'completed'"
        ),
        knowledge="product_name belongs to products; channel and status belong to orders.",
        tags=("single_db_lab", "targeted_v2", "product", "join_path"),
    )


def _products_with_completed_sales_count(category: str) -> Task:
    return Task(
        question=f"How many active {category} products have at least one completed-order sale?",
        sql=(
            "SELECT COUNT(DISTINCT T1.product_id) FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.category = '{category}' AND T1.active = 1 AND T3.status = 'completed'"
        ),
        knowledge="active is on products; completed sales require an order_items to orders join.",
        tags=("single_db_lab", "targeted_v2", "products", "join_path"),
    )


def _active_products_without_completed_sales_count(category: str) -> Task:
    return Task(
        question=f"How many active {category} products have zero completed-order sales?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.product_id FROM products AS T1 "
            "LEFT JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "LEFT JOIN orders AS T3 ON T2.order_id = T3.order_id AND T3.status = 'completed' "
            f"WHERE T1.category = '{category}' AND T1.active = 1 "
            "GROUP BY T1.product_id HAVING COUNT(T3.order_id) = 0)"
        ),
        knowledge="zero completed sales is an anti-join from products through order_items to completed orders.",
        tags=("single_db_lab", "targeted_v2", "anti_join", "products"),
    )


def _carrier_delivery_after_date(carrier: str, start: str) -> Task:
    return Task(
        question=f"What is the average delivery time in days for {carrier} shipments shipped on or after {start}?",
        sql=(
            "SELECT ROUND(AVG(julianday(delivery_date) - julianday(shipped_date)), 2) "
            f"FROM shipments WHERE carrier = '{carrier}' AND shipped_date >= '{start}'"
        ),
        knowledge="shipped_date and delivery_date are both stored on shipments.",
        tags=("single_db_lab", "targeted_v2", "shipment", "date_math"),
    )


def _shipping_cost_for_carrier(carrier: str) -> Task:
    return Task(
        question=f"What total shipping cost is recorded for {carrier} shipments?",
        sql=f"SELECT ROUND(SUM(shipping_cost), 2) FROM shipments WHERE carrier = '{carrier}'",
        knowledge="shipping_cost and carrier are stored on shipments.",
        tags=("single_db_lab", "targeted_v2", "shipment", "aggregation"),
    )


def _shipping_cost_for_region_and_carrier(region: str, carrier: str) -> Task:
    return Task(
        question=f"What shipping cost total came from {region} completed orders shipped by {carrier}?",
        sql=(
            "SELECT ROUND(SUM(T3.shipping_cost), 2) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN shipments AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed' AND T3.carrier = '{carrier}'"
        ),
        knowledge="region comes from customers; carrier and shipping_cost come from shipments.",
        tags=("single_db_lab", "targeted_v2", "shipment", "join_path"),
    )


def _orders_delivered_after_days(min_days: int) -> Task:
    return Task(
        question=f"How many shipped orders took more than {min_days} days to deliver?",
        sql=(
            "SELECT COUNT(*) FROM shipments "
            f"WHERE julianday(delivery_date) - julianday(shipped_date) > {min_days}"
        ),
        knowledge="delivery time uses delivery_date minus shipped_date on shipments.",
        tags=("single_db_lab", "targeted_v2", "shipment", "date_math"),
    )


def _returns_after_slow_shipment_count(min_days: int) -> Task:
    return Task(
        question=f"How many returns are tied to shipments that took more than {min_days} days?",
        sql=(
            "SELECT COUNT(*) FROM returns AS T1 "
            "INNER JOIN shipments AS T2 ON T1.order_id = T2.order_id "
            f"WHERE julianday(T2.delivery_date) - julianday(T2.shipped_date) > {min_days}"
        ),
        knowledge="returns join to shipments by order_id; shipment duration comes from shipments dates.",
        tags=("single_db_lab", "targeted_v2", "returns", "shipment"),
    )


def _refund_total_for_region_and_reason(region: str, reason: str) -> Task:
    return Task(
        question=f"What refund total came from {region} customers for returns with reason {reason}?",
        sql=(
            "SELECT ROUND(SUM(T3.refund_amount), 2) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN returns AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T3.reason = '{reason}'"
        ),
        knowledge="refund_amount and reason are on returns; region comes through orders to customers.",
        tags=("single_db_lab", "targeted_v2", "returns", "join_path"),
    )


def _refund_count_by_tier_and_reason(tier: str, reason: str) -> Task:
    return Task(
        question=f"How many returns with reason {reason} came from {tier} loyalty customers?",
        sql=(
            "SELECT COUNT(*) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN returns AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.loyalty_tier = '{tier}' AND T3.reason = '{reason}'"
        ),
        knowledge="loyalty_tier comes from customers; return reason comes from returns.",
        tags=("single_db_lab", "targeted_v2", "returns", "join_path"),
    )


def _returned_products_for_region(region: str) -> Task:
    return Task(
        question=f"List product names returned by {region} customers sorted alphabetically.",
        sql=(
            "SELECT DISTINCT T4.product_name FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN returns AS T3 ON T2.order_id = T3.order_id "
            "INNER JOIN products AS T4 ON T3.product_id = T4.product_id "
            f"WHERE T1.region = '{region}' ORDER BY T4.product_name ASC"
        ),
        knowledge="returned product names require customers to orders to returns to products.",
        tags=("single_db_lab", "targeted_v2", "returns", "order_by"),
        order_sensitive=True,
    )


def _customers_with_returns_in_region(region: str) -> Task:
    return Task(
        question=f"How many distinct {region} customers have at least one return?",
        sql=(
            "SELECT COUNT(DISTINCT T1.customer_id) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN returns AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}'"
        ),
        knowledge="returns join to customers through orders.",
        tags=("single_db_lab", "targeted_v2", "returns", "join_path"),
    )


def _ticket_count_by_region_and_resolution(region: str, resolved: int) -> Task:
    resolution = "resolved" if resolved else "unresolved"
    return Task(
        question=f"How many {resolution} support tickets were opened by {region} customers?",
        sql=(
            "SELECT COUNT(*) FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.region = '{region}' AND T2.resolved = {resolved}"
        ),
        knowledge="support_tickets.resolved uses 1 for resolved and 0 for unresolved.",
        tags=("single_db_lab", "targeted_v2", "support", "join_path"),
    )


def _unresolved_ticket_customers_by_issue(issue_type: str) -> Task:
    return Task(
        question=f"List customers with unresolved {issue_type} tickets sorted by customer name.",
        sql=(
            "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T2.issue_type = '{issue_type}' AND T2.resolved = 0 ORDER BY T1.customer_name ASC"
        ),
        knowledge="issue_type and resolved are stored on support_tickets.",
        tags=("single_db_lab", "targeted_v2", "support", "order_by"),
        order_sensitive=True,
    )


def _ticket_count_for_order_status(status: str) -> Task:
    return Task(
        question=f"How many support tickets are linked to {status} orders?",
        sql=(
            "SELECT COUNT(*) FROM support_tickets AS T1 "
            "INNER JOIN orders AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T2.status = '{status}'"
        ),
        knowledge="support tickets may link to orders through support_tickets.order_id.",
        tags=("single_db_lab", "targeted_v2", "support", "join_path"),
    )


def _avg_ticket_score_by_region(region: str) -> Task:
    return Task(
        question=f"What is the average resolved-ticket satisfaction score for {region} customers?",
        sql=(
            "SELECT ROUND(AVG(T2.satisfaction_score), 2) FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.region = '{region}' AND T2.resolved = 1"
        ),
        knowledge="only resolved tickets have satisfaction_score values.",
        tags=("single_db_lab", "targeted_v2", "support", "aggregation"),
    )


def _orders_with_support_issue_type(issue_type: str) -> Task:
    return Task(
        question=f"List order IDs linked to {issue_type} support tickets sorted ascending.",
        sql=(
            "SELECT DISTINCT order_id FROM support_tickets "
            f"WHERE issue_type = '{issue_type}' AND order_id IS NOT NULL ORDER BY order_id ASC"
        ),
        knowledge="support_tickets.order_id links a ticket to an order when present.",
        tags=("single_db_lab", "targeted_v2", "support", "order_by"),
        order_sensitive=True,
    )


def _completed_order_count_by_region_channel(region: str, channel: str) -> Task:
    return Task(
        question=f"How many completed {channel} orders were placed by {region} customers?",
        sql=(
            "SELECT COUNT(*) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.region = '{region}' AND T2.channel = '{channel}' AND T2.status = 'completed'"
        ),
        knowledge="region is stored on customers; channel and status are stored on orders.",
        tags=("single_db_lab", "targeted_v2", "orders", "join_path"),
    )


def _repeat_customer_count_by_region(region: str) -> Task:
    return Task(
        question=f"How many {region} customers placed more than one completed order?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.customer_id, COUNT(*) AS completed_orders "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed' "
            "GROUP BY T1.customer_id HAVING completed_orders > 1)"
        ),
        knowledge="repeat customers are counted after grouping completed orders by customer.",
        tags=("single_db_lab", "targeted_v2", "having", "join_path"),
    )


def _customers_with_at_least_completed_units(region: str, min_units: int) -> Task:
    return Task(
        question=f"How many {region} customers bought at least {min_units} units across completed orders?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.customer_id, SUM(T3.quantity) AS units "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed' "
            f"GROUP BY T1.customer_id HAVING units >= {min_units})"
        ),
        knowledge="customer unit totals require customers to orders to order_items.",
        tags=("single_db_lab", "targeted_v2", "having", "join_path"),
    )


def _orders_with_more_than_products(min_product_count: int) -> Task:
    return Task(
        question=f"How many completed orders contain more than {min_product_count} distinct products?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.order_id, COUNT(DISTINCT T2.product_id) AS product_count "
            "FROM orders AS T1 INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            "WHERE T1.status = 'completed' GROUP BY T1.order_id "
            f"HAVING product_count > {min_product_count})"
        ),
        knowledge="distinct product count is calculated from order_items grouped by completed order.",
        tags=("single_db_lab", "targeted_v2", "having", "order_items"),
    )


def _order_unit_total_over(min_units: int) -> Task:
    return Task(
        question=f"List completed order IDs with total units greater than {min_units}, sorted ascending.",
        sql=(
            "SELECT T1.order_id FROM orders AS T1 "
            "INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            "WHERE T1.status = 'completed' GROUP BY T1.order_id "
            f"HAVING SUM(T2.quantity) > {min_units} ORDER BY T1.order_id ASC"
        ),
        knowledge="total units are SUM(order_items.quantity) grouped by order_id.",
        tags=("single_db_lab", "targeted_v2", "having", "order_by"),
        order_sensitive=True,
    )


def _month_revenue_by_category(month: str, category: str) -> Task:
    return Task(
        question=f"What discounted completed-order revenue came from {category} products in {month}?",
        sql=(
            "SELECT ROUND(SUM(T2.quantity * T2.unit_price * (1 - T3.discount_pct / 100.0)), 2) "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.category = '{category}' AND T3.status = 'completed' AND substr(T3.order_date, 1, 7) = '{month}'"
        ),
        knowledge="category comes from products; month filtering uses orders.order_date.",
        tags=("single_db_lab", "targeted_v2", "revenue", "date_filter"),
    )


def _month_return_count(month: str) -> Task:
    return Task(
        question=f"How many returns were recorded in {month}?",
        sql=f"SELECT COUNT(*) FROM returns WHERE substr(return_date, 1, 7) = '{month}'",
        knowledge="return_date is stored on returns.",
        tags=("single_db_lab", "targeted_v2", "returns", "date_filter"),
    )


def _quarter_revenue_for_tier(tier: str, start: str, end: str) -> Task:
    return Task(
        question=f"What discounted completed-order revenue came from {tier} loyalty customers from {start} through {end}?",
        sql=(
            "SELECT ROUND(SUM(T3.quantity * T3.unit_price * (1 - T2.discount_pct / 100.0)), 2) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.loyalty_tier = '{tier}' AND T2.status = 'completed' "
            f"AND T2.order_date BETWEEN '{start}' AND '{end}'"
        ),
        knowledge="loyalty_tier is on customers; discounted revenue uses orders and order_items.",
        tags=("single_db_lab", "targeted_v2", "revenue", "join_path"),
    )


def _new_customer_completed_orders(start: str) -> Task:
    return Task(
        question=f"How many completed orders were placed by customers who signed up on or after {start}?",
        sql=(
            "SELECT COUNT(*) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.signup_date >= '{start}' AND T2.status = 'completed'"
        ),
        knowledge="signup_date is stored on customers; order status is stored on orders.",
        tags=("single_db_lab", "targeted_v2", "customers", "join_path"),
    )


def _product_names_for_return_reason(reason: str) -> Task:
    return Task(
        question=f"List product names with returns for reason {reason}, sorted alphabetically.",
        sql=(
            "SELECT DISTINCT T2.product_name FROM returns AS T1 "
            "INNER JOIN products AS T2 ON T1.product_id = T2.product_id "
            f"WHERE T1.reason = '{reason}' ORDER BY T2.product_name ASC"
        ),
        knowledge="return reason is on returns; product_name is on products.",
        tags=("single_db_lab", "targeted_v2", "returns", "products"),
        order_sensitive=True,
    )


def _refund_total_for_carrier(carrier: str) -> Task:
    return Task(
        question=f"What refund amount is tied to returned orders shipped by {carrier}?",
        sql=(
            "SELECT ROUND(SUM(T1.refund_amount), 2) FROM returns AS T1 "
            "INNER JOIN shipments AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T2.carrier = '{carrier}'"
        ),
        knowledge="refund_amount is on returns; carrier is on shipments.",
        tags=("single_db_lab", "targeted_v2", "returns", "shipment"),
    )


def _return_rate_for_region(region: str) -> Task:
    return Task(
        question=f"What share of completed orders from {region} customers had at least one return?",
        sql=(
            "SELECT ROUND(CAST(COUNT(DISTINCT T3.order_id) AS REAL) / COUNT(DISTINCT T2.order_id), 3) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "LEFT JOIN returns AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed'"
        ),
        knowledge="the denominator is completed orders; the numerator is distinct returned completed orders.",
        tags=("single_db_lab", "targeted_v2", "ratio", "returns"),
    )


def _category_return_count(category: str) -> Task:
    return Task(
        question=f"How many return rows are for products in the {category} category?",
        sql=(
            "SELECT COUNT(*) FROM returns AS T1 "
            "INNER JOIN products AS T2 ON T1.product_id = T2.product_id "
            f"WHERE T2.category = '{category}'"
        ),
        knowledge="product category comes from products; return rows come from returns.",
        tags=("single_db_lab", "targeted_v2", "returns", "products"),
    )


def _product_names_never_ordered() -> Task:
    return Task(
        question="List product names that never appear in any order item, sorted alphabetically.",
        sql=(
            "SELECT T1.product_name FROM products AS T1 "
            "LEFT JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "WHERE T2.item_id IS NULL ORDER BY T1.product_name ASC"
        ),
        knowledge="products with no order_items rows require a left anti-join.",
        tags=("single_db_lab", "targeted_v2", "anti_join", "products"),
        order_sensitive=True,
    )


def _top_region_by_completed_units() -> Task:
    return Task(
        question="Which customer region bought the most units in completed orders?",
        sql=(
            "SELECT T1.region, SUM(T3.quantity) AS units FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "WHERE T2.status = 'completed' GROUP BY T1.region ORDER BY units DESC, T1.region ASC LIMIT 1"
        ),
        knowledge="units are stored on order_items; regions are stored on customers.",
        tags=("single_db_lab", "targeted_v2", "ranking", "join_path"),
        order_sensitive=True,
    )


def _carrier_count_by_region(region: str) -> Task:
    return Task(
        question=f"List carriers used for completed {region} orders with shipment counts, sorted by carrier.",
        sql=(
            "SELECT T3.carrier, COUNT(*) AS shipment_count FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN shipments AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed' "
            "GROUP BY T3.carrier ORDER BY T3.carrier ASC"
        ),
        knowledge="carrier is on shipments; region is on customers.",
        tags=("single_db_lab", "targeted_v2", "shipment", "order_by"),
        order_sensitive=True,
    )


def _unresolved_ticket_order_ids() -> Task:
    return Task(
        question="List order IDs for unresolved support tickets sorted ascending.",
        sql=(
            "SELECT order_id FROM support_tickets "
            "WHERE resolved = 0 AND order_id IS NOT NULL ORDER BY order_id ASC"
        ),
        knowledge="unresolved tickets have support_tickets.resolved = 0.",
        tags=("single_db_lab", "targeted_v2", "support", "order_by"),
        order_sensitive=True,
    )


def _category_revenue_rank_for_channel(channel: str) -> Task:
    return Task(
        question=f"Which category has the highest discounted completed-order revenue in the {channel} channel?",
        sql=(
            "SELECT T1.category, ROUND(SUM(T2.quantity * T2.unit_price * (1 - T3.discount_pct / 100.0)), 2) AS revenue "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T3.status = 'completed' AND T3.channel = '{channel}' "
            "GROUP BY T1.category ORDER BY revenue DESC, T1.category ASC LIMIT 1"
        ),
        knowledge="category belongs to products; discount_pct belongs to orders; quantity and unit_price belong to order_items.",
        tags=("single_db_lab", "targeted_v3", "grouped_ranking", "alias_ownership"),
        order_sensitive=True,
    )


def _category_revenue_rank_for_region(region: str) -> Task:
    return Task(
        question=f"Which category has the highest discounted completed-order revenue for {region} customers?",
        sql=(
            "SELECT T1.category, ROUND(SUM(T2.quantity * T2.unit_price * (1 - T3.discount_pct / 100.0)), 2) AS revenue "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "INNER JOIN customers AS T4 ON T3.customer_id = T4.customer_id "
            f"WHERE T3.status = 'completed' AND T4.region = '{region}' "
            "GROUP BY T1.category ORDER BY revenue DESC, T1.category ASC LIMIT 1"
        ),
        knowledge="regional category revenue joins products to order_items to orders to customers.",
        tags=("single_db_lab", "targeted_v3", "grouped_ranking", "alias_ownership"),
        order_sensitive=True,
    )


def _category_revenue_rank_for_month(month: str) -> Task:
    return Task(
        question=f"Which category had the highest discounted completed-order revenue in {month}?",
        sql=(
            "SELECT T1.category, ROUND(SUM(T2.quantity * T2.unit_price * (1 - T3.discount_pct / 100.0)), 2) AS revenue "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T3.status = 'completed' AND substr(T3.order_date, 1, 7) = '{month}' "
            "GROUP BY T1.category ORDER BY revenue DESC, T1.category ASC LIMIT 1"
        ),
        knowledge="month filtering uses orders.order_date; revenue uses item values and orders.discount_pct.",
        tags=("single_db_lab", "targeted_v3", "grouped_ranking", "alias_ownership"),
        order_sensitive=True,
    )


def _return_item_share_for_category(category: str) -> Task:
    return Task(
        question=f"What share of completed-order item rows in {category} products had a matching return row?",
        sql=(
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T2.item_id), 3) "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T2.order_id = T4.order_id AND T2.product_id = T4.product_id "
            f"WHERE T1.category = '{category}' AND T3.status = 'completed'"
        ),
        knowledge="the denominator is completed-order item rows; the numerator is matching return rows.",
        tags=("single_db_lab", "targeted_v3", "ratio_denominator", "alias_ownership"),
    )


def _return_item_share_for_channel(channel: str) -> Task:
    return Task(
        question=f"What share of completed-order item rows from the {channel} channel had a matching return row?",
        sql=(
            "SELECT ROUND(CAST(COUNT(T3.return_id) AS REAL) / COUNT(T2.item_id), 3) "
            "FROM orders AS T1 INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            "LEFT JOIN returns AS T3 ON T2.order_id = T3.order_id AND T2.product_id = T3.product_id "
            f"WHERE T1.channel = '{channel}' AND T1.status = 'completed'"
        ),
        knowledge="item-row return share divides matching returns by completed-order item rows.",
        tags=("single_db_lab", "targeted_v3", "ratio_denominator", "alias_ownership"),
    )


def _carrier_fastest_delivery() -> Task:
    return Task(
        question="Which carrier has the fastest average delivery time?",
        sql=(
            "SELECT T1.carrier, ROUND(AVG(julianday(T1.delivery_date) - julianday(T1.shipped_date)), 2) AS avg_days "
            "FROM shipments AS T1 GROUP BY T1.carrier ORDER BY avg_days ASC, T1.carrier ASC LIMIT 1"
        ),
        knowledge="carrier, shipped_date, and delivery_date are all stored on shipments.",
        tags=("single_db_lab", "targeted_v3", "grouped_ranking", "alias_ownership"),
        order_sensitive=True,
    )


def _carrier_slowest_delivery_after_date(start: str) -> Task:
    return Task(
        question=f"Which carrier has the slowest average delivery time for shipments on or after {start}?",
        sql=(
            "SELECT T1.carrier, ROUND(AVG(julianday(T1.delivery_date) - julianday(T1.shipped_date)), 2) AS avg_days "
            "FROM shipments AS T1 "
            f"WHERE T1.shipped_date >= '{start}' "
            "GROUP BY T1.carrier ORDER BY avg_days DESC, T1.carrier ASC LIMIT 1"
        ),
        knowledge="shipment ranking groups by shipments.carrier and computes delivery duration from shipment dates.",
        tags=("single_db_lab", "targeted_v3", "grouped_ranking", "alias_ownership"),
        order_sensitive=True,
    )


def _carrier_delivery_rank_for_region(region: str) -> Task:
    return Task(
        question=f"Which carrier has the slowest average delivery time for completed {region} orders?",
        sql=(
            "SELECT T3.carrier, ROUND(AVG(julianday(T3.delivery_date) - julianday(T3.shipped_date)), 2) AS avg_days "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN shipments AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed' "
            "GROUP BY T3.carrier ORDER BY avg_days DESC, T3.carrier ASC LIMIT 1"
        ),
        knowledge="carrier is on shipments; region is on customers; completed status is on orders.",
        tags=("single_db_lab", "targeted_v3", "grouped_ranking", "alias_ownership"),
        order_sensitive=True,
    )


def _active_product_no_completed_sales_names_by_category(category: str) -> Task:
    return Task(
        question=f"List active {category} products with no completed-order sales, sorted alphabetically.",
        sql=(
            "SELECT T1.product_name FROM products AS T1 "
            "LEFT JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "LEFT JOIN orders AS T3 ON T2.order_id = T3.order_id AND T3.status = 'completed' "
            f"WHERE T1.active = 1 AND T1.category = '{category}' "
            "GROUP BY T1.product_id, T1.product_name HAVING COUNT(T3.order_id) = 0 "
            "ORDER BY T1.product_name ASC"
        ),
        knowledge="this is a left anti-join over completed-order sales grouped by product.",
        tags=("single_db_lab", "targeted_v3", "anti_join_list", "alias_ownership"),
        order_sensitive=True,
    )


def _active_product_no_completed_sales_count_by_channel(channel: str) -> Task:
    return Task(
        question=f"How many active products have no completed-order sales in the {channel} channel?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.product_id FROM products AS T1 "
            "LEFT JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            f"LEFT JOIN orders AS T3 ON T2.order_id = T3.order_id AND T3.status = 'completed' AND T3.channel = '{channel}' "
            "WHERE T1.active = 1 GROUP BY T1.product_id HAVING COUNT(T3.order_id) = 0)"
        ),
        knowledge="the channel filter belongs in the joined orders condition for this completed-sales anti-join.",
        tags=("single_db_lab", "targeted_v3", "anti_join_list", "alias_ownership"),
    )


def _repeat_customer_names() -> Task:
    return Task(
        question="List customers who placed more than one completed order, sorted by customer name.",
        sql=(
            "SELECT T1.customer_name FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "WHERE T2.status = 'completed' GROUP BY T1.customer_id, T1.customer_name "
            "HAVING COUNT(T2.order_id) > 1 ORDER BY T1.customer_name ASC"
        ),
        knowledge="repeat customers require grouping completed orders by customer and applying HAVING.",
        tags=("single_db_lab", "targeted_v3", "global_having", "alias_ownership"),
        order_sensitive=True,
    )


def _repeat_customer_count_by_tier(tier: str) -> Task:
    return Task(
        question=f"How many {tier} customers placed more than one completed order?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.customer_id FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.loyalty_tier = '{tier}' AND T2.status = 'completed' "
            "GROUP BY T1.customer_id HAVING COUNT(T2.order_id) > 1)"
        ),
        knowledge="repeat-customer counts are computed after grouping completed orders by customer.",
        tags=("single_db_lab", "targeted_v3", "global_having", "alias_ownership"),
    )


def _top_completed_order_product_by_channel(channel: str) -> Task:
    return Task(
        question=f"Which product had the most completed-order units in the {channel} channel?",
        sql=(
            "SELECT T1.product_name, SUM(T2.quantity) AS units FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T3.status = 'completed' AND T3.channel = '{channel}' "
            "GROUP BY T1.product_id, T1.product_name ORDER BY units DESC, T1.product_name ASC LIMIT 1"
        ),
        knowledge="quantity belongs to order_items; product_name belongs to products; channel and status belong to orders.",
        tags=("single_db_lab", "targeted_v3", "grouped_ranking", "alias_ownership"),
        order_sensitive=True,
    )


def _refund_total_for_slow_shipments(min_days: int) -> Task:
    return Task(
        question=f"What refund amount is tied to returns whose order shipment took more than {min_days} days?",
        sql=(
            "SELECT ROUND(SUM(T1.refund_amount), 2) FROM returns AS T1 "
            "INNER JOIN shipments AS T2 ON T1.order_id = T2.order_id "
            f"WHERE julianday(T2.delivery_date) - julianday(T2.shipped_date) > {min_days}"
        ),
        knowledge="refund_amount is on returns; shipment duration is computed from shipment dates.",
        tags=("single_db_lab", "targeted_v3", "shipment_return_join", "alias_ownership"),
    )


def _refund_total_for_slow_shipments_by_reason(reason: str, min_days: int) -> Task:
    return Task(
        question=f"What refund amount for {reason} returns is tied to shipments that took more than {min_days} days?",
        sql=(
            "SELECT ROUND(SUM(T1.refund_amount), 2) FROM returns AS T1 "
            "INNER JOIN shipments AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.reason = '{reason}' AND julianday(T2.delivery_date) - julianday(T2.shipped_date) > {min_days}"
        ),
        knowledge="return reason and refund_amount are on returns; delivery duration is on shipments.",
        tags=("single_db_lab", "targeted_v3", "shipment_return_join", "alias_ownership"),
    )


def _refund_total_for_slow_shipments_by_carrier(carrier: str, min_days: int) -> Task:
    return Task(
        question=f"What refund amount is tied to returned {carrier} shipments that took more than {min_days} days?",
        sql=(
            "SELECT ROUND(SUM(T1.refund_amount), 2) FROM returns AS T1 "
            "INNER JOIN shipments AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T2.carrier = '{carrier}' AND julianday(T2.delivery_date) - julianday(T2.shipped_date) > {min_days}"
        ),
        knowledge="carrier and delivery duration are on shipments; refund_amount is on returns.",
        tags=("single_db_lab", "targeted_v3", "shipment_return_join", "alias_ownership"),
    )


def _train_exp051_support_tasks() -> list[Task]:
    return [
        _unresolved_ticket_count_all(),
        _unresolved_ticket_count_by_region("West"),
        _unresolved_ticket_count_by_region("South"),
        _unresolved_ticket_count_by_tier("Silver"),
        _unresolved_ticket_count_by_tier("Gold"),
        _unresolved_ticket_customers_for_region("West"),
        _unresolved_ticket_customers_for_region("South"),
        _unresolved_ticket_customers_for_tier("Silver"),
        _ticket_customers_by_issue_and_resolution("shipping", 1),
        _ticket_customers_by_issue_and_resolution("return", 0),
        _ticket_count_by_issue_without_resolution("shipping"),
        _ticket_count_by_issue_without_resolution("return"),
        _resolved_ticket_customers_any_issue(),
        _ticket_count_with_no_order_link(),
    ]


def _train_exp052_date_tasks() -> list[Task]:
    return [
        _completed_orders_after_date("2024-05-01"),
        _completed_orders_on_or_after_date("2024-05-01"),
        _completed_orders_before_date("2024-04-01"),
        _completed_orders_through_date("2024-04-30"),
        _completed_orders_in_month("2024-02"),
        _completed_orders_in_date_range("2024-02-01", "2024-04-30"),
        _top_product_units_after_date("2024-04-01"),
        _top_product_units_on_or_after_date("2024-04-01"),
        _returns_after_date("2024-05-01"),
        _returns_in_month("2024-07"),
        _shipment_count_after_ship_date("2024-05-01"),
        _shipment_count_delivered_through("2024-04-30"),
        _revenue_strictly_after_date("2024-03-01"),
        _revenue_on_or_after_date("2024-03-01"),
    ]


def _train_exp053_return_ratio_tasks() -> list[Task]:
    return [
        _return_item_share_for_product("Nimbus Jacket"),
        _return_item_share_for_product("Trail Boots"),
        _return_item_share_for_product("Travel Backpack"),
        _return_item_share_for_region("West"),
        _return_item_share_for_region("South"),
        _return_item_share_for_tier("Gold"),
        _return_item_share_for_tier("Silver"),
        _return_order_share_for_category("Bags"),
        _return_order_share_for_region("West"),
        _return_order_share_for_channel("web"),
        _return_item_share_for_channel_and_category("web", "Apparel"),
        _return_item_share_for_channel_and_category("marketplace", "Bags"),
        _returned_item_row_count_for_category("Footwear"),
        _completed_item_row_count_for_category("Footwear"),
    ]


def _train_exp054_having_tasks() -> list[Task]:
    return [
        _repeat_customer_count_min_orders(2),
        _repeat_customer_count_min_orders(3),
        _repeat_customer_names_min_orders(2),
        _repeat_customer_names_min_orders(3),
        _products_with_more_than_completed_units(2),
        _products_with_more_than_completed_units(3),
        _categories_with_more_than_completed_units(4),
        _categories_with_more_than_completed_units(8),
        _customers_with_more_than_return_count(0),
        _customers_with_more_than_return_count(1),
        _orders_with_at_least_distinct_products(2),
        _orders_with_at_least_distinct_products(3),
        _channels_with_more_than_completed_orders(4),
        _regions_with_more_than_completed_orders(4),
    ]


def _train_exp055_antijoin_tasks() -> list[Task]:
    return [
        _customers_with_no_support_tickets(),
        _customers_with_no_unresolved_tickets(),
        _orders_with_no_returns(),
        _completed_orders_with_no_returns(),
        _products_with_no_returns(),
        _active_products_with_no_returns(),
        _customers_with_completed_orders_no_returns(),
        _products_with_no_completed_sales_in_region("South"),
        _products_with_no_completed_sales_in_region("West"),
        _customers_with_no_completed_orders(),
        _customers_with_no_marketplace_completed_orders(),
        _orders_with_no_support_ticket(),
        _products_with_no_completed_sales_after_date("2024-05-01"),
        _products_with_no_completed_sales_before_date("2024-03-01"),
    ]


def _train_exp059_alias_contrast_tasks() -> list[Task]:
    return [
        Task(
            question="List Gold customers with unresolved support tickets, sorted by customer name.",
            sql=(
                "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
                "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
                "WHERE T1.loyalty_tier = 'Gold' AND T2.resolved = 0 ORDER BY T1.customer_name"
            ),
            knowledge="loyalty_tier belongs to customers; resolved belongs to support_tickets.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "support"),
            order_sensitive=True,
        ),
        Task(
            question="List customers who placed marketplace completed orders, sorted by customer name.",
            sql=(
                "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
                "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
                "WHERE T2.channel = 'marketplace' AND T2.status = 'completed' ORDER BY T1.customer_name"
            ),
            knowledge="channel and status belong to orders; customer_name belongs to customers.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "order_customer_join"),
            order_sensitive=True,
        ),
        Task(
            question="What is the total refund amount for returned Bags products?",
            sql=(
                "SELECT ROUND(SUM(T1.refund_amount), 2) FROM returns AS T1 "
                "INNER JOIN products AS T2 ON T1.product_id = T2.product_id "
                "WHERE T2.category = 'Bags'"
            ),
            knowledge="refund_amount belongs to returns; category belongs to products.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "return_product_join"),
        ),
        Task(
            question="What is the total FedEx shipping cost for orders that had returns?",
            sql=(
                "SELECT ROUND(SUM(T1.shipping_cost), 2) FROM shipments AS T1 "
                "INNER JOIN returns AS T2 ON T1.order_id = T2.order_id "
                "WHERE T1.carrier = 'FedEx'"
            ),
            knowledge="carrier and shipping_cost belong to shipments; returns only establishes the returned-order join.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "shipment_return_join"),
        ),
        Task(
            question="How many South customers placed completed store orders?",
            sql=(
                "SELECT COUNT(DISTINCT T1.customer_id) FROM customers AS T1 "
                "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
                "WHERE T1.region = 'South' AND T2.channel = 'store' AND T2.status = 'completed'"
            ),
            knowledge="region belongs to customers; channel and status belong to orders.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "order_customer_join"),
        ),
        Task(
            question="Which active Apparel products were sold in completed web orders, sorted by product name?",
            sql=(
                "SELECT DISTINCT T1.product_name FROM products AS T1 "
                "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
                "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
                "WHERE T1.active = 1 AND T1.category = 'Apparel' AND T3.channel = 'web' "
                "AND T3.status = 'completed' ORDER BY T1.product_name"
            ),
            knowledge="active and category belong to products; channel and status belong to orders.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "product_order_join"),
            order_sensitive=True,
        ),
        Task(
            question="What average satisfaction score did resolved shipping tickets from West customers receive?",
            sql=(
                "SELECT ROUND(AVG(T2.satisfaction_score), 2) FROM customers AS T1 "
                "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
                "WHERE T1.region = 'West' AND T2.issue_type = 'shipping' AND T2.resolved = 1"
            ),
            knowledge="region belongs to customers; issue_type, resolved, and satisfaction_score belong to support_tickets.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "support"),
        ),
        Task(
            question="How many completed orders used UPS shipments and came from Gold customers?",
            sql=(
                "SELECT COUNT(DISTINCT T2.order_id) FROM customers AS T1 "
                "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
                "INNER JOIN shipments AS T3 ON T2.order_id = T3.order_id "
                "WHERE T1.loyalty_tier = 'Gold' AND T2.status = 'completed' AND T3.carrier = 'UPS'"
            ),
            knowledge="loyalty_tier belongs to customers; status belongs to orders; carrier belongs to shipments.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "multi_join"),
        ),
        Task(
            question="What refund amount is tied to late_delivery returns on Trail Boots?",
            sql=(
                "SELECT ROUND(SUM(T1.refund_amount), 2) FROM returns AS T1 "
                "INNER JOIN products AS T2 ON T1.product_id = T2.product_id "
                "WHERE T1.reason = 'late_delivery' AND T2.product_name = 'Trail Boots'"
            ),
            knowledge="reason and refund_amount belong to returns; product_name belongs to products.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "return_product_join"),
        ),
        Task(
            question="List Midwest customers with completed orders that have support tickets, sorted by customer name.",
            sql=(
                "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
                "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
                "INNER JOIN support_tickets AS T3 ON T2.order_id = T3.order_id "
                "WHERE T1.region = 'Midwest' AND T2.status = 'completed' ORDER BY T1.customer_name"
            ),
            knowledge="region belongs to customers; status belongs to orders; support_tickets links by order_id.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "multi_join"),
            order_sensitive=True,
        ),
        Task(
            question="How many inactive products appeared in returns?",
            sql=(
                "SELECT COUNT(DISTINCT T1.product_id) FROM products AS T1 "
                "INNER JOIN returns AS T2 ON T1.product_id = T2.product_id WHERE T1.active = 0"
            ),
            knowledge="active belongs to products; returns establishes whether the product appeared in returns.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "return_product_join"),
        ),
        Task(
            question="What is the total quantity of active Footwear products in completed orders?",
            sql=(
                "SELECT SUM(T2.quantity) FROM products AS T1 "
                "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
                "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
                "WHERE T1.active = 1 AND T1.category = 'Footwear' AND T3.status = 'completed'"
            ),
            knowledge="active and category belong to products; quantity belongs to order_items; status belongs to orders.",
            tags=("single_db_lab", "targeted_exp059", "contrastive", "alias_ownership", "product_order_join"),
        ),
    ]


def _train_exp060_boundary_contrast_tasks() -> list[Task]:
    return [
        Task(
            question="How many completed orders were placed strictly before 2024-04-01?",
            sql="SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date < '2024-04-01'",
            knowledge="strictly before excludes the boundary date.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "date_filter"),
        ),
        Task(
            question="How many completed orders were placed on or before 2024-04-01?",
            sql="SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date <= '2024-04-01'",
            knowledge="on or before includes the boundary date.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "date_filter"),
        ),
        Task(
            question="How many returns happened strictly after 2024-04-19?",
            sql="SELECT COUNT(*) FROM returns WHERE return_date > '2024-04-19'",
            knowledge="strictly after excludes the boundary date.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "date_filter"),
        ),
        Task(
            question="How many returns happened on or after 2024-04-19?",
            sql="SELECT COUNT(*) FROM returns WHERE return_date >= '2024-04-19'",
            knowledge="on or after includes the boundary date.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "date_filter"),
        ),
        Task(
            question="List orders delivered in at most 3 days, sorted by order id.",
            sql=(
                "SELECT order_id FROM shipments "
                "WHERE julianday(delivery_date) - julianday(shipped_date) <= 3 ORDER BY order_id"
            ),
            knowledge="at most 3 days means duration <= 3.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "date_math"),
            order_sensitive=True,
        ),
        Task(
            question="List orders delivered in more than 3 days, sorted by order id.",
            sql=(
                "SELECT order_id FROM shipments "
                "WHERE julianday(delivery_date) - julianday(shipped_date) > 3 ORDER BY order_id"
            ),
            knowledge="more than 3 days means duration > 3.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "date_math"),
            order_sensitive=True,
        ),
        Task(
            question="How many products have completed sales quantity of at least 3 units?",
            sql=(
                "SELECT COUNT(*) FROM (SELECT T1.product_id, SUM(T1.quantity) AS units FROM order_items AS T1 "
                "INNER JOIN orders AS T2 ON T1.order_id = T2.order_id WHERE T2.status = 'completed' "
                "GROUP BY T1.product_id HAVING units >= 3)"
            ),
            knowledge="at least 3 means HAVING units >= 3.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "having"),
        ),
        Task(
            question="How many products have completed sales quantity greater than 3 units?",
            sql=(
                "SELECT COUNT(*) FROM (SELECT T1.product_id, SUM(T1.quantity) AS units FROM order_items AS T1 "
                "INNER JOIN orders AS T2 ON T1.order_id = T2.order_id WHERE T2.status = 'completed' "
                "GROUP BY T1.product_id HAVING units > 3)"
            ),
            knowledge="greater than 3 means HAVING units > 3.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "having"),
        ),
        Task(
            question="How many support tickets opened through 2024-04-30 remain unresolved?",
            sql="SELECT COUNT(*) FROM support_tickets WHERE opened_date <= '2024-04-30' AND resolved = 0",
            knowledge="through 2024-04-30 includes tickets opened on that date.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "support"),
        ),
        Task(
            question="How many support tickets opened after 2024-04-30 remain unresolved?",
            sql="SELECT COUNT(*) FROM support_tickets WHERE opened_date > '2024-04-30' AND resolved = 0",
            knowledge="after 2024-04-30 excludes tickets opened on that date.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "support"),
        ),
        Task(
            question="What completed order revenue was booked before May 2024?",
            sql=(
                "SELECT ROUND(SUM(T1.quantity * T1.unit_price * (1 - T2.discount_pct / 100.0)), 2) "
                "FROM order_items AS T1 INNER JOIN orders AS T2 ON T1.order_id = T2.order_id "
                "WHERE T2.status = 'completed' AND T2.order_date < '2024-05-01'"
            ),
            knowledge="before May 2024 means order_date < 2024-05-01.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "revenue"),
        ),
        Task(
            question="What completed order revenue was booked during May 2024 or later?",
            sql=(
                "SELECT ROUND(SUM(T1.quantity * T1.unit_price * (1 - T2.discount_pct / 100.0)), 2) "
                "FROM order_items AS T1 INNER JOIN orders AS T2 ON T1.order_id = T2.order_id "
                "WHERE T2.status = 'completed' AND T2.order_date >= '2024-05-01'"
            ),
            knowledge="during May 2024 or later means order_date >= 2024-05-01.",
            tags=("single_db_lab", "targeted_exp060", "contrastive", "boundary_semantics", "revenue"),
        ),
    ]


def _train_exp061_antijoin_contrast_tasks() -> list[Task]:
    return [
        Task(
            question="List customers with no unresolved support tickets, sorted by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 "
                "LEFT JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id AND T2.resolved = 0 "
                "WHERE T2.ticket_id IS NULL ORDER BY T1.customer_name"
            ),
            knowledge="put the unresolved predicate in the LEFT JOIN ON clause before checking for NULL.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List customers with no resolved support tickets, sorted by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 "
                "LEFT JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id AND T2.resolved = 1 "
                "WHERE T2.ticket_id IS NULL ORDER BY T1.customer_name"
            ),
            knowledge="put the resolved predicate in the LEFT JOIN ON clause before checking for NULL.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List products with no completed web sales, sorted by product name.",
            sql=(
                "SELECT T1.product_name FROM products AS T1 LEFT JOIN ("
                "SELECT DISTINCT T2.product_id FROM order_items AS T2 INNER JOIN orders AS T3 "
                "ON T2.order_id = T3.order_id WHERE T3.status = 'completed' AND T3.channel = 'web'"
                ") AS T4 ON T1.product_id = T4.product_id WHERE T4.product_id IS NULL ORDER BY T1.product_name"
            ),
            knowledge="anti-join against the set of products with completed web sales.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List products with no completed marketplace sales, sorted by product name.",
            sql=(
                "SELECT T1.product_name FROM products AS T1 LEFT JOIN ("
                "SELECT DISTINCT T2.product_id FROM order_items AS T2 INNER JOIN orders AS T3 "
                "ON T2.order_id = T3.order_id WHERE T3.status = 'completed' AND T3.channel = 'marketplace'"
                ") AS T4 ON T1.product_id = T4.product_id WHERE T4.product_id IS NULL ORDER BY T1.product_name"
            ),
            knowledge="anti-join against the set of products with completed marketplace sales.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List completed orders with no return rows, sorted by order id.",
            sql=(
                "SELECT T1.order_id FROM orders AS T1 LEFT JOIN returns AS T2 ON T1.order_id = T2.order_id "
                "WHERE T1.status = 'completed' AND T2.return_id IS NULL ORDER BY T1.order_id"
            ),
            knowledge="the completed-order filter is on the preserved orders table; NULL check is on returns.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List web orders with no support ticket, sorted by order id.",
            sql=(
                "SELECT T1.order_id FROM orders AS T1 LEFT JOIN support_tickets AS T2 ON T1.order_id = T2.order_id "
                "WHERE T1.channel = 'web' AND T2.ticket_id IS NULL ORDER BY T1.order_id"
            ),
            knowledge="the web filter is on the preserved orders table; NULL check is on support_tickets.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="How many active products had no returns?",
            sql=(
                "SELECT COUNT(*) FROM products AS T1 LEFT JOIN returns AS T2 ON T1.product_id = T2.product_id "
                "WHERE T1.active = 1 AND T2.return_id IS NULL"
            ),
            knowledge="active is on the preserved products table; NULL check is on returns.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
        ),
        Task(
            question="How many inactive products had no completed sales?",
            sql=(
                "SELECT COUNT(*) FROM products AS T1 LEFT JOIN ("
                "SELECT DISTINCT T2.product_id FROM order_items AS T2 INNER JOIN orders AS T3 "
                "ON T2.order_id = T3.order_id WHERE T3.status = 'completed'"
                ") AS T4 ON T1.product_id = T4.product_id WHERE T1.active = 0 AND T4.product_id IS NULL"
            ),
            knowledge="active is on products; anti-join against the set of products with completed sales.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
        ),
        Task(
            question="List customers with no completed store orders, sorted by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 LEFT JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
                "AND T2.status = 'completed' AND T2.channel = 'store' WHERE T2.order_id IS NULL ORDER BY T1.customer_name"
            ),
            knowledge="completed store predicates belong in the orders LEFT JOIN ON clause.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List customers with no completed marketplace orders, sorted by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 LEFT JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
                "AND T2.status = 'completed' AND T2.channel = 'marketplace' WHERE T2.order_id IS NULL "
                "ORDER BY T1.customer_name"
            ),
            knowledge="completed marketplace predicates belong in the orders LEFT JOIN ON clause.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List carriers with no shipments delivered after 2024-06-01, sorted by carrier.",
            sql=(
                "SELECT DISTINCT T1.carrier FROM shipments AS T1 LEFT JOIN shipments AS T2 ON T1.carrier = T2.carrier "
                "AND T2.delivery_date > '2024-06-01' WHERE T2.shipment_id IS NULL ORDER BY T1.carrier"
            ),
            knowledge="the delivery-date predicate belongs in the self LEFT JOIN ON clause.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List regions with no unresolved support tickets, sorted by region.",
            sql=(
                "SELECT DISTINCT T1.region FROM customers AS T1 LEFT JOIN ("
                "SELECT DISTINCT T2.region FROM customers AS T2 INNER JOIN support_tickets AS T3 "
                "ON T2.customer_id = T3.customer_id WHERE T3.resolved = 0"
                ") AS T4 ON T1.region = T4.region WHERE T4.region IS NULL ORDER BY T1.region"
            ),
            knowledge="anti-join regions against the set of regions that have unresolved support tickets.",
            tags=("single_db_lab", "targeted_exp061", "contrastive", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
    ]


def _challenge_v2_tasks() -> list[Task]:
    return [
        Task(
            question="List Bronze customers with resolved support tickets, sorted by customer name.",
            sql=(
                "SELECT DISTINCT T1.customer_name FROM customers AS T1 INNER JOIN support_tickets AS T2 "
                "ON T1.customer_id = T2.customer_id WHERE T1.loyalty_tier = 'Bronze' AND T2.resolved = 1 "
                "ORDER BY T1.customer_name"
            ),
            knowledge="loyalty_tier belongs to customers; resolved belongs to support_tickets.",
            tags=("single_db_lab", "challenge_v2", "alias_ownership", "support"),
            order_sensitive=True,
        ),
        Task(
            question="What is the total refund amount for returned Apparel products?",
            sql=(
                "SELECT ROUND(SUM(T1.refund_amount), 2) FROM returns AS T1 "
                "INNER JOIN products AS T2 ON T1.product_id = T2.product_id WHERE T2.category = 'Apparel'"
            ),
            knowledge="refund_amount belongs to returns; category belongs to products.",
            tags=("single_db_lab", "challenge_v2", "alias_ownership", "return_product_join"),
        ),
        Task(
            question="How many completed orders used USPS shipments and came from Silver customers?",
            sql=(
                "SELECT COUNT(DISTINCT T2.order_id) FROM customers AS T1 INNER JOIN orders AS T2 "
                "ON T1.customer_id = T2.customer_id INNER JOIN shipments AS T3 ON T2.order_id = T3.order_id "
                "WHERE T1.loyalty_tier = 'Silver' AND T2.status = 'completed' AND T3.carrier = 'USPS'"
            ),
            knowledge="loyalty_tier belongs to customers; status belongs to orders; carrier belongs to shipments.",
            tags=("single_db_lab", "challenge_v2", "alias_ownership", "multi_join"),
        ),
        Task(
            question="Which active Bags products were sold in completed marketplace orders, sorted by product name?",
            sql=(
                "SELECT DISTINCT T1.product_name FROM products AS T1 INNER JOIN order_items AS T2 "
                "ON T1.product_id = T2.product_id INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
                "WHERE T1.active = 1 AND T1.category = 'Bags' AND T3.channel = 'marketplace' "
                "AND T3.status = 'completed' ORDER BY T1.product_name"
            ),
            knowledge="active and category belong to products; channel and status belong to orders.",
            tags=("single_db_lab", "challenge_v2", "alias_ownership", "product_order_join"),
            order_sensitive=True,
        ),
        Task(
            question="How many completed orders were placed strictly after 2024-03-01?",
            sql="SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date > '2024-03-01'",
            knowledge="strictly after excludes the boundary date.",
            tags=("single_db_lab", "challenge_v2", "boundary_semantics", "date_filter"),
        ),
        Task(
            question="How many completed orders were placed on or after 2024-03-01?",
            sql="SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date >= '2024-03-01'",
            knowledge="on or after includes the boundary date.",
            tags=("single_db_lab", "challenge_v2", "boundary_semantics", "date_filter"),
        ),
        Task(
            question="List orders delivered in at least 4 days, sorted by order id.",
            sql=(
                "SELECT order_id FROM shipments WHERE julianday(delivery_date) - julianday(shipped_date) >= 4 "
                "ORDER BY order_id"
            ),
            knowledge="at least 4 days means duration >= 4.",
            tags=("single_db_lab", "challenge_v2", "boundary_semantics", "date_math"),
            order_sensitive=True,
        ),
        Task(
            question="List orders delivered in fewer than 4 days, sorted by order id.",
            sql=(
                "SELECT order_id FROM shipments WHERE julianday(delivery_date) - julianday(shipped_date) < 4 "
                "ORDER BY order_id"
            ),
            knowledge="fewer than 4 days means duration < 4.",
            tags=("single_db_lab", "challenge_v2", "boundary_semantics", "date_math"),
            order_sensitive=True,
        ),
        Task(
            question="How many products have completed sales quantity of at most 4 units?",
            sql=(
                "SELECT COUNT(*) FROM (SELECT T1.product_id, SUM(T1.quantity) AS units FROM order_items AS T1 "
                "INNER JOIN orders AS T2 ON T1.order_id = T2.order_id WHERE T2.status = 'completed' "
                "GROUP BY T1.product_id HAVING units <= 4)"
            ),
            knowledge="at most 4 means HAVING units <= 4.",
            tags=("single_db_lab", "challenge_v2", "boundary_semantics", "having"),
        ),
        Task(
            question="What completed order revenue was booked through March 2024?",
            sql=(
                "SELECT ROUND(SUM(T1.quantity * T1.unit_price * (1 - T2.discount_pct / 100.0)), 2) "
                "FROM order_items AS T1 INNER JOIN orders AS T2 ON T1.order_id = T2.order_id "
                "WHERE T2.status = 'completed' AND T2.order_date <= '2024-03-31'"
            ),
            knowledge="through March 2024 means order_date <= 2024-03-31.",
            tags=("single_db_lab", "challenge_v2", "boundary_semantics", "revenue"),
        ),
        Task(
            question="List customers with no unresolved return tickets, sorted by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 LEFT JOIN support_tickets AS T2 "
                "ON T1.customer_id = T2.customer_id AND T2.resolved = 0 AND T2.issue_type = 'return' "
                "WHERE T2.ticket_id IS NULL ORDER BY T1.customer_name"
            ),
            knowledge="unresolved return predicates belong in the support_tickets LEFT JOIN ON clause.",
            tags=("single_db_lab", "challenge_v2", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List products with no completed store sales, sorted by product name.",
            sql=(
                "SELECT T1.product_name FROM products AS T1 LEFT JOIN ("
                "SELECT DISTINCT T2.product_id FROM order_items AS T2 INNER JOIN orders AS T3 "
                "ON T2.order_id = T3.order_id WHERE T3.status = 'completed' AND T3.channel = 'store'"
                ") AS T4 ON T1.product_id = T4.product_id WHERE T4.product_id IS NULL ORDER BY T1.product_name"
            ),
            knowledge="anti-join against the set of products with completed store sales.",
            tags=("single_db_lab", "challenge_v2", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="List customers with no completed web orders, sorted by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 LEFT JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
                "AND T2.status = 'completed' AND T2.channel = 'web' WHERE T2.order_id IS NULL ORDER BY T1.customer_name"
            ),
            knowledge="completed web predicates belong in the orders LEFT JOIN ON clause.",
            tags=("single_db_lab", "challenge_v2", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
        Task(
            question="How many active Apparel products had no returns?",
            sql=(
                "SELECT COUNT(*) FROM products AS T1 LEFT JOIN returns AS T2 ON T1.product_id = T2.product_id "
                "WHERE T1.active = 1 AND T1.category = 'Apparel' AND T2.return_id IS NULL"
            ),
            knowledge="active and category are on the preserved products table; NULL check is on returns.",
            tags=("single_db_lab", "challenge_v2", "anti_join", "left_join_predicate"),
        ),
        Task(
            question="List customers with no support tickets tied to completed orders, sorted by customer name.",
            sql=(
                "SELECT T1.customer_name FROM customers AS T1 LEFT JOIN ("
                "SELECT DISTINCT T2.customer_id FROM orders AS T2 INNER JOIN support_tickets AS T3 "
                "ON T2.order_id = T3.order_id WHERE T2.status = 'completed'"
                ") AS T4 ON T1.customer_id = T4.customer_id WHERE T4.customer_id IS NULL ORDER BY T1.customer_name"
            ),
            knowledge="anti-join against the set of customers with support tickets tied to completed orders.",
            tags=("single_db_lab", "challenge_v2", "anti_join", "left_join_predicate"),
            order_sensitive=True,
        ),
    ]


def _dev_v2_tasks() -> list[Task]:
    return [
        _unresolved_ticket_count_by_region_dev("Northeast"),
        _completed_orders_after_date_dev("2024-04-15"),
        _completed_orders_in_month_dev("2024-04"),
        _return_item_share_for_tier_dev("Bronze"),
        _return_item_share_for_region_dev("Midwest"),
        _repeat_customer_count_min_orders_dev(4),
        _repeat_customer_names_min_orders_dev(4),
        _products_with_no_completed_sales_in_region_dev("Midwest"),
        _products_with_no_completed_sales_before_date_dev("2024-04-01"),
        _top_product_units_after_date_dev("2024-04-15"),
        _ticket_count_by_issue_without_resolution_dev("product"),
        _active_products_no_completed_sales_after_date_dev("2024-04-01"),
    ]


def _challenge_tasks() -> list[Task]:
    return [
        _unresolved_ticket_count_by_region_challenge("Northeast"),
        _unresolved_ticket_customers_for_tier_challenge("Bronze"),
        _ticket_customers_by_issue_and_resolution_challenge("return", 1),
        _ticket_count_by_issue_without_resolution_challenge("product"),
        _completed_orders_after_date_challenge("2024-04-15"),
        _completed_orders_on_or_after_date_challenge("2024-04-15"),
        _completed_orders_in_month_challenge("2024-04"),
        _revenue_strictly_after_date_challenge("2024-05-01"),
        _returns_in_month_challenge("2024-03"),
        _return_item_share_for_region_challenge("Northeast"),
        _return_item_share_for_tier_challenge("Bronze"),
        _return_order_share_for_channel_challenge("marketplace"),
        _return_item_share_for_channel_and_category_challenge("store", "Footwear"),
        _repeat_customer_count_min_orders_challenge(4),
        _repeat_customer_names_min_orders_challenge(4),
        _products_with_more_than_completed_units_challenge(5),
        _regions_with_more_than_completed_orders_challenge(5),
        _products_with_no_completed_sales_in_region_challenge("Northeast"),
        _products_with_no_completed_sales_before_date_challenge("2024-04-01"),
        _products_with_no_completed_sales_after_date_challenge("2024-04-01"),
        _carrier_delivery_rank_for_region_challenge("South"),
        _refund_total_for_slow_shipments_by_reason_challenge("damaged", 4),
        _active_product_no_completed_sales_count_by_channel_challenge("marketplace"),
        _top_completed_order_product_by_region_challenge("West"),
    ]


def _with_tags(task: Task, *extra_tags: str) -> Task:
    return Task(
        question=task.question,
        sql=task.sql,
        knowledge=task.knowledge,
        tags=task.tags + extra_tags,
        order_sensitive=task.order_sensitive,
        numeric_tolerance=task.numeric_tolerance,
    )


def _unresolved_ticket_count_all() -> Task:
    return Task(
        question="How many support tickets are currently unresolved?",
        sql="SELECT COUNT(*) FROM support_tickets WHERE resolved = 0",
        knowledge="unresolved means support_tickets.resolved = 0; issue_type is not required unless the question asks for a specific type.",
        tags=("single_db_lab", "targeted_exp051", "support", "no_issue_filter"),
    )


def _unresolved_ticket_count_by_region(region: str) -> Task:
    return Task(
        question=f"How many unresolved tickets were opened by {region} customers?",
        sql=(
            "SELECT COUNT(*) FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.region = '{region}' AND T2.resolved = 0"
        ),
        knowledge="for all unresolved tickets, filter on resolved = 0 and do not add an issue_type predicate.",
        tags=("single_db_lab", "targeted_exp051", "support", "no_issue_filter", "join_path"),
    )


def _unresolved_ticket_count_by_tier(tier: str) -> Task:
    return Task(
        question=f"How many unresolved tickets came from {tier} loyalty customers?",
        sql=(
            "SELECT COUNT(*) FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.loyalty_tier = '{tier}' AND T2.resolved = 0"
        ),
        knowledge="loyalty_tier is on customers; unresolved uses support_tickets.resolved = 0.",
        tags=("single_db_lab", "targeted_exp051", "support", "no_issue_filter", "join_path"),
    )


def _unresolved_ticket_customers_for_region(region: str) -> Task:
    return Task(
        question=f"List {region} customers who have unresolved tickets, sorted by customer name.",
        sql=(
            "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.region = '{region}' AND T2.resolved = 0 ORDER BY T1.customer_name ASC"
        ),
        knowledge="issue_type is not part of the filter unless the question names a ticket type.",
        tags=("single_db_lab", "targeted_exp051", "support", "no_issue_filter", "order_by"),
        order_sensitive=True,
    )


def _unresolved_ticket_customers_for_tier(tier: str) -> Task:
    return Task(
        question=f"List {tier} loyalty customers with unresolved tickets, sorted by customer name.",
        sql=(
            "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T1.loyalty_tier = '{tier}' AND T2.resolved = 0 ORDER BY T1.customer_name ASC"
        ),
        knowledge="unresolved means resolved = 0; this question does not restrict issue_type.",
        tags=("single_db_lab", "targeted_exp051", "support", "no_issue_filter", "order_by"),
        order_sensitive=True,
    )


def _ticket_customers_by_issue_and_resolution(issue_type: str, resolved: int) -> Task:
    resolution = "resolved" if resolved else "unresolved"
    return Task(
        question=f"List customers with {resolution} {issue_type} tickets sorted by customer name.",
        sql=(
            "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            f"WHERE T2.issue_type = '{issue_type}' AND T2.resolved = {resolved} ORDER BY T1.customer_name ASC"
        ),
        knowledge="issue_type is used only when a specific ticket type is named.",
        tags=("single_db_lab", "targeted_exp051", "support", "issue_filter", "order_by"),
        order_sensitive=True,
    )


def _ticket_count_by_issue_without_resolution(issue_type: str) -> Task:
    return Task(
        question=f"How many support tickets have issue type {issue_type}?",
        sql=f"SELECT COUNT(*) FROM support_tickets WHERE issue_type = '{issue_type}'",
        knowledge="when the question asks only for issue_type, do not add a resolved predicate.",
        tags=("single_db_lab", "targeted_exp051", "support", "issue_filter"),
    )


def _resolved_ticket_customers_any_issue() -> Task:
    return Task(
        question="List customers with resolved support tickets sorted by customer name.",
        sql=(
            "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            "WHERE T2.resolved = 1 ORDER BY T1.customer_name ASC"
        ),
        knowledge="resolved means support_tickets.resolved = 1; issue_type is not required unless named.",
        tags=("single_db_lab", "targeted_exp051", "support", "no_issue_filter", "order_by"),
        order_sensitive=True,
    )


def _ticket_count_with_no_order_link() -> Task:
    return Task(
        question="How many support tickets are not linked to an order?",
        sql="SELECT COUNT(*) FROM support_tickets WHERE order_id IS NULL",
        knowledge="support_tickets.order_id can be NULL.",
        tags=("single_db_lab", "targeted_exp051", "support", "nullable"),
    )


def _completed_orders_after_date(date: str) -> Task:
    return Task(
        question=f"How many completed orders were placed after {date}?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date > '{date}'",
        knowledge="after a date means a strict greater-than comparison.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "strict_after"),
    )


def _completed_orders_on_or_after_date(date: str) -> Task:
    return Task(
        question=f"How many completed orders were placed on or after {date}?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date >= '{date}'",
        knowledge="on or after a date means a greater-than-or-equal comparison.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "inclusive_after"),
    )


def _completed_orders_before_date(date: str) -> Task:
    return Task(
        question=f"How many completed orders were placed before {date}?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date < '{date}'",
        knowledge="before a date means a strict less-than comparison.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "strict_before"),
    )


def _completed_orders_through_date(date: str) -> Task:
    return Task(
        question=f"How many completed orders were placed through {date}?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date <= '{date}'",
        knowledge="through a date means an inclusive less-than-or-equal comparison.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "inclusive_before"),
    )


def _completed_orders_in_month(month: str) -> Task:
    return Task(
        question=f"How many completed orders were placed during {month}?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = 'completed' AND substr(order_date, 1, 7) = '{month}'",
        knowledge="during a YYYY-MM month uses substr(order_date, 1, 7).",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "month_filter"),
    )


def _completed_orders_in_date_range(start: str, end: str) -> Task:
    return Task(
        question=f"How many completed orders were placed from {start} through {end}?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date BETWEEN '{start}' AND '{end}'",
        knowledge="from X through Y is an inclusive BETWEEN date range.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "date_range"),
    )


def _top_product_units_after_date(date: str) -> Task:
    return Task(
        question=f"After {date}, which product had the highest completed-order unit count?",
        sql=(
            "SELECT T1.product_name, SUM(T2.quantity) AS units FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T3.status = 'completed' AND T3.order_date > '{date}' "
            "GROUP BY T1.product_id, T1.product_name ORDER BY units DESC, T1.product_name ASC LIMIT 1"
        ),
        knowledge="after a date is strict; units are on order_items.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "ranking"),
        order_sensitive=True,
    )


def _top_product_units_on_or_after_date(date: str) -> Task:
    return Task(
        question=f"On or after {date}, which product had the highest completed-order unit count?",
        sql=(
            "SELECT T1.product_name, SUM(T2.quantity) AS units FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T3.status = 'completed' AND T3.order_date >= '{date}' "
            "GROUP BY T1.product_id, T1.product_name ORDER BY units DESC, T1.product_name ASC LIMIT 1"
        ),
        knowledge="on or after a date is inclusive; units are on order_items.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "ranking"),
        order_sensitive=True,
    )


def _returns_after_date(date: str) -> Task:
    return Task(
        question=f"How many returns were recorded after {date}?",
        sql=f"SELECT COUNT(*) FROM returns WHERE return_date > '{date}'",
        knowledge="return_date is stored on returns; after is strict.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "returns"),
    )


def _returns_in_month(month: str) -> Task:
    return Task(
        question=f"How many returns were recorded during {month}?",
        sql=f"SELECT COUNT(*) FROM returns WHERE substr(return_date, 1, 7) = '{month}'",
        knowledge="month filtering uses substr(return_date, 1, 7).",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "returns"),
    )


def _shipment_count_after_ship_date(date: str) -> Task:
    return Task(
        question=f"How many shipments were shipped after {date}?",
        sql=f"SELECT COUNT(*) FROM shipments WHERE shipped_date > '{date}'",
        knowledge="shipped_date is stored on shipments; after is strict.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "shipment"),
    )


def _shipment_count_delivered_through(date: str) -> Task:
    return Task(
        question=f"How many shipments were delivered through {date}?",
        sql=f"SELECT COUNT(*) FROM shipments WHERE delivery_date <= '{date}'",
        knowledge="through a date is inclusive.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "shipment"),
    )


def _revenue_strictly_after_date(date: str) -> Task:
    return Task(
        question=f"What discounted completed-order revenue was generated after {date}?",
        sql=(
            "SELECT ROUND(SUM(T2.quantity * T2.unit_price * (1 - T1.discount_pct / 100.0)), 2) "
            "FROM orders AS T1 INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.status = 'completed' AND T1.order_date > '{date}'"
        ),
        knowledge="after a date is strict greater-than; revenue uses order_items and orders.discount_pct.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "revenue"),
    )


def _revenue_on_or_after_date(date: str) -> Task:
    return Task(
        question=f"What discounted completed-order revenue was generated on or after {date}?",
        sql=(
            "SELECT ROUND(SUM(T2.quantity * T2.unit_price * (1 - T1.discount_pct / 100.0)), 2) "
            "FROM orders AS T1 INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.status = 'completed' AND T1.order_date >= '{date}'"
        ),
        knowledge="on or after a date is inclusive greater-than-or-equal.",
        tags=("single_db_lab", "targeted_exp052", "date_semantics", "revenue"),
    )


def _return_item_share_for_product(product_name: str) -> Task:
    return Task(
        question=f"What share of completed-order item rows for {product_name} had a matching return row?",
        sql=(
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T2.item_id), 3) "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T2.order_id = T4.order_id AND T2.product_id = T4.product_id "
            f"WHERE T1.product_name = '{product_name}' AND T3.status = 'completed'"
        ),
        knowledge="the numerator counts matching returns; the denominator counts completed-order item rows.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "alias_ownership"),
    )


def _return_item_share_for_region(region: str) -> Task:
    return Task(
        question=f"What share of completed-order item rows from {region} customers had a matching return row?",
        sql=(
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T3.item_id), 3) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T3.order_id = T4.order_id AND T3.product_id = T4.product_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed'"
        ),
        knowledge="item-row return share divides returned item rows by completed-order item rows.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "join_path"),
    )


def _return_item_share_for_tier(tier: str) -> Task:
    return Task(
        question=f"What share of completed-order item rows from {tier} loyalty customers had a matching return row?",
        sql=(
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T3.item_id), 3) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN order_items AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T3.order_id = T4.order_id AND T3.product_id = T4.product_id "
            f"WHERE T1.loyalty_tier = '{tier}' AND T2.status = 'completed'"
        ),
        knowledge="loyalty tier filters customers; return share is item-row based.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "join_path"),
    )


def _return_order_share_for_category(category: str) -> Task:
    return Task(
        question=f"What share of completed orders containing {category} products had at least one return?",
        sql=(
            "SELECT ROUND(CAST(COUNT(DISTINCT T4.order_id) AS REAL) / COUNT(DISTINCT T3.order_id), 3) "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T3.order_id = T4.order_id "
            f"WHERE T1.category = '{category}' AND T3.status = 'completed'"
        ),
        knowledge="this is order-level return share, not item-row return share.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "order_denominator"),
    )


def _return_order_share_for_region(region: str) -> Task:
    return Task(
        question=f"What share of completed orders from {region} customers had at least one return?",
        sql=(
            "SELECT ROUND(CAST(COUNT(DISTINCT T3.order_id) AS REAL) / COUNT(DISTINCT T2.order_id), 3) "
            "FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "LEFT JOIN returns AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}' AND T2.status = 'completed'"
        ),
        knowledge="order return share uses distinct returned orders over distinct completed orders.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "order_denominator"),
    )


def _return_order_share_for_channel(channel: str) -> Task:
    return Task(
        question=f"What share of completed {channel} orders had at least one return?",
        sql=(
            "SELECT ROUND(CAST(COUNT(DISTINCT T2.order_id) AS REAL) / COUNT(DISTINCT T1.order_id), 3) "
            "FROM orders AS T1 LEFT JOIN returns AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.channel = '{channel}' AND T1.status = 'completed'"
        ),
        knowledge="order return share counts distinct returned orders divided by distinct completed orders.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "order_denominator"),
    )


def _return_item_share_for_channel_and_category(channel: str, category: str) -> Task:
    return Task(
        question=f"What share of completed-order {category} item rows from the {channel} channel had a matching return row?",
        sql=(
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T2.item_id), 3) "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T2.order_id = T4.order_id AND T2.product_id = T4.product_id "
            f"WHERE T1.category = '{category}' AND T3.channel = '{channel}' AND T3.status = 'completed'"
        ),
        knowledge="item-row return share uses matching returns over completed-order item rows.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "alias_ownership"),
    )


def _returned_item_row_count_for_category(category: str) -> Task:
    return Task(
        question=f"How many completed-order item rows in {category} had a matching return row?",
        sql=(
            "SELECT COUNT(T4.return_id) FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T2.order_id = T4.order_id AND T2.product_id = T4.product_id "
            f"WHERE T1.category = '{category}' AND T3.status = 'completed'"
        ),
        knowledge="count return_id from the returns alias, not the orders alias.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "alias_ownership"),
    )


def _completed_item_row_count_for_category(category: str) -> Task:
    return Task(
        question=f"How many completed-order item rows are in the {category} category?",
        sql=(
            "SELECT COUNT(T2.item_id) FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.category = '{category}' AND T3.status = 'completed'"
        ),
        knowledge="completed-order item row denominator counts order_items.item_id.",
        tags=("single_db_lab", "targeted_exp053", "return_ratio", "denominator"),
    )


def _repeat_customer_count_min_orders(min_orders: int) -> Task:
    return Task(
        question=f"How many customers placed at least {min_orders} completed orders?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT customer_id FROM orders "
            "WHERE status = 'completed' GROUP BY customer_id "
            f"HAVING COUNT(order_id) >= {min_orders})"
        ),
        knowledge="count customers after grouping completed orders by customer.",
        tags=("single_db_lab", "targeted_exp054", "having", "grouped_count"),
    )


def _repeat_customer_names_min_orders(min_orders: int) -> Task:
    return Task(
        question=f"List customers who placed at least {min_orders} completed orders, sorted by customer name.",
        sql=(
            "SELECT T1.customer_name FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "WHERE T2.status = 'completed' GROUP BY T1.customer_id, T1.customer_name "
            f"HAVING COUNT(T2.order_id) >= {min_orders} ORDER BY T1.customer_name ASC"
        ),
        knowledge="repeat-customer lists require GROUP BY and HAVING after joining customers to orders.",
        tags=("single_db_lab", "targeted_exp054", "having", "order_by"),
        order_sensitive=True,
    )


def _products_with_more_than_completed_units(min_units: int) -> Task:
    return Task(
        question=f"How many products sold more than {min_units} units across completed orders?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.product_id FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "WHERE T3.status = 'completed' GROUP BY T1.product_id "
            f"HAVING SUM(T2.quantity) > {min_units})"
        ),
        knowledge="product unit thresholds require grouping completed-order item quantities by product.",
        tags=("single_db_lab", "targeted_exp054", "having", "products"),
    )


def _categories_with_more_than_completed_units(min_units: int) -> Task:
    return Task(
        question=f"List categories with more than {min_units} completed-order units, sorted alphabetically.",
        sql=(
            "SELECT T1.category FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "WHERE T3.status = 'completed' GROUP BY T1.category "
            f"HAVING SUM(T2.quantity) > {min_units} ORDER BY T1.category ASC"
        ),
        knowledge="category unit thresholds require GROUP BY category and HAVING over SUM(quantity).",
        tags=("single_db_lab", "targeted_exp054", "having", "category"),
        order_sensitive=True,
    )


def _customers_with_more_than_return_count(min_returns: int) -> Task:
    return Task(
        question=f"How many customers have more than {min_returns} return rows?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.customer_id FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN returns AS T3 ON T2.order_id = T3.order_id "
            f"GROUP BY T1.customer_id HAVING COUNT(T3.return_id) > {min_returns})"
        ),
        knowledge="return-count thresholds require grouping return rows by customer.",
        tags=("single_db_lab", "targeted_exp054", "having", "returns"),
    )


def _orders_with_at_least_distinct_products(min_products: int) -> Task:
    return Task(
        question=f"How many completed orders contain at least {min_products} distinct products?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT T1.order_id FROM orders AS T1 "
            "INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            "WHERE T1.status = 'completed' GROUP BY T1.order_id "
            f"HAVING COUNT(DISTINCT T2.product_id) >= {min_products})"
        ),
        knowledge="distinct product thresholds require grouping order_items by completed order.",
        tags=("single_db_lab", "targeted_exp054", "having", "order_items"),
    )


def _channels_with_more_than_completed_orders(min_orders: int) -> Task:
    return Task(
        question=f"List channels with more than {min_orders} completed orders, sorted alphabetically.",
        sql=(
            "SELECT channel FROM orders WHERE status = 'completed' GROUP BY channel "
            f"HAVING COUNT(order_id) > {min_orders} ORDER BY channel ASC"
        ),
        knowledge="channel is on orders and completed-order thresholds use HAVING.",
        tags=("single_db_lab", "targeted_exp054", "having", "orders"),
        order_sensitive=True,
    )


def _regions_with_more_than_completed_orders(min_orders: int) -> Task:
    return Task(
        question=f"List customer regions with more than {min_orders} completed orders, sorted alphabetically.",
        sql=(
            "SELECT T1.region FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "WHERE T2.status = 'completed' GROUP BY T1.region "
            f"HAVING COUNT(T2.order_id) > {min_orders} ORDER BY T1.region ASC"
        ),
        knowledge="region comes from customers; completed-order thresholds use GROUP BY and HAVING.",
        tags=("single_db_lab", "targeted_exp054", "having", "join_path"),
        order_sensitive=True,
    )


def _customers_with_no_support_tickets() -> Task:
    return Task(
        question="List customers with no support tickets, sorted by customer name.",
        sql=(
            "SELECT T1.customer_name FROM customers AS T1 "
            "LEFT JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            "WHERE T2.ticket_id IS NULL ORDER BY T1.customer_name ASC"
        ),
        knowledge="customers with no support tickets require a left anti-join.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "support"),
        order_sensitive=True,
    )


def _customers_with_no_unresolved_tickets() -> Task:
    return Task(
        question="List customers with no unresolved support tickets, sorted by customer name.",
        sql=(
            "SELECT T1.customer_name FROM customers AS T1 "
            "LEFT JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id AND T2.resolved = 0 "
            "WHERE T2.ticket_id IS NULL ORDER BY T1.customer_name ASC"
        ),
        knowledge="the unresolved predicate belongs in the left join condition for this anti-join.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "support"),
        order_sensitive=True,
    )


def _orders_with_no_returns() -> Task:
    return Task(
        question="List order IDs with no returns, sorted ascending.",
        sql=(
            "SELECT T1.order_id FROM orders AS T1 "
            "LEFT JOIN returns AS T2 ON T1.order_id = T2.order_id "
            "WHERE T2.return_id IS NULL ORDER BY T1.order_id ASC"
        ),
        knowledge="orders with no returns require a left anti-join to returns.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "returns"),
        order_sensitive=True,
    )


def _completed_orders_with_no_returns() -> Task:
    return Task(
        question="How many completed orders have no returns?",
        sql=(
            "SELECT COUNT(*) FROM orders AS T1 "
            "LEFT JOIN returns AS T2 ON T1.order_id = T2.order_id "
            "WHERE T1.status = 'completed' AND T2.return_id IS NULL"
        ),
        knowledge="completed orders with no returns require orders filtered by status and a left anti-join.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "returns"),
    )


def _products_with_no_returns() -> Task:
    return Task(
        question="List products with no returns, sorted by product name.",
        sql=(
            "SELECT T1.product_name FROM products AS T1 "
            "LEFT JOIN returns AS T2 ON T1.product_id = T2.product_id "
            "WHERE T2.return_id IS NULL ORDER BY T1.product_name ASC"
        ),
        knowledge="products with no returns require a left anti-join to returns.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "returns"),
        order_sensitive=True,
    )


def _active_products_with_no_returns() -> Task:
    return Task(
        question="How many active products have no returns?",
        sql=(
            "SELECT COUNT(*) FROM products AS T1 "
            "LEFT JOIN returns AS T2 ON T1.product_id = T2.product_id "
            "WHERE T1.active = 1 AND T2.return_id IS NULL"
        ),
        knowledge="active is on products; no returns requires a left anti-join.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "returns"),
    )


def _customers_with_completed_orders_no_returns() -> Task:
    return Task(
        question="How many customers have completed orders but no returns on those completed orders?",
        sql=(
            "SELECT COUNT(DISTINCT T1.customer_id) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "LEFT JOIN returns AS T3 ON T2.order_id = T3.order_id "
            "WHERE T2.status = 'completed' AND T3.return_id IS NULL"
        ),
        knowledge="this anti-join is scoped to completed orders.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "returns"),
    )


def _products_with_no_completed_sales_in_region(region: str) -> Task:
    return Task(
        question=f"List active products with no completed-order sales from {region} customers, sorted by product name.",
        sql=(
            "SELECT T1.product_name FROM products AS T1 "
            "LEFT JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "LEFT JOIN orders AS T3 ON T2.order_id = T3.order_id AND T3.status = 'completed' "
            f"LEFT JOIN customers AS T4 ON T3.customer_id = T4.customer_id AND T4.region = '{region}' "
            "WHERE T1.active = 1 GROUP BY T1.product_id, T1.product_name "
            "HAVING COUNT(T4.customer_id) = 0 ORDER BY T1.product_name ASC"
        ),
        knowledge="the region-scoped no-sales query uses left joins and HAVING after grouping by product.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "products"),
        order_sensitive=True,
    )


def _customers_with_no_completed_orders() -> Task:
    return Task(
        question="List customers with no completed orders, sorted by customer name.",
        sql=(
            "SELECT T1.customer_name FROM customers AS T1 "
            "LEFT JOIN orders AS T2 ON T1.customer_id = T2.customer_id AND T2.status = 'completed' "
            "WHERE T2.order_id IS NULL ORDER BY T1.customer_name ASC"
        ),
        knowledge="the completed status predicate belongs in the left join condition.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "orders"),
        order_sensitive=True,
    )


def _customers_with_no_marketplace_completed_orders() -> Task:
    return Task(
        question="List customers with no completed marketplace orders, sorted by customer name.",
        sql=(
            "SELECT T1.customer_name FROM customers AS T1 "
            "LEFT JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "AND T2.status = 'completed' AND T2.channel = 'marketplace' "
            "WHERE T2.order_id IS NULL ORDER BY T1.customer_name ASC"
        ),
        knowledge="channel and status predicates belong in the left join condition for this anti-join.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "orders"),
        order_sensitive=True,
    )


def _orders_with_no_support_ticket() -> Task:
    return Task(
        question="List completed order IDs with no support ticket, sorted ascending.",
        sql=(
            "SELECT T1.order_id FROM orders AS T1 "
            "LEFT JOIN support_tickets AS T2 ON T1.order_id = T2.order_id "
            "WHERE T1.status = 'completed' AND T2.ticket_id IS NULL ORDER BY T1.order_id ASC"
        ),
        knowledge="orders with no support ticket require a left anti-join to support_tickets.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "support"),
        order_sensitive=True,
    )


def _products_with_no_completed_sales_after_date(date: str) -> Task:
    return Task(
        question=f"List active products with no completed-order sales after {date}, sorted by product name.",
        sql=(
            "SELECT T1.product_name FROM products AS T1 "
            "LEFT JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "LEFT JOIN orders AS T3 ON T2.order_id = T3.order_id AND T3.status = 'completed' "
            f"AND T3.order_date > '{date}' "
            "WHERE T1.active = 1 GROUP BY T1.product_id, T1.product_name "
            "HAVING COUNT(T3.order_id) = 0 ORDER BY T1.product_name ASC"
        ),
        knowledge="date and completed status predicates belong in the left join condition.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "date_semantics"),
        order_sensitive=True,
    )


def _products_with_no_completed_sales_before_date(date: str) -> Task:
    return Task(
        question=f"List active products with no completed-order sales before {date}, sorted by product name.",
        sql=(
            "SELECT T1.product_name FROM products AS T1 "
            "LEFT JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "LEFT JOIN orders AS T3 ON T2.order_id = T3.order_id AND T3.status = 'completed' "
            f"AND T3.order_date < '{date}' "
            "WHERE T1.active = 1 GROUP BY T1.product_id, T1.product_name "
            "HAVING COUNT(T3.order_id) = 0 ORDER BY T1.product_name ASC"
        ),
        knowledge="before a date is strict; predicates belong in the left join condition.",
        tags=("single_db_lab", "targeted_exp055", "anti_join", "date_semantics"),
        order_sensitive=True,
    )


def _challenge(task: Task) -> Task:
    return _with_tags(task, "challenge_v1")


def _dev2(task: Task) -> Task:
    return _with_tags(task, "dev_v2")


def _unresolved_ticket_count_all_dev() -> Task:
    return _dev2(_unresolved_ticket_count_all())


def _unresolved_ticket_count_by_region_dev(region: str) -> Task:
    return _dev2(_unresolved_ticket_count_by_region(region))


def _completed_orders_after_date_dev(date: str) -> Task:
    return _dev2(_completed_orders_after_date(date))


def _completed_orders_in_month_dev(month: str) -> Task:
    return _dev2(_completed_orders_in_month(month))


def _return_item_share_for_tier_dev(tier: str) -> Task:
    return _dev2(_return_item_share_for_tier(tier))


def _return_item_share_for_region_dev(region: str) -> Task:
    return _dev2(_return_item_share_for_region(region))


def _repeat_customer_count_min_orders_dev(min_orders: int) -> Task:
    return _dev2(_repeat_customer_count_min_orders(min_orders))


def _repeat_customer_names_min_orders_dev(min_orders: int) -> Task:
    return _dev2(_repeat_customer_names_min_orders(min_orders))


def _orders_with_no_returns_dev() -> Task:
    return _dev2(_orders_with_no_returns())


def _products_with_no_returns_dev() -> Task:
    return _dev2(_products_with_no_returns())


def _top_product_units_after_date_dev(date: str) -> Task:
    return _dev2(_top_product_units_after_date(date))


def _support_ticket_customers_no_issue_filter_dev() -> Task:
    return _dev2(_resolved_ticket_customers_any_issue())


def _ticket_count_by_issue_without_resolution_dev(issue_type: str) -> Task:
    return _dev2(_ticket_count_by_issue_without_resolution(issue_type))


def _products_with_no_completed_sales_in_region_dev(region: str) -> Task:
    return _dev2(_products_with_no_completed_sales_in_region(region))


def _products_with_no_completed_sales_before_date_dev(date: str) -> Task:
    return _dev2(_products_with_no_completed_sales_before_date(date))


def _active_products_no_completed_sales_after_date_dev(date: str) -> Task:
    return _dev2(_products_with_no_completed_sales_after_date(date))


def _unresolved_ticket_count_all_challenge() -> Task:
    return _challenge(_unresolved_ticket_count_all())


def _unresolved_ticket_count_by_region_challenge(region: str) -> Task:
    return _challenge(_unresolved_ticket_count_by_region(region))


def _unresolved_ticket_customers_for_tier_challenge(tier: str) -> Task:
    return _challenge(_unresolved_ticket_customers_for_tier(tier))


def _ticket_customers_by_issue_and_resolution_challenge(issue_type: str, resolved: int) -> Task:
    return _challenge(_ticket_customers_by_issue_and_resolution(issue_type, resolved))


def _ticket_count_with_no_order_link_challenge() -> Task:
    return _challenge(_ticket_count_with_no_order_link())


def _ticket_count_by_issue_without_resolution_challenge(issue_type: str) -> Task:
    return _challenge(_ticket_count_by_issue_without_resolution(issue_type))


def _completed_orders_after_date_challenge(date: str) -> Task:
    return _challenge(_completed_orders_after_date(date))


def _completed_orders_on_or_after_date_challenge(date: str) -> Task:
    return _challenge(_completed_orders_on_or_after_date(date))


def _completed_orders_in_month_challenge(month: str) -> Task:
    return _challenge(_completed_orders_in_month(month))


def _revenue_strictly_after_date_challenge(date: str) -> Task:
    return _challenge(_revenue_strictly_after_date(date))


def _returns_in_month_challenge(month: str) -> Task:
    return _challenge(_returns_in_month(month))


def _return_item_share_for_region_challenge(region: str) -> Task:
    return _challenge(_return_item_share_for_region(region))


def _return_item_share_for_tier_challenge(tier: str) -> Task:
    return _challenge(_return_item_share_for_tier(tier))


def _return_order_share_for_channel_challenge(channel: str) -> Task:
    return _challenge(_return_order_share_for_channel(channel))


def _return_item_share_for_channel_and_category_challenge(channel: str, category: str) -> Task:
    return _challenge(_return_item_share_for_channel_and_category(channel, category))


def _repeat_customer_count_min_orders_challenge(min_orders: int) -> Task:
    return _challenge(_repeat_customer_count_min_orders(min_orders))


def _repeat_customer_names_min_orders_challenge(min_orders: int) -> Task:
    return _challenge(_repeat_customer_names_min_orders(min_orders))


def _products_with_more_than_completed_units_challenge(min_units: int) -> Task:
    return _challenge(_products_with_more_than_completed_units(min_units))


def _regions_with_more_than_completed_orders_challenge(min_orders: int) -> Task:
    return _challenge(_regions_with_more_than_completed_orders(min_orders))


def _customers_with_no_support_tickets_challenge() -> Task:
    return _challenge(_customers_with_no_support_tickets())


def _orders_with_no_returns_challenge() -> Task:
    return _challenge(_orders_with_no_returns())


def _products_with_no_returns_challenge() -> Task:
    return _challenge(_products_with_no_returns())


def _products_with_no_completed_sales_in_region_challenge(region: str) -> Task:
    return _challenge(_products_with_no_completed_sales_in_region(region))


def _products_with_no_completed_sales_before_date_challenge(date: str) -> Task:
    return _challenge(_products_with_no_completed_sales_before_date(date))


def _products_with_no_completed_sales_after_date_challenge(date: str) -> Task:
    return _challenge(_products_with_no_completed_sales_after_date(date))


def _carrier_delivery_rank_for_region_challenge(region: str) -> Task:
    return _challenge(_carrier_delivery_rank_for_region(region))


def _refund_total_for_slow_shipments_by_reason_challenge(reason: str, min_days: int) -> Task:
    return _challenge(_refund_total_for_slow_shipments_by_reason(reason, min_days))


def _active_product_no_completed_sales_count_by_channel_challenge(channel: str) -> Task:
    return _challenge(_active_product_no_completed_sales_count_by_channel(channel))


def _top_completed_order_product_by_channel_challenge(channel: str) -> Task:
    return _challenge(_top_completed_order_product_by_channel(channel))


def _top_completed_order_product_by_region(region: str) -> Task:
    return Task(
        question=f"Which product had the most completed-order units for {region} customers?",
        sql=(
            "SELECT T1.product_name, SUM(T2.quantity) AS units FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "INNER JOIN customers AS T4 ON T3.customer_id = T4.customer_id "
            f"WHERE T3.status = 'completed' AND T4.region = '{region}' "
            "GROUP BY T1.product_id, T1.product_name ORDER BY units DESC, T1.product_name ASC LIMIT 1"
        ),
        knowledge="quantity belongs to order_items; region belongs to customers through orders.",
        tags=("single_db_lab", "grouped_ranking", "join_path"),
        order_sensitive=True,
    )


def _top_completed_order_product_by_region_challenge(region: str) -> Task:
    return _challenge(_top_completed_order_product_by_region(region))


def _revenue_after_date(start: str) -> Task:
    return Task(
        question=f"What discounted completed-order revenue was generated on or after {start}?",
        sql=(
            "SELECT ROUND(SUM(T2.quantity * T2.unit_price * (1 - T1.discount_pct / 100.0)), 2) "
            "FROM orders AS T1 INNER JOIN order_items AS T2 ON T1.order_id = T2.order_id "
            f"WHERE T1.status = 'completed' AND T1.order_date >= '{start}'"
        ),
        knowledge="completed-order revenue applies discount_pct to item totals.",
        tags=("single_db_lab", "eval_holdout", "revenue"),
    )


def _category_revenue_rank() -> Task:
    return Task(
        question="Which product category has the highest discounted completed-order revenue?",
        sql=(
            "SELECT T1.category, ROUND(SUM(T2.quantity * T2.unit_price * (1 - T3.discount_pct / 100.0)), 2) AS revenue "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "WHERE T3.status = 'completed' GROUP BY T1.category ORDER BY revenue DESC, T1.category ASC LIMIT 1"
        ),
        knowledge="category revenue applies discount_pct and excludes non-completed orders.",
        tags=("single_db_lab", "eval_holdout", "ranking"),
        order_sensitive=True,
    )


def _return_rate_by_category(category: str) -> Task:
    return Task(
        question=f"What share of completed-order item rows in {category} had a return?",
        sql=(
            "SELECT ROUND(CAST(COUNT(T4.return_id) AS REAL) / COUNT(T2.item_id), 3) "
            "FROM products AS T1 INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            "LEFT JOIN returns AS T4 ON T2.order_id = T4.order_id AND T2.product_id = T4.product_id "
            f"WHERE T1.category = '{category}' AND T3.status = 'completed'"
        ),
        knowledge="return share is returned completed-order item rows divided by completed-order item rows.",
        tags=("single_db_lab", "eval_holdout", "returns"),
    )


def _carrier_slowest_delivery() -> Task:
    return Task(
        question="Which carrier has the slowest average delivery time?",
        sql=(
            "SELECT carrier, ROUND(AVG(julianday(delivery_date) - julianday(shipped_date)), 2) AS avg_days "
            "FROM shipments GROUP BY carrier ORDER BY avg_days DESC, carrier ASC LIMIT 1"
        ),
        knowledge="delivery time in days is julianday(delivery_date) - julianday(shipped_date).",
        tags=("single_db_lab", "eval_holdout", "date_math"),
        order_sensitive=True,
    )


def _customer_return_refunds(tier: str) -> Task:
    return Task(
        question=f"What total refund amount came from {tier} loyalty customers?",
        sql=(
            "SELECT ROUND(SUM(T4.refund_amount), 2) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN returns AS T4 ON T2.order_id = T4.order_id "
            f"WHERE T1.loyalty_tier = '{tier}'"
        ),
        knowledge="refund amount comes from returns.refund_amount joined through orders to customers.",
        tags=("single_db_lab", "eval_holdout", "returns"),
    )


def _unresolved_ticket_customers() -> Task:
    return Task(
        question="List customers with unresolved support tickets sorted by customer name.",
        sql=(
            "SELECT DISTINCT T1.customer_name FROM customers AS T1 "
            "INNER JOIN support_tickets AS T2 ON T1.customer_id = T2.customer_id "
            "WHERE T2.resolved = 0 ORDER BY T1.customer_name ASC"
        ),
        knowledge="unresolved means support_tickets.resolved = 0.",
        tags=("single_db_lab", "eval_holdout", "support"),
        order_sensitive=True,
    )


def _discounted_order_count(min_discount: float) -> Task:
    return Task(
        question=f"How many completed orders had a discount of at least {min_discount:g} percent?",
        sql=f"SELECT COUNT(*) FROM orders WHERE status = 'completed' AND discount_pct >= {min_discount:g}",
        knowledge="discount_pct is stored as a percent value.",
        tags=("single_db_lab", "eval_holdout", "discount"),
    )


def _active_product_no_sales() -> Task:
    return Task(
        question="Which active products have no completed-order sales?",
        sql=(
            "SELECT T1.product_name FROM products AS T1 "
            "LEFT JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "LEFT JOIN orders AS T3 ON T2.order_id = T3.order_id AND T3.status = 'completed' "
            "WHERE T1.active = 1 GROUP BY T1.product_id, T1.product_name "
            "HAVING COUNT(T3.order_id) = 0 ORDER BY T1.product_name ASC"
        ),
        knowledge="active products have active = 1; completed-order sales require orders.status = 'completed'.",
        tags=("single_db_lab", "eval_holdout", "anti_join"),
        order_sensitive=True,
    )


def _region_refund_total(region: str) -> Task:
    return Task(
        question=f"What refund total is associated with customers in the {region} region?",
        sql=(
            "SELECT ROUND(SUM(T3.refund_amount), 2) FROM customers AS T1 "
            "INNER JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "INNER JOIN returns AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T1.region = '{region}'"
        ),
        knowledge="refund total uses returns.refund_amount joined to customers through orders.",
        tags=("single_db_lab", "eval_holdout", "returns"),
    )


def _highest_quantity_order_product(start: str) -> Task:
    return Task(
        question=f"After {start}, which product had the most completed-order units?",
        sql=(
            "SELECT T1.product_name, SUM(T2.quantity) AS units FROM products AS T1 "
            "INNER JOIN order_items AS T2 ON T1.product_id = T2.product_id "
            "INNER JOIN orders AS T3 ON T2.order_id = T3.order_id "
            f"WHERE T3.status = 'completed' AND T3.order_date > '{start}' "
            "GROUP BY T1.product_id, T1.product_name ORDER BY units DESC, T1.product_name ASC LIMIT 1"
        ),
        knowledge="units means SUM(order_items.quantity) over completed orders after the date.",
        tags=("single_db_lab", "eval_holdout", "ranking"),
        order_sensitive=True,
    )


def _repeat_customer_count() -> Task:
    return Task(
        question="How many customers placed more than one completed order?",
        sql=(
            "SELECT COUNT(*) FROM (SELECT customer_id, COUNT(*) AS completed_orders FROM orders "
            "WHERE status = 'completed' GROUP BY customer_id HAVING completed_orders > 1)"
        ),
        knowledge="repeat customers have more than one completed order.",
        tags=("single_db_lab", "eval_holdout", "having"),
    )


def _late_delivery_refunds() -> Task:
    return Task(
        question="What refund amount is tied to returns whose order shipment took more than 4 days?",
        sql=(
            "SELECT ROUND(SUM(T1.refund_amount), 2) FROM returns AS T1 "
            "INNER JOIN shipments AS T2 ON T1.order_id = T2.order_id "
            "WHERE julianday(T2.delivery_date) - julianday(T2.shipped_date) > 4"
        ),
        knowledge="shipment duration in days is julianday(delivery_date) - julianday(shipped_date).",
        tags=("single_db_lab", "eval_holdout", "date_math"),
    )


def _write_train_jsonl(path: Path, tasks: list[Task], *, split_name: str) -> None:
    rows = [
        _train_row(index=index, task=task, split_name=split_name)
        for index, task in enumerate(tasks, start=1)
    ]
    _write_jsonl(path, rows)


def _write_eval_jsonl(path: Path, tasks: list[Task], *, split_name: str) -> None:
    rows = [
        _eval_row(index=index, task=task, split_name=split_name)
        for index, task in enumerate(tasks, start=1)
    ]
    _write_jsonl(path, rows)


def _train_row(index: int, task: Task, *, split_name: str) -> dict[str, Any]:
    row_id = f"{DB_ID}_{split_name}_{index:03d}"
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
            "created_by": "scripts/create_storefront_sql_lab.py",
            "teacher_model": None,
            "source_path": "scripts/create_storefront_sql_lab.py",
        },
        "tags": list(task.tags),
    }


def _eval_row(index: int, task: Task, *, split_name: str) -> dict[str, Any]:
    case_id = f"{DB_ID}_{split_name}_{index:03d}"
    return {
        "schema_version": "sql_eval_case:v1",
        "case_id": case_id,
        "source_benchmark": "synthetic",
        "source_split": split_name,
        "task_id": case_id,
        "fixture_id": DB_ID,
        "db_id": DB_ID,
        "db_path": str(DB_PATH.relative_to(ROOT)),
        "dialect": "sqlite",
        "question": task.question,
        "schema_text": SCHEMA_TEXT,
        "knowledge_text": task.knowledge,
        "column_value_notes": COLUMN_VALUE_NOTES,
        "gold_sql": task.sql,
        "task_type": "select",
        "order_sensitive": task.order_sensitive,
        "numeric_tolerance": task.numeric_tolerance,
        "tags": list(task.tags),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
