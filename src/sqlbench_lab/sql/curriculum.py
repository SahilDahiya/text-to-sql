"""Versioned curriculum specification for SQL grounding."""

from __future__ import annotations

CURRICULUM_SPEC_VERSION = "sql-curriculum:v1"
CURRICULUM_TIERS = {
    1: "schema_and_identifier_grounding",
    2: "single_table_filter_projection_order",
    3: "joins_and_relationships",
    4: "date_range_and_value_grounding",
    5: "aggregation_grouping_arithmetic_nulls",
    6: "nested_queries_and_composition",
}


def curriculum_tier_name(tier: int) -> str:
    try:
        return CURRICULUM_TIERS[tier]
    except KeyError as exc:
        raise ValueError(f"unsupported curriculum tier: {tier}") from exc
