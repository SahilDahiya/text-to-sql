"""Shared SQL training type definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SQLSFTTrainingSummary:
    experiment_id: str
    base_model: str
    adapter_dir: str
    train_row_count: int
    dry_run: bool
    trainable_parameters: int | None = None
    total_parameters: int | None = None
    training_metrics: dict[str, float] | None = None
