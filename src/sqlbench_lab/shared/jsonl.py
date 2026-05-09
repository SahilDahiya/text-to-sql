"""JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file as JSON objects with useful parse errors."""

    resolved_path = Path(path)
    rows: list[dict[str, Any]] = []
    with resolved_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{resolved_path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{resolved_path}:{line_number}: JSONL rows must be objects")
            rows.append(payload)
    return rows

