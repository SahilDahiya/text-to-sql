"""Immutable promotion comparison for continued LiveSQLBench ISFT."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SQLPromotionDecision:
    decision: str
    target_base_pass_rate: float
    target_candidate_pass_rate: float
    target_delta: float
    guardrails: list[dict[str, Any]]
    reasons: list[str]
    base_result_paths: list[str]
    candidate_result_paths: list[str]


def compare_sql_promotion(
    *,
    base_target: str | Path,
    candidate_target: str | Path,
    base_guardrails: list[str | Path] | tuple[str | Path, ...],
    candidate_guardrails: list[str | Path] | tuple[str | Path, ...],
    target_min_improvement: float,
    max_guardrail_regression: float,
    output_path: str | Path | None = None,
) -> SQLPromotionDecision:
    if target_min_improvement < 0 or max_guardrail_regression < 0:
        raise ValueError("promotion thresholds must be non-negative")
    if len(base_guardrails) != len(candidate_guardrails):
        raise ValueError("base and candidate guardrail counts must match")
    base_target_payload = _read_result(base_target)
    candidate_target_payload = _read_result(candidate_target)
    reasons = _assert_comparable(base_target_payload, candidate_target_payload, label="target")
    target_delta = float(candidate_target_payload["pass_rate"]) - float(base_target_payload["pass_rate"])
    guardrails: list[dict[str, Any]] = []
    for index, (base_path, candidate_path) in enumerate(zip(base_guardrails, candidate_guardrails), start=1):
        base_payload = _read_result(base_path)
        candidate_payload = _read_result(candidate_path)
        reasons.extend(_assert_comparable(base_payload, candidate_payload, label=f"guardrail-{index}"))
        delta = float(candidate_payload["pass_rate"]) - float(base_payload["pass_rate"])
        guardrails.append(
            {
                "index": index,
                "base_path": str(Path(base_path).resolve()),
                "candidate_path": str(Path(candidate_path).resolve()),
                "base_pass_rate": float(base_payload["pass_rate"]),
                "candidate_pass_rate": float(candidate_payload["pass_rate"]),
                "delta": delta,
                "passed": delta >= -max_guardrail_regression,
            }
        )

    if reasons:
        decision = "investigate"
    elif target_delta < target_min_improvement:
        decision = "reject"
        reasons = [f"target improvement {target_delta:.6f} is below {target_min_improvement:.6f}"]
    elif not all(bool(guardrail["passed"]) for guardrail in guardrails):
        decision = "reject"
        reasons = [
            f"guardrail {guardrail['index']} regressed by {guardrail['delta']:.6f} beyond {-max_guardrail_regression:.6f}"
            for guardrail in guardrails
            if not guardrail["passed"]
        ]
    else:
        decision = "promote"
        reasons = ["target gate improved and all frozen guardrails passed"]

    result = SQLPromotionDecision(
        decision=decision,
        target_base_pass_rate=float(base_target_payload["pass_rate"]),
        target_candidate_pass_rate=float(candidate_target_payload["pass_rate"]),
        target_delta=target_delta,
        guardrails=guardrails,
        reasons=reasons,
        base_result_paths=[str(Path(base_target).resolve()), *[str(Path(path).resolve()) for path in base_guardrails]],
        candidate_result_paths=[str(Path(candidate_target).resolve()), *[str(Path(path).resolve()) for path in candidate_guardrails]],
    )
    if output_path is not None:
        resolved = Path(output_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(asdict(result), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def _read_result(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"eval result does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"invalid eval result artifact: {resolved}")
    required = {"eval_dataset", "dataset_fingerprint", "eval_db_ids", "db_disjoint_verified", "scorer_version", "generation_config", "case_count", "pass_rate"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"eval result is missing promotion fields at {resolved}: {', '.join(missing)}")
    return payload


def _assert_comparable(base: dict[str, Any], candidate: dict[str, Any], *, label: str) -> list[str]:
    reasons: list[str] = []
    for field in ("eval_dataset", "dataset_fingerprint", "eval_db_ids", "db_disjoint_verified", "scorer_version", "generation_config"):
        if base[field] != candidate[field]:
            reasons.append(f"{label} {field} differs")
    if not base["db_disjoint_verified"] or not candidate["db_disjoint_verified"]:
        reasons.append(f"{label} is not database-disjoint")
    base_ids = sorted(str(record.get("case_id", "")) for record in base["records"])
    candidate_ids = sorted(str(record.get("case_id", "")) for record in candidate["records"])
    if base_ids != candidate_ids:
        reasons.append(f"{label} case IDs differ")
    if int(base["case_count"]) != len(base_ids) or int(candidate["case_count"]) != len(candidate_ids):
        reasons.append(f"{label} case count does not match records")
    return reasons
