"""Human review packets for the semi-automated ISFT loop."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loaders import load_sql_eval_cases, load_sql_train_examples
from .manifest import load_sql_sft_manifest

REVIEW_DECISIONS = {"approve", "reject", "request_extra_review"}
REVIEW_PHASES = {"artifacts", "baseline", "evaluation"}


@dataclass(frozen=True)
class ReviewPacketSummary:
    packet_id: str
    phase: str
    markdown_path: str
    json_path: str


def build_review_packet(
    *,
    iteration_id: str,
    phase: str,
    manifest_path: str | Path,
    output_path: str | Path,
    result_path: str | Path | None = None,
    conversation_path: str | Path | None = None,
) -> ReviewPacketSummary:
    if not iteration_id.strip():
        raise ValueError("iteration_id is required")
    if phase not in REVIEW_PHASES:
        raise ValueError(f"phase must be one of {sorted(REVIEW_PHASES)}")
    manifest_file = Path(manifest_path).resolve()
    manifest = load_sql_sft_manifest(manifest_file)
    train_files = tuple(manifest.resolve_workspace_path(path) for path in manifest.train_datasets)
    eval_file = manifest.resolve_workspace_path(manifest.eval_plan.target_dataset)
    eval_cases = load_sql_eval_cases(eval_file)
    train_rows = [row for path in train_files for row in load_sql_train_examples(path)]
    if phase in {"baseline", "evaluation"} and result_path is None:
        raise ValueError(f"{phase} review packets require result_path")
    result = _load_result(result_path) if result_path is not None else None
    conversation = _read_optional_text(conversation_path)
    packet_id = f"{iteration_id}:{phase}"
    payload = {
        "packet_id": packet_id,
        "iteration_id": iteration_id,
        "phase": phase,
        "manifest_path": str(manifest_file),
        "manifest_sha256": _sha256(manifest_file),
        "train_datasets": [str(path) for path in train_files],
        "train_sha256": {str(path): _sha256(path) for path in train_files},
        "eval_dataset": str(eval_file),
        "eval_sha256": _sha256(eval_file),
        "train_row_count": len(train_rows),
        "eval_case_count": len(eval_cases),
        "initial_adapter_dir": (
            str(manifest.resolve_workspace_path(manifest.initial_adapter_dir))
            if manifest.initial_adapter_dir is not None
            else None
        ),
        "adapter_dir": str(manifest.resolve_workspace_path(manifest.output_paths.adapter_dir)),
        "train_summary_path": str(manifest.resolve_workspace_path(manifest.output_paths.train_summary_json)),
        "eval_summary_path": str(manifest.resolve_workspace_path(manifest.output_paths.eval_summary_json)),
        "result_path": str(Path(result_path).resolve()) if result_path is not None else None,
        "result_sha256": _sha256(Path(result_path).resolve()) if result_path is not None else None,
        "conversation_path": str(Path(conversation_path).resolve()) if conversation_path is not None else None,
    }
    markdown = _render_markdown(payload, train_rows=train_rows, eval_cases=eval_cases, result=result, conversation=conversation)
    markdown_file = Path(output_path).resolve()
    if markdown_file.suffix != ".md":
        raise ValueError("review packet output must end in .md")
    json_file = markdown_file.with_suffix(".json")
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(markdown_file, markdown)
    _write_atomic(json_file, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    return ReviewPacketSummary(packet_id, phase, str(markdown_file), str(json_file))


def record_human_review(
    *,
    packet_path: str | Path,
    reviewer: str,
    decision: str,
    output_path: str | Path,
    notes: str = "",
    extra_questions: list[str] | tuple[str, ...] = (),
) -> Path:
    packet_file = Path(packet_path).resolve()
    packet = json.loads(packet_file.read_text(encoding="utf-8"))
    if not isinstance(packet, dict) or not isinstance(packet.get("packet_id"), str):
        raise ValueError(f"invalid review packet: {packet_file}")
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(REVIEW_DECISIONS)}")
    questions = [question.strip() for question in extra_questions if question.strip()]
    if decision == "request_extra_review" and not questions:
        raise ValueError("request_extra_review requires at least one extra question")
    review = {
        "packet_id": packet["packet_id"],
        "packet_path": str(packet_file),
        "packet_sha256": _sha256(packet_file),
        "phase": packet.get("phase"),
        "reviewer": reviewer.strip(),
        "decision": decision,
        "notes": notes.strip(),
        "extra_questions": questions,
    }
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(output_file, json.dumps(review, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    return output_file


def require_approved_review(
    review_path: str | Path,
    *,
    manifest_path: str | Path,
    phase: str = "artifacts",
) -> None:
    review_file = Path(review_path).resolve()
    review = json.loads(review_file.read_text(encoding="utf-8"))
    if not isinstance(review, dict) or review.get("decision") != "approve":
        raise ValueError(f"human approval is required before training: {review_file}")
    if review.get("phase") != phase:
        raise ValueError(f"review phase must be {phase!r}: {review_file}")
    packet_path = review.get("packet_path")
    if not isinstance(packet_path, str) or not packet_path.strip():
        raise ValueError(f"review is missing packet_path: {review_file}")
    packet_file = Path(packet_path).resolve()
    if not packet_file.is_file():
        raise FileNotFoundError(f"review packet JSON is missing beside review: {packet_file}")
    packet = json.loads(packet_file.read_text(encoding="utf-8"))
    if review.get("packet_sha256") != _sha256(packet_file):
        raise ValueError(f"review packet changed after human approval: {packet_file}")
    if Path(packet["manifest_path"]).resolve() != Path(manifest_path).resolve():
        raise ValueError("approved review belongs to a different manifest")


def _load_result(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"eval result must contain records: {resolved}")
    return payload


def _render_markdown(
    payload: dict[str, Any],
    *,
    train_rows: list[Any],
    eval_cases: list[Any],
    result: dict[str, Any] | None,
    conversation: str | None,
) -> str:
    lines = [
        f"# ISFT Review Packet: {payload['packet_id']}",
        "",
        "## Human Decision",
        "- Decision: `approve`, `reject`, or `request_extra_review`",
        "- Reviewer:",
        "- Notes:",
        "- Extra review questions:",
        "",
        "## Artifact Evidence",
        f"- Manifest: `{payload['manifest_path']}`",
        f"- Train rows: {payload['train_row_count']}",
        f"- Eval cases: {payload['eval_case_count']}",
        f"- Initial adapter: `{payload['initial_adapter_dir'] or 'none'}`",
        f"- Output adapter/checkpoint: `{payload['adapter_dir']}`",
        f"- Training summary: `{payload['train_summary_path']}`",
        f"- Evaluation summary: `{payload['eval_summary_path']}`",
        "",
        "## Training Label Evidence",
        "",
        "| Row | Task | Database | Target SQL | Source | Verification |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in train_rows:
        lines.append(
            "| {row_id} | {task_id} | {db_id} | `{target_sql}` | {source} | {verification} |".format(
                row_id=_cell(row.row_id),
                task_id=_cell(row.task_id),
                db_id=_cell(row.db_id),
                target_sql=_cell(row.target_sql),
                source=_cell(row.provenance.target_source),
                verification=_cell(row.verification.verification_id),
            )
        )
    if result is not None:
        lines.extend([
            "",
            "## Evaluation Evidence",
            f"- Result: `{payload['result_path']}`",
            f"- Result SHA-256: `{payload['result_sha256']}`",
            f"- Passed: {result.get('passed_count')}/{result.get('case_count')}",
            "",
            "| Case | Passed | Predicted SQL | Gold SQL | Predicted rows | Gold rows | Errors |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        gold_sql_by_case = {case.case_id: case.gold_sql for case in eval_cases}
        result_case_ids = {str(record.get("case_id", "")) for record in result["records"]}
        missing_case_ids = sorted(result_case_ids - set(gold_sql_by_case))
        if missing_case_ids:
            raise ValueError("eval result references unknown cases: " + ", ".join(missing_case_ids[:10]))
        for record in result["records"]:
            prediction_error = str(record.get("prediction_error") or "")
            gold_error = str(record.get("gold_error") or "")
            lines.append(
                "| {case_id} | {passed} | `{sql}` | `{gold_sql}` | `{predicted_rows}` | `{gold_rows}` | {errors} |".format(
                    case_id=_cell(record.get("case_id", "")),
                    passed=record.get("passed", False),
                    sql=_cell(record.get("predicted_sql", "")),
                    gold_sql=_cell(gold_sql_by_case.get(str(record.get("case_id", "")), "")),
                    predicted_rows=_cell(json.dumps(record.get("predicted_rows", []), ensure_ascii=True, default=str)),
                    gold_rows=_cell(json.dumps(record.get("gold_rows", []), ensure_ascii=True, default=str)),
                    errors=_cell("; ".join(error for error in (prediction_error, gold_error) if error)),
                )
            )
    if conversation is not None:
        lines.extend(["", "## Coding-Agent Conversation", "", conversation])
    lines.extend(["", "## Review Checklist", "- Are the train labels permitted and verified?", "- Are the generated SQL and execution results understood?", "- Should the next iteration change the training set?", "- Is extra review needed before proceeding?"])
    return "\n".join(lines) + "\n"


def _read_optional_text(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"conversation file does not exist: {resolved}")
    return resolved.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
