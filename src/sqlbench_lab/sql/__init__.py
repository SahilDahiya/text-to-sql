"""SQL training and local validation primitives for the LiveSQLBench lane."""

from .eval_runner import SQLEvalRunSummary, extract_generated_sql, run_sql_eval
from .eval_types import SQLCaseEvalRecord
from .evaluator import SQLEvaluationResult, evaluate_postgresql_case, evaluate_sql_case, evaluate_sqlite_case
from .livesqlbench_adapter import (
    LiveSQLBenchImportSummary,
    LiveSQLBenchTask,
    LiveSQLBenchVerificationSummary,
    build_livesqlbench_artifacts,
    discover_livesqlbench_tasks,
    verify_livesqlbench_targets,
)
from .loaders import load_sql_eval_cases, load_sql_train_examples
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .mixture import SQLMixtureAuditSummary, audit_sql_mixture
from .models import SQLEvalCase, SQLProvenance, SQLTrainExample, SQLVerification
from .rendering import build_eval_messages, build_train_messages
from .review import ReviewPacketSummary, build_review_packet, record_human_review
from .training import SQLSFTTrainingSummary, run_sql_sft, tokenize_sql_sft_messages

__all__ = [
    "SQLCaseEvalRecord",
    "SQLEvalCase",
    "SQLEvalRunSummary",
    "SQLEvaluationResult",
    "SQLMixtureAuditSummary",
    "SQLProvenance",
    "SQLSFTExperimentManifest",
    "SQLSFTTrainingSummary",
    "SQLTrainExample",
    "SQLVerification",
    "ReviewPacketSummary",
    "LiveSQLBenchImportSummary",
    "LiveSQLBenchTask",
    "LiveSQLBenchVerificationSummary",
    "audit_sql_mixture",
    "build_livesqlbench_artifacts",
    "build_eval_messages",
    "build_train_messages",
    "build_review_packet",
    "discover_livesqlbench_tasks",
    "evaluate_postgresql_case",
    "evaluate_sql_case",
    "evaluate_sqlite_case",
    "extract_generated_sql",
    "load_sql_eval_cases",
    "load_sql_sft_manifest",
    "load_sql_train_examples",
    "run_sql_eval",
    "run_sql_sft",
    "record_human_review",
    "tokenize_sql_sft_messages",
    "verify_livesqlbench_targets",
]
