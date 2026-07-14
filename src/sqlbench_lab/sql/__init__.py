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
from .models import SQLEvalCase, SQLProvenance, SQLTrainExample, SQLVerification
from .rendering import build_eval_messages, build_train_messages, render_sql_sft_prompt
from .training import SQLSFTTrainingSummary, run_sql_sft, tokenize_sql_sft_messages

__all__ = [
    "SQLCaseEvalRecord",
    "SQLEvalCase",
    "SQLEvalRunSummary",
    "SQLEvaluationResult",
    "SQLProvenance",
    "SQLSFTExperimentManifest",
    "SQLSFTTrainingSummary",
    "SQLTrainExample",
    "SQLVerification",
    "LiveSQLBenchImportSummary",
    "LiveSQLBenchTask",
    "LiveSQLBenchVerificationSummary",
    "build_livesqlbench_artifacts",
    "build_eval_messages",
    "build_train_messages",
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
    "render_sql_sft_prompt",
    "tokenize_sql_sft_messages",
    "verify_livesqlbench_targets",
]
