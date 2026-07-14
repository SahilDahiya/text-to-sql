"""SQL training and local validation primitives for the LiveSQLBench lane."""

from .eval_analysis import SQLEvalAnalysisSummary, SQLEvalFailureAnalysis, analyze_sql_eval_result
from .eval_runner import SQLEvalRunSummary, extract_generated_sql, run_sql_eval
from .eval_types import SQLCaseEvalRecord
from .curriculum import CURRICULUM_SPEC_VERSION, CURRICULUM_TIERS, curriculum_tier_name
from .evaluator import SQLEvaluationResult, evaluate_postgresql_case, evaluate_sql_case, evaluate_sqlite_case
from .failure_mining import (
    SQLCorrectionCollectionSummary,
    build_next_sql_train_mixture,
    collect_verified_sql_corrections,
)
from .leakage import SQLLeakageAuditSummary, assert_no_sql_dataset_leakage, audit_sql_dataset_leakage
from .livesqlbench_adapter import (
    LiveSQLBenchImportSummary,
    LiveSQLBenchTask,
    LiveSQLBenchVerificationSummary,
    build_livesqlbench_artifacts,
    discover_livesqlbench_tasks,
    verify_livesqlbench_targets,
)
from .loaders import load_sql_correction_examples, load_sql_eval_cases, load_sql_train_examples
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .mixture import SQLMixtureAuditSummary, assert_mixture_coverage, audit_sql_mixture
from .models import SQLCorrectionExample, SQLEvalCase, SQLProvenance, SQLTaskMetadata, SQLTrainExample, SQLVerification
from .promotion import SQLPromotionDecision, compare_sql_promotion
from .rendering import build_eval_messages, build_train_messages
from .training import SQLSFTTrainingSummary, run_sql_sft, tokenize_sql_sft_messages

__all__ = [
    "SQLCaseEvalRecord",
    "SQLCorrectionCollectionSummary",
    "SQLCorrectionExample",
    "SQLEvalAnalysisSummary",
    "SQLEvalCase",
    "SQLEvalFailureAnalysis",
    "SQLEvalRunSummary",
    "SQLEvaluationResult",
    "SQLLeakageAuditSummary",
    "SQLMixtureAuditSummary",
    "SQLPromotionDecision",
    "SQLProvenance",
    "SQLSFTExperimentManifest",
    "SQLSFTTrainingSummary",
    "SQLTaskMetadata",
    "SQLTrainExample",
    "SQLVerification",
    "CURRICULUM_SPEC_VERSION",
    "CURRICULUM_TIERS",
    "LiveSQLBenchImportSummary",
    "LiveSQLBenchTask",
    "LiveSQLBenchVerificationSummary",
    "analyze_sql_eval_result",
    "assert_mixture_coverage",
    "assert_no_sql_dataset_leakage",
    "audit_sql_dataset_leakage",
    "audit_sql_mixture",
    "build_livesqlbench_artifacts",
    "build_next_sql_train_mixture",
    "build_eval_messages",
    "build_train_messages",
    "collect_verified_sql_corrections",
    "compare_sql_promotion",
    "discover_livesqlbench_tasks",
    "evaluate_postgresql_case",
    "evaluate_sql_case",
    "evaluate_sqlite_case",
    "extract_generated_sql",
    "load_sql_correction_examples",
    "load_sql_eval_cases",
    "load_sql_sft_manifest",
    "load_sql_train_examples",
    "run_sql_eval",
    "run_sql_sft",
    "tokenize_sql_sft_messages",
    "curriculum_tier_name",
    "verify_livesqlbench_targets",
]
