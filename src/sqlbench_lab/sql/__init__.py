"""SQL training and evaluation primitives."""

from .benchmark_import import SQLBenchmarkImportSummary, import_sql_benchmark
from .bird_lab import (
    BIRDSchemaLabSummary,
    BIRDTrainLabSummary,
    generate_bird_regional_sales_normalization_micro_lab,
    generate_bird_regional_sales_schema_lab,
    generate_bird_regional_sales_unit_price_contrast_lab,
    generate_bird_superstore_schema_lab,
)
from .eval_analysis import SQLEvalAnalysisSummary, SQLEvalFailureAnalysis, analyze_sql_eval_result
from .eval_runner import SQLEvalRunSummary, extract_generated_sql, run_sql_eval, run_sql_eval_with_repair
from .eval_types import SQLCaseEvalRecord, SQLRepairAttemptRecord, SQLRepairEvalCaseRecord, SQLRepairEvalRunSummary
from .evaluator import SQLEvaluationResult, evaluate_sqlite_case
from .fixtures import build_sqlite_fixture
from .leakage import SQLLeakageAuditSummary, assert_no_sql_dataset_leakage, audit_sql_dataset_leakage
from .loaders import load_sql_eval_cases, load_sql_repair_examples, load_sql_train_examples
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .models import SQLEvalCase, SQLRepairExample, SQLTrainExample
from .profile_metadata import SQLProfileMetadataSummary, attach_sqlite_profile_metadata
from .prompt_optimization import SQLPromptCandidateSummary, record_sql_prompt_candidate
from .repair_collection import SQLRepairCollectionSummary, collect_sql_repair_data
from .rendering import build_eval_messages, build_repair_eval_messages, build_repair_messages, build_train_messages
from .training import SQLSFTTrainingSummary, run_sql_sft, tokenize_sql_sft_messages

__all__ = [
    "SQLBenchmarkImportSummary",
    "BIRDSchemaLabSummary",
    "BIRDTrainLabSummary",
    "SQLCaseEvalRecord",
    "SQLEvalCase",
    "SQLEvalAnalysisSummary",
    "SQLEvalFailureAnalysis",
    "SQLEvalRunSummary",
    "SQLEvaluationResult",
    "SQLLeakageAuditSummary",
    "SQLRepairAttemptRecord",
    "SQLRepairEvalCaseRecord",
    "SQLRepairEvalRunSummary",
    "SQLRepairExample",
    "SQLRepairCollectionSummary",
    "SQLSFTExperimentManifest",
    "SQLSFTTrainingSummary",
    "SQLProfileMetadataSummary",
    "SQLPromptCandidateSummary",
    "SQLTrainExample",
    "attach_sqlite_profile_metadata",
    "build_eval_messages",
    "build_repair_eval_messages",
    "build_repair_messages",
    "build_sqlite_fixture",
    "build_train_messages",
    "analyze_sql_eval_result",
    "assert_no_sql_dataset_leakage",
    "audit_sql_dataset_leakage",
    "collect_sql_repair_data",
    "evaluate_sqlite_case",
    "extract_generated_sql",
    "generate_bird_superstore_schema_lab",
    "generate_bird_regional_sales_normalization_micro_lab",
    "generate_bird_regional_sales_schema_lab",
    "generate_bird_regional_sales_unit_price_contrast_lab",
    "import_sql_benchmark",
    "load_sql_eval_cases",
    "load_sql_repair_examples",
    "load_sql_sft_manifest",
    "load_sql_train_examples",
    "record_sql_prompt_candidate",
    "run_sql_eval",
    "run_sql_eval_with_repair",
    "run_sql_sft",
    "tokenize_sql_sft_messages",
]
