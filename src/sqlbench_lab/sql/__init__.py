"""SQL training and evaluation primitives."""

from .benchmark_import import SQLBenchmarkImportSummary, import_sql_benchmark
from .eval_analysis import SQLEvalAnalysisSummary, SQLEvalFailureAnalysis, analyze_sql_eval_result
from .eval_runner import SQLEvalRunSummary, extract_generated_sql, run_sql_eval
from .eval_types import SQLCaseEvalRecord
from .evaluator import SQLEvaluationResult, evaluate_sqlite_case
from .fixtures import build_sqlite_fixture
from .loaders import load_sql_eval_cases, load_sql_repair_examples, load_sql_train_examples
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .models import SQLEvalCase, SQLRepairExample, SQLTrainExample
from .rendering import build_eval_messages, build_repair_messages, build_train_messages
from .training import SQLSFTTrainingSummary, run_sql_sft, tokenize_sql_sft_messages

__all__ = [
    "SQLBenchmarkImportSummary",
    "SQLCaseEvalRecord",
    "SQLEvalCase",
    "SQLEvalAnalysisSummary",
    "SQLEvalFailureAnalysis",
    "SQLEvalRunSummary",
    "SQLEvaluationResult",
    "SQLRepairExample",
    "SQLSFTExperimentManifest",
    "SQLSFTTrainingSummary",
    "SQLTrainExample",
    "build_eval_messages",
    "build_repair_messages",
    "build_sqlite_fixture",
    "build_train_messages",
    "analyze_sql_eval_result",
    "evaluate_sqlite_case",
    "extract_generated_sql",
    "import_sql_benchmark",
    "load_sql_eval_cases",
    "load_sql_repair_examples",
    "load_sql_sft_manifest",
    "load_sql_train_examples",
    "run_sql_eval",
    "run_sql_sft",
    "tokenize_sql_sft_messages",
]
