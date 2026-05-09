"""SQL training and evaluation primitives."""

from .evaluator import SQLEvaluationResult, evaluate_sqlite_case
from .fixtures import build_sqlite_fixture
from .loaders import load_sql_eval_cases, load_sql_repair_examples, load_sql_train_examples
from .manifest import SQLSFTExperimentManifest, load_sql_sft_manifest
from .models import SQLEvalCase, SQLRepairExample, SQLTrainExample
from .rendering import build_repair_messages, build_train_messages
from .training import SQLSFTTrainingSummary, run_sql_sft, tokenize_sql_sft_messages

__all__ = [
    "SQLEvalCase",
    "SQLEvaluationResult",
    "SQLRepairExample",
    "SQLSFTExperimentManifest",
    "SQLSFTTrainingSummary",
    "SQLTrainExample",
    "build_repair_messages",
    "build_sqlite_fixture",
    "build_train_messages",
    "evaluate_sqlite_case",
    "load_sql_eval_cases",
    "load_sql_repair_examples",
    "load_sql_sft_manifest",
    "load_sql_train_examples",
    "run_sql_sft",
    "tokenize_sql_sft_messages",
]
