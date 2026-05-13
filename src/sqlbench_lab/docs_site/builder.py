"""Build the SQLBench Lab browser docs."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from typing import Any

from sqlbench_lab.paths import WORKSPACE_ROOT

EXPERIMENT_RE = re.compile(r"exp(?P<number>\d+)")


@dataclass(frozen=True)
class DocsSiteSummary:
    """Summary of a generated static docs site."""

    output_dir: Path
    page_count: int
    asset_count: int


@dataclass(frozen=True)
class EvalRecord:
    """Evaluation result surfaced in browser docs."""

    label: str
    dataset: str
    passed: int
    total: int
    pass_rate: float
    analysis_path: str | None
    failure_counts: dict[str, int]


@dataclass(frozen=True)
class ExperimentRecord:
    """Experiment manifest plus discovered local outputs."""

    experiment_id: str
    number: int
    manifest_path: str
    base_model: str
    adapter_name: str
    method: str
    prompt_style: str
    stage: str
    notes: str
    train_datasets: list[str]
    backend: str
    epochs: float | None
    learning_rate: float | None
    packing: bool | None
    train_rows: int | None
    train_loss: float | None
    train_runtime: float | None
    train_summary_path: str | None
    evals: list[EvalRecord]


PIPELINE_STAGES: list[dict[str, str]] = [
    {
        "stage": "Benchmark Intake",
        "job": "Import public Spider/BIRD rows into canonical JSONL without hidden splits.",
        "artifact": "datasets/sql/train/*.jsonl, datasets/sql/eval/*.jsonl",
        "command": "uv run python -m sqlbench_lab.cli sql import-benchmark ...",
        "risk": "Protected-data leakage or untracked row selection.",
    },
    {
        "stage": "Dataset Contract",
        "job": "Validate train/eval schemas and fail on malformed records.",
        "artifact": "sql_train:v1, sql_eval:v1 JSONL",
        "command": "uv run python -m sqlbench_lab.cli sql validate-train --dataset ...",
        "risk": "Invalid durable state feeding training.",
    },
    {
        "stage": "Leakage Audit",
        "job": "Check question SQL overlap and optional train/eval DB disjointness.",
        "artifact": "CLI audit summary",
        "command": "uv run python -m sqlbench_lab.cli sql audit-leakage ...",
        "risk": "Inflated scores from repeated examples or seen DBs.",
    },
    {
        "stage": "Schema Lab",
        "job": "Create controlled train/eval slices for one DB before expanding.",
        "artifact": "bird_*_schema_lab_*.jsonl",
        "command": "uv run python -m sqlbench_lab.cli sql generate-bird-lab ...",
        "risk": "Overfitting a template instead of learning schema use.",
    },
    {
        "stage": "Profile Metadata",
        "job": "Attach compact SQLite-derived column/value notes to real benchmark rows before broader scaling.",
        "artifact": "datasets/sql/*/*_profile_notes*.jsonl",
        "command": "uv run python -m sqlbench_lab.cli sql profile-metadata ...",
        "risk": "Training on raw DDL when the missing skill is value grounding and schema linking.",
    },
    {
        "stage": "Prompt Rendering",
        "job": "Render canonical chat prompts with schema and optional value notes.",
        "artifact": "Rendered SFT rows",
        "command": "uv run python -m sqlbench_lab.cli sql run-sft --dry-run ...",
        "risk": "Training on a prompt shape evaluation never uses.",
    },
    {
        "stage": "Training Manifest",
        "job": "Pin model, LoRA, trainer, datasets, and output paths per experiment.",
        "artifact": "experiments/sql/*.json",
        "command": "uv run python -m sqlbench_lab.cli sql validate-manifest --manifest ...",
        "risk": "Unrepeatable experiment settings.",
    },
    {
        "stage": "One-Shot SFT",
        "job": "Train the direct text-to-SQL adapter. No repair-stage learning in current focus.",
        "artifact": "artifacts/sql/<experiment>/adapter",
        "command": "uv run python -m sqlbench_lab.cli sql run-sft --manifest ...",
        "risk": "Optimizing repair behavior before the first shot is strong.",
    },
    {
        "stage": "Smoke Eval",
        "job": "Run fast result-equivalence evals after each adapter.",
        "artifact": "results/sql/<experiment>/adapter__*.json",
        "command": "uv run python -m sqlbench_lab.cli sql eval --model adapter ...",
        "risk": "Trusting loss when SQL behavior regressed.",
    },
    {
        "stage": "Failure Analysis",
        "job": "Classify failures by syntax, schema, runtime, wrong result, or empty output.",
        "artifact": "results/sql/<experiment>/*.analysis.json",
        "command": "uv run python -m sqlbench_lab.cli sql analyze-eval --result ...",
        "risk": "Adding broad data when one missing skill is blocking.",
    },
    {
        "stage": "Prompt Optimization",
        "job": "Iterate MIPROv2/GEPA prompt candidates on prompt-dev while logging every candidate to MLflow.",
        "artifact": "results/sql/<experiment>/prompt_candidates/*.json",
        "command": "uv run --group observability python -m sqlbench_lab.cli sql optimize-prompt ... --mlflow",
        "risk": "Selecting a prompt by memory or screenshots instead of comparable tracked evidence.",
    },
    {
        "stage": "Observability",
        "job": "Track train/eval metrics, manifests, artifacts, and tags in MLflow.",
        "artifact": "mlruns/, MLflow UI",
        "command": "uv run python -m sqlbench_lab.cli observe ui",
        "risk": "Losing which intervention moved which metric.",
    },
    {
        "stage": "Experiment Ledger",
        "job": "Record durable lessons in Linear comments, not one issue per run.",
        "artifact": "Linear learning/project comments",
        "command": "Manual via Linear connector",
        "risk": "Issue explosion with no usable memory.",
    },
    {
        "stage": "DB Expansion",
        "job": "Add DBs gradually and hold out unseen DBs to measure generalization.",
        "artifact": "Train/eval split policy",
        "command": "uv run python -m sqlbench_lab.cli sql audit-leakage --require-db-disjoint ...",
        "risk": "Mistaking memorized DB patterns for transferable SQL ability.",
    },
    {
        "stage": "LiveSQLBench Gate",
        "job": "Promote only adapters that pass local gates into official benchmark tooling.",
        "artifact": "LiveSQLBench run outputs",
        "command": "Official benchmark command, not local approximate score",
        "risk": "Reporting unofficial results as competition results.",
    },
]


TOOLING_ROWS: list[dict[str, str]] = [
    {
        "tool": "Transformers",
        "status": "Active",
        "value": "Base model/tokenizer loading and generation path.",
        "next": "Keep as the model substrate.",
    },
    {
        "tool": "Datasets",
        "status": "Useful",
        "value": "Better large-set transforms and splits as we move beyond small JSONL labs.",
        "next": "Adopt where it removes local loader code.",
    },
    {
        "tool": "PEFT",
        "status": "Active",
        "value": "LoRA adapter construction and checkpoint format.",
        "next": "Keep adapter settings explicit in manifests.",
    },
    {
        "tool": "TRL SFTTrainer",
        "status": "Active",
        "value": "Standard SFT loop, formatting hooks, masking, and packing knobs.",
        "next": "Run controlled packed/unpacked comparisons.",
    },
    {
        "tool": "bitsandbytes",
        "status": "Planned",
        "value": "QLoRA and lower memory; more valuable for larger models or longer context.",
        "next": "Add when local/cloud GPU memory becomes the bottleneck.",
    },
    {
        "tool": "Accelerate",
        "status": "Implicit",
        "value": "Trainer device placement and future multi-GPU/cloud launch path.",
        "next": "Expose config once distributed runs become real.",
    },
    {
        "tool": "FlashAttention",
        "status": "Blocked locally",
        "value": "Speed/memory improvement on supported GPUs.",
        "next": "Use SDPA/eager locally on RTX 2080 Ti; use cloud GPU for FA2 tests.",
    },
    {
        "tool": "vLLM",
        "status": "Planned",
        "value": "Faster batched inference for larger eval sweeps and LiveSQLBench prep.",
        "next": "Add after one-shot quality improves.",
    },
    {
        "tool": "Axolotl",
        "status": "Watch",
        "value": "Recipe-driven training for larger, more standardized cloud runs.",
        "next": "Compare after the in-repo loop is stable.",
    },
]

HISTORY_ROWS: list[dict[str, str]] = [
    {
        "phase": "Exp001-006",
        "focus": "Initial direct SQL SFT, Spider/BIRD imports, identifier-copy prompting.",
        "signal": "BIRD stayed low, Spider guardrail was easier to preserve.",
        "lesson": "Generic real rows and prompt wording were not enough; BIRD needed targeted schema/value work.",
    },
    {
        "phase": "Exp007",
        "focus": "Controlled TRL SFTTrainer migration.",
        "signal": "BIRD improved to 3/25 while Spider dipped to 17/25.",
        "lesson": "TRL is worth keeping, but backend changes must be scored as quality experiments, not just tooling swaps.",
    },
    {
        "phase": "Exp008-009",
        "focus": "Packing, bf16/tf32, and FlashAttention on local/cloud hardware.",
        "signal": "Packing improved runtime, but quality regressed; cloud FlashAttention validated speed but not quality.",
        "lesson": "A faster recipe is rejected if fixed eval quality collapses. Do not stack QLoRA or Liger onto a bad packed recipe.",
    },
    {
        "phase": "Exp010-017",
        "focus": "Local overnight unpacked queue to separate data quality from runtime tooling.",
        "signal": "Exp014 matched best BIRD and improved Spider, but no run beat 3/25 BIRD.",
        "lesson": "More real BIRD rows alone did not fix BIRD; schema-targeted data mattered more than volume.",
    },
    {
        "phase": "Exp018-020",
        "focus": "PremSQL prompt and stratified BIRD/Spider slices.",
        "signal": "Stratified BIRD 110 stayed schema-error heavy.",
        "lesson": "The missing capability was not only benchmark coverage; validation schemas exposed unresolved schema linking.",
    },
    {
        "phase": "Exp021-022",
        "focus": "Single-DB superstore lab and computed-order fix.",
        "signal": "Superstore moved from 36/40 to 40/40.",
        "lesson": "A controlled DB lab can isolate one missing skill and verify a targeted fix without benchmark noise.",
    },
    {
        "phase": "Exp023",
        "focus": "Two-DB expansion with superstore plus regional_sales.",
        "signal": "Superstore held 40/40; regional_sales reached 37/40.",
        "lesson": "DB expansion should happen gradually, with same-DB and future unseen-DB eval lanes separated.",
    },
    {
        "phase": "Exp024",
        "focus": "Broad regional_sales normalization curriculum.",
        "signal": "Regional_sales regressed to 33/40 with identifier errors.",
        "lesson": "Broad additions can teach the wrong shortcut; reject regressions even when the intent is correct.",
    },
    {
        "phase": "Exp025-026",
        "focus": "Normalization micro-lab and column value notes.",
        "signal": "Both preserved 37/40 but did not fix unit-price aggregation.",
        "lesson": "Sidecar examples and passive notes may not override a stable shorter decode path.",
    },
    {
        "phase": "Exp027",
        "focus": "Direct unit-price target-shape contrast rows.",
        "signal": "Still 37/40.",
        "lesson": "A small contrast set outside the canonical slot was too weak for this adapter.",
    },
    {
        "phase": "Exp028",
        "focus": "Move normalized unit price into the canonical regional_sales train slot.",
        "signal": "Regional_sales improved to 38/40; superstore held 40/40.",
        "lesson": "Canonical slot placement worked where sidecar examples did not; remaining gap is quoted Order Quantity.",
    },
    {
        "phase": "Exp029",
        "focus": "Add SQLite profile-derived notes to regional_sales train and eval prompts.",
        "signal": "Regional_sales reached 40/40 and superstore held 40/40; train runtime rose to about 22.4 minutes.",
        "lesson": "Profile metadata fixed the remaining quoted-identifier failures, but full notes are context-expensive and should be compressed before broad DB expansion.",
    },
    {
        "phase": "Unseen Gate 001",
        "focus": "Run the Exp029 adapter on a DB-disjoint BIRD holdout from restaurant and airline.",
        "signal": "Adapter scored 4/50 versus base 3/50; failures were still schema-error heavy.",
        "lesson": "Seen-DB curriculum wins did not transfer. The next loop needs broader train DB coverage and cleaner schema-linking support, not more same-DB polishing.",
    },
    {
        "phase": "Exp030",
        "focus": "Add 100 real BIRD train rows from sales and bike_share_1 while keeping restaurant and airline unseen.",
        "signal": "Unseen holdout moved from 4/50 to 5/50; regional_sales and superstore both held 40/40.",
        "lesson": "More train DB coverage reduced schema errors without damaging seen labs, but the gain was tiny and value errors increased. Expansion alone is not enough.",
    },
    {
        "phase": "Exp031",
        "focus": "Attach generic SQLite profile metadata to sales, bike_share_1, and the fixed unseen holdout.",
        "signal": "Unseen holdout moved from 5/50 to 7/50; raw and profile-note holdout prompts both scored 7/50; seen guardrails held 40/40.",
        "lesson": "Minimal deterministic profile notes are feasible and mildly helpful, but two notes per row are too weak. The next metadata work should improve note selection/schema linking, not just add longer notes locally.",
    },
    {
        "phase": "Exp032 plan",
        "focus": "Use DSPy prompt optimization as a separate inference recipe lane before another SFT run.",
        "signal": "Restaurant plus airline is prompt-dev; works_cycles plus public_review_platform is the fresh 50-case unseen gate. Baseline candidate c000_current_system is logged to MLflow.",
        "lesson": "Optimization over a known holdout is allowed only when it is labeled as dev, every candidate is tracked in MLflow, and selected prompts are followed by a fresh DB-disjoint evaluation.",
    },
    {
        "phase": "Exp032 manual candidates",
        "focus": "Add prompt override evals and test two hand-written prompt candidates before DSPy.",
        "signal": "c000 baseline scored 7/50 on prompt-dev and 7/50 on the fresh gate. c001_schema_grounded and c002_value_grounded each scored 6/50 on prompt-dev and were rejected.",
        "lesson": "Longer instruction prompts did not fix transfer. Next prompt optimization should use failure feedback/search rather than manual instruction padding.",
    },
    {
        "phase": "Exp033 setup",
        "focus": "Return to training with explicit schema-linking supervision instead of more prompt-only search.",
        "signal": "Added schema_linking_notes to the data contract. Train rows use gold-SQL-derived result-shape/table/column notes; eval rows use question/schema/value-note-derived notes to avoid gold leakage.",
        "lesson": "DSPy-style optimization belongs upstream of SFT as data/program supervision. The trainer remains TRL/LoRA; the new experiment tests whether richer supervised inputs transfer better than prompt padding.",
    },
]

RUNBOOK_ROWS: list[dict[str, str]] = [
    {
        "task": "Build browser docs",
        "command": "uv run python -m sqlbench_lab.cli docs build",
        "output": "site/index.html",
        "gate": "Docs source changed first; generated site opens.",
    },
    {
        "task": "Serve browser docs",
        "command": "uv run python -m sqlbench_lab.cli docs serve",
        "output": "http://127.0.0.1:8000",
        "gate": "Use for browsing; generated site remains ignored by git.",
    },
    {
        "task": "Import benchmark slice",
        "command": "uv run --group training python -m sqlbench_lab.cli sql import-benchmark --benchmark bird --split validation --artifact eval --limit 25 --output datasets/sql/eval/bird_validation_sample_v1.jsonl",
        "output": "Canonical JSONL",
        "gate": "Only public/dev data allowed; hidden/protected data never enters train.",
    },
    {
        "task": "Generate BIRD DB lab",
        "command": "uv run python -m sqlbench_lab.cli sql generate-bird-lab --db-id regional_sales --curriculum-version v3 --train-output datasets/sql/train/bird_regional_sales_schema_lab_train_v3_unit_price_slot.jsonl --eval-output /tmp/bird_regional_sales_schema_lab_dev_v3_check.jsonl",
        "output": "Train/eval JSONL",
        "gate": "No exact train/eval SQL overlap.",
    },
    {
        "task": "Audit same-DB leakage",
        "command": "uv run python -m sqlbench_lab.cli sql audit-leakage --train-dataset datasets/sql/train/<train>.jsonl --eval-dataset datasets/sql/eval/<eval>.jsonl",
        "output": "Audit summary",
        "gate": "No task/question/SQL overlap.",
    },
    {
        "task": "Audit unseen-DB leakage",
        "command": "uv run python -m sqlbench_lab.cli sql audit-leakage --train-dataset datasets/sql/train/<train>.jsonl --eval-dataset datasets/sql/eval/<unseen>.jsonl --require-db-disjoint",
        "output": "Audit summary",
        "gate": "No overlapping db_id.",
    },
    {
        "task": "Train adapter",
        "command": "uv run --group training --group observability python -m sqlbench_lab.cli sql run-sft --manifest experiments/sql/<experiment>.json --mlflow",
        "output": "artifacts/sql/<experiment>/train_summary.json",
        "gate": "Manifest validated and MLflow run logged.",
    },
    {
        "task": "Evaluate adapter",
        "command": "uv run --group training --group observability python -m sqlbench_lab.cli sql eval --manifest experiments/sql/<experiment>.json --model adapter --dataset datasets/sql/eval/<eval>.jsonl --mlflow",
        "output": "results/sql/<experiment>/adapter__*.json",
        "gate": "Result-equivalence score recorded as local, not official.",
    },
    {
        "task": "Analyze failures",
        "command": "uv run python -m sqlbench_lab.cli sql analyze-eval --result results/sql/<experiment>/adapter__<eval>.json",
        "output": "Sibling .analysis.json",
        "gate": "Next intervention follows failure type, not guesswork.",
    },
    {
        "task": "Optimize prompt candidate",
        "command": "uv run --group observability python -m sqlbench_lab.cli sql optimize-prompt --experiment-id <experiment> --candidate-id <candidate> --prompt-dev-dataset datasets/sql/eval/<prompt-dev>.jsonl --mlflow",
        "output": "results/sql/<experiment>/prompt_candidates/<candidate>.json",
        "gate": "Every MIPROv2/GEPA candidate has MLflow tags before selection.",
    },
    {
        "task": "Open MLflow",
        "command": "uv run --group observability python -m sqlbench_lab.cli observe ui",
        "output": "http://127.0.0.1:5000",
        "gate": "Compare one dataset at a time.",
    },
]

RESEARCH_PAPER_ROWS: list[dict[str, str]] = [
    {
        "paper": "Automatic Metadata Extraction for Text-to-SQL",
        "source": "https://arxiv.org/pdf/2505.19988",
        "theme": "DB profiling, profile summaries, query-log analysis, SQL-to-text examples.",
        "use": "Build a SQLite profiler that emits value-shape notes, text-numeric notes, top values, and join candidates before the next SFT run.",
        "priority": "Now",
    },
    {
        "paper": "BIRD: Big Bench for Large-Scale Database Grounded Text-to-SQL",
        "source": "https://arxiv.org/abs/2305.03111",
        "theme": "Real database values, dirty contents, external knowledge, efficiency.",
        "use": "Keep BIRD as the main training/eval substrate; avoid treating Spider-style schema-only success as enough.",
        "priority": "Now",
    },
    {
        "paper": "Spider",
        "source": "https://arxiv.org/abs/1809.08887",
        "theme": "Cross-domain schema generalization with train/test database split.",
        "use": "Use Spider for broad SQL pattern coverage and DB-disjoint split hygiene, but do not let it mask BIRD value-grounding gaps.",
        "priority": "Now",
    },
    {
        "paper": "Spider 2.0",
        "source": "https://arxiv.org/abs/2411.07763",
        "theme": "Enterprise workflows, huge schemas, dialect docs, project files, multi-query tasks.",
        "use": "Use as a north-star for why LiveSQLBench needs agents, metadata retrieval, dialect handling, and workflow artifacts.",
        "priority": "Future",
    },
    {
        "paper": "LiveSQLBench",
        "source": "https://livesqlbench.ai/",
        "theme": "Live benchmark ladder with Base-Lite, Base-Full, Large, and agent-oriented releases.",
        "use": "Keep official score claims isolated from local BIRD/Spider lab scores; move from lite to full to large only after local gates.",
        "priority": "Now",
    },
    {
        "paper": "Text-to-SQL Empowered by Large Language Models / DAIL-SQL",
        "source": "https://arxiv.org/abs/2308.15363",
        "theme": "Prompt representation, example selection, example organization, token efficiency, SFT comparison.",
        "use": "Improve few-shot/example selection and track token budget as a first-class metric before scaling prompts.",
        "priority": "Near",
    },
    {
        "paper": "DIN-SQL",
        "source": "https://arxiv.org/abs/2304.11015",
        "theme": "Decomposed text-to-SQL with self-correction.",
        "use": "Use decomposition as a later inference/agent pattern; do not mix it into current one-shot SFT scoring.",
        "priority": "Near",
    },
    {
        "paper": "C3",
        "source": "https://arxiv.org/abs/2307.07306",
        "theme": "Clear prompting, calibration with hints, consistent output.",
        "use": "Mine prompt-format lessons for deterministic SQL-only output and explicit dialect/schema constraints.",
        "priority": "Near",
    },
    {
        "paper": "RESDSQL",
        "source": "https://arxiv.org/abs/2302.05965",
        "theme": "Decoupled schema linking and skeleton parsing.",
        "use": "Treat schema linking as its own train/eval target before expecting end-to-end SQL improvements.",
        "priority": "Near",
    },
    {
        "paper": "CHESS",
        "source": "https://arxiv.org/abs/2405.16755",
        "theme": "Retriever, schema selector, candidate generator, iterative refinement, unit tester.",
        "use": "Reference design for LiveSQLBench agent mode and for separating schema selection from SQL generation.",
        "priority": "Future",
    },
    {
        "paper": "MAC-SQL",
        "source": "https://arxiv.org/abs/2312.11242",
        "theme": "Multi-agent decomposition, sub-database selection, SQL refinement, fine-tuned SQL-Llama.",
        "use": "Future agent scaffold: decomposer plus selector plus refiner; keep separate from one-shot model quality.",
        "priority": "Future",
    },
    {
        "paper": "SQLFixAgent",
        "source": "https://huggingface.co/papers/2406.13408",
        "theme": "Execution/error-aware correction with consistency-enhanced multi-agent repair.",
        "use": "Use after one-shot plateau, when collected repair rows can train or evaluate a repair-stage adapter.",
        "priority": "Future",
    },
    {
        "paper": "CHASE-SQL",
        "source": "https://arxiv.org/abs/2410.01943",
        "theme": "Multi-path reasoning and preference-optimized candidate selection.",
        "use": "Future reranking lane: generate several SQL candidates, execute or score them, then select.",
        "priority": "Future",
    },
    {
        "paper": "XiYan-SQL",
        "source": "https://arxiv.org/abs/2411.08599",
        "theme": "Multi-generator ensemble, M-Schema, SFT plus ICL, refiner, selection model.",
        "use": "Longer-term architecture for candidate diversity and selection; useful once base one-shot generator is stable.",
        "priority": "Future",
    },
    {
        "paper": "Qwen2.5-Coder Technical Report",
        "source": "https://arxiv.org/abs/2409.12186",
        "theme": "Code-specialized data mixture, synthetic data, executable-code filtering, model-size ladder.",
        "use": "Guides model choice and data hygiene for SQL/code tasks; supports trying coder bases if Qwen3.5 base stalls.",
        "priority": "Near",
    },
    {
        "paper": "Survey on LLMs for Text-to-SQL",
        "source": "https://arxiv.org/abs/2407.15186",
        "theme": "Benchmarks, prompting, fine-tuning, base models, future directions.",
        "use": "Use as the living literature map when adding new research rows to this page.",
        "priority": "Reference",
    },
]

TRAINING_PIPELINE_ROWS: list[dict[str, str]] = [
    {
        "stage": "Goal And Metric Contract",
        "basic": "Define task, target model, local eval, official benchmark boundary, and pass/fail metric.",
        "advanced": "Separate seen-DB, unseen-DB, benchmark-dev, and official scores; define promotion gates before training.",
        "repo": "Keep result-equivalence as primary SQL metric and never call local scores official LiveSQLBench scores.",
    },
    {
        "stage": "Data Inventory And Rights",
        "basic": "List raw sources, licenses, splits, and allowed use.",
        "advanced": "Track source hashes, generation prompts, synthetic-data lineage, and protected-split exclusions.",
        "repo": "Public Spider/BIRD train/dev only; hidden or protected benchmark data is excluded from train.",
    },
    {
        "stage": "Data Contracts",
        "basic": "Validate JSONL schema, required fields, dialect, db_id, task_id, SQL target.",
        "advanced": "Add semantic validators: executable SQL, schema references, literal/value checks, token-length budgets.",
        "repo": "Keep strict loaders and fail hard before writing durable datasets.",
    },
    {
        "stage": "Split And Leakage Hygiene",
        "basic": "Train/validation/test split with duplicate question and target checks.",
        "advanced": "DB-disjoint holdouts, SQL AST overlap checks, value/literal overlap audits, synthetic template family splits.",
        "repo": "Use db_id as split unit when measuring generalization.",
    },
    {
        "stage": "Context Construction",
        "basic": "Render question, dialect, schema, optional evidence, and assistant SQL target.",
        "advanced": "Add profile metadata, schema linking, retrieved examples, compressed database docs, and context budget accounting.",
        "repo": "Next high-value lane is SQLite profiling metadata before broader data expansion.",
    },
    {
        "stage": "Tokenization And Loss Masking",
        "basic": "Tokenize with correct chat template and train only assistant completion when desired.",
        "advanced": "Verify masks by unit test, track prompt/completion token histograms, evaluate packing side effects.",
        "repo": "Assistant-SQL-only loss stays explicit in manifests.",
    },
    {
        "stage": "Training Method",
        "basic": "Start with SFT and LoRA on a small model.",
        "advanced": "Compare full fine-tune, LoRA, QLoRA, DoRA, DPO/ORPO/KTO, reward modeling, GRPO only when data supports it.",
        "repo": "Current lane remains direct one-shot SFT; repair/preference/RL wait until one-shot plateaus.",
    },
    {
        "stage": "Runtime Recipe",
        "basic": "Choose batch size, accumulation, learning rate, epochs, precision, seed, and checkpoint cadence.",
        "advanced": "Use Accelerate/FSDP/DeepSpeed for scale, gradient checkpointing, FlashAttention when hardware supports it, and packing only after quality gates.",
        "repo": "RTX 2080 Ti uses non-FlashAttention local runs; cloud is for targeted high-confidence experiments.",
    },
    {
        "stage": "Evaluation",
        "basic": "Run fixed eval sets and compare base vs adapter.",
        "advanced": "Add per-failure taxonomy, cost/latency, inference determinism, candidate diversity, ablations, and confidence intervals.",
        "repo": "Every experiment needs eval JSON plus analysis JSON before deciding next action.",
    },
    {
        "stage": "Observability",
        "basic": "Log params, metrics, artifacts, and model outputs.",
        "advanced": "Track dataset lineage, git commit, source hashes, hardware, package lock, token stats, failure buckets, and qualitative notes.",
        "repo": "MLflow is browser/comparison layer; manifests and JSON artifacts remain durable contract.",
    },
    {
        "stage": "Model Registry And Serving",
        "basic": "Save adapter, tokenizer, config, and train summary.",
        "advanced": "Merge/serve adapters, vLLM batch eval, compatibility tests, rollback plan, and model card.",
        "repo": "Add vLLM after one-shot quality is stable enough to justify larger eval sweeps.",
    },
    {
        "stage": "Feedback And Next Data",
        "basic": "Read failures and add targeted examples.",
        "advanced": "Collect repair rows, preference pairs, negative candidates, SQL-to-text examples, and active-learning queues.",
        "repo": "Failures decide data. Do not add broad rows when one schema/value skill is missing.",
    },
]

FRAMEWORK_ROWS: list[dict[str, str]] = [
    {
        "framework": "Transformers Trainer",
        "source": "https://huggingface.co/docs/transformers/main/trainer",
        "role": "General model loading, training loop, generation, TrainingArguments.",
        "repo_use": "Baseline backend and eval path; keep only if it gives control TRL hides.",
    },
    {
        "framework": "Datasets",
        "source": "https://huggingface.co/docs/datasets/v1.1.3/processing.html",
        "role": "Map/filter/split/shuffle dataset transforms and Arrow-backed scaling.",
        "repo_use": "Useful when DB labs grow beyond small JSONL files.",
    },
    {
        "framework": "PEFT",
        "source": "https://huggingface.co/docs/peft/developer_guides/quantization",
        "role": "LoRA/adapter training and quantized PEFT workflows.",
        "repo_use": "Current adapter substrate; keep LoRA config explicit in manifests.",
    },
    {
        "framework": "TRL SFTTrainer",
        "source": "https://huggingface.co/docs/trl/sft_trainer",
        "role": "SFT-specific trainer with packing, chat templates, assistant/completion loss controls.",
        "repo_use": "Canonical SFT backend for Qwen SQL experiments.",
    },
    {
        "framework": "bitsandbytes / QLoRA",
        "source": "https://arxiv.org/abs/2305.14314",
        "role": "4-bit NF4, double quantization, paged optimizers for memory-efficient tuning.",
        "repo_use": "Add when memory blocks larger model/context experiments, not as a quality fix by itself.",
    },
    {
        "framework": "Accelerate",
        "source": "https://huggingface.co/docs/transformers/accelerate",
        "role": "Launch/device abstraction for single-GPU, multi-GPU, FSDP, and DeepSpeed setups.",
        "repo_use": "Use for cloud/distributed reproducibility after local recipe is stable.",
    },
    {
        "framework": "Axolotl",
        "source": "https://docs.axolotl.ai/",
        "role": "Recipe-driven full fine-tuning, LoRA, QLoRA, DPO/ORPO/KTO, GRPO, RM/PRM.",
        "repo_use": "Mirror best repo recipe later; do not make it the source of truth yet.",
    },
    {
        "framework": "vLLM",
        "source": "https://docs.vllm.ai/en/v0.7.0/serving/openai_compatible_server.html",
        "role": "High-throughput OpenAI-compatible serving and LoRA module serving.",
        "repo_use": "Future eval acceleration and LiveSQLBench serving path.",
    },
    {
        "framework": "MLflow",
        "source": "https://mlflow.org/docs/latest/ml/tracking",
        "role": "Experiment tracking for params, metrics, artifacts, datasets, and run comparison.",
        "repo_use": "Keep dashboard and comparison layer; log manifest/result artifact paths.",
    },
    {
        "framework": "DVC",
        "source": "https://dvc.org/doc/user-guide/project-structure/dvc-files",
        "role": "Git-tracked pointers for large datasets and model artifacts.",
        "repo_use": "Consider once ignored external/results/artifacts need reproducible cross-machine sync.",
    },
]

HYGIENE_ROWS: list[dict[str, str]] = [
    {
        "rule": "One variable per experiment",
        "why": "Without isolation, we cannot tell whether data, prompt, trainer, model, or decoding caused movement.",
        "check": "Manifest diff names the changed variable and fixed variables.",
    },
    {
        "rule": "Dataset lineage is part of the model",
        "why": "A checkpoint without source rows, generation command, and leakage audit is not reproducible.",
        "check": "Manifest points to train/eval datasets; generated data has command or builder code.",
    },
    {
        "rule": "Never score on trained dev rows",
        "why": "Repair rows, SQL-to-text rows, and synthetic examples collected from eval failures contaminate that eval.",
        "check": "Once an eval slice feeds training, retire it for headline scoring.",
    },
    {
        "rule": "Track failure mix, not only pass rate",
        "why": "A flat score can hide useful shifts from wrong-result failures into syntax/schema failures or vice versa.",
        "check": "Every eval has analysis JSON with failure_counts.",
    },
    {
        "rule": "Local lab scores are not benchmark scores",
        "why": "Official benchmarks include hidden data, runner constraints, dialect details, and scoring policies.",
        "check": "Docs and MLflow tags distinguish lab, local, same-DB, unseen-DB, and official.",
    },
    {
        "rule": "Speed optimizations need quality gates",
        "why": "Packing and FlashAttention can improve runtime while damaging completion boundaries or learned behavior.",
        "check": "Runtime improvements promote only if fixed eval gates hold.",
    },
    {
        "rule": "Keep research as operating memory",
        "why": "Papers should change the next experiment or pipeline design, not sit as decorative bibliography.",
        "check": "Every research row has a repo-use decision.",
    },
]


def build_docs_site(output_dir: str | Path = "site") -> DocsSiteSummary:
    """Generate browser docs from structured repo state."""

    output_path = _resolve_output_path(output_dir)
    static_path = output_path / "static"
    experiments_path = output_path / "experiments"
    static_path.mkdir(parents=True, exist_ok=True)
    experiments_path.mkdir(parents=True, exist_ok=True)

    experiments = _load_experiments()
    pages = {
        "index.html": _render_home(experiments),
        "pipeline.html": _render_pipeline(),
        "training.html": _render_training(experiments),
        "learnings.html": _render_learnings(),
        "research.html": _render_research(),
        "runbook.html": _render_runbook(),
        "evaluation.html": _render_evaluation(experiments),
        "livesqlbench.html": _render_livesqlbench(),
        "observability.html": _render_observability(experiments),
        "agent-workflow.html": _render_agent_workflow(),
        "documentation.html": _render_documentation(),
        "tooling.html": _render_tooling(),
        "experiments/index.html": _render_experiments_index(experiments),
    }
    for experiment in experiments:
        pages[f"experiments/{_experiment_slug(experiment)}.html"] = _render_experiment(experiment)

    for relative_path, content in pages.items():
        target = output_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    css_path = static_path / "styles.css"
    css_path.write_text(_styles(), encoding="utf-8")

    return DocsSiteSummary(
        output_dir=output_path,
        page_count=len(pages),
        asset_count=1,
    )


def serve_docs_site(
    output_dir: str | Path = "site",
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> int:
    """Build and serve the docs site until interrupted."""

    summary = build_docs_site(output_dir)
    handler = _handler_for(summary.output_dir)
    with TCPServer((host, port), handler) as httpd:
        print(f"serving SQLBench Lab docs at http://{host}:{port}")
        print(f"site root: {summary.output_dir}")
        httpd.serve_forever()
    return 0


def _resolve_output_path(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path


def _handler_for(directory: Path) -> type[SimpleHTTPRequestHandler]:
    class DocsHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

    return DocsHandler


def _load_experiments() -> list[ExperimentRecord]:
    manifests = sorted((WORKSPACE_ROOT / "experiments/sql").glob("*.json"))
    records = [_load_experiment(path) for path in manifests]
    return sorted(records, key=lambda record: record.number)


def _load_experiment(path: Path) -> ExperimentRecord:
    data = _load_json(path)
    experiment_id = _string(data.get("experiment_id"), path.stem)
    training_method = _mapping(data.get("training_method"))
    student = _mapping(data.get("student"))
    prompt = _mapping(data.get("prompt"))
    train_inputs = _mapping(data.get("train_inputs"))
    trainer = _mapping(data.get("trainer"))
    train_summary_path = _find_train_summary_path(experiment_id, data)
    train_summary = _load_json_optional(train_summary_path)
    metrics = _mapping(train_summary.get("training_metrics")) if train_summary else {}

    return ExperimentRecord(
        experiment_id=experiment_id,
        number=_experiment_number(experiment_id),
        manifest_path=_repo_path(path),
        base_model=_string(student.get("base_model"), "unknown"),
        adapter_name=_string(student.get("adapter_name"), "unknown"),
        method=_string(training_method.get("method"), "unknown"),
        prompt_style=_string(prompt.get("style"), "unknown"),
        stage=_string(training_method.get("stage"), "unknown"),
        notes=_string(training_method.get("notes"), ""),
        train_datasets=[_string(item, "") for item in train_inputs.get("train_datasets", [])],
        backend=_string(trainer.get("backend"), "unknown"),
        epochs=_float_or_none(trainer.get("num_train_epochs")),
        learning_rate=_float_or_none(trainer.get("learning_rate")),
        packing=_bool_or_none(trainer.get("packing")),
        train_rows=_int_or_none(train_summary.get("train_row_count")) if train_summary else None,
        train_loss=_float_or_none(metrics.get("train_loss")),
        train_runtime=_float_or_none(metrics.get("train_runtime")),
        train_summary_path=_repo_path(train_summary_path) if train_summary_path and train_summary_path.exists() else None,
        evals=_load_eval_records(experiment_id),
    )


def _load_eval_records(experiment_id: str) -> list[EvalRecord]:
    result_root = WORKSPACE_ROOT / "results/sql" / experiment_id
    if not result_root.exists():
        return []
    evals: list[EvalRecord] = []
    for path in sorted(result_root.glob("adapter__*.json")):
        if path.name.endswith(".analysis.json"):
            continue
        data = _load_json(path)
        analysis_path = _matching_analysis_path(path)
        analysis = _load_json_optional(analysis_path)
        failure_counts = _mapping(analysis.get("failure_counts")) if analysis else {}
        evals.append(
            EvalRecord(
                label=path.stem.replace("adapter__", ""),
                dataset=_string(data.get("eval_dataset"), "unknown"),
                passed=_int_or_none(data.get("passed_count")) or 0,
                total=_int_or_none(data.get("case_count")) or 0,
                pass_rate=_float_or_none(data.get("pass_rate")) or 0.0,
                analysis_path=_repo_path(analysis_path) if analysis_path and analysis_path.exists() else None,
                failure_counts={str(key): int(value) for key, value in failure_counts.items()},
            )
        )
    return evals


def _matching_analysis_path(result_path: Path) -> Path | None:
    standard_path = result_path.with_suffix(".analysis.json")
    if standard_path.exists():
        return standard_path
    candidates = sorted(result_path.parent.glob("*analysis.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _find_train_summary_path(experiment_id: str, data: dict[str, Any]) -> Path | None:
    output_paths = _mapping(data.get("output_paths"))
    manifest_path = output_paths.get("train_summary_json")
    if manifest_path:
        path = WORKSPACE_ROOT / str(manifest_path)
        if path.exists():
            return path
    path = WORKSPACE_ROOT / "artifacts/sql" / experiment_id / "train_summary.json"
    return path if path.exists() else None


def _render_home(experiments: list[ExperimentRecord]) -> str:
    latest = experiments[-1] if experiments else None
    latest_score = _best_latest_score(latest)
    cards = [
        _metric_card("Tracked experiments", str(len(experiments)), "Manifests under experiments/sql"),
        _metric_card("Latest run", f"Exp{latest.number:03d}" if latest else "None", latest.experiment_id if latest else ""),
        _metric_card("Latest best eval", latest_score, "Local result-equivalence score, not official benchmark"),
        _metric_card("Current mode", "One-shot SFT", "No repair-stage training in the active plan"),
    ]
    body = f"""
        <section class="page-head">
          <p class="eyebrow">SQLBench Lab</p>
          <h1>Dense browser map for training, eval, and competition readiness.</h1>
          <p class="lead">This view organizes the repo by decisions, artifacts, gates, and experiment evidence. Markdown remains agent-readable source context; this site is the working cockpit.</p>
        </section>
        <section class="metrics">{''.join(cards)}</section>
        <section class="grid two">
          <article class="panel">
            <h2>Operating Question</h2>
            <p>Can a small Qwen adapter learn one-shot text-to-SQL on real Spider/BIRD style data, then generalize across held-out databases well enough to justify LiveSQLBench runs?</p>
            <div class="callout">Current evidence says schema naming and value grounding dominate. Exp031 added compact profile metadata, preserved 40/40 on both seen labs, and moved the fixed DB-disjoint restaurant plus airline holdout from 5/50 to 7/50.</div>
          </article>
          <article class="panel">
            <h2>Next Useful Move</h2>
            <ol class="tight">
              <li>Use Exp031 as the new local baseline for the metadata lane: 7/50 on restaurant plus airline.</li>
              <li>For DSPy MIPROv2/GEPA work, restaurant plus airline is prompt-dev; works_cycles plus public_review_platform is the fresh unseen gate.</li>
              <li>Exp033 moves back to training with schema-linking notes instead of more prompt-only search.</li>
              <li>Candidate selection and repair remain separate lanes, not mixed into one-shot SFT scoring.</li>
              <li>Promote only stable one-shot behavior toward LiveSQLBench.</li>
            </ol>
          </article>
        </section>
        <section class="panel">
          <h2>System Map</h2>
          <div class="system-map">
            {_system_node("Data", "Import, validate, audit leakage", "evaluation.html")}
            {_system_node("Training", "Manifest-driven TRL/LoRA SFT", "training.html")}
            {_system_node("Eval", "Result equivalence plus failure taxonomy", "evaluation.html")}
            {_system_node("Observability", "MLflow and artifact trails", "observability.html")}
            {_system_node("LiveSQLBench", "Lite to Full to Large gates", "livesqlbench.html")}
          </div>
        </section>
    """
    return _page("Home", "index", body)


def _render_pipeline() -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td><span class="stage-number">{index:02d}</span></td>
          <td><strong>{_escape(item['stage'])}</strong><p>{_escape(item['job'])}</p></td>
          <td><code>{_escape(item['artifact'])}</code></td>
          <td><code>{_escape(item['command'])}</code></td>
          <td>{_escape(item['risk'])}</td>
        </tr>
        """
        for index, item in enumerate(PIPELINE_STAGES, start=1)
    )
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Pipeline</p>
          <h1>Every stage has an artifact, a command, and a failure mode.</h1>
        </section>
        <section class="panel full">
          <table class="dense-table">
            <thead>
              <tr><th>#</th><th>Stage</th><th>Artifact</th><th>Command</th><th>Risk Controlled</th></tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
    """
    return _page("Pipeline", "pipeline", body)


def _render_training(experiments: list[ExperimentRecord]) -> str:
    rows = "\n".join(_experiment_summary_row(exp) for exp in experiments[-12:])
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Training</p>
          <h1>One-shot text-to-SQL experiments, ordered by intervention.</h1>
          <p class="lead">Use this page to compare what changed, what moved, and what regressed. It intentionally foregrounds eval behavior over training loss.</p>
        </section>
        <section class="panel full">
          <h2>Recent Experiment Ledger</h2>
          <table class="dense-table">
            <thead>
              <tr><th>Run</th><th>Intervention</th><th>Train Rows</th><th>Loss</th><th>Eval Signal</th><th>Open Read</th></tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
        <section class="grid three">
          {_principle_card("Do One Thing Properly", "Active scope is direct one-shot SQL generation. Repair rows can be collected, but repair-stage SFT is deferred.")}
          {_principle_card("Small DB Labs First", "Use one or two DBs to isolate schema/value failures, then expand and measure when transfer starts.")}
          {_principle_card("Holdout Discipline", "As DB count grows, keep unseen DBs untouched by training and report seen-DB and unseen-DB performance separately.")}
        </section>
    """
    return _page("Training", "training", body)


def _render_learnings() -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['phase'])}</strong></td>
          <td>{_escape(row['focus'])}</td>
          <td>{_escape(row['signal'])}</td>
          <td>{_escape(row['lesson'])}</td>
        </tr>
        """
        for row in HISTORY_ROWS
    )
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Learnings</p>
          <h1>Practical fine-tuning lessons from the run history.</h1>
          <p class="lead">This page keeps the learning signal from old planning notes without preserving a long chronological markdown log.</p>
        </section>
        <section class="panel full">
          <table class="dense-table">
            <thead><tr><th>Phase</th><th>Focus</th><th>Signal</th><th>Rule Captured</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>DB Expansion Rule</h2>
            <p>Treat database ID as the scientific split unit. Same-DB dev can share a train DB only when exact task, question, and SQL overlap are absent. Unseen-DB dev must also be DB-disjoint.</p>
          </article>
          <article class="panel">
            <h2>Current Partition</h2>
            <table class="key-table">
              <tr><th>Train labs</th><td>superstore, regional_sales, sales, and bike_share_1.</td></tr>
              <tr><th>Reserve</th><td>restaurant and airline stay unseen until a new reserve is chosen before looking.</td></tr>
              <tr><th>Benchmark</th><td>BIRD validation and LiveSQLBench are measurement only, never training sources.</td></tr>
            </table>
          </article>
        </section>
    """
    return _page("Learnings", "learnings", body)


def _render_research() -> str:
    paper_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['paper'])}</strong><br><a href="{_escape(row['source'])}">source</a></td>
          <td>{_badge(row['priority'])}</td>
          <td>{_escape(row['theme'])}</td>
          <td>{_escape(row['use'])}</td>
        </tr>
        """
        for row in RESEARCH_PAPER_ROWS
    )
    pipeline_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['stage'])}</strong></td>
          <td>{_escape(row['basic'])}</td>
          <td>{_escape(row['advanced'])}</td>
          <td>{_escape(row['repo'])}</td>
        </tr>
        """
        for row in TRAINING_PIPELINE_ROWS
    )
    framework_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['framework'])}</strong><br><a href="{_escape(row['source'])}">source</a></td>
          <td>{_escape(row['role'])}</td>
          <td>{_escape(row['repo_use'])}</td>
        </tr>
        """
        for row in FRAMEWORK_ROWS
    )
    hygiene_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['rule'])}</strong></td>
          <td>{_escape(row['why'])}</td>
          <td>{_escape(row['check'])}</td>
        </tr>
        """
        for row in HYGIENE_ROWS
    )
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Research</p>
          <h1>Text-to-SQL literature and modern fine-tuning pipeline map.</h1>
          <p class="lead">Last researched 2026-05-12. Each source is kept only if it changes a concrete SQLBench Lab design choice, experiment, or hygiene rule.</p>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>Immediate Read</h2>
            <p>The literature points away from blind row scaling and toward database-grounded context: profiling metadata, schema linking, prompt optimization, candidate selection, execution feedback, and strict split hygiene. Exp031 implemented deterministic SQLite profile notes; Exp033 adds schema-linking notes as supervised SFT context. The active gap is making metadata and linking transfer to fresh unseen DBs without leaking gold SQL into eval.</p>
          </article>
          <article class="panel">
            <h2>Research Boundary</h2>
            <p>Agent, repair, reranking, and preference/RL methods are useful future lanes, but they must not pollute the one-shot SFT metric. Promote them only after the direct generator has a stable local baseline and clean artifacts.</p>
          </article>
        </section>
        <section class="panel full">
          <h2>Text-to-SQL Research Map</h2>
          <table class="dense-table">
            <thead><tr><th>Paper / System</th><th>Priority</th><th>Core Idea</th><th>How We Use It</th></tr></thead>
            <tbody>{paper_rows}</tbody>
          </table>
        </section>
        <section class="panel full">
          <h2>Modern Fine-Tuning Pipeline</h2>
          <table class="dense-table">
            <thead><tr><th>Stage</th><th>Basic</th><th>Advanced</th><th>SQLBench Contract</th></tr></thead>
            <tbody>{pipeline_rows}</tbody>
          </table>
        </section>
        <section class="panel full">
          <h2>Framework Stack</h2>
          <table class="dense-table">
            <thead><tr><th>Framework / Method</th><th>Role</th><th>Repo Decision</th></tr></thead>
            <tbody>{framework_rows}</tbody>
          </table>
        </section>
        <section class="panel full">
          <h2>Training Hygiene Rules</h2>
          <table class="dense-table">
            <thead><tr><th>Rule</th><th>Why It Matters</th><th>Check</th></tr></thead>
            <tbody>{hygiene_rows}</tbody>
          </table>
        </section>
    """
    return _page("Research", "research", body)


def _render_runbook() -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['task'])}</strong></td>
          <td><code>{_escape(row['command'])}</code></td>
          <td><code>{_escape(row['output'])}</code></td>
          <td>{_escape(row['gate'])}</td>
        </tr>
        """
        for row in RUNBOOK_ROWS
    )
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Runbook</p>
          <h1>Commands are documented beside the artifact and gate they affect.</h1>
          <p class="lead">Use this as the operational index before running imports, training, eval, analysis, docs, or observability commands.</p>
        </section>
        <section class="panel full">
          <table class="dense-table">
            <thead><tr><th>Task</th><th>Command</th><th>Output</th><th>Gate</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
    """
    return _page("Runbook", "runbook", body)


def _render_evaluation(experiments: list[ExperimentRecord]) -> str:
    latest = experiments[-1] if experiments else None
    failure_rows = _failure_rows(latest) if latest else ""
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Evaluation</p>
          <h1>Result equivalence is the contract; benchmark claims need official tooling.</h1>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>Eval Lanes</h2>
            <table class="key-table">
              <tr><th>Smoke</th><td>Fast sanity checks against known local fixtures.</td></tr>
              <tr><th>DB Lab</th><td>Controlled BIRD train-split DBs such as superstore and regional_sales.</td></tr>
              <tr><th>Unseen DB</th><td>Train/eval DB-disjoint checks as expansion begins.</td></tr>
              <tr><th>Competition</th><td>LiveSQLBench official runner only; local approximations stay labeled local.</td></tr>
            </table>
          </article>
          <article class="panel">
            <h2>Failure Taxonomy</h2>
            <table class="key-table">
              <tr><th>syntax</th><td>SQL cannot parse.</td></tr>
              <tr><th>schema</th><td>Column/table name or join path is invalid.</td></tr>
              <tr><th>runtime</th><td>SQL parses but execution fails.</td></tr>
              <tr><th>wrong result</th><td>Executes but rows differ from gold.</td></tr>
              <tr><th>empty</th><td>No usable SQL was produced.</td></tr>
            </table>
          </article>
        </section>
        <section class="panel full">
          <h2>Latest Failure Readout</h2>
          <table class="dense-table">
            <thead><tr><th>Eval</th><th>Score</th><th>Failures</th><th>Analysis</th></tr></thead>
            <tbody>{failure_rows or '<tr><td colspan="4">No eval artifacts discovered.</td></tr>'}</tbody>
          </table>
        </section>
    """
    return _page("Evaluation", "evaluation", body)


def _render_livesqlbench() -> str:
    body = """
        <section class="page-head compact">
          <p class="eyebrow">LiveSQLBench</p>
          <h1>Compete only after local gates prove one-shot behavior is stable.</h1>
        </section>
        <section class="grid three">
          <article class="panel">
            <h2>Base-Lite</h2>
            <p><strong>18 DBs / 270 tasks.</strong> First official target after local DB expansion and unseen-DB gates.</p>
          </article>
          <article class="panel">
            <h2>Base-Full v1</h2>
            <p><strong>22 new DBs / 600 tasks.</strong> Validates whether the adapter handles fresh DBs, not just extra queries.</p>
          </article>
          <article class="panel">
            <h2>Large-v1</h2>
            <p><strong>18 industrial-scale DBs / 480 tasks.</strong> Requires stronger inference throughput and likely cloud GPU support.</p>
          </article>
        </section>
        <section class="panel full">
          <h2>Promotion Gates</h2>
          <ol class="tight columns">
            <li>Local result-equivalence eval passes on current seen DB labs.</li>
            <li>Unseen-DB validation is reported separately and does not collapse.</li>
            <li>MLflow has train, eval, manifest, and artifact references for the candidate.</li>
            <li>Official runner is used for any public LiveSQLBench number.</li>
            <li>No hidden/protected benchmark data enters train artifacts.</li>
          </ol>
        </section>
    """
    return _page("LiveSQLBench", "livesqlbench", body)


def _render_observability(experiments: list[ExperimentRecord]) -> str:
    latest = experiments[-1] if experiments else None
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Observability</p>
          <h1>MLflow should answer what changed, why, and where the evidence lives.</h1>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>Dashboard</h2>
            <p>Launch the local dashboard:</p>
            <pre><code>uv run python -m sqlbench_lab.cli observe ui</code></pre>
            <p>Training, eval, and prompt-optimization commands must log with <code>--mlflow</code>. Keep run names tied to manifest experiment IDs, and give every optimizer candidate a stable candidate ID.</p>
          </article>
          <article class="panel">
            <h2>Latest Artifact Trail</h2>
            <table class="key-table">
              <tr><th>Experiment</th><td>{_escape(latest.experiment_id if latest else 'none')}</td></tr>
              <tr><th>Manifest</th><td><code>{_escape(latest.manifest_path if latest else 'none')}</code></td></tr>
              <tr><th>Train Summary</th><td><code>{_escape(latest.train_summary_path if latest and latest.train_summary_path else 'not found')}</code></td></tr>
              <tr><th>Eval Count</th><td>{len(latest.evals) if latest else 0}</td></tr>
            </table>
          </article>
        </section>
        <section class="panel full">
          <h2>Minimum Run Tags</h2>
          <div class="tag-cloud">
            <span>experiment_id</span><span>base_model</span><span>adapter_name</span><span>trainer_backend</span>
            <span>prompt_style</span><span>train_dataset</span><span>eval_dataset</span><span>packing</span>
            <span>lora_r</span><span>learning_rate</span><span>seen_db_policy</span><span>unseen_db_policy</span>
            <span>optimizer</span><span>candidate_id</span><span>prompt_dev_dataset</span><span>fresh_gate_dataset</span>
          </div>
        </section>
    """
    return _page("Observability", "observability", body)


def _render_agent_workflow() -> str:
    body = """
        <section class="page-head compact">
          <p class="eyebrow">Agent Workflow</p>
          <h1>How work should move without losing experiment memory.</h1>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>Repo Rules</h2>
            <ul class="tight">
              <li>Build or read the browser docs first when entering the repo.</li>
              <li>Use uv for Python commands.</li>
              <li>Fail hard on invalid state; no silent fallbacks.</li>
              <li>Keep SQL training, eval, and LiveSQLBench adapters explicit.</li>
              <li>Do not train on hidden or protected benchmark data.</li>
            </ul>
          </article>
          <article class="panel">
            <h2>Learning Ledger Policy</h2>
            <ul class="tight">
              <li>Record meaningful experiment lessons in Linear comments.</li>
              <li>Do not create a new learning issue per experiment.</li>
              <li>Create a new subissue only for a reusable lesson or workstream.</li>
              <li>Reference manifest, dataset, result, and analysis paths.</li>
            </ul>
          </article>
        </section>
        <section class="panel full">
          <h2>Remembered Next Plan</h2>
          <p>Exp031 compared Exp030 against the same fixed holdout after adding compact profile metadata to real BIRD rows. The result was 7/50, up from 5/50, with both seen guardrails preserved. Exp032 showed prompt-dev gains that did not transfer cleanly to the fresh gate. Exp033 therefore returns to training with explicit schema-linking notes: train rows use gold-SQL-derived supervision, while eval rows use question/schema/value-note-derived notes only.</p>
          <table class="key-table">
            <tr><th>Paper pattern</th><td>Profile columns, summarize useful value/shape metadata, then use schema linking before candidate selection.</td></tr>
            <tr><th>Repo now</th><td>Raw DDL for real BIRD rows, with hand-authored/profile notes only in regional_sales lab data.</td></tr>
            <tr><th>Next implementation</th><td>Train Exp033 with schema_linking_notes through the existing TRL/LoRA path, then compare prompt-dev, fresh-gate, and seen guardrail scores against Exp031.</td></tr>
          </table>
        </section>
        <section class="panel full">
          <h2>Experiment Closeout Checklist</h2>
          <ol class="tight columns">
            <li>Manifest committed.</li>
            <li>Dataset generation command captured.</li>
            <li>Leakage audit run where relevant.</li>
            <li>SFT summary exists.</li>
            <li>Adapter eval results and analysis exist.</li>
            <li>Prompt optimization candidates are logged individually when Exp032-style loops are used.</li>
            <li>MLflow run contains key tags and artifact paths.</li>
            <li>Linear comment records the lesson, not just the metric.</li>
          </ol>
        </section>
    """
    return _page("Agent Workflow", "agent-workflow", body)


def _render_documentation() -> str:
    body = """
        <section class="page-head compact">
          <p class="eyebrow">Documentation</p>
          <h1>Docs are maintained as structured browser pages, not markdown notes.</h1>
          <p class="lead">The goal is dense operational memory: pages should organize decisions, artifacts, commands, gates, and lessons so a future run can move faster.</p>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>How To Change Docs</h2>
            <ol class="tight">
              <li>Edit <code>src/sqlbench_lab/docs_site/builder.py</code> first.</li>
              <li>Use structured sections, tables, cards, and artifact links instead of prose dumps.</li>
              <li>Add the page to the nav when it is a durable operating surface.</li>
              <li>Add or update <code>tests/test_docs_site.py</code> for required pages and sections.</li>
              <li>Run <code>uv run python -m sqlbench_lab.cli docs build</code>.</li>
              <li>Open <code>site/index.html</code> or serve with <code>uv run python -m sqlbench_lab.cli docs serve</code>.</li>
            </ol>
          </article>
          <article class="panel">
            <h2>What Belongs Here</h2>
            <table class="key-table">
              <tr><th>Pipeline</th><td>Stages, artifacts, commands, and risks.</td></tr>
              <tr><th>Training</th><td>Experiment comparisons and intervention logic.</td></tr>
              <tr><th>Evaluation</th><td>Metrics, failure taxonomy, leakage policy, and gates.</td></tr>
              <tr><th>Competition</th><td>LiveSQLBench target ladder and official-score policy.</td></tr>
              <tr><th>Workflow</th><td>Agent behavior, Linear ledger rules, and closeout checklists.</td></tr>
            </table>
          </article>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>Markdown Policy</h2>
            <p>Do not add markdown docs for repo knowledge, training plans, experiment reads, LiveSQLBench notes, or tooling roadmaps. Markdown is only for tool-mandated files such as <code>AGENTS.md</code> or external issue text.</p>
            <div class="callout">If a doc change matters to the project, it belongs in the browser docs source first. Generated HTML is output; <code>builder.py</code> is the tracked source.</div>
          </article>
          <article class="panel">
            <h2>Shape Rules</h2>
            <ul class="tight">
              <li>Prefer tables for comparisons and gates.</li>
              <li>Prefer cards for independent operating principles.</li>
              <li>Prefer artifact paths over vague references.</li>
              <li>Keep local lab scores distinct from official benchmark scores.</li>
              <li>Keep practical lessons close to experiment evidence.</li>
            </ul>
          </article>
        </section>
        <section class="panel full">
          <h2>Code Contracts Worth Linking</h2>
          <div class="tag-cloud">
            <span>schemas/sql_train_example_v1.schema.json</span>
            <span>schemas/sql_repair_example_v1.schema.json</span>
            <span>schemas/sql_eval_case_v1.schema.json</span>
            <span>schemas/sql_sft_experiment_v1.schema.json</span>
            <span>experiments/sql/*.json</span>
            <span>results/sql/&lt;experiment&gt;/*.json</span>
            <span>artifacts/sql/&lt;experiment&gt;/train_summary.json</span>
          </div>
        </section>
    """
    return _page("Documentation", "documentation", body)


def _render_tooling() -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['tool'])}</strong></td>
          <td>{_badge(row['status'])}</td>
          <td>{_escape(row['value'])}</td>
          <td>{_escape(row['next'])}</td>
        </tr>
        """
        for row in TOOLING_ROWS
    )
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Tooling</p>
          <h1>Use standard training tools where they improve control or throughput.</h1>
        </section>
        <section class="panel full">
          <table class="dense-table">
            <thead><tr><th>Tool</th><th>Status</th><th>Value</th><th>Next Decision</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
    """
    return _page("Tooling", "tooling", body)


def _render_experiments_index(experiments: list[ExperimentRecord]) -> str:
    rows = "\n".join(_experiment_summary_row(exp, nested=True) for exp in experiments)
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Experiments</p>
          <h1>Manifest-backed run index.</h1>
        </section>
        <section class="panel full">
          <table class="dense-table">
            <thead>
              <tr><th>Run</th><th>Intervention</th><th>Train Rows</th><th>Loss</th><th>Eval Signal</th><th>Open Read</th></tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
    """
    return _page("Experiments", "experiments", body, depth=1)


def _render_experiment(experiment: ExperimentRecord) -> str:
    eval_rows = "\n".join(_eval_row(record) for record in experiment.evals)
    dataset_items = "".join(f"<li><code>{_escape(path)}</code></li>" for path in experiment.train_datasets)
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Exp{experiment.number:03d}</p>
          <h1>{_escape(_experiment_title(experiment))}</h1>
          <p class="lead">{_escape(experiment.notes or 'No notes in manifest.')}</p>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>Configuration</h2>
            <table class="key-table">
              <tr><th>Base</th><td>{_escape(experiment.base_model)}</td></tr>
              <tr><th>Adapter</th><td>{_escape(experiment.adapter_name)}</td></tr>
              <tr><th>Method</th><td>{_escape(experiment.method)}</td></tr>
              <tr><th>Stage</th><td>{_escape(experiment.stage)}</td></tr>
              <tr><th>Prompt</th><td>{_escape(experiment.prompt_style)}</td></tr>
              <tr><th>Backend</th><td>{_escape(experiment.backend)}</td></tr>
              <tr><th>Epochs</th><td>{_format_number(experiment.epochs)}</td></tr>
              <tr><th>LR</th><td>{_format_number(experiment.learning_rate)}</td></tr>
              <tr><th>Packing</th><td>{_format_bool(experiment.packing)}</td></tr>
            </table>
          </article>
          <article class="panel">
            <h2>Training</h2>
            <table class="key-table">
              <tr><th>Rows</th><td>{_format_int(experiment.train_rows)}</td></tr>
              <tr><th>Loss</th><td>{_format_number(experiment.train_loss)}</td></tr>
              <tr><th>Runtime</th><td>{_format_seconds(experiment.train_runtime)}</td></tr>
              <tr><th>Manifest</th><td><code>{_escape(experiment.manifest_path)}</code></td></tr>
              <tr><th>Summary</th><td><code>{_escape(experiment.train_summary_path or 'not found')}</code></td></tr>
            </table>
          </article>
        </section>
        <section class="panel full">
          <h2>Train Inputs</h2>
          <ul class="dataset-list">{dataset_items}</ul>
        </section>
        <section class="panel full">
          <h2>Eval Results</h2>
          <table class="dense-table">
            <thead><tr><th>Eval</th><th>Dataset</th><th>Score</th><th>Failures</th><th>Analysis</th></tr></thead>
            <tbody>{eval_rows or '<tr><td colspan="5">No adapter eval artifacts discovered.</td></tr>'}</tbody>
          </table>
        </section>
    """
    return _page(f"Exp{experiment.number:03d}", "experiments", body, depth=1)


def _experiment_summary_row(experiment: ExperimentRecord, *, nested: bool = False) -> str:
    href = f"experiments/{_experiment_slug(experiment)}.html"
    if nested:
        href = f"{_experiment_slug(experiment)}.html"
    return f"""
        <tr>
          <td><a href="{href}">Exp{experiment.number:03d}</a><br><small>{_escape(experiment.experiment_id)}</small></td>
          <td>{_escape(_experiment_title(experiment))}<br><small>{_escape(experiment.backend)} / {_escape(experiment.prompt_style)}</small></td>
          <td>{_format_int(experiment.train_rows)}</td>
          <td>{_format_number(experiment.train_loss)}</td>
          <td>{_eval_badges(experiment)}</td>
          <td>{_open_read(experiment)}</td>
        </tr>
    """


def _experiment_title(experiment: ExperimentRecord) -> str:
    name = experiment.experiment_id
    if "__" in name:
        name = name.split("__", 1)[1]
    name = EXPERIMENT_RE.sub("", name)
    return name.strip("_").replace("_", " ") or experiment.experiment_id


def _eval_badges(experiment: ExperimentRecord) -> str:
    if not experiment.evals:
        return '<span class="muted">no evals</span>'
    return "".join(
        f'<span class="score-pill">{_escape(_short_dataset(record.dataset))}: {record.passed}/{record.total}</span>'
        for record in experiment.evals
    )


def _open_read(experiment: ExperimentRecord) -> str:
    if experiment.number == 29:
        return "Profile notes fixed regional_sales to 40/40 while preserving superstore; runtime cost increased."
    if experiment.number == 28:
        return "Best two-DB lab run; remaining gap is quoted Order Quantity."
    if experiment.number == 27:
        return "Unit price contrast improved target shape but still missed identifier quoting."
    if experiment.number == 26:
        return "Column notes helped grounding, but did not fully fix target shape."
    if experiment.number == 25:
        return "Normalization micro-lab alone was too narrow."
    if experiment.number == 24:
        return "Regional_sales expansion exposed value and column-shape gaps."
    if experiment.number == 23:
        return "Two-DB expansion started transfer measurement."
    if experiment.evals:
        return "Use eval artifacts for readout."
    return "Manifest exists; no local eval artifact discovered."


def _failure_rows(experiment: ExperimentRecord) -> str:
    return "\n".join(
        f"""
        <tr>
          <td>{_escape(record.label)}</td>
          <td>{record.passed}/{record.total} ({record.pass_rate:.1%})</td>
          <td>{_failure_badges(record.failure_counts)}</td>
          <td>{_analysis_cell(record.analysis_path)}</td>
        </tr>
        """
        for record in experiment.evals
    )


def _eval_row(record: EvalRecord) -> str:
    return f"""
        <tr>
          <td>{_escape(record.label)}</td>
          <td><code>{_escape(record.dataset)}</code></td>
          <td>{record.passed}/{record.total} ({record.pass_rate:.1%})</td>
          <td>{_failure_badges(record.failure_counts)}</td>
          <td>{_analysis_cell(record.analysis_path)}</td>
        </tr>
    """


def _failure_badges(failure_counts: dict[str, int]) -> str:
    if not failure_counts:
        return '<span class="muted">none recorded</span>'
    return "".join(
        f'<span class="failure-pill">{_escape(name)}: {count}</span>'
        for name, count in sorted(failure_counts.items())
    )


def _analysis_cell(path: str | None) -> str:
    if not path:
        return '<span class="muted">not found</span>'
    return f"<code>{_escape(path)}</code>"


def _short_dataset(dataset: str) -> str:
    name = Path(dataset).stem
    for token in ("bird_", "_schema_lab", "_dev", "_v1", "_column_notes"):
        name = name.replace(token, "")
    return name or dataset


def _best_latest_score(experiment: ExperimentRecord | None) -> str:
    if not experiment or not experiment.evals:
        return "No eval"
    best = max(experiment.evals, key=lambda record: record.pass_rate)
    return f"{best.passed}/{best.total}"


def _metric_card(label: str, value: str, detail: str) -> str:
    return f"""
        <article class="metric">
          <span>{_escape(label)}</span>
          <strong>{_escape(value)}</strong>
          <p>{_escape(detail)}</p>
        </article>
    """


def _system_node(title: str, subtitle: str, href: str) -> str:
    return f"""
        <a class="system-node" href="{href}">
          <strong>{_escape(title)}</strong>
          <span>{_escape(subtitle)}</span>
        </a>
    """


def _principle_card(title: str, text: str) -> str:
    return f"""
        <article class="panel principle">
          <h2>{_escape(title)}</h2>
          <p>{_escape(text)}</p>
        </article>
    """


def _badge(value: str) -> str:
    class_name = "badge"
    lowered = value.lower()
    if "active" in lowered:
        class_name += " good"
    elif "planned" in lowered or "useful" in lowered:
        class_name += " wait"
    elif "blocked" in lowered:
        class_name += " bad"
    return f'<span class="{class_name}">{_escape(value)}</span>'


def _page(title: str, active: str, body: str, *, depth: int = 0) -> str:
    prefix = "../" if depth else ""
    nav = _nav(active, prefix)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)} | SQLBench Lab</title>
  <link rel="stylesheet" href="{prefix}static/styles.css">
</head>
<body>
  <aside class="sidebar">
    <a class="brand" href="{prefix}index.html">
      <span class="brand-mark">SQL</span>
      <span><strong>SQLBench Lab</strong><small>Training cockpit</small></span>
    </a>
    {nav}
  </aside>
  <main class="content">
    {body}
  </main>
</body>
</html>
"""


def _nav(active: str, prefix: str) -> str:
    items = [
        ("index", "Home", "index.html"),
        ("pipeline", "Pipeline", "pipeline.html"),
        ("training", "Training", "training.html"),
        ("learnings", "Learnings", "learnings.html"),
        ("research", "Research", "research.html"),
        ("runbook", "Runbook", "runbook.html"),
        ("evaluation", "Evaluation", "evaluation.html"),
        ("livesqlbench", "LiveSQLBench", "livesqlbench.html"),
        ("observability", "Observability", "observability.html"),
        ("tooling", "Tooling", "tooling.html"),
        ("agent-workflow", "Agent Workflow", "agent-workflow.html"),
        ("documentation", "Documentation", "documentation.html"),
        ("experiments", "Experiments", "experiments/index.html"),
    ]
    links = "\n".join(
        f'<a class="{"active" if key == active else ""}" href="{prefix}{href}">{label}</a>'
        for key, label, href in items
    )
    return f'<nav class="nav">{links}</nav>'


def _styles() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f7f4;
  --panel: #ffffff;
  --ink: #1e2428;
  --muted: #647078;
  --line: #d8ddd9;
  --accent: #1f7a68;
  --accent-dark: #15584c;
  --amber: #9a6500;
  --red: #a13c3c;
  --blue: #27638f;
  --code: #f0f3f1;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
a { color: var(--accent-dark); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre {
  background: var(--code);
  border-radius: 6px;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 0.87rem;
}
code { padding: 0.1rem 0.3rem; }
pre { overflow-x: auto; padding: 0.9rem; }
.sidebar {
  position: fixed;
  inset: 0 auto 0 0;
  width: 248px;
  background: #e9ede8;
  border-right: 1px solid var(--line);
  padding: 18px 14px;
  overflow-y: auto;
}
.brand {
  display: grid;
  grid-template-columns: 42px 1fr;
  gap: 10px;
  align-items: center;
  color: var(--ink);
  margin-bottom: 18px;
}
.brand-mark {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 8px;
  background: var(--accent);
  color: white;
  font-weight: 800;
  letter-spacing: 0;
}
.brand small {
  display: block;
  color: var(--muted);
  margin-top: 1px;
}
.nav { display: grid; gap: 4px; }
.nav a {
  color: var(--ink);
  padding: 8px 10px;
  border-radius: 7px;
  font-size: 0.94rem;
}
.nav a.active, .nav a:hover {
  background: #d9e5df;
  text-decoration: none;
}
.content {
  margin-left: 248px;
  padding: 28px;
  max-width: 1440px;
}
.page-head {
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--line);
}
.page-head.compact h1 { max-width: 980px; }
.eyebrow {
  margin: 0 0 4px;
  color: var(--accent-dark);
  text-transform: uppercase;
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
}
h1, h2 { letter-spacing: 0; }
h1 {
  margin: 0;
  font-size: clamp(2rem, 4vw, 4rem);
  line-height: 1.02;
}
h2 {
  margin: 0 0 10px;
  font-size: 1.04rem;
}
.lead {
  max-width: 980px;
  color: var(--muted);
  font-size: 1.06rem;
  margin: 12px 0 0;
}
.grid { display: grid; gap: 14px; margin: 14px 0; }
.grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.panel, .metric {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
.panel.full { margin: 14px 0; overflow-x: auto; }
.panel p { margin: 0 0 10px; color: var(--muted); }
.panel p:last-child { margin-bottom: 0; }
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}
.metric span { color: var(--muted); font-size: 0.82rem; }
.metric strong {
  display: block;
  margin-top: 3px;
  font-size: 1.45rem;
  line-height: 1.1;
}
.metric p { margin: 8px 0 0; color: var(--muted); font-size: 0.9rem; }
.callout {
  margin-top: 12px;
  border-left: 4px solid var(--accent);
  background: #edf5f1;
  padding: 10px 12px;
  border-radius: 0 6px 6px 0;
}
.system-map {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
}
.system-node {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  color: var(--ink);
  background: #fbfcfb;
}
.system-node strong, .system-node span { display: block; }
.system-node span { color: var(--muted); font-size: 0.88rem; margin-top: 4px; }
.dense-table, .key-table {
  width: 100%;
  border-collapse: collapse;
}
.dense-table th, .dense-table td,
.key-table th, .key-table td {
  border-bottom: 1px solid var(--line);
  padding: 9px 8px;
  text-align: left;
  vertical-align: top;
}
.dense-table th, .key-table th {
  color: #425058;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.dense-table td { font-size: 0.91rem; }
.dense-table td p { margin: 4px 0 0; color: var(--muted); }
.dense-table small { color: var(--muted); }
.key-table th { width: 140px; }
.stage-number {
  display: inline-grid;
  place-items: center;
  width: 32px;
  height: 28px;
  border-radius: 7px;
  background: #dfe8e3;
  color: var(--accent-dark);
  font-weight: 800;
}
.score-pill, .failure-pill, .badge, .tag-cloud span {
  display: inline-block;
  margin: 2px 4px 2px 0;
  padding: 4px 7px;
  border-radius: 999px;
  font-size: 0.78rem;
  white-space: nowrap;
}
.score-pill { background: #e7f2ed; color: var(--accent-dark); }
.failure-pill { background: #f4e8e8; color: var(--red); }
.badge { background: #e8edf1; color: var(--blue); font-weight: 700; }
.badge.good { background: #e2f1ea; color: var(--accent-dark); }
.badge.wait { background: #fff1d6; color: var(--amber); }
.badge.bad { background: #f4e3e3; color: var(--red); }
.muted { color: var(--muted); }
.tight { margin: 0; padding-left: 1.2rem; }
.tight li { margin-bottom: 5px; }
.columns {
  columns: 2;
  column-gap: 28px;
}
.dataset-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 18px;
  margin: 0;
  padding-left: 1.2rem;
}
.tag-cloud { display: flex; flex-wrap: wrap; gap: 5px; }
.tag-cloud span { background: #eef1ef; }
@media (max-width: 980px) {
  .sidebar {
    position: static;
    width: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .content { margin-left: 0; padding: 18px; }
  .metrics, .grid.two, .grid.three, .system-map, .dataset-list {
    grid-template-columns: 1fr;
  }
  .columns { columns: 1; }
}
"""


def _experiment_slug(experiment: ExperimentRecord) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", experiment.experiment_id)


def _experiment_number(experiment_id: str) -> int:
    match = EXPERIMENT_RE.search(experiment_id)
    return int(match.group("number")) if match else 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_int(value: int | None) -> str:
    return "unknown" if value is None else f"{value:,}"


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "true" if value else "false"


def _format_number(value: float | None) -> str:
    if value is None:
        return "unknown"
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    minutes = value / 60
    return f"{minutes:.1f} min"
