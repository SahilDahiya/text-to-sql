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
    initial_adapter_dir: str | None
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
    {
        "phase": "Exp034 setup",
        "focus": "Run the schema-linking SFT hypothesis with a compact context budget that can finish locally.",
        "signal": "Exp033 full notes were too slow on the RTX 2080 Ti. Exp034 caps schema-linking notes to two tables and six columns, keeps DB-disjoint holdouts, and starts with one epoch.",
        "lesson": "Training metadata must be useful and cheap enough to iterate. First prove compact schema-linking transfers on the fixed unseen gates, then decide whether longer notes or more epochs are worth the compute.",
    },
    {
        "phase": "Exp034 result",
        "focus": "Evaluate compact schema-linking SFT as a one-shot training improvement.",
        "signal": "One epoch trained 188 rows in 35.6 minutes with train_loss 0.2323. Prompt-dev improved to 10/50, but the fresh unseen gate stayed 7/50. Seen guardrails were 29/40 superstore and 28/40 regional-sales.",
        "lesson": "Reject Exp034 as a promoted checkpoint. Compact schema-linking helps the tuned holdout but does not transfer; the long-tail runtime also shows the pipeline needs token-length reporting and an explicit max-length policy before larger SFT runs.",
    },
    {
        "phase": "Blacksmith reset",
        "focus": "Step back from tiny one-shot SFT tweaks after Exp034 failed to transfer.",
        "signal": "Recent systems that move BIRD/Spider/LiveSQLBench use candidate pools, execution, selection, broad data, metadata retrieval, or agents; they are not just LoRA runs on a few hundred rows.",
        "lesson": "Keep the small-model constraint. Do blacksmith work next: token-length instrumentation, candidate-pool execution/selection, and larger small-model training data before another goldsmith SFT variant.",
    },
    {
        "phase": "Exp036 setup",
        "focus": "Try the broad-data blacksmith move while staying model-path only.",
        "signal": "Imported 8,364 BIRD train rows after excluding restaurant, airline, works_cycles, and public_review_platform. Profile metadata and compact train-only schema linking were attached. Token report showed raw p50 1,642 and max 193,754, so the train file was filtered to 3,536 rows that fit 1,536 rendered tokens. The first long run died before writing a checkpoint, so Exp036 now saves every 100 steps with auto-resume enabled.",
        "lesson": "Broad data must still obey DB-level holdout, token-budget hygiene, and checkpoint hygiene. Exp036 tests scale plus metadata without repair, agent loops, or hidden benchmark data.",
    },
    {
        "phase": "Exp037 and Exp038 setup",
        "focus": "Test whether Qwen2.5-Coder changes the one-shot unseen-DB ceiling without changing the data boundary.",
        "signal": "Exp037 uses Qwen2.5-Coder-1.5B and Exp038 uses Qwen2.5-Coder-3B. Both keep the Exp036 train file, DB-level holdouts, LoRA target modules, 1,536-token budget, and eval gates fixed.",
        "lesson": "Model-family comparisons must change one thing at a time. First measure Exp036 as the Qwen3.5-0.8B baseline, then promote Qwen2.5-Coder only if the same prompt-dev and fresh unseen gates move.",
    },
    {
        "phase": "Exp039-040",
        "focus": "Build a controlled storefront single-DB lab and compare base-model SFT with a SQL-adapter warm start.",
        "signal": "Base SFT improved dev but failed most held-out eval cases. Warm-starting from the Exp031 SQL/profile adapter reached 11/12 dev and 4/12 eval, with remaining failures dominated by table ownership and join-path errors.",
        "lesson": "A prior SQL adapter helps, but a tiny single-DB train split can overfit dev-shaped tasks. The next storefront run should expand train coverage around failure classes while keeping dev/eval frozen.",
    },
    {
        "phase": "Exp041",
        "focus": "Create storefront train_v2 without touching held-out dev/eval, then run a one-epoch warm-start SFT.",
        "signal": "Train_v2 expands from 40 to 97 rows with targeted examples for column ownership, multi-hop joins, anti-joins, HAVING, ratios, shipments, returns, and support tickets. Leakage and SQL execution checks passed, but the one-epoch adapter regressed to 6/12 dev and 3/12 held-out eval.",
        "lesson": "Clean comprehensive data is necessary but not sufficient. One epoch reduced overfit pressure but underfit the storefront mapping; reject this checkpoint and tune the schedule or continue from the stronger Exp040 storefront adapter before claiming improvement.",
    },
    {
        "phase": "Exp042",
        "focus": "Continue from the stronger Exp040 storefront adapter on train_v2 with a lower learning rate.",
        "signal": "The correction run trained 97 rows for one epoch at 1e-4 LR and preserved most dev behavior at 10/12, but frozen held-out eval fell to 3/12. Eval failures remained alias/table ownership heavy: six schema errors plus three result mismatches.",
        "lesson": "Continuation SFT on more same-DB examples still did not teach robust alias ownership. Stop adding same-representation rows as the primary move; the next artifact should add explicit schema-linking or execution-checked candidate selection.",
    },
    {
        "phase": "Exp043-046",
        "focus": "Tune LoRA adapter shape on the frozen storefront train_v2/dev/eval setup.",
        "signal": "The r8/alpha16 baseline reached 9/12 dev and 4/12 held-out eval. Moving to r16/alpha32 reached 11/12 dev and 5/12 eval. Raising dropout to 0.10 kept dev at 11/12 and improved held-out eval to 6/12.",
        "lesson": "LoRA capacity and dropout matter for this one DB, and Exp046 is the best storefront adapter so far. The remaining failures are still alias/table ownership, anti-join syntax, and missing HAVING behavior, so the next improvement should target those composition families or add execution-checked selection.",
    },
    {
        "phase": "Exp048",
        "focus": "Improve the storefront train data without training on frozen dev/eval answers.",
        "signal": "Train_v3 keeps the 97 train_v2 rows and adds 33 execution-checked near-neighbor rows for grouped ranking, item-row return ratios, anti-join lists, global HAVING, shipment-return joins, and explicit alias ownership. With the Exp046 r16/alpha32/dropout0.10/bias-none recipe fixed, Exp048 reached 10/12 dev and 10/12 frozen held-out eval; Exp046 was 11/12 dev and 6/12 held-out eval.",
        "lesson": "Targeted composition-family data can improve same-DB held-out behavior without copying held-out answers. Promote Exp048 over Exp046 for this one-DB lab, but keep investigating the remaining alias ownership miss on returns.return_id and the unresolved-ticket overfilter that invented issue_type = 'support'.",
    },
    {
        "phase": "Exp048 vLLM serving",
        "focus": "Serve the promoted storefront adapter through an OpenAI-compatible vLLM endpoint.",
        "signal": "vLLM 0.22.1 can start locally on the WSL RTX 2080 Ti with gpu_memory_utilization=0.75, eager mode, FlashInfer sampler disabled, FlashInfer autotune disabled, and TRITON_ATTN. The frozen held-out endpoint eval completed but scored 9/12, below the local HF 10/12. Stress probes with max_new_tokens=128 succeeded through c160: c8 was 32/32 at 0.5826 rps with p95 18.3s, c64 was 64/64 at 1.8890 rps with p95 32.7s, and c160 was 160/160 at 2.3994 rps with p95 63.4s.",
        "lesson": "Serving is mechanically wired but not promoted. The extra vLLM miss is a deterministic alias ownership slip in eval_002, while eval_003 and eval_006 match the local HF failures. Treat vLLM backend parity as an eval gate, not a deployment afterthought. For this local GPU, admitted concurrency is not the same as useful interactive concurrency; use c8-c16 for interactive probes and c64+ only for batch/background work.",
    },
    {
        "phase": "Exp049 QLoRA",
        "focus": "Repeat Exp048 with QLoRA while keeping data, prompt, LoRA shape, trainer schedule, and eval gates fixed.",
        "signal": "The QLoRA path loads the base model with bitsandbytes 4-bit NF4, double quantization, bfloat16 compute intent, device_map=auto, and PEFT prepare_model_for_kbit_training. Training completed in 1,379s over 390 steps with final train_loss 0.08428 and wrote a normal PEFT adapter. Local HF scored 9/12 dev and 9/12 frozen held-out eval. vLLM endpoint eval on port 8003 also scored 9/12; c8 load was 32/32 at 0.6515 rps with p50 10.21s and p95 17.17s.",
        "lesson": "QLoRA is now a working training path, but it is not a quality upgrade for this 0.8B one-DB lab. It saves training memory and serves like a normal LoRA adapter, but it regressed below Exp048 local HF quality. Use QLoRA when memory unlocks a larger model or longer context, not as a replacement for the current Exp048 adapter.",
    },
    {
        "phase": "Exp048 vs Exp049 stress",
        "focus": "Run fresh-server LoRA and QLoRA vLLM concurrency ladders with the same frozen eval prompts and request counts.",
        "signal": "Both adapters served cleanly through c32 with zero request failures. LoRA c32 handled 64/64 at 1.3195 rps with p50 18.60s and p95 30.11s. QLoRA c32 handled 64/64 at 1.4715 rps with p50 15.04s and p95 25.64s. Both runs reported the same 230,912-token GPU KV cache and 150.33x theoretical concurrency at 1,536 tokens/request.",
        "lesson": "QLoRA has no observed serving stress penalty in this setup because inference still serves a normal PEFT adapter on the same base model. The quality gate, not concurrency, remains the blocker for Exp049.",
    },
    {
        "phase": "Exp050-057 one-DB ladder",
        "focus": "Break the next storefront work into controlled same-DB experiments instead of one undiagnosable data expansion.",
        "signal": "Exp050 showed Exp048 at 15/24 on challenge_v1. Exp051-Exp055 each added 14 rows over train_v3 for one failure family: support no-issue filters, date boundary semantics, return-ratio denominator and alias ownership, grouped HAVING counts, and anti-join or missing-relationship queries. The best isolated ablations reached 17/24 challenge, but moved failures between schema validity and semantics. Exp056 combined the supplements into train_v4 with 200 rows and reached 11/12 dev_v2, 12/12 eval_v1, and 22/24 challenge_v1, with only schema-error failures on challenge. Exp057 repeated train_v4 with QLoRA, ran faster with fewer loaded total parameters, reached 12/12 dev_v2 and 22/24 challenge_v1, but regressed eval_v1 to 10/12. Train_v4 has no exact question or gold-SQL overlap with dev_v1, dev_v2, eval_v1, or challenge_v1.",
        "lesson": "For a production single-DB text-to-SQL model, dataset diversification should be inside the database semantics: same schema, new literals, new phrasings, and isolated failure families. The isolated ablations diagnosed useful row families, but Exp056 proved the families compose and is the promoted LoRA checkpoint. Exp057 keeps QLoRA as a credible memory/runtime tradeoff, not the preferred quality checkpoint, because stable eval parity matters more than train loss, runtime, or one targeted gate.",
    },
    {
        "phase": "Exp058-063 contrast ladder",
        "focus": "Turn the next one-DB improvement into hard-negative data and richer eval slices rather than another broad expansion.",
        "signal": "Exp058 added per-tag eval slicing and showed Exp056 failures concentrated in join_path, return_ratio, and grouped_ranking slices despite 22/24 challenge_v1. Exp063 introduced challenge_v2 with alias-ownership, boundary, and anti-join contrast cases; the promoted Exp056 adapter scored 8/15, with anti_join/left_join_predicate at 1/5, boundary_semantics at 4/6, and alias_ownership at 3/4. The train ladder kept the Exp056 recipe fixed: Exp059 added 12 alias contrast rows and reached 9/15 challenge_v2 while preserving 12/12 eval_v1; Exp060 and Exp061 improved targeted slices but regressed eval_v1 to 11/12. Exp062 combined all three into train_v5 with 236 rows and reached 12/15 challenge_v2, but also regressed eval_v1 to 11/12.",
        "lesson": "Same-DB diversification should include adversarial near-neighbor examples and sliced gates, but promotion still requires a clean protected eval gate. Aggregate same-DB scores can hide that the model memorized common shapes but still confuses predicate ownership, inclusive/exclusive operators, anti-join semantics, and SQL alias validity. Exp062 is a rejected ablation, not a promoted replacement for Exp056.",
    },
]

SERVING_STRESS_ROWS: list[dict[str, str]] = [
    {"concurrency": "1", "requests": "4", "success": "4/4", "rps": "0.0784", "p50": "14.41s", "p95": "16.98s", "max": "16.98s"},
    {"concurrency": "2", "requests": "8", "success": "8/8", "rps": "0.1756", "p50": "11.43s", "p95": "15.05s", "max": "15.05s"},
    {"concurrency": "4", "requests": "16", "success": "16/16", "rps": "0.3003", "p50": "12.36s", "p95": "17.69s", "max": "18.71s"},
    {"concurrency": "8", "requests": "32", "success": "32/32", "rps": "0.5826", "p50": "11.42s", "p95": "18.33s", "max": "19.36s"},
    {"concurrency": "16", "requests": "32", "success": "32/32", "rps": "0.6612", "p50": "21.12s", "p95": "34.93s", "max": "36.08s"},
    {"concurrency": "32", "requests": "32", "success": "32/32", "rps": "1.4934", "p50": "14.76s", "p95": "20.50s", "max": "21.25s"},
    {"concurrency": "64", "requests": "64", "success": "64/64", "rps": "1.8890", "p50": "24.36s", "p95": "32.70s", "max": "33.81s"},
    {"concurrency": "128", "requests": "128", "success": "128/128", "rps": "2.2966", "p50": "43.22s", "p95": "53.25s", "max": "55.16s"},
    {"concurrency": "160", "requests": "160", "success": "160/160", "rps": "2.3994", "p50": "53.90s", "p95": "63.44s", "max": "65.91s"},
]

LORA_QLORA_STRESS_ROWS: list[dict[str, str]] = [
    {"adapter": "Exp048 LoRA", "concurrency": "1", "requests": "8", "success": "8/8", "rps": "0.0818", "p50": "12.74s", "p95": "17.54s", "avg_chars": "231.9"},
    {"adapter": "Exp049 QLoRA", "concurrency": "1", "requests": "8", "success": "8/8", "rps": "0.0898", "p50": "9.79s", "p95": "19.95s", "avg_chars": "231.2"},
    {"adapter": "Exp048 LoRA", "concurrency": "2", "requests": "12", "success": "12/12", "rps": "0.1414", "p50": "14.06s", "p95": "20.60s", "avg_chars": "231.2"},
    {"adapter": "Exp049 QLoRA", "concurrency": "2", "requests": "12", "success": "12/12", "rps": "0.1912", "p50": "9.85s", "p95": "15.96s", "avg_chars": "226.8"},
    {"adapter": "Exp048 LoRA", "concurrency": "4", "requests": "16", "success": "16/16", "rps": "0.2849", "p50": "12.55s", "p95": "17.65s", "avg_chars": "242.1"},
    {"adapter": "Exp049 QLoRA", "concurrency": "4", "requests": "16", "success": "16/16", "rps": "0.3040", "p50": "9.86s", "p95": "17.48s", "avg_chars": "242.6"},
    {"adapter": "Exp048 LoRA", "concurrency": "8", "requests": "32", "success": "32/32", "rps": "0.5324", "p50": "11.85s", "p95": "20.14s", "avg_chars": "231.3"},
    {"adapter": "Exp049 QLoRA", "concurrency": "8", "requests": "32", "success": "32/32", "rps": "0.6761", "p50": "9.78s", "p95": "16.70s", "avg_chars": "227.9"},
    {"adapter": "Exp048 LoRA", "concurrency": "16", "requests": "48", "success": "48/48", "rps": "0.9600", "p50": "13.68s", "p95": "22.23s", "avg_chars": "231.2"},
    {"adapter": "Exp049 QLoRA", "concurrency": "16", "requests": "48", "success": "48/48", "rps": "1.0555", "p50": "11.04s", "p95": "20.02s", "avg_chars": "226.8"},
    {"adapter": "Exp048 LoRA", "concurrency": "32", "requests": "64", "success": "64/64", "rps": "1.3195", "p50": "18.60s", "p95": "30.11s", "avg_chars": "233.9"},
    {"adapter": "Exp049 QLoRA", "concurrency": "32", "requests": "64", "success": "64/64", "rps": "1.4715", "p50": "15.04s", "p95": "25.64s", "avg_chars": "230.8"},
]

GPT51_COST_ROWS: list[dict[str, str]] = [
    {
        "scenario": "Lean SQL",
        "tokens": "800 input + 300 output/reasoning",
        "cost_10k_week": "$40",
        "cost_10k_month": "$173",
        "cost_100k_week": "$400",
        "cost_100k_month": "$1,732",
    },
    {
        "scenario": "Typical SQL",
        "tokens": "1,500 input + 600 output/reasoning",
        "cost_10k_week": "$78.75",
        "cost_10k_month": "$341",
        "cost_100k_week": "$787.50",
        "cost_100k_month": "$3,410",
    },
    {
        "scenario": "Heavy schema/reasoning",
        "tokens": "3,000 input + 1,000 output/reasoning",
        "cost_10k_week": "$137.50",
        "cost_10k_month": "$595",
        "cost_100k_week": "$1,375",
        "cost_100k_month": "$5,954",
    },
    {
        "scenario": "Typical with 70% input cached",
        "tokens": "1,500 input + 600 output/reasoning",
        "cost_10k_week": "$66.94",
        "cost_10k_month": "$290",
        "cost_100k_week": "$669",
        "cost_100k_month": "$2,898",
    },
    {
        "scenario": "Heavy with 70% input cached",
        "tokens": "3,000 input + 1,000 output/reasoning",
        "cost_10k_week": "$113.88",
        "cost_10k_month": "$493",
        "cost_100k_week": "$1,139",
        "cost_100k_month": "$4,931",
    },
]

SELF_HOST_COST_ROWS: list[dict[str, str]] = [
    {"setup": "RunPod L4 24GB, 1 replica, 24/7", "weekly": "$65.52", "monthly": "$284", "read": "Cheapest simple always-on option; provider availability varies."},
    {"setup": "RunPod L4 24GB, 2 replicas, 24/7", "weekly": "$131.04", "monthly": "$568", "read": "Better deploy safety and peak latency headroom."},
    {"setup": "AWS g6.xlarge L4, 1 replica, 24/7", "weekly": "~$134.40", "monthly": "~$582", "read": "More managed cloud posture at higher cost."},
    {"setup": "AWS g6.xlarge L4, 2 replicas, 24/7", "weekly": "~$268.80", "monthly": "~$1,164", "read": "Production-style redundancy, still cheaper than GPT-5.1 at 100k/week."},
    {"setup": "RunPod L4 peak-only 30h/week", "weekly": "$11.70", "monthly": "$51", "read": "Only acceptable when cold starts and scheduled windows are fine."},
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
        "gate": "Manifest validated, MLflow run logged, and long runs use checkpoint saves.",
    },
    {
        "task": "Evaluate adapter",
        "command": "uv run --group training --group observability python -m sqlbench_lab.cli sql eval --manifest experiments/sql/<experiment>.json --model adapter --dataset datasets/sql/eval/<eval>.jsonl --mlflow",
        "output": "results/sql/<experiment>/adapter__*.json",
        "gate": "Result-equivalence score recorded as local, not official.",
    },
    {
        "task": "Prepare local vLLM runtime env",
        "command": "uv python install 3.12 && export CPATH=$HOME/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12 CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13 PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0",
        "output": "Python headers and bundled CUDA compiler are visible to vLLM JIT paths.",
        "gate": "Required on the local WSL RTX 2080 Ti box; cloud hosts with aligned system CUDA may not need these exports.",
    },
    {
        "task": "Print vLLM serve command",
        "command": "uv run python -m sqlbench_lab.cli sql vllm-serve-command --manifest experiments/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010.json --model adapter --port 8001 --served-model-name storefront-sql --lora-name storefront-sql --max-model-len 1536 --gpu-memory-utilization 0.75 --enforce-eager --no-enable-flashinfer-autotune --attention-backend TRITON_ATTN",
        "output": "vllm serve Qwen/Qwen3.5-0.8B-Base ... --attention-backend TRITON_ATTN --enable-lora ...",
        "gate": "Serve base plus the Exp048 LoRA adapter; do not merge or quantize until endpoint eval passes.",
    },
    {
        "task": "Evaluate vLLM endpoint",
        "command": "uv run python -m sqlbench_lab.cli sql eval --manifest experiments/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010.json --model adapter --dataset datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl --openai-base-url http://127.0.0.1:8001 --openai-model storefront-sql --result-label vllm_eval",
        "output": "results/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010/adapter__storefront_sales_lab_eval_v1__vllm_eval.json",
        "gate": "Must match or beat the local Exp048 held-out gate of 10/12 before treating vLLM serving as promoted.",
    },
    {
        "task": "Probe endpoint concurrency",
        "command": "uv run python -m sqlbench_lab.cli sql openai-load-test --manifest experiments/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010.json --model adapter --dataset datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl --openai-base-url http://127.0.0.1:8001 --openai-model storefront-sql --requests 32 --concurrency 8 --output artifacts/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010/vllm_load_c8.json",
        "output": "Latency/RPS JSON",
        "gate": "Record p50, p95, success count, and request rate before increasing concurrency.",
    },
    {
        "task": "Report token lengths",
        "command": "uv run --group training python -m sqlbench_lab.cli sql token-report --manifest experiments/sql/<experiment>.json --dataset datasets/sql/eval/<eval>.jsonl --output artifacts/sql/<experiment>/token_report.json",
        "output": "Prompt token p50/p90/p95/max",
        "gate": "Long-tail rows are known before training or candidate generation.",
    },
    {
        "task": "Filter train token budget",
        "command": "uv run --group training python -m sqlbench_lab.cli sql filter-train-by-token-budget --input datasets/sql/train/<train>.jsonl --output datasets/sql/train/<train>_token1536.jsonl --base-model Qwen/Qwen3.5-0.8B-Base --prompt-style canonical_chat --max-tokens 1536",
        "output": "Budget-clean train JSONL",
        "gate": "No silent trainer truncation becomes the actual experiment.",
    },
    {
        "task": "Evaluate candidate pool",
        "command": "uv run --group training --group observability python -m sqlbench_lab.cli sql eval-candidates --manifest experiments/sql/<experiment>.json --model adapter --dataset datasets/sql/eval/<eval>.jsonl --candidates 5 --result-label <label> --mlflow",
        "output": "results/sql/<experiment>/candidates__*.json",
        "gate": "Compare first@1, pass@N, and selected@1 before deciding whether generation or selection is the bottleneck.",
    },
    {
        "task": "Evaluate execution-guided repair",
        "command": "uv run --group training python -m sqlbench_lab.cli sql eval-repair --manifest experiments/sql/<experiment>.json --model adapter --dataset datasets/sql/eval/<eval>.jsonl --max-repair-attempts 1",
        "output": "results/sql/<experiment>/repair__adapter__<eval>.json",
        "gate": "Compare first_pass and final_pass without rewriting the one-shot score.",
    },
    {
        "task": "Step SQL environment",
        "command": "cd separate_projects/db_sql_agent_env && uv run python -m db_sql_agent_env.cli env-step --dataset ../../datasets/sql/eval/<eval>.jsonl --case-id <case_id> --sql \"SELECT ...\"",
        "output": "Structured JSON with validation, execution, evaluation, reward, and repair observation fields.",
        "gate": "Read-only execution returns syntax/schema/execution feedback without mutating the database.",
    },
    {
        "task": "Run extracted SQL agent env",
        "command": "cd separate_projects/db_sql_agent_env && uv run python -m pytest",
        "output": "Standalone env-step project test results.",
        "gate": "The future separate project stays dependency-light and does not import sqlbench_lab.",
    },
    {
        "task": "Import standalone seed data",
        "command": "cd separate_projects/db_sql_agent_env && uv run python -m db_sql_agent_env.cli import-seed --train ../../datasets/sql/train/storefront_sales_lab_train_v4.jsonl --eval ../../datasets/sql/eval/storefront_sales_lab_challenge_v2.jsonl --output data/storefront_seed",
        "output": "train.jsonl, eval.jsonl, and dataset_summary.json in the seed directory.",
        "gate": "Fails on exact train/eval question or SQL overlap unless --allow-overlap is explicit.",
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
        "use": "Use as a later reference for selective metadata retrieval. Do not repeat static metadata stuffing; require an ablation that proves retrieved context helps a frozen small model.",
        "priority": "Near",
    },
    {
        "paper": "The Death of Schema Linking?",
        "source": "https://openreview.net/pdf?id=fglyh5pa7d",
        "theme": "Ablates schema linking, augmentation, selection, and correction on BIRD.",
        "use": "Stop expecting schema-linking alone to move the needle. Treat augmentation, candidate selection, and correction as the bigger system levers.",
        "priority": "Now",
    },
    {
        "paper": "DPC",
        "source": "https://openreview.net/pdf/41f5d29ae18adc86a2ab7c9a7f109b6a6a41c3e2.pdf",
        "theme": "Training-free candidate selection over multiple SQL candidates.",
        "use": "Blacksmith move: generate N SQL candidates, execute/filter them, then select by result/logic consistency instead of trusting one generation.",
        "priority": "Now",
    },
    {
        "paper": "ContextualAI BIRD SQL pipeline",
        "source": "https://github.com/ContextualAI/bird-sql",
        "theme": "Local BIRD pipeline with candidate generation, SQL execution, reward scoring, and final selection.",
        "use": "Use as a concrete local architecture pattern: candidate pool first, execution second, reward/selector third.",
        "priority": "Now",
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
        "use": "Do not wait for a perfect one-shot generator. Use candidate diversity plus selection as a first-class architecture lane.",
        "priority": "Near",
    },
    {
        "paper": "OmniSQL",
        "source": "https://arxiv.org/abs/2503.02240",
        "theme": "Million-scale synthetic text-to-SQL data across synthetic databases, then open model SFT.",
        "use": "Blacksmith training-data move: expand from hundreds of rows to broad synthetic DB/query coverage before more tiny LoRA tuning.",
        "priority": "Near",
    },
    {
        "paper": "BASE-SQL",
        "source": "https://arxiv.org/abs/2502.10739",
        "theme": "Open-source Qwen2.5-Coder-32B-Instruct baseline with a small multi-call generation recipe.",
        "use": "Reference only. Do not make larger models the product path; small-model capability is the value proposition.",
        "priority": "Reference",
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

BLACKSMITH_ROWS: list[dict[str, str]] = [
    {
        "move": "Candidate Pool + Execution + Selection",
        "why": "One-shot generation is a narrow bottleneck. If the model can produce the right SQL sometimes, pass@N plus execution filtering can recover wins without changing weights.",
        "first_artifact": "Add eval mode that generates N candidates per case, executes each candidate, stores result signatures/errors, and reports pass@N versus selected@1.",
        "gate": "Fresh unseen gate improves by at least +5/50 without regressing seen guardrails below the Exp031 baseline.",
    },
    {
        "move": "Real Metadata Retrieval Layer",
        "why": "Static prompt metadata did not transfer, so retrieval is not trusted by default. It only matters if it selects less but better context than raw schema/profile stuffing.",
        "first_artifact": "After candidate-pool eval exists, add an ablation that compares raw prompt, static notes, and retrieved context on the same frozen adapter.",
        "gate": "Promote only if retrieved context improves fresh unseen selected@1 without increasing token budget or weakening seen guardrails.",
    },
    {
        "move": "Data Scale, Not More Hand Rows",
        "why": "Our rows are high-touch but too few. Recent SFT systems use broad generated coverage across many databases and query shapes.",
        "first_artifact": "Generate or import a large synthetic DB-disjoint SQL curriculum with execution-validated SQL and balanced query complexity.",
        "gate": "Hold out whole synthetic DBs plus BIRD train DBs; promote only if unseen DB accuracy rises, not just same-template accuracy.",
    },
    {
        "move": "LiveSQLBench Agent Skeleton",
        "why": "The portfolio target is a DB-specific SQL agent, not only a one-shot adapter. The first agent artifact is a safe environment step that can execute a candidate SQL action and return structured repair feedback.",
        "first_artifact": "Use the extracted db_sql_agent_env env-step command to produce syntax/schema/execution/result observations for one case and one SQL action; later wrap the same contract in OpenEnv with inspect-schema, inspect-values, repair-sql, and final-answer actions.",
        "gate": "The environment step must be read-only, structured, test-covered, and able to explain real schema/syntax failures before any RL or self-healing loop is added.",
    },
    {
        "move": "Execution-Guided Repair Gate",
        "why": "A consuming agent should be able to send an execution error back for correction, but that must not hide whether the direct generator regressed.",
        "first_artifact": "Run Exp056 on challenge_v2 with one-shot@1, repair final@1, pass@5, and selected@1 reported as separate artifacts.",
        "gate": "Promote repair as an endpoint workflow only if final@1 improves on strong execution-blocking failures while eval_v1 one-shot guardrails stay clean.",
    },
    {
        "move": "Fine-Tuning Stop Rule",
        "why": "Without a stop rule, we will keep doing clean but low-leverage LoRA runs.",
        "first_artifact": "Track one-shot@1, candidate pass@N, selected@1, seen guardrails, token budget, and train rows for each experiment.",
        "gate": "Stop pure SFT when one-shot@1 plateaus, when pass@N shows the answer is already in the candidate pool, or when added data improves seen DBs but not fresh unseen DBs.",
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
        "repo_use": "Implemented for Exp049 and Exp057 with bitsandbytes 4-bit NF4 plus PEFT k-bit preparation. Exp057 on train_v4 ran faster and loaded fewer total parameters than Exp056 LoRA, reached 12/12 dev_v2 and 22/24 challenge_v1, but regressed stable eval_v1 from 12/12 to 10/12; use QLoRA as a memory/runtime tradeoff unless it matches the full LoRA quality gates.",
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
        "repo_use": "Serve Exp048 through the OpenAI-compatible completions API, then score the same frozen SQL eval files with --openai-base-url before promotion. Local WSL RTX 2080 Ti startup works with eager mode, capped GPU memory, FlashInfer sampler disabled, and TRITON_ATTN; do not promote until the endpoint eval also passes.",
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
        "serving.html": _render_serving(),
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
        initial_adapter_dir=_optional_repo_path(student.get("initial_adapter_dir")),
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
          <p class="lead">This view organizes the repo by decisions, artifacts, gates, and experiment evidence. Markdown and browser pages are both repo docs; this site is the structured working cockpit.</p>
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
              <li>Run Exp036 as the broad-data model-path blacksmith move: BIRD train minus both unseen gates, profile notes, compact train-only schema linking, and token-budget filtering.</li>
              <li>Then run Exp037 with Qwen2.5-Coder-1.5B on the exact same data and gates.</li>
              <li>Try Exp038 with Qwen2.5-Coder-3B only after the 1.5B run proves the model-family move is worth the extra memory.</li>
              <li>Keep restaurant plus airline as prompt-dev; keep works_cycles plus public_review_platform as the fresh unseen gate.</li>
              <li>Compare Exp036, Exp037, and Exp038 one-shot against Exp031 and Exp034 before any candidate-pool work.</li>
              <li>Do not mix repair or agent loops into the current score.</li>
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
          <h2>MLOps Run Contract</h2>
          <p><code>sqlbench_lab.mlops.run_contract</code> is the first production-loop artifact for TAP-631 and the dev environment boundary for TAP-648. It converts existing manifest, train summary, eval result, analysis, endpoint eval, and load-test JSON into one machine-readable contract plus a deterministic promotion decision.</p>
          <table class="key-table">
            <tr><th>Environment</th><td>Only <code>dev</code> is supported now. The default contract records <code>gs://mistri-sqlbench-dev-artifacts</code>, <code>gs://mistri-sqlbench-dev-datasets</code>, <code>gs://mistri-sqlbench-dev-models</code>, and the dev train, serving, and pipeline service-account names. Any non-dev environment fails fast.</td></tr>
            <tr><th>Inputs</th><td>environment, experiment_id, manifest_path, base_model, adapter_name, adapter_method, train_datasets, and output_root.</td></tr>
            <tr><th>Train</th><td>train_row_count, dry_run, trainable_parameters, total_parameters, train_loss, and train_runtime_seconds.</td></tr>
            <tr><th>Eval gates</th><td>Offline and endpoint eval gates keep result_path, analysis_path, passed_count, pass_rate, failure_counts, failed_case_ids, and protected/required thresholds.</td></tr>
            <tr><th>Load gates</th><td>Request count, concurrency, success count, RPS, p50, p95, max latency, and required success rate.</td></tr>
            <tr><th>Decision</th><td><code>promote</code>, <code>reject</code>, or <code>investigate</code>. Missing required evidence investigates; failed thresholds reject.</td></tr>
          </table>
          <div class="callout">The first tests prove the contract can represent Exp056 as promotable, Exp062 as rejected for protected eval regression, and Exp049 as rejected when offline and endpoint quality gates miss.</div>
        </section>
        <section class="panel full">
          <h2>Local Dev Metaflow Flow</h2>
          <p><code>flows/sql_adapter_offline_dev_flow.py</code> is the TAP-632/TAP-633 orchestration artifact. It keeps Metaflow thin: the flow validates the manifest through the repo CLI, replays explicit train/eval artifacts, optionally gates endpoint eval and load-test artifacts, runs failure analysis through the repo CLI, builds the MLOps run contract, and emits the dev promotion decision.</p>
          <table class="key-table">
            <tr><th>Command</th><td><code>uv run --group mlops python flows/sql_adapter_offline_dev_flow.py run</code></td></tr>
            <tr><th>Default target</th><td>Exp056 replay mode with the promoted train summary and dev_v2, eval_v1, and challenge_v1 result files.</td></tr>
            <tr><th>Offline steps</th><td>start, validate_inputs, train_adapter, eval_dev, eval_eval, eval_challenge.</td></tr>
            <tr><th>Serving steps</th><td>start_temp_dev_endpoint, wait_for_health, endpoint_eval, load_test, stop_temp_dev_endpoint, decide_dev_promote_or_reject, end.</td></tr>
            <tr><th>Serving replay</th><td>Pass <code>--endpoint-eval-result</code>, <code>--endpoint-min-passed</code>, and <code>--load-test-result</code> to require endpoint quality and concurrency evidence in the final decision.</td></tr>
            <tr><th>Serving execution</th><td><code>--run-endpoint-eval</code> and <code>--run-load-test</code> call the repo CLI against an explicit <code>--openai-base-url</code> and <code>--openai-model</code>; the flow never starts a hidden GPU server.</td></tr>
            <tr><th>Boundary</th><td>Dev only. The planner rejects non-dev environments and does not define prod paths or prod promotion behavior.</td></tr>
          </table>
          <div class="callout">Replay mode is intentional: it proves orchestration, artifact capture, failure analysis, endpoint/load gates, and promotion logic without requiring a GPU train/eval or serving rerun.</div>
        </section>
        <section class="panel full">
          <h2>Dev GCS Sync Plan</h2>
          <p><code>sqlbench_lab.mlops.gcs_sync</code> is the TAP-634 artifact boundary for cloud persistence. It builds a deterministic dev-only sync manifest from the run contract and promotion decision; it does not upload files.</p>
          <table class="key-table">
            <tr><th>Schema</th><td><code>sql_adapter_gcs_sync_plan:v1</code></td></tr>
            <tr><th>Run prefix</th><td><code>gs://mistri-sqlbench-dev-artifacts/sql-adapter-runs/dev/{{experiment_id}}/{{run_id}}/</code></td></tr>
            <tr><th>Model URI</th><td><code>gs://mistri-sqlbench-dev-models/adapters/{{adapter_name}}/</code></td></tr>
            <tr><th>Artifacts</th><td>manifest, train summary, eval results, eval analyses, load tests, run contract, and promotion decision.</td></tr>
            <tr><th>Boundary</th><td>The planner rejects non-dev contracts and never defines prod paths or prod promotion pointers.</td></tr>
          </table>
          <div class="callout">The Metaflow flow now exposes <code>gcs_sync_plan</code> as a run artifact after <code>decide_dev_promote_or_reject</code>; real upload remains a later explicit step.</div>
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
    ledger_experiments = list(experiments[-16:])
    anchor = next((exp for exp in experiments if "exp028" in exp.experiment_id), None)
    if anchor is not None and anchor not in ledger_experiments:
        ledger_experiments = [anchor, *ledger_experiments]
    rows = "\n".join(_experiment_summary_row(exp) for exp in ledger_experiments)
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
    blacksmith_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['move'])}</strong></td>
          <td>{_escape(row['why'])}</td>
          <td>{_escape(row['first_artifact'])}</td>
          <td>{_escape(row['gate'])}</td>
        </tr>
        """
        for row in BLACKSMITH_ROWS
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
          <p class="lead">Last researched 2026-05-13. Each source is kept only if it changes a concrete SQLBench Lab design choice, experiment, or hygiene rule.</p>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>Immediate Read</h2>
            <p>Exp034 confirmed that small one-shot SFT changes are not enough. The active model-path blacksmith sequence is Exp036 then Exp037: first measure broad BIRD training coverage on Qwen3.5-0.8B, then keep the same data boundary and switch only to Qwen2.5-Coder-1.5B.</p>
          </article>
          <article class="panel">
            <h2>Research Boundary</h2>
            <p>Current scope is one-shot model quality. Candidate selection, repair, and agent workflows remain separate lanes and are not part of the active Exp036 measurement boundary.</p>
          </article>
        </section>
        <section class="panel full">
          <h2>Blacksmith Moves</h2>
          <table class="dense-table">
            <thead><tr><th>Move</th><th>Why</th><th>First Artifact</th><th>Promotion Gate</th></tr></thead>
            <tbody>{blacksmith_rows}</tbody>
          </table>
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


def _render_serving() -> str:
    stress_rows = "\n".join(
        f"""
        <tr>
          <td><strong>c{_escape(row['concurrency'])}</strong></td>
          <td>{_escape(row['requests'])}</td>
          <td>{_escape(row['success'])}</td>
          <td>{_escape(row['rps'])}</td>
          <td>{_escape(row['p50'])}</td>
          <td>{_escape(row['p95'])}</td>
          <td>{_escape(row['max'])}</td>
        </tr>
        """
        for row in SERVING_STRESS_ROWS
    )
    lora_qlora_stress_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['adapter'])}</strong></td>
          <td>c{_escape(row['concurrency'])}</td>
          <td>{_escape(row['requests'])}</td>
          <td>{_escape(row['success'])}</td>
          <td>{_escape(row['rps'])}</td>
          <td>{_escape(row['p50'])}</td>
          <td>{_escape(row['p95'])}</td>
          <td>{_escape(row['avg_chars'])}</td>
        </tr>
        """
        for row in LORA_QLORA_STRESS_ROWS
    )
    gpt_cost_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['scenario'])}</strong></td>
          <td>{_escape(row['tokens'])}</td>
          <td>{_escape(row['cost_10k_week'])}</td>
          <td>{_escape(row['cost_10k_month'])}</td>
          <td>{_escape(row['cost_100k_week'])}</td>
          <td>{_escape(row['cost_100k_month'])}</td>
        </tr>
        """
        for row in GPT51_COST_ROWS
    )
    self_host_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{_escape(row['setup'])}</strong></td>
          <td>{_escape(row['weekly'])}</td>
          <td>{_escape(row['monthly'])}</td>
          <td>{_escape(row['read'])}</td>
        </tr>
        """
        for row in SELF_HOST_COST_ROWS
    )
    body = f"""
        <section class="page-head compact">
          <p class="eyebrow">Serving</p>
          <h1>Deployment quality, concurrency, and cost are separate gates.</h1>
          <p class="lead">Exp048 is the current best local adapter, but the OpenAI-compatible vLLM endpoint is not promoted until endpoint quality matches local HF quality.</p>
        </section>
        <section class="grid three">
          {_metric_card("Local HF quality", "10/12", "Exp048 frozen same-DB held-out eval")}
          {_metric_card("vLLM endpoint quality", "9/12", "One extra alias ownership miss versus local HF")}
          {_metric_card("Local stress ceiling", "160/160", "Accepted concurrent clients; p95 was 63.44s")}
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>Promotion Boundary</h2>
            <p>The served endpoint is mechanically working, but it is not the blessed deployment path. The vLLM backend must match or beat the local HF frozen eval score before product promotion.</p>
            <div class="callout">Do not claim production readiness from transport success alone. Quality parity, latency, and cost must be recorded as separate measurements. Exp049 QLoRA also served successfully but stayed at 9/12, so it is rejected as a replacement for Exp048.</div>
          </article>
          <article class="panel">
            <h2>Local Runtime Shape</h2>
            <table class="key-table">
              <tr><th>GPU</th><td>RTX 2080 Ti, WSL, fp16 fallback.</td></tr>
              <tr><th>vLLM</th><td>0.22.1 with TRITON_ATTN and eager mode.</td></tr>
              <tr><th>KV cache</th><td>230,912 tokens; reported 150.33x at 1,536 tokens/request.</td></tr>
              <tr><th>Practical read</th><td>c8-c16 for interactive probes; c64+ only for batch/background traffic.</td></tr>
            </table>
          </article>
        </section>
        <section class="panel full">
          <h2>QLoRA Serving Check</h2>
          <p>Exp049 repeated Exp048 with QLoRA only: same storefront train_v3 data, same prompt, same r16/alpha32/dropout0.10 LoRA shape, and same dev/eval gates. Training completed and produced a standard adapter, but quality regressed.</p>
          <table>
            <thead><tr><th>Gate</th><th>Result</th><th>Decision</th></tr></thead>
            <tbody>
              <tr><td>Local HF dev</td><td>9/12</td><td>Below Exp048 10/12.</td></tr>
              <tr><td>Local HF held-out eval</td><td>9/12</td><td>Below Exp048 10/12.</td></tr>
              <tr><td>vLLM held-out eval</td><td>9/12</td><td>Serving works, not promoted.</td></tr>
              <tr><td>vLLM c8 load</td><td>32/32, 0.6515 rps, p50 10.21s, p95 17.17s</td><td>Latency is usable for probes; quality is the blocker.</td></tr>
            </tbody>
          </table>
          <div class="callout">The QLoRA adapter is useful as proof that the memory-efficient training path works. It is not the current best storefront model.</div>
        </section>
        <section class="panel full">
          <h2>LoRA vs QLoRA Stress</h2>
          <p>A paired fresh-server ladder compared Exp048 LoRA and Exp049 QLoRA on the same local RTX 2080 Ti, vLLM flags, frozen held-out eval prompts, max_new_tokens=128, and shaped request counts. Both adapters served cleanly through c32; no QLoRA-specific inference stress penalty showed up.</p>
          <table class="dense-table">
            <thead><tr><th>Adapter</th><th>Concurrency</th><th>Requests</th><th>Success</th><th>RPS</th><th>p50</th><th>p95</th><th>Avg Chars</th></tr></thead>
            <tbody>{lora_qlora_stress_rows}</tbody>
          </table>
          <div class="callout">Read this as a transport/runtime result only. Exp049 still fails the quality gate at 9/12, so higher stress tolerance does not make it the promoted model.</div>
        </section>
        <section class="panel full">
          <h2>Local Adapter Inference</h2>
          <p>Start the Exp048 adapter as an OpenAI-compatible vLLM completions endpoint, then send the same rendered SQL prompt shape used by repo eval.</p>
          <pre><code>uv python install 3.12
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12
export CUDA_HOME=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

uv run --group serving vllm serve Qwen/Qwen3.5-0.8B-Base \\
  --host 127.0.0.1 \\
  --port 8001 \\
  --max-model-len 1536 \\
  --gpu-memory-utilization 0.75 \\
  --enforce-eager \\
  --no-enable-flashinfer-autotune \\
  --attention-backend TRITON_ATTN \\
  --language-model-only \\
  --served-model-name storefront-sql \\
  --enable-lora \\
  --max-lora-rank 16 \\
  --lora-modules storefront-sql=$PWD/artifacts/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010/adapter</code></pre>
          <pre><code>curl http://127.0.0.1:8001/v1/completions \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "storefront-sql",
    "prompt": "&lt;|system|&gt;You are a precise text-to-SQL model. Return only the final SQL statement.&lt;|user|&gt;Dialect:\\nSQLite\\n\\nSchema:\\n...\\n\\nQuestion:\\n...&lt;|assistant|&gt;",
    "max_tokens": 128,
    "temperature": 0
  }}'</code></pre>
          <div class="callout">For application traffic, build prompts with the repo renderer rather than hand-writing chat markers. The curl example is only the wire-format shape.</div>
        </section>
        <section class="panel full">
          <h2>Raw Serving Notebook</h2>
          <p>The runnable notebook <code>notebooks/sql_local_serving_kv_cache_walkthrough.ipynb</code> walks one held-out storefront case from manifest to vLLM command, rendered prompt, raw completion request, execution scoring, endpoint gates, and the local vLLM KV-cache allocation functions.</p>
          <pre><code>uv run --with jupyter jupyter lab notebooks/sql_local_serving_kv_cache_walkthrough.ipynb</code></pre>
          <div class="callout">The server-start cell is guarded by <code>START_VLLM_SERVER = False</code> so opening the notebook does not accidentally launch a long-running vLLM process.</div>
        </section>
        <section class="panel full">
          <h2>Repo Endpoint Evaluation</h2>
          <p>Use the CLI path below to score the running endpoint with frozen eval files. Always set a result label so remote endpoint output cannot overwrite local HF eval artifacts.</p>
          <pre><code>uv run python -m sqlbench_lab.cli sql eval \\
  --manifest experiments/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010.json \\
  --model adapter \\
  --dataset datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl \\
  --openai-base-url http://127.0.0.1:8001 \\
  --openai-model storefront-sql \\
  --result-label vllm_eval \\
  --max-new-tokens 128</code></pre>
          <pre><code>uv run python -m sqlbench_lab.cli sql openai-load-test \\
  --manifest experiments/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010.json \\
  --model adapter \\
  --dataset datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl \\
  --openai-base-url http://127.0.0.1:8001 \\
  --openai-model storefront-sql \\
  --requests 32 \\
  --concurrency 8 \\
  --output artifacts/sql/qwen35_0_8b__exp048_storefront_v3_lora_r16_a32_d010/vllm_load_c8.json \\
  --max-new-tokens 128</code></pre>
        </section>
        <section class="panel full">
          <h2>GPT-5.1 API Inference</h2>
          <p>Use GPT-5.1 medium reasoning when accuracy is worth token-priced inference and no local adapter parity risk is acceptable. Keep the stable system prompt and schema prefix first to improve prompt-cache hits.</p>
          <pre><code>curl https://api.openai.com/v1/responses \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "gpt-5.1",
    "reasoning": {{"effort": "medium"}},
    "text": {{"verbosity": "low"}},
    "input": [
      {{
        "role": "system",
        "content": "You are a precise text-to-SQL model. Return only the final SQL statement. Use the declared SQL dialect and stay grounded in the provided schema."
      }},
      {{
        "role": "user",
        "content": "Dialect:\\nSQLite\\n\\nDatabase ID:\\nstorefront_sales\\n\\nSchema:\\n...\\n\\nQuestion:\\n..."
      }}
    ],
    "max_output_tokens": 512
  }}'</code></pre>
          <div class="callout">For cost control, set a tight max_output_tokens ceiling, keep n=1, avoid unnecessary tools, and record input, cached input, output, and reasoning output tokens per request.</div>
        </section>
        <section class="panel full">
          <h2>Local vLLM Stress Test</h2>
          <table class="dense-table">
            <thead><tr><th>Concurrency</th><th>Requests</th><th>Success</th><th>RPS</th><th>p50</th><th>p95</th><th>Max</th></tr></thead>
            <tbody>{stress_rows}</tbody>
          </table>
        </section>
        <section class="panel full">
          <h2>GPT-5.1 Medium Reasoning Cost</h2>
          <p>Rates checked from OpenAI API pricing: GPT-5.1 input $1.25/M tokens, cached input $0.125/M tokens, output $10/M tokens. Medium reasoning cost is modeled as additional output tokens because completion tokens include visible answer and reasoning budget.</p>
          <table class="dense-table">
            <thead><tr><th>Scenario</th><th>Token Budget / Query</th><th>10k / Week</th><th>10k / Month</th><th>100k / Week</th><th>100k / Month</th></tr></thead>
            <tbody>{gpt_cost_rows}</tbody>
          </table>
        </section>
        <section class="panel full">
          <h2>Self-Hosted Cost Comparison</h2>
          <p>At 100k queries/week, query volume is still modest: about 0.165 qps average. Self-hosting cost is mostly GPU uptime, not token count, so one or two L4-class replicas can be materially cheaper if model quality is good enough.</p>
          <table class="dense-table">
            <thead><tr><th>Setup</th><th>Weekly</th><th>Monthly</th><th>Read</th></tr></thead>
            <tbody>{self_host_rows}</tbody>
          </table>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>10k Queries / Week</h2>
            <p>GPT-5.1 is financially reasonable at this volume: typical SQL is roughly $79/week, while always-on self-hosting is roughly $65-$170/week depending on provider. Pick the better quality path first.</p>
          </article>
          <article class="panel">
            <h2>100k Queries / Week</h2>
            <p>Self-hosting becomes economically attractive: GPT-5.1 typical SQL is roughly $3.4k/month, while one to two L4 replicas are roughly $284-$1.2k/month before engineering and on-call cost.</p>
          </article>
        </section>
    """
    return _page("Serving", "serving", body)


def _render_evaluation(experiments: list[ExperimentRecord]) -> str:
    latest = experiments[-1] if experiments else None
    failure_rows = _failure_rows(latest) if latest else ""
    repair_gate_rows = """
              <tr><th>Fixed checkpoint</th><td>Use Exp056, the promoted full-LoRA storefront adapter. Do not train or change prompt/data during this measurement.</td></tr>
              <tr><th>One-shot@1</th><td><code>uv run --group training --group observability python -m sqlbench_lab.cli sql eval --manifest experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json --model adapter --dataset datasets/sql/eval/storefront_sales_lab_challenge_v2.jsonl --result-label exp064_one_shot_challenge_v2 --mlflow</code></td></tr>
              <tr><th>Repair final@1</th><td><code>uv run --group training python -m sqlbench_lab.cli sql eval-repair --manifest experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json --model adapter --dataset datasets/sql/eval/storefront_sales_lab_challenge_v2.jsonl --max-repair-attempts 1</code></td></tr>
              <tr><th>pass@5 and selected@1</th><td><code>uv run --group training --group observability python -m sqlbench_lab.cli sql eval-candidates --manifest experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json --model adapter --dataset datasets/sql/eval/storefront_sales_lab_challenge_v2.jsonl --candidates 5 --temperature 0.7 --top-p 0.95 --result-label exp064_candidates_challenge_v2 --mlflow</code></td></tr>
              <tr><th>Guardrail</th><td>Repeat one-shot eval_v1 for Exp056 and require the protected 12/12 behavior to remain the reference. Repair and candidate selection are endpoint workflow evidence, not replacements for the direct SFT score.</td></tr>
              <tr><th>Readout</th><td>If pass@5 is high and selected@1 is low, the bottleneck is selection. If repair final@1 improves only syntax/schema/execution failures, keep repair narrowly scoped. If neither moves, the generator/data lane needs work.</td></tr>
    """
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
        <section class="panel full">
          <h2>Execution-Guided Repair Experiment</h2>
          <p>Prepare this as a separate endpoint workflow gate: one-shot score, repair score, pass@N, and selected@1 are tracked separately so downstream correction cannot mask model-regression risk.</p>
          <table class="key-table">
            {repair_gate_rows}
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
          <p>Exp031 compared Exp030 against the same fixed holdout after adding compact profile metadata to real BIRD rows. Exp034 showed that compact schema-linking SFT can lift the tuned prompt-dev holdout to 10/50 but still fails to improve the fresh unseen gate. Exp036 is the broad-data Qwen3.5-0.8B baseline; Exp037 and Exp038 keep the same data contract and test whether Qwen2.5-Coder provides the missing code prior.</p>
          <table class="key-table">
            <tr><th>Paper pattern</th><td>Profile columns, retrieve relevant metadata, generate multiple candidates, execute candidates, then select or repair.</td></tr>
            <tr><th>Repo now</th><td>One-shot SFT is measurable but plateauing at 7/50 on the fresh unseen gate.</td></tr>
            <tr><th>Next implementation</th><td>Train and evaluate Exp036 first. Then run Exp037 on the same train/eval boundary; Exp038 follows only if the 1.5B coder model improves the fresh unseen gate.</td></tr>
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
          <h1>Docs can live in browser pages or markdown.</h1>
          <p class="lead">The goal is dense operational memory: choose the format that makes decisions, artifacts, commands, gates, and lessons easiest to reuse.</p>
        </section>
        <section class="grid two">
          <article class="panel">
            <h2>How To Change Docs</h2>
            <ol class="tight">
              <li>Edit <code>src/sqlbench_lab/docs_site/builder.py</code> first for browser docs.</li>
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
            <p>Markdown docs are allowed for repo knowledge, training plans, experiment reads, LiveSQLBench notes, and tooling roadmaps when markdown is the clearer format.</p>
            <div class="callout">Use browser docs for durable operating surfaces that need navigation, tables, cross-links, or generated experiment indexes. Generated HTML is output; <code>builder.py</code> is the tracked browser-docs source.</div>
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
              <tr><th>Initial adapter</th><td><code>{_escape(experiment.initial_adapter_dir or 'none')}</code></td></tr>
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
        ("serving", "Serving", "serving.html"),
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


def _optional_repo_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    if not path.is_absolute():
        return value
    return _repo_path(path)


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
