# Text-to-SQL Fine-Tuning Interview Narrative

Purpose: a rehearsable interview reference for ML engineer and applied AI engineer roles.

This is an interview prep note, not an official benchmark report and not repo documentation. The claims are grounded in local experiment manifests, result artifacts, repo code, and the Linear learning ledger `TAP-532`.

## Interview Card

Core claim:

I built a measurement-driven SQL fine-tuning lab for a production-shaped one-database text-to-SQL endpoint.

Product frame:

A user asks a natural-language question. Another agent calls a specialized SQL-generation endpoint. The endpoint returns SQL for one known database. The surrounding system can validate, execute, repair, or reject the SQL, but the SFT lane measures the direct SQL generator by itself.

Best checkpoint:

Exp056: full LoRA, Qwen/Qwen3.5-0.8B-Base, train_v4, 200 same-DB storefront rows, LoRA r16/alpha32/dropout0.10. It reached dev_v2 11/12, eval_v1 12/12, and challenge_v1 22/24.

Main battle scar:

Exp062 improved the newest challenge_v2 gate from 8/15 to 12/15, but regressed protected eval_v1 from 12/12 to 11/12. I rejected it. That is the central production lesson: a model can improve on a new stress test and still fail promotion if it breaks behavior the consuming agent relies on.

What I can defend:

- Data design: targeted curricula, isolated ablations, hard-negative contrast rows, leakage avoidance.
- Eval design: dev, stable eval, fresh challenge, DB-disjoint holdout, slice analysis, result-equivalence scoring.
- Training design: LoRA, QLoRA, assistant-SQL-only loss, fixed prompt contracts.
- Failure analysis: alias ownership, boundary semantics, anti-join placement, return-ratio denominator, HAVING/grouped-count logic, duplicate-producing joins.
- LLMOps shape: manifest-driven runs, CLI control surface, dev-only environment contract, artifact contract, replayable Metaflow flow.

Do not overclaim:

- These are local/lab results, not official LiveSQLBench scores.
- The one-DB lab proves endpoint-candidate quality, not general SQL ability.
- Exp056 was the promoted checkpoint. Exp062 was rejected.
- The MLOps loop is dev-only. It is not production deployment by itself.

## 30-Second Version

I built a measurement-driven text-to-SQL fine-tuning lab for a one-database endpoint that another agent would call. I fine-tuned Qwen 0.8B with LoRA adapters, measured SQL by execution/result equivalence, and promoted checkpoints only if they passed fixed dev, eval, and challenge gates.

The key scar was Exp062. It improved the newest hard challenge from 8/15 to 12/15, but it regressed the protected eval gate from 12/12 to 11/12, so I rejected it. That is the main thing I learned: in an agent-consumed endpoint, better on the latest slice is not enough if stable behavior regresses.

## 90-Second Version

The project was framed as a production-shaped assignment: build a specialized SQL endpoint for one known database so another agent could answer natural-language questions over that database. I separated the direct SFT generator from repair, reranking, candidate selection, and full agent workflows, because I wanted to know whether the model itself had improved.

The strongest phase was the storefront single-DB lab. I fixed the schema, canonical prompt, trainer, LoRA recipe, and eval gates, then changed one thing at a time: train split, LoRA capacity, QLoRA, isolated data families, and hard-negative contrast rows. Exp056 became the promoted full-LoRA checkpoint: 200 train rows, dev_v2 11/12, eval_v1 12/12, challenge_v1 22/24.

Then I added a fresh sliced challenge gate. Exp056 scored 8/15, with weaknesses in alias ownership, boundary semantics, and anti-join predicate placement. I added contrast data in Exp059 through Exp062. Exp062 improved challenge_v2 to 12/15, but eval_v1 fell to 11/12. I rejected it because the protected eval represented behavior the agent would rely on.

## 5-Minute Version

The useful interview framing is not "I beat a benchmark." It is: I built a controlled lab for deciding whether a fine-tuned SQL generator was reliable enough to become an endpoint candidate for one database.

The endpoint boundary was natural-language question plus database context in, SQL out. Around that endpoint, a production system could add schema-aware prompt construction, SQL validation, execution sandboxing, repair, result inspection, and telemetry. But I measured the generator directly first so downstream guardrails could not hide model regressions.

I started with broader text-to-SQL experiments and learned two early lessons. First, same-DB performance is not the same as unseen-DB generalization. Exp029 reached 40/40 on regional_sales and superstore same-DB gates, but a DB-disjoint restaurant+airline holdout was only 4/50. Second, more examples can hurt. Exp024 added relevant-looking regional_sales normalization rows, but regional_sales dropped from 37/40 to 33/40.

That led to the storefront one-DB lab. The variables were tight:

- one SQLite schema
- one-shot SQL output
- fixed canonical prompt
- TRL SFTTrainer
- assistant-SQL-only loss
- LoRA adapters against Qwen/Qwen3.5-0.8B-Base
- SQLite result-equivalence scoring
- separate dev, eval, and challenge gates
- endpoint-style promotion decisions

The training sequence showed that data composition mattered more than the adapter mechanism. Exp039 and Exp040 were weak starting points. Exp046 found a better LoRA shape on train_v2. Exp048 kept the LoRA recipe fixed and changed only the train split to train_v3, moving eval_v1 to 10/12 and challenge_v1 to 15/24.

After that I treated data families as ablations. Exp051 support rows regressed. Exp052 date-boundary rows preserved eval but did not move the challenge enough. Exp053 return-ratio rows were the strongest isolated family. Exp054 HAVING rows improved challenge but regressed eval. Exp055 anti-join rows helped some slices but were not enough alone.

Exp056 combined the useful families into train_v4: support, date, return-ratio, HAVING, and anti-join supplements. It trained on 200 rows with full LoRA r16/alpha32/dropout0.10. It reached dev_v2 11/12, eval_v1 12/12, and challenge_v1 22/24, so I promoted it as the best local endpoint candidate.

Then I tested QLoRA. Exp057 kept train_v4 fixed but changed to 4-bit NF4 QLoRA. It was operationally attractive: runtime was about 2014 seconds versus about 2230 seconds for Exp056, and dev_v2 improved to 12/12. But eval_v1 regressed to 10/12, so I kept QLoRA as an efficiency tradeoff, not the quality checkpoint.

The strongest scar came after that. A fresh challenge_v2 showed Exp056 was brittle:

- alias_ownership: 3/4
- boundary_semantics: 4/6
- anti_join / left_join_predicate: 1/5
- total challenge_v2: 8/15

I added hard-negative contrast data. These were near-miss examples where the wrong answer looks very close to the right one. Exp059 added alias-ownership contrast rows. Exp060 added boundary contrast rows. Exp061 added anti-join contrast rows. Exp062 bundled all 36 contrast rows.

Exp062 improved challenge_v2 to 12/15 and held dev_v2 at 12/12, but it regressed eval_v1 to 11/12. I rejected it. That is what made the lab real: I did not promote the model that looked best on the newest hard test because it broke a protected gate.

## System Shape

```text
User question
  -> agent
  -> SQL-generation endpoint
  -> schema/context prompt
  -> fine-tuned adapter
  -> generated SQL
  -> validator / executor / repair / telemetry
  -> database
  -> result returned through the agent
```

The SFT experiments measured the fine-tuned adapter at the direct SQL-generation step. Repair, reranking, pass@N, execution-guided correction, and agent workflows are useful, but they are separate measurement lanes.

How to say it:

I was not training a general benchmark model. I was training a specialized SQL endpoint candidate for one known database. The question was whether the endpoint was reliable on that database and whether model updates preserved behavior the consuming agent expected.

## Measurement Design

The eval design was the backbone. I did not rely on one aggregate score.

The surfaces were:

- quick dev gates for fast iteration
- stable eval gates as protected holdouts
- fresh challenge gates to reveal hidden brittleness
- DB-disjoint holdouts when testing generalization beyond same-DB behavior
- slice analysis for failure families
- result-equivalence scoring by executing predicted SQL and target SQL on SQLite

Syntax errors and invalid SQL counted as not executing correctly. A query that executes but returns different rows also fails. That distinction matters because execution failure, row-count mismatch, and row-value mismatch point to different fixes.

How to say it:

I promoted checkpoints through gates, not through vibes. For an agent-consumed endpoint, a pass-rate improvement was not enough if the failure mix moved in the wrong direction or a protected workflow regressed.

## Training And Data Design

My strongest practical learning was that data composition mattered more than the training API.

The data loop was:

1. Run fixed eval.
2. Slice failures by mechanism.
3. Add the smallest targeted data family that should move that mechanism.
4. Keep model, prompt, trainer, and gates fixed where possible.
5. Re-run dev, protected eval, and challenge gates.
6. Promote, reject, or investigate based on all gates, not only the newest one.

Hard-negative contrast data means training rows where the wrong answer is tempting because it looks very close to the right one. I was not adding random extra examples. I was adding examples that forced the model to choose the exact semantic distinction it had been missing.

Example boundary contrast:

```text
Question A: How many completed orders were placed strictly before 2024-04-01?
Correct SQL: ... WHERE status = 'completed' AND order_date < '2024-04-01'

Question B: How many completed orders were placed on or before 2024-04-01?
Correct SQL: ... WHERE status = 'completed' AND order_date <= '2024-04-01'
```

The words are almost the same, but the operator must change. If the model treats both as the same pattern, execution returns the wrong result.

Example anti-join contrast:

```sql
-- Correct shape for "customers with no unresolved support tickets"
SELECT T1.customer_name
FROM customers AS T1
LEFT JOIN support_tickets AS T2
  ON T1.customer_id = T2.customer_id AND T2.resolved = 0
WHERE T2.ticket_id IS NULL
ORDER BY T1.customer_name
```

The important part is that `T2.resolved = 0` belongs in the `ON` clause. Putting it in the `WHERE` clause changes the meaning of the left join and can destroy the anti-join.

How to say it:

Data was not "more rows is better." Data was "what failure family am I trying to move, what stayed fixed, and what protected gate proves I did not damage something else?"

## Failure Examples

| Failure family | Concrete example | Why it matters |
|---|---|---|
| alias ownership | `status` belongs to `orders`, while `customer_name` belongs to `customers` | The SQL can look plausible but filter the wrong table or alias. |
| boundary semantics | `strictly before` needs `<`; `on or before` needs `<=` | One operator changes the business answer. |
| anti-join predicate placement | `T2.resolved = 0` belongs in the `LEFT JOIN ON` clause before `WHERE T2.ticket_id IS NULL` | Moving the filter can break "no related rows" logic. |
| return-ratio denominator | return rate can mean returns per order, item, customer, or product depending on wording | A plausible denominator can still answer the wrong question. |
| HAVING/grouped-count logic | "products with at least 3 completed units" needs grouping then `HAVING units >= 3` | Filtering before aggregation gives the wrong population. |
| duplicate-producing joins | joining orders, items, returns, and tickets can multiply rows | Aggregates and counts become inflated. |
| support-ticket filters | resolved/unresolved, issue type, and ticket date belong to support tickets | Filtering the wrong table changes the support workflow answer. |
| revenue discount calculation | revenue used `quantity * unit_price * (1 - discount_pct / 100.0)` | Ignoring discounts or integer division creates wrong totals. |

How to say it:

The useful unit of analysis was not just pass/fail. It was failure mechanism. That let me decide whether to add data, change schema grounding, add repair, or reject the checkpoint.

## LoRA And QLoRA

The practical LoRA story is tradeoff-driven.

Exp056 full LoRA was the preferred quality checkpoint:

- train rows: 200
- train loss: about 0.0670
- dev_v2: 11/12
- eval_v1: 12/12
- challenge_v1: 22/24

Exp057 QLoRA was efficient and competitive but not promoted:

- train rows: 200
- train loss: about 0.0719
- runtime: about 2014 seconds versus about 2230 seconds for Exp056
- dev_v2: 12/12
- eval_v1: 10/12
- challenge_v1: 22/24

How to say it:

QLoRA was a credible memory/runtime option, but I would not promote it unless it matched full LoRA on the stable eval gate. For this lab, full LoRA was the quality checkpoint and QLoRA was the efficiency tradeoff.

## LLMOps Layer

After the training experiments, I started turning the lab into a dev-only LLMOps loop. This was not production deployment yet. It was the control plane I would want before production deployment.

The dev-loop slices were:

- `TAP-648`: dev-only environment contract
- `TAP-631`: SQL adapter workflow contract and artifact layout
- `TAP-632`: local/dev Metaflow offline flow

The dev environment contract was deliberately narrow:

- only `environment=dev` is supported
- dev GCS boundaries are explicit:
  - `gs://mistri-sqlbench-dev-artifacts`
  - `gs://mistri-sqlbench-dev-datasets`
  - `gs://mistri-sqlbench-dev-models`
- dev service accounts are explicit:
  - `sqlbench-dev-pipeline-sa`
  - `sqlbench-dev-train-sa`
  - `sqlbench-dev-serving-sa`
- non-dev environments fail fast
- there is no prod bucket, prod pointer, prod IAM, or prod promotion behavior

The run contract turns existing artifacts into one machine-readable decision object:

- manifest and experiment identity
- training summary
- offline eval gates
- endpoint eval gates when present
- load-test gates when present
- failure counts and failed case IDs
- promotion decision: `promote`, `reject`, or `investigate`

The first Metaflow flow is intentionally thin. It does not replace the repo CLI or invent a second training path. It validates the manifest through the repo CLI, replays existing Exp056 train/eval artifacts, runs failure analysis through the repo CLI, builds the LLMOps run contract, and emits the dev promotion decision.

The latest serving work made this more concrete. The repo now has a dev vLLM serving image contract and a local FastAPI/HTMX SQL Ask app.

- The vLLM image packages an OpenAI-compatible LoRA serving endpoint with a strict environment contract.
- The local rehearsal target is `local_gpu_docker`: same serving image, local NVIDIA runtime, local or GCS model/adapter URIs, and endpoint eval/load gates.
- The recorded local GPU Docker evidence for Exp056 passed eval_v1 12/12 through the LoRA request model.
- The c8/r32 local load test succeeded 32/32 with zero recorded failures.
- The dev promotion bundle recorded decision `promote` with gates: train, dev_v2, eval_v1, challenge_v1, endpoint_eval, and local load.
- The FastAPI app is a local manual console: it reuses the eval prompt renderer, calls the explicit OpenAI-compatible endpoint, extracts generated SQL, and executes one read-only SQLite `SELECT` or `WITH` statement.

That changes the claim from "I only had offline experiments" to "I had a local production-shaped runtime rehearsal." The remaining work is productionalization: registry, deployment target, auth, monitoring, rollback, security, and promotion governance.

Concrete verification:

```text
uv run pytest tests/test_mlops_contract.py tests/test_mlops_offline_dev_flow.py tests/test_docs_site.py
# 11 passed

uv run python -m sqlbench_lab.cli docs build
# built successfully

uv run --group mlops python flows/sql_adapter_offline_dev_flow.py run
# decision: promote
# passed_gates: train, dev_v2, eval_v1, challenge_v1

uv run --group web pytest tests/test_sql_query_app.py tests/test_mlops_container_contract.py tests/test_mlops_dev_cloud_contracts.py
# 25 passed

docker run --rm sqlbench-lab-dev-cli:dev sql validate-manifest --manifest experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json
# validated SQL SFT manifest: qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010 (200 train row(s), 12 smoke case(s))

docker build -f docker/sqlbench-dev-cli.Dockerfile --build-arg 'INSTALL_GROUPS=mlops web' -t sqlbench-lab-dev-cli:web-local .
# built successfully

docker run -d --rm --name sqlbench-query-app-smoke -p 18080:8080 sqlbench-lab-dev-cli:web-local web query-app --openai-base-url http://127.0.0.1:8000 --openai-model qwen35_0_8b_storefront_v4_lora_r16_a32_d010_exp056 --host 0.0.0.0 --port 8080
curl http://127.0.0.1:18080/
# HTTP 200
```

How to say it:

I did not jump from notebook training to "production." I first built a dev-only promotion loop: explicit environment boundary, explicit artifact contract, a replayable flow, a local Docker serving rehearsal, and a small FastAPI app that proves the endpoint can be exercised like a product surface.

## Manifest And CLI Control Surface

The manifest is the experiment's source of truth. It made experiments easier to train, evaluate, debug, replay, and later operationalize.

A manifest tied together:

- experiment ID
- base model
- adapter name
- train files
- eval files
- prompt/template contract
- LoRA or QLoRA config
- trainer settings
- artifact paths
- result paths
- promotion metadata

That mattered because the CLI and flow could consume the same JSON instead of hardcoding paths in commands, notebooks, or scripts.

How the manifest helped:

- Training: the trainer knew which model, adapter, data files, and hyperparameters belonged to the run.
- Evaluation: the eval command knew which adapter and eval gates to run.
- Analysis: failure analysis could link results back to experiment identity and datasets.
- Reproducibility: a future run could replay the same contract.
- MLOps: the Metaflow flow could validate and package the run without inventing another experiment format.
- Productionalization: a platform engineer would have a structured object to map into artifacts, model registry entries, endpoint candidates, and promotion records.

The CLI is the control surface. It kept the loop boring and repeatable:

```text
validate manifest
train adapter
evaluate gates
analyze failures
build docs
emit run contract
replay flow
```

How to say it:

The CLI made local development look like the future production loop. The same contract that trained an adapter locally could be consumed by offline eval, docs, failure analysis, and the dev promotion flow.

## What Was Still Missing For Production

I would not hand this to platform engineering as production-ready serving yet. I would hand it over as a strong dev-stage control plane and endpoint-candidate workflow.

Still missing before production:

- model registry integration
- artifact signing/checksums
- dev-to-staging promotion boundary
- temporary serving endpoint eval
- load tests for latency, throughput, and concurrency
- SQL sandboxing and read-only enforcement
- query cost/time limits
- schema drift detection
- online telemetry
- rollback plan
- human approval policy for model promotion
- agent-visible error contract
- repair/reranking gates measured separately from one-shot SFT

How to say it:

The lab is close to LLMOps in the sense that it has repeatable experiments, manifests, eval gates, run contracts, a dev replay flow, local Docker serving evidence, endpoint eval/load artifacts, and a product-like FastAPI probe. It is not full production LLMOps until serving, security, monitoring, rollback, and promotion governance are in place.

## How I Would Productionalize It

Use this if the interviewer asks: "How would you turn this into production?"

I would not rewrite the training system. I would harden the boundary around the existing contract: manifest, adapter artifact, eval gates, serving image, endpoint eval, load test, and promotion decision. The goal would be to make promotion a controlled state change, not a manual copy of files.

### 1. Registry

I would register each candidate as a versioned model package:

- base model ID and revision
- LoRA adapter URI
- manifest URI
- training summary URI
- eval result URIs
- endpoint eval and load-test URIs
- artifact hashes
- approved serving image digest
- promotion decision and approver

The registry record should point to immutable artifacts, not mutable local paths. The repo already moves in this direction with run contracts, publish records, artifact hashes, and current/rollback pointers. Production would formalize that in a model registry or an internal release registry.

How to say it:

The registry is not just "where the model lives." It is the release object that ties the adapter to the exact data, code, evals, image, and decision that produced it.

### 2. Deployment Target

I would choose a GPU serving target where I control the NVIDIA driver/runtime compatibility. The current vLLM image hit a real Cloud Run GPU problem: the managed L4 driver was too old for the CUDA runtime. So I would not force that target.

Safer production targets:

- GCE GPU VM for the first controlled deployment
- GKE GPU node pool if the org already runs Kubernetes
- Vertex custom GPU endpoint if the platform team wants managed endpoint lifecycle

The serving contract would stay the same:

- OpenAI-compatible `/v1/completions`
- explicit model ID for the LoRA adapter
- pinned serving image digest
- base model URI
- adapter URI
- health endpoint
- max context length, max sequences, GPU memory utilization

How to say it:

The deployment target is an infrastructure choice, but the serving contract should not change. I would keep the same OpenAI-compatible API and promote only after endpoint eval and load gates pass on the actual target.

### 3. Auth

I would separate human access, service access, and model artifact access.

- The calling agent gets a service identity, not a shared API key.
- The endpoint only accepts requests from approved services or through an internal gateway.
- The serving runtime can read only the approved base model and adapter artifacts.
- Developers can run dev endpoints, but cannot write production pointers.
- Production promotion requires a separate role.

For a simple first version, this could be an internal API gateway in front of the endpoint with IAM or signed service-to-service tokens. The important thing is that the database-query agent is authenticated as a workload, not as a person.

How to say it:

Auth is part of the model contract because this endpoint can generate database queries. I would not expose it as an unauthenticated text completion service.

### 4. Monitoring

I would monitor both runtime health and SQL behavior.

Runtime metrics:

- request count
- error count
- timeout count
- p50/p95/p99 latency
- tokens or generated character count
- GPU memory pressure
- startup time
- replica health

SQL behavior metrics:

- empty SQL rate
- syntax failure rate
- schema failure rate
- execution failure rate
- read-only rejection count
- row-limit truncation count
- top failing question patterns
- drift in table/column usage

For offline/nearline monitoring, I would sample production questions into a review queue, remove sensitive values if needed, and replay a frozen eval set before every promotion. I would not use production traffic directly as training data without a privacy and leakage review.

How to say it:

Monitoring is not just "is the container up?" For text-to-SQL, I need to know whether the endpoint is producing executable, safe, schema-valid SQL and whether failures are concentrating in a new slice.

### 5. Rollback

Rollback should be a pointer change, not a rebuild.

I would keep:

- current pointer: approved production adapter version
- previous pointer: last known good version
- rollback pointer: explicit fallback package
- immutable artifact URIs for every promoted version
- endpoint config snapshot

If a new model regresses, rollback means repoint the serving deployment to the previous approved adapter and redeploy or reload. The repo already has dev current and rollback pointer concepts; production would add approval and audit around them.

How to say it:

Rollback has to be designed before launch. If the only rollback plan is "find the old files and rebuild the container," the release system is not ready.

### 6. Security

Text-to-SQL needs security at three layers: model endpoint, SQL execution, and data access.

Model endpoint:

- private network or authenticated gateway
- request size limits
- rate limits
- structured logging without sensitive payload leakage

SQL execution:

- read-only database user
- one-statement enforcement
- allow only `SELECT` or approved `WITH`
- query timeout
- row limit
- cost guardrail
- blocked dangerous functions if needed

Data access:

- database permissions scoped to the agent use case
- no write privileges
- no unrestricted cross-tenant access
- audit log from natural-language question to generated SQL to execution result

How to say it:

The model is not the security boundary. The SQL executor and database permissions are the security boundary. The model can propose SQL; the system decides whether it is allowed to run.

### 7. Promotion Governance

Promotion should require evidence, not confidence.

Minimum promotion packet:

- manifest validated
- training completed
- protected eval passed
- challenge eval passed or explicitly waived
- endpoint eval passed on the serving target
- load test passed
- failure analysis reviewed
- artifact hashes recorded
- serving image digest recorded
- rollback target exists
- human approval recorded

I would use three possible decisions:

- `promote`: all required gates passed
- `reject`: a protected gate failed
- `investigate`: evidence is incomplete or mixed

The Exp062 lesson is exactly why governance matters. It improved a new challenge gate but regressed a protected eval gate, so it should not be promotable without an explicit exception.

How to say it:

Promotion governance is how you prevent a good experiment from becoming a bad release. The release decision should be reproducible from artifacts.

### Short Interview Answer

I would productionalize it by keeping the experiment contract intact and adding release controls around it. The model registry would store the immutable adapter package, manifest, evals, hashes, serving image digest, and decision. The deployment target would be a GPU environment with driver control, probably GCE, GKE, or Vertex, because the Cloud Run GPU attempt exposed CUDA driver mismatch risk. Auth would be service-to-service, not public completion access. Monitoring would track runtime metrics plus SQL-specific failures like syntax, schema, execution, empty SQL, and rejected non-read-only queries. Rollback would be a pointer change to the last approved adapter. Security would live in the SQL execution boundary: read-only user, one statement, row limits, timeouts, and audit logs. Promotion would require protected eval, endpoint eval, load test, artifact hashes, rollback target, and human approval.

## Execution-Guided Repair Plan

Yes, I would add execution-guided repair, but as an endpoint workflow gate, not as a way to blur the one-shot SFT score.

The exact experiment I would run next:

1. Freeze the promoted Exp056 adapter.
2. Freeze the storefront challenge_v2 dataset.
3. Measure one-shot@1: direct model output executes and matches target result.
4. Measure repair final@1: after a failed first attempt, send the execution/schema/syntax observation back once and evaluate repaired SQL.
5. Measure pass@5: generate five candidates and ask whether any candidate is correct.
6. Measure selected@1: apply a non-gold selector and check whether the selected candidate is correct.

Why this split matters:

- If one-shot@1 regresses but repair hides it, the endpoint may become dependent on a fragile correction loop.
- If pass@5 is high but selected@1 is low, generation can already produce the answer and selection is the bottleneck.
- If repair final@1 improves only syntax/schema/execution failures, repair is a useful guardrail but not evidence that SFT learned deeper semantics.

Prepared commands:

```bash
uv run --group training --group observability python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json \
  --model adapter \
  --dataset datasets/sql/eval/storefront_sales_lab_challenge_v2.jsonl \
  --result-label exp064_one_shot_challenge_v2 \
  --mlflow

uv run --group training python -m sqlbench_lab.cli sql eval-repair \
  --manifest experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json \
  --model adapter \
  --dataset datasets/sql/eval/storefront_sales_lab_challenge_v2.jsonl \
  --max-repair-attempts 1

uv run --group training --group observability python -m sqlbench_lab.cli sql eval-candidates \
  --manifest experiments/sql/qwen35_0_8b__exp056_storefront_v4_lora_r16_a32_d010.json \
  --model adapter \
  --dataset datasets/sql/eval/storefront_sales_lab_challenge_v2.jsonl \
  --candidates 5 \
  --temperature 0.7 \
  --top-p 0.95 \
  --result-label exp064_candidates_challenge_v2 \
  --mlflow
```

Promotion rule:

Repair can be promoted as an agent-endpoint behavior only if it improves execution-blocking failures without changing the protected one-shot eval story. I would not rename repair final@1 as the model's one-shot SQL accuracy.

## Background Scars

### Same-DB Success Did Not Transfer

Exp029 reached 40/40 on both regional_sales and superstore same-DB dev surfaces. But the DB-disjoint restaurant+airline holdout was only 4/50. Exp030 moved it to 5/50. Exp031 moved it to 7/50.

Interview phrasing:

I learned to label the measurement boundary honestly. Same-DB local performance is useful for a specialized endpoint, but it is not a benchmark claim and not unseen-DB generalization.

### More Data Can Regress The Model

Exp024 added extra regional_sales normalization examples and regional_sales dev dropped from 37/40 to 33/40 while superstore stayed 40/40. Later, Exp051 support-ticket rows also regressed the storefront gates.

Interview phrasing:

The model did not reward blind row scaling. Even rows that looked semantically relevant could shift the decode path or output validity in a bad direction.

### Passive Metadata Was Not Enough

Exp026 added column notes, but regional_sales stayed 37/40. Exp027 added direct contrast rows and still stayed 37/40. Exp028 changed the canonical training slot and moved regional_sales to 38/40. Exp029 profile notes then reached 40/40, but unseen transfer remained weak.

Interview phrasing:

Metadata was useful, but notes by themselves did not always change the SQL the model preferred to generate. I had to treat prompt/context, target-shape supervision, and eval transfer as separate questions.

### Fresh Gates Exposed Hidden Weakness

Exp056 looked strong on dev_v2, eval_v1, and challenge_v1. Then challenge_v2 showed 8/15, with anti-join / left-join predicate at 1/5.

Interview phrasing:

That was a useful failure. It meant the old gates were not lying, but they were incomplete. The new gate made the weakness measurable.

### Best New-Gate Score Was Rejected

Exp062 improved challenge_v2 to 12/15, but eval_v1 dropped from Exp056's 12/12 to 11/12.

Interview phrasing:

This is the cleanest scar. I rejected the checkpoint that did best on the newest stress test because it regressed the protected holdout.

## Answer Bank

### What exactly did you fine-tune?

I fine-tuned Qwen/Qwen3.5-0.8B-Base with LoRA adapters for one-shot SQL generation. The strongest case study is the storefront single-DB lab: a SQLite storefront schema, canonical chat prompt, assistant-SQL-only loss, TRL SFTTrainer, and result-equivalence scoring against fixed dev/eval/challenge files.

### Why fine-tune instead of just prompt a general model?

Because the target was a specialized endpoint for one database, not open-ended SQL over arbitrary schemas. Prompting a general model can work, but the failures were repeated and domain-specific: alias ownership, return-ratio denominators, date boundaries, anti-join predicate placement, support-ticket filters, and duplicate-producing joins. Fine-tuning let me test whether targeted examples changed those decode paths while fixed eval gates protected existing behavior.

### How would another agent use it?

The agent would treat the model as a database-query tool. It would pass the user's natural-language question and database context to the endpoint, receive SQL, and then execute or validate that SQL in the surrounding system. The SFT lab measured direct SQL quality before adding repair, reranking, or agentic workflows.

### How was eval designed for effective measurement?

I used separate gates for separate questions. Dev gates were for iteration speed. Stable eval gates were protected holdouts. Challenge gates exposed brittleness. DB-disjoint gates tested whether same-DB learning transferred. Slice analysis explained failures by mechanism. SQL was evaluated by execution/result equivalence, not string match alone.

### What broke in production?

I would not claim this was deployed to real customers. The production-shaped failure to discuss is endpoint-candidate promotion risk: Exp062 looked better on the newest challenge but regressed protected eval. In a real agent workflow, that could mean the agent silently gets worse on existing user behaviors while the team celebrates a better new slice score.

### What changed from experiment to experiment?

The rule was to isolate the changed variable:

- Exp046 to Exp048: same LoRA recipe, changed train_v2 to train_v3 targeted data.
- Exp048 to Exp051-055: same model/prompt/LoRA recipe, added one supplement family at a time.
- Exp056 to Exp057: same train_v4 data, changed full LoRA to QLoRA.
- Exp056 to Exp059-061: same promoted recipe, added one hard-negative contrast family at a time.
- Exp062: combined the hard-negative families, then rejected because stable eval regressed.

### What did you learn about SFT?

SFT is controlled measurement plus data composition. The training API is the easy part. The hard part is deciding what changed, keeping evals fixed, understanding failure mechanisms, and rejecting endpoint candidates that only help the newest slice while making the consuming agent less reliable.

### What would you do next?

I would keep the lanes separate:

- one-shot SFT generator quality
- metadata/context retrieval
- candidate generation and pass@N
- execution-guided repair
- candidate selection or reward scoring
- agentic workflows

For the endpoint specifically, the next dev-loop slices are serving and runtime gates around the promoted adapter: temporary vLLM endpoint eval, load testing, schema validation, SQL execution safety, structured failure telemetry, latency/throughput checks, and agent-visible error handling.

## Evidence Map

Local repo artifacts:

- Experiment manifests: `experiments/sql/*.json`
- Training summaries: `artifacts/sql/*/train_summary.json`
- Eval results and slice analysis: `results/sql/*/*.json`
- Storefront train splits: `datasets/sql/train/storefront_sales_lab_*.jsonl`
- Storefront eval gates: `datasets/sql/eval/storefront_sales_lab_*.jsonl`
- SQL eval and analysis code: `src/sqlbench_lab/sql/eval_runner.py`, `src/sqlbench_lab/sql/eval_analysis.py`, `src/sqlbench_lab/sql/evaluator.py`
- Training path: `src/sqlbench_lab/sql/training.py`
- MLOps run contract: `src/sqlbench_lab/mlops/run_contract.py`
- Local/dev flow planning: `src/sqlbench_lab/mlops/offline_dev_flow.py`
- Metaflow entrypoint: `flows/sql_adapter_offline_dev_flow.py`

Linear evidence:

- `TAP-532`: practical fine-tuning learning ledger
- child learning issues include `TAP-533` through `TAP-543`, plus later `TAP-545`, `TAP-547`, and `TAP-549`
- parent comments record Exp029 through Exp062 decisions, including promoted and rejected checkpoints
- `TAP-648`: dev-only environment contract
- `TAP-631`: SQL adapter workflow contract and artifact layout
- `TAP-632`: local/dev Metaflow offline flow

## Guardrails

Do not say:

- "I got an official LiveSQLBench score."
- "The model generalizes to all SQL databases."
- "Exp062 was the best checkpoint."
- "QLoRA was better."
- "The one-DB lab proves production readiness by itself."
- "This was deployed to real customers" unless that is actually true.
- "The dev loop is production infrastructure."

Say instead:

- "These were lab/local results with explicit measurement boundaries."
- "Exp056 was the promoted full-LoRA one-DB checkpoint."
- "Exp062 improved the fresh challenge gate but was rejected because stable eval regressed."
- "QLoRA was a useful efficiency tradeoff, not the quality winner."
- "The main skill I can defend is the experiment discipline: fixed gates, isolated variables, failure taxonomy, and promote/reject decisions."
- "The product-shaped target was an endpoint consumed by another agent, and the lab measured whether a checkpoint was reliable enough to be an endpoint candidate."
- "The MLOps work is dev-only: it proves repeatability, artifact contracts, and promotion decisions before productionization."

## Core Sentence To Memorize

I built a measurement-driven SQL fine-tuning lab for a one-database text-to-SQL endpoint, where the important discipline was not just training adapters, but protecting eval gates, isolating data changes, classifying failures, and rejecting checkpoints that improved a new slice while regressing behavior another agent would rely on.
