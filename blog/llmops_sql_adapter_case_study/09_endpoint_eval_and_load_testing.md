# Endpoint Eval and Load Testing

Correct once is not the same as reliable under traffic.

That is why endpoint eval and load testing are separate gates.

## Offline Eval vs Endpoint Eval

Offline eval runs the local evaluation path.

Endpoint eval calls the live OpenAI-compatible endpoint:

```text
HTTP request -> endpoint -> generated SQL -> eval result
```

Endpoint eval can catch serving-specific bugs:

- wrong OpenAI model name
- adapter not loaded
- base model served by accident
- prompt mismatch
- timeout
- server error
- context length mismatch

Offline eval cannot prove those things.

## Endpoint Eval Command Shape

The repo's SQL eval command can call an OpenAI-compatible endpoint:

```bash
uv run python -m sqlbench_lab.cli sql eval \
  --manifest experiments/sql/<experiment>.json \
  --model adapter \
  --dataset datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl \
  --openai-base-url http://127.0.0.1:8000/v1 \
  --openai-model <adapter_model_name> \
  --result-label endpoint_eval
```

The important part is:

```text
--openai-model must be the adapter model name
```

If it points at the base model, the endpoint eval is measuring the wrong model.

## Load Test Command Shape

Load testing probes whether the endpoint holds up under concurrent requests:

```bash
uv run python -m sqlbench_lab.cli sql openai-load-test \
  --manifest experiments/sql/<experiment>.json \
  --model adapter \
  --dataset datasets/sql/eval/storefront_sales_lab_eval_v1.jsonl \
  --openai-base-url http://127.0.0.1:8000/v1 \
  --openai-model <adapter_model_name> \
  --requests 32 \
  --concurrency 8 \
  --output artifacts/dev/<run_id>/load_test.json
```

Load test output tracks:

- request count
- concurrency
- success count
- failure count
- timeout count
- requests per second
- p50 latency
- p95 latency
- max latency
- generated SQL length stats

This is not the same as quality eval.

Quality eval asks:

```text
were the answers correct?
```

Load test asks:

```text
did the service respond reliably under expected pressure?
```

You need both.

## How Gates Work

The run contract can include:

- offline eval gates
- endpoint eval gate
- load-test gate

If endpoint eval is required but missing, the decision should be `investigate`.

If endpoint eval exists but fails threshold, the decision should be `reject`.

If load test exists but success rate is below threshold, the decision should be `reject`.

That keeps incomplete evidence separate from bad evidence.

## Interview Answer

```text
I separated endpoint eval from offline eval. Offline eval showed the adapter could generate correct SQL in the local path. Endpoint eval showed the live OpenAI-compatible service generated correct SQL when called like a real client. Then load testing measured request success, timeout count, concurrency, throughput, and latency.

That separation matters because an endpoint can fail for reasons unrelated to model quality: wrong model name, adapter not loaded, timeout, GPU memory pressure, or serving the base model by accident.
```

Short line:

```text
Endpoint eval tests correctness through the service. Load test tests service behavior under pressure.
```

## Sources

- `src/sqlbench_lab/cli.py`
- `src/sqlbench_lab/sql/eval_runner.py`
- `src/sqlbench_lab/mlops/run_contract.py`
- `tests/test_mlops_offline_dev_flow.py`
