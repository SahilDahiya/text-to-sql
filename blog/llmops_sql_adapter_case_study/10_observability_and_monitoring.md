# Observability and Monitoring

LLMOps monitoring is not only uptime.

For a text-to-SQL endpoint, monitoring has to answer two questions:

```text
Is the service healthy?
Is the model behavior still useful?
```

Infrastructure monitoring answers the first question.

Quality monitoring answers the second.

## Run Observability

A dev run observability record tracks:

- environment
- experiment ID
- adapter name
- base model
- run ID
- git SHA
- container image URI
- train row count
- train runtime
- eval results
- failure counts
- decision
- passed gates
- failed gates
- GCS prefix
- current pointer URI

This makes a run inspectable after it finishes.

Without this, you get a common ML failure mode:

```text
the run happened, but nobody can reconstruct what changed or why it passed
```

## Endpoint Monitoring

Endpoint monitoring tracks serving behavior:

- endpoint ID
- serving target
- OpenAI model name
- adapter name
- base model
- endpoint image URI
- endpoint logs URI
- startup time
- GPU memory notes
- failure mode
- request count
- success count
- failure count
- timeout count
- concurrency
- requests per second
- p50/p95/p99 latency
- generated SQL length
- endpoint eval pass rate
- syntax failure count
- schema failure count
- runtime failure count
- empty SQL count

This combines quality and infrastructure signals.

A normal service dashboard might stop at latency and error rate.

An LLMOps dashboard also needs failure shape:

```text
are SQL syntax failures increasing?
are schema failures increasing?
are empty SQL responses increasing?
did repair rate spike?
```

## Cost and Capacity

Cost/capacity records track:

- training machine type
- training accelerator type/count
- training runtime hours
- training estimated cost
- endpoint machine type
- endpoint accelerator type/count
- endpoint uptime hours
- endpoint estimated cost
- request count
- peak concurrency
- requests per second
- capacity ladder points

This matters because model choices affect serving economics.

QLoRA might help memory/cost, but if it regresses protected eval it is not the quality winner.

Cost does not replace quality gates. It sits next to them.

## Why Monitoring Must Include Failure Counts

For text-to-SQL, a request can fail in multiple ways:

- endpoint timeout
- model returns empty SQL
- SQL syntax error
- SQL schema error
- SQL runtime error
- SQL executes but returns wrong rows

If all of these become one "error" number, the next action is unclear.

Failure counts make the next action easier:

- syntax failures may need repair or decoding constraints
- schema failures may need better schema context
- row mismatches may need new eval slices or data changes
- timeouts may need serving/capacity work

## Interview Answer

```text
I treated observability as part of the model lifecycle. For each dev run, I recorded experiment ID, adapter, base model, git SHA, train rows, eval scores, failure counts, promotion decision, and artifact prefix. For endpoints, I tracked request volume, success/error/timeout counts, latency, generated SQL length, endpoint eval pass rate, and SQL failure buckets like syntax, schema, and runtime errors.

The point was that LLMOps monitoring is not just uptime. It has to show whether model quality and failure shape are changing.
```

Short line:

```text
Infrastructure monitoring tells me whether the service is up. Quality monitoring tells me whether it is still useful.
```

## Sources

- `src/sqlbench_lab/mlops/observability.py`
- `src/sqlbench_lab/mlops/endpoint_monitoring.py`
- `src/sqlbench_lab/mlops/cost_capacity.py`
