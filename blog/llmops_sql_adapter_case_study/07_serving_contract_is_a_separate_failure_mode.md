# Serving Contract Is a Separate Failure Mode

Offline eval passing does not prove serving works.

That is the core serving lesson from this repo.

Offline eval answers:

```text
Can the adapter generate correct SQL in the local eval path?
```

Serving answers:

```text
Can a live endpoint load the right base model, load the right LoRA adapter, expose the right model name, and respond like a real client expects?
```

Those are different questions.

## What the Endpoint Had to Know

The dev endpoint plan captured the serving contract:

- environment: `dev`
- serving target: local GPU Docker, GCE GPU VM, GKE GPU node pool, or Vertex custom GPU endpoint
- rejected target: Cloud Run GPU for the current image/runtime constraint
- base model
- optional mirrored base model URI
- adapter name
- adapter URI
- OpenAI model name clients should request
- max model length
- max number of sequences
- max LoRA rank
- GPU memory utilization
- health path
- completions path

The key detail:

```text
clients must request the LoRA adapter model ID, not the base served model alias
```

If the endpoint serves the base model by accident, offline adapter quality is irrelevant.

## Base Model vs Adapter Model

The vLLM server loads:

```text
base model + LoRA adapter
```

The base model is the foundation model:

```text
Qwen/Qwen3.5-0.8B-Base
```

The adapter is the trained SQL behavior:

```text
qwen35_0_8b_storefront_v4_lora_r16_a32_d010_exp056
```

The endpoint contract keeps these names separate:

- base served model name
- OpenAI model name
- adapter name

The entrypoint enforces:

```text
SQLBENCH_OPENAI_MODEL must equal SQLBENCH_ADAPTER_NAME
```

That is a safety check. It prevents clients from accidentally calling the base model when they meant to call the LoRA adapter.

## Why Local GPU Rehearsal Matters

The local GPU Docker target is useful because it rehearses the same serving contract before moving to cloud.

It uses:

- same serving image
- same environment variables
- same adapter materialization path
- same OpenAI-compatible endpoint
- local NVIDIA runtime
- local GCP credentials mount when reading GCS

This is not production. But it is stronger than saying "the model worked in Python."

It tests the serving path.

## What Can Break Only at Serving Time

Serving introduces failures offline eval does not catch:

- adapter URI points to missing files
- adapter is missing `adapter_config.json`
- adapter is missing `adapter_model.safetensors`
- base model mirror is incomplete
- model name sent by client is wrong
- LoRA rank does not match runtime config
- context length is too small
- GPU memory utilization is too high
- CUDA/driver mismatch
- endpoint starts but times out
- endpoint serves base model instead of adapter

That is why serving deserves its own gate.

## Interview Answer

```text
I treated serving as a separate failure mode. Offline eval proved the adapter could generate SQL in the local eval path, but endpoint eval had to prove the live service loaded the right base model, loaded the right LoRA adapter, exposed the adapter model name, and responded through the OpenAI-compatible API.

The serving contract made base model, adapter name, adapter URI, model name, max context length, LoRA rank, GPU memory settings, and health/completions paths explicit. That prevented a common production bug: thinking the adapter is served when the endpoint is actually serving the base model.
```

Short line:

```text
Serving is not just deployment. It is another eval surface.
```

## Sources

- `src/sqlbench_lab/mlops/dev_endpoint.py`
- `docker/sqlbench-vllm-entrypoint.py`
- `tests/test_mlops_dev_cloud_contracts.py`
