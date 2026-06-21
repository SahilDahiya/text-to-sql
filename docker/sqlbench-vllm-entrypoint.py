"""Dev vLLM entrypoint that syncs one GCS LoRA adapter then starts OpenAI serving."""

from __future__ import annotations

import os
import shutil
import shlex
from pathlib import Path

from google.cloud import storage


def main() -> None:
    base_model = _required_env("SQLBENCH_BASE_MODEL")
    base_model_uri = os.environ.get("SQLBENCH_BASE_MODEL_URI", "").strip()
    base_served_model = _required_env("SQLBENCH_BASE_SERVED_MODEL")
    openai_model = _required_env("SQLBENCH_OPENAI_MODEL")
    adapter_name = _required_env("SQLBENCH_ADAPTER_NAME")
    adapter_uri = _required_env("SQLBENCH_ADAPTER_URI")
    host = os.environ.get("SQLBENCH_HOST", "0.0.0.0")
    port = _positive_int_env("SQLBENCH_PORT", 8000)
    max_model_len = _positive_int_env("SQLBENCH_MAX_MODEL_LEN", 4096)
    max_num_seqs = _positive_int_env("SQLBENCH_MAX_NUM_SEQS", 64)
    max_lora_rank = _positive_int_env("SQLBENCH_MAX_LORA_RANK", 16)
    gpu_memory_utilization = _float_env("SQLBENCH_GPU_MEMORY_UTILIZATION", 0.75)
    if openai_model != adapter_name:
        raise ValueError("SQLBENCH_OPENAI_MODEL must equal SQLBENCH_ADAPTER_NAME so clients request the LoRA model")

    adapter_dir = Path("/models/adapters") / adapter_name
    _materialize_uri_prefix(adapter_uri, adapter_dir)
    _assert_adapter_materialized(adapter_dir)
    if base_model_uri:
        base_model_dir = Path("/models/base") / _safe_name(base_model)
        _materialize_uri_prefix(base_model_uri, base_model_dir)
        _assert_base_model_materialized(base_model_dir)
        served_base_model = str(base_model_dir)
    else:
        served_base_model = base_model

    command = [
        "vllm",
        "serve",
        served_base_model,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        base_served_model,
        "--enable-lora",
        "--max-lora-rank",
        str(max_lora_rank),
        "--lora-modules",
        f"{adapter_name}={adapter_dir}",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.2f}",
    ]
    extra_args = os.environ.get("SQLBENCH_VLLM_EXTRA_ARGS", "").strip()
    if extra_args:
        command.extend(shlex.split(extra_args))
    os.execvp(command[0], command)


def _materialize_uri_prefix(uri: str, destination: Path) -> None:
    if uri.startswith("file://"):
        _copy_local_prefix(uri, destination)
        return
    _download_gcs_prefix(uri, destination)


def _copy_local_prefix(file_uri: str, destination: Path) -> None:
    source = Path(file_uri[len("file://") :]).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"local model or adapter directory does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source_path in source.rglob("*"):
        if not source_path.is_file():
            continue
        relative_name = source_path.relative_to(source)
        target = destination / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"no local files copied from URI: {file_uri}")


def _download_gcs_prefix(gcs_uri: str, destination: Path) -> None:
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        raise RuntimeError(f"no GCS objects found under adapter URI: {gcs_uri}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_name = blob.name[len(prefix) :].lstrip("/")
        if not relative_name:
            continue
        target = destination / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"no adapter files copied from GCS URI: {gcs_uri}")


def _parse_gcs_uri(value: str) -> tuple[str, str]:
    if not value.startswith("gs://"):
        raise ValueError("model or adapter URI must start with gs:// or file://")
    without_scheme = value[len("gs://") :].strip("/")
    if "/" not in without_scheme:
        raise ValueError("GCS model or adapter URI must include a bucket and prefix")
    bucket, prefix = without_scheme.split("/", 1)
    if not bucket or not prefix:
        raise ValueError("GCS model or adapter URI must include a bucket and prefix")
    return bucket, prefix.rstrip("/") + "/"


def _assert_adapter_materialized(adapter_dir: Path) -> None:
    required_files = ("adapter_config.json", "adapter_model.safetensors")
    missing = [filename for filename in required_files if not (adapter_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"downloaded adapter is missing required files: {', '.join(missing)}")


def _assert_base_model_materialized(base_model_dir: Path) -> None:
    required_files = ("config.json", "tokenizer_config.json")
    missing = [filename for filename in required_files if not (base_model_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"downloaded base model is missing required files: {', '.join(missing)}")
    weight_files = [
        path
        for pattern in ("*.safetensors", "*.bin")
        for path in base_model_dir.glob(pattern)
    ]
    if not weight_files:
        raise FileNotFoundError("downloaded base model is missing safetensors or bin weight files")


def _safe_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    resolved = "".join(safe).strip("_")
    if not resolved:
        raise ValueError("safe model name must not be empty")
    return resolved


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    value = default if raw_value is None else float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


if __name__ == "__main__":
    main()
