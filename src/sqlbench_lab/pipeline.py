"""The minimal two-task LiveSQLBench supervised fine-tuning loop."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL model. Return only the final SQL statement. "
    "Use the SQLite dialect and the declared schema."
)
PROTECTED_FIELDS = {"sol_sql", "test_cases", "external_knowledge"}
ALLOWED_TARGET_SOURCE = "independent_verified"


@dataclass(frozen=True)
class Task:
    task_id: str
    db_id: str
    question: str
    db_path: Path


@dataclass(frozen=True)
class Target:
    task_id: str
    split: str
    target_sql: str
    schema_text: str
    knowledge_text: str
    verification: dict[str, str]


@dataclass(frozen=True)
class Example:
    task_id: str
    db_id: str
    question: str
    schema_text: str
    knowledge_text: str
    db_path: Path
    target_sql: str
    split: str
    verification: dict[str, str]


def prepare(
    *,
    public_data: str | Path,
    target_manifest: str | Path,
    db_root: str | Path,
    train_output: str | Path,
    dev_output: str | Path,
) -> dict[str, Any]:
    """Verify exactly one train and one dev target, then write both artifacts."""

    targets = _load_targets(target_manifest)
    if len(targets) != 2 or {target.split for target in targets} != {"train", "dev"}:
        raise ValueError(
            "the first loop requires exactly one train target and one dev target"
        )
    tasks = _load_tasks(public_data, db_root, {target.task_id for target in targets})
    examples = []
    for target in targets:
        task = tasks.get(target.task_id)
        if task is None:
            raise ValueError(
                f"target references task absent from public data: {target.task_id}"
            )
        _verify_target(task.db_path, target.target_sql)
        examples.append(
            Example(
                task_id=task.task_id,
                db_id=task.db_id,
                question=task.question,
                schema_text=target.schema_text,
                knowledge_text=target.knowledge_text,
                db_path=task.db_path,
                target_sql=target.target_sql,
                split=target.split,
                verification=target.verification,
            )
        )

    train = [example for example in examples if example.split == "train"]
    dev = [example for example in examples if example.split == "dev"]
    _write_jsonl(train_output, [_example_payload(example) for example in train])
    _write_jsonl(dev_output, [_example_payload(example) for example in dev])
    return {
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_output": str(Path(train_output).resolve()),
        "dev_output": str(Path(dev_output).resolve()),
    }


def render_prompt(example: Example | dict[str, Any]) -> str:
    """Render the single prompt used by both training and generation."""

    if isinstance(example, Example):
        dialect = "sqlite"
        db_id = example.db_id
        schema_text = example.schema_text
        knowledge_text = example.knowledge_text
        question = example.question
    else:
        dialect = str(example["dialect"])
        db_id = str(example["db_id"])
        schema_text = str(example["schema_text"])
        knowledge_text = str(example["knowledge_text"])
        question = str(example["question"])
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nDialect:\n{dialect}\n\n"
        f"Database ID:\n{db_id}\n\n"
        f"Schema:\n{schema_text.strip()}\n\n"
        f"Verified knowledge:\n{knowledge_text.strip()}\n\n"
        f"Question:\n{question.strip()}\n"
        "<|assistant|>\n"
    )


def run_eval(
    *,
    dataset: str | Path,
    model_path: str | Path,
    output: str | Path,
    adapter_path: str | Path | None = None,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Run deterministic generation and local SQLite result evaluation."""

    examples = _load_artifact(dataset)
    tokenizer, model, torch = _load_model(model_path, adapter_path)
    records = []
    for example in examples:
        prompt = render_prompt(example)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_length = int(encoded["input_ids"].shape[-1])
        raw_prediction = tokenizer.decode(
            generated[0][input_length:], skip_special_tokens=True
        )
        predicted_sql = _extract_sql(raw_prediction)
        passed, error = _evaluate_sql(example, predicted_sql)
        records.append(
            {
                "task_id": example["task_id"],
                "db_id": example["db_id"],
                "predicted_sql": predicted_sql,
                "passed": passed,
                "error": error,
            }
        )
    summary = {
        "model": str(model_path),
        "adapter": str(adapter_path) if adapter_path is not None else None,
        "dataset": str(Path(dataset).resolve()),
        "records": records,
        "passed": sum(bool(record["passed"]) for record in records),
        "total": len(records),
    }
    _write_json(output, summary)
    return summary


def run_train(
    *,
    dataset: str | Path,
    model_path: str | Path,
    adapter_output: str | Path,
    max_length: int = 8192,
) -> dict[str, Any]:
    """Run one direct SQL LoRA SFT pass from the base model."""

    examples = _load_artifact(dataset)
    if not examples:
        raise ValueError("training dataset is empty")
    tokenizer, model, torch = _load_model(model_path, None, for_training=True)
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments

    encoded_examples = []
    for example in examples:
        prompt_ids = tokenizer(render_prompt(example), add_special_tokens=False)[
            "input_ids"
        ]
        target_ids = tokenizer(
            example["target_sql"] + (tokenizer.eos_token or ""),
            add_special_tokens=False,
        )["input_ids"]
        if len(prompt_ids) + len(target_ids) > max_length:
            raise ValueError(
                f"training example exceeds max_length: {example['task_id']}"
            )
        input_ids = prompt_ids + target_ids
        encoded_examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + target_ids,
            }
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    output_path = Path(adapter_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            fp16=False,
        ),
        train_dataset=_TrainingDataset(encoded_examples),
        data_collator=_Collator(tokenizer.pad_token_id, torch),
    )
    result = trainer.train()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    summary = {
        "model": str(model_path),
        "adapter_output": str(output_path),
        "train_rows": len(encoded_examples),
        "metrics": {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, (int, float))
        },
    }
    _write_json(output_path / "train_summary.json", summary)
    return summary


def _load_tasks(
    public_data: str | Path, db_root: str | Path, task_ids: set[str]
) -> dict[str, Task]:
    root = Path(db_root).resolve()
    tasks: dict[str, Task] = {}
    for row in _read_jsonl(public_data):
        task_id = _required_text(row, "instance_id")
        if task_id not in task_ids:
            continue
        if str(row.get("category", "")).casefold() != "query":
            raise ValueError(f"first loop accepts Query tasks only: {task_id}")
        for field in PROTECTED_FIELDS:
            value = row.get(field)
            if value not in (None, [], ""):
                raise ValueError(
                    f"protected public field is populated: {task_id}.{field}"
                )
        db_id = _required_text(row, "selected_database")
        db_dir = root / db_id
        schema_path = db_dir / f"{db_id}_schema.txt"
        db_path = db_dir / f"{db_id}_template.sqlite"
        if not schema_path.is_file() or not db_path.is_file():
            raise FileNotFoundError(
                f"missing public database files for {task_id}: {db_dir}"
            )
        if task_id in tasks:
            raise ValueError(f"duplicate public task: {task_id}")
        tasks[task_id] = Task(
            task_id=task_id,
            db_id=db_id,
            question=_required_text(row, "query"),
            db_path=db_path,
        )
    missing = sorted(task_ids - tasks.keys())
    if missing:
        raise ValueError(f"tasks missing from public data: {', '.join(missing)}")
    return tasks


def _load_targets(path: str | Path) -> list[Target]:
    targets = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        task_id = _required_text(row, "task_id")
        if task_id in seen:
            raise ValueError(f"duplicate target: {task_id}")
        seen.add(task_id)
        split = _required_text(row, "split")
        if split not in {"train", "dev"}:
            raise ValueError(f"unsupported first-loop split: {split}")
        source = _required_text(row, "target_source")
        if source != ALLOWED_TARGET_SOURCE:
            raise ValueError("targets must be independently verified")
        verification = row.get("verification")
        if (
            not isinstance(verification, dict)
            or verification.get("status") != "execution_verified"
        ):
            raise ValueError(f"target is not execution_verified: {task_id}")
        for field in ("verified_by", "verification_id", "verified_at"):
            _required_text(verification, field)
        targets.append(
            Target(
                task_id=task_id,
                split=split,
                target_sql=_required_text(row, "target_sql"),
                schema_text=_required_text(row, "schema_text"),
                knowledge_text=_required_text(row, "knowledge_text"),
                verification={
                    field: str(verification[field])
                    for field in (
                        "status",
                        "verified_by",
                        "verification_id",
                        "verified_at",
                    )
                },
            )
        )
    if not targets:
        raise ValueError("target manifest is empty")
    return targets


def _verify_target(db_path: Path, target_sql: str) -> None:
    first_word = target_sql.lstrip().split(None, 1)[0].casefold()
    if first_word not in {"select", "with"}:
        raise ValueError("first-loop targets must be read-only SELECT statements")
    if ";" in target_sql.rstrip().rstrip(";"):
        raise ValueError("target must contain one SQL statement")
    observations = []
    for _ in range(2):
        with _read_only_connection(db_path) as connection:
            cursor = connection.execute(target_sql)
            rows = cursor.fetchall()
            columns = tuple(description[0] for description in cursor.description or ())
            observations.append((columns, rows))
    if observations[0] != observations[1]:
        raise ValueError(f"target result is not deterministic: {db_path}")


def _load_artifact(path: str | Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    for row in rows:
        for field in (
            "task_id",
            "db_id",
            "question",
            "schema_text",
            "knowledge_text",
            "db_path",
            "target_sql",
        ):
            _required_text(row, field)
    return rows


def _read_only_connection(db_path: str | Path) -> sqlite3.Connection:
    resolved = Path(db_path).resolve()
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def _evaluate_sql(
    example: dict[str, Any], predicted_sql: str
) -> tuple[bool, str | None]:
    try:
        with _read_only_connection(example["db_path"]) as connection:
            predicted = connection.execute(predicted_sql)
            predicted_rows = predicted.fetchall()
            predicted_columns = tuple(
                description[0] for description in predicted.description or ()
            )
            gold = connection.execute(example["target_sql"])
            gold_rows = gold.fetchall()
            gold_columns = tuple(
                description[0] for description in gold.description or ()
            )
        return predicted_columns == gold_columns and predicted_rows == gold_rows, None
    except sqlite3.Error as exc:
        return False, str(exc)


def _load_model(
    model_path: str | Path,
    adapter_path: str | Path | None,
    *,
    for_training: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    if not torch.cuda.is_available():
        raise RuntimeError("the first loop requires CUDA")
    root = Path(model_path).resolve()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(root / "tokenizer.json"),
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    model = AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        dtype=torch.float32 if for_training else torch.float16,
    ).cuda()
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, Path(adapter_path).resolve(), is_trainable=False
        )
    if for_training:
        model.train()
    else:
        model.eval()
    return tokenizer, model, torch


def _extract_sql(text: str) -> str:
    cleaned = text.strip()
    if "<|assistant|>" in cleaned:
        cleaned = cleaned.split("<|assistant|>", 1)[-1].strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    match = re.search(r";", cleaned)
    return cleaned[: match.end()].strip() if match else cleaned


def _example_payload(example: Example) -> dict[str, Any]:
    return {
        "task_id": example.task_id,
        "db_id": example.db_id,
        "dialect": "sqlite",
        "question": example.question,
        "schema_text": example.schema_text,
        "knowledge_text": example.knowledge_text,
        "db_path": str(example.db_path),
        "target_sql": example.target_sql,
        "source_split": example.split,
        "verification": example.verification,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"JSONL row must be an object at line {line_number}: {path}"
            )
        rows.append(payload)
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _required_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


class _TrainingDataset:
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class _Collator:
    def __init__(self, pad_token_id: int, torch: Any) -> None:
        self.pad_token_id = pad_token_id
        self.torch = torch

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention_mask, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }
