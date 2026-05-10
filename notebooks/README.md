# Notebooks

read_when: you want interactive training or evaluation runs

## SQL SFT Training

- `sql_sft_training_loop.ipynb` runs the current manifest-driven LoRA SFT path for
  `Qwen/Qwen3.5-0.8B-Base`.
- The notebook can enable MLflow logging for the run and writes local tracking state to
  `sqlite:///./mlflow.db` by default.

## Experiment Exploration

- `sql_exp002_to_exp004_explorer.ipynb` compares the exp002, exp003, and exp004
  one-shot SQL artifacts. It is intentionally breadcrumb-style: use it to inspect
  manifest changes, dataset shape, pass-rate movement, failure buckets, and
  exp003/exp004 case flips.

Open it from the repo root so relative artifact and dataset paths resolve cleanly.
