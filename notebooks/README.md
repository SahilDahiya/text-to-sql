# Notebooks

read_when: you want interactive training or evaluation runs

## SQL SFT Training

- `sql_sft_training_loop.ipynb` runs the current manifest-driven LoRA SFT path for
  `Qwen/Qwen3.5-0.8B-Base`.
- The notebook can enable MLflow logging for the run and writes local tracking state to
  `sqlite:///./mlflow.db` by default.

Open it from the repo root so relative artifact and dataset paths resolve cleanly.
