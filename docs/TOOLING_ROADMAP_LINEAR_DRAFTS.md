# Tooling Roadmap Linear Drafts

read_when: Linear auth is unavailable and the SQLBench tooling roadmap needs issue tracking

These are the Linear issue drafts for the next tooling lane. Add them to the
`SQLBench Lab Training Program` project when the Linear connector is authenticated.

## Umbrella: Standardize SQL Training And Evaluation Tooling

Priority: High

Description:

Track the next infrastructure lane after exp006. The goal is to separate trainer
standardization, quantized training, inference serving, and optional external training
launcher work into clean experiments.

Order:

1. TRL SFTTrainer alternate backend.
2. bitsandbytes QLoRA option.
3. vLLM-backed eval path.
4. Axolotl recipe mirror only if the custom/TRL runner becomes limiting.

Non-goals:

- Do not change the SQL data recipe in the same issue as a tooling migration.
- Do not add Axolotl and TRL in the same experiment.
- Do not score LiveSQLBench until the fixed local eval path has parity.

## Issue: Adopt TRL SFTTrainer As Alternate SQL SFT Backend

Priority: High

Description:

Plan exp007 as a controlled trainer-backend migration.

Scope:

- Add `trl` dependency.
- Extend SQL SFT manifest/trainer config with a backend selector.
- Keep the existing `transformers.Trainer` backend working.
- Add a TRL `SFTTrainer` backend that reuses the repo-owned SQL prompt renderer.
- Start with packing disabled and the same train mix as exp006.
- Train `qwen35_0_8b__exp007_trl_sft_identifier_copy`.
- Eval on fixed BIRD 25 and Spider 25.

Success criteria:

- BIRD near exp006 adapter `2/25`.
- Spider near exp006 adapter `18/25`.
- No generated-format regression.
- Runtime is not worse than the current trainer path by an unacceptable amount.

Non-goals:

- Do not add bitsandbytes here.
- Do not change the dataset recipe.
- Do not add Axolotl here.

## Issue: Add bitsandbytes QLoRA Option

Priority: High

Description:

Plan exp008 after the TRL backend is working. The goal is memory-efficient LoRA training
without changing the data recipe.

Scope:

- Add `bitsandbytes` dependency.
- Add a manifest quantization config, default disabled.
- Support 4-bit NF4 loading through `BitsAndBytesConfig`.
- Keep non-quantized loading as the default path.
- Train `qwen35_0_8b__exp008_trl_qlora_identifier_copy`.
- Use the same train/eval mix as exp007.

Success criteria:

- BIRD stays near exp007.
- Spider stays near exp007.
- GPU memory drops meaningfully.
- Runtime remains acceptable.

Non-goals:

- Do not use bitsandbytes before the TRL path has parity.
- Do not change the dataset recipe in the QLoRA experiment.

## Issue: Add vLLM-Backed SQL Eval Path

Priority: Medium

Description:

Add a serving-backed evaluation path for faster fixed-set evals and future LiveSQLBench
readiness.

Scope:

- Add a vLLM eval backend beside the local `transformers.generate` backend.
- Support an OpenAI-compatible local server URL.
- Preserve the same result JSON and `.analysis.json` shape.
- Compare vLLM output parity and speed against local eval on fixed BIRD 25 and Spider 25.
- Document adapter serving or merged-adapter requirements.

Success criteria:

- Same eval cases and scoring code.
- Comparable pass/fail behavior on at least one existing adapter.
- Faster wall-clock eval or clear path to faster larger-set eval.

Non-goals:

- Do not use vLLM for training.
- Do not change result-equivalence scoring.

## Issue: Decide Whether To Mirror Training In Axolotl

Priority: Low

Description:

Evaluate Axolotl only after the TRL and QLoRA paths are understood. Axolotl may be useful
as a production-grade YAML launcher, but it should not replace the repo-owned runner until
we know the recipe is worth standardizing.

Scope:

- Create an Axolotl YAML that mirrors the best TRL/QLoRA experiment.
- Preserve the same prompt format and train/eval split.
- Train one adapter only if the YAML mirrors the repo path cleanly.
- Compare adapter behavior to the repo runner.

Success criteria:

- Axolotl recipe is reproducible from checked-in config.
- Result is comparable to the repo runner.
- Debuggability remains acceptable.

Non-goals:

- Do not add Axolotl before exp007/exp008.
- Do not treat Axolotl as the source of truth for prompt formatting.
