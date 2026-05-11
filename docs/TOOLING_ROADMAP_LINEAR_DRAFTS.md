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
2. TRL packing/bf16/tf32 optimization pass.
3. Liger Kernel acceleration trial if packing helps but runtime is still high.
4. bitsandbytes QLoRA option.
5. vLLM-backed eval path.
6. Axolotl recipe mirror only if the custom/TRL runner becomes limiting.

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

## Comment: Exp007 TRL Baseline Result And Next Optimization Split

Use this as a Linear comment on the TRL backend issue once Linear auth is available.

Exp007 is implemented and trained as the controlled TRL SFTTrainer backend gate.

Result:

- BIRD fixed eval improved to `3/25`, the best local fixed-BIRD result so far.
- Spider fixed eval dropped to `17/25` versus exp006's `18/25` guardrail.
- Train runtime was about `1346s`, slower than exp006's about `1026s` on the same train mix.

Read:

TRL is worth keeping, but the first TRL run is not enough to call the backend migration
done. The BIRD gain is useful; the Spider and runtime regressions mean the next work should
optimize the TRL recipe before adding QLoRA or Axolotl.

Recommended next split:

- exp008: TRL packing plus explicit `bf16`/`tf32`, same data recipe.
- exp009: Liger Kernel on top of the best exp008 recipe if runtime remains high.
- exp010: bitsandbytes QLoRA only after the optimized TRL full-precision path is understood.

Success bar for exp008:

- BIRD stays at or above `3/25`.
- Spider recovers to at least `18/25`.
- Runtime improves versus exp007 or the slower runtime is justified by better eval quality.

Do not mix data changes into exp008. This should stay a tooling/runtime experiment.

## Issue: Optimize TRL SFTTrainer With Packing And Precision Flags

Priority: High

Description:

Plan exp008 as a controlled TRL optimization pass. Keep the exp007 data recipe, prompt,
LoRA config, and eval sets fixed while testing TRL packing and explicit precision settings.

Scope:

- Extend the SQL trainer manifest config with TRL-specific fields:
  - `packing`
  - `packing_strategy`
  - `max_length`
  - `bf16`
  - `tf32`
  - `gradient_checkpointing`
- Pass those fields through the TRL `SFTConfig`.
- Create `qwen35_0_8b__exp008_trl_packing_identifier_copy`.
- Start with `packing=true`, `packing_strategy=bfd`, `max_length=1024`, `bf16=true`,
  `tf32=true`, and gradient checkpointing disabled unless memory requires it.
- Train and eval on fixed BIRD 25 and Spider 25.
- Log MLflow metrics for train runtime, token count, mean token accuracy, BIRD pass rate,
  and Spider pass rate.

Success criteria:

- BIRD is at least exp007 `3/25`.
- Spider returns to at least exp006 `18/25`.
- Runtime improves against exp007's about `1346s`, or quality improvement justifies the cost.
- Manifest defaults keep older experiments reproducible.

Non-goals:

- Do not add bitsandbytes in exp008.
- Do not add Liger Kernel until packing behavior is measured.
- Do not change train/eval rows.

## Issue: Trial Liger Kernel For TRL Runtime

Priority: Medium

Description:

Plan exp009 only after exp008 has a stable packed TRL recipe. The goal is runtime
improvement without changing data or scoring.

Scope:

- Add optional `liger-kernel` dependency only if the installed stack supports it cleanly.
- Add a manifest flag for `use_liger_kernel`, default disabled.
- Create `qwen35_0_8b__exp009_trl_liger_identifier_copy` from the best exp008 manifest.
- Train and eval on the same fixed BIRD 25 and Spider 25.
- Compare runtime and eval quality against exp008.

Success criteria:

- Eval quality is no worse than exp008 within one case on each fixed eval set.
- Runtime or memory behavior improves enough to justify the dependency.
- The flag is optional and older experiments are unaffected.

Non-goals:

- Do not introduce Liger before exp008 packing is measured.
- Do not combine with bitsandbytes in the first Liger trial.

## Issue: Add bitsandbytes QLoRA Option

Priority: High

Description:

Plan exp010 after the TRL backend and packing/precision pass are understood. The goal is
memory-efficient LoRA training without changing the data recipe.

Scope:

- Add `bitsandbytes` dependency.
- Add a manifest quantization config, default disabled.
- Support 4-bit NF4 loading through `BitsAndBytesConfig`.
- Keep non-quantized loading as the default path.
- Train `qwen35_0_8b__exp010_trl_qlora_identifier_copy`.
- Use the same train/eval mix as the best optimized TRL run.

Success criteria:

- BIRD stays near exp007.
- Spider stays near exp007.
- GPU memory drops meaningfully.
- Runtime remains acceptable.

Non-goals:

- Do not use bitsandbytes before the optimized TRL path has parity.
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
