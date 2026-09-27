# BONES BC validation

The permanent BC component trains the existing local Ψ₀ policy on
`walk and turn around`, then loads its adapter into the same Ψ₀ → SONIC →
MJLab path. The objective is recognizable, more reliable physical execution.
Lower offline loss alone does not complete this experiment.

## Data and policy contract

The pinned dataset is [wsagi/SONIC-VLA-BonesSeed-V2](https://huggingface.co/datasets/wsagi/SONIC-VLA-BonesSeed-V2/tree/4d84c8009ad601d24c1fe493c45b2715536b7bef).
Walking episodes are 47–53. The local reader preserves the upstream converter's
body and Dex3 ordering, using `action.wbc` for demonstrated hand targets.

| Tensor | Layout |
| --- | --- |
| State | 29 body joints, 14 reordered Dex3 joints, 2 absent neck channels |
| Action | 64 SONIC tokens, 14 reordered Dex3 targets, 2 absent neck channels |
| Loss mask | First 78 dimensions valid, final 2 ignored |
| Chunk | 30 future actions, repeating the last action at episode boundaries |

Dimensions and chunk length come from the starting checkpoint's
`run_config.json`; normalization uses its saved bounds. Constant state channels
become zero, constant action channels retain their values, and normalized
values are clipped to `[-1, 1]`, matching upstream. Some raw BONES tokens are
below `-0.625`; per-episode token/hand ranges and fractions outside checkpoint
bounds are printed and saved in the BC config so clipping is visible.

The source recordings run at 50 Hz. `dataset.action_hz` defaults to 30 Hz to
match the current planner runtime. The reader selects nearest source frames
on that physical clock, retaining alignment of image, state, and action.
This differs from the upstream raw converter, which keeps every source frame
and writes 30 Hz timestamps. Do not change the training clock without changing
the rollout clock consistently.

The BC objective uses uniform `sigma`, Gaussian noise, interpolated noisy
actions, and the `noise - expert_action` velocity target. Its reduction matches
upstream: sum over time, mean over batch and all 80 action dimensions after
masking. The saved state-token dropout probability is used during training.
Other augmentations are disabled initially.

Following `AIM.md`, all base weights stay frozen. Rank-16 LoRA adapts the action
expert's used attention projections, with alpha equal to rank and fp32 adapter
weights. The VLM and CLIP text encoder remain frozen and in evaluation mode.
There is one local policy implementation for training and inference.

## Artifacts and commands

Use the Ψ₀ SONIC, Qwen processor, CLIP, and SONIC artifacts documented in
[README.md](README.md). Install the same pinned environment with
`uv sync --extra cu128 --frozen`. Download only the walking data:

```sh
uvx hf download wsagi/SONIC-VLA-BonesSeed-V2 --repo-type dataset \
  --revision 4d84c8009ad601d24c1fe493c45b2715536b7bef \
  --include 'meta/*' \
  --include 'data/chunk-000/episode_00004[7-9].parquet' \
  --include 'data/chunk-000/episode_00005[0-3].parquet' \
  --include 'videos/chunk-000/observation.images.ego_view/episode_00004[7-9].mp4' \
  --include 'videos/chunk-000/observation.images.ego_view/episode_00005[0-3].mp4' \
  --local-dir artifacts/bones-walk
```

Phase 0 loads the base checkpoint strictly and checks a real observation:

```sh
uv run --extra cu128 --frozen python src/train_bc.py \
  --baseline-only --output results/bones-base-offline
```

This writes shape/range/flow-loss metrics, the transformed observation,
instruction, state, pooled text, sampled actions, and RNG states. These saved
inputs support comparison with a separately run original Ψ₀ environment.
Full-model numerical and physical checks remain necessary before training.

The default [config](configs/bc_bones_walk.json) selects only episode 47:

```sh
uv run --extra cu128 --frozen python src/train_bc.py \
  --output results/bones-overfit
```

The training command performs the checkpoint and sample checks before enabling
adapters or creating the optimizer. Confirm strong one-episode fitting before
expanding the split. It logs training flow loss, optional validation flow loss,
learning rate, step, and gradient norm; non-finite loss or gradients abort the
run. Validation uses a fixed flow RNG and disables state dropout. Outputs include
`metrics.jsonl`, `bc_config.json`, and `ckpt_<step>/adapter.safetensors` with
the config used for that checkpoint. Output directories must be new. Training
resume and optimizer-state checkpoints are not implemented.

For Phase 2, copy the config and set `train_episodes` to `[47,48,49,50,51]`,
`val_episodes` to `[52,53]`, and adjust the training step budget. Pass the new
file with `--config`. The split is at episode level: 723 training and 240
validation observations after resampling. At most the seven selected videos
are cached in CPU memory; frame shuffling does not repeatedly decode videos.

Evaluate a trained adapter on held-out observations with the split config:

```sh
uv run --extra cu128 --frozen python src/train_bc.py \
  --config configs/bc_bones_walk_split.json --baseline-only \
  --bc-checkpoint results/bones-walk/ckpt_2000 \
  --output results/bones-bc-offline
```

The adapter config records the base step and configuration hash. Both are
checked on load; the original pinned base weights remain required.

For the physical comparison, run the same command with and without the adapter:

```sh
uv run --extra cu128 --frozen python src/evaluate_bc.py \
  --num-envs 1 --seconds 10 --record-video results/bones-base.mp4 \
  --output results/bones-base-rollout.json

uv run --extra cu128 --frozen python src/evaluate_bc.py \
  --bc-checkpoint results/bones-overfit/ckpt_500 \
  --num-envs 1 --seconds 10 --record-video results/bones-bc.mp4 \
  --output results/bones-bc-rollout.json
```

This small synchronous evaluation consumes a full chunk before replanning.
Both runs use the same configured seed and action clock. It records path
length, displacement, net/max heading change, and falls for each environment.
Judge walk-and-turn behavior from these measurements and video; no reward or
automatic success threshold is introduced. Only the first 64 predicted channels
are sent to SONIC. Hand targets stay in BC, while rollout has no Dex3 control.

## Demonstration replay

Replay the demonstrated body tokens before interpreting a missing turn as a
BC failure:

```sh
uv run --extra cu128 --frozen python src/evaluate_bc.py \
  --replay-episode 47 --record-video results/bones-replay-47.mp4 \
  --output results/bones-replay-47.json
```

Replay starts from the same standing MJLab reset as policy evaluation. It uses
the raw 64-D body tokens at the source 50 Hz rate, without normalization,
quantization, or clipping, and stops at the episode end (9.1 seconds for episode
47). It requires only the dataset and SONIC decoder, not Ψ₀ artifacts. The JSON
also reports the heading change recorded in the demonstration. If native replay
works, repeat with `--replay-hz 30` and distinct output/video paths to check the
resampling used by BC. A failed replay can reflect initial-state or simulator
differences; it does not by itself establish a training failure.

Episode 47's recorded root orientation shows approximately 0.885 rad (51°)
maximum heading change and 0.002 rad net change. Compare replay against these
recorded values rather than assuming the task label implies a 180° turn.

## Validation status

Locally checked: all seven real walking episodes load; hand/state conversion,
normalization, chunk boundaries, upstream action-expert numerical regression,
strict small-checkpoint loading, flow loss, adapter gradients, optimizer updates,
save/reload, and Euler sampling have focused tests.

Pending on the training machine: the full released-checkpoint/VLM regression,
one-episode overfit, 5/2 training, and original-versus-BC physical execution.
The local development workspace has CPU Torch and no released Ψ₀ artifacts.
