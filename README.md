# Batched Ψ₀ → SONIC → MJLab

One local Ψ₀ planner, one batched SONIC decoder, and multiple G1 environments,
with a small flow-matching BC component for demonstrated SONIC actions.

SONIC is the only controller in Milestone 1. MJLab owns both `robot_state()` and batched `rgb()` observations.

The [BONES BC experiment](BC_BONES.md) documents the local dataset conversion,
one-episode overfit, 5/2 episode split, adapter training, and physical comparison.
It uses the same checkpoint-compatible policy as the runtime. Its frozen base
and action-expert LoRA follow `AIM.md`.

## Model artifacts

Supply a SONIC compatible Ψ₀ run with `run_config.json`,
`clip_pooled_cache.pt`, and `checkpoints/ckpt_<step>/model.safetensors`.
Supply the SONIC `model_decoder.onnx`. The loader changes the public decoder's
fixed batch metadata to a dynamic batch axis in memory.

The checkpoint-compatible Ψ₀ inference implementation is vendored under
`src/planner/_psi0` from physical-superintelligence-lab/Psi0 commit
`4f3720d45e102b36d7c3e9465ab8062274170518`. The local project owns its full
Python dependency graph and does not require a Psi0 checkout or a second
environment. Qwen's processor files and the CLIP text encoder are downloaded
as pinned local artifacts; the runtime does not fetch model files.

## Install

Install the pinned interpreter with `uv python install 3.12.13`. For local CPU
development, run `uv sync --extra cpu --frozen`. On a CUDA 12.8 machine, run
`uv sync --extra cu128 --frozen`. The extras are mutually
exclusive. Ruff, ty, and pytest are pinned in the development dependency group.
To run the example on CPU, use `uv run --extra cpu --frozen` and pass
`--device cpu`.

On the remote RTX 5090, confirm that Torch and ONNX Runtime see CUDA before
downloading the model artifacts:

```sh
nvidia-smi
uv run --extra cu128 --frozen python -c \
  'import onnxruntime as ort, torch; ort.preload_dlls(); print(torch.__version__, torch.cuda.get_device_name(), ort.get_available_providers())'
```

The output must name the RTX 5090 and include `CUDAExecutionProvider`.

Download the released artifacts at pinned revisions:

```sh
uvx hf download USC-PSI-Lab/psi-model \
  --revision 4c6f9776fc5b18d87945254175e38bb74b9d7748 \
  --include "psi0/sonic-checkpoints/multi-task.psi-dream.2609092156/**" \
  --local-dir artifacts/psi-model

uvx hf download nvidia/GEAR-SONIC \
  --revision 6733128a3d8a523b1418b06bca3cdf61c8b0987f \
  --include model_decoder.onnx \
  --local-dir artifacts/sonic

uvx hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --exclude "*.safetensors" --exclude "*.bin" \
  --local-dir artifacts/qwen3-vl-2b-instruct

uvx hf download openai/clip-vit-large-patch14 \
  config.json merges.txt model.safetensors special_tokens_map.json \
  tokenizer.json tokenizer_config.json vocab.json \
  --revision 32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --local-dir artifacts/clip-vit-large-patch14
```

## Run

```sh
uv run --extra cu128 --frozen python src/runtime.py \
  --num-envs 1 --num-episodes 1 --viewer \
  --psi-run-dir artifacts/psi-model/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156 \
  --ckpt-step 40000 \
  --qwen-model artifacts/qwen3-vl-2b-instruct \
  --clip-model artifacts/clip-vit-large-patch14 \
  --sonic-bundle artifacts/sonic
```

The viewer is optional and passive. Without `--viewer`, no interactive viewer
is constructed and no viewer state is copied from GPU to CPU. Record the run
with `--record-video results/walk.mp4` when needed.

For batched headless evaluation:

```sh
uv run --extra cu128 --frozen python src/runtime.py \
  --num-envs 32 --num-episodes 256 \
  --psi-run-dir artifacts/psi-model/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156 \
  --ckpt-step 40000 \
  --qwen-model artifacts/qwen3-vl-2b-instruct \
  --clip-model artifacts/clip-vit-large-patch14 \
  --sonic-bundle artifacts/sonic \
  --save-metrics results/walk-to-target.json
```

`WalkToTarget-v0` places a non-colliding green marker 2 m in front of G1. Its
reward is progress, with a +5 success bonus and a -5 fall penalty. Success
requires the upright robot to remain within 0.2 m of the target below 0.2 m/s
for 0.5 s. The JSON contains per-episode outcomes, aggregate success and fall
rates, and runtime throughput. The planner action clock is 30 Hz; SONIC and
MJLab run at 50 Hz. After executing 15 of a chunk's 30 actions, Ψ₀ generates
the next chunk on a separate CUDA stream while simulation continues. Test-time
RTC guides the new chunk toward the shifted previous chunk. The
`rtc_deadline_misses` metric counts replans that did not finish before the
current chunk expired.

The MJLab G1 has no actuated hands or neck, so the Ψ₀ wrapper packs its 45-D
state as 12 leg joints, three waist joints, 14 arm joints, 14 neutral hand
values, and two neutral neck values. The wrapper returns the first 64 dimensions
of each predicted action, snapped to SONIC's 1/16 finite scalar quantization
grid in the controller's [-0.625, 0.625] token range, as the body token.

The observation camera uses SONIC's G1 head-camera mount and the ZED Mini WVGA
view used by the checkpoint data. MJLab renders the native 672×376 image; the
planner restores the dataset's eight-row bottom pad before applying the saved
480×270 checkpoint transform.

## Push-box diagnostic probe

`--push-box` selects a flat scene with a 3 kg, 36 cm box centered 1.5 m in
front of G1. The same head camera and Ψ₀ → 64-D SONIC body token → SONIC
controller path run without assistance. `--prompt` supplies the task text;
the default remains the walk-to-target instruction for the original scene.

Run each base prompt with the same seed and episode length. For example, use
these four prompts with `--push-box --seed 0 --episode-seconds 8` and the model
artifact flags shown above:

```text
walk forward
go to the box
push the box forward
go to the box and push it forward
```

One invocation looks like:

```sh
uv run --extra cu128 --frozen python src/runtime.py \
  --push-box --prompt "go to the box" --seed 0 --episode-seconds 8 \
  --psi-run-dir artifacts/psi-model/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156 \
  --ckpt-step 40000 \
  --qwen-model artifacts/qwen3-vl-2b-instruct \
  --clip-model artifacts/clip-vit-large-patch14 \
  --sonic-bundle artifacts/sonic \
  --save-metrics results/push-box/base-go-to-box.json \
  --record-video results/push-box/base-go-to-box.mp4
```

The JSON records root and box displacement, heading change, robot-to-box
distance at each control step and its minimum, contact, fall, Ψ₀ inference
latencies, termination reason, and a behavior label. Labels use simple distance,
contact, and displacement thresholds; inspect the video when judging whether
motion is useful. The run ends on a fall or the time limit. After recording all
base runs, repeat each command with `--bc-checkpoint PATH`, keeping every other
argument and the scene unchanged. Use distinct output paths for those runs.
