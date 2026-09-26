# Batched Ψ₀ → SONIC → MJLab

Milestone 1 runtime for one Ψ₀ planner, one batched SONIC decoder, and multiple G1 environments. No RL, LoRA, dataset writer, or Dora is included.

SONIC is the only controller in Milestone 1. MJLab owns both `robot_state()` and batched `rgb()` observations.

## Model artifacts

Supply a SONIC compatible Ψ₀ run with `run_config.json`,
`clip_pooled_cache.pt`, and `checkpoints/ckpt_<step>/model.safetensors`.
Supply the SONIC `model_decoder.onnx`. The loader changes the public decoder's
fixed batch metadata to a dynamic batch axis in memory.

The checkpoint-compatible Ψ₀ inference implementation is vendored under
`src/planner/_psi0` from physical-superintelligence-lab/Psi0 commit
`4f3720d45e102b36d7c3e9465ab8062274170518`. The local project owns its full
Python dependency graph and does not require a Psi0 checkout or a second
environment. The Qwen processor/config and fallback CLIP encoder are also
loaded at revisions pinned in the source.

## Install

Install the pinned interpreter with `uv python install 3.12.13`. For local CPU
development, run `uv sync --extra cpu --frozen`. On a CUDA 12.8 machine, run
`uv sync --extra cu128 --frozen`. The extras are mutually
exclusive. To run the example on CPU, use `uv run --extra cpu --frozen` and
pass `--device cpu`.

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
uv run --extra cu128 --frozen hf download USC-PSI-Lab/psi-model \
  --revision 4c6f9776fc5b18d87945254175e38bb74b9d7748 \
  --include "psi0/sonic-checkpoints/multi-task.psi-dream.2609092156/**" \
  --local-dir artifacts/psi-model

uv run --extra cu128 --frozen hf download nvidia/GEAR-SONIC \
  --revision 6733128a3d8a523b1418b06bca3cdf61c8b0987f \
  --include model_decoder.onnx \
  --local-dir artifacts/sonic
```

## Run

```sh
uv run --extra cu128 --frozen python src/runtime.py \
  --num-envs 2 --chunks 5 \
  --psi-run-dir artifacts/psi-model/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156 \
  --ckpt-step 40000 \
  --sonic-bundle artifacts/sonic
```

Run with `--num-envs 1, 2, 4, 8` and compare the JSON latency, throughput, displacement, and peak Torch GPU allocation. The Ψ₀ model is invoked once per batch. SONIC also runs once per control step with a batch input. The planner action clock is 30 Hz; SONIC and MJLab run at 50 Hz. The checkpoint's `action_exec_horizon` determines when Ψ₀ replans.

The MJLab G1 has no actuated hands, so the Ψ₀ wrapper packs the 45-D checkpoint
state from 29 body joint positions in SONIC order, 14 neutral hand values, and
two zero padding values. The wrapper returns the first 64 dimensions of each
predicted action as the SONIC body token.

The observation camera uses SONIC's G1 head-camera mount and the ZED Mini WVGA
view used by the checkpoint data. MJLab renders the native 672×376 image; the
planner restores the dataset's eight-row bottom pad before applying the saved
480×270 checkpoint transform.
