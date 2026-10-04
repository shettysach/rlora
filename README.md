# Carry-box probe

This branch ports the scene for `simple/G1WholebodyXMoveBendCarryBoxSonic-v0` to
the batched MJLab runtime. G1 has 29 SONIC-controlled body joints and 14
position-controlled Dex3 joints. The scene has a floor box and a table inside
SIMPLE's furnished HSSD room (`hssd:scene0`). It uses
the published 78D action layout: 64 SONIC body-token values followed by 14 hand
targets, at 50 Hz. The 43D planner state contains the same body and hand joint
order as the [published evaluation data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval).

The task succeeds after the box is released onto the table's upper surface for
over 0.9 seconds, matching the [SIMPLE task](https://github.com/physical-superintelligence-lab/SIMPLE/blob/main/src/simple/tasks/g1_wholebody_xmove_bend_carry_box_sonic.py).
`--mode replay` feeds recorded actions to the decoder. `--mode policy` runs the
published fine-tuned Psi0 checkpoint directly for action chunks. Both modes execute
body and hand actions in MJLab.
Replay actions are copied to the device once at startup and sliced into chunks.

## Setup

```sh
uv sync --extra cu128 --frozen

uv run --extra cu128 --frozen --with usd-core==26.8 python tools/setup_hssd.py

uvx hf download USC-PSI-Lab/psi-data \
  --repo-type dataset \
  --include simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip \
  --local-dir artifacts/psi-data

uvx hf download nvidia/GEAR-SONIC model_decoder.onnx \
  --revision 6733128a3d8a523b1418b06bca3cdf61c8b0987f \
  --local-dir artifacts/sonic

uvx hf download USC-PSI-Lab/psi-model \
  --include 'psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223/run_config.json' \
  --include 'psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223/checkpoints/ckpt_40000/model.safetensors' \
  --local-dir artifacts/psi-model

uvx hf download Qwen/Qwen3-VL-2B-Instruct \
  --include '*.json' '*.txt' '*.model' \
  --local-dir artifacts/qwen3-vl-2b
```

The runtime checks the decoder's SHA-256 so a v1.1 decoder cannot silently
consume the older task's body tokens. The Dex3 MJCF and meshes came from the
existing `dex3_hands` branch; source and license are recorded in
`assets/g1/`.
`tools/setup_hssd.py` downloads the pinned SIMPLE room archive into
`artifacts/hssd/`, extracts it temporarily, and converts it into
`assets/hssd/scene0/`. Both the download and generated room files are ignored by
Git. Rerunning the command reuses the cached archive and rebuilds the room.
The CC BY-NC 4.0 license and source/conversion details remain in `assets/hssd/`.
USD is needed only for setup; running the scene requires no Isaac or USD dependency.

## Inspect and replay

```sh
ARCHIVE=artifacts/psi-data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip

uv run --extra cu128 --frozen python src/runtime.py \
  --mode scene --eval-archive "$ARCHIVE" --episode-indices 0 --viewer

uv run --extra cu128 --frozen python src/runtime.py \
  --mode replay --eval-archive "$ARCHIVE" --episode-indices 0 \
  --max-steps 800 --viewer
```

`--episode-indices 0 1 2 3 4` runs the five published initial states as one
batch. `--viewer` enables the passive viewer; omit it for throughput.
`--max-steps 50` is a quick smoke run.

## Run the fine-tuned policy

The [task-specific checkpoint](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223)
at step 40000 uses the original 43D-state, six-block action head. This branch
loads that head and its Qwen3-VL weights in process, generates 78D actions,
and executes the configured 30 actions per prediction using plain flow inference.
RTC guidance is disabled for this probe.
Control steps run MJLab's action manager and physics substeps directly. Episode
outcomes and resets are handled by the runtime, avoiding the general RL step's
per-step reset-index synchronization. This scene has only actions and reset events.
The native MuJoCo head camera renders fresh images only before each policy
prediction. It downloads the batch of generalized positions and renders the
worlds sequentially into one image batch. Fixed scene poses are cached until
reset. The planner uploads BF16 image patches, builds multimodal positions from
CPU metadata, and keeps image-grid metadata on CPU for the default SDPA backend.
Unchanged token/mask/grid metadata reuses its device tensors across predictions;
image pixels are always updated.
Replay and scene mode skip head-camera rendering; the optional viewer is separate.
For headless policy runs, set `MUJOCO_GL=egl` before launching.

```sh
MUJOCO_GL=egl uv run --extra cu128 --frozen python src/runtime.py \
  --mode policy \
  --psi-run-dir artifacts/psi-model/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223 \
  --qwen-model artifacts/qwen3-vl-2b \
  --eval-archive "$ARCHIVE" --episode-indices 0 \
  --prompt 'xmove to the table and bend to pick up the box'
```

The runtime keeps placement and fall checks on the GPU and checks batch termination
between action chunks. It does not collect diagnostics, timings, metrics, or videos.
The visual assets live in `assets/`. The default scene uses the actual HSSD
room geometry and authored diffuse textures from SIMPLE's default
`mujoco_isaac` evaluation, rendered entirely by native MuJoCo. It includes the
Isaac box markings and approximates episode 0's lighting and Pearl tabletop.
MuJoCo approximates the MDL/PBR materials and area lights; pixel equivalence to
Isaac RTX is not established. Other episodes' task lighting/material variations
are not replayed yet. See [assets/hssd/UPSTREAM.md](assets/hssd/UPSTREAM.md).
Physics and closed-loop success-rate parity still require validation on the
target system. The published checker-floor reference videos use a different
visual backend from the current default official evaluator.
See [differences.md](differences.md) for evaluation differences and
[PERF.md](PERF.md) for CPU transfers and synchronization.
