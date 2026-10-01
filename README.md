# Carry-box probe

This branch ports the scene for `simple/G1WholebodyXMoveBendCarryBoxSonic-v0` to
the batched MJLab runtime. G1 has 29 SONIC-controlled body joints and 14
position-controlled Dex3 joints. The scene has a floor box and a table. It uses
the published 78D action layout: 64 SONIC body-token values followed by 14 hand
targets, at 50 Hz. The 43D planner state contains the same body and hand joint
order as the [published evaluation data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval).

The task succeeds after the box is released onto the table's upper surface for
over 0.9 seconds, matching the [SIMPLE task](https://github.com/physical-superintelligence-lab/SIMPLE/blob/main/src/simple/tasks/g1_wholebody_xmove_bend_carry_box_sonic.py).
`--mode replay` feeds recorded actions to the decoder. `--mode policy` asks the
official Psi0 SONIC HTTP server for action chunks. Both modes execute body and
hand actions in MJLab and emit per-episode diagnostics.

## Setup

```sh
uv sync --extra cu128 --frozen

uvx hf download USC-PSI-Lab/psi-data \
  --repo-type dataset \
  --include simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip \
  --local-dir artifacts/psi-data

uvx hf download nvidia/GEAR-SONIC model_decoder.onnx \
  --revision 6733128a3d8a523b1418b06bca3cdf61c8b0987f \
  --local-dir artifacts/sonic
```

The runtime checks the decoder's SHA-256 so a v1.1 decoder cannot silently
consume the older task's body tokens. The Dex3 MJCF and meshes came from the
existing `dex3_hands` branch; source and license are recorded in
`src/sim/assets/g1/`.

## Inspect and replay

```sh
ARCHIVE=artifacts/psi-data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip

uv run --extra cu128 --frozen python src/runtime.py \
  --mode scene --eval-archive "$ARCHIVE" --episode-indices 0

uv run --extra cu128 --frozen python src/runtime.py \
  --mode replay --eval-archive "$ARCHIVE" --episode-indices 0 \
  --max-steps 800 --record-video results/carry-replay.mp4 \
  --save-metrics results/carry-replay.json
```

`--episode-indices 0 1 2 3 4` runs the five published initial states as one
batch. `--record-video` saves the head-camera view of the first batch member.
`--max-steps 50` is a quick smoke run.

## Run the fine-tuned policy

Start the [published Psi0 server](https://psi-lab.ai/SIMPLE/docs/tutorials/wholebody_loco_manipulation.html)
with the task-specific `sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223`
checkpoint at step 40000, `--rtc`, `--action-exec-horizon 24`, and port 8014.
That release has a different Ψ₀ action-head architecture from the checkpoint
loaded by this repo's older direct Python wrapper. The server runs the released
model and returns denormalized 78D actions; this branch retains batched SONIC
execution in MJLab.

```sh
uv run --extra cu128 --frozen python src/runtime.py \
  --mode policy --psi-url http://127.0.0.1:8014 \
  --eval-archive "$ARCHIVE" --episode-indices 0 \
  --prompt 'xmove to the table and bend to pick up the box' \
  --record-video results/carry-policy.mp4 \
  --save-metrics results/carry-policy.json
```

The JSON reports placement success, falls, body displacement and heading,
closest robot-box distance, box displacement and height, hand-box and
box-table contacts, latency, and termination reason. The room geometry,
lighting, physics, and camera rendering are approximations of SIMPLE. Matched
trajectory or success-rate comparisons therefore require validation on the
target system. The published evaluation videos are demonstrations, not
fine-tuned-policy rollouts.
