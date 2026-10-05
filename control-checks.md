# Carry-box control comparison

`--actuation torque` uses batched motor actuators with a clipped PD torque
computed from each joint's latest position and velocity on every 5 ms physics
substep. The existing position servo remains the default until full replay
checks on the target GPU show no unexplained major regression. Both paths use
the same direct SONIC ONNX output and normalized-action conversion.

This is a **source-derived approximation**, not exact SIMPLE control. The
published 78D action contains a 64D token and 14 hand targets; it does not
include the external controller's `LowCmd`. The torque path uses the local G1
actuator gains and effort limits, assumes desired motor velocity and feedforward
torque are zero, and keeps the same decoder. These assumptions require a real
`LowCmd` trace to verify. The optional static history warmup and one-step delay
are diagnostics, not calibrated replacements for the external controller.

Run from the repository root on the GPU device, replacing the archive path if
needed:

```sh
ARCHIVE=artifacts/psi-data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip
uv run --extra cu128 --frozen python src/runtime.py --mode replay \
  --eval-archive "$ARCHIVE" --episode-indices 0 1 2 3 4 \
  --max-steps 1700 --actuation position --trace-output artifacts/control/position.npz
uv run --extra cu128 --frozen python src/runtime.py --mode replay \
  --eval-archive "$ARCHIVE" --episode-indices 0 1 2 3 4 \
  --max-steps 1700 --actuation torque --trace-output artifacts/control/torque.npz
uv run --extra cu128 --frozen python tools/compare_replay_trace.py \
  --eval-archive "$ARCHIVE" --trace artifacts/control/position.npz
uv run --extra cu128 --frozen python tools/compare_replay_trace.py \
  --eval-archive "$ARCHIVE" --trace artifacts/control/torque.npz
```

The trace records one initial state and one post-action state per control step,
plus the applied targets and final-substep motor torques. The comparison aligns
trace state 0 with recorded observation 0. Check first-100-frame joint RMSE,
full joint and box trajectories, saturation, lift timing, and success. To isolate
the optional timing approximations, repeat a torque replay with
`--startup static-152` or `--control-delay 1`, changing only one option at a time.
The scene viewer and policy runs use the selected actuation mode too.
