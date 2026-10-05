# GPU carry-box replay investigation

Run this from the `rlora` repository root on the RTX 5090 machine. The question is
which local control path best reproduces the five published SIMPLE replays, and
whether a one-control-step delay improves that match. Keep Ψ₀ and RTC out of this
experiment: replay uses recorded 78D actions.

The checkpoint's released training configuration says `rtc: false`,
`action_chunk_size: 30`, and `action_exec_horizon: 30`. Its `max_delay: 8` field
does **not** establish that this fine-tune was trained with a one-step command
delay. SIMPLE's replay code describes forwarding a cached `LowCmd` roughly one
step behind the newly sent token, but the exact timing and motor fields are not
in the published archive. Our `--control-delay 1` shifts both decoded body and
hand commands by one 20 ms control tick; treat it as a diagnostic approximation.

## Run the comparison

Check that the archive and SONIC bundle are present. Use the exact same commit,
archive, seed, episode order, and step budget for every run. Do not use a viewer.
Save the terminal output as well as the NPZ files.

```sh
set -o pipefail
ARCHIVE=artifacts/psi-data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip
test -f "$ARCHIVE" && test -f artifacts/sonic/model_decoder.onnx
mkdir -p artifacts/control
export MUJOCO_GL=egl

for actuation in position torque; do
  for delay in 0 1; do
    name="${actuation}-delay${delay}"
    uv run --extra cu128 --frozen python src/runtime.py --mode replay \
      --eval-archive "$ARCHIVE" --episode-indices 0 1 2 3 4 \
      --seed 0 --max-steps 1700 --actuation "$actuation" \
      --startup cold --control-delay "$delay" \
      --trace-output "artifacts/control/${name}.npz" \
      2>&1 | tee "artifacts/control/${name}.log"
    uv run --extra cu128 --frozen python tools/compare_replay_trace.py \
      --eval-archive "$ARCHIVE" --trace "artifacts/control/${name}.npz" \
      2>&1 | tee "artifacts/control/${name}-comparison.log"
  done
done
```

If the five-environment batch fails due to memory, run one episode at a time
with the same four variants and give each trace a unique episode suffix. Report
the failure and peak VRAM; do not silently change the comparison.

## Interpret the traces

For each episode and variant, report:

| Variant | Early joint RMSE | Full joint RMSE | Base XYZ RMSE | Box XYZ RMSE | Peak box Z | Success/step |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| position, immediate | | | | | | |
| position, delayed | | | | | | |
| torque, immediate | | | | | | |
| torque, delayed | | | | | | |

The comparison script prints these values and maximum absolute torque. It aligns
trace state 0 with recorded observation 0 and compares at most the recorded
episode length. Check the first 100 frames separately because later differences
can compound after contact or a fall. Describe **where** divergence begins: joint
motion, base drift, grasp/lift, or table placement. A success flag alone is not
enough to choose a controller.

If one variant clearly improves the early trajectory and outcomes, rerun that
variant with `--startup static-152`, keeping its actuation and delay unchanged.
This prefill is only an approximation of SIMPLE's external controller startup.

## Decision and report

Give a short recommendation backed by the table:

- Which actuation and delay should be used for the next comparison, and why?
- Is the difference large enough to change the default, or is evidence mixed?
- Which episode and step first show a meaningful mismatch?
- Did any run error, exhaust GPU memory, or show abnormal motor torque?

Do not claim exact SIMPLE parity from these four runs. If none tracks the early
joint trajectory well, the next high-value measurement is a SIMPLE-side trace
of the actual per-joint `LowCmd` (`q`, `dq`, `kp`, `kd`, `tau`) and its control-step
index, alongside joint state and applied torque. Compare those fields with our
`target`, `joint_pos`, `joint_vel`, and `torque` arrays before changing gains or
adding a local `LowCmd` interface. Only after replay behavior is understood
should we implement the documented **test-time** RTC policy mode (24 executed
actions per 30-action chunk) and compare policy evaluation outcomes.

Sources: [released fine-tune configuration](https://huggingface.co/USC-PSI-Lab/psi-model/commit/7e3076049dd86c322d2ca19e6c0288b1ec25c61b),
[SIMPLE replay control path](https://github.com/physical-superintelligence-lab/SIMPLE/blob/6d10628794d9c7de4596b4f2afb2c054a637c2bc/src/simple/agents/replay_wbc_agent.py),
[SIMPLE policy evaluation setup](https://psi-lab.ai/SIMPLE/docs/sonic-wbc/evaluation.html).

## Follow-up: why placement never succeeds

**Instructions for the agent on the GPU device.** The first comparison found
zero successes even where the box remained near table height. Diagnose the
success rule using a fresh trace; do not ask the user to paste shell commands.

1. From the repository root, verify that the current `src/runtime.py` records
   `placement_time_s` and `tools/compare_replay_trace.py` prints placement step
   diagnostics. Record the commit hash. Check that the evaluation archive and pinned
   SONIC decoder are present. If the checkout lacks this instrumentation, update
   it before running.
2. Run `src/runtime.py` in replay mode with episodes 0–4, seed 0, 1700 steps,
   torque actuation, cold startup, zero control delay, and `MUJOCO_GL=egl`. Save
   a **new, uniquely named** NPZ trace and terminal log under
   `artifacts/control/`; never reuse an earlier trace path.
3. Open that NPZ and verify it contains `hand_contact`, `table_contact`,
   `table_height_contact`, `box_height_ok`, `placed`, and `placement_time_s`, each
   with five environment columns. Then run `tools/compare_replay_trace.py` on
   that exact file and confirm it prints five `placement steps:` lines. If a
   check fails, stop and report the actual error and trace keys.
4. For each episode, report those five condition counts, final accumulated
   placement time, success flag, and the first/last steps where `placed` is
   true. Identify the condition that prevents success. `placed` must accumulate
   for more than 0.9 seconds; it need not be consecutive.
5. If the box appears to rest on the table but `table_contact=0`, inspect Warp
   contact reporting and geometry IDs. If raw table contact exists but
   `table_height_contact=0`, inspect contact Z values against the checker’s
   0.4 ± 0.05 m requirement. If hand contact persists, inspect release behavior.
   Change the checker only if a concrete implementation error is demonstrated;
   rerun the focused success test and affected replay after any fix.

Send the commit hash, fresh trace path, five placement lines, and a short
conclusion. Do not infer success from box height alone or report an older NPZ as
the new run.
