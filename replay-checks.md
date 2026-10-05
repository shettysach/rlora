# Carry-box replay verification

## Scope

Checked the local physics configuration and compiled model against SIMPLE at
`6d10628794d9c7de4596b4f2afb2c054a637c2bc`, and inspected all five trajectories
in the published carry-box evaluation archive. Cached reference engine and
robot sources match the Git blob hashes in that revision's tree manifest.

The timeout results at 800 steps do not establish a pickup failure. The command
previously supplied was too short to replay any of the five recordings fully.

## Recorded trajectories and budget

The archive is 50 Hz; local action execution is also 50 Hz (four 5 ms physics
substeps). Counts and heights below come from recorded object poses, not local
simulation outcomes.

| Episode | Recorded steps | Duration (s) | Maximum box center Z in first 800 steps (m) | Final box center Z (m) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1054 | 21.08 | 0.548 | 0.691 |
| 1 | 1595 | 31.90 | 0.240 | 0.670 |
| 2 | 1220 | 24.40 | 0.333 | 0.680 |
| 3 | 1001 | 20.02 | 0.549 | 0.698 |
| 4 | 1075 | 21.50 | 0.627 | 0.673 |

All five start the box at approximately `(-0.55, -0.06, 0.216905)`.
Robot starting poses differ. These recordings do not supply five different box
starting positions.

SIMPLE's reference evaluator budgets replay at `source_length + 10`; policy at
`max(2 * source_length, 1500)`. The runtime now uses these per-episode budgets
and holds the final recorded action during replay's extra ten steps.
`--max-steps` optionally caps each episode for diagnostics.

## Physical settings before the contact fix

| Setting | Local | Pinned SIMPLE reference |
| --- | --- | --- |
| Box mass | 2 kg | 0.1 kg |
| Box half-extents | `(0.115, 0.19, 0.217)` m | Same |
| Box friction | `(0.8, 0.005, 0.0001)` | `(0.8, 0.05, 0.005)` |
| Box contact dimension | 3 (MuJoCo default) | 4 |
| Box `solref` | `(0.02, 1)` | `(0.005, 2)` |
| Table friction | `(0.8, 0.005, 0.0001)` | `(2, 0.04, 0.0005)` |
| Table contact dimension/priority | 3 / 0 (defaults) | 6 / 10 |
| Body actuation | MJLab motor-derived position gains, armature, action scaling | Explicit torque PD using SONIC commands and torque clipping |
| Initial velocity | Joint velocities zero; recorded base velocity not loaded | Recorded base velocity restored |
| SONIC startup | Zero history, immediate replay | 152 idle controller boundaries while holding the initial pose, then scene restored |

The box mass, box/table friction and contact dimensions, box `solref`, and table
priority have now been changed to the reference values in the asset XML files.
Compiling both assets with MuJoCo confirms the new mass and contact parameters.
Body actuation and initialization remain as described below. Success checking
has subsequently been aligned with the reference, as described below.
The replay measurements in this document were taken **before** these changes;
a new replay is needed to measure their effect.

The local compiled box has a free joint and enabled collisions (`contype=1`,
`conaffinity=1`). It is not fixed or a visual-only object. Hand collision geoms
exist and hand position gains match the reference direct-position path:
thumb KP 5, index/middle KP 2.5, KD 1. This does not establish complete hand or
body dynamics parity: the reference applies explicit torque PD every substep,
whereas the local configuration uses MuJoCo position actuators.

Box/table sizes and table position agree with saved episode 0: table full size
`(1.25, 0.78999733, 0.1)`, center `(0.3, 0, 0.4)`. Local Y half-extent is rounded
to `0.395`. The local table top is at Z 0.45. Both success checkers now use
table center Z 0.4 with a contact-height tolerance of 0.05, require no hand contact
and box center Z at least 0.4, and scan registered contacts without another
distance filter. Local placement time now accumulates in FP64 to reproduce
SIMPLE's Python-float arithmetic and strict > 0.9 s boundary. A focused scalar
reference test covers contact-height boundaries, hand contact, inactive contact
slots, interrupted placement, and the 45-step threshold. The local pelvis-height
failure rule and evaluation budgets described here were subsequently removed or
aligned with the official evaluator.

These differences can affect lift forces, grasp friction, joint tracking, and
release. Their individual contribution to replay failure has not been isolated.
An all-timeout outcome alone does not identify which difference is responsible.

## Local CPU replay observation

Ran episode 0 for all 1054 recorded actions plus 10 final-action steps using
the actual SONIC ONNX decoder and local MJLab/Warp physics. Omitted only the
visual room entity (its geoms have zero collision masks); no rendering or viewer
was used. The robot, box, table, action managers, task checker, and physical
parameters were retained. This is a CPU diagnosis, not a repeat of the user's
GPU execution.

- Hand/box contact was detected on 148 control steps.
- Maximum box center Z was `0.237026` m, versus `0.857923` m in the recording.
- Final box center was approximately `(-0.453038, -0.063924, 0.223353)` m,
  versus recorded `(-0.078525, -0.006618, 0.690574)` m.
- Accumulated placement time was zero; success and fall flags remained false.

Thus collisions are active, but the full local replay does not reproduce the
recorded pickup and placement. A longer budget alone does not fix episode 0
on CPU. The physical/control mismatches need to be addressed before interpreting
policy failures. No production physics parameters were changed during that
measurement; the subsequent asset changes are described above.

## Next GPU replay

After the README artifact setup, run from the repository root:

```sh
uv run --extra cu128 --frozen python src/runtime.py \
  --mode replay \
  --eval-archive artifacts/psi-data/simple-eval/G1WholebodyXMoveBendCarryBoxSonic-v0.zip \
  --episode-indices 0 1 2 3 4 \
  --max-steps 1700 --seed 0
```

For visual inspection, select episode 0 and add `--viewer`. Inspect approach,
hand closure, box lift, placement, and release; success flags alone cannot
distinguish a physical failure from a checker mismatch.

Before policy success-rate comparisons, match box/contact parameters, SONIC
actuation and initialization, then compare replay trajectories against recorded
robot and box poses. The previous checkpoint numerical tests validate planner
inference, not these physical settings.

## Reference files

- `src/simple/engines/mujoco.py`: `_build_primitive_object`, `_build_primitive`.
- `src/simple/robots/g1_sonic.py`: `apply_action`, torque calculations.
- `src/simple/evals/sonic_wbc.py`: `episode_budget_steps`, startup and pose restore.
- `src/simple/tasks/g1_wholebody_xmove_bend_carry_box_sonic.py`: contact/success checks.

Local definitions are in `assets/simple/box.xml`, `assets/simple/table.xml`,
`src/sim/config.py`, `src/sim/env.py`, and `src/carry_box.py`.
