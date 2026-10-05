# SIMPLE carry-box evaluation parity

We have a checkpoint-compatible Ψ₀ → SONIC → MJLab runtime. The visual scene
now uses the actual furnished HSSD scene0 geometry and diffuse textures from
SIMPLE’s default `mujoco_isaac` evaluator, rendered by native MuJoCo. Full
physics, photometric, and closed-loop policy equivalence remain unproven.

## What matches

- Task-specific Psi-0 checkpoint at step 40000 and the pinned compatible SONIC
  decoder.
- 43D joint-position state and 78D actions: 64 SONIC token values followed by
  14 Dex3 hand targets.
- Recorded robot/joint/box starting poses from the five published level-0
  episodes; box dimensions and basic table placement.
- 50 Hz action execution, with 5 ms physics steps.
- Checkpoint-derived image preprocessing and normalization.
- Actual HSSD scene0 geometry, authored UV textures, room placement, hidden
  original tea table, replacement slab, and Isaac box markings.
- Calibrated torso-mounted head camera, 640 × 360 projection, and 0.01–10 m
  clipping range. See [visual sources](assets/hssd/UPSTREAM.md).

## Visual target

The default official evaluator uses MuJoCo physics with Isaac rendering of
`hssd:scene0`, room `107734119_175999932`. This port now uses that room's actual
meshes and diffuse textures in MuJoCo, rather than the earlier checker-floor
visual target. Room placement, table replacement, and ceiling visibility follow
the inspected Isaac loader.

The published checker-floor reference videos illustrate recorded episodes from
a different visual backend. Their initial states and action sequences remain
useful for initialization and replay; matching those videos does not establish
agreement with the current default evaluation's images.

MuJoCo approximates MDL/PBR materials and cylinder lights. The saved lighting
and slab material choice are applied per episode. No comparison against actual
Isaac RGB or closed-loop policy success is claimed.

## Remaining differences

| Area | Current repository | Official pipeline / remaining work |
| --- | --- | --- |
| Rendering | Native MuJoCo head images, rendered sequentially from batched state. | Actual HSSD geometry/textures are ported. MDL/PBR materials, normal maps, reflections, cylinder lights, and RTX tone mapping are approximated. Pixel parity needs actual Isaac observations. |
| Per-episode appearance | Saved light positions, relative intensities/color temperatures, and table material choices are applied per rendered episode. | MuJoCo classic materials approximate the five MDL choices; cylinder length/radius, PBR shading, and exact photometric calibration remain unmatched. |
| Physical model and actuation | Batched torque control recomputes clipped PD torque every 5 ms; position actuators remain the default pending full replay validation. Torque-mode hands use the inspected Dex3 defaults: Kp 1.5, Kd 0.1, closing bounds, and a 0.25 rad feedback clamp each physics substep. Box/contact settings, solver `impratio=10`, and elliptic friction cone match the pinned SIMPLE builder. | The available ONNX bundle has joint actions but no external `LowCmd` trace. Desired body motor velocity and feedforward torque are assumed zero. Dex3 publishes at 500 Hz, so the 5 ms hand clamp approximates its command timing. Two no-slip solver iterations remain unsupported in MuJoCo Warp. Exact motor and replay parity remain unverified. |
| Initial velocity | Restores recorded base pose, joint positions, box pose, and base velocity; joint velocities remain zero. | Official evaluation similarly zeroes joint velocities and restores `observation.base_vel`. |
| Controller startup | Cold SONIC history by default; an optional 152-step static-pose prefill is available for A/B comparison. | Official evaluator runs a 152-boundary external-controller initialization sequence while holding the start pose, then restores the episode. Static-pose ONNX prefill is only an approximation. |
| Replanning | Plain flow executes 30 actions per prediction. `--rtc` executes 24 per 30-action prediction and guides the six-action overlap using the previous normalized chunk, with zero inference delay and ten flow steps by default. | The documented RTC configuration executes 24 actions, but the official server/client has an external lockstep path. Closed-loop outcome parity is untested. |
| Episode budget | Replay uses demonstration length + 10; policy uses `max(2 × demonstration length, 1500)`. `--max-steps` optionally caps each episode. | Same per-episode budgets. Batched simulation continues for other episodes after one reaches its budget. |
| Stopping rule | Records success or budget exhaustion for each episode. | Official evaluation also stops on task success or budget exhaustion. |
| Success rule | Matches the pinned SIMPLE checker: no hand/box contact, box/table contact within 0.05 m of table center Z 0.4, box center at or above 0.4, and accumulated placement time strictly above 0.9 s. Uses registered contacts without an extra distance filter and FP64 accumulation to match Python floats. | Upstream uses `table.pose.position[2]` despite describing it as the table top. Neither check requires uninterrupted placement. Contact generation and closed-loop outcomes still depend on the remaining physics differences. |

The compiled physical parameters were verified unchanged by the visual work.
That establishes isolation of this change, not physical parity with SIMPLE.
HSSD validation checks compiled geometry and camera range, native RGB,
body/hand stepping, unchanged simulation state during rendering, and image
agreement across batch origins for identical local poses.
CUDA rendering integration and full checkpoint rollouts still need validation
on the target GPU system.

## Next parity checks

1. Match physical parameters and replay recorded actions against reference
   trajectories before attributing behavior differences to the VLA.
2. Match episode initialization, controller history, budgets, and termination.
3. Compare planner outputs for identical recorded inputs, then closed-loop
   outcomes with equivalent inference settings.

CPU transfers and synchronization are inventoried separately in [PERF.md](PERF.md).

## Official references

Inspected SIMPLE source revision: `6d10628794d9c7de4596b4f2afb2c054a637c2bc`.

- [Task and scene configuration](https://github.com/physical-superintelligence-lab/SIMPLE/blob/6d10628794d9c7de4596b4f2afb2c054a637c2bc/src/simple/tasks/g1_wholebody_xmove_bend_carry_box_sonic.py)
- [Evaluation implementation](https://github.com/physical-superintelligence-lab/SIMPLE/blob/6d10628794d9c7de4596b4f2afb2c054a637c2bc/src/simple/evals/sonic_wbc.py)
- [MuJoCo engine](https://github.com/physical-superintelligence-lab/SIMPLE/blob/6d10628794d9c7de4596b4f2afb2c054a637c2bc/src/simple/engines/mujoco.py)
- [Documented evaluation configuration](https://psi-lab.ai/SIMPLE/docs/sonic-wbc/evaluation.html)
- [Published evaluation data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval)
