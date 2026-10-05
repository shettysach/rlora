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

MuJoCo approximates MDL/PBR materials and cylinder lights. Episode 0 supplies
the current task lighting and Pearl slab material. Other recorded episodes'
appearance variations are not replayed, and no comparison against actual Isaac
RGB or closed-loop policy success is claimed.

## Remaining differences

| Area | Current repository | Official pipeline / remaining work |
| --- | --- | --- |
| Rendering | Native MuJoCo head images, rendered sequentially from batched state. | Actual HSSD geometry/textures are ported. MDL/PBR materials, normal maps, reflections, cylinder lights, and RTX tone mapping are approximated. Pixel parity needs actual Isaac observations. |
| Per-episode appearance | Room geometry is fixed; task lighting/materials approximate episode 0. | Saved episode configurations use different light placements/intensities/temperatures and slab MDL materials. Their full appearance variations are not replayed. |
| Physical model and actuation | Imported G1 geometry and MJLab position actuators. Box mass (0.1 kg), box/table friction, contact dimensions, box `solref`, and table priority now match the pinned SIMPLE builder. | SIMPLE applies SONIC commands through explicit torque PD control. Body gains, armature, saturation, solver behavior, and recorded-action trajectories still need comparison. Replay outcomes after the contact changes are unverified. |
| Initial velocity | Restores poses and zero joint velocities; does not load recorded base velocity. | Official evaluation restores `observation.base_vel`. |
| Controller startup | Zero SONIC observation history; actions begin immediately. | Official evaluator runs a 152-boundary initialization sequence while holding the start pose, then restores the episode. |
| Replanning | Executes the checkpoint's 30 actions per prediction, or 0.60 s; plain flow inference. | Documented RTC configuration executes 24, or 0.48 s, with RTC guidance. Observation frequency differs independently of guidance. |
| Episode budget | Default 800 steps, or 16 s. | Current policy evaluator uses `max(2 × demonstration length, 1500)`, at least 30 s. |
| Failure rule | Pelvis height below 0.5 m records failure. | Official evaluation loop has no equivalent height-based termination; bending can trigger a local failure. |
| Success rule | Released box/table contact accumulates placement time above 0.9 s; uses table-top height 0.45 m and a contact-distance filter. | Upstream uses `table.pose.position[2]` as the height threshold despite describing it as the table top. Reconcile height/contact details before comparing outcomes. Neither check requires uninterrupted placement. |

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
