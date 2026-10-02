# SIMPLE carry-box evaluation parity

We have a checkpoint-compatible Ψ₀ → SONIC → MJLab runtime. The visual scene
now targets the blue checker-floor MuJoCo observations in the five published
`G1WholebodyXMoveBendCarryBoxSonic-v0` episodes. Full physics and closed-loop
policy equivalence have not been established.

## What matches

- Task-specific Psi-0 checkpoint at step 40000 and the pinned compatible SONIC
  decoder.
- 43D joint-position state and 78D actions: 64 SONIC token values followed by
  14 Dex3 hand targets.
- Recorded robot/joint/box starting poses from the five published level-0
  episodes; box dimensions and basic table placement.
- 50 Hz action execution, with 5 ms physics steps.
- Checkpoint-derived image preprocessing and normalization.
- Blue checker floor, tan box, gray tabletop, lighting, and calibrated
  torso-mounted head camera, using native MuJoCo rendering. See
  [visual sources and validation](assets/simple/UPSTREAM.md).

## Visual target

The published evaluation videos show the MuJoCo checker-floor scene. SIMPLE's
current evaluator also supports a furnished HSSD room through Isaac rendering.
Those are different observation distributions. This port targets the published
MuJoCo videos first and uses only MJLab/MuJoCo at runtime.

The earlier assessment that these published videos required an HSSD room was
incorrect. Room reconstruction is not required to reproduce their visual scene.
Native MuJoCo also avoids the checker filtering and lighting differences found
when rendering this scene through Warp's camera path.

At 15 recorded poses across five episodes, native rendering of our scene and an
upstream scene reconstruction produced mean absolute RGB error of 0.12 on the
0–255 scale, down from 33.47 before the visual changes. This compares two scenes
under the same renderer. The final five-environment runtime check, with a
documented Mesa shadow workaround, measured 0.065 against the reconstruction
and 4.42 against the compressed published videos. These measurements do not
establish policy success or Isaac equivalence.

## Remaining differences

| Area | Current repository | Official pipeline / remaining work |
| --- | --- | --- |
| Rendering | Native MuJoCo head images, rendered sequentially from batched state. | Matches the published scene target; Isaac room/materials/RTX output are not reproduced. Driver/version behavior and compressed-video pixels can differ. |
| Physical model and actuation | Imported G1 geometry, MJLab position actuators, existing local object mass/friction and solver settings. | SIMPLE applies SONIC commands through explicit torque PD control. Compare compiled parameters, saturation, contacts, and recorded-action trajectories. The local box is 2 kg; the inspected upstream primitive builder uses 0.1 kg. |
| Initial velocity | Restores poses and zero joint velocities; does not load recorded base velocity. | Official evaluation restores `observation.base_vel`. |
| Controller startup | Zero SONIC observation history; actions begin immediately. | Official evaluator runs a 152-boundary initialization sequence while holding the start pose, then restores the episode. |
| Replanning | Executes the checkpoint's 30 actions per prediction, or 0.60 s; plain flow inference. | Documented RTC configuration executes 24, or 0.48 s, with RTC guidance. Observation frequency differs independently of guidance. |
| Episode budget | Default 800 steps, or 16 s. | Current policy evaluator uses `max(2 × demonstration length, 1500)`, at least 30 s. |
| Failure rule | Pelvis height below 0.5 m records failure. | Official evaluation loop has no equivalent height-based termination; bending can trigger a local failure. |
| Success rule | Released box/table contact accumulates placement time above 0.9 s; uses table-top height 0.45 m and a contact-distance filter. | Upstream uses `table.pose.position[2]` as the height threshold despite describing it as the table top. Reconcile height/contact details before comparing outcomes. Neither check requires uninterrupted placement. |

The compiled physical parameters were verified unchanged by the visual work.
That establishes isolation of this change, not physical parity with SIMPLE.
Five-environment CPU checks verified native RGB and body/hand stepping,
unchanged simulation state during rendering, and image agreement across batch
origins for identical local poses.
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
