# SIMPLE carry-box visuals

The default visual target is the blue checker-floor MuJoCo scene visible in the
five published `G1WholebodyXMoveBendCarryBoxSonic-v0` evaluation videos. Those
videos do not show the HSSD room used by SIMPLE's Isaac rendering path.

Sources:

- [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) commit
  `6d10628794d9c7de4596b4f2afb2c054a637c2bc`,
  `src/simple/engines/mujoco.py`: checker texture, repeat, lights, primitive box
  color, table color, head-camera mount, and projection.
- [SIMPLE assets](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE) revision
  `1ce0fa3956706b408df2c7c0e26b0298aa7411fd`, `robots_g1_sonic.zip`:
  `robots/g1_sonic/g1_29dof_with_hand.xml` supplies the checker material and
  directional light.
- [Published evaluation data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval):
  the table/box dimensions and poses, 640 x 360 resolution, and camera focal
  lengths (`fx = fy = 224.06641222710715`).

`scene.xml` contains the checker material and two upstream lights. The scene
callback also restores MuJoCo's default headlight and haze, overriding MJLab's
viewer defaults. `box.xml` and `table.xml` contain the existing task geometry
with upstream colors. Their mass, friction, and other dynamics retain this
repository's existing settings; physics parity is a separate task.

The camera is attached to `torso_link` at `(0.05762354, 0.01752999, 0.4298702)`.
Its quaternion combines SIMPLE's `(0.91496, 0, 0.40355, 0)` head orientation with
the Isaac-to-MuJoCo camera-axis conversion, then is normalized. Its vertical FOV
is `2 * atan(360 / (2 * fy)) = 77.5521422` degrees.

The VLA camera uses native MuJoCo OpenGL rendering to match SIMPLE's checker
filtering and lighting. At each prediction, the runtime downloads four batched
state arrays and renders each world with one shared renderer. Rendering
removes the batch placement offsets so lights and floor texture are identical
across environments. This adds host transfers and sequential image rendering;
physics and planner inference remain batched. OpenGL is initialized lazily,
so scene/replay mode does not create an offscreen camera renderer.

OpenGL requires an EGL-capable driver for headless rendering (`MUJOCO_GL=egl`).
Driver differences and video compression can still affect pixel comparisons.
CPU transfers and synchronization are inventoried in [PERF.md](../../PERF.md).
Isaac MDL materials and RTX rendering are not reproduced.

In the CPU validation setup, MuJoCo 3.11.0 with Mesa EGL produced shadow
speckles on the table and floor in both the upstream and ported scenes. Running
with `MESA_EXTENSION_OVERRIDE=-GL_ARB_clip_control` removed those speckles while
preserving shadows. This is a Mesa-specific validation workaround; it is not
applied automatically to NVIDIA runs. MuJoCo's
[classic renderer](https://github.com/google-deepmind/mujoco/blob/3.11.0/src/render/classic/render_gl3.c)
uses different shadow depth offsets depending on that extension.

Validation used frames 0, 200, and 400 from each of the five published episodes:

- An initial scene-only comparison reduced mean absolute RGB error against an
  upstream scene reconstruction from 33.47 to 0.12 on the 0–255 scale.
- The final five-environment runtime comparison, with the Mesa workaround,
  measured mean error 0.065 against that reconstruction and 4.42 against the
  compressed published videos. Maximum per-frame errors were 0.30 and 7.97,
  respectively. This establishes camera/scene agreement for the sampled poses,
  not Isaac rendering or policy success.
- RGB requests left simulation positions, velocities, controls, mocap poses,
  and time unchanged. Identical local poses at different batch origins differed
  by at most 0.003 mean RGB units. Five environments also executed a finite
  body/hand step.
- Compiled physical parameters matched the previous scene exactly. This does
  not establish physics parity with SIMPLE.

Comparison images are in `artifacts/visual-parity/comparison.png` and
`artifacts/visual-parity/runtime_episode_000000.png` through episode 4.
Validation ran on CPU with Mesa EGL; the NVIDIA CUDA driver was unavailable.
