# SIMPLE carry-box primitives and camera

The default visual scene is now the actual HSSD room used by SIMPLE's default
`mujoco_isaac` evaluator. Its assets, conversion, licensing, and limitations are
in [assets/hssd/UPSTREAM.md](../hssd/UPSTREAM.md).

Source: [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) commit
`6d10628794d9c7de4596b4f2afb2c054a637c2bc`,
`src/simple/engines/isaacsim.py` and `src/simple/engines/mujoco.py`; the published
carry-box evaluation archive supplies the geometry, saved poses, and appearance.

- `scene.xml`: six native lights at episode 0's saved cylinder-light positions,
  with approximate cool color and illumination. Cylinder area-light photometry
  is not reproduced by MuJoCo spot lights.
- `table.xml`: the existing collision slab with an approximation of episode 0's
  Pearl MDL material. The original HSSD tea table is hidden, as in Isaac.
- `box.xml`: the existing collision box, with the nine render-only tape/ink
  markings authored by the Isaac engine. The optional external cardboard
  texture and MDL shading are approximated by solid native materials.
- `checker.xml`: retained source of the earlier published-video checker floor;
  it is no longer the runtime default.

Physical box/table mass, friction, and contact settings retain their prior
values. Decorative markings have zero density and no collision participation.

The camera is attached to `torso_link` at `(0.05762354, 0.01752999, 0.4298702)`.
Its quaternion combines SIMPLE's `(0.91496, 0, 0.40355, 0)` head orientation with
the Isaac-to-MuJoCo camera-axis conversion, then is normalized. Its vertical FOV
is `2 * atan(360 / (2 * 224.06641222710715)) = 77.5521422` degrees at 640 × 360.
The scene uses a fixed 2 m statistic extent and corresponding clipping factors
for the official Isaac implementation's hard-coded 0.01–10 m camera range.

The VLA uses one lazy native MuJoCo renderer, downloads batched generalized
positions, rebases free/mocap bodies to local coordinates, and renders the worlds
sequentially into the final RGB batch without advancing physics. Fixed origins
and room/table mocap poses are cached until reset. Scene/replay mode skips
offscreen RGB.
CPU transfers are listed in [PERF.md](../../PERF.md).

For headless rendering use `MUJOCO_GL=egl`. In CPU validation, MuJoCo 3.11.0 with
Mesa EGL produced shadow speckles; `MESA_EXTENSION_OVERRIDE=-GL_ARB_clip_control`
removed them while retaining shadows. This Mesa-specific workaround is not
applied automatically to NVIDIA runs. It affects MuJoCo's extension-dependent
[classic shadow depth offsets](https://github.com/google-deepmind/mujoco/blob/3.11.0/src/render/classic/render_gl3.c).
