# HSSD scene0 for the official carry-box evaluator

The default runtime scene is SIMPLE's furnished `hssd:scene0`, room
`107734119_175999932`. The geometry comes from the actual published USD asset,
including furniture, walls, openings, ceiling, carpet, and authored UV textures.
Runtime rendering and simulation require only MuJoCo/MJLab.

## Sources

- [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) revision
  `6d10628794d9c7de4596b4f2afb2c054a637c2bc`: `src/simple/engines/isaacsim.py`,
  `src/simple/scenes/hssd.py`, and `src/simple/resources/hssd-scenes/config.yaml`.
- [SIMPLE asset archive](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE/resolve/1ce0fa3956706b408df2c7c0e26b0298aa7411fd/scenes_hssd_107734119_175999932.zip),
  revision `1ce0fa3956706b408df2c7c0e26b0298aa7411fd`.
- Episode 0's saved `environment_config` from the published carry-box archive
  supplies the fixed room placement, six light positions, and Pearl table material.
- HSSD attribution and CC BY-NC 4.0 terms are in [LICENSE.md](LICENSE.md).

## Conversion

`scene0/room.xml` contains 403 visual mesh parts, 273,684 triangles, and 150
materials. `scene0/manifest.json` maps every exported mesh back to its USD prim.
Material subsets, winding, normals, nonuniform transforms, UVs, and authored UV
scales are preserved. Diffuse color factors are converted from linear values to
sRGB for MuJoCo's classic renderer. Glass transmission is approximated by alpha.

The room is rotated by 90 degrees about X into Z-up coordinates. Its selected
tea-table top is translated to the saved center offset `[0.25, 0, 0]`; the
vertical translation remains zero, matching SIMPLE's G1 SONIC special case.
The original tea-table prim
`/World/furniture/node_b914fb6bcc81386bfa1ff7a3eb8412b7ac581ff` is hidden and
replaced by the task's slab at `[0.3, 0, 0.4]`. The source USD marks the ceiling
invisible, but the inspected Isaac loader explicitly makes it visible, so the
port includes it.

Room meshes are visual-only (`contype=conaffinity=0`, `density=0`). SIMPLE's
MuJoCo engine builds task actors and the ground plane rather than HSSD room
collision meshes. The existing robot, box, table, and floor dynamics are retained.
A mocap root gives the room the correct per-environment origin in MJLab.

`assets/simple/scene.xml` approximates episode 0's six cylinder lights with
native spot lights at their saved world positions. `table.xml` approximates
Pearl; `box.xml` includes the Isaac engine's nine render-only tape/ink markings.
Camera clipping matches the inspected Isaac implementation's hard-coded
0.01–10 m range. The prior calibrated camera mount and projection are retained.

## Download and build

Run from the repo root:

```sh
uv run --extra cu128 --frozen --with usd-core==26.8 python tools/setup_hssd.py
```

The script downloads the revision pinned above into `artifacts/hssd/`, extracts
it temporarily, and writes the converted assets to `assets/hssd/scene0/`.
Downloads and generated files are ignored by Git. Rerunning the command reuses
the cached archive and rebuilds the room.
USD is only a conversion dependency. It is not added to the runtime manifest.
The converter repairs the archive's absolute `/props` references and its
`black.usd`/`Black.usd` case mismatch without editing the source USD.

## Limits

MuJoCo's classic renderer approximates Isaac's MDL/PBR materials, reflections,
normal maps, transmission, cylinder area lights, and tone mapping. This is a
geometry/texture port, not a claim of pixel equivalence to RTX rendering.
The task lighting and slab material currently use episode 0's appearance; other
recorded episodes have different lighting and MDL materials that are not replayed.
Native RGB remains sequential and uses host state transfers listed in
[PERF.md](../../PERF.md). Full physics and policy parity are still outstanding.

## Validation

- Compiles in the pinned MuJoCo/MJLab environment: 512 total geoms, 453 total
  meshes, 57 generalized positions, and two mocap roots.
- Existing body masses/inertias, joint dynamics, collision parameters, and
  actuator gains/ranges match the previous checker scene. All 403 imported
  room geoms have collision participation disabled.
- Two environments rendered native 640 × 360 RGB without changing simulator
  positions, velocities, controls, mocap poses, or time. Identical local poses
  at different batch origins differed by 0.00059 mean RGB units on the 0–255
  scale. RGB remained valid after a body/hand step.
- Head-camera images at recorded frames 0, 200, and 400 were inspected, along
  with the configured interior viewer camera.
- Ruff and ty checks passed. Validation used CPU/Mesa EGL; an NVIDIA CUDA
  driver and actual Isaac RGB were unavailable in this workspace.
