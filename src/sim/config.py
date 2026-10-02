from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTUATOR_4010,
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_WAIST,
    KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from shared.g1 import BODY_JOINTS, HAND_JOINTS

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

ASSETS = Path(__file__).resolve().parents[2] / "assets"
G1_DEX3_XML = ASSETS / "g1/g1_29dof_with_hand.xml"


def _dex3_spec() -> MjSpec:
    return mujoco.MjSpec.from_file(str(G1_DEX3_XML))  # ty: ignore[unresolved-attribute]


def _box_spec() -> MjSpec:
    return mujoco.MjSpec.from_file(str(ASSETS / "simple/box.xml"))  # ty: ignore[unresolved-attribute]


def _table_spec() -> MjSpec:
    return mujoco.MjSpec.from_file(str(ASSETS / "simple/table.xml"))  # ty: ignore[unresolved-attribute]


def _scene_visuals(spec: MjSpec) -> None:
    visuals = mujoco.MjSpec.from_file(str(ASSETS / "simple/scene.xml"))  # ty: ignore[unresolved-attribute]
    spec.attach(visuals, prefix="simple/", frame=spec.worldbody.add_frame())
    spec.geom("terrain").material = "simple/groundplane"
    # SIMPLE builds its scene with MuJoCo's default headlight and haze.
    spec.visual.headlight.diffuse[:] = visuals.visual.headlight.diffuse
    spec.visual.headlight.ambient[:] = visuals.visual.headlight.ambient
    spec.visual.headlight.specular[:] = visuals.visual.headlight.specular
    spec.visual.rgba.haze[:] = visuals.visual.rgba.haze
    # SIMPLE's head_stereo_left mount and 110-degree horizontal FOV.
    spec.body("robot/torso_link").add_camera(
        name="observation_camera",
        pos=(0.05762354, 0.01752999, 0.4298702),
        quat=(0.65925355, 0.25570444, -0.25570444, -0.65925355),
        fovy=77.5521422,
    )


def make_env_cfg(num_envs: int) -> ManagerBasedRlEnvCfg:
    g1_actuator_7520_14 = replace(
        G1_ACTUATOR_7520_14,
        target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
    )
    g1_actuator_7520_22 = replace(
        G1_ACTUATOR_7520_22,
        target_names_expr=(
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_knee_joint",
        ),
    )
    actuators = (
        G1_ACTUATOR_5020,
        g1_actuator_7520_14,
        g1_actuator_7520_22,
        G1_ACTUATOR_4010,
        G1_ACTUATOR_WAIST,
        G1_ACTUATOR_ANKLE,
    )
    hand_actuators = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hand_thumb_0_joint",),
            stiffness=5.0,
            damping=1.0,
            effort_limit=2.45,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hand_thumb_[12]_joint",),
            stiffness=5.0,
            damping=1.0,
            effort_limit=1.4,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hand_(index|middle)_[01]_joint",),
            stiffness=2.5,
            damping=1.0,
            effort_limit=1.4,
        ),
    )
    robot = EntityCfg(
        spec_fn=_dex3_spec,
        init_state=replace(KNEES_BENT_KEYFRAME, pos=(-1.2, 0.0, 0.76)),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators + hand_actuators,
            soft_joint_pos_limit_factor=0.9,
        ),
    )
    scale = {
        pattern: 0.25 * cast(float, actuator.effort_limit) / actuator.stiffness
        for actuator in actuators
        for pattern in actuator.target_names_expr
    }
    scene = SceneCfg(
        num_envs=num_envs,
        terrain=TerrainEntityCfg(
            terrain_type="plane", textures=(), materials=(), lights=()
        ),
        spec_fn=_scene_visuals,
        entities={
            "robot": robot,
            "box": EntityCfg(
                spec_fn=_box_spec,
                init_state=EntityCfg.InitialStateCfg(pos=(-0.55, -0.06, 0.217)),
            ),
            "table": EntityCfg(
                spec_fn=_table_spec,
                init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.4)),
            ),
        },
    )
    return ManagerBasedRlEnvCfg(
        decimation=4,
        scene=scene,
        actions={
            "joint_position": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=BODY_JOINTS,
                scale=scale,
                use_default_offset=True,
            ),
            "hands": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=HAND_JOINTS,
                use_default_offset=False,
            ),
        },
        sim=SimulationCfg(njmax=256, mujoco=MujocoCfg(timestep=0.005)),
        viewer=ViewerConfig(
            distance=4.0,
            elevation=-20.0,
            azimuth=135.0,
            lookat=(1.0, 0.0, 0.8),
            max_extra_envs=0,
        ),
        episode_length_s=0.0,
    )
