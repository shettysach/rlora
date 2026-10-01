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
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from shared.g1 import BODY_JOINTS, HAND_JOINTS

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

G1_DEX3_XML = Path(__file__).parent / "assets/g1/g1_29dof_with_hand.xml"
MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]
MJJOINT_FREE = mujoco.mjtJoint.mjJNT_FREE  # ty: ignore[unresolved-attribute]


def _dex3_spec() -> MjSpec:
    return mujoco.MjSpec.from_file(str(G1_DEX3_XML))  # ty: ignore[unresolved-attribute]


def _box_spec() -> MjSpec:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    box = spec.worldbody.add_body(name="box")
    box.add_joint(name="free_joint", type=MJJOINT_FREE)
    box.add_geom(
        name="collision",
        type=MJGEOM_BOX,
        size=(0.115, 0.19, 0.217),
        mass=2.0,
        friction=(0.8, 0.005, 0.0001),
        rgba=(0.55, 0.40, 0.25, 1.0),
    )
    return spec


def _table_spec() -> MjSpec:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    table = spec.worldbody.add_body(name="top")
    table.add_geom(
        name="collision",
        type=MJGEOM_BOX,
        size=(0.625, 0.395, 0.05),
        friction=(0.8, 0.005, 0.0001),
        rgba=(0.7, 0.68, 0.62, 1.0),
    )
    return spec


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
        terrain=TerrainEntityCfg(terrain_type="plane"),
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
        sensors=(
            CameraSensorCfg(
                name="observation_camera",
                parent_body="robot/torso_link",
                # SONIC's G1 head-camera mount with ZED Mini WVGA optics.
                pos=(0.06, 0.0, 0.45),
                quat=(
                    0.66075695,
                    0.25234433,
                    -0.25258157,
                    -0.66024628,
                ),
                width=640,
                height=360,
                data_types=("rgb",),
                fovy=77.5521422,
            ),
        ),
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
