from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTUATOR_4010,
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_WAIST,
    get_g1_robot_cfg,
)
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from walk_to_target import GOAL_RADIUS, GOAL_X

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER  # ty: ignore[unresolved-attribute]


def _add_goal(spec: MjSpec) -> None:
    spec.worldbody.add_geom(
        name="walk_target",
        type=MJGEOM_CYLINDER,
        pos=(GOAL_X, 0.0, 0.006),
        size=(GOAL_RADIUS, 0.006, 0.0),
        rgba=(0.05, 0.9, 0.1, 1.0),
        contype=0,
        conaffinity=0,
        mass=0.0,
    )


def make_env_cfg(num_envs: int) -> ManagerBasedRlEnvCfg:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
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
    robot = get_g1_robot_cfg()
    robot.articulation = EntityArticulationInfoCfg(
        actuators=actuators, soft_joint_pos_limit_factor=0.9
    )
    scale = {
        pattern: 0.25 * cast(float, actuator.effort_limit) / actuator.stiffness
        for actuator in actuators
        for pattern in actuator.target_names_expr
    }
    scene = SceneCfg(
        num_envs=num_envs,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot},
        sensors=(
            CameraSensorCfg(
                name="observation_camera",
                parent_body="robot/torso_link",
                # SONIC's G1 head-camera mount with ZED Mini WVGA optics.
                pos=(0.06, 0.0, 0.45),
                quat=(
                    0.6515477423451215,
                    0.2752506903280633,
                    -0.2754699671476764,
                    -0.6510291038951777,
                ),
                width=672,
                height=376,
                data_types=("rgb",),
                fovy=54.0,
            ),
        ),
        spec_fn=_add_goal,
    )
    return ManagerBasedRlEnvCfg(
        decimation=4,
        scene=scene,
        actions={
            "joint_position": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=scale,
                use_default_offset=True,
            )
        },
        sim=SimulationCfg(njmax=128, mujoco=MujocoCfg(timestep=0.005)),
        viewer=ViewerConfig(
            distance=4.0,
            elevation=-20.0,
            azimuth=135.0,
            lookat=(1.0, 0.0, 0.8),
            max_extra_envs=0,
        ),
        episode_length_s=0.0,
    )
