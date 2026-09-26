from __future__ import annotations

from dataclasses import replace
from typing import cast

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
        episode_length_s=0.0,
    )
