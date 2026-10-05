from __future__ import annotations

from contextlib import nullcontext

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from carry_box_data import LIGHT_NAMES, REFERENCE_LIGHT_COLOR, Appearance
from shared.g1 import BODY_JOINTS, HAND_JOINTS, MJLAB_HAND_FROM_PSI0
from shared.state import RobotState
from sim.config import make_env_cfg, motor_parameters


class MjlabEnv:
    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        appearances: list[Appearance] | None = None,
        actuation: str = "position",
    ):
        self.device = torch.device(device)
        self.appearances = appearances
        self.actuation = actuation
        self.env = ManagerBasedRlEnv(
            cfg=make_env_cfg(num_envs, actuation=actuation),
            device=device,
            render_mode=None,
        )
        self._renderer = None
        self._render_data = mujoco.MjData(self.env.sim.mj_model)  # ty: ignore[unresolved-attribute]
        self._render_static = None
        self.robot = self.env.scene["robot"]
        self.box = self.env.scene["box"]
        body_ids, _ = self.robot.find_joints(BODY_JOINTS, preserve_order=True)
        hand_ids, _ = self.robot.find_joints(HAND_JOINTS, preserve_order=True)
        self.body_joint_ids = torch.as_tensor(body_ids, device=self.device)
        self.planner_joint_ids = torch.as_tensor(
            body_ids + hand_ids, device=self.device
        )
        scales, kp, kd, limits = motor_parameters()
        self.body_scale = torch.as_tensor(scales[:29], device=self.device)
        self.motor_kp = torch.as_tensor(kp, device=self.device)
        self.motor_kd = torch.as_tensor(kd, device=self.device)
        self.motor_limit = torch.as_tensor(limits, device=self.device)
        self.last_target = torch.empty(num_envs, 43, device=self.device)
        self.last_torque = torch.empty_like(self.last_target)
        model = self.env.sim.mj_model
        joint_ids = [
            model.joint(f"robot/{name}").id for name in BODY_JOINTS + HAND_JOINTS
        ]
        self.actuator_ids = torch.as_tensor(
            [
                int(np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)[0])
                for joint_id in joint_ids
            ],
            device=self.device,
        )
        self.mjlab_hand_from_psi0 = torch.as_tensor(
            MJLAB_HAND_FROM_PSI0, device=self.device
        )
        if self.device.type == "cuda":
            import warp as wp

            self.cuda_stream = wp.stream_to_torch(self.env.sim.wp_device)
        else:
            self.cuda_stream = None
        with self.compute_context():
            self.env.reset()

    def compute_context(self):
        return (
            torch.cuda.stream(self.cuda_stream)
            if self.cuda_stream is not None
            else nullcontext()
        )

    def robot_state(self) -> RobotState:
        data = self.robot.data
        return RobotState(
            root_ang_vel_b=data.root_link_ang_vel_b,
            projected_gravity_b=data.projected_gravity_b,
            joint_pos=data.joint_pos.index_select(-1, self.body_joint_ids),
            joint_vel=data.joint_vel.index_select(-1, self.body_joint_ids),
        )

    def planner_state(self) -> torch.Tensor:
        return self.robot.data.joint_pos.index_select(-1, self.planner_joint_ids)

    def box_pose(self) -> torch.Tensor:
        return self.box.data.root_link_pose_w

    def hold_action(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw actions that hold the current joint positions for the first delayed tick."""
        joints = self.planner_state()
        body_default = self.robot.data.default_joint_pos.index_select(
            -1, self.body_joint_ids
        )
        body = (joints[:, :29] - body_default) / self.body_scale
        hand_order = torch.argsort(self.mjlab_hand_from_psi0)
        hands = joints[:, 29:].index_select(-1, hand_order)
        return body, hands

    def rgb(self) -> torch.Tensor:
        """Return CPU uint8 RGB [n, 360, 640, 3] without advancing physics."""
        model = self.env.sim.mj_model
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height=360, width=640)
            self._render_free_joints = model.jnt_qposadr[
                model.jnt_type == mujoco.mjtJoint.mjJNT_FREE  # ty: ignore[unresolved-attribute]
            ]
            if self.appearances is not None:
                self._render_light_ids = [
                    model.light(f"simple/{name}").id for name in LIGHT_NAMES
                ]
                self._render_table_geom = model.geom("table/collision").id
                self._render_table_material = model.mat("table/pearl").id
        with self.compute_context():
            data = self.env.sim.data
            qpos = data.qpos.cpu().numpy()
            if self._render_static is None:
                # The room, table, and world origins stay fixed between resets.
                origins = self.env.scene.env_origins.cpu().numpy()
                mocap_pos = data.mocap_pos.cpu().numpy() - origins[:, None]
                mocap_quat = data.mocap_quat.cpu().numpy()
                self._render_static = origins, mocap_pos, mocap_quat
        origins, mocap_pos, mocap_quat = self._render_static
        images = np.empty((self.env.num_envs, 360, 640, 3), dtype=np.uint8)
        for index, (image, positions, mc_pos, mc_quat, origin) in enumerate(
            zip(images, qpos, mocap_pos, mocap_quat, origins)
        ):
            if self.appearances is not None:
                appearance = self.appearances[index]
                ids = self._render_light_ids
                model.light_pos[ids] = appearance.light_positions
                color_scale = appearance.light_diffuse / REFERENCE_LIGHT_COLOR
                model.light_diffuse[ids] = appearance.light_diffuse
                model.light_ambient[ids] = color_scale * (0.025, 0.028, 0.035)
                model.light_specular[ids] = color_scale * 0.12
                model.geom_rgba[self._render_table_geom] = appearance.table_rgba
                model.mat_rgba[self._render_table_material] = appearance.table_rgba
                model.mat_specular[self._render_table_material] = (
                    appearance.table_specular
                )
                model.mat_shininess[self._render_table_material] = (
                    appearance.table_shininess
                )
            self._render_data.qpos[:] = positions
            # Rebase each world so its camera sees the room at the same local pose.
            for address in self._render_free_joints:
                self._render_data.qpos[address : address + 3] -= origin
            self._render_data.mocap_pos[:] = mc_pos
            self._render_data.mocap_quat[:] = mc_quat
            mujoco.mj_forward(model, self._render_data)  # ty: ignore[unresolved-attribute]
            self._renderer.update_scene(self._render_data, camera="observation_camera")
            self._renderer.render(out=image)
        return torch.from_numpy(images)

    def step(self, body_action: torch.Tensor, hand_action: torch.Tensor) -> None:
        """Advance physics; the runtime owns episode outcomes and resets."""
        hand_action = hand_action.index_select(-1, self.mjlab_hand_from_psi0)
        env = self.env
        self.last_target = torch.cat(
            (
                self.robot.data.default_joint_pos.index_select(-1, self.body_joint_ids)
                + body_action * self.body_scale,
                hand_action,
            ),
            dim=-1,
        )
        if self.actuation == "position":
            env.action_manager.process_action(
                torch.cat((body_action, hand_action), dim=-1).to(self.device)
            )
        # This scene has only action terms and reset events. Bypass RL stepping
        # so an empty termination manager cannot cause a CUDA nonzero() wait.
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            if self.actuation == "torque":
                joint_pos = self.robot.data.joint_pos.index_select(
                    -1, self.planner_joint_ids
                )
                joint_vel = self.robot.data.joint_vel.index_select(
                    -1, self.planner_joint_ids
                )
                self.last_torque = torch.clamp(
                    self.motor_kp * (self.last_target - joint_pos)
                    - self.motor_kd * joint_vel,
                    -self.motor_limit,
                    self.motor_limit,
                )
                self.robot.set_joint_effort_target(
                    self.last_torque, joint_ids=self.planner_joint_ids
                )
            else:
                env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
        env.episode_length_buf += 1
        env.common_step_counter += 1
        # mj_step leaves derived state one substep behind integration.
        env.sim.forward()
        if self.actuation == "position":
            self.last_torque = env.sim.data.actuator_force.index_select(
                -1, self.actuator_ids
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        base_pose: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        box_pose: torch.Tensor | None = None,
        base_vel: torch.Tensor | None = None,
    ) -> None:
        self._render_static = None
        self.env.reset(seed=seed)
        if base_pose is not None:
            assert joint_pos is not None and box_pose is not None
            base_pose = base_pose.to(self.device).clone()
            box_pose = box_pose.to(self.device).clone()
            base_pose[:, :3] += self.env.scene.env_origins
            box_pose[:, :3] += self.env.scene.env_origins
            self.robot.write_root_link_pose_to_sim(base_pose)
            if base_vel is not None:
                self.robot.write_root_link_velocity_to_sim(base_vel.to(self.device))
            self.robot.write_joint_state_to_sim(
                joint_pos.to(self.device),
                torch.zeros_like(joint_pos, device=self.device),
                joint_ids=self.planner_joint_ids,
            )
            self.box.write_root_link_pose_to_sim(box_pose)
            self.env.sim.forward()
        self.last_target.zero_()
        self.last_torque.zero_()

    @property
    def step_dt(self) -> float:
        return self.env.step_dt

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self.env.close()
