from __future__ import annotations

from contextlib import nullcontext

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from shared.g1 import BODY_JOINTS, HAND_JOINTS, MJLAB_HAND_FROM_PSI0
from shared.state import RobotState
from sim.config import make_env_cfg


class MjlabEnv:
    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.env = ManagerBasedRlEnv(
            cfg=make_env_cfg(num_envs),
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

    def rgb(self) -> torch.Tensor:
        """Return CPU uint8 RGB [n, 360, 640, 3] without advancing physics."""
        model = self.env.sim.mj_model
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height=360, width=640)
            self._render_free_joints = model.jnt_qposadr[
                model.jnt_type == mujoco.mjtJoint.mjJNT_FREE  # ty: ignore[unresolved-attribute]
            ]
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
        for image, positions, mc_pos, mc_quat, origin in zip(
            images, qpos, mocap_pos, mocap_quat, origins
        ):
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
        hand_action = hand_action.index_select(-1, self.mjlab_hand_from_psi0)
        self.env.step(torch.cat((body_action, hand_action), dim=-1))

    def reset(
        self,
        *,
        seed: int | None = None,
        base_pose: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        box_pose: torch.Tensor | None = None,
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
            self.robot.write_joint_state_to_sim(
                joint_pos.to(self.device),
                torch.zeros_like(joint_pos, device=self.device),
                joint_ids=self.planner_joint_ids,
            )
            self.box.write_root_link_pose_to_sim(box_pose)
            self.env.sim.forward()

    @property
    def step_dt(self) -> float:
        return self.env.step_dt

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self.env.close()
