from __future__ import annotations

from contextlib import nullcontext

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
        # MJLab senses automatically on step/reset. Keep its captured pipeline
        # for explicit RGB requests; this scene has no other context sensors.
        self._render_camera = self.env.sim.sense
        self.env.sim.sense = lambda: None  # ty: ignore[invalid-assignment]
        self.robot = self.env.scene["robot"]
        self.box = self.env.scene["box"]
        self.camera = self.env.scene["observation_camera"]
        body_ids, _ = self.robot.find_joints(BODY_JOINTS, preserve_order=True)
        hand_ids, _ = self.robot.find_joints(HAND_JOINTS, preserve_order=True)
        self.body_joint_ids = torch.as_tensor(body_ids, device=self.device)
        self.hand_joint_ids = torch.as_tensor(hand_ids, device=self.device)
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
            root_pos_w=data.root_link_pos_w,
            root_quat_w=data.root_link_quat_w,
            root_lin_vel_w=data.root_link_lin_vel_w,
            root_ang_vel_b=data.root_link_ang_vel_b,
            projected_gravity_b=data.projected_gravity_b,
            joint_pos=data.joint_pos.index_select(-1, self.body_joint_ids),
            joint_vel=data.joint_vel.index_select(-1, self.body_joint_ids),
            hand_pos=data.joint_pos.index_select(-1, self.hand_joint_ids),
        )

    def planner_state(self) -> torch.Tensor:
        state = self.robot_state()
        assert state.hand_pos is not None
        return torch.cat((state.joint_pos, state.hand_pos), dim=-1)

    def box_pose(self) -> torch.Tensor:
        return self.box.data.root_link_pose_w

    def rgb(self) -> torch.Tensor:
        with self.compute_context():
            self._render_camera()
            return self.camera.data.rgb

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
        self.env.reset(seed=seed)
        if base_pose is not None:
            assert joint_pos is not None and box_pose is not None
            base_pose = base_pose.to(self.device).clone()
            box_pose = box_pose.to(self.device).clone()
            base_pose[:, :3] += self.env.scene.env_origins
            box_pose[:, :3] += self.env.scene.env_origins
            self.robot.write_root_link_pose_to_sim(base_pose)
            joint_ids = torch.cat((self.body_joint_ids, self.hand_joint_ids))
            self.robot.write_joint_state_to_sim(
                joint_pos.to(self.device),
                torch.zeros_like(joint_pos, device=self.device),
                joint_ids=joint_ids,
            )
            self.box.write_root_link_pose_to_sim(box_pose)
            self.env.sim.forward()

    @property
    def step_dt(self) -> float:
        return self.env.step_dt

    def close(self) -> None:
        self.env.close()
