from __future__ import annotations

from contextlib import nullcontext

import torch
from mjlab.envs import ManagerBasedRlEnv

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
        self.robot = self.env.scene["robot"]
        self.camera = self.env.scene["observation_camera"]
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
            root_ang_vel_b=data.root_link_ang_vel_b,
            projected_gravity_b=data.projected_gravity_b,
            joint_pos=data.joint_pos,
            joint_vel=data.joint_vel,
        )

    def rgb(self) -> torch.Tensor:
        return self.camera.data.rgb

    def step(self, action: torch.Tensor) -> None:
        self.env.step(action)

    @property
    def step_dt(self) -> float:
        return self.env.step_dt

    def close(self) -> None:
        self.env.close()
