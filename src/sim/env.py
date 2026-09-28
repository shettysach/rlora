from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import cast

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from shared.state import RobotState
from sim.config import make_env_cfg


class MjlabEnv:
    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        record_video: Path | None = None,
        push_box: bool = False,
        seed: int | None = None,
    ):
        self.device = torch.device(device)
        self.record_video = record_video
        self.video_frames: list[np.ndarray] = []
        self.env = ManagerBasedRlEnv(
            cfg=make_env_cfg(num_envs, push_box=push_box),
            device=device,
            render_mode="rgb_array" if record_video is not None else None,
        )
        self.robot = self.env.scene["robot"]
        self.box = self.env.scene["box"] if push_box else None
        self.box_contact = self.env.scene["robot_box_contact"] if push_box else None
        self.camera = self.env.scene["observation_camera"]
        if self.device.type == "cuda":
            import warp as wp

            self.cuda_stream = wp.stream_to_torch(self.env.sim.wp_device)
        else:
            self.cuda_stream = None
        with self.compute_context():
            self.env.reset(seed=seed)

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
            joint_pos=data.joint_pos,
            joint_vel=data.joint_vel,
        )

    def rgb(self) -> torch.Tensor:
        return self.camera.data.rgb

    def box_position(self) -> torch.Tensor:
        assert self.box is not None
        return self.box.data.root_link_pos_w

    def touching_box(self) -> torch.Tensor:
        assert self.box_contact is not None
        return self.box_contact.data.found.any(dim=-1)

    def step(self, action: torch.Tensor) -> None:
        self.env.step(action)
        if self.record_video is not None:
            frame = cast(np.ndarray, self.env.render())
            self.video_frames.append(frame[0] if frame.ndim == 4 else frame)

    def reset(self, seed: int | None = None) -> None:
        self.env.reset(seed=seed)

    @property
    def step_dt(self) -> float:
        return self.env.step_dt

    def close(self) -> None:
        self.env.close()
        if self.record_video is not None and self.video_frames:
            import mediapy

            self.record_video.parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_video(
                str(self.record_video),
                self.video_frames,
                fps=round(1.0 / self.step_dt),
            )
