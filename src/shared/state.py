from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RobotState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    hand_pos: torch.Tensor | None = None
