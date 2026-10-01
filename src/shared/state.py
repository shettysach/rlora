from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RobotState:
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
