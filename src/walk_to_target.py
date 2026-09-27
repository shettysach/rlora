from __future__ import annotations

from dataclasses import dataclass

import torch

from shared.state import RobotState

TASK_NAME = "WalkToTarget-v0"
INSTRUCTION = "Walk forward to the green target and stop."
GOAL_X = 2.0
GOAL_RADIUS = 0.2
STOP_SPEED = 0.2
STOP_HOLD_SECONDS = 0.5
FALL_HEIGHT = 0.5


@dataclass
class WalkToTarget:
    target_xy: torch.Tensor
    previous_xy: torch.Tensor
    initial_distance: torch.Tensor
    previous_distance: torch.Tensor
    path_length: torch.Tensor
    reward: torch.Tensor
    stopped_for: torch.Tensor
    success_time: torch.Tensor
    success: torch.Tensor
    fell: torch.Tensor

    @classmethod
    def start(cls, state: RobotState, episode_seconds: float) -> WalkToTarget:
        xy = state.root_pos_w[:, :2].clone()
        target_xy = xy.new_tensor((GOAL_X, 0.0))
        distance = torch.linalg.vector_norm(xy - target_xy, dim=-1)
        return cls(
            target_xy=target_xy,
            previous_xy=xy,
            initial_distance=distance,
            previous_distance=distance.clone(),
            path_length=torch.zeros_like(distance),
            reward=torch.zeros_like(distance),
            stopped_for=torch.zeros_like(distance),
            success_time=torch.full_like(distance, episode_seconds),
            success=torch.zeros_like(distance, dtype=torch.bool),
            fell=torch.zeros_like(distance, dtype=torch.bool),
        )

    def update(self, state: RobotState, elapsed: float, step_dt: float) -> None:
        xy = state.root_pos_w[:, :2]
        distance = torch.linalg.vector_norm(xy - self.target_xy, dim=-1)
        speed = torch.linalg.vector_norm(state.root_lin_vel_w[:, :2], dim=-1)
        upright = state.projected_gravity_b[:, 2] < -0.7
        stopped_at_goal = (distance <= GOAL_RADIUS) & upright & (speed < STOP_SPEED)
        self.stopped_for = torch.where(
            stopped_at_goal,
            self.stopped_for + step_dt,
            torch.zeros_like(self.stopped_for),
        )
        new_success = (self.stopped_for >= STOP_HOLD_SECONDS) & ~self.success
        new_fall = (state.root_pos_w[:, 2] < FALL_HEIGHT) & ~self.fell
        self.success_time = torch.where(
            new_success,
            torch.full_like(self.success_time, elapsed + step_dt),
            self.success_time,
        )
        self.success |= new_success
        self.fell |= new_fall
        self.reward += self.previous_distance - distance
        self.reward += 5.0 * new_success - 5.0 * new_fall
        self.path_length += torch.linalg.vector_norm(xy - self.previous_xy, dim=-1)
        self.previous_xy = xy.clone()
        self.previous_distance = distance

    def results(self, state: RobotState, count: int) -> list[dict]:
        final_distance = torch.linalg.vector_norm(
            state.root_pos_w[:, :2] - self.target_xy, dim=-1
        )
        final_speed = torch.linalg.vector_norm(state.root_lin_vel_w[:, :2], dim=-1)
        return [
            {
                "success": bool(self.success[index].item()),
                "final_goal_distance_m": final_distance[index].item(),
                "initial_goal_distance_m": self.initial_distance[index].item(),
                "progress_m": (
                    self.initial_distance[index] - final_distance[index]
                ).item(),
                "fell": bool(self.fell[index].item()),
                "episode_time_s": self.success_time[index].item(),
                "path_length_m": self.path_length[index].item(),
                "final_root_speed_mps": final_speed[index].item(),
                "return": self.reward[index].item(),
            }
            for index in range(count)
        ]
