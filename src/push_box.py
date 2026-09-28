from __future__ import annotations

import torch

from shared.state import RobotState


class PushBoxProbe:
    def __init__(self, state: RobotState, box_pos: torch.Tensor):
        self.start_root = state.root_pos_w[:, :2].clone()
        self.start_box = box_pos[:, :2].clone()
        self.start_yaw = self._yaw(state.root_quat_w).clone()
        self.previous_root = self.start_root.clone()
        self.path_length = torch.zeros(self.start_root.shape[0], device=box_pos.device)
        self.contact = torch.zeros_like(self.path_length, dtype=torch.bool)
        self.contact_duration = torch.zeros_like(self.path_length)
        self.previous_time = 0.0
        self.fell = torch.zeros_like(self.contact)
        self.trace: list[list[dict]] = [[] for _ in range(len(self.contact))]
        self.initial_distance = torch.linalg.vector_norm(
            self.start_root - self.start_box, dim=-1
        )
        self.minimum_distance = self.initial_distance.clone()

    @staticmethod
    def _yaw(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def update(
        self,
        state: RobotState,
        box_pos: torch.Tensor,
        contact: torch.Tensor,
        time_s: float,
    ) -> None:
        root = state.root_pos_w[:, :2]
        box = box_pos[:, :2]
        distance = torch.linalg.vector_norm(root - box, dim=-1)
        self.path_length += torch.linalg.vector_norm(root - self.previous_root, dim=-1)
        self.previous_root = root.clone()
        self.minimum_distance = torch.minimum(self.minimum_distance, distance)
        self.contact |= contact
        self.contact_duration += contact.float() * (time_s - self.previous_time)
        self.previous_time = time_s
        self.fell |= state.root_pos_w[:, 2] < 0.5
        heading = torch.atan2(
            torch.sin(self._yaw(state.root_quat_w) - self.start_yaw),
            torch.cos(self._yaw(state.root_quat_w) - self.start_yaw),
        )
        for i, trace in enumerate(self.trace):
            trace.append(
                {
                    "time_s": time_s,
                    "robot_to_box_distance_m": distance[i].item(),
                    "root_xy_m": root[i].tolist(),
                    "box_xy_m": box[i].tolist(),
                    "heading_change_rad": heading[i].item(),
                    "contact": bool(contact[i].item()),
                }
            )

    def results(
        self, state: RobotState, box_pos: torch.Tensor, reason: str
    ) -> list[dict]:
        root_delta = state.root_pos_w[:, :2] - self.start_root
        box_delta = box_pos[:, :2] - self.start_box
        heading = self._yaw(state.root_quat_w) - self.start_yaw
        heading = torch.atan2(torch.sin(heading), torch.cos(heading))
        results = []
        for i, trace in enumerate(self.trace):
            displacement = torch.linalg.vector_norm(root_delta[i]).item()
            box_displacement = torch.linalg.vector_norm(box_delta[i]).item()
            approach = (self.initial_distance[i] - self.minimum_distance[i]).item()
            if box_delta[i, 0].item() >= 0.3 and self.contact_duration[i].item() >= 0.5:
                behavior = "sustained useful push"
            elif box_displacement >= 0.05:
                behavior = "moves box"
            elif bool(self.contact[i]):
                behavior = "approaches and establishes contact"
            elif approach >= 0.3:
                behavior = "approaches box"
            elif self.path_length[i].item() >= 0.3:
                behavior = "locomotes but ignores box"
            else:
                behavior = "no meaningful motion"
            results.append(
                {
                    "behavior": behavior,
                    "robot_root_displacement_m": displacement,
                    "robot_root_delta_xy_m": root_delta[i].tolist(),
                    "robot_heading_change_rad": heading[i].item(),
                    "robot_path_length_m": self.path_length[i].item(),
                    "initial_robot_to_box_distance_m": self.initial_distance[i].item(),
                    "minimum_robot_to_box_distance_m": self.minimum_distance[i].item(),
                    "box_displacement_m": box_displacement,
                    "box_delta_xy_m": box_delta[i].tolist(),
                    "robot_box_contact": bool(self.contact[i].item()),
                    "robot_box_contact_duration_s": self.contact_duration[i].item(),
                    "fell": bool(self.fell[i].item()),
                    "termination_reason": reason,
                    "trace": trace,
                }
            )
        return results
