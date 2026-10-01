"""Success and diagnostics for the SIMPLE floor-to-table box task."""

from __future__ import annotations

import torch
import warp as wp

from sim.env import MjlabEnv

TABLE_TOP_Z = 0.45
SUCCESS_REWARD = 0.9


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class CarryBoxTask:
    def __init__(self, env: MjlabEnv):
        self.env = env
        model = env.env.sim.mj_model
        self.box_geom = model.geom("box/collision").id
        self.table_geom = model.geom("table/collision").id
        self.hand_geoms = torch.tensor(
            [
                geom_id
                for geom_id in range(model.ngeom)
                if "hand" in model.body(int(model.geom(geom_id).bodyid[0])).name
                and model.body(int(model.geom(geom_id).bodyid[0])).name.startswith(
                    "robot/"
                )
            ],
            device=env.device,
        )
        self.reset()

    def reset(self) -> None:
        robot = self.env.robot_state()
        box = self.env.box_pose()
        self.initial_root = robot.root_pos_w.clone()
        self.initial_yaw = _yaw(robot.root_quat_w)
        self.initial_box = box[:, :3].clone()
        self.minimum_distance = torch.linalg.vector_norm(
            robot.root_pos_w[:, :2] - box[:, :2], dim=-1
        )
        self.max_box_height = box[:, 2].clone()
        count = len(self.minimum_distance)
        self.reward = torch.zeros(count, device=self.env.device)
        self.success = torch.zeros(count, dtype=torch.bool, device=self.env.device)
        self.fell = torch.zeros_like(self.success)
        self.hand_contact_ever = torch.zeros_like(self.success)
        self.table_contact_ever = torch.zeros_like(self.success)
        self.first_success_step = torch.full((count,), -1, device=self.env.device)
        self.distance_trace: list[torch.Tensor] = []
        self.box_height_trace: list[torch.Tensor] = []

    def _contacts(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.env.env.sim.wp_data
        count = int(data.nacon.numpy()[0])
        hand = torch.zeros_like(self.success)
        table = torch.zeros_like(self.success)
        if count == 0:
            return hand, table
        geom = wp.to_torch(data.contact.geom)[:count]
        world = wp.to_torch(data.contact.worldid)[:count].long()
        pos = wp.to_torch(data.contact.pos)[:count]
        dist = wp.to_torch(data.contact.dist)[:count]
        box_first = geom[:, 0] == self.box_geom
        box_second = geom[:, 1] == self.box_geom
        hand_mask = (box_first & torch.isin(geom[:, 1], self.hand_geoms)) | (
            box_second & torch.isin(geom[:, 0], self.hand_geoms)
        )
        table_mask = (
            (box_first & (geom[:, 1] == self.table_geom))
            | (box_second & (geom[:, 0] == self.table_geom))
        ) & ((pos[:, 2] - TABLE_TOP_Z).abs() <= 0.05)
        hand_mask &= dist <= 0.01
        table_mask &= dist <= 0.01
        hand = torch.bincount(world[hand_mask], minlength=len(hand)) > 0
        table = torch.bincount(world[table_mask], minlength=len(table)) > 0
        return hand, table

    def update(self, step: int) -> None:
        robot = self.env.robot_state()
        box = self.env.box_pose()
        hand_contact, table_contact = self._contacts()
        self.hand_contact_ever |= hand_contact
        self.table_contact_ever |= table_contact
        distance = torch.linalg.vector_norm(
            robot.root_pos_w[:, :2] - box[:, :2], dim=-1
        )
        self.distance_trace.append(distance)
        self.box_height_trace.append(box[:, 2])
        self.minimum_distance = torch.minimum(self.minimum_distance, distance)
        self.max_box_height = torch.maximum(self.max_box_height, box[:, 2])
        placed = table_contact & ~hand_contact & (box[:, 2] >= TABLE_TOP_Z)
        self.reward += placed.float() * self.env.step_dt
        newly_successful = (self.reward > SUCCESS_REWARD) & ~self.success
        self.first_success_step = torch.where(
            newly_successful, step, self.first_success_step
        )
        self.success |= newly_successful
        self.fell |= robot.root_pos_w[:, 2] < 0.5

    def results(self, steps: int) -> list[dict]:
        robot = self.env.robot_state()
        box = self.env.box_pose()
        heading = _yaw(robot.root_quat_w) - self.initial_yaw
        heading = torch.atan2(torch.sin(heading), torch.cos(heading))
        distances = torch.stack(self.distance_trace).T.tolist()
        heights = torch.stack(self.box_height_trace).T.tolist()
        return [
            {
                "success": bool(self.success[i].item()),
                "fell": bool(self.fell[i].item()),
                "termination_reason": "task_success"
                if self.success[i]
                else "fell"
                if self.fell[i]
                else "time_limit",
                "first_success_step": int(self.first_success_step[i].item()),
                "root_displacement_m": (
                    robot.root_pos_w[i] - self.initial_root[i]
                ).tolist(),
                "heading_change_rad": heading[i].item(),
                "minimum_robot_box_distance_m": self.minimum_distance[i].item(),
                "robot_box_distance_trace_m": distances[i],
                "box_displacement_m": (box[i, :3] - self.initial_box[i]).tolist(),
                "max_box_height_m": self.max_box_height[i].item(),
                "box_height_trace_m": heights[i],
                "hand_box_contact": bool(self.hand_contact_ever[i].item()),
                "box_table_contact": bool(self.table_contact_ever[i].item()),
                "placement_reward": self.reward[i].item(),
                "executed_steps": steps,
            }
            for i in range(len(self.success))
        ]
