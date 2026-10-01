"""Placement and fall checks for the SIMPLE floor-to-table box task."""

from __future__ import annotations

import torch
import warp as wp

from sim.env import MjlabEnv

TABLE_TOP_Z = 0.45
SUCCESS_REWARD = 0.9


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
        self.contact_indices = torch.arange(
            env.env.sim.wp_data.contact.worldid.shape[0], device=env.device
        )
        self.reset()

    def reset(self) -> None:
        count = self.env.box_pose().shape[0]
        self.reward = torch.zeros(count, device=self.env.device)
        self.success = torch.zeros(count, dtype=torch.bool, device=self.env.device)
        self.fell = torch.zeros_like(self.success)

    def _contacts(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.env.env.sim.wp_data
        active = self.contact_indices < wp.to_torch(data.nacon)[0]
        geom = wp.to_torch(data.contact.geom)
        world = wp.to_torch(data.contact.worldid).long().masked_fill(~active, 0)
        pos = wp.to_torch(data.contact.pos)
        dist = wp.to_torch(data.contact.dist)
        box_first = geom[:, 0] == self.box_geom
        box_second = geom[:, 1] == self.box_geom
        hand_mask = (box_first & torch.isin(geom[:, 1], self.hand_geoms)) | (
            box_second & torch.isin(geom[:, 0], self.hand_geoms)
        )
        table_mask = (
            (box_first & (geom[:, 1] == self.table_geom))
            | (box_second & (geom[:, 0] == self.table_geom))
        ) & ((pos[:, 2] - TABLE_TOP_Z).abs() <= 0.05)
        hand_mask &= active & (dist <= 0.01)
        table_mask &= active & (dist <= 0.01)
        hand = torch.zeros_like(self.reward, dtype=torch.int32)
        table = torch.zeros_like(hand)
        hand.scatter_add_(0, world, hand_mask.int())
        table.scatter_add_(0, world, table_mask.int())
        return hand > 0, table > 0

    def update(self) -> None:
        box = self.env.box_pose()
        hand_contact, table_contact = self._contacts()
        placed = table_contact & ~hand_contact & (box[:, 2] >= TABLE_TOP_Z)
        self.reward += placed.float() * self.env.step_dt
        self.success |= self.reward > SUCCESS_REWARD
        self.fell |= self.env.robot.data.root_link_pos_w[:, 2] < 0.5
