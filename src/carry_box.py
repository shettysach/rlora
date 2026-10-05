"""Placement success check for the SIMPLE floor-to-table box task."""

from __future__ import annotations

import torch
import warp as wp

from sim.env import MjlabEnv

# SIMPLE's checker uses the table pose's center Z, despite calling it the top.
TABLE_CENTER_Z = 0.4
SUCCESS_PLACEMENT_TIME_S = 0.9


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
        # Match SIMPLE's Python-float accumulation at the strict > 0.9 boundary.
        self.placement_time_s = torch.zeros(
            count, dtype=torch.float64, device=self.env.device
        )
        self.success = torch.zeros(count, dtype=torch.bool, device=self.env.device)

    def _contacts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.env.env.sim.wp_data
        active = self.contact_indices < wp.to_torch(data.nacon)[0]
        geom = wp.to_torch(data.contact.geom)
        world = wp.to_torch(data.contact.worldid).long().masked_fill(~active, 0)
        pos = wp.to_torch(data.contact.pos)
        box_first = geom[:, 0] == self.box_geom
        box_second = geom[:, 1] == self.box_geom
        hand_mask = (box_first & torch.isin(geom[:, 1], self.hand_geoms)) | (
            box_second & torch.isin(geom[:, 0], self.hand_geoms)
        )
        table_mask = (box_first & (geom[:, 1] == self.table_geom)) | (
            box_second & (geom[:, 0] == self.table_geom)
        )
        table_height_mask = table_mask & (
            (pos[:, 2].double() - TABLE_CENTER_Z).abs() <= 0.05
        )
        # SIMPLE scans all registered contacts without another distance filter.
        hand_mask &= active
        table_mask &= active
        table_height_mask &= active
        hand = torch.zeros_like(self.placement_time_s, dtype=torch.int32)
        table = torch.zeros_like(hand)
        table_height = torch.zeros_like(hand)
        hand.scatter_add_(0, world, hand_mask.int())
        table.scatter_add_(0, world, table_mask.int())
        table_height.scatter_add_(0, world, table_height_mask.int())
        return hand > 0, table > 0, table_height > 0

    def update(self) -> None:
        box = self.env.box_pose()
        self.hand_contact, self.table_contact, self.table_height_contact = (
            self._contacts()
        )
        self.box_height_ok = box[:, 2].double() >= TABLE_CENTER_Z
        self.placed = (
            self.table_height_contact & ~self.hand_contact & self.box_height_ok
        )
        self.placement_time_s += self.placed.double() * self.env.step_dt
        self.success |= self.placement_time_s > SUCCESS_PLACEMENT_TIME_S
