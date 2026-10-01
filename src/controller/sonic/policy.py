from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from controller.sonic.model import SonicModel
from shared.g1 import DEFAULT_JOINT_POS_MJLAB, MJLAB_FROM_SONIC, SONIC_FROM_MJLAB
from shared.state import RobotState


class SonicPolicy:
    DECODER_SHA256 = "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"
    TOKEN = slice(0, 64)
    BASE_ANGULAR_VELOCITY = slice(64, 94)
    JOINT_POSITIONS = slice(94, 384)
    JOINT_VELOCITIES = slice(384, 674)
    LAST_ACTIONS = slice(674, 964)
    GRAVITY = slice(964, 994)

    def __init__(
        self,
        bundle_dir: Path,
        batch_size: int,
        device: str = "cuda",
        cuda_stream: torch.cuda.Stream | None = None,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        decoder = bundle_dir / "model_decoder.onnx"
        if hashlib.sha256(decoder.read_bytes()).hexdigest() != self.DECODER_SHA256:
            raise ValueError(
                "Expected the pinned older SONIC decoder for carry-box replay"
            )
        self.model = SonicModel(
            decoder,
            batch_size,
            self.GRAVITY.stop,
            self.device,
            cuda_stream,
        )
        self.default_joint_pos = torch.as_tensor(
            DEFAULT_JOINT_POS_MJLAB, device=self.device
        )
        self.sonic_from_mjlab = torch.as_tensor(SONIC_FROM_MJLAB, device=self.device)
        self.mjlab_from_sonic = torch.as_tensor(MJLAB_FROM_SONIC, device=self.device)

    def reset(self) -> None:
        self.model.input.zero_()

    def act(
        self,
        *,
        reference: torch.Tensor,
        robot_state: RobotState,
    ) -> torch.Tensor:
        joint_pos = (robot_state.joint_pos - self.default_joint_pos).index_select(
            -1, self.sonic_from_mjlab
        )
        joint_vel = robot_state.joint_vel.index_select(-1, self.sonic_from_mjlab)
        self.model.input[:, self.TOKEN].copy_(
            reference.clamp(-0.625, 0.625).mul(16).round().div(16)
        )
        self._history(self.BASE_ANGULAR_VELOCITY, robot_state.root_ang_vel_b)
        self._history(self.JOINT_POSITIONS, joint_pos)
        self._history(self.JOINT_VELOCITIES, joint_vel)
        self._history(self.GRAVITY, robot_state.projected_gravity_b)
        # The decoder consumes the previous action before the current one replaces it.
        action_sonic = self.model.run()
        self._history(self.LAST_ACTIONS, action_sonic)
        return action_sonic.index_select(-1, self.mjlab_from_sonic)

    def _history(self, field: slice, value: torch.Tensor) -> None:
        history = self.model.input[:, field].view(self.batch_size, 10, -1)
        history[:, :-1].copy_(history[:, 1:].clone())
        history[:, -1].copy_(value)
