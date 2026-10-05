"""Preprocessing and direct inference for the released carry-box Psi-0 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from planner._psi0 import Psi0Model


class Psi0Planner:
    def __init__(
        self,
        run_dir: Path,
        ckpt_step: int,
        qwen_model: Path,
        device: str = "cuda",
        inference_steps: int = 10,
        rtc: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.inference_steps = inference_steps
        self.rtc = rtc
        saved = json.loads((run_dir / "run_config.json").read_text())
        config = saved["model"]
        self.exec_horizon = 24 if rtc else config["action_exec_horizon"]
        self.previous_actions: torch.Tensor | None = None
        field = saved["data"]["transform"]["field"]
        image_config = saved["data"]["transform"]["model"]
        self.image_transform = v2.Compose(
            (
                v2.Resize(
                    image_config["resize"]["size"],
                    interpolation=v2.InterpolationMode.NEAREST,
                ),
                v2.CenterCrop(image_config["center_crop"]["size"]),
            )
        )
        self.state_min = torch.tensor(field["state_min"], device=device)
        self.state_max = torch.tensor(field["state_max"], device=device)
        self.action_min = torch.tensor(field["action_min"], device=device)
        self.action_max = torch.tensor(field["action_max"], device=device)
        self.model = Psi0Model.from_pretrained(
            run_dir, ckpt_step, config, qwen_model, self.device
        )

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        span = self.state_max - self.state_min
        constant = span.abs() < 1e-4 * (
            self.state_max.abs() + self.state_min.abs() + 1e-8
        )
        normalized = (state - self.state_min) / span.masked_fill(constant, 1) * 2 - 1
        return normalized.masked_fill(constant, 0).clamp(-1, 1)

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        joint_positions: torch.Tensor,
        instructions: list[str],
    ) -> torch.Tensor:
        states = self._normalize_state(joint_positions.to(self.device)).unsqueeze(1)
        observations = [
            [self.image_transform(Image.fromarray(frame))]
            for frame in images.cpu().numpy()
        ]
        actions = self.model.predict_action(
            observations=observations,
            states=states,
            instructions=instructions,
            num_inference_steps=self.inference_steps,
            prev_actions=self.previous_actions,
            execution_horizon=self.exec_horizon if self.rtc else None,
        ).float()
        if self.rtc:
            self.previous_actions = torch.cat(
                (
                    actions[:, self.exec_horizon :],
                    torch.zeros_like(actions[:, : self.exec_horizon]),
                ),
                dim=1,
            )
        output = (
            0.5 * (actions + 1) * (self.action_max - self.action_min) + self.action_min
        )
        return output
