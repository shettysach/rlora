from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from planner._psi0 import Psi0Model
from shared.state import RobotState


class Psi0Planner:
    """Checkpoint-specific Psi-0 preprocessing and batched inference."""

    def __init__(
        self,
        run_dir: Path,
        ckpt_step: int,
        clip_model: Path,
        device: str = "cuda",
        inference_steps: int = 8,
    ):
        self.device = torch.device(device)
        self.inference_steps = inference_steps

        saved = json.loads((run_dir / "run_config.json").read_text())
        model_config = saved["model"]
        transform = saved["data"]["transform"]
        field = transform["field"]

        image_transform = transform["model"]
        self.image_transform = v2.Compose(
            (
                v2.Resize(image_transform["resize"]["size"]),
                v2.CenterCrop(image_transform["center_crop"]["size"]),
            )
        )
        self.state_min = torch.tensor(
            field["state_min"], device=self.device, dtype=torch.float32
        )
        self.state_max = torch.tensor(
            field["state_max"], device=self.device, dtype=torch.float32
        )
        self.action_min = torch.tensor(
            field["action_min"], device=self.device, dtype=torch.float32
        )
        self.action_max = torch.tensor(
            field["action_max"], device=self.device, dtype=torch.float32
        )

        self.model = Psi0Model.from_pretrained(
            run_dir, ckpt_step, model_config, self.device
        )
        self.exec_horizon = model_config["action_exec_horizon"]
        self.clip_model = clip_model
        self.text_encoder = None
        cache_path = run_dir / model_config["pooled_cache_path"]
        cache = torch.load(cache_path, map_location=self.device, weights_only=True)
        self.pooled = {
            key.lower(): value.to(self.device) for key, value in cache.items()
        }

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        span = self.state_max - self.state_min
        constant = span.abs() < 1e-4 * (
            self.state_max.abs() + self.state_min.abs() + 1e-8
        )
        normalized = (state - self.state_min) / span.masked_fill(constant, 1) * 2 - 1
        return normalized.masked_fill(constant, 0).clamp(-1, 1)

    def _pooled_projections(self, instructions: list[str]) -> torch.Tensor:
        missing = [text for text in instructions if text not in self.pooled]
        if missing:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer

            if self.text_encoder is None:
                tokenizer = CLIPTokenizer.from_pretrained(
                    self.clip_model, local_files_only=True
                )
                text_model = CLIPTextModelWithProjection.from_pretrained(
                    self.clip_model,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                )
                nn.Module.to(text_model, self.device)
                text_model.eval().requires_grad_(False)
                self.text_encoder = tokenizer, text_model
            tokenizer, text_model = self.text_encoder
            for text in dict.fromkeys(missing):
                tokens = tokenizer(
                    [text], padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                self.pooled[text] = text_model(**tokens).text_embeds[0]
        return torch.stack([self.pooled[text] for text in instructions])

    @torch.inference_mode()
    def predict(
        self, images: torch.Tensor, states: RobotState, instructions: list[str]
    ) -> torch.Tensor:
        batch = states.joint_pos.shape[0]
        missing_joints = states.joint_pos.new_zeros((batch, 16))
        planner_state = torch.cat((states.joint_pos, missing_joints), dim=-1)
        normalized = self._normalize_state(planner_state).unsqueeze(1)
        # The checkpoint's ZED Mini videos contain eight padded rows below the
        # native 672x376 image.
        images = torch.cat(
            (images, images.new_zeros((batch, 8, images.shape[2], 3))), dim=1
        )
        observations = [
            [self.image_transform(Image.fromarray(frame))]
            for frame in images.cpu().numpy()
        ]
        lowered = [instruction.lower() for instruction in instructions]
        actions = self.model.predict_action(
            observations=observations,
            states=normalized,
            instructions=lowered,
            num_inference_steps=self.inference_steps,
            pooled_projections=self._pooled_projections(lowered),
        ).float()
        low = self.action_min[:64]
        high = self.action_max[:64]
        body_token = 0.5 * (actions[..., :64] + 1) * (high - low) + low
        return body_token.clamp(-0.625, 0.625).mul(16).round().div(16)
