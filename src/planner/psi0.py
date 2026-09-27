from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from planner._psi0 import Psi0Model
from planner._psi0.lora import load_adapter
from planner._psi0.preprocessing import normalize_bounds


class Psi0Planner:
    """Checkpoint-specific Psi-0 preprocessing and batched inference."""

    def __init__(
        self,
        run_dir: Path,
        ckpt_step: int,
        qwen_model: Path,
        clip_model: Path,
        device: str = "cuda",
        inference_steps: int = 8,
        bc_checkpoint: Path | None = None,
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
            run_dir, ckpt_step, model_config, qwen_model, self.device
        )
        if bc_checkpoint is not None:
            adapter_config = json.loads((bc_checkpoint / "bc_config.json").read_text())
            base = adapter_config["model"]
            config_hash = hashlib.sha256(
                (run_dir / "run_config.json").read_bytes()
            ).hexdigest()
            if (
                base["base_config_sha256"] != config_hash
                or base["ckpt_step"] != ckpt_step
            ):
                raise ValueError(
                    "BC adapter was trained against a different base checkpoint"
                )
            load_adapter(
                self.model.action_header,
                bc_checkpoint / "adapter.safetensors",
                base["lora_rank"],
            )
        self.horizon = model_config["action_chunk_size"]
        self.clip_model = clip_model
        self.text_encoder = None
        cache_path = run_dir / model_config["pooled_cache_path"]
        cache = torch.load(cache_path, map_location=self.device, weights_only=True)
        self.pooled = {
            key.lower(): value.to(self.device) for key, value in cache.items()
        }
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.previous_actions: torch.Tensor | None = None

    def reset(self) -> None:
        self.previous_actions = None

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return normalize_bounds(state, self.state_min, self.state_max, state=True)

    @torch.no_grad()
    def prepare_batch(self, samples: list[dict]) -> dict:
        """Prepare converted demonstrations using the checkpoint's transforms."""
        instructions = [sample["instruction"].lower() for sample in samples]
        states = torch.stack([sample["states"] for sample in samples]).to(self.device)
        actions = torch.stack([sample["actions"] for sample in samples]).to(self.device)
        return {
            "observations": [
                [self.image_transform(sample["image"])] for sample in samples
            ],
            "states": self._normalize_state(states),
            "instructions": instructions,
            "actions": normalize_bounds(actions, self.action_min, self.action_max),
            "actions_mask": torch.stack(
                [sample["actions_mask"] for sample in samples]
            ).to(self.device),
            "pooled_projections": self._pooled_projections(instructions),
        }

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

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        joint_positions: torch.Tensor,
        instructions: list[str],
        executed_actions: int = 0,
        inference_delay: int = 6,
    ) -> torch.Tensor:
        stream_context = (
            torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        )
        with stream_context:
            joint_positions = joint_positions.to(self.device)
            batch = joint_positions.shape[0]
            missing_joints = joint_positions.new_zeros((batch, 16))
            planner_state = torch.cat((joint_positions, missing_joints), dim=-1)
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
            previous_actions = None
            if self.previous_actions is not None:
                tail = self.previous_actions[:, executed_actions:]
                previous_actions = torch.nn.functional.pad(
                    tail, (0, 0, 0, executed_actions)
                )
            actions = self.model.predict_action(
                observations=observations,
                states=normalized,
                instructions=lowered,
                num_inference_steps=self.inference_steps,
                pooled_projections=self._pooled_projections(lowered),
                previous_actions=previous_actions,
                inference_delay=inference_delay,
                execution_horizon=executed_actions,
            ).float()
        if self.stream is not None:
            self.stream.synchronize()
        self.previous_actions = actions
        low = self.action_min[:64]
        high = self.action_max[:64]
        body_token = 0.5 * (actions[..., :64] + 1) * (high - low) + low
        return body_token.clamp(-0.625, 0.625).mul(16).round().div(16)
