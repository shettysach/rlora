"""Inference-only Psi-0 model used by the planner."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)
from PIL import Image
from qwen_vl_utils import process_vision_info
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.utils import is_flash_attn_2_available


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2) * (-math.log(10_000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.pe = nn.Parameter(encoding.unsqueeze(1), requires_grad=False)


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dimension, 2 * dimension)
        self.norm = nn.LayerNorm(dimension, elementwise_affine=False, eps=1e-6)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = condition.unsqueeze(1)
        scale, shift = self.linear(self.silu(condition)).chunk(2, dim=-1)
        return self.norm(value) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dimension, 6 * dimension)
        self.norm = nn.LayerNorm(dimension, elementwise_affine=False, eps=1e-6)

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        modulation = self.linear(self.silu(condition))
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(6, dim=-1)
        normalized = self.norm(value) * (1 + scale_attention[:, None])
        normalized = normalized + shift_attention[:, None]
        return normalized, gate_attention, shift_mlp, scale_mlp, gate_mlp


class TimeNetwork(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.w = nn.Parameter(
            torch.exp(torch.arange(128) * (-math.log(10_000) / 127)),
            requires_grad=False,
        )
        self.out_net = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        phase = timestep[:, None] * self.w
        return self.out_net(torch.cat((phase.cos(), phase.sin()), dim=-1))


class JointVLAAttnProcessor:
    def __call__(
        self,
        attention: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        head_dim = attention.inner_dim // attention.heads

        to_k = cast(nn.Linear, attention.to_k)
        to_v = cast(nn.Linear, attention.to_v)
        add_k_proj = cast(nn.Linear, attention.add_k_proj)
        add_v_proj = cast(nn.Linear, attention.add_v_proj)
        to_out = cast(nn.ModuleList, attention.to_out)

        query = attention.to_q(hidden_states)
        key = to_k(hidden_states)
        value = to_v(hidden_states)
        query = query.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        context_key = add_k_proj(encoder_hidden_states)
        context_value = add_v_proj(encoder_hidden_states)
        context_key = context_key.view(
            batch_size, -1, attention.heads, head_dim
        ).transpose(1, 2)
        context_value = context_value.view(
            batch_size, -1, attention.heads, head_dim
        ).transpose(1, 2)
        action_length = hidden_states.shape[1]
        # The final block consumes context but never updates it.
        if not attention.context_pre_only:
            context_query = cast(nn.Linear, attention.add_q_proj)(encoder_hidden_states)
            context_query = context_query.view(
                batch_size, -1, attention.heads, head_dim
            ).transpose(1, 2)
            query = torch.cat((query, context_query), dim=2)
        key = torch.cat((key, context_key), dim=2)
        value = torch.cat((value, context_value), dim=2)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask[:, None, None]
        )
        output = output.transpose(1, 2).reshape(batch_size, -1, attention.inner_dim)
        if attention.context_pre_only:
            # Preserve the action slice's batch stride for BF16 projection rounding.
            output = F.pad(output, (0, 0, 0, encoder_hidden_states.shape[1]))
        action_output = to_out[1](to_out[0](output[:, :action_length]))
        context_output = output[:, action_length:]
        if not attention.context_pre_only:
            context_output = cast(nn.Linear, attention.to_add_out)(context_output)
        return action_output, context_output


class ObservationProjection(nn.Module):
    def __init__(self, odim: int, view_feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.enc_pos = PositionalEncoding(hidden_dim)
        self.views_proj = nn.Linear(view_feature_dim, hidden_dim)
        self._obs_proc = nn.Sequential(nn.Dropout(0), nn.Linear(odim, hidden_dim))

    def forward(
        self, views: torch.Tensor, state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_tokens = self.views_proj(views[:, 0])
        state_token = self._obs_proc(state) + self.enc_pos.pe[
            : state.shape[1]
        ].transpose(0, 1)
        mask = torch.cat(
            (attention_mask, torch.ones_like(attention_mask[:, : state.shape[1]])),
            dim=1,
        )
        return torch.cat((view_tokens, state_token), dim=1), mask


class ActionProjectionIn(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, horizon: int) -> None:
        super().__init__()
        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, hidden_dim),
        )
        self.dec_pos = nn.Parameter(torch.empty(horizon, hidden_dim))
        nn.init.xavier_uniform_(self.dec_pos)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.ac_proj(action) + self.dec_pos


class ActionProjectionOut(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, action_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        value = self.norm_final(value) * (1 + scale[:, None]) + shift[:, None]
        return self.linear(value)


class VLATransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, context_pre_only: bool) -> None:
        super().__init__()
        self.norm1_act = AdaLayerNormZero(hidden_dim)
        self.norm1_obs = (
            AdaLayerNormContinuous(hidden_dim)
            if context_pre_only
            else AdaLayerNormZero(hidden_dim)
        )
        self.context_pre_only = context_pre_only
        self.attn = Attention(
            query_dim=hidden_dim,
            added_kv_proj_dim=hidden_dim,
            dim_head=hidden_dim // heads,
            heads=heads,
            out_dim=hidden_dim,
            context_pre_only=context_pre_only,
            bias=True,
            eps=1e-6,
            processor=cast(AttnProcessor, JointVLAAttnProcessor()),
        )
        self.norm2_act = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ff_act = FeedForward(
            hidden_dim,
            hidden_dim,
            activation_fn="gelu-approximate",
            final_dropout=False,
            bias=True,
        )
        if not context_pre_only:
            self.norm2_obs = nn.LayerNorm(
                hidden_dim, elementwise_affine=False, eps=1e-6
            )
            self.ff_obs = FeedForward(
                hidden_dim,
                hidden_dim,
                activation_fn="gelu-approximate",
                final_dropout=False,
                bias=True,
            )

    def forward(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, gate_attention, shift_mlp, scale_mlp, gate_mlp = self.norm1_act(
            action, condition
        )
        if self.context_pre_only:
            normalized_context = self.norm1_obs(context, condition)
        else:
            (
                normalized_context,
                context_gate,
                context_shift,
                context_scale,
                context_ff_gate,
            ) = self.norm1_obs(context, condition)
        attention_output, context_output = self.attn(
            normalized,
            encoder_hidden_states=normalized_context,
            attention_mask=attention_mask,
        )
        action = action + gate_attention[:, None] * attention_output
        normalized = self.norm2_act(action) * (1 + scale_mlp[:, None])
        normalized = normalized + shift_mlp[:, None]
        action = action + gate_mlp[:, None] * self.ff_act(normalized)
        if not self.context_pre_only:
            context = context + context_gate[:, None] * context_output
            normalized_context = self.norm2_obs(context) * (1 + context_scale[:, None])
            normalized_context = normalized_context + context_shift[:, None]
            context = context + context_ff_gate[:, None] * self.ff_obs(
                normalized_context
            )
        return action, context


class ActionTransformerModel(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        odim: int,
        view_feature_dim: int,
        hidden_dim: int,
        num_blocks: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.time_ins_embed = TimeNetwork(hidden_dim)
        self.obs_proj = ObservationProjection(odim, view_feature_dim, hidden_dim)
        self.action_proj_in = ActionProjectionIn(action_dim, hidden_dim, horizon)
        self.transformer_blocks = nn.ModuleList(
            VLATransformerBlock(hidden_dim, heads, index == num_blocks - 1)
            for index in range(num_blocks)
        )
        self.action_proj_out = ActionProjectionOut(hidden_dim, action_dim)

    def forward(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.time_ins_embed(timestep)
        action_tokens = self.action_proj_in(action)
        for block in self.transformer_blocks:
            action_tokens, context = block(
                action_tokens, context, condition, attention_mask
            )
        return self.action_proj_out(action_tokens, condition)


class Psi0Model(nn.Module):
    def __init__(
        self,
        action_header: ActionTransformerModel,
        vlm: Qwen3VLForConditionalGeneration,
        processor,
        scheduler: FlowMatchEulerDiscreteScheduler,
        action_dim: int,
        horizon: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.action_header = action_header
        self.vlm = vlm
        # Transformers 4.57's causal wrapper captures hidden_states[-1] before
        # this norm. Bypass it so the backbone returns the same conditioning.
        self.vlm.model.language_model.norm = nn.Identity()
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.scheduler = scheduler
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device
        self._conditioning_metadata = None

    @classmethod
    def from_pretrained(
        cls,
        run_dir: Path,
        checkpoint_step: int,
        model_config: dict,
        qwen_model: Path,
        device: torch.device,
    ) -> Psi0Model:
        checkpoint = (
            run_dir / "checkpoints" / f"ckpt_{checkpoint_step}" / "model.safetensors"
        )

        vlm_config = AutoConfig.from_pretrained(qwen_model, local_files_only=True)
        vlm_config._attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        vlm_config.dtype = torch.bfloat16
        vlm_config.text_config.dtype = torch.bfloat16
        vlm_config.vision_config.dtype = torch.bfloat16
        vlm = Qwen3VLForConditionalGeneration(vlm_config)
        vlm.bfloat16()
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            vlm_state = {
                key.removeprefix("vlm_model."): weights.get_tensor(key)
                for key in weights.keys()  # noqa: SIM118 - safe_open is not iterable
                if key.startswith("vlm_model.")
            }
        vlm_state["lm_head.weight"] = vlm_state[
            "model.language_model.embed_tokens.weight"
        ]
        if vlm.config.text_config.vocab_size != vlm_state["lm_head.weight"].shape[0]:
            vlm.resize_token_embeddings(
                vlm_state["lm_head.weight"].shape[0],
                pad_to_multiple_of=192,
                mean_resizing=False,
            )
        vlm.load_state_dict(vlm_state, strict=True)
        del vlm_state
        nn.Module.to(vlm, device)
        vlm.eval().requires_grad_(False)

        action_header = ActionTransformerModel(
            action_dim=model_config["action_dim"],
            horizon=model_config["action_chunk_size"],
            odim=model_config["odim"],
            view_feature_dim=model_config["view_feature_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_blocks=model_config["num_blocks"],
            heads=model_config["nhead"],
        )
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            action_state = {
                key.removeprefix("action_header."): weights.get_tensor(key)
                for key in weights.keys()  # noqa: SIM118 - safe_open is not iterable
                if key.startswith("action_header.")
            }
        action_header.load_state_dict(action_state, strict=True)
        action_header.to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
        # Rounding the fixed frequencies changes the phase at large timesteps.
        action_header.time_ins_embed.w = nn.Parameter(
            action_state["time_ins_embed.w"].to(device), requires_grad=False
        )
        del action_state

        processor = AutoProcessor.from_pretrained(qwen_model, local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=model_config["train_diffusion_steps"]
        )
        return cls(
            action_header,
            vlm,
            processor,
            scheduler,
            model_config["action_dim"],
            model_config["action_chunk_size"],
            device,
        )

    @torch.no_grad()
    def _conditioning(
        self,
        observations: list[list[Image.Image]],
        states: torch.Tensor,
        instructions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conversations = []
        texts = []
        for images, instruction in zip(observations, instructions):
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": instruction})
            messages = [{"role": "user", "content": content}]
            conversations.append(messages)
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        image_inputs, video_inputs = cast(
            tuple[object, object],
            process_vision_info(conversations, image_patch_size=16),
        )
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Match the vision dtype before upload, halving the BF16 pixel payload.
        pixels = inputs.pop("pixel_values").to(self.vlm.model.visual.dtype)
        metadata = (inputs.input_ids, inputs.attention_mask, inputs.image_grid_thw)
        # Compare CPU metadata; reuse its device tensors while prompts/grids stay fixed.
        if self._conditioning_metadata is None or any(
            not torch.equal(value, cached)
            for value, cached in zip(metadata, self._conditioning_metadata[0])
        ):
            # Qwen builds positions in Python. Compute them before upload to avoid
            # token/grid downloads and synchronization from CUDA.
            inputs["position_ids"], _ = self.vlm.model.get_rope_index(
                inputs.input_ids,
                image_grid_thw=inputs.image_grid_thw,
                attention_mask=inputs.attention_mask,
            )
            # SDPA uses CPU shapes/splits; FA2 needs CUDA cumulative lengths.
            image_grid = inputs.pop("image_grid_thw")
            if self.vlm.model.visual.config._attn_implementation == "flash_attention_2":
                image_grid = image_grid.to(self.device)
            inputs["attention_mask"] = inputs.attention_mask.bool()
            self._conditioning_metadata = metadata, inputs.to(self.device), image_grid
        _, inputs, image_grid = self._conditioning_metadata
        attention_mask = inputs.attention_mask
        states = states.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            output = self.vlm.model(
                input_ids=inputs.input_ids,
                position_ids=inputs.position_ids,
                attention_mask=attention_mask,
                pixel_values=pixels.to(self.device),
                image_grid_thw=image_grid,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            views = output.last_hidden_state[:, None]
        return views, states, attention_mask

    @torch.no_grad()
    def predict_action(
        self,
        observations: list[list[Image.Image]],
        states: torch.Tensor,
        instructions: list[str],
        num_inference_steps: int,
    ) -> torch.Tensor:
        views, states, attention_mask = self._conditioning(
            observations, states, instructions
        )
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            # Each denoising step starts from the same projected conditioning.
            context, context_mask = self.action_header.obs_proj(
                views, states, attention_mask
            )
            joint_mask = F.pad(context_mask, (self.horizon, 0), value=True)
            action = torch.randn(
                states.shape[0],
                self.horizon,
                self.action_dim,
                device=self.device,
            )
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            self.scheduler.set_begin_index(0)
            timesteps = cast(torch.Tensor, self.scheduler.timesteps)
            for timestep in timesteps:
                batch_timestep = timestep.expand(states.shape[0])
                prediction = self.action_header(
                    action,
                    context,
                    batch_timestep,
                    joint_mask,
                )
                step = cast(
                    FlowMatchEulerDiscreteSchedulerOutput,
                    self.scheduler.step(
                        prediction, timestep, cast(torch.FloatTensor, action)
                    ),
                )
                action = step.prev_sample
        return action.float()
