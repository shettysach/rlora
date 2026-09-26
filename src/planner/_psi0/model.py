"""Inference-only Psi-0 model used by the planner."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)
from PIL import Image
from qwen_vl_utils import process_vision_info
from safetensors import safe_open
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.utils import is_flash_attn_2_available

QWEN3VL_VARIANT = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3VL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


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


class JointVLAAttnProcessor(AttnProcessor):
    def __call__(
        self,
        attention: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        head_dim = attention.inner_dim // attention.heads

        to_k = cast(nn.Linear, attention.to_k)
        to_v = cast(nn.Linear, attention.to_v)
        norm_q = cast(nn.Module, attention.norm_q)
        norm_k = cast(nn.Module, attention.norm_k)
        add_k_proj = cast(nn.Linear, attention.add_k_proj)
        add_v_proj = cast(nn.Linear, attention.add_v_proj)
        norm_added_k = cast(nn.Module, attention.norm_added_k)
        to_out = cast(nn.ModuleList, attention.to_out)

        query = attention.to_q(hidden_states)
        key = to_k(hidden_states)
        value = to_v(hidden_states)
        query = query.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        query = norm_q(query)
        key = norm_k(key)

        context_key = add_k_proj(encoder_hidden_states)
        context_value = add_v_proj(encoder_hidden_states)
        context_key = context_key.view(
            batch_size, -1, attention.heads, head_dim
        ).transpose(1, 2)
        context_value = context_value.view(
            batch_size, -1, attention.heads, head_dim
        ).transpose(1, 2)
        context_key = norm_added_k(context_key)

        action_length = hidden_states.shape[1]
        key = torch.cat((key, context_key), dim=2)
        value = torch.cat((value, context_value), dim=2)
        action_mask = torch.ones(
            batch_size,
            action_length,
            dtype=torch.bool,
            device=attention_mask.device,
        )
        joint_mask = torch.cat((action_mask, attention_mask.bool()), dim=1)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=joint_mask[:, None, None]
        )
        output = output.transpose(1, 2).reshape(batch_size, -1, attention.inner_dim)
        output = to_out[1](to_out[0](output))
        return output


class ObservationProjection(nn.Module):
    def __init__(self, odim: int, view_feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.enc_pos = PositionalEncoding(hidden_dim)
        self.views_proj = nn.Linear(view_feature_dim, hidden_dim)
        self._obs_proc = nn.Sequential(nn.Dropout(0), nn.Linear(odim, hidden_dim))

    def forward(
        self, views: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.views_proj(views), attention_mask.float()


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
    def __init__(self, hidden_dim: int, heads: int, qk_norm: str) -> None:
        super().__init__()
        self.norm1_act = AdaLayerNormZero(hidden_dim)
        self.norm1_obs = AdaLayerNormContinuous(hidden_dim)
        self.attn = Attention(
            query_dim=hidden_dim,
            added_kv_proj_dim=hidden_dim,
            dim_head=hidden_dim // heads,
            heads=heads,
            out_dim=hidden_dim,
            context_pre_only=True,
            bias=True,
            qk_norm=qk_norm,
            eps=1e-6,
            processor=JointVLAAttnProcessor(),
        )
        self.norm2_act = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ff_act = FeedForward(
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
    ) -> torch.Tensor:
        normalized, gate_attention, shift_mlp, scale_mlp, gate_mlp = self.norm1_act(
            action, condition
        )
        normalized_context = self.norm1_obs(context, condition)
        attention_output = self.attn(
            normalized,
            encoder_hidden_states=normalized_context,
            attention_mask=attention_mask,
        )
        action = action + gate_attention[:, None] * attention_output
        normalized = self.norm2_act(action) * (1 + scale_mlp[:, None])
        normalized = normalized + shift_mlp[:, None]
        return action + gate_mlp[:, None] * self.ff_act(normalized)


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
        qk_norm: str,
        pooled_projection_dim: int,
    ) -> None:
        super().__init__()
        self.time_ins_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_dim,
            pooled_projection_dim=pooled_projection_dim,
        )
        self.obs_proj = ObservationProjection(odim, view_feature_dim, hidden_dim)
        self.state_pos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.state_null = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_proj_in = ActionProjectionIn(action_dim, hidden_dim, horizon)
        self.transformer_blocks = nn.ModuleList(
            VLATransformerBlock(hidden_dim, heads, qk_norm) for _ in range(num_blocks)
        )
        self.action_proj_out = ActionProjectionOut(hidden_dim, action_dim)

    def forward(
        self,
        action: torch.Tensor,
        views: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.time_ins_embed(timestep, pooled_projection)
        action_tokens = self.action_proj_in(action)
        state_token = self.obs_proj._obs_proc[1](state[:, -1])[:, None]
        state_token = state_token + self.state_pos
        action_tokens = torch.cat((state_token, action_tokens), dim=1)
        contexts, attention_mask = self.obs_proj(views, attention_mask)
        for block, context in zip(self.transformer_blocks, contexts.unbind(1)):
            action_tokens = block(action_tokens, context, condition, attention_mask)
        return self.action_proj_out(action_tokens[:, 1:], condition)


class Psi0Model(nn.Module):
    def __init__(
        self,
        action_header: ActionTransformerModel,
        vlm: Qwen3VLForConditionalGeneration,
        processor,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vlm_layer_indices: list[int],
        action_dim: int,
        horizon: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.action_header = action_header
        self.vlm = vlm
        self.processor = processor
        self.scheduler = scheduler
        self.vlm_layer_indices = vlm_layer_indices
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        run_dir: Path,
        checkpoint_step: int,
        model_config: dict,
        device: torch.device,
    ) -> Psi0Model:
        checkpoint = (
            run_dir / "checkpoints" / f"ckpt_{checkpoint_step}" / "model.safetensors"
        )

        vlm_config = AutoConfig.from_pretrained(
            QWEN3VL_VARIANT, revision=QWEN3VL_REVISION
        )
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
        vlm.eval()

        action_header = ActionTransformerModel(
            action_dim=model_config["action_dim"],
            horizon=model_config["action_chunk_size"],
            odim=model_config["odim"],
            view_feature_dim=model_config["view_feature_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_blocks=model_config["num_blocks"],
            heads=model_config["nhead"],
            qk_norm=model_config["qk_norm"],
            pooled_projection_dim=model_config["pooled_projection_dim"],
        )
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            action_state = {
                key.removeprefix("action_header."): weights.get_tensor(key)
                for key in weights.keys()  # noqa: SIM118 - safe_open is not iterable
                if key.startswith("action_header.")
            }
        action_header.load_state_dict(action_state, strict=True)
        del action_state
        action_header.to(device, dtype=torch.bfloat16).eval()

        processor = AutoProcessor.from_pretrained(
            QWEN3VL_VARIANT, revision=QWEN3VL_REVISION
        )
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=model_config["train_diffusion_steps"]
        )
        return cls(
            action_header,
            vlm,
            processor,
            scheduler,
            model_config["vlm_layer_indices"],
            model_config["action_dim"],
            model_config["action_chunk_size"],
            device,
        )

    def _collate_text(
        self, input_ids: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.processor.tokenizer.pad_token_id
        lengths = torch.tensor([tokens.numel() for tokens in input_ids])
        padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        mask = torch.arange(padded.shape[1])[None] < lengths[:, None]
        return padded.to(self.device), mask.to(self.device)

    @torch.inference_mode()
    def predict_action(
        self,
        observations: list[list[Image.Image]],
        states: torch.Tensor,
        instructions: list[str],
        num_inference_steps: int,
        pooled_projections: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = []
        pixel_values = []
        image_grids = []
        for images, instruction in zip(observations, instructions):
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": instruction})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = cast(
                tuple[object, object],
                process_vision_info([messages], image_patch_size=16),
            )
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            input_ids.append(inputs.input_ids[0])
            pixel_values.append(inputs.pixel_values)
            image_grids.append(inputs.image_grid_thw)

        token_ids, attention_mask = self._collate_text(input_ids)
        pixels = torch.cat(pixel_values).to(self.device)
        grids = torch.cat(image_grids).to(self.device)
        states = states.to(self.device)
        pooled_projections = pooled_projections.to(self.device)

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            output = self.vlm(
                input_ids=token_ids,
                attention_mask=attention_mask,
                pixel_values=pixels,
                image_grid_thw=grids,
                output_hidden_states=True,
                return_dict=True,
            )
            views = torch.stack(
                [output.hidden_states[index] for index in self.vlm_layer_indices], dim=1
            )
            action = torch.randn(
                states.shape[0],
                self.horizon,
                self.action_dim,
                device=self.device,
            )
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = cast(torch.Tensor, self.scheduler.timesteps)
            for timestep in timesteps:
                batch_timestep = timestep.expand(states.shape[0])
                prediction = self.action_header(
                    action,
                    views,
                    states,
                    batch_timestep,
                    pooled_projections,
                    attention_mask,
                )
                step = cast(
                    FlowMatchEulerDiscreteSchedulerOutput,
                    self.scheduler.step(
                        prediction, timestep, cast(torch.FloatTensor, action)
                    ),
                )
                action = step.prev_sample
        return action.float()
