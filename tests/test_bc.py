"""Core compatibility checks; no upstream package or model downloads required."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from data.sonic_bones import SonicBones, convert_sample
from planner._psi0.lora import add_lora, load_adapter, save_adapter
from planner._psi0.model import ActionTransformerModel, Psi0Model
from planner._psi0.preprocessing import normalize_bounds
from training.bc import train


def small_header():
    return ActionTransformerModel(
        action_dim=80,
        horizon=3,
        odim=45,
        view_feature_dim=16,
        hidden_dim=32,
        num_blocks=2,
        heads=4,
        qk_norm="rms_norm",
        pooled_projection_dim=8,
    ).eval()


def reference():
    return json.loads(
        (Path(__file__).parent / "fixtures/psi0_action_expert.json").read_text()
    )


def reference_header():
    header = small_header()
    weights = {}
    for name, shape in reference()["keys"].items():
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        weights[name] = (
            torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.1
        )
    header.load_state_dict(weights, strict=True)
    return header


def header_inputs():
    values = reference()["inputs"]
    return tuple(
        torch.tensor(values[key])
        for key in ("actions", "views", "states", "timestep", "pooled", "mask")
    )


def small_policy():
    header = reference_header()
    _, views, states, _, pooled, mask = header_inputs()
    vlm = nn.Linear(1, 1).eval().requires_grad_(False)
    policy = Psi0Model(
        header,
        vlm,
        None,
        FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        [0, 1],
        80,
        3,
        torch.device("cpu"),
    )
    policy._conditioning = Mock(return_value=(views, states, mask))
    actions = torch.randn(2, 3, 80)
    actions_mask = torch.ones_like(actions)
    actions_mask[..., 78:] = 0
    batch = {
        "observations": [],
        "states": states,
        "instructions": ["walk"] * 2,
        "actions": actions,
        "actions_mask": actions_mask,
        "pooled_projections": pooled,
    }
    return policy, batch


def test_converter_matches_upstream_example():
    # Upstream raw_sonic_to_psi_lerobot.py at the pinned commit: pack_psi_state,
    # source_full_hand14 and build_act(end_effector="dex3") on index-coded data.
    source = np.arange(43, dtype=np.float32)
    token = np.arange(64, dtype=np.float32) / 16 - 0.625
    state, action, mask = convert_sample(source, source + 100, token)
    expected_hands = [26, 27, 28, 22, 23, 24, 25, 40, 41, 42, 36, 37, 38, 39]
    assert state.tolist() == [*range(22), *range(29, 36), *expected_hands, 0, 0]
    torch.testing.assert_close(action[:64], torch.from_numpy(token))
    assert action[64:78].tolist() == [value + 100 for value in expected_hands]
    assert action[78:].tolist() == [0, 0]
    assert mask.tolist() == [1] * 78 + [0] * 2
    with pytest.raises(ValueError, match="finite"):
        convert_sample(source, source, np.full(64, np.nan))


def test_normalization_constant_dimensions_match_upstream():
    low = torch.tensor([-2.0, 1.0, 0.0])
    high = torch.tensor([2.0, 1.0, 0.0])
    values = torch.tensor([0.0, 0.4, 0.0])
    torch.testing.assert_close(
        normalize_bounds(values, low, high), torch.tensor([0.0, 0.4, 0.0])
    )
    torch.testing.assert_close(
        normalize_bounds(values, low, high, state=True), torch.zeros(3)
    )


def test_action_expert_matches_saved_upstream_output():
    with torch.no_grad():
        output = reference_header()(*header_inputs())
    torch.testing.assert_close(
        output, torch.tensor(reference()["output"]), rtol=1e-5, atol=1e-6
    )


def test_flow_loss_mask_reduction_and_adapter_gradients():
    policy, batch = small_policy()
    add_lora(policy.action_header, rank=2)
    torch.manual_seed(42)
    sigma = torch.rand(2)
    noise = torch.randn_like(batch["actions"])
    noisy = (1 - sigma[:, None, None]) * batch["actions"] + sigma[:, None, None] * noise
    _, views, states, _, pooled, mask = header_inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = policy.action_header(
            noisy, views, states, sigma * 1000, pooled, mask
        )
    expected = (
        (
            (prediction.float() - noise + batch["actions"]).square()
            * batch["actions_mask"]
        )
        .sum(1)
        .mean()
    )
    torch.manual_seed(42)
    loss = policy.bc_loss(batch)
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)
    loss.backward()
    parameters = dict(policy.named_parameters())
    adapters = {
        name: value for name, value in parameters.items() if value.requires_grad
    }
    assert adapters and all("lora_" in name for name in adapters)
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in adapters.values()
    )
    assert any(
        value.grad.abs().sum() > 0
        for name, value in adapters.items()
        if name.endswith("lora_b")
    )
    assert all(
        value.grad is None for value in parameters.values() if not value.requires_grad
    )


def test_adapter_starts_identically_and_reloads(tmp_path):
    header = reference_header()
    inputs = header_inputs()
    expected = header(*inputs)
    add_lora(header, rank=2)
    torch.testing.assert_close(header(*inputs), expected, rtol=0, atol=0)
    with torch.no_grad():
        for name, parameter in header.named_parameters():
            if name.endswith("lora_b"):
                parameter.fill_(0.05)
    updated = header(*inputs)
    assert not torch.allclose(updated, expected)
    save_adapter(header, tmp_path / "adapter.safetensors")
    restored = reference_header()
    load_adapter(restored, tmp_path / "adapter.safetensors", rank=2)
    torch.testing.assert_close(restored(*inputs), updated, rtol=0, atol=0)


def test_sampling_is_finite_and_matches_euler_flow():
    policy, batch = small_policy()
    torch.manual_seed(123)
    action = torch.randn(2, 3, 80)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(3, device="cpu")
    _, views, states, _, pooled, mask = header_inputs()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        for timestep in scheduler.timesteps:
            velocity = policy.action_header(
                action, views, states, timestep.expand(2), pooled, mask
            )
            action = scheduler.step(velocity, timestep, action).prev_sample
    torch.manual_seed(123)
    result = policy.predict_action(
        [], batch["states"], batch["instructions"], 3, pooled
    )
    assert result.shape == (2, 3, 80) and torch.isfinite(result).all()
    torch.testing.assert_close(result, action.float(), rtol=0, atol=0)


def test_checkpoint_loader_is_strict(tmp_path, monkeypatch):
    import planner._psi0.model as module

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [0],
        },
    )
    vlm = Qwen3VLForConditionalGeneration(config).bfloat16()
    header = reference_header().bfloat16()
    weights = {
        f"vlm_model.{key}": value.clone()
        for key, value in vlm.state_dict().items()
        if key != "lm_head.weight"
    }
    weights.update(
        {f"action_header.{key}": value for key, value in header.state_dict().items()}
    )
    checkpoint = tmp_path / "checkpoints/ckpt_1/model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    save_file(weights, str(checkpoint))
    monkeypatch.setattr(
        module.AutoConfig, "from_pretrained", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(
        module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: None
    )
    model_config = {
        "action_dim": 80,
        "action_chunk_size": 3,
        "odim": 45,
        "view_feature_dim": 16,
        "hidden_dim": 32,
        "num_blocks": 2,
        "nhead": 4,
        "qk_norm": "rms_norm",
        "pooled_projection_dim": 8,
        "train_diffusion_steps": 1000,
        "vlm_layer_indices": [0, 1],
    }
    loaded = Psi0Model.from_pretrained(
        tmp_path, 1, model_config, tmp_path, torch.device("cpu")
    )
    for name, expected in header.state_dict().items():
        torch.testing.assert_close(loaded.action_header.state_dict()[name], expected)
    del weights["action_header.action_proj_out.linear.weight"]
    save_file(weights, str(checkpoint))
    with pytest.raises(RuntimeError, match="Missing key"):
        Psi0Model.from_pretrained(
            tmp_path, 1, model_config, tmp_path, torch.device("cpu")
        )


def test_chunks_keep_episode_boundaries_and_resample_physical_time(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"fps": 50}))
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 7, "task": "walk and turn around"})
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {"episode_index": 47, "length": 5, "tasks": ["walk and turn around"]}
        )
    )
    data = SonicBones(tmp_path, [47], task="walk and turn around", horizon=3)
    assert data.frames[47].tolist() == [0, 2, 3]
    states = torch.zeros(3, 45)
    actions = torch.arange(3).float()[:, None].expand(3, 80)
    data._episode = lambda episode: (
        states,
        actions,
        torch.ones_like(actions),
        [None] * 3,
    )
    assert data[1]["actions"][:, 0].tolist() == [1, 2, 2]
    assert data[2]["actions"][:, 0].tolist() == [2, 2, 2]


def test_trainer_updates_only_adapters_and_saves_metrics(tmp_path):
    policy, batch = small_policy()
    add_lora(policy.action_header, rank=2)
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}
    config = {
        "seed": 0,
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "max_steps": 2,
            "gradient_accumulation": 2,
            "max_grad_norm": 1,
            "eval_every": 1,
        },
    }
    output = tmp_path / "run"
    train(policy, lambda: [batch], lambda: [batch], config=config, output=output)
    records = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]
    assert all(np.isfinite(record["validation_flow_loss"]) for record in records)
    assert (output / "ckpt_2/adapter.safetensors").is_file()
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in policy.named_parameters()
        if not parameter.requires_grad
    )
