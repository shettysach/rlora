import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from planner._psi0.model import Psi0Model
from planner.psi0 import Psi0Planner


def test_planner_resize_preserves_upstream_nearest_pixels(tmp_path, monkeypatch):
    config = {
        "model": {"action_exec_horizon": 30},
        "data": {
            "transform": {
                "field": {
                    key: [0]
                    for key in ("state_min", "state_max", "action_min", "action_max")
                },
                "model": {"resize": {"size": [2, 2]}, "center_crop": {"size": [2, 2]}},
            }
        },
    }
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(Psi0Model, "from_pretrained", lambda *args: None)
    planner = Psi0Planner(tmp_path, 40000, tmp_path, device="cpu")
    pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    actual = np.array(planner.image_transform(Image.fromarray(pixels)))
    np.testing.assert_array_equal(actual, pixels[1::2, 1::2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_plain_sampling_matches_upstream_cpu_sigma_arithmetic(monkeypatch):
    class Head(torch.nn.Module):
        def obs_proj(self, views, states, mask):
            return views[:, 0], mask

        def forward(self, action, context, timestep, mask):
            return (action * 0.37 + timestep[:, None, None] * 0.0001).bfloat16()

    vlm = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(norm=None))
    )
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    model = Psi0Model(
        Head(),
        vlm,
        processor,
        FlowMatchEulerDiscreteScheduler(),
        78,
        30,
        torch.device("cuda"),
    )
    states = torch.zeros(1, 1, 43, device="cuda")
    context = torch.zeros(1, 1, 2, 8, device="cuda")
    mask = torch.ones(1, 2, device="cuda", dtype=torch.bool)
    monkeypatch.setattr(model, "_conditioning", lambda *args: (context, states, mask))
    noise = torch.linspace(-2, 2, 30 * 78, device="cuda").reshape(1, 30, 78)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: noise.clone())
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(10)
    expected = noise.clone()
    for timestep in scheduler.timesteps:
        velocity = model.action_header(
            expected, context[:, 0], timestep.expand(1).cuda(), mask
        )
        expected = scheduler.step(velocity, timestep, expected).prev_sample
    actual = model.predict_action([], states, [], 10)
    torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
