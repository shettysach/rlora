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


def test_test_time_rtc_guides_only_the_overlapping_actions(monkeypatch):
    class Head(torch.nn.Module):
        def obs_proj(self, views, states, mask):
            return views[:, 0], mask

        def forward(self, action, context, timestep, mask):
            return action * 0.25

    vlm = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(norm=None))
    )
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    model = Psi0Model(
        Head(),
        vlm,
        processor,
        FlowMatchEulerDiscreteScheduler(),
        2,
        30,
        torch.device("cpu"),
    )
    states = torch.zeros(2, 1, 1)
    views = torch.zeros(2, 1, 2, 4)
    mask = torch.ones(2, 2, dtype=torch.bool)
    monkeypatch.setattr(model, "_conditioning", lambda *args: (views, states, mask))
    noise = torch.zeros(2, 30, 2)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: noise.clone())

    plain = model.predict_action([], states, [], 4)
    previous = torch.zeros_like(noise)
    previous[0, :6] = 1
    previous[1, :6] = -1
    guided = model.predict_action(
        [], states, [], 4, prev_actions=previous, execution_horizon=24
    )

    assert (guided[0, :6] > plain[0, :6]).all()
    assert (guided[1, :6] < plain[1, :6]).all()
    torch.testing.assert_close(guided[:, 6:], plain[:, 6:], rtol=0, atol=0)


def test_planner_rtc_keeps_normalized_unexecuted_tail(tmp_path, monkeypatch):
    config = {
        "model": {"action_exec_horizon": 30},
        "data": {
            "transform": {
                "field": {
                    "state_min": [0],
                    "state_max": [1],
                    "action_min": [-2],
                    "action_max": [2],
                },
                "model": {"resize": {"size": [2, 2]}, "center_crop": {"size": [2, 2]}},
            }
        },
    }
    (tmp_path / "run_config.json").write_text(json.dumps(config))

    class Model:
        def __init__(self):
            self.previous = []

        def predict_action(self, **kwargs):
            self.previous.append(kwargs["prev_actions"])
            return torch.arange(30, dtype=torch.float32).view(1, 30, 1) / 30

    model = Model()
    monkeypatch.setattr(Psi0Model, "from_pretrained", lambda *args: model)
    planner = Psi0Planner(tmp_path, 40000, tmp_path, device="cpu", rtc=True)
    image = torch.zeros(1, 2, 2, 3, dtype=torch.uint8)
    state = torch.zeros(1, 1)
    planner.predict(image, state, ["carry box"])
    planner.predict(image, state, ["carry box"])

    assert planner.exec_horizon == 24
    assert model.previous[0] is None
    expected = torch.cat((torch.arange(24, 30) / 30, torch.zeros(24)))
    torch.testing.assert_close(model.previous[1][0, :, 0], expected)
