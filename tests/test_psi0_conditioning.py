import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature

from planner._psi0.model import ActionTransformerModel, Psi0Model


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
    ],
)
def test_cpu_metadata_preserves_batched_conditioning(monkeypatch, device):
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    config._attn_implementation = "sdpa"
    # Two image grids and different prompt lengths exercise right padding.
    inputs = {
        "input_ids": torch.tensor([[1, 5, 3, 6, 7, 0, 0, 0], [1, 8, 5, 3, 3, 6, 7, 9]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1] * 8]),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]),
        "pixel_values": torch.randn(12, 12),
    }

    class Processor:
        tokenizer = type("Tokenizer", (), {})()

        def apply_chat_template(self, *args, **kwargs):
            return "image instruction"

        def __call__(self, **kwargs):
            return BatchFeature({key: value.clone() for key, value in inputs.items()})

    model = Psi0Model(
        ActionTransformerModel(
            action_dim=78,
            horizon=2,
            odim=43,
            view_feature_dim=32,
            hidden_dim=32,
            num_blocks=1,
            heads=4,
        ),
        Qwen3VLForConditionalGeneration(config).bfloat16().to(device).eval(),
        Processor(),
        FlowMatchEulerDiscreteScheduler(),
        78,
        2,
        torch.device(device),
    )
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        expected = model.vlm.model(
            **{key: value.to(device) for key, value in inputs.items()},
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, None]

    calls = []
    get_positions = model.vlm.model.get_rope_index

    def cpu_positions(input_ids, **kwargs):
        assert input_ids.device.type == "cpu"
        assert kwargs["image_grid_thw"].device.type == "cpu"
        calls.append(input_ids)
        return get_positions(input_ids, **kwargs)

    monkeypatch.setattr(model.vlm.model, "get_rope_index", cpu_positions)
    forwarded = []
    model.vlm.model.register_forward_pre_hook(
        lambda module, args, kwargs: forwarded.append(
            (kwargs["image_grid_thw"].device.type, kwargs["pixel_values"].dtype)
        ),
        with_kwargs=True,
    )
    states = torch.randn(2, 1, 43, device=device)
    views, actual_states, mask = model._conditioning(
        [[Image.new("RGB", (32, 32))]] * 2, states, ["short", "longer instruction"]
    )
    assert len(calls) == 1  # The backbone receives explicit positions.
    assert forwarded == [("cpu", torch.bfloat16)]
    torch.testing.assert_close(views, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_states, states)
    torch.testing.assert_close(mask, inputs["attention_mask"].to(device).bool())

    changed_ids = inputs["input_ids"].clone()
    changed_ids[0, 4] = 8
    changed_mask = inputs["attention_mask"].clone()
    changed_mask[1, -1] = 0
    for name, value, position_calls in (
        ("pixel_values", inputs["pixel_values"] + 0.25, 1),
        ("input_ids", changed_ids, 2),
        ("attention_mask", changed_mask, 3),
        ("image_grid_thw", torch.tensor([[1, 2, 2], [1, 4, 2]]), 4),
    ):
        inputs[name] = value
        # Compare each updated batch with the original, uncached backbone call.
        with monkeypatch.context() as reference_patch:
            reference_patch.setattr(model.vlm.model, "get_rope_index", get_positions)
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                expected = model.vlm.model(
                    **{key: tensor.to(device) for key, tensor in inputs.items()},
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, None]
        views, _, mask = model._conditioning(
            [[Image.new("RGB", (32, 32))]] * 2, states, ["short", "longer instruction"]
        )
        assert len(calls) == position_calls
        assert forwarded[-1] == ("cpu", torch.bfloat16)
        torch.testing.assert_close(views, expected, rtol=0, atol=0)
        torch.testing.assert_close(mask, inputs["attention_mask"].to(device).bool())
