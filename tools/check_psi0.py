"""CUDA compatibility probe. Run from the repo root with .venv/bin/python.

Uses unchanged upstream class ASTs; excludes training/config/logging imports.
Raw comparisons are written incrementally so failures preserve measured results.
"""

import ast
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.attention_processor import Attention
from PIL import Image
from pydantic import BaseModel
from safetensors import safe_open
from torchvision.transforms import v2
from transformers.feature_extraction_utils import BatchFeature

sys.path.insert(0, str(Path("src").resolve()))
from planner._psi0 import model as local
from planner.psi0 import Psi0Planner

UPSTREAM = Path("/tmp/rlora-psi0-reference")
COMMIT = "2830f9367fc12ca49d9b4df4348726daae9db15d"
RUN = Path(
    "artifacts/psi-model/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223"
)
QWEN = Path("artifacts/qwen3-vl-2b")
OUT = Path("artifacts/psi0-checks")
OUT.mkdir(parents=True, exist_ok=True)
rows = []


def compare(name, actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    delta = actual.float() - expected.float()
    row = {
        "name": name,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "backend": "sdpa",
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "exact": torch.equal(actual, expected),
        "max_abs": delta.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
    }
    rows.append(row)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(row), flush=True)


def upstream_model():
    assert (
        subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
        == COMMIT
    )
    source = UPSTREAM / "src/psi/models/psi0.py"
    assert source.read_bytes() == subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "show", f"{COMMIT}:src/psi/models/psi0.py"]
    )
    tree = ast.parse(source.read_text())
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.ImportFrom)
            and (n.module or "").startswith("psi")
            or isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "overwatch" for t in n.targets)
        )
    ]
    env = {
        "LaunchConfig": object,
        "overwatch": logging.getLogger("reference"),
        "count_parameters": lambda m: sum(p.numel() for p in m.parameters()),
        "__name__": "reference",
    }
    exec(compile(tree, str(source), "exec"), env)  # noqa: S102 - pinned upstream source
    return SimpleNamespace(**env)


def upstream_transforms():
    env = {
        "BaseModel": BaseModel,
        "FieldTransform": BaseModel,
        "resolve_path": lambda p: p,
    }
    for file, names in [
        ("augmentation.py", {"ResizeImage", "CenterCrop"}),
        (
            "transform_psi0_sonic.py",
            {"SonicActionStateTransform", "parse_modality_key"},
        ),
    ]:
        source = UPSTREAM / "src/psi/config" / file
        tree = ast.parse(source.read_text())
        tree.body = [
            n
            for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            and not (
                isinstance(n, ast.ImportFrom)
                and (n.level or (n.module or "").startswith("psi"))
            )
            or isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in names
        ]
        exec(compile(tree, str(source), "exec"), env)  # noqa: S102 - upstream classes
    return SimpleNamespace(**env)


def attention_matrix(ref):
    for dtype in [torch.float32, torch.bfloat16]:
        for batch in [1, 2, 4]:
            for dim in [64, 768]:
                for final in [False, True]:
                    torch.manual_seed(100 + batch + dim)
                    attn = (
                        Attention(
                            query_dim=dim,
                            added_kv_proj_dim=dim,
                            heads=8,
                            dim_head=dim // 8,
                            out_dim=dim,
                            context_pre_only=final,
                            bias=True,
                        )
                        .to("cuda", dtype=dtype)
                        .eval()
                    )
                    action = torch.randn(batch, 30, dim, device="cuda", dtype=dtype)
                    context = torch.randn(batch, 65, dim, device="cuda", dtype=dtype)
                    mask = torch.ones(batch, 65, device="cuda", dtype=torch.bool)
                    mask[:, -7:] = False
                    call = (
                        lambda dim=dim, attn=attn, action=action, context=context, mask=mask: (
                            ref.JointVLAAttnProcessor(dim)(attn, action, context, mask)
                        )
                    )
                    expected = call()
                    label = f"attention/{dtype}/b{batch}/d{dim}/final{final}"
                    compare(label + "/repeat", call()[0], expected[0])
                    actual = local.JointVLAAttnProcessor()(
                        attn, action, context, F.pad(mask, (30, 0), value=True)
                    )
                    compare(label + "/action", actual[0], expected[0])
                    if not final:
                        compare(label + "/context", actual[1], expected[1])
                    else:
                        original_pad = F.pad
                        with patch.object(local.F, "pad", lambda x, *a, **k: x):
                            unpadded = local.JointVLAAttnProcessor()(
                                attn,
                                action,
                                context,
                                original_pad(mask, (30, 0), value=True),
                            )
                        compare(label + "/no-padding", unpadded[0], expected[0])


def capture(model, reference, images, state, prompts, noise):
    result = {}
    hooks = []

    def record(name):
        def hook(module, args, output):
            values = output if isinstance(output, tuple) else (output,)
            for i, value in enumerate(values):
                if torch.is_tensor(value):
                    result.setdefault(f"{name}/{i}", []).append(
                        value.detach().cpu().clone()
                    )

        return hook

    for name, module in [
        ("obs", model.action_header.obs_proj),
        ("time", model.action_header.time_ins_embed),
        ("action-in", model.action_header.action_proj_in),
        *[
            (f"block{i}", b)
            for i, b in enumerate(model.action_header.transformer_blocks)
        ],
    ]:
        hooks.append(module.register_forward_hook(record(name)))
    vlm = model.vlm_model if reference else model.vlm
    inputs_seen = []

    def vlm_inputs(module, args, kwargs):
        inputs_seen.append(
            {
                k: v.detach().cpu().clone()
                for k, v in kwargs.items()
                if torch.is_tensor(v)
            }
        )

    hooks.append(
        (vlm if reference else vlm.model).register_forward_pre_hook(
            vlm_inputs, with_kwargs=True
        )
    )

    def features(module, args, output):
        result["features"] = (
            (
                output.hidden_states[-1][:, None]
                if reference
                else output.last_hidden_state[:, None]
            )
            .detach()
            .cpu()
        )

    hooks.append((vlm if reference else vlm.model).register_forward_hook(features))
    scheduler = model.noise_scheduler if reference else model.scheduler
    original_step = scheduler.step

    def step(prediction=None, timestep=None, sample=None, **kwargs):
        velocity = kwargs.get("model_output", prediction)
        result.setdefault("velocity", []).append(velocity.detach().cpu().clone())
        output = original_step(velocity, timestep, sample)
        result.setdefault("actions", []).append(
            output.prev_sample.detach().cpu().clone()
        )
        return output

    vlm.model.rope_deltas = None

    def randn(*args, **kwargs):
        return noise.clone()

    try:
        with patch.object(torch, "randn", randn), patch.object(scheduler, "step", step):
            kwargs = {
                "observations": images,
                "states": state,
                "instructions": prompts,
                "num_inference_steps": 10,
            }
            if reference:
                kwargs["traj2ds"] = None
            result["final"] = model.predict_action(**kwargs).cpu()
    finally:
        for hook in hooks:
            hook.remove()
    result["inputs"] = inputs_seen[0]
    if reference:
        data = result["inputs"]
        result["positions"] = vlm.model.get_rope_index(
            data["input_ids"],
            image_grid_thw=data["image_grid_thw"],
            attention_mask=data["attention_mask"],
        )[0].cpu()
    else:
        result["positions"] = result["inputs"]["position_ids"]
    return result


def compare_capture(label, actual, expected):
    for key in ["input_ids", "attention_mask", "image_grid_thw", "pixel_values"]:
        a, b = actual["inputs"][key], expected["inputs"][key]
        if key == "pixel_values":
            a, b = a.bfloat16(), b.bfloat16()
        compare(label + "/" + key, a, b)
    for key in ["positions", "features", "final"]:
        compare(label + "/" + key, actual[key], expected[key])
    for key in expected:
        if isinstance(expected[key], list) and key in actual:
            for i, (a, b) in enumerate(zip(actual[key], expected[key])):
                compare(f"{label}/{key}/step{i}", a, b)


@torch.inference_mode()
def main():
    ref = upstream_model()
    attention_matrix(ref)
    saved = json.loads((RUN / "run_config.json").read_text())
    transforms = upstream_transforms()
    field = transforms.SonicActionStateTransform(**saved["data"]["transform"]["field"])
    image_cfg = saved["data"]["transform"]["model"]
    reference_transform = v2.Compose(
        [
            transforms.ResizeImage(**image_cfg["resize"])(),
            transforms.CenterCrop(**image_cfg["center_crop"])(),
        ]
    )
    planner = Psi0Planner(RUN, 40000, QWEN)
    model = planner.model
    assert model.vlm.model.visual.config._attn_implementation == "sdpa"
    assert model.vlm.model.language_model.config._attn_implementation == "sdpa"
    # Restore the norm in a separate causal wrapper with identical fine-tuned weights.
    vlm = copy.deepcopy(model.vlm).cpu()
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

    norm = Qwen3VLTextRMSNorm(
        vlm.config.text_config.hidden_size, eps=vlm.config.text_config.rms_norm_eps
    ).bfloat16()
    with safe_open(
        RUN / "checkpoints/ckpt_40000/model.safetensors", framework="pt"
    ) as weights:
        norm.weight.copy_(
            weights.get_tensor("vlm_model.model.language_model.norm.weight")
        )
    vlm.model.language_model.norm = norm
    reference = ref.Psi0Model(SimpleNamespace(**saved["model"]), vlm)
    reference.action_header.load_state_dict(
        model.action_header.state_dict(), strict=True
    )
    reference.action_header.bfloat16()
    reference.action_header.time_ins_embed.w = torch.nn.Parameter(
        model.action_header.time_ins_embed.w.cpu().clone(), requires_grad=False
    )
    reference.vlm_processor = copy.deepcopy(model.processor)
    reference.noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000
    )
    reference.action_horizon, reference.action_dim, reference.device = (
        30,
        78,
        torch.device("cuda"),
    )
    reference.eval().requires_grad_(False)
    rng = np.random.default_rng(292285)
    for batch in [1, 2, 4]:
        raw = [
            Image.fromarray(
                rng.integers(0, 256, (360 + i * 16, 480 + i * 32, 3), dtype=np.uint8)
            )
            for i in range(batch)
        ]
        state_raw = rng.uniform(
            np.array(field.state_min), np.array(field.state_max), (batch, 1, 43)
        ).astype(np.float32)
        state = planner._normalize_state(torch.from_numpy(state_raw).cuda())
        compare(
            f"b{batch}/state-normalization",
            state,
            torch.from_numpy(field.normalize_state_func(state_raw)),
        )
        for i, img in enumerate(raw):
            compare(
                f"b{batch}/image-transform/row{i}",
                torch.from_numpy(np.array(planner.image_transform(img))),
                torch.from_numpy(np.array(reference_transform(img))),
            )
        prompts = [
            "walk forward" + " and carry the box carefully" * i for i in range(batch)
        ]
        torch.manual_seed(292285 + batch)
        noise = torch.randn(batch, 30, 78, device="cuda")
        torch.save(noise.cpu(), OUT / f"noise-b{batch}.pt")
        cases = [
            ("transformed", [[reference_transform(img)] for img in raw], prompts),
            (
                "pixels-changed",
                [
                    [reference_transform(Image.fromarray(255 - np.array(img)))]
                    for img in raw
                ],
                prompts,
            ),
            ("mixed-grids", [[img] for img in raw], [p + " stop" for p in prompts]),
        ]
        for case, images, instructions in cases:
            label = f"b{batch}/{case}"
            model.cpu()
            torch.cuda.empty_cache()
            reference.cuda()
            expected = capture(reference, True, images, state, instructions, noise)
            repeated = capture(reference, True, images, state, instructions, noise)
            compare_capture(label + "/repeat", repeated, expected)
            reference.cpu()
            torch.cuda.empty_cache()
            model.cuda()
            local_images = (
                [[planner.image_transform(img)] for img in raw]
                if case == "transformed"
                else images
            )
            actual = capture(model, False, local_images, state, instructions, noise)
            compare_capture(label, actual, expected)
            if case == "transformed":
                # Isolate model execution with the exact reference processor outputs.
                inputs = {
                    k: v
                    for k, v in expected["inputs"].items()
                    if k
                    in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
                }
                with patch.object(
                    type(model.processor),
                    "__call__",
                    lambda self, inputs=inputs, **kwargs: BatchFeature(
                        {k: v.clone() for k, v in inputs.items()}
                    ),
                ):
                    isolated = capture(model, False, images, state, instructions, noise)
                compare_capture(label + "/identical-inputs", isolated, expected)
            compare(
                label + "/denormalized",
                0.5
                * (actual["final"] + 1)
                * (planner.action_max.cpu() - planner.action_min.cpu())
                + planner.action_min.cpu(),
                field.denormalize(expected["final"]),
            )
            torch.save(
                {"reference": expected, "local": actual}, OUT / f"b{batch}-{case}.pt"
            )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
