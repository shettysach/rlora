"""BONES experiment entry point; the trainer consumes prepared policy batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.sonic_bones import SonicBones
from planner._psi0.lora import add_lora
from planner.psi0 import Psi0Planner
from training.bc import train, validate


@torch.no_grad()
def check_baseline(planner: Psi0Planner, batch: dict) -> tuple[dict, torch.Tensor]:
    """Exercise the strict checkpoint loader's policy before enabling adapters."""
    actions = planner.model.predict_action(
        batch["observations"],
        batch["states"],
        batch["instructions"],
        planner.inference_steps,
        batch["pooled_projections"],
    )
    expected = (batch["states"].shape[0], planner.horizon, planner.model.action_dim)
    if actions.shape != expected or not torch.isfinite(actions).all():
        raise ValueError(f"Invalid baseline action chunk: {tuple(actions.shape)}")
    tokens = (actions[..., :64] + 1) * 0.5 * (
        planner.action_max[:64] - planner.action_min[:64]
    ) + planner.action_min[:64]
    metrics = {
        "shape": list(actions.shape),
        "normalized_min": actions.min().item(),
        "normalized_max": actions.max().item(),
        "normalized_outside_unit_fraction": (actions.abs() > 1).float().mean().item(),
        "sonic_token_min": tokens.min().item(),
        "sonic_token_max": tokens.max().item(),
        "flow_loss": planner.model.bc_loss(batch).item(),
    }
    return metrics, actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Ψ₀ BONES flow-matching BC")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bc_bones_walk.json")
    )
    parser.add_argument("--output", type=Path, default=Path("results/bc-bones-overfit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        help="Evaluate a saved adapter; requires --baseline-only",
    )
    args = parser.parse_args()
    if args.bc_checkpoint is not None and not args.baseline_only:
        parser.error("--bc-checkpoint is an evaluation option; use --baseline-only")
    config = json.loads(args.config.read_text())
    dataset_config, model_config, settings = (
        config[key] for key in ("dataset", "model", "training")
    )
    if set(dataset_config["train_episodes"]) & set(dataset_config["val_episodes"]):
        parser.error("Training and validation episodes must be disjoint")
    for key in ("batch_size", "max_steps", "gradient_accumulation", "eval_every"):
        if settings[key] < 1:
            parser.error(f"training.{key} must be positive")
    if dataset_config["action_hz"] <= 0 or settings["learning_rate"] <= 0:
        parser.error("Action rate and learning rate must be positive")
    if args.output.exists() and not args.baseline_only:
        parser.error(f"Output directory already exists: {args.output}")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    # Record absolute artifact paths so saved adapters identify their base exactly.
    for key in ("run_dir", "qwen_model", "clip_model"):
        model_config[key] = str(Path(model_config[key]).resolve())
    dataset_config["root"] = str(Path(dataset_config["root"]).resolve())
    planner = Psi0Planner(
        Path(model_config["run_dir"]),
        model_config["ckpt_step"],
        Path(model_config["qwen_model"]),
        Path(model_config["clip_model"]),
        device=args.device,
        inference_steps=model_config["inference_steps"],
        bc_checkpoint=args.bc_checkpoint,
    )
    saved = json.loads((Path(model_config["run_dir"]) / "run_config.json").read_text())
    model_config["base_config_sha256"] = hashlib.sha256(
        (Path(model_config["run_dir"]) / "run_config.json").read_bytes()
    ).hexdigest()
    if saved["model"]["noise_scheduler"] != "flow" or saved["model"]["rtc"]:
        raise ValueError(
            "BC currently supports the released non-RTC flow SONIC checkpoint"
        )

    def dataset(episodes):
        return SonicBones(
            Path(dataset_config["root"]),
            episodes,
            task=dataset_config["task"],
            horizon=planner.horizon,
            state_dim=len(planner.state_min),
            action_dim=planner.model.action_dim,
            action_hz=dataset_config["action_hz"],
        )

    training_data = dataset(dataset_config["train_episodes"])
    validation_data = (
        dataset(dataset_config["val_episodes"])
        if dataset_config["val_episodes"]
        else None
    )

    data_summary = []
    for split, data in (("train", training_data), ("validation", validation_data)):
        if data is None:
            continue
        for episode in data.episodes:
            _, actions, _, _ = data._episode(episode)
            low, high = planner.action_min.cpu(), planner.action_max.cpu()
            outside = (actions < low) | (actions > high)
            data_summary.append(
                {
                    "split": split,
                    "episode": episode,
                    "observations": len(actions),
                    "token_min": actions[:, :64].min().item(),
                    "token_max": actions[:, :64].max().item(),
                    "hands_min": actions[:, 64:78].min().item(),
                    "hands_max": actions[:, 64:78].max().item(),
                    "token_bounds_clipped_fraction": outside[:, :64]
                    .float()
                    .mean()
                    .item(),
                    "hand_bounds_clipped_fraction": outside[:, 64:78]
                    .float()
                    .mean()
                    .item(),
                }
            )
    config["data_summary"] = data_summary
    print(json.dumps({"data_summary": data_summary}, indent=2), flush=True)

    def batches(data, shuffle=False):
        loader = DataLoader(
            data, batch_size=settings["batch_size"], shuffle=shuffle, collate_fn=list
        )
        return lambda: (planner.prepare_batch(samples) for samples in loader)

    training_batches = batches(training_data, shuffle=True)
    validation_batches = (
        batches(validation_data) if validation_data is not None else None
    )
    evaluation_data = (
        validation_data
        if args.baseline_only and validation_data is not None
        else training_data
    )
    baseline_batch = planner.prepare_batch([evaluation_data[0]])
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(planner.device)
        if planner.device.type == "cuda"
        else None
    )
    baseline, sampled_actions = check_baseline(planner, baseline_batch)
    if validation_batches is not None:
        baseline["validation_flow_loss"] = validate(
            planner.model, validation_batches, config["seed"]
        )
    if not all(
        np.isfinite(value) for value in baseline.values() if isinstance(value, float)
    ):
        raise FloatingPointError("Non-finite baseline metrics")
    print(json.dumps({"baseline": baseline}, indent=2), flush=True)
    if args.baseline_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "evaluation.json").write_text(
            json.dumps(baseline, indent=2) + "\n"
        )
        baseline_batch["observations"][0][0].save(args.output / "observation.png")
        (args.output / "instructions.json").write_text(
            json.dumps(baseline_batch["instructions"]) + "\n"
        )
        torch.save(
            {
                "states": baseline_batch["states"].cpu(),
                "pooled_projections": baseline_batch["pooled_projections"].cpu(),
                "actions": sampled_actions.cpu(),
                "rng_state": rng_state,
                "cuda_rng_state": cuda_rng_state,
            },
            args.output / "reference.pt",
        )
        return
    add_lora(planner.model.action_header, model_config["lora_rank"])
    config["baseline"] = baseline
    train(
        planner.model,
        training_batches,
        validation_batches,
        config=config,
        output=args.output,
    )


if __name__ == "__main__":
    main()
