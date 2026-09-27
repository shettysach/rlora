"""Single-device BC over prepared policy batches."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from planner._psi0.lora import save_adapter


@torch.no_grad()
def validate(policy, batches, seed: int = 0) -> float:
    # Fixed flow noise/time makes validation checkpoints comparable, without
    # consuming the training RNG. The VLM stays frozen and in eval mode.
    devices = [policy.device] if policy.device.type == "cuda" else []
    policy.action_header.eval()
    total, count = 0.0, 0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in batches():
            loss = policy.bc_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation flow loss")
            size = batch["actions"].shape[0]
            total += loss.item() * size
            count += size
    return total / count


def train(policy, batches, validation_batches, *, config: dict, output: Path) -> None:
    settings = config["training"]
    parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    output.mkdir(parents=True, exist_ok=False)
    config_text = json.dumps(config, indent=2) + "\n"
    (output / "bc_config.json").write_text(config_text)
    iterator = iter(batches())
    with (output / "metrics.jsonl").open("w") as metrics:
        for step in range(1, settings["max_steps"] + 1):
            policy.action_header.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(settings["gradient_accumulation"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(batches())
                    batch = next(iterator)
                loss = policy.bc_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training flow loss at step {step}"
                    )
                (loss / settings["gradient_accumulation"]).backward()
                total_loss += loss.detach().item() / settings["gradient_accumulation"]
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, settings["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            record = {
                "step": step,
                "train_flow_loss": total_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": norm.item(),
            }
            if step % settings["eval_every"] == 0 or step == settings["max_steps"]:
                if validation_batches is not None:
                    record["validation_flow_loss"] = validate(
                        policy, validation_batches, config["seed"]
                    )
                checkpoint = output / f"ckpt_{step}"
                checkpoint.mkdir()
                save_adapter(policy.action_header, checkpoint / "adapter.safetensors")
                (checkpoint / "bc_config.json").write_text(config_text)
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
    policy.action_header.eval()
