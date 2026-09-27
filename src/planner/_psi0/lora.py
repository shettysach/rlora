"""Small action-expert LoRA; base checkpoint names remain unchanged."""

import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, rank: int):
        # Reuse the base parameters without allocating a second large matrix.
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base = F.linear(input, self.weight, self.bias)
        # alpha == rank, so the adapter scale is one. Keep optimizer weights fp32.
        with torch.autocast(input.device.type, enabled=False):
            delta = F.linear(F.linear(input.float(), self.lora_a), self.lora_b)
        return base + delta.to(base.dtype)


def add_lora(action_header: nn.Module, rank: int) -> None:
    """Adapt attention projections only; every base parameter stays frozen."""
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    action_header.requires_grad_(False)
    for name, module in list(action_header.named_modules()):
        # The layerwise expert discards context-query outputs; add_q_proj is
        # retained for checkpoint compatibility and has no effect on actions.
        if (
            isinstance(module, nn.Linear)
            and ".attn." in name
            and not name.endswith("add_q_proj")
        ):
            parent_name, _, child = name.rpartition(".")
            setattr(
                action_header.get_submodule(parent_name),
                child,
                LoRALinear(module, rank),
            )


def adapter_parameters(action_header: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter
        for name, parameter in action_header.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }


def save_adapter(action_header: nn.Module, path: Path) -> None:
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in adapter_parameters(action_header).items()
        },
        str(path),
    )


def load_adapter(action_header: nn.Module, path: Path, rank: int) -> None:
    add_lora(action_header, rank)
    weights = load_file(str(path))
    parameters = adapter_parameters(action_header)
    if weights.keys() != parameters.keys():
        raise ValueError("Adapter parameter names do not match the action expert")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if weights[name].shape != parameter.shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
            parameter.copy_(weights[name])
    action_header.eval().requires_grad_(False)
