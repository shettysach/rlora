"""SONIC bounds normalization shared by demonstration and rollout inputs."""

import torch


def normalize_bounds(
    value: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    state: bool = False,
) -> torch.Tensor:
    span = high - low
    constant = span.abs() < 1e-4 * (high.abs() + low.abs() + 1e-8)
    normalized = (value - low) / span.masked_fill(constant, 1) * 2 - 1
    # Upstream keeps constant action channels as-is, but zeroes constant states.
    return torch.where(
        constant, torch.zeros_like(value) if state else value, normalized
    ).clamp(-1, 1)
