from __future__ import annotations

from typing import Any, cast

import torch
from mjlab.viewer import EnvProtocol, NativeMujocoViewer


class SimViewer(NativeMujocoViewer):
    """Passive viewer for a simulation stepped by the Ψ₀ runtime."""

    def __init__(self, env: Any) -> None:
        super().__init__(
            cast(EnvProtocol, env),
            _ViewerOnlyPolicy(),
            frame_rate=50.0,
            enable_perturbations=False,
        )
        self.setup()
        self.sync()

    def sync(self) -> None:
        self.sync_env_to_viewer()


class _ViewerOnlyPolicy:
    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        raise RuntimeError("The passive viewer cannot step the simulation")
