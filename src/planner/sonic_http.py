"""Small client for the published Psi0 SONIC evaluation server."""

from __future__ import annotations

import base64
import json
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np


def _packed(array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array)
    return {
        "__numpy__": base64.b64encode(array.tobytes()).decode("ascii"),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _unpacked(value: dict) -> np.ndarray:
    return np.frombuffer(
        base64.b64decode(value["__numpy__"]), np.dtype(value["dtype"])
    ).reshape(value["shape"])


class SonicHttpPlanner:
    def __init__(self, url: str, batch_size: int):
        self.url = url.rstrip("/")
        with urllib.request.urlopen(self.url + "/info", timeout=30) as response:
            info = json.load(response)["action"]
        if info["action_dim"] != 78 or info["action_chunk_size"] != 30:
            raise ValueError(f"Psi0 server must return [30,78], got {info}")
        self.horizon = 30
        self.exec_horizon = info["action_exec_horizon"]
        self.client_ids = [str(uuid.uuid4()) for _ in range(batch_size)]
        self.reset_pending = [True] * batch_size
        self.executor = ThreadPoolExecutor(max_workers=batch_size)

    def reset(self) -> None:
        self.reset_pending = [True] * len(self.client_ids)

    def predict(
        self, images: np.ndarray, states: np.ndarray, instructions: list[str]
    ) -> np.ndarray:
        actions = list(
            self.executor.map(
                self._infer,
                range(len(self.client_ids)),
                images,
                states,
                instructions,
            )
        )
        return np.stack(actions)

    def _infer(
        self, index: int, image: np.ndarray, state: np.ndarray, instruction: str
    ) -> np.ndarray:
        history = {"client_id": self.client_ids[index]}
        if self.reset_pending[index]:
            history["reset"] = True
            self.reset_pending[index] = False
        request = {
            "image": {"observation.images.egocentric": _packed(image.astype(np.uint8))},
            "instruction": instruction,
            "history": history,
            "state": {"states": _packed(state.astype(np.float32))},
            "condition": {},
            "gt_action": [],
            "dataset_name": "simple",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        payload = json.dumps(request).encode()
        with urllib.request.urlopen(
            urllib.request.Request(
                self.url + "/act",
                data=payload,
                headers={"Content-Type": "application/json"},
            ),
            timeout=300,
        ) as response:
            action = _unpacked(json.load(response)["action"]).astype(np.float32)
        if action.shape != (30, 78) or not np.isfinite(action).all():
            raise ValueError(f"Psi0 server returned invalid action {action.shape}")
        return action

    def close(self) -> None:
        self.executor.shutdown(wait=True)
