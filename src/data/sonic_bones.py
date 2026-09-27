"""BONES LeRobot v2.1 reader and the upstream SONIC → Ψ₀ mapping."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

VIDEO_KEY = "observation.images.ego_view"
BODY_INDICES = [*range(22), *range(29, 36)]
HAND_INDICES = [26, 27, 28, 22, 23, 24, 25, 40, 41, 42, 36, 37, 38, 39]


def convert_sample(
    state: np.ndarray,
    wbc: np.ndarray,
    token: np.ndarray,
    state_dim: int = 45,
    action_dim: int = 80,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve 29 body + 14 hand states, and 64 token + 14 hand targets."""
    state, wbc, token = (
        np.asarray(value, dtype=np.float32) for value in (state, wbc, token)
    )
    for name, value, width in (
        ("state", state, 43),
        ("action.wbc", wbc, 43),
        ("token", token, 64),
    ):
        if value.shape[-1] != width or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite {width}-D vectors")
    if state_dim < 43 or action_dim < 78:
        raise ValueError("Checkpoint is too small for BONES state/action dimensions")
    packed_state = np.concatenate(
        (state[..., BODY_INDICES], state[..., HAND_INDICES]), axis=-1
    )
    packed_action = np.concatenate((token, wbc[..., HAND_INDICES]), axis=-1)
    # The final two checkpoint channels represent neck joints absent from BONES.
    states = np.pad(packed_state, [(0, 0)] * (state.ndim - 1) + [(0, state_dim - 43)])
    actions = np.pad(
        packed_action, [(0, 0)] * (token.ndim - 1) + [(0, action_dim - 78)]
    )
    mask = np.zeros_like(actions)
    mask[..., :78] = 1
    return torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(mask)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SonicBones(Dataset):
    def __init__(
        self,
        root: Path,
        episodes: list[int],
        *,
        task: str,
        horizon: int = 30,
        state_dim: int = 45,
        action_dim: int = 80,
        action_hz: float = 30,
    ):
        self.root = root
        self.info = json.loads((root / "meta/info.json").read_text())
        metadata = {
            row["episode_index"]: row
            for row in read_jsonl(root / "meta/episodes.jsonl")
        }
        self.tasks = {
            row["task_index"]: row["task"].lower()
            for row in read_jsonl(root / "meta/tasks.jsonl")
        }
        if not episodes or len(set(episodes)) != len(episodes):
            raise ValueError("Select at least one episode, without duplicates")
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.task = task.lower()
        self.episodes = episodes
        # This experiment has seven short episodes; decode each selected video
        # once rather than re-reading it as shuffled training revisits frames.
        self._episode = cache(self._read_episode)
        self.frames = {}
        self.samples = []
        for episode in episodes:
            if episode not in metadata or self.task not in metadata[episode]["tasks"]:
                raise ValueError(f"Episode {episode} does not contain task {task!r}")
            length = metadata[episode]["length"]
            # Sample the 50 Hz source in physical time on the checkpoint's 30 Hz clock.
            frames = np.rint(np.arange(0, length, self.info["fps"] / action_hz)).astype(
                np.int64
            )
            self.frames[episode] = np.minimum(frames, length - 1)
            self.samples.extend((episode, frame) for frame in range(len(frames)))

    def _path(self, template: str, episode: int) -> Path:
        return self.root / template.format(
            episode_chunk=episode // self.info["chunks_size"],
            episode_index=episode,
            video_key=VIDEO_KEY,
        )

    def _read_episode(self, episode: int) -> tuple:
        table = pq.read_table(self._path(self.info["data_path"], episode))
        arrays = [
            np.asarray(table[key].to_pylist(), dtype=np.float32)
            for key in ("observation.state", "action.wbc", "action.motion_token")
        ]
        frames = self.frames[episode]
        if len(arrays[0]) <= frames[-1]:
            raise ValueError(f"Episode {episode}: metadata/data length mismatch")
        if not all(
            self.tasks[index] == self.task for index in table["task_index"].to_pylist()
        ):
            raise ValueError(f"Episode {episode}: unexpected task in data")
        states, actions, mask = convert_sample(
            arrays[0][frames],
            arrays[1][frames],
            arrays[2][frames],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        wanted = set(frames.tolist())
        images = {}
        with av.open(str(self._path(self.info["video_path"], episode))) as video:
            for index, frame in enumerate(video.decode(video=0)):
                if index in wanted:
                    images[index] = frame.to_image()
        if wanted != images.keys():
            raise ValueError(f"Episode {episode}: missing video frames")
        return states, actions, mask, [images[index] for index in frames]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        episode, frame = self.samples[index]
        states, actions, mask, images = self._episode(episode)
        # LeRobot repeats the boundary action when a chunk reaches the episode end.
        chunk = torch.arange(frame, frame + self.horizon).clamp_max(len(actions) - 1)
        return {
            "image": images[frame],
            "states": states[frame : frame + 1],
            "actions": actions[chunk],
            "actions_mask": mask[chunk],
            "instruction": self.task,
        }
