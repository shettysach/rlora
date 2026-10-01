"""Read the five published SIMPLE carry-box evaluation episodes."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq

from shared.g1 import BODY_JOINTS, HAND_JOINTS

ARCHIVE_ROOT = "G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0/"


@dataclass
class Episode:
    index: int
    instruction: str
    base_pose: np.ndarray
    joint_pos: np.ndarray
    box_pose: np.ndarray
    actions: np.ndarray


def load_episodes(archive: Path, indices: list[int]) -> list[Episode]:
    with ZipFile(archive) as source:
        metadata = [
            json.loads(line)
            for line in source.read(ARCHIVE_ROOT + "meta/episodes.jsonl").splitlines()
        ]
        info = json.loads(source.read(ARCHIVE_ROOT + "meta/info.json"))
        state_names = tuple(info["features"]["observation.state"]["names"])
        if state_names != BODY_JOINTS + HAND_JOINTS:
            raise ValueError("Recorded state joint order differs from G1 Dex3")
        if info["features"]["action"]["shape"] != [78]:
            raise ValueError("Expected 64 body token and 14 hand targets")

        episodes = []
        for index in indices:
            entry = metadata[index]
            scene = json.loads(entry["environment_config"])
            table = scene["dr_state_dict"]["scene"]["table"]
            if table["pose"]["position"] != [0.3, 0.0, 0.4]:
                raise ValueError("Recorded table pose differs from the MJLab scene")
            data = pq.read_table(
                io.BytesIO(
                    source.read(
                        ARCHIVE_ROOT + f"data/chunk-000/episode_{index:06d}.parquet"
                    )
                ),
                columns=[
                    "observation.state",
                    "observation.base_pose",
                    "observation.object_poses",
                    "action",
                ],
            )

            def values(name: str, table=data) -> np.ndarray:
                return np.asarray(table[name].to_pylist(), dtype=np.float32)

            episodes.append(
                Episode(
                    index=index,
                    instruction=entry["tasks"][0],
                    base_pose=values("observation.base_pose")[0],
                    joint_pos=values("observation.state")[0],
                    box_pose=values("observation.object_poses")[0],
                    actions=values("action"),
                )
            )
    return episodes
