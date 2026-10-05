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
LIGHT_NAMES = tuple(f"Light_{row}_{column}" for row in range(2) for column in range(3))
REFERENCE_INTENSITY = 9695.757778779214
REFERENCE_TEMPERATURE = 7815.292395897912
REFERENCE_LIGHT_COLOR = np.array([0.18, 0.2, 0.23])
# MuJoCo classic-material approximations for the five saved Isaac MDL choices.
TABLE_MATERIALS = {
    "Pearl": ((0.85, 0.84, 0.8, 1.0), 0.6, 0.5),
    "Polyethylene_Dark_Gray": ((0.19, 0.2, 0.21, 1.0), 0.2, 0.2),
    "blue_eyes": ((0.17, 0.21, 0.26, 1.0), 0.7, 0.7),
    "bark_oak_dark_contrast": ((0.31, 0.2, 0.12, 1.0), 0.15, 0.15),
    "Gunmetal_Matte_Metallic": ((0.3, 0.31, 0.33, 1.0), 0.45, 0.35),
}


@dataclass
class Appearance:
    light_positions: np.ndarray
    light_diffuse: np.ndarray
    table_rgba: tuple[float, float, float, float]
    table_specular: float
    table_shininess: float


def _temperature_rgb(kelvin: float) -> np.ndarray:
    """Approximate blackbody sRGB, used only for relative light tint."""
    t = kelvin / 100
    red = 255 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    green = (
        99.4708025861 * np.log(t) - 161.1195681661
        if t <= 66
        else 288.1221695283 * (t - 60) ** -0.0755148492
    )
    blue = 255 if t >= 66 else 138.5177312231 * np.log(t - 10) - 305.0447927307
    return np.clip([red, green, blue], 0, 255) / 255


def _appearance(scene: dict) -> Appearance:
    lights = scene["lighting"]
    positions = []
    diffuse = []
    reference_rgb = _temperature_rgb(REFERENCE_TEMPERATURE)
    for name in LIGHT_NAMES:
        light = lights[name]
        center = np.asarray(light["center_light_postion"], dtype=np.float64)
        q = np.asarray(light["center_light_orientation"], dtype=np.float64)
        local = np.asarray(light["pose"]["position"], dtype=np.float64)
        xyz = q[1:]
        positions.append(
            center + local + 2 * np.cross(xyz, np.cross(xyz, local) + q[0] * local)
        )
        diffuse.append(
            REFERENCE_LIGHT_COLOR
            * light["light_intensity"]
            / REFERENCE_INTENSITY
            * _temperature_rgb(light["light_color_temperature"])
            / reference_rgb
        )
    material = scene["material"]["table_material"]["name"]
    rgba, specular, shininess = TABLE_MATERIALS[material]
    return Appearance(
        np.asarray(positions), np.asarray(diffuse), rgba, specular, shininess
    )


@dataclass
class Episode:
    index: int
    instruction: str
    base_pose: np.ndarray
    joint_pos: np.ndarray
    box_pose: np.ndarray
    actions: np.ndarray
    appearance: Appearance


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
            scene = json.loads(entry["environment_config"])["dr_state_dict"]
            table = scene["scene"]["table"]
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
                    appearance=_appearance(scene),
                )
            )
    return episodes
