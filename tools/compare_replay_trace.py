"""Compare a local carry-box replay trace with the published observations."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq

ARCHIVE_ROOT = "G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0/"


def _rmse(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - reference) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--eval-archive", type=Path, required=True)
    args = parser.parse_args()

    with (
        np.load(args.trace, allow_pickle=False) as trace,
        ZipFile(args.eval_archive) as archive,
    ):
        print(
            f"actuation={trace['actuation'].item()} startup={trace['startup'].item()} "
            f"delay={trace['control_delay'].item()}"
        )
        for column, episode_index in enumerate(trace["episode_indices"]):
            data = pq.read_table(
                io.BytesIO(
                    archive.read(
                        ARCHIVE_ROOT
                        + f"data/chunk-000/episode_{episode_index:06d}.parquet"
                    )
                ),
                columns=[
                    "observation.state",
                    "observation.base_pose",
                    "observation.object_poses",
                ],
            )
            frames = min(data.num_rows, len(trace["joint_pos"]))
            recorded_joints = np.asarray(data["observation.state"].to_pylist()[:frames])
            recorded_base = np.asarray(
                data["observation.base_pose"].to_pylist()[:frames]
            )
            recorded_box = np.asarray(
                data["observation.object_poses"].to_pylist()[:frames]
            )
            joints = trace["joint_pos"][:frames, column]
            base = trace["base_pose"][:frames, column]
            box = trace["box_pose"][:frames, column]
            early = min(frames, 101)
            torque = trace["torque"][:, column]
            print(
                f"episode {episode_index}: frames={frames}, "
                f"joint_rmse_early={_rmse(joints[:early], recorded_joints[:early]):.4f}, "
                f"joint_rmse_all={_rmse(joints, recorded_joints):.4f}, "
                f"base_xyz_rmse={_rmse(base[:, :3], recorded_base[:, :3]):.4f}, "
                f"box_xyz_rmse={_rmse(box[:, :3], recorded_box[:, :3]):.4f}, "
                f"box_max_z={box[:, 2].max():.3f}/{recorded_box[:, 2].max():.3f}, "
                f"max_abs_torque={np.abs(torque).max():.2f}, "
                f"success={bool(trace['success'][column])}"
            )


if __name__ == "__main__":
    main()
