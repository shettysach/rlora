import json
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import evaluate_bc


@pytest.mark.parametrize(
    "rate,expected", [(None, [0, 1, 2, 3, 4]), (30, [0, 0, 2, 2, 3])]
)
def test_replay_preserves_raw_tokens_and_physical_clock(
    tmp_path, monkeypatch, rate, expected
):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 50,
                "chunks_size": 1000,
                "data_path": "episode_{episode_index:06d}.parquet",
            }
        )
    )
    tokens = np.zeros((5, 64), dtype=np.float32)
    tokens[:, 0] = [-0.75, -0.625, 0, 0.25, 0.625]
    yaw = np.arange(5) * 0.1
    quaternions = np.column_stack((np.cos(yaw / 2), np.zeros((5, 2)), np.sin(yaw / 2)))
    pq.write_table(
        pa.table(
            {
                "action.motion_token": tokens.tolist(),
                "observation.root_orientation": quaternions.tolist(),
            }
        ),
        root / "episode_000047.parquet",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"seed": 0, "dataset": {"root": str(root), "task": "walk and turn around"}}
        )
    )
    references = []

    class Env:
        step_dt = 0.02
        cuda_stream = None

        def __init__(self, *args, **kwargs):
            self.state = SimpleNamespace(
                root_pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            )

        def compute_context(self):
            from contextlib import nullcontext

            return nullcontext()

        def reset(self):
            pass

        def robot_state(self):
            return self.state

        def step(self, action):
            self.state.root_pos_w[:, 0] += 0.01

        def close(self):
            pass

    class Sonic:
        def __init__(self, *args, **kwargs):
            pass

        def act(self, *, reference, robot_state):
            references.append(reference.clone())
            return torch.zeros(1, 29)

    monkeypatch.setitem(sys.modules, "sim.env", SimpleNamespace(MjlabEnv=Env))
    monkeypatch.setattr(evaluate_bc, "SonicPolicy", Sonic)
    output = tmp_path / "metrics.json"
    argv = [
        "evaluate_bc",
        "--config",
        str(config),
        "--replay-episode",
        "47",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    if rate is not None:
        argv += ["--replay-hz", str(rate)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate_bc.main()
    torch.testing.assert_close(
        torch.cat(references), torch.from_numpy(tokens[expected])
    )
    result = json.loads(output.read_text())
    assert result["seconds"] == 0.1
    assert result["replay"]["demonstrated_net_turn_rad"] == pytest.approx(0.4)
    assert result["episodes"][0]["path_length_m"] == pytest.approx(0.05)
