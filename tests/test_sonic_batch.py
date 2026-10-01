import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from controller.sonic.policy import SonicPolicy
from shared.state import RobotState


def test_one_decoder_keeps_two_histories_independent(
    tmp_path: Path, monkeypatch
) -> None:
    input_dim = 994
    # Decoder output copies token[0:29], making batch correspondence observable.
    weights = np.zeros((input_dim, 29), dtype=np.float32)
    weights[np.arange(29), np.arange(29)] = 1
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weights"], ["output"])],
        "decoder",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["batch", input_dim]
            )
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 29])],
        [numpy_helper.from_array(weights, name="weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, tmp_path / "model_decoder.onnx")
    monkeypatch.setattr(
        SonicPolicy,
        "DECODER_SHA256",
        hashlib.sha256((tmp_path / "model_decoder.onnx").read_bytes()).hexdigest(),
    )

    policy = SonicPolicy(tmp_path, batch_size=2, device="cpu")
    zeros = lambda *shape: torch.zeros(shape)
    state = RobotState(
        root_ang_vel_b=zeros(2, 3),
        projected_gravity_b=zeros(2, 3),
        joint_pos=zeros(2, 29),
        joint_vel=zeros(2, 29),
    )
    tokens = zeros(2, 64)
    tokens[0, :29] = 0.25
    tokens[1, :29] = 0.5
    actions = policy.act(reference=tokens, robot_state=state)
    assert actions.shape == (2, 29)
    assert torch.all(actions[0] == 0.25)
    assert torch.all(actions[1] == 0.5)
    history = policy.model.input[:, policy.LAST_ACTIONS].view(2, 10, 29)
    assert torch.all(history[0, -1] == 0.25)
    assert torch.all(history[1, -1] == 0.5)

    # Run beyond the history length with different per-env tokens and velocities.
    expected_actions = []
    expected_velocities = []
    for step in range(12):
        tokens[0] = (step % 5) / 16
        tokens[1] = -(step % 7) / 16
        state.root_ang_vel_b[0] = step
        state.root_ang_vel_b[1] = -step
        actions = policy.act(reference=tokens, robot_state=state)
        expected_actions.append(tokens[:, :29].clone())
        expected_velocities.append(state.root_ang_vel_b.clone())
        torch.testing.assert_close(actions, expected_actions[-1])
    torch.testing.assert_close(history, torch.stack(expected_actions[-10:], dim=1))
    velocity_history = policy.model.input[:, policy.BASE_ANGULAR_VELOCITY].view(
        2, 10, 3
    )
    torch.testing.assert_close(
        velocity_history, torch.stack(expected_velocities[-10:], dim=1)
    )

    policy.reset()
    policy.act(reference=tokens, robot_state=state)
    assert torch.all(history[:, :-1] == 0)
    assert torch.all(velocity_history[:, :-1] == 0)
    torch.testing.assert_close(history[:, -1], tokens[:, :29])
    torch.testing.assert_close(velocity_history[:, -1], state.root_ang_vel_b)
