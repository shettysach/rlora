from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from controller.sonic.policy import SonicPolicy
from shared.state import RobotState


def test_one_decoder_keeps_two_histories_independent(tmp_path: Path) -> None:
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

    policy = SonicPolicy(tmp_path, batch_size=2, device="cpu")
    zeros = lambda *shape: torch.zeros(shape)
    state = RobotState(
        zeros(2, 3),
        zeros(2, 4),
        zeros(2, 3),
        zeros(2, 3),
        zeros(2, 3),
        zeros(2, 29),
        zeros(2, 29),
    )
    tokens = zeros(2, 64)
    tokens[0, :29] = 1
    tokens[1, :29] = 2
    actions = policy.act(reference=tokens, robot_state=state)
    assert actions.shape == (2, 29)
    assert torch.all(actions[0] == 1)
    assert torch.all(actions[1] == 2)
    history = policy.model.input[:, policy.LAST_ACTIONS].view(2, 10, 29)
    assert torch.all(history[0, -1] == 1)
    assert torch.all(history[1, -1] == 2)
