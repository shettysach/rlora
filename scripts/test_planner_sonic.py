from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_yaw,
    quat_conjugate,
    quat_mul,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controller.sonic.policy import SonicPolicy
from shared.g1 import DEFAULT_JOINT_POS_MJLAB, SONIC_FROM_MJLAB
from sim.env import MjlabEnv

CONTROL_HZ = 50
PLANNER_HZ = 30
CONTEXT_FRAMES = 4
IDLE = 0
WALK = 2

OBSERVATION_DIMS = {
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}

TORCH_DTYPES = {
    "tensor(float)": torch.float32,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
}

NUMPY_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


class OnnxModel:
    def __init__(
        self,
        path: Path,
        device: torch.device,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        providers: list[str | tuple[str, dict[str, str]]] = ["CPUExecutionProvider"]
        device_id = 0
        if device.type == "cuda":
            device_id = (
                torch.cuda.current_device()
                if device.index is None
                else device.index
            )
            ort.preload_dlls()
            options = {"device_id": str(device_id)}
            if cuda_stream is not None:
                options["user_compute_stream"] = str(cuda_stream.cuda_stream)
            providers.insert(0, ("CUDAExecutionProvider", options))

        self.session = ort.InferenceSession(path, providers=providers)
        self.inputs = {
            value.name: torch.zeros(
                tuple(value.shape), dtype=TORCH_DTYPES[value.type], device=device
            )
            for value in self.session.get_inputs()
        }
        self.outputs = {
            value.name: torch.empty(
                tuple(value.shape), dtype=TORCH_DTYPES[value.type], device=device
            )
            for value in self.session.get_outputs()
        }
        self.binding = self.session.io_binding()
        for value in self.session.get_inputs():
            tensor = self.inputs[value.name]
            self.binding.bind_input(
                value.name,
                device.type,
                device_id,
                NUMPY_DTYPES[value.type],
                tensor.shape,
                tensor.data_ptr(),
            )
        for value in self.session.get_outputs():
            tensor = self.outputs[value.name]
            self.binding.bind_output(
                value.name,
                device.type,
                device_id,
                NUMPY_DTYPES[value.type],
                tensor.shape,
                tensor.data_ptr(),
            )

    def run(
        self, inputs: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        if inputs is not None:
            for name, value in inputs.items():
                self.inputs[name].copy_(value)
        self.session.run_with_iobinding(self.binding)
        return self.outputs


def standing_qpos() -> torch.Tensor:
    root = torch.tensor((0.0, 0.0, 0.788740, 1.0, 0.0, 0.0, 0.0))
    joints = torch.from_numpy(DEFAULT_JOINT_POS_MJLAB)
    return torch.cat((root, joints))


class Planner:
    def __init__(self, path: Path) -> None:
        self.model = OnnxModel(path, torch.device("cpu"))
        self.context = standing_qpos().repeat(1, CONTEXT_FRAMES, 1)

    def generate(self, mode: int) -> torch.Tensor:
        root = self.context[0, -1]
        facing = quat_apply_yaw(root[3:7], root.new_tensor((1.0, 0.0, 0.0)))
        inputs = {
            "context_mujoco_qpos": self.context,
            "target_vel": root.new_tensor([-1.0]),
            "mode": torch.tensor([mode], dtype=torch.int64),
            "movement_direction": (
                facing[None] if mode == WALK else facing.new_zeros(1, 3)
            ),
            "facing_direction": facing[None],
            "random_seed": torch.tensor([1234], dtype=torch.int64),
            "has_specific_target": torch.zeros((1, 1), dtype=torch.int64),
            "specific_target_positions": torch.zeros((1, CONTEXT_FRAMES, 3)),
            "specific_target_headings": torch.zeros((1, CONTEXT_FRAMES)),
            "allowed_pred_num_tokens": torch.ones((1, 11), dtype=torch.int64),
            "height": root.new_tensor([-1.0]),
        }
        output = self.model.run(inputs)
        frame_count = int(output["num_pred_frames"].item())
        qpos = output["mujoco_qpos"][0, :frame_count].clone()
        self.context = qpos[-CONTEXT_FRAMES:][None].clone()
        return qpos


def resample_qpos(qpos: torch.Tensor) -> torch.Tensor:
    output_frames = math.floor(len(qpos) * CONTROL_HZ / PLANNER_HZ)
    positions = torch.arange(output_frames, dtype=torch.float64) * PLANNER_HZ / CONTROL_HZ
    index0 = positions.floor().long().clamp(max=len(qpos) - 1)
    index1 = (index0 + 1).clamp(max=len(qpos) - 1)
    blend = (positions - index0).to(qpos.dtype)
    qpos0 = qpos.index_select(0, index0)
    qpos1 = qpos.index_select(0, index1)
    result = torch.lerp(qpos0, qpos1, blend[:, None])

    quat0 = qpos0[:, 3:7]
    quat1 = qpos1[:, 3:7]
    dot = (quat0 * quat1).sum(-1)
    quat1 = torch.where((dot < 0).unsqueeze(-1), -quat1, quat1)
    dot = dot.abs().clamp(-1, 1)
    angle = torch.acos(dot)
    same = angle < torch.finfo(qpos.dtype).eps * 4
    denominator = torch.where(same, torch.ones_like(angle), torch.sin(angle))
    interpolated = (
        quat0 * (torch.sin((1 - blend) * angle) / denominator)[:, None]
        + quat1 * (torch.sin(blend * angle) / denominator)[:, None]
    )
    result[:, 3:7] = torch.where(same[:, None], quat0, interpolated)
    return result.contiguous()


def observation_slices(names: list[str]) -> dict[str, slice]:
    result = {}
    offset = 0
    for name in names:
        size = OBSERVATION_DIMS[name]
        result[name] = slice(offset, offset + size)
        offset += size
    return result


class ReferenceEncoder:
    def __init__(
        self,
        bundle: Path,
        qpos: torch.Tensor,
        env: MjlabEnv,
        device: torch.device,
    ) -> None:
        with (bundle / "observation_config.yaml").open() as stream:
            config = yaml.safe_load(stream)
        names = [
            item["name"]
            for item in config["encoder"]["encoder_observations"]
            if item.get("enabled", False)
        ]
        self.slices = observation_slices(names)
        self.model = OnnxModel(
            bundle / "model_encoder.onnx", device, env.cuda_stream
        )
        self.input = next(iter(self.model.inputs.values()))
        self.output = next(iter(self.model.outputs.values()))
        self.sonic_from_mjlab = torch.as_tensor(SONIC_FROM_MJLAB, device=device)

        state = env.robot_state()
        qpos = qpos.to(device)
        rotation = quat_mul(state.root_quat_w[0], quat_conjugate(qpos[0, 3:7]))
        self.root_quat = quat_mul(rotation.expand(len(qpos), -1), qpos[:, 3:7])
        self.joint_pos = qpos[:, 7:]
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.joint_vel[:-1] = torch.diff(self.joint_pos, dim=0) * CONTROL_HZ
        self.joint_vel[-1] = self.joint_vel[-2]

    def encode(self, frame: int, root_quat: torch.Tensor) -> torch.Tensor:
        indices = (
            frame + torch.arange(10, device=self.joint_pos.device) * 5
        ).clamp(max=len(self.joint_pos) - 1)
        joint_pos = self.joint_pos.index_select(0, indices).index_select(
            1, self.sonic_from_mjlab
        )
        joint_vel = self.joint_vel.index_select(0, indices).index_select(
            1, self.sonic_from_mjlab
        )
        reference_quat = self.root_quat.index_select(0, indices)
        relative_quat = quat_mul(
            quat_conjugate(root_quat[0]).expand_as(reference_quat), reference_quat
        )
        orientation = matrix_from_quat(relative_quat)[..., :2]
        self.input[0, self.slices["encoder_mode_4"]].zero_()
        self.input[0, self.slices["motion_joint_positions_10frame_step5"]].copy_(
            joint_pos.flatten()
        )
        self.input[0, self.slices["motion_joint_velocities_10frame_step5"]].copy_(
            joint_vel.flatten()
        )
        self.input[0, self.slices["motion_anchor_orientation_10frame_step5"]].copy_(
            orientation.flatten()
        )
        self.model.run()
        return self.output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run planner_sonic through SONIC in MJLab")
    parser.add_argument("--sonic-bundle", type=Path, required=True)
    parser.add_argument("--walk-segments", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    planner = Planner(args.sonic_bundle / "planner_sonic.onnx")
    chunks = [planner.generate(mode=WALK) for _ in range(args.walk_segments)]
    chunks.append(planner.generate(mode=IDLE))
    trajectory = resample_qpos(torch.cat(chunks))

    env = MjlabEnv(1, device=args.device)
    viewer = None
    previous_stream = (
        torch.cuda.current_stream() if env.cuda_stream is not None else None
    )
    if env.cuda_stream is not None:
        torch.cuda.set_stream(env.cuda_stream)
    try:
        device = torch.device(args.device)
        encoder = ReferenceEncoder(args.sonic_bundle, trajectory, env, device)
        controller = SonicPolicy(
            args.sonic_bundle,
            1,
            device=args.device,
            cuda_stream=env.cuda_stream,
        )
        controller.reset()
        if args.viewer:
            from sim.viewer import SimViewer

            viewer = SimViewer(env.env)

        initial_xy = env.robot_state().root_pos_w[:, :2].clone()
        for frame in range(len(trajectory)):
            state = env.robot_state()
            token = encoder.encode(frame, state.root_quat_w)
            action = controller.act(reference=token, robot_state=state)
            env.step(action)
            if viewer is not None:
                viewer.sync()

        displacement = env.robot_state().root_pos_w[:, :2] - initial_xy
        print(
            json.dumps(
                {
                    "frames": len(trajectory),
                    "duration_s": len(trajectory) / CONTROL_HZ,
                    "root_displacement_xy": displacement.cpu().tolist(),
                },
                indent=2,
            )
        )
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
        if previous_stream is not None:
            torch.cuda.set_stream(previous_stream)


if __name__ == "__main__":
    main()
