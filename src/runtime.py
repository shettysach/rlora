from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from controller.sonic.policy import SonicPolicy
from planner.psi0 import Psi0Planner
from sim.env import MjlabEnv

PLANNER_HZ = 30.0  # SONIC Ψ₀ action chunks are sampled at 30 Hz.
INSTRUCTION = "grasp the backrest of the chair and push it straight under the table"


def run(
    num_envs: int,
    chunks: int,
    psi_run_dir: Path,
    ckpt_step: int,
    sonic_bundle: Path,
    device: str = "cuda",
) -> dict:
    checkpoint = psi_run_dir / "checkpoints" / f"ckpt_{ckpt_step}" / "model.safetensors"
    for path in (
        psi_run_dir / "run_config.json",
        checkpoint,
        sonic_bundle / "model_decoder.onnx",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")
    env = MjlabEnv(num_envs, device=device)
    previous_stream = (
        torch.cuda.current_stream() if env.cuda_stream is not None else None
    )
    if env.cuda_stream is not None:
        torch.cuda.set_stream(env.cuda_stream)
    try:
        planner = Psi0Planner(psi_run_dir, ckpt_step, device=device)
        controller = SonicPolicy(
            sonic_bundle, num_envs, device=device, cuda_stream=env.cuda_stream
        )
        start_pos = env.robot_state().root_pos_w.clone()
        planner_times, sonic_times = [], []
        control_steps = 0
        gpu_used_peak = 0
        wall_start = time.perf_counter()

        def sync() -> None:
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        def sample_gpu_usage() -> None:
            nonlocal gpu_used_peak
            if device.startswith("cuda"):
                free, total = torch.cuda.mem_get_info()
                gpu_used_peak = max(gpu_used_peak, total - free)

        for _ in range(chunks):
            images, states = env.rgb(), env.robot_state()
            sync()
            before = time.perf_counter()
            reference_chunks = planner.predict(
                images, states, [INSTRUCTION] * num_envs
            )
            sync()
            planner_times.append(time.perf_counter() - before)
            sample_gpu_usage()
            # Execute only the checkpoint's configured horizon, then replan.
            elapsed = 0.0
            duration = planner.exec_horizon / PLANNER_HZ
            while elapsed < duration:
                action_idx = int(elapsed * PLANNER_HZ)
                sync()
                before = time.perf_counter()
                joints = controller.act(
                    reference=reference_chunks[:, action_idx],
                    robot_state=env.robot_state(),
                )
                sync()
                sonic_times.append(time.perf_counter() - before)
                env.step(joints)
                control_steps += 1
                elapsed += env.step_dt
            sample_gpu_usage()

        sync()
        wall_seconds = time.perf_counter() - wall_start
        displacement = (env.robot_state().root_pos_w - start_pos).cpu().tolist()
        return {
            "num_envs": num_envs,
            "chunks": chunks,
            "control_steps": control_steps,
            "planner_latency_ms": 1000 * sum(planner_times) / len(planner_times),
            "planner_throughput_envs_per_s": num_envs
            * len(planner_times)
            / sum(planner_times),
            "sonic_latency_ms": 1000 * sum(sonic_times) / len(sonic_times),
            "control_steps_per_s": num_envs * control_steps / wall_seconds,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if device.startswith("cuda")
            else None,
            "gpu_used_bytes_sampled_peak": gpu_used_peak
            if device.startswith("cuda")
            else None,
            "root_displacement_xyz": displacement,
        }
    finally:
        env.close()
        if previous_stream is not None:
            torch.cuda.set_stream(previous_stream)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batched Ψ₀ → SONIC → MJLab walking demo"
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--psi-run-dir", type=Path, required=True)
    parser.add_argument("--ckpt-step", type=int, required=True)
    parser.add_argument("--sonic-bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    print(
        json.dumps(
            run(
                args.num_envs,
                args.chunks,
                args.psi_run_dir,
                args.ckpt_step,
                args.sonic_bundle,
                args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
