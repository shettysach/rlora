from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from controller.sonic.policy import SonicPolicy
from planner.psi0 import Psi0Planner
from sim.env import MjlabEnv
from walk_to_target import INSTRUCTION, TASK_NAME, WalkToTarget

PLANNER_HZ = 30.0


def run(
    num_envs: int,
    num_episodes: int,
    episode_seconds: float,
    psi_run_dir: Path,
    ckpt_step: int,
    clip_model: Path,
    sonic_bundle: Path,
    device: str = "cuda",
    viewer_enabled: bool = False,
    record_video: Path | None = None,
) -> dict:
    checkpoint = psi_run_dir / "checkpoints" / f"ckpt_{ckpt_step}" / "model.safetensors"
    for path in (
        psi_run_dir / "run_config.json",
        checkpoint,
        sonic_bundle / "model_decoder.onnx",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")
    if not clip_model.is_dir():
        raise FileNotFoundError(f"Required model artifact is missing: {clip_model}")

    env = MjlabEnv(num_envs, device=device, record_video=record_video)
    viewer = None
    previous_stream = (
        torch.cuda.current_stream() if env.cuda_stream is not None else None
    )
    if env.cuda_stream is not None:
        torch.cuda.set_stream(env.cuda_stream)
    try:
        planner = Psi0Planner(psi_run_dir, ckpt_step, clip_model, device=device)
        controller = SonicPolicy(
            sonic_bundle, num_envs, device=device, cuda_stream=env.cuda_stream
        )
        if viewer_enabled:
            from sim.viewer import SimViewer

            viewer = SimViewer(env.env)

        planner_times: list[float] = []
        sonic_times: list[float] = []
        episodes: list[dict] = []
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

        num_batches = math.ceil(num_episodes / num_envs)
        for batch in range(num_batches):
            env.reset()
            controller.reset()
            state = env.robot_state()
            task = WalkToTarget.start(state, episode_seconds)
            elapsed = 0.0

            while elapsed < episode_seconds:
                images, state = env.rgb(), env.robot_state()
                sync()
                before = time.perf_counter()
                reference = planner.predict(
                    images, state, [INSTRUCTION] * num_envs
                )
                sync()
                planner_times.append(time.perf_counter() - before)
                sample_gpu_usage()

                chunk_elapsed = 0.0
                chunk_duration = min(
                    planner.exec_horizon / PLANNER_HZ,
                    episode_seconds - elapsed,
                )
                while chunk_elapsed < chunk_duration:
                    action_idx = int(chunk_elapsed * PLANNER_HZ)
                    sync()
                    before = time.perf_counter()
                    joints = controller.act(
                        reference=reference[:, action_idx],
                        robot_state=env.robot_state(),
                    )
                    sync()
                    sonic_times.append(time.perf_counter() - before)
                    env.step(joints)
                    if viewer is not None:
                        viewer.sync()

                    task.update(env.robot_state(), elapsed, env.step_dt)
                    control_steps += 1
                    elapsed += env.step_dt
                    chunk_elapsed += env.step_dt
                sample_gpu_usage()

            active = min(num_envs, num_episodes - batch * num_envs)
            episodes.extend(task.results(env.robot_state(), active))

        sync()
        wall_seconds = time.perf_counter() - wall_start
        success_count = sum(episode["success"] for episode in episodes)
        fall_count = sum(episode["fell"] for episode in episodes)
        return {
            "task": TASK_NAME,
            "instruction": INSTRUCTION,
            "num_envs": num_envs,
            "num_episodes": num_episodes,
            "success_rate": success_count / num_episodes,
            "fall_rate": fall_count / num_episodes,
            "mean_final_goal_distance_m": sum(
                episode["final_goal_distance_m"] for episode in episodes
            )
            / num_episodes,
            "mean_progress_m": sum(episode["progress_m"] for episode in episodes)
            / num_episodes,
            "mean_episode_time_s": sum(
                episode["episode_time_s"] for episode in episodes
            )
            / num_episodes,
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
            "episodes": episodes,
        }
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
        if previous_stream is not None:
            torch.cuda.set_stream(previous_stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WalkToTarget-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--psi-run-dir", type=Path, required=True)
    parser.add_argument("--ckpt-step", type=int, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--sonic-bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record-video", type=Path)
    parser.add_argument("--save-metrics", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.num_episodes < 1:
        parser.error("--num-episodes must be positive")
    if args.episode_seconds <= 0:
        parser.error("--episode-seconds must be positive")

    result = run(
        args.num_envs,
        args.num_episodes,
        args.episode_seconds,
        args.psi_run_dir,
        args.ckpt_step,
        args.clip_model,
        args.sonic_bundle,
        args.device,
        args.viewer,
        args.record_video,
    )
    output = json.dumps(result, indent=2)
    print(output)
    if args.save_metrics is not None:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.save_metrics.write_text(output + "\n")


if __name__ == "__main__":
    main()
