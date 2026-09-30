from __future__ import annotations

import argparse
import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

from controller.sonic.policy import SonicPolicy
from planner.psi0 import Psi0Planner
from sim.env import MjlabEnv
from walk_to_target import INSTRUCTION, TASK_NAME, WalkToTarget

PLANNER_HZ = 30.0
RTC_REPLAN_AFTER = 15
RTC_INITIAL_DELAY = 6


def run(
    *,
    num_envs: int,
    num_episodes: int,
    episode_seconds: float,
    psi_run_dir: Path,
    ckpt_step: int,
    qwen_model: Path,
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
    for path in (qwen_model, clip_model):
        if not path.is_dir():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")

    env = MjlabEnv(num_envs, device=device, record_video=record_video)
    viewer = None
    previous_stream = (
        torch.cuda.current_stream() if env.cuda_stream is not None else None
    )
    if env.cuda_stream is not None:
        torch.cuda.set_stream(env.cuda_stream)
    try:
        planner = Psi0Planner(
            psi_run_dir, ckpt_step, qwen_model, clip_model, device=device
        )
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
        instructions = [INSTRUCTION] * num_envs
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()

        def sync_sim() -> None:
            if env.cuda_stream is not None:
                env.cuda_stream.synchronize()

        def sample_gpu_usage() -> None:
            nonlocal gpu_used_peak
            if device.startswith("cuda"):
                free, total = torch.cuda.mem_get_info(device)
                gpu_used_peak = max(gpu_used_peak, total - free)

        def replan(
            images: torch.Tensor,
            joints: torch.Tensor,
            executed: int,
            delay: int,
        ) -> tuple[torch.Tensor, float]:
            started = time.perf_counter()
            result = planner.predict(images, joints, instructions, executed, delay)
            return result, time.perf_counter() - started

        rtc_deadline_misses = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            for batch_start in range(0, num_episodes, num_envs):
                env.reset()
                controller.reset()
                planner.reset()
                state = env.robot_state()
                task = WalkToTarget.start(state, episode_seconds)
                elapsed = 0.0
                inference_delays = deque([RTC_INITIAL_DELAY], maxlen=6)

                sync_sim()
                before = time.perf_counter()
                reference = planner.predict(
                    env.rgb().cpu(),
                    state.joint_pos.cpu(),
                    instructions,
                )
                planner_times.append(time.perf_counter() - before)
                sample_gpu_usage()
                action_cursor = 0.0
                replan_cursor = 0
                pending: Future[tuple[torch.Tensor, float]] | None = None

                while elapsed < episode_seconds:
                    if pending is not None and (
                        pending.done() or action_cursor >= planner.horizon
                    ):
                        if not pending.done():
                            rtc_deadline_misses += 1
                        reference, latency = pending.result()
                        planner_times.append(latency)
                        delay = int(action_cursor) - replan_cursor
                        inference_delays.append(delay)
                        action_cursor = float(delay)
                        pending = None
                        sample_gpu_usage()

                    chunk_seconds_left = (planner.horizon - action_cursor) / PLANNER_HZ
                    if (
                        pending is None
                        and action_cursor >= RTC_REPLAN_AFTER
                        and elapsed + chunk_seconds_left < episode_seconds
                    ):
                        sync_sim()
                        replan_cursor = int(action_cursor)
                        state = env.robot_state()
                        pending = executor.submit(
                            replan,
                            env.rgb().cpu(),
                            state.joint_pos.cpu(),
                            replan_cursor,
                            max(inference_delays),
                        )

                    action_idx = min(int(action_cursor), planner.horizon - 1)
                    sync_sim()
                    before = time.perf_counter()
                    state = env.robot_state()
                    joints = controller.act(
                        reference=reference[:, action_idx],
                        robot_state=state,
                    )
                    sync_sim()
                    sonic_times.append(time.perf_counter() - before)
                    env.step(joints)
                    if viewer is not None:
                        viewer.sync()

                    state = env.robot_state()
                    task.update(state, elapsed, env.step_dt)
                    control_steps += 1
                    elapsed += env.step_dt
                    action_cursor += PLANNER_HZ * env.step_dt

                active = min(num_envs, num_episodes - batch_start)
                episodes.extend(task.results(env.robot_state(), active))

        sync_sim()
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
            "rtc_deadline_misses": rtc_deadline_misses,
            "sonic_latency_ms": 1000 * sum(sonic_times) / len(sonic_times),
            "control_steps_per_s": num_envs * control_steps / wall_seconds,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
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
    psi_run_dir = Path(
        "artifacts/psi-model/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156"
    )
    checkpoint_step = 40000
    qwen_model = Path("artifacts/qwen3-vl-2b-instruct")
    clip_model = Path("artifacts/clip-vit-large-patch14")
    sonic_bundle = Path("artifacts/sonic")

    parser = argparse.ArgumentParser(description="Evaluate WalkToTarget-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--psi-run-dir", type=Path, default=psi_run_dir)
    parser.add_argument("--ckpt-step", type=int, default=checkpoint_step)
    parser.add_argument("--qwen-model", type=Path, default=qwen_model)
    parser.add_argument("--clip-model", type=Path, default=clip_model)
    parser.add_argument("--sonic-bundle", type=Path, default=sonic_bundle)
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
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        episode_seconds=args.episode_seconds,
        psi_run_dir=args.psi_run_dir,
        ckpt_step=args.ckpt_step,
        qwen_model=args.qwen_model,
        clip_model=args.clip_model,
        sonic_bundle=args.sonic_bundle,
        device=args.device,
        viewer_enabled=args.viewer,
        record_video=args.record_video,
    )
    output = json.dumps(result, indent=2)
    print(output)
    if args.save_metrics is not None:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.save_metrics.write_text(output + "\n")


if __name__ == "__main__":
    main()
