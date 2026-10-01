"""Run the SIMPLE carry-box reference in a batched MJLab scene."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from carry_box import CarryBoxTask
from carry_box_data import Episode, load_episodes
from controller.sonic.policy import SonicPolicy
from planner.sonic_http import SonicHttpPlanner
from sim.env import MjlabEnv


def _reference_chunk(episodes: list[Episode], start: int, length: int) -> np.ndarray:
    return np.stack(
        [
            episode.actions[
                np.minimum(np.arange(start, start + length), len(episode.actions) - 1)
            ]
            for episode in episodes
        ]
    )


def run(args: argparse.Namespace) -> dict:
    episodes = load_episodes(args.eval_archive, args.episode_indices)
    count = len(episodes)
    env = MjlabEnv(count, device=args.device, record_video=args.record_video)
    viewer = None
    planner = None
    try:
        env.reset(
            seed=args.seed,
            base_pose=torch.from_numpy(np.stack([e.base_pose for e in episodes])),
            joint_pos=torch.from_numpy(np.stack([e.joint_pos for e in episodes])),
            box_pose=torch.from_numpy(np.stack([e.box_pose for e in episodes])),
        )
        task = CarryBoxTask(env)
        if args.viewer:
            from sim.viewer import SimViewer

            viewer = SimViewer(env.env)
        if args.mode == "scene":
            return {
                "mode": "scene",
                "episode_indices": args.episode_indices,
                "image_shape": list(env.rgb().shape),
                "state_shape": list(env.planner_state().shape),
                "box_pose": env.box_pose().tolist(),
                "robot_pose": env.robot_state().root_pos_w.tolist(),
            }

        controller = SonicPolicy(
            args.sonic_bundle, count, device=args.device, cuda_stream=env.cuda_stream
        )
        controller.reset()
        if args.mode == "policy":
            planner = SonicHttpPlanner(args.psi_url, count)
            planner.reset()
        planner_latencies = []
        sonic_latencies = []
        executed = 0
        started = time.perf_counter()
        while executed < args.max_steps:
            if planner is None:
                horizon = min(30, args.max_steps - executed)
                chunk = _reference_chunk(episodes, executed, horizon)
            else:
                before = time.perf_counter()
                chunk = planner.predict(
                    env.rgb().cpu().numpy(),
                    env.planner_state().cpu().numpy(),
                    [args.prompt or e.instruction for e in episodes],
                )
                planner_latencies.append(time.perf_counter() - before)
                horizon = min(planner.exec_horizon, args.max_steps - executed)
            actions = torch.as_tensor(chunk, device=args.device)
            for index in range(horizon):
                reference = actions[:, index]
                before = time.perf_counter()
                joints = controller.act(
                    reference=reference[:, :64], robot_state=env.robot_state()
                )
                if env.cuda_stream is not None:
                    env.cuda_stream.synchronize()
                sonic_latencies.append(time.perf_counter() - before)
                env.step(joints, reference[:, 64:78])
                executed += 1
                task.update(executed)
                if viewer is not None:
                    viewer.sync()
                if bool((task.success | task.fell).all().item()):
                    break
            if bool((task.success | task.fell).all().item()):
                break

        results = task.results(executed)
        for episode, result in zip(episodes, results):
            result["episode_index"] = episode.index
        return {
            "task": "simple/G1WholebodyXMoveBendCarryBoxSonic-v0",
            "mode": args.mode,
            "episode_indices": args.episode_indices,
            "instruction": args.prompt or episodes[0].instruction,
            "seed": args.seed,
            "control_steps": executed,
            "wall_seconds": time.perf_counter() - started,
            "success_rate": sum(r["success"] for r in results) / count,
            "planner_latency_ms": 1000 * np.mean(planner_latencies)
            if planner_latencies
            else None,
            "sonic_latency_ms": 1000 * np.mean(sonic_latencies),
            "episodes": results,
        }
    finally:
        if planner is not None:
            planner.close()
        if viewer is not None:
            viewer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MJLab carry-box task probe")
    parser.add_argument(
        "--mode", choices=("scene", "replay", "policy"), default="scene"
    )
    parser.add_argument("--eval-archive", type=Path, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sonic-bundle", type=Path, default=Path("artifacts/sonic"))
    parser.add_argument("--psi-url", default="http://127.0.0.1:8014")
    parser.add_argument("--prompt", help="Override the recorded task instruction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record-video", type=Path)
    parser.add_argument("--save-metrics", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if len(set(args.episode_indices)) != len(args.episode_indices):
        parser.error("--episode-indices must be distinct")
    result = run(args)
    output = json.dumps(result, indent=2)
    print(output)
    if args.save_metrics is not None:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.save_metrics.write_text(output + "\n")


if __name__ == "__main__":
    main()
