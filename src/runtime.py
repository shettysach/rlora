"""Run the SIMPLE carry-box reference in a batched MJLab scene."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import sleep

import numpy as np
import torch

from carry_box import CarryBoxTask
from carry_box_data import Episode, load_episodes
from controller.sonic.policy import SonicPolicy
from planner.psi0 import Psi0Planner
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


def run(args: argparse.Namespace) -> None:
    episodes = load_episodes(args.eval_archive, args.episode_indices)
    count = len(episodes)
    env = MjlabEnv(
        count, device=args.device, appearances=[e.appearance for e in episodes]
    )
    viewer = None
    planner = None
    controller = None
    try:
        with env.compute_context():
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
            while viewer is not None and viewer.is_running():
                viewer.sync()
                sleep(0.02)
            return

        controller = SonicPolicy(
            args.sonic_bundle, count, device=args.device, cuda_stream=env.cuda_stream
        )
        controller.reset()
        if args.mode == "replay":
            replay_actions = torch.as_tensor(
                _reference_chunk(episodes, 0, args.max_steps), device=args.device
            )
        if args.mode == "policy":
            planner = Psi0Planner(
                args.psi_run_dir,
                args.ckpt_step,
                args.qwen_model,
                device=args.device,
                inference_steps=args.inference_steps,
            )
        with env.compute_context():
            terminal_steps = torch.zeros(count, dtype=torch.int32, device=env.device)
            succeeded = torch.zeros(count, dtype=torch.bool, device=env.device)
            fell = torch.zeros_like(succeeded)
        executed = 0
        while executed < args.max_steps:
            if planner is None:
                horizon = min(30, args.max_steps - executed)
                actions = replay_actions[:, executed : executed + horizon]
            else:
                images = env.rgb()
                if env.cuda_stream is not None:
                    torch.cuda.current_stream(env.device).wait_stream(env.cuda_stream)
                actions = planner.predict(
                    images,
                    env.planner_state(),
                    [args.prompt or e.instruction for e in episodes],
                )
                horizon = min(planner.exec_horizon, args.max_steps - executed)
            if env.cuda_stream is not None:
                env.cuda_stream.wait_stream(torch.cuda.current_stream(env.device))
            for index in range(horizon):
                reference = actions[:, index]
                with env.compute_context():
                    joints = controller.act(
                        reference=reference[:, :64], robot_state=env.robot_state()
                    )
                    env.step(joints, reference[:, 64:78])
                    executed += 1
                    task.update()
                    newly_done = (terminal_steps == 0) & (task.success | task.fell)
                    terminal_steps.masked_fill_(newly_done, executed)
                    succeeded |= newly_done & task.success
                    fell |= newly_done & task.fell
                if viewer is not None:
                    viewer.sync()
            with env.compute_context():
                all_done = bool((terminal_steps != 0).all().item())
            if all_done:
                break
        with env.compute_context():
            steps = terminal_steps.cpu().tolist()
            successes = succeeded.cpu().tolist()
            falls = fell.cpu().tolist()
        for episode, step, success, fall in zip(episodes, steps, successes, falls):
            print(
                f"episode {episode.index}: success={success}, fell={fall}, "
                f"terminal_step={step if step else '-'}"
            )
    finally:
        if controller is not None:
            # Release ONNX Runtime's CUDA resources before closing the viewer.
            if env.cuda_stream is not None:
                env.cuda_stream.synchronize()
            controller.model.close()
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
    parser.add_argument(
        "--psi-run-dir", type=Path, default=Path("artifacts/psi0-carry")
    )
    parser.add_argument("--ckpt-step", type=int, default=40000)
    parser.add_argument(
        "--qwen-model", type=Path, default=Path("artifacts/qwen3-vl-2b")
    )
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--prompt", help="Override the recorded task instruction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if len(set(args.episode_indices)) != len(args.episode_indices):
        parser.error("--episode-indices must be distinct")
    run(args)


if __name__ == "__main__":
    main()
