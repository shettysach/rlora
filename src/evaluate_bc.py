"""Run the BONES walking instruction through the shared Ψ₀/SONIC/MJLab stack."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from controller.sonic.policy import SonicPolicy
from planner.psi0 import Psi0Planner


def heading(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop BONES walk-and-turn evaluation"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bc_bones_walk.json")
    )
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--sonic-bundle", type=Path, default=Path("artifacts/sonic"))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-video", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/bones-rollout.json")
    )
    args = parser.parse_args()
    if args.num_envs < 1 or args.seconds <= 0:
        parser.error("Environment count and duration must be positive")
    from sim.env import MjlabEnv

    config = json.loads(args.config.read_text())
    torch.manual_seed(config["seed"])
    model = config["model"]
    planner = Psi0Planner(
        Path(model["run_dir"]),
        model["ckpt_step"],
        Path(model["qwen_model"]),
        Path(model["clip_model"]),
        device=args.device,
        inference_steps=model["inference_steps"],
        bc_checkpoint=args.bc_checkpoint,
    )
    env = MjlabEnv(args.num_envs, device=args.device, record_video=args.record_video)
    try:
        with env.compute_context():
            sonic = SonicPolicy(
                args.sonic_bundle,
                args.num_envs,
                device=args.device,
                cuda_stream=env.cuda_stream,
            )
            env.reset()
            state = env.robot_state()
            initial_xy = state.root_pos_w[:, :2].clone()
            previous_xy = initial_xy.clone()
            previous_heading = heading(state.root_quat_w)
            path_length = torch.zeros(args.num_envs, device=args.device)
            turn = torch.zeros_like(path_length)
            max_turn = torch.zeros_like(path_length)
            fell = torch.zeros_like(path_length, dtype=torch.bool)
            actions = None
            cursor = 0.0
            instructions = [config["dataset"]["task"]] * args.num_envs
            for _ in range(math.ceil(args.seconds / env.step_dt)):
                if actions is None or cursor >= planner.horizon:
                    if env.cuda_stream is not None:
                        env.cuda_stream.synchronize()
                    # This synchronous evaluation consumes the entire chunk;
                    # there is no unexecuted prefix to guide the next sample.
                    planner.reset()
                    actions = planner.predict(
                        env.rgb().cpu(), state.joint_pos.cpu(), instructions
                    )
                    cursor = 0.0
                env.step(
                    sonic.act(reference=actions[:, int(cursor)], robot_state=state)
                )
                state = env.robot_state()
                xy = state.root_pos_w[:, :2]
                yaw = heading(state.root_quat_w)
                delta = yaw - previous_heading
                turn += torch.atan2(delta.sin(), delta.cos())
                max_turn = torch.maximum(max_turn, turn.abs())
                path_length += torch.linalg.vector_norm(xy - previous_xy, dim=-1)
                fell |= (state.root_pos_w[:, 2] < 0.5) | (
                    state.projected_gravity_b[:, 2] > -0.7
                )
                previous_xy, previous_heading = xy.clone(), yaw
                cursor += config["dataset"]["action_hz"] * env.step_dt
            result = {
                "instruction": instructions[0],
                "seed": config["seed"],
                "bc_checkpoint": str(args.bc_checkpoint)
                if args.bc_checkpoint
                else None,
                "seconds": args.seconds,
                "action_hz": config["dataset"]["action_hz"],
                "replan_after": planner.horizon,
                "episodes": [
                    {
                        "path_length_m": path_length[index].item(),
                        "displacement_m": torch.linalg.vector_norm(
                            state.root_pos_w[index, :2] - initial_xy[index]
                        ).item(),
                        "net_turn_rad": turn[index].item(),
                        "max_turn_rad": max_turn[index].item(),
                        "fell": fell[index].item(),
                    }
                    for index in range(args.num_envs)
                ],
            }
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
