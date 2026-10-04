import pytest
import torch

from carry_box import CarryBoxTask
from sim.config import make_env_cfg
from sim.env import MjlabEnv


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
    ],
)
def test_physics_step_matches_mjlab_without_reset_index_discovery(monkeypatch, device):
    def physics_scene(count):
        cfg = make_env_cfg(count)
        # Keep the actual robot/actuators/box/table; omit the large visual room.
        del cfg.scene.entities["room"]
        return cfg

    monkeypatch.setattr("sim.env.make_env_cfg", physics_scene)
    env = MjlabEnv(2, device=device)
    reference = MjlabEnv(2, device=device)
    tolerances = {"rtol": 0, "atol": 0} if device == "cpu" else {}

    def unexpected_nonzero(*args, **kwargs):
        pytest.fail("Physics stepping must not discover reset indices")

    try:
        # Both simulations share the Warp device stream; enqueue checks there too.
        with env.compute_context():
            task = CarryBoxTask(env)
            reference_task = CarryBoxTask(reference)
            initial = env.env.sim.data.qpos.clone()
            for step in range(4):
                if step == 2:
                    for simulation, outcomes in (
                        (env, task),
                        (reference, reference_task),
                    ):
                        simulation.reset(seed=1)
                        outcomes.reset()
                body = torch.linspace(-0.05, 0.05, 58, device=device).reshape(2, 29)
                hands = torch.linspace(0, 0.1, 28, device=device).reshape(2, 14)
                body *= step + 1
                reference.env.step(
                    torch.cat(
                        (body, hands.index_select(-1, reference.mjlab_hand_from_psi0)),
                        dim=-1,
                    )
                )
                reference_task.update()
                with monkeypatch.context() as guard:
                    guard.setattr(torch.Tensor, "nonzero", unexpected_nonzero)
                    guard.setattr(torch, "nonzero", unexpected_nonzero)
                    env.step(body, hands)
                    task.update()

                for field in (
                    "qpos",
                    "qvel",
                    "ctrl",
                    "mocap_pos",
                    "mocap_quat",
                    "time",
                ):
                    torch.testing.assert_close(
                        getattr(env.env.sim.data, field).clone(),
                        getattr(reference.env.sim.data, field).clone(),
                        **tolerances,
                    )
                for field in ("action", "prev_action", "prev_prev_action"):
                    torch.testing.assert_close(
                        getattr(env.env.action_manager, field),
                        getattr(reference.env.action_manager, field),
                        **tolerances,
                    )
                state, reference_state = env.robot_state(), reference.robot_state()
                for field in (
                    "root_ang_vel_b",
                    "projected_gravity_b",
                    "joint_pos",
                    "joint_vel",
                ):
                    torch.testing.assert_close(
                        getattr(state, field),
                        getattr(reference_state, field),
                        **tolerances,
                    )
                for field in ("placement_time_s", "success", "fell"):
                    torch.testing.assert_close(
                        getattr(task, field), getattr(reference_task, field)
                    )
                torch.testing.assert_close(
                    env.box_pose(), reference.box_pose(), **tolerances
                )
                torch.testing.assert_close(
                    env.planner_state(), reference.planner_state(), **tolerances
                )
                torch.testing.assert_close(
                    env.env.episode_length_buf, reference.env.episode_length_buf
                )
                assert env.env._sim_step_counter == reference.env._sim_step_counter
                assert env.env.common_step_counter == reference.env.common_step_counter
            assert not torch.equal(initial, env.env.sim.data.qpos.clone())
    finally:
        reference.close()
        env.close()
