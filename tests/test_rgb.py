import os
from types import SimpleNamespace

import mujoco
import pytest
import torch

from sim.env import MjlabEnv

pytestmark = pytest.mark.skipif(
    os.environ.get("MUJOCO_GL") != "egl", reason="Requires MUJOCO_GL=egl"
)


def test_rgb_rebases_worlds_preserves_state_and_refreshes_static_poses():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <visual><global offwidth="640" offheight="360"/></visual>
          <worldbody>
            <light pos="0 0 3"/>
            <geom type="plane" size="5 5 .1"/>
            <body pos="0 0 1">
              <freejoint/><geom type="sphere" size=".05"/>
              <camera name="observation_camera" xyaxes="1 0 0 0 0 1"/>
            </body>
            <body pos="0 1 .6">
              <freejoint/><geom type="box" size=".15 .15 .15" rgba="1 0 0 1"/>
            </body>
            <body mocap="true" pos="0 1 .3">
              <geom type="box" size=".4 .4 .1" rgba="0 1 0 1"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    initial = mujoco.MjData(model)
    origins = torch.tensor([[0, 0, 0], [4, 4, 0]], dtype=torch.float32)
    qpos = torch.tensor(initial.qpos, dtype=torch.float32).repeat(2, 1)
    qpos[:, :3] += origins
    qpos[:, 7:10] += origins
    mocap = torch.tensor(initial.mocap_pos, dtype=torch.float32).repeat(2, 1, 1)
    mocap += origins[:, None]
    data = SimpleNamespace(
        qpos=qpos,
        mocap_pos=mocap,
        mocap_quat=torch.tensor(initial.mocap_quat).repeat(2, 1, 1),
    )
    env = object.__new__(MjlabEnv)
    env.cuda_stream = None
    env._renderer = None
    env._render_data = initial
    env._render_static = None
    env.env = SimpleNamespace(
        num_envs=2,
        sim=SimpleNamespace(mj_model=model, data=data),
        scene=SimpleNamespace(env_origins=origins),
        reset=lambda seed: data.mocap_pos.add_(torch.tensor([0, 0, 0.5])),
    )
    try:
        before_qpos = data.qpos.clone()
        before_mocap = data.mocap_pos.clone()
        first = env.rgb()
        assert first.shape == (2, 360, 640, 3)
        assert first.dtype == torch.uint8 and first.device.type == "cpu"
        torch.testing.assert_close(first[0], first[1], rtol=0, atol=0)
        saved = first.clone()
        torch.testing.assert_close(data.qpos, before_qpos)
        torch.testing.assert_close(data.mocap_pos, before_mocap)
        static = env._render_static

        data.qpos[1, 7] += 0.3
        second = env.rgb()
        assert env._render_static is static
        assert not torch.equal(second[0], second[1])
        torch.testing.assert_close(first, saved, rtol=0, atol=0)

        env.reset()
        third = env.rgb()
        assert env._render_static is not static
        assert not torch.equal(second[0], third[0])
    finally:
        if env._renderer is not None:
            env._renderer.close()
