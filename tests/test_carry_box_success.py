from types import SimpleNamespace

import mujoco
import pytest
import torch

from carry_box import CarryBoxTask


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
def test_success_matches_simple_contact_rule_and_python_accumulation(
    monkeypatch, device
):
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <body name="box"><geom name="box/collision" size="0.1" /></body>
          <body name="table"><geom name="table/collision" size="0.1" /></body>
          <body name="robot/hand"><geom name="hand" size="0.1" /></body>
        </worldbody></mujoco>
    """)
    box, table, hand = (
        model.geom(name).id for name in ("box/collision", "table/collision", "hand")
    )
    # Valid placement, center-height threshold, contact-height boundaries, hands
    # still touching, a registered distant hand contact, and no table contact.
    count = 8
    geometries = [(box, table)] * 7 + [(hand, box), (box, hand), (table, box)]
    worlds = [0, 1, 2, 3, 4, 5, 6, 5, 6, 99]  # Last slot is inactive garbage.
    positions = torch.zeros(len(worlds), 3, device=device)
    positions[:, 2] = torch.tensor(
        [0.449, 0.4, 0.35, 0.45, 0.451, 0.4, 0.4, 0.4, 0.4, 0.4], device=device
    )
    contacts = SimpleNamespace(
        geom=torch.tensor(geometries, device=device),
        worldid=torch.tensor(worlds, device=device),
        pos=positions,
        dist=torch.full((len(worlds),), 0.02, device=device),
    )
    data = SimpleNamespace(contact=contacts, nacon=torch.tensor([9], device=device))
    poses = torch.zeros(count, 7, device=device)
    poses[:, 2] = 0.67
    poses[1, 2] = 0.41  # Above table center, below its physical top.
    roots = torch.zeros(count, 3, device=device)
    roots[:, 2] = 0.8
    env = SimpleNamespace(
        device=torch.device(device),
        step_dt=0.02,
        box_pose=lambda: poses,
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=roots)),
        env=SimpleNamespace(sim=SimpleNamespace(mj_model=model, wp_data=data)),
    )
    monkeypatch.setattr("carry_box.wp.to_torch", lambda value: value)
    task = CarryBoxTask(env)
    elapsed = [0.0] * count
    succeeded = [False] * count
    for step in range(70):
        # Pause placement for five steps; accumulated time must not reset.
        positions[0, 2] = 0.2 if 15 <= step < 20 else 0.449
        hand_contact = [False] * count
        table_any = [False] * count
        table_contact = [False] * count
        for (first, second), world, pos in zip(
            contacts.geom[:9].tolist(),
            contacts.worldid[:9].tolist(),
            positions[:9].tolist(),
        ):
            # Independent scalar reference: SIMPLE scans registered contacts,
            # casts their height to Python float, and ignores contact distance.
            hand_contact[world] |= {first, second} == {box, hand}
            table_any[world] |= {first, second} == {box, table}
            table_contact[world] |= {first, second} == {box, table} and abs(
                float(pos[2]) - 0.4
            ) <= 0.05
        placed = [
            table_contact[world] and not hand_contact[world] and pose[2] >= 0.4
            for world, pose in enumerate(poses.tolist())
        ]
        for world in range(count):
            if placed[world]:
                elapsed[world] += 0.02
            succeeded[world] |= elapsed[world] > 0.9
        task.update()
        assert task.hand_contact.tolist() == hand_contact
        assert task.table_contact.tolist() == table_any
        assert task.table_height_contact.tolist() == table_contact
        assert task.placed.tolist() == placed
        torch.testing.assert_close(
            task.placement_time_s,
            torch.tensor(elapsed, dtype=torch.float64, device=device),
            rtol=0,
            atol=0,
        )
        assert task.success.tolist() == succeeded
        if step == 44:
            assert task.success[1].item()  # Python accumulation passes on step 45.
            assert not task.success[0].item()  # Five missing placement steps.
