# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real GPU physics contract checks. Run from the repository with python -m."""

import argparse
import json
from types import SimpleNamespace
import numpy as np
import mujoco
import torch
from protomotions.robot_configs.astro_p2 import AstroP2RobotConfig
from protomotions.simulator.mjlab.config import MjLabSimulatorConfig
from protomotions.simulator.mjlab.simulator import MjLabSimulator
from protomotions.simulator.base_simulator.simulator_state import StateConversion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--motion-output", default=None)
    args = p.parse_args()
    device = torch.device("cuda:0")
    robot = AstroP2RobotConfig()
    scene = SimpleNamespace(num_scenes=lambda: 0, num_objects_per_scene=0)
    terrain = SimpleNamespace(sim_config=SimpleNamespace(static_friction=1.0))
    sim = MjLabSimulator(
        MjLabSimulatorConfig(
            num_envs=args.num_envs,
            headless=True,
            experiment_name="mjlab_contract_smoke",
        ),
        robot,
        terrain,
        device,
        scene,
    )
    sim._initialize_with_markers(None)
    state = sim.get_default_robot_reset_state()
    sim.reset_envs(state)
    if args.motion_output:
        from pathlib import Path

        sample = sim.get_robot_state()
        motion = {"fps": 50, "state_conversion": StateConversion.COMMON}
        for name in (
            "rigid_body_pos",
            "rigid_body_rot",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "dof_pos",
            "dof_vel",
        ):
            value = getattr(sample, name)[0].detach().cpu()
            motion[name] = value.unsqueeze(0).repeat((300,) + (1,) * value.ndim)
        # Synthetic support labels for lifecycle tests, not measured motion data.
        motion["rigid_body_contacts"] = torch.zeros(
            300, robot.kinematic_info.num_bodies, dtype=torch.bool
        )
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            motion["rigid_body_contacts"][:, sim._body_names.index(name)] = True
        Path(args.motion_output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(motion, args.motion_output)
    measured = sim.get_root_state()
    torch.testing.assert_close(measured.root_pos, state.root_pos, atol=1e-5, rtol=0)
    # Round trip nonzero world angular velocity at rotated base orientation.
    state.root_rot[:] = torch.tensor([0, 0, 0.70710678, 0.70710678], device=device)
    state.root_ang_vel[:] = torch.tensor([0.2, -0.3, 0.4], device=device)
    sim.reset_envs(state)
    torch.testing.assert_close(
        sim.get_root_state().root_ang_vel, state.root_ang_vel, atol=2e-5, rtol=0
    )
    # Compare link-origin body pose and velocity to independent CPU MuJoCo.
    md = mujoco.MjData(sim._sim.mj_model)
    md.qpos[:] = sim._sim.data.qpos[0].cpu().numpy()
    md.qvel[:] = sim._sim.data.qvel[0].cpu().numpy()
    mujoco.mj_forward(sim._sim.mj_model, md)
    body = sim._get_simulator_bodies_state()
    for i, name in enumerate(sim._robot.body_names):
        bid = sim._sim.mj_model.body("robot/" + name).id
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            sim._sim.mj_model, md, mujoco.mjtObj.mjOBJ_XBODY, bid, vel, 0
        )
        np.testing.assert_allclose(
            body.rigid_body_pos[0, i].cpu().numpy(), md.xpos[bid], atol=2e-5
        )
        np.testing.assert_allclose(
            body.rigid_body_vel[0, i].cpu().numpy(), vel[3:], atol=2e-5
        )
    before = sim._sim.data.qpos[1:].clone()
    sim.reset_envs(
        sim.get_default_robot_reset_state()[torch.tensor([0], device=device)],
        env_ids=torch.tensor([0], device=device),
    )
    torch.testing.assert_close(sim._sim.data.qpos[1:], before, atol=0, rtol=0)
    sim.reset_envs(sim.get_default_robot_reset_state())
    calls = [0]
    sim.register_post_physics_substep(lambda: calls.__setitem__(0, calls[0] + 1))
    actions = robot.default_dof_pos.to(device).repeat(args.num_envs, 1)
    for _ in range(args.steps):
        sim.step(actions)
        assert torch.isfinite(sim._sim.data.qpos).all()
        assert torch.isfinite(
            sim.get_bodies_contact_buf().rigid_body_contact_forces
        ).all()
    assert calls[0] == args.steps * 4
    snapshot = sim.save_runtime_state()
    sim.step(actions)
    sim.load_runtime_state(snapshot)
    torch.testing.assert_close(
        sim._sim.data.qpos.clone(), snapshot["data"]["qpos"], atol=0, rtol=0
    )
    from protomotions.simulator.mjlab.physics_guard import MjLabPhysicsGuard
    import tempfile

    with tempfile.TemporaryDirectory(prefix="mjlab-guard-") as folder:
        guard = MjLabPhysicsGuard(sim, folder=folder)
        sim._physics_guard = guard
        sim._sim.data.qvel[0, 6] = float("nan")
        sim.step(actions)
        assert guard.invalid[0] and not guard.invalid[1:].any()
        assert torch.isfinite(sim._sim.data.qvel).all()
        del sim._physics_guard
    print(
        "MJLAB_PHYSICS_SMOKE_PASS "
        + json.dumps(
            dict(
                num_envs=args.num_envs,
                steps=args.steps,
                contacts=float(
                    sim.get_bodies_contact_buf()
                    .rigid_body_contact_forces.norm(dim=-1)
                    .max()
                ),
                min_height=float(sim.get_root_state().root_pos[:, 2].min()),
            )
        )
    )
    sim.close()


if __name__ == "__main__":
    main()
