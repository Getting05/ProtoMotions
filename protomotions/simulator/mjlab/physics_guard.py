# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse learner validity handling with MuJoCo-native slot recovery."""

from protomotions.utils.physics_guard import PhysicsGuard


class MjLabPhysicsGuard(PhysicsGuard):
    def raw(self):
        d = self.sim._robot.data
        return dict(
            joint_vel=d.joint_vel.clone(),
            joint_pos=d.joint_pos.clone(),
            body_lin_vel_w=d.body_link_lin_vel_w.clone(),
            root_pose=d.root_link_pose_w.clone(),
            body_ang_vel_w=d.body_link_ang_vel_w.clone(),
            qpos=self.sim._sim.data.qpos.clone(),
            qvel=self.sim._sim.data.qvel.clone(),
            contact_forces=self.sim._contacts.data.force.clone(),
        )

    def park(self, ids):
        if not ids.numel():
            return
        sim = self.sim
        state = sim.get_default_robot_reset_state()[ids].convert_to_sim(
            sim.data_conversion
        )
        sim._set_simulator_env_state(state, env_ids=ids)
        sim._common_actions[ids] = sim.robot_config.default_dof_pos.to(sim.device)
        sim._previous_actions[ids] = sim._common_actions[ids]
        sim._prev_prev_actions[ids] = sim._common_actions[ids]

    def step(self):
        self.control_step += 1
        self.invalid.copy_(self.retired)
        self.before = self.raw()
        self.inspect_and_recover(self.before, -1)
        for substep in range(self.sim.decimation):
            self.sim._apply_control()
            self.park(self.invalid.nonzero().flatten())
            self.sim._scene.write_data_to_sim()
            self.sim._sim.step()
            self.sim._sim.forward()
            self.inspect_and_recover(self.raw(), substep)
            self.sim._scene.update(1.0 / self.sim.config.sim.fps)
            for callback in self.sim._substep_callbacks:
                callback()
        self.consecutive[~self.invalid] = 0
