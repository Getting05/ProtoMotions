# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched GPU MuJoCo backend retaining ProtoMotions' control/state contracts."""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import torch
import warp as wp
from mjlab.scene import Scene, SceneCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg

from protomotions.assets import resolve_asset_root
from protomotions.robot_configs.base import ControlType
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import SimBodyOrdering
from protomotions.simulator.base_simulator.simulator_state import (
    RootOnlyState,
    RobotState,
    ObjectState,
    StateConversion,
)


class MjLabSimulator(Simulator):
    def __init__(self, config, robot_config, terrain, device, scene_lib):
        if device.type != "cuda":
            raise ValueError("mjlab training requires a CUDA device")
        if scene_lib.num_scenes():
            raise NotImplementedError(
                "mjlab backend currently requires an empty SceneLib"
            )
        if not config.headless:
            raise ValueError(
                "Use headless mjlab training; offscreen evaluation is supported"
            )
        if config.projectile.num_projectiles > 0:
            raise NotImplementedError(
                "mjlab projectile perturbations are not implemented"
            )
        if (
            terrain is not None
            and hasattr(terrain, "config")
            and any(terrain.config.terrain_proportions[:-1])
        ):
            raise NotImplementedError("mjlab currently supports flat terrain only")
        self._substep_callbacks = []
        self._renderer = None
        super().__init__(config, robot_config, terrain, device, scene_lib)

    def register_post_physics_substep(self, callback):
        self._substep_callbacks.append(callback)

    def _robot_spec(self):
        path = (
            Path(resolve_asset_root(self.robot_config.asset.asset_root))
            / self.robot_config.asset.asset_file_name
        )
        tree = ET.parse(path)
        root = tree.getroot()
        for tag in ("sensor", "actuator", "keyframe", "size"):
            for element in root.findall(tag):
                root.remove(element)
        if self.robot_config.asset.replace_cylinder_with_capsule:
            for geom in root.findall(".//geom"):
                if geom.get("type") == "cylinder":
                    geom.set("type", "capsule")
        actuators = ET.SubElement(root, "actuator")
        implicit = self.control_type == ControlType.BUILT_IN_PD
        for joint in root.findall(".//worldbody//joint"):
            name = joint.get("name")
            if name not in self.robot_config.control.control_info:
                continue
            info = self.robot_config.control.control_info[name]
            joint.set("stiffness", "0")
            joint.set("damping", "0")
            joint.set("frictionloss", str(info.friction or 0.0))
            joint.set("armature", str(info.armature or 0.0))
            attrs = dict(
                name=name + "_mjlab",
                joint=name,
                gear="1",
                forcelimited="true",
                forcerange=f"{-info.effort_limit} {info.effort_limit}",
            )
            if implicit:
                attrs.update(kp=str(info.stiffness), kv=str(info.damping))
            ET.SubElement(actuators, "position" if implicit else "motor", attrs)
        # Resolve meshes relative to the original asset, without changing its XML.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=path.parent, delete=False
        ) as f:
            f.write(ET.tostring(root, encoding="unicode"))
            tmp = f.name
        try:
            return mujoco.MjSpec.from_file(tmp)
        finally:
            os.unlink(tmp)

    def _create_simulation(self):
        torch.cuda.set_device(self.device)
        wp.init()
        with wp.ScopedDevice(str(self.device)):
            robot = EntityCfg(
                spec_fn=self._robot_spec,
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        XmlActuatorCfg(target_names_expr=tuple(self._dof_names)),
                    )
                ),
            )
            sensor = ContactSensorCfg(
                name="body_contacts",
                primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
                fields=("force", "found"),
                reduce="netforce",
            )
            self._scene = Scene(
                SceneCfg(
                    num_envs=self.num_envs, entities={"robot": robot}, sensors=(sensor,)
                ),
                device=str(self.device),
            )
            self._scene.spec.worldbody.add_geom(
                name="ground",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=(0, 0, 0.1),
                friction=(self.terrain.sim_config.static_friction, 0.005, 0.0001),
            )
            p = self.config.sim
            self._sim = Simulation(
                self.num_envs,
                SimulationCfg(
                    nconmax=p.nconmax,
                    njmax=p.njmax,
                    contact_sensor_maxmatch=p.contact_sensor_maxmatch,
                    mujoco=MujocoCfg(
                        timestep=1.0 / p.fps,
                        integrator=p.integrator,
                        iterations=p.iterations,
                        ls_iterations=p.ls_iterations,
                        cone=p.cone,
                    ),
                ),
                spec=self._scene.spec,
                device=str(self.device),
            )
            self._scene.initialize(
                mj_model=self._sim.mj_model, model=self._sim.model, data=self._sim.data
            )
        self._robot = self._scene.entities["robot"]
        self._contacts = self._scene.sensors["body_contacts"]
        names = [n.removeprefix("robot/") for n in self._contacts.primary_names]
        self._contact_order = torch.tensor(
            [names.index(n) for n in self._robot.body_names], device=self.device
        )
        self._apply_materials()
        self._sim.forward()

    def _apply_materials(self):
        dr = self._domain_randomization or {}
        unsupported = set(dr) - {"action_noise", "center_of_mass", "friction"}
        if unsupported:
            raise NotImplementedError(
                f"Unsupported mjlab physical randomization: {unsupported}"
            )
        self._sim.expand_model_fields(("geom_friction",))
        ids = self._robot.indexing.geom_ids
        self._sim.model.geom_friction[:, ids, 0] = self.config.default_robot_friction
        if "friction" in dr:
            # PhysX two-coefficient/restitution mappings need an explicit contract.
            raise NotImplementedError(
                "Use mjlab material configuration; PhysX friction/restitution DR is not equivalent"
            )
        if "center_of_mass" in dr:
            self._sim.expand_model_fields(("body_ipos",))
            common_ids = dr["center_of_mass"]["body_indices"]
            names = [self._body_names[i] for i in common_ids]
            body_ids = [self._sim.mj_model.body("robot/" + n).id for n in names]
            self._sim.model.body_ipos[:, body_ids] += dr["center_of_mass"]["com"].to(
                self.device
            )
            from mjlab.managers.event_manager import RecomputeLevel

            self._sim.recompute_constants(RecomputeLevel.set_const)

    def _get_sim_body_ordering(self):
        return SimBodyOrdering(
            body_names=list(self._robot.body_names),
            dof_names=list(self._robot.joint_names),
        )

    def _physics_step(self):
        guard = getattr(self, "_physics_guard", None)
        if guard is not None:
            return guard.step()
        for _ in range(self.decimation):
            self._apply_control()
            self._scene.write_data_to_sim()
            self._sim.step()
            self._sim.forward()
            self._scene.update(1.0 / self.config.sim.fps)
            for callback in self._substep_callbacks:
                callback()

    def _set_simulator_env_state(
        self, new_states, new_object_states=None, env_ids=None
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._sim.reset(env_ids)
        self._scene.reset(env_ids)
        root = torch.cat(
            (
                new_states.root_pos,
                new_states.root_rot,
                new_states.root_vel,
                new_states.root_ang_vel,
            ),
            -1,
        )
        self._robot.write_root_state_to_sim(root, env_ids)
        self._robot.write_joint_state_to_sim(
            new_states.dof_pos, new_states.dof_vel, env_ids=env_ids
        )
        self._robot.set_joint_position_target(new_states.dof_pos, env_ids=env_ids)
        self._robot.set_joint_effort_target(
            torch.zeros_like(new_states.dof_pos), env_ids=env_ids
        )
        self._common_actions[env_ids] = 0
        self._scene.write_data_to_sim()
        self._sim.forward()

    def _get_simulator_root_state(self, env_ids=None):
        d = self._robot.data
        s = slice(None) if env_ids is None else env_ids
        return RootOnlyState(
            root_pos=d.root_link_pos_w[s].clone(),
            root_rot=d.root_link_quat_w[s].clone(),
            root_vel=d.root_link_lin_vel_w[s].clone(),
            root_ang_vel=d.root_link_ang_vel_w[s].clone(),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_bodies_state(self, env_ids=None):
        d = self._robot.data
        s = slice(None) if env_ids is None else env_ids
        return RobotState(
            rigid_body_pos=d.body_link_pos_w[s].clone(),
            rigid_body_rot=d.body_link_quat_w[s].clone(),
            rigid_body_vel=d.body_link_lin_vel_w[s].clone(),
            rigid_body_ang_vel=d.body_link_ang_vel_w[s].clone(),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_state(self, env_ids=None):
        s = slice(None) if env_ids is None else env_ids
        return RobotState(
            dof_pos=self._robot.data.joint_pos[s].clone(),
            dof_vel=self._robot.data.joint_vel[s].clone(),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_forces(self, env_ids=None):
        s = slice(None) if env_ids is None else env_ids
        return RobotState(
            dof_forces=self._robot.data.qfrc_actuator[s].clone(),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_bodies_contact_buf(self, env_ids=None):
        s = slice(None) if env_ids is None else env_ids
        force = self._contacts.data.force[:, self._contact_order].reshape(
            self.num_envs, -1, 3
        )
        return RobotState(
            rigid_body_contact_forces=force[s].clone(),
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_limits_for_verification(self):
        m = self._sim.mj_model
        limits = torch.tensor(
            [m.joint("robot/" + n).range.tolist() for n in self._robot.joint_names],
            device=self.device,
        )
        return limits[:, 0], limits[:, 1]

    def _apply_simulator_pd_targets(self, pd_targets):
        self._robot.set_joint_position_target(pd_targets)

    def _apply_simulator_torques(self, torques):
        self._robot.set_joint_effort_target(torques)

    def _apply_root_velocity_impulse(self, linear_velocity, angular_velocity, env_ids):
        vel = self._robot.data.root_link_vel_w[env_ids].clone()
        vel[:, :3] += linear_velocity
        vel[:, 3:] += angular_velocity
        self._robot.write_root_link_velocity_to_sim(vel, env_ids)
        self._sim.forward()

    def park_envs(self, env_ids, *args, **kwargs):
        # Worlds do not collide; avoid unbounded freefall below the plane.
        return None

    def _get_simulator_object_root_state(self, env_ids=None):
        return ObjectState(state_conversion=StateConversion.SIMULATOR)

    def _get_simulator_object_contact_buf(self, env_ids=None):
        return ObjectState(state_conversion=StateConversion.SIMULATOR)

    def _create_projectiles(self, config):
        if config.num_projectiles:
            raise NotImplementedError("mjlab projectiles")

    def _set_projectile_root_states(self, *args, **kwargs):
        raise NotImplementedError("mjlab projectiles")

    def _get_projectile_positions_rotations(self):
        raise NotImplementedError("mjlab projectiles")

    def _init_camera(self):
        pass

    def _update_simulator_markers(self, markers_state=None):
        pass

    def _write_viewport_to_file(self, file_name):
        import imageio.v3 as iio

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._sim.mj_model, height=720, width=1280)
        data = mujoco.MjData(self._sim.mj_model)
        data.qpos[:] = self._sim.data.qpos[0].cpu().numpy()
        data.qvel[:] = self._sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(self._sim.mj_model, data)
        self._renderer.update_scene(data)
        iio.imwrite(file_name, self._renderer.render())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        self._simulation_running = False

    def save_runtime_state(self):
        fields = (
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "qacc_warmstart",
            "time",
            "qfrc_applied",
            "xfrc_applied",
        )
        return {
            "backend": "mjlab",
            "version": 1,
            "num_envs": self.num_envs,
            "data": {n: getattr(self._sim.data, n).clone() for n in fields},
            "actions": {
                n: getattr(self, n).clone()
                for n in (
                    "_common_actions",
                    "_previous_actions",
                    "_prev_prev_actions",
                    "_steps_since_reset",
                )
            },
            "targets": {
                n: getattr(self._robot.data, n).clone()
                for n in ("joint_pos_target", "joint_vel_target", "joint_effort_target")
            },
        }

    def load_runtime_state(self, state):
        if (state["backend"], state["version"], state["num_envs"]) != (
            "mjlab",
            1,
            self.num_envs,
        ):
            raise ValueError(
                "Incompatible mjlab runtime state; use a new warm-start experiment"
            )
        for n, v in state["data"].items():
            getattr(self._sim.data, n).copy_(v.to(self.device))
        for n, v in state["actions"].items():
            getattr(self, n).copy_(v.to(self.device))
        for n, v in state["targets"].items():
            getattr(self._robot.data, n).copy_(v.to(self.device))
        self._sim.forward()
