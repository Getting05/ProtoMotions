"""Stateful P2 point-IMU emulation with unfiltered joint velocity.

Only this independent environment uses it. Velocities are simulator inputs to
sensor emulation, never extra policy channels. Acceleration is a 5 ms finite
difference approximation of the deployed point accelerometer.
"""
from __future__ import annotations

import json
from pathlib import Path
import torch
from protomotions.utils import rotations
from protomotions.envs.obs.p2_reduced_coord import pack_p2_sensors

CONTRACT = json.loads(Path(__file__).with_name('p2_deploy_contract.json').read_text())


def pack_deploy_sensors(dof_pos, dof_vel, imu_quat, imu_gyro, imu_accel):
    """Match Eigen normalization/XYZW coefficients; preserve SDK quaternion sign."""
    return pack_p2_sensors(dof_pos, dof_vel, imu_quat, imu_gyro, imu_accel,
                           list(range(29)), canonicalize_quaternion=False)


def imu_specific_force(root_rot, root_vel, root_ang_vel, previous_vel,
                       previous_ang_vel, dt, imu_position_base):
    r = rotations.quat_rotate(root_rot, imu_position_base.expand_as(root_vel), True)
    acceleration = ((root_vel - previous_vel) / dt
                    + torch.cross((root_ang_vel - previous_ang_vel) / dt, r, dim=-1)
                    + torch.cross(root_ang_vel, torch.cross(root_ang_vel, r, dim=-1), dim=-1))
    acceleration = acceleration + acceleration.new_tensor([0., 0., 9.81])
    return rotations.quat_rotate_inverse(root_rot, acceleration, True)


class P2DeploySensorBank:
    def __init__(self, num_envs, history_steps, device):
        self.dt = CONTRACT['physics_dt']
        self.position = torch.tensor(CONTRACT['imu_position_base'], device=device)
        self.ids = list(range(29))
        self.previous_vel = torch.zeros(num_envs, 3, device=device)
        self.previous_ang_vel = torch.zeros_like(self.previous_vel)
        self.current = torch.zeros(num_envs, 71, device=device)
        self.history = torch.zeros(num_envs, history_steps + 1, 71, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.sample_count = 0

    def seed(self, state, ids):
        """Reset only selected sensors and their finite-difference history."""
        self.previous_vel[ids] = state.root_vel
        self.previous_ang_vel[ids] = state.root_ang_vel
        accel = imu_specific_force(state.root_rot, state.root_vel, state.root_ang_vel,
                                   state.root_vel, state.root_ang_vel, self.dt, self.position)
        gyro = rotations.quat_rotate_inverse(state.root_rot, state.root_ang_vel, True)
        obs = pack_deploy_sensors(state.dof_pos, state.dof_vel, state.root_rot, gyro, accel)
        self.current[ids] = obs
        self.history[ids] = obs[:, None]
        self.valid[ids] = True

    def sample(self, state):
        # Joint velocity is used directly, without filtering or smoothing.
        accel = imu_specific_force(state.root_rot, state.root_vel, state.root_ang_vel,
                                   self.previous_vel, self.previous_ang_vel, self.dt, self.position)
        gyro = rotations.quat_rotate_inverse(state.root_rot, state.root_ang_vel, True)
        self.current.copy_(pack_deploy_sensors(state.dof_pos, state.dof_vel, state.root_rot,
                                             gyro, accel))
        self.previous_vel.copy_(state.root_vel)
        self.previous_ang_vel.copy_(state.root_ang_vel)
        self.valid.fill_(True)
        self.sample_count += 1

    def commit(self):
        self.history[:, 1:] = self.history[:, :-1].clone()
        self.history[:, 0] = self.current

    def state_dict(self):
        return {'version': CONTRACT['contract_version'], 'sample_count': self.sample_count,
                **{name: getattr(self, name).clone() for name in (
                    'previous_vel', 'previous_ang_vel', 'current', 'history', 'valid')}}

    def load_state_dict(self, state):
        if state['version'] != CONTRACT['contract_version']:
            raise ValueError('Incompatible P2 deployment sensor checkpoint')
        for name in ('previous_vel', 'previous_ang_vel', 'current', 'history', 'valid'):
            target = getattr(self, name)
            if target.shape != state[name].shape:
                raise ValueError(f'P2 sensor resume shape mismatch: {name}')
            target.copy_(state[name].to(target.device))
        self.sample_count = state['sample_count']


def current_deploy_obs(sensor_obs, motor_dof_ids, include_projected_gravity=True,
                       contract_version=3, quaternion_sign='preserve'):
    return sensor_obs if include_projected_gravity else sensor_obs[:, :68]


def historical_deploy_obs(sensor_history, history_steps):
    return sensor_history[:, history_steps].flatten(1)


def reference_deploy_obs(motion_lib, motion_ids, motion_times, num_state_history_steps,
                         dt, motor_dof_ids, history_steps, **kwargs):
    """AMP experts: use the same 200 Hz point IMU and raw joint velocities.

    Chunk samples to bound full-body interpolation workspace. One preceding
    physics sample supplies acceleration differences for the oldest requested
    observation; no filter state or filter warmup is used.
    """
    from types import SimpleNamespace
    if abs(dt - CONTRACT['inference_dt']) > 1e-8 or motor_dof_ids != list(range(29)):
        raise ValueError('P2 deployment AMP requires 50 Hz and matching DOF order')
    factor = round(dt / CONTRACT['physics_dt'])
    count = max(history_steps) * factor + 1
    outputs = []
    for start in range(0, len(motion_ids), 32):
        ids = motion_ids[start:start+32]
        times = motion_times[start:start+32]
        offsets = torch.arange(count, -1, -1, device=ids.device) * CONTRACT['physics_dt']
        sample_times = (times[:, None] - offsets).clamp(min=0.)
        lengths = getattr(motion_lib, 'motion_lengths', None)
        if lengths is not None:
            sample_times = torch.minimum(sample_times, lengths[ids, None].to(sample_times))
        flat = motion_lib.get_motion_state(ids[:, None].expand_as(sample_times).reshape(-1),
                                          sample_times.reshape(-1))
        batch, ticks = sample_times.shape
        q = flat.dof_pos.reshape(batch, ticks, -1)
        dq = flat.dof_vel.reshape(batch, ticks, -1)
        rot = flat.rigid_body_rot.reshape(batch, ticks, -1, 4)[:, :, 0]
        vel = flat.rigid_body_vel.reshape(batch, ticks, -1, 3)[:, :, 0]
        omega = flat.rigid_body_ang_vel.reshape(batch, ticks, -1, 3)[:, :, 0]
        def state(i):
            return SimpleNamespace(dof_pos=q[:, i], dof_vel=dq[:, i], root_rot=rot[:, i],
                                   root_vel=vel[:, i], root_ang_vel=omega[:, i])
        bank = P2DeploySensorBank(batch, 1, ids.device)
        bank.seed(state(0), torch.arange(batch, device=ids.device))
        selected = {}
        wanted = {count - step * factor: step for step in history_steps}
        for i in range(1, ticks):
            bank.sample(state(i))
            if i in wanted:
                selected[wanted[i]] = bank.current.clone()
        outputs.append(torch.stack([selected[step] for step in history_steps], 1).flatten(1))
    return torch.cat(outputs, 0)
