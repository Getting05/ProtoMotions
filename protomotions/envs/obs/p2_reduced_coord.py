"""P2 motor + pelvis IMU observations (SI units, XYZW quaternion).

The simulated accelerometer measures specific force R^-1 (dv/dt - g),
averaged over a control interval, at the root origin with aligned axes.
It is not a hardware IMU model with mounting offsets, bias or latency.
Only the 68/71 sensor channels are exposed to the networks; simulator
velocities are used internally to emulate acceleration, never concatenated.
"""

from __future__ import annotations

import torch
from torch import Tensor
from protomotions.utils import rotations


def pack_p2_sensors(dof_pos: Tensor, dof_vel: Tensor, imu_quat: Tensor,
                    imu_gyro: Tensor, imu_accel: Tensor, motor_dof_ids: list[int],
                    include_projected_gravity: bool = True,
                    canonicalize_quaternion: bool = True) -> Tensor:
    """Hardware-facing packer; q/dq follow model DOF order, gyro/accel body frame.

    Layout: q[29], dq[29], quat[4], gyro[3], accel[3], gravity[3 optional].
    Quaternion is normalized; canonicalize_quaternion optionally fixes w >= 0.
    The V2 deployment packer disables sign canonicalization to match Eigen/SDK.
    """
    quat = torch.nn.functional.normalize(imu_quat, dim=-1)
    if canonicalize_quaternion:
        quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    parts = [dof_pos[..., motor_dof_ids], dof_vel[..., motor_dof_ids],
             quat, imu_gyro, imu_accel]
    if include_projected_gravity:
        gravity = torch.zeros_like(imu_gyro)
        gravity[..., 2] = -1.0
        parts.append(rotations.quat_rotate_inverse(quat, gravity, True))
    return torch.cat(parts, dim=-1)


def compute_p2_reduced_coord_obs(dof_pos: Tensor, dof_vel: Tensor,
                                root_rot: Tensor, root_ang_vel: Tensor,
                                root_vel: Tensor, historical_body_vel: Tensor,
                                progress_buf: Tensor, dt: float,
                                motor_dof_ids: list[int],
                                include_projected_gravity: bool = True) -> Tensor:
    acceleration = (root_vel - historical_body_vel[:, 0, 0]) / dt
    # Reset teleport/reference history must not create a fictitious IMU impulse.
    acceleration = torch.where((progress_buf > 0).unsqueeze(-1),
                               acceleration, torch.zeros_like(acceleration))
    acceleration = acceleration + acceleration.new_tensor([0.0, 0.0, 9.81])
    gyro = rotations.quat_rotate_inverse(root_rot, root_ang_vel, True)
    accel = rotations.quat_rotate_inverse(root_rot, acceleration, True)
    return pack_p2_sensors(dof_pos, dof_vel, root_rot, gyro, accel,
                           motor_dof_ids, include_projected_gravity)


def compute_p2_historical_reduced_coord_obs(
    historical_dof_pos: Tensor, historical_dof_vel: Tensor,
    historical_body_rot: Tensor, historical_body_vel: Tensor,
    historical_body_ang_vel: Tensor, dt: float, motor_dof_ids: list[int],
    history_steps: list[int], include_projected_gravity: bool = True,
    progress_buf: Tensor = None,
) -> Tensor:
    frames = []
    for step in history_steps:
        i = step - 1
        quat = historical_body_rot[:, i, 0]
        acceleration = (historical_body_vel[:, i, 0] - historical_body_vel[:, i + 1, 0]) / dt
        if progress_buf is not None:
            acceleration = torch.where((progress_buf > step).unsqueeze(-1),
                                       acceleration, torch.zeros_like(acceleration))
        acceleration = acceleration + acceleration.new_tensor([0.0, 0.0, 9.81])
        frames.append(pack_p2_sensors(
            historical_dof_pos[:, i], historical_dof_vel[:, i], quat,
            rotations.quat_rotate_inverse(quat, historical_body_ang_vel[:, i, 0], True),
            rotations.quat_rotate_inverse(quat, acceleration, True),
            motor_dof_ids, include_projected_gravity))
    return torch.stack(frames, dim=1).flatten(1)


def compute_p2_historical_reduced_coord_from_motion_lib(
    motion_lib, motion_ids: Tensor, motion_times: Tensor,
    num_state_history_steps: int, dt: float, motor_dof_ids: list[int],
    history_steps: list[int], include_projected_gravity: bool = True,
) -> Tensor:
    frames = []
    for step in history_steps:
        times = (motion_times - step * dt).clamp(min=0.0)
        lengths = getattr(motion_lib, "motion_lengths", None)
        if lengths is not None:
            times = torch.minimum(times, lengths[motion_ids].to(times))
        previous_times = (times - dt).clamp(min=0.0)
        state = motion_lib.get_motion_state(motion_ids, times)
        previous = motion_lib.get_motion_state(motion_ids, previous_times)
        # Match rollout masking for unavailable history at the start of a clip.
        acceleration = (state.rigid_body_vel[:, 0] - previous.rigid_body_vel[:, 0]) / dt
        acceleration = torch.where((times >= dt).unsqueeze(-1), acceleration,
                                   torch.zeros_like(acceleration))
        acceleration = acceleration + acceleration.new_tensor([0.0, 0.0, 9.81])
        quat = state.rigid_body_rot[:, 0]
        frames.append(pack_p2_sensors(
            state.dof_pos, state.dof_vel, quat,
            rotations.quat_rotate_inverse(quat, state.rigid_body_ang_vel[:, 0], True),
            rotations.quat_rotate_inverse(quat, acceleration, True),
            motor_dof_ids, include_projected_gravity))
    return torch.stack(frames, dim=1).flatten(1)
