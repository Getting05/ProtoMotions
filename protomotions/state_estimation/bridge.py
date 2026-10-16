"""Flat-ground bridge into the existing URDF 463D reconstruction path."""
import numpy as np
from scipy.spatial.transform import Rotation
from deployment.urdf_kinematics import URDFKinematics
from .physics import ANCHOR


def reconstruct_bodies(kinematics: URDFKinematics,q,qd,pelvis_quat,omega_world,estimate):
    """estimate=[pelvis ground height, pelvis-origin velocity in WAIST heading].

    Orientation/angular velocity must already be transformed from calibrated IMU
    coordinates to pelvis/world axes. Absolute xy/yaw remain gauge choices.
    Output COM velocities match IsaacLab max_coords_obs; height is translated by
    FK to the waist anchor downstream. This bridge rejects no terrain itself:
    callers MUST provide flat-ground data, as in the estimator contract.
    """
    batch=len(q); zeros=np.zeros((batch,3))
    poses=kinematics.reconstruct(q,qd,pelvis_quat,omega_world,zeros,estimate[:,0],
                                 root_velocity_point='origin',body_velocity_point='origin')
    anchor=kinematics.body_names.index(ANCHOR)
    r=Rotation.from_quat(poses['body_rot'][:,anchor]).as_matrix()
    yaw=np.arctan2(r[:,1,0],r[:,0,0])
    velocity_world=Rotation.from_euler('z',yaw[:,None]).apply(estimate[:,1:])
    return kinematics.reconstruct(q,qd,pelvis_quat,omega_world,velocity_world,estimate[:,0],
                                   root_velocity_point='origin',body_velocity_point='com')
