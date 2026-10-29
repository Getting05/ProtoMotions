"""Measurement-only GPC reconstruction and separately scoped truth diagnostics.

No simulator import: usable in CPU checks and on a real robot sensor stream.
XY/yaw are gauges absent from max_coords_obs. Simulator alignment is exclusively
for visualization/error scoring and NEVER feeds the estimator or controller.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .bridge import reconstruct_bodies
from .physics import ANCHOR, mv, tilt_from_gravity


def yaw(matrix):
    return np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])


def heading_rotation(quat):
    return Rotation.from_euler('z', yaw(Rotation.from_quat(quat).as_matrix()))


def encode_gpc(poses, anchor_index):
    from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
    tensors=[torch.as_tensor(poses[k],dtype=torch.float32) for k in
             ('body_pos','body_rot','body_vel','body_ang_vel')]
    batch=len(tensors[0])
    obs=compute_humanoid_max_coords_observations(
        *tensors,torch.zeros(batch),torch.zeros(batch,0,dtype=torch.bool),
        True,True,False,True,anchor_index)
    if obs.shape[-1]!=463: raise ValueError(f'Expected P2 463D GPC contract, got {obs.shape}')
    return obs.numpy()


def from_measurements(proprio, estimate, physics, kinematics):
    """Only 129 measured channels and 4 estimated channels enter reconstruction."""
    x=torch.as_tensor(proprio,device=physics.imu_rotation.device,dtype=torch.float32)
    if x.shape!=(129,): raise ValueError('Expected one 129D sensor frame')
    gravity=mv(physics.imu_rotation,x[90:93])
    r=tilt_from_gravity(gravity).detach().cpu().numpy()
    omega_b=mv(physics.imu_rotation,x[93:96]).detach().cpu().numpy()
    xx=x.detach().cpu().numpy()
    poses=reconstruct_bodies(kinematics,xx[None,:30],xx[None,30:60],
        Rotation.from_matrix(r).as_quat()[None],(r@omega_b)[None],np.asarray(estimate)[None])
    return poses,encode_gpc(poses,kinematics.body_names.index(ANCHOR)),r,omega_b


def transformed(poses, rotation, translation):
    result={k:v.copy() for k,v in poses.items()}
    result['body_pos'][0]=rotation.apply(poses['body_pos'][0])+translation
    result['body_rot'][0]=(rotation*Rotation.from_quat(poses['body_rot'][0])).as_quat()
    for k in ('body_vel','body_ang_vel'): result[k][0]=rotation.apply(poses[k][0])
    return result


class DisplayPlacement:
    """Aligned display or IMU yaw + estimated velocity integration after warmup.

    Drift starts at the simulator pelvis XY/yaw once per reset; z is always the
    estimated height. It measures drift SINCE warmup, not absolute localization.
    """
    def __init__(self, body_names, dt):
        self.root=body_names.index('pelvis'); self.anchor=body_names.index(ANCHOR)
        self.dt=dt; self.reset()

    def reset(self):
        self.rotation=None; self.xy=None; self.previous_velocity=None

    def place(self, poses, truth, ground, tilt, omega_b, origin_velocity):
        root=self.root
        rt=Rotation.from_quat(truth['body_rot'][0,root]).as_matrix()
        aligned_rot=Rotation.from_euler('z',yaw(rt)-yaw(tilt))
        translation=np.r_[truth['body_pos'][0,root,:2],ground]
        aligned=transformed(poses,aligned_rot,translation)
        if self.rotation is None:
            self.rotation=aligned_rot.as_matrix()@tilt
            self.xy=translation[:2].copy()
            first=True
        else:
            predicted=self.rotation@Rotation.from_rotvec(omega_b*self.dt).as_matrix()
            self.rotation=Rotation.from_euler('z',yaw(predicted)-yaw(tilt)).as_matrix()@tilt
            first=False
        world=Rotation.from_matrix(self.rotation@tilt.T)
        # Pelvis origin velocity is recovered from COM using merged URDF offset
        # by the caller; the poses contain COM velocities used by GPC.
        velocity=world.apply(origin_velocity)[:2]
        if not first: self.xy+=(velocity+self.previous_velocity)*(.5*self.dt)
        self.previous_velocity=velocity
        drift=transformed(poses,world,np.r_[self.xy,ground])
        return aligned,drift


def diagnostic_frame(estimate, target, poses, aligned, drift, truth, truth_obs,
                     measured_q, truth_q, kinematics):
    """Truth is read ONLY here (and for the explicitly labelled display gauge)."""
    anchor=kinematics.body_names.index(ANCHOR)
    estimate_obs=encode_gpc(poses,anchor)
    diff=estimate_obs[0]-np.asarray(truth_obs).reshape(463)
    ah=heading_rotation(aligned['body_rot'][0,anchor]).inv()
    th=heading_rotation(truth['body_rot'][0,anchor]).inv()
    errors={
        'position_cm':np.linalg.norm(aligned['body_pos'][0]-truth['body_pos'][0],axis=-1)*100,
        'drift_position_cm':np.linalg.norm(drift['body_pos'][0]-truth['body_pos'][0],axis=-1)*100,
        'rotation_deg':(Rotation.from_quat(truth['body_rot'][0]).inv()*Rotation.from_quat(aligned['body_rot'][0])).magnitude()*180/np.pi,
        'velocity_mps':np.linalg.norm(ah.apply(aligned['body_vel'][0])-th.apply(truth['body_vel'][0]),axis=-1),
        'angular_velocity_rps':np.linalg.norm(ah.apply(aligned['body_ang_vel'][0])-th.apply(truth['body_ang_vel'][0]),axis=-1),
        'encoder_angle_deg':np.abs(np.arctan2(np.sin(measured_q-truth_q),np.cos(measured_q-truth_q)))*180/np.pi,
    }
    slices={'height_m':(0,1),'relative_position_m':(1,91),'rotation_6d':(91,277),
            'linear_velocity_mps':(277,370),'angular_velocity_rps':(370,463)}
    groups={key:float(np.sqrt(np.mean(diff[lo:hi]**2))) for key,(lo,hi) in slices.items()}
    return dict(estimate=np.asarray(estimate).tolist(),target=np.asarray(target).tolist(),
        mean={k:float(v.mean()) for k,v in errors.items()},per_body={k:v.tolist() for k,v in errors.items()},
        gpc_rmse=groups,gpc_max_abs={k:float(np.abs(diff[lo:hi]).max()) for k,(lo,hi) in slices.items()},
        # This mixed-unit number is only a raw feature diagnostic, not physical RMSE.
        gpc_raw_mae=float(np.abs(diff).mean()),gpc_estimate=estimate_obs[0].tolist())
