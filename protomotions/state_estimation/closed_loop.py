"""Flat-ground P2 observation-only replacement for frozen Stage1 evaluation.

The bridge accepts measured proprioception, estimated pelvis h/v, and explicitly
retained ideal pelvis XY/yaw. No simulator or contact access exists here.
Rewards/terminations retain the original context; observations use a new context.
"""
import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from deployment.urdf_kinematics import URDFKinematics
from .bridge import reconstruct_bodies
from .inference import RealtimeEstimator
from .physics import mv, tilt_from_gravity


class BatchedEstimator:
    """Fixed cohorts, synchronized resets; same windows/model as RealtimeEstimator."""
    def __init__(self, checkpoint, device):
        runtime = RealtimeEstimator(checkpoint, device)
        self.physics, self.model = runtime.physics, runtime.model
        self.history_length, self.dt = runtime.history_length, runtime.dt
        self.reset()

    def reset(self):
        self.history, self.candidates = [], []

    @torch.inference_mode()
    def update(self, x, valid):
        candidates = self.physics(x)
        # Lowest geometrical foot is assumed stationary during bootstrap only.
        # No true contact, root height or root velocity is used.
        index = candidates[..., 0].argmax(-1)
        bootstrap = candidates[torch.arange(len(x), device=x.device), index]
        if not bool(valid.all()):
            self.reset()
            return bootstrap, False
        self.history.append(x.clone())
        self.candidates.append(candidates)
        self.history = self.history[-self.history_length:]
        self.candidates = self.candidates[-self.history_length:]
        if len(self.history) < self.history_length:
            return bootstrap, False
        result = self.model(torch.stack(self.history, 1), torch.stack(self.candidates, 1))
        return result['state'][:, -1], True


class EpisodeNoise:
    """Training distribution: Gaussian samples + episode-constant sensor biases.

    Bias and white noise are independent by environment/channel. Zero delay only.
    A fresh instance per cohort provides identical seeded streams in paired runs.
    """
    GROUPS = ((0,30,'q'),(30,60,'dq'),(60,90,'torque'),(90,93,'gravity'),
              (93,96,'gyro'),(96,99,'accel'))

    def __init__(self, count, config, seed, device):
        if config.get('max_delay_frames', 0):
            raise ValueError('This evaluation supports the current zero-delay noise config')
        self.config = config
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.bias = torch.zeros(count, 129, device=device)
        for lo, hi, name in self.GROUPS:
            self.bias[:, lo:hi] = torch.randn((count, hi-lo), generator=self.generator,
                                            device=device) * config.get(name+'_bias_std', 0.)

    def __call__(self, x):
        y = x.clone() + self.bias
        for lo, hi, name in self.GROUPS:
            y[:, lo:hi] += torch.randn(y[:, lo:hi].shape, generator=self.generator,
                                      device=x.device) * self.config.get(name+'_std', 0.)
        y[:, 90:93] = torch.nn.functional.normalize(y[:, 90:93], dim=-1)
        return y


class ObservationBridge:
    def __init__(self, physics):
        self.physics = physics
        self.kinematics = URDFKinematics(physics.urdf, physics.body_names, physics.dof_names)
        self.anchor = physics.body_names.index('waist_roll_link')

    def reconstruct(self, x, estimate, pelvis_xy, pelvis_yaw, ground_height):
        """No hidden true height/velocity: XY/yaw are explicitly retained inputs."""
        gravity = mv(self.physics.imu_rotation, x[:, 90:93])
        tilt = tilt_from_gravity(gravity).cpu().numpy()
        omega = mv(self.physics.imu_rotation, x[:, 93:96]).cpu().numpy()
        tilt_yaw = np.arctan2(tilt[:, 1, 0], tilt[:, 0, 0])
        rz = Rotation.from_euler('z', (pelvis_yaw.cpu().numpy()-tilt_yaw)[:, None]).as_matrix()
        rotation = rz @ tilt
        q = x.cpu().numpy()
        poses = reconstruct_bodies(self.kinematics, q[:, :30], q[:, 30:60],
            Rotation.from_matrix(rotation).as_quat(), np.einsum('bij,bj->bi', rotation, omega),
            estimate.cpu().numpy())
        poses['body_pos'][..., :2] += pelvis_xy.cpu().numpy()[:, None, :]
        poses['body_pos'][..., 2] += ground_height.cpu().numpy()[:, None]
        return {key: torch.as_tensor(value, device=x.device, dtype=x.dtype) for key, value in poses.items()}

    def observations(self, components, truth_context, poses, x):
        from protomotions.envs.context_views import CurrentStateView
        from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion
        # Fail closed if a future checkpoint adds unhandled policy inputs.
        if set(components) != {'max_coords_obs', 'mimic_target_poses'}:
            raise ValueError('Audit new observation components before using this bridge')
        allowed = {'current.rigid_body_pos', 'current.rigid_body_rot', 'current.rigid_body_vel',
                   'current.rigid_body_ang_vel', 'current.anchor_idx', 'ground_heights',
                   'body_contacts', 'mimic.future_pos', 'mimic.future_rot',
                   'mimic.future_vel', 'mimic.future_ang_vel', 'mimic.anchor_idx'}
        for component in components.values():
            if not set(component.get_bindings_dict().values()) <= allowed:
                raise ValueError('Unreviewed observation state dependency')
        if components['max_coords_obs'].static_params.get('observe_contacts', False):
            raise ValueError('True contacts must not enter the actor')
        state = RobotState(state_conversion=StateConversion.COMMON,
            rigid_body_pos=poses['body_pos'], rigid_body_rot=poses['body_rot'],
            rigid_body_vel=poses['body_vel'], rigid_body_ang_vel=poses['body_ang_vel'],
            dof_pos=x[:, :30], dof_vel=x[:, 30:60], dof_forces=x[:, 60:90])
        ctx = copy.copy(truth_context)
        ctx.current = CurrentStateView(state, self.anchor)
        ctx.noisy = ctx.current
        ctx.body_contacts = torch.zeros_like(truth_context.body_contacts)
        # Only flat-ground elevation and reference trajectory remain from context.
        obs = {name: component.compute(ctx) for name, component in components.items()}
        if obs['max_coords_obs'].shape[-1] != 463:
            raise ValueError('P2 observation size changed')
        return obs
