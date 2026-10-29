"""Measured-state bridge for the frozen P2 Stage3 steering actor.

The only retained external state is flat ground elevation and ideal pelvis XY/yaw
passed to reconstruct(). Commands are supplied by the steering controller. The
actor receives only max_coords_obs and task_obs, never nearest_surface or other
unused simulator observations. Rewards and evaluation keep the true context.
"""
import copy
import torch

from .closed_loop import ObservationBridge


class SteeringObservationBridge(ObservationBridge):
    def observations(self, components, truth_context, poses, x):
        from protomotions.envs.context_views import CurrentStateView
        from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion

        if set(components) != {'max_coords_obs', 'task_obs'}:
            raise ValueError('Audit any new steering actor input before replacement')
        allowed = {
            'max_coords_obs': {'current.rigid_body_pos', 'current.rigid_body_rot',
                              'current.rigid_body_vel', 'current.rigid_body_ang_vel',
                              'current.anchor_idx', 'ground_heights', 'body_contacts'},
            'task_obs': {'current.anchor_rot', 'steering.tar_dir',
                         'steering.tar_speed', 'steering.tar_face_dir'},
        }
        for name, component in components.items():
            if not set(component.get_bindings_dict().values()) <= allowed[name]:
                raise ValueError('Unreviewed steering observation dependency: '+name)
        if components['max_coords_obs'].static_params.get('observe_contacts', False):
            raise ValueError('True contact must not enter the actor')
        state = RobotState(state_conversion=StateConversion.COMMON,
            rigid_body_pos=poses['body_pos'], rigid_body_rot=poses['body_rot'],
            rigid_body_vel=poses['body_vel'], rigid_body_ang_vel=poses['body_ang_vel'],
            dof_pos=x[:, :30], dof_vel=x[:, 30:60], dof_forces=x[:, 60:90])
        ctx = copy.copy(truth_context)
        ctx.current = CurrentStateView(state, self.anchor)
        ctx.noisy = ctx.current
        ctx.body_contacts = torch.zeros_like(truth_context.body_contacts)
        obs = {name: component.compute(ctx) for name, component in components.items()}
        if obs['max_coords_obs'].shape[-1] != 463 or obs['task_obs'].shape[-1] != 5:
            raise ValueError('Unexpected P2 steering actor observation shape')
        return obs
