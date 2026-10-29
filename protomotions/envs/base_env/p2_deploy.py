"""Independent P2 reduced environment matching rl_deploy sensor/action semantics."""
from __future__ import annotations

import torch
from types import SimpleNamespace
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.obs.p2_deploy_sensors import P2DeploySensorBank, CONTRACT


class P2DeployEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        self._deploy_bank = None
        self._deploy_resetting = False
        super().__init__(*args, **kwargs)
        if not hasattr(self.simulator, 'register_post_physics_substep') and not hasattr(self.simulator, '_scene'):
            raise ValueError('P2 deployment sensor environment requires IsaacLab or mjlab')
        physics_dt = 1. / self.simulator.config.sim.fps
        if abs(physics_dt - CONTRACT['physics_dt']) > 1e-9 or abs(self.dt - .02) > 1e-9:
            raise ValueError('P2 deployment contract requires 200 Hz physics / 50 Hz policy')
        self._install_deploy_sensor_hook()

    def _install_deploy_sensor_hook(self):
        if hasattr(self.simulator, "register_post_physics_substep"):
            def sample_after_substep():
                if not self._deploy_resetting and self._deploy_bank is not None:
                    guard = getattr(self.simulator, "_physics_guard", None)
                    self._sample_deploy_sensors(None if guard is None else guard.invalid)
            self.simulator.register_post_physics_substep(sample_after_substep)
            return
        # All IsaacLab step paths (normal, physics guard and trace) update this
        # same scene after each physical substep. The wrapper is instance-local.
        update_scene = self.simulator._scene.update
        def update_and_sample(dt, *update_args, **update_kwargs):
            result = update_scene(dt, *update_args, **update_kwargs)
            if not self._deploy_resetting and self._deploy_bank is not None:
                guard = getattr(self.simulator, '_physics_guard', None)
                if guard is None:
                    self._sample_deploy_sensors()
                elif not getattr(guard, '_p2_sensor_hook_installed', False):
                    # Guard inspects/recoveries run *after* scene.update. Never
                    # construct validated RobotState before that recovery.
                    recover = guard.inspect_and_recover
                    def recover_and_sample(snapshot, substep):
                        result = recover(snapshot, substep)
                        if substep >= 0 and not self._deploy_resetting:
                            self._sample_deploy_sensors(guard.invalid)
                        return result
                    guard.inspect_and_recover = recover_and_sample
                    guard._p2_sensor_hook_installed = True
            return result
        self.simulator._scene.update = update_and_sample

    def _sample_deploy_sensors(self, invalid=None):
        state = self.simulator.get_robot_state()
        if invalid is not None and invalid.any():
            ids = invalid.nonzero().flatten()
            reset_state = SimpleNamespace(**{name: getattr(state, name)[ids] for name in (
                'dof_pos', 'dof_vel', 'root_rot', 'root_vel', 'root_ang_vel')})
            self._deploy_bank.seed(reset_state, ids)
        self._deploy_bank.sample(state)

    def get_action_size(self):
        return 29

    def _step(self, action):
        if action.shape[-1] != 29:
            raise ValueError('P2 reduced deployment policy must output 29 actions')
        return super()._step(torch.cat((action, action.new_zeros(action.shape[:-1] + (1,))), -1))

    def _build_context(self, env_ids=None):
        context = super()._build_context(env_ids)
        if self._deploy_bank is None:
            self._deploy_bank = P2DeploySensorBank(self.num_envs, self.config.num_state_history_steps,
                                                  self.device)
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids)
        reset = (context.progress_buf == 0) | ~self._deploy_bank.valid[ids]
        if reset.any():
            state = SimpleNamespace(**{name: getattr(context.current, name)[reset] for name in (
                'dof_pos', 'dof_vel', 'root_rot', 'root_vel', 'root_ang_vel')})
            self._deploy_bank.seed(state, ids[reset])
        context.p2_deploy = SimpleNamespace(sensor_obs=self._deploy_bank.current[ids],
                                            sensor_history=self._deploy_bank.history[ids])
        return context

    def post_physics_step(self):
        if self._deploy_bank is not None:
            self._deploy_bank.commit()
        return super().post_physics_step()

    def reset_envs(self, *args, **kwargs):
        self._deploy_resetting = True
        try:
            return super().reset_envs(*args, **kwargs)
        finally:
            self._deploy_resetting = False

    def save_state(self):
        snapshot = super().save_state()
        if self._deploy_bank is not None:
            snapshot['p2_deploy_sensors'] = self._deploy_bank.state_dict()
        return snapshot

    def restore_state(self, snapshot):
        super().restore_state(snapshot)
        if 'p2_deploy_sensors' in snapshot:
            self._deploy_bank.load_state_dict(snapshot['p2_deploy_sensors'])

    def get_state_dict(self):
        state = super().get_state_dict()
        state['p2_deploy_sensors'] = self._deploy_bank.state_dict()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if 'p2_deploy_sensors' not in state:
            raise ValueError('Resume requires deployment-aligned sensor state; use a new run')
        self._deploy_bank.load_state_dict(state['p2_deploy_sensors'])


class P2DeployLift2mmEnv(P2DeployEnv):
    def compute_ref_reset_state(self, env_ids, motion_ids, motion_times, sample_flat=False):
        states, objects = super().compute_ref_reset_state(env_ids, motion_ids, motion_times, sample_flat)
        states.root_pos[:, 2] += .002
        return states, objects
