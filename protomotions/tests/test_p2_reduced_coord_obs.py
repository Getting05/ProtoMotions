"""Run with python -m unittest protomotions.tests.test_p2_reduced_coord_obs."""
import importlib
import pickle
import unittest
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from protomotions.envs.component_factories import reduced_coord_obs_factory, p2_motor_dof_ids
from protomotions.envs.obs.p2_reduced_coord import (
    compute_p2_reduced_coord_obs, pack_p2_sensors,
    compute_p2_historical_reduced_coord_obs,
    compute_p2_historical_reduced_coord_from_motion_lib,
)
from protomotions.robot_configs.factory import robot_config


def all_input_keys(config):
    if is_dataclass(config):
        for field in fields(config):
            value = getattr(config, field.name)
            if field.name == 'in_keys':
                yield from value
            else:
                yield from all_input_keys(value)
    elif isinstance(config, dict):
        for value in config.values():
            yield from all_input_keys(value)
    elif isinstance(config, (list, tuple)):
        for value in config:
            yield from all_input_keys(value)


class ReducedCoordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = robot_config('p2')
        cls.ids = p2_motor_dof_ids(cls.robot)

    def inputs(self):
        return dict(dof_pos=torch.arange(30.).expand(2, -1).clone(),
                    dof_vel=torch.arange(30.).expand(2, -1).clone() + 100,
                    root_rot=torch.tensor([[0., 0., 0., 1.]]).expand(2, -1).clone(),
                    root_ang_vel=torch.tensor([[1., 2., 3.]]).expand(2, -1).clone(),
                    root_vel=torch.zeros(2, 3), historical_body_vel=torch.zeros(2, 1, 31, 3),
                    progress_buf=torch.ones(2, dtype=torch.long), dt=0.02, motor_dof_ids=self.ids)

    def test_layout_head_exclusion_and_static_imu(self):
        x = self.inputs()
        obs = compute_p2_reduced_coord_obs(**x)
        self.assertEqual(obs.shape, (2, 71))
        torch.testing.assert_close(obs[:, :29], x['dof_pos'][:, self.ids])
        torch.testing.assert_close(obs[:, 29:58], x['dof_vel'][:, self.ids])
        torch.testing.assert_close(obs[:, 62:65], x['root_ang_vel'])
        torch.testing.assert_close(obs[:, 65:68], torch.tensor([[0., 0., 9.81]]).expand(2, -1))
        torch.testing.assert_close(obs[:, 68:71], torch.tensor([[0., 0., -1.]]).expand(2, -1))
        head = self.robot.kinematic_info.dof_names.index('head_joint')
        x['dof_pos'][:, head] = 9999
        x['dof_vel'][:, head] = -9999
        torch.testing.assert_close(obs, compute_p2_reduced_coord_obs(**x))
        self.assertEqual(compute_p2_reduced_coord_obs(**x, include_projected_gravity=False).shape[1], 68)

    def test_acceleration_freefall_reset_orientation(self):
        x = self.inputs()
        x['root_vel'][:, 2] = -9.81 * x['dt']
        obs = compute_p2_reduced_coord_obs(**x)
        torch.testing.assert_close(obs[:, 65:68], torch.zeros(2, 3), atol=1e-5, rtol=0)
        x['progress_buf'].zero_()
        x['root_vel'].fill_(1000)
        x['root_rot'][:] = torch.tensor([2**-0.5, 0., 0., 2**-0.5])
        obs = compute_p2_reduced_coord_obs(**x)
        torch.testing.assert_close(obs[:, 65:68], torch.tensor([[0., 9.81, 0.]]).expand(2, -1), atol=1e-5, rtol=0)
        torch.testing.assert_close(obs[:, 68:71], torch.tensor([[0., -1., 0.]]).expand(2, -1), atol=1e-5, rtol=0)
        x['root_rot'].neg_()
        torch.testing.assert_close(obs, compute_p2_reduced_coord_obs(**x))

    def test_factory_resolution_and_pickle(self):
        x = self.inputs()
        ctx = SimpleNamespace(current=SimpleNamespace(**{k: x[k] for k in ('dof_pos', 'dof_vel', 'root_rot', 'root_ang_vel', 'root_vel')}),
                              historical=SimpleNamespace(rigid_body_vel=x['historical_body_vel']),
                              progress_buf=x['progress_buf'], dt=x['dt'])
        ctx.p2_deploy = SimpleNamespace(sensor_obs=compute_p2_reduced_coord_obs(**x))
        factory = pickle.loads(pickle.dumps(reduced_coord_obs_factory(self.robot)))
        torch.testing.assert_close(factory.compute(ctx), compute_p2_reduced_coord_obs(**x))

    def test_amp_reference_matches_history(self):
        n, history, bodies = 2, 18, 31
        dt = 0.02
        def state(times):
            quat = torch.zeros(n, bodies, 4); quat[..., 3] = 1
            vel = torch.zeros(n, bodies, 3); vel[..., 0] = times[:, None] * 2
            return SimpleNamespace(dof_pos=times[:, None].expand(n, 30),
                                   dof_vel=torch.ones(n, 30), rigid_body_rot=quat,
                                   rigid_body_vel=vel, rigid_body_ang_vel=torch.zeros(n, bodies, 3))
        times = torch.ones(n)
        states = [state(times - step * dt) for step in range(1, history)]
        def stack(name): return torch.stack([getattr(s, name) for s in states], 1)
        steps = [1, 2, 4, 8, 16]
        actual = compute_p2_historical_reduced_coord_obs(
            stack('dof_pos'), stack('dof_vel'), stack('rigid_body_rot'), stack('rigid_body_vel'),
            stack('rigid_body_ang_vel'), dt, self.ids, steps, progress_buf=torch.full((n,), 100))
        motion_lib = SimpleNamespace(get_motion_state=lambda ids, t: state(t), motion_lengths=torch.tensor([2.]))
        expected = compute_p2_historical_reduced_coord_from_motion_lib(
            motion_lib, torch.zeros(n, dtype=torch.long), times, 17, dt, self.ids, steps)
        self.assertEqual(actual.shape, (n, 355))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)

    def test_all_experiments_input_graph(self):
        from examples.experiments.gpc import reduced_coord_context
        args = SimpleNamespace(batch_size=16, training_max_steps=32, motion_file='unused.pt',
                               scenes_file=None, tracker_checkpoint='tracker.ckpt', prior_checkpoint='prior.ckpt')
        modules = ['mimic.fsq', 'mimic.fsq_lift_foot_smooth'] + ['gpc.' + x for x in (
            'prior', 'sft_target_prior_peft', 'target_prior_peft', 'steering_prior_peft',
            'steering_prior_peft_stopfix', 'task_target_prior_peft', 'task_steering_headvel_prior_peft',
            'task_target_prior_peft_amp', 'task_steering_headvel_prior_peft_amp')]
        fake = {'env': SimpleNamespace(control_components={'mimic': SimpleNamespace(future_steps=[1, 2])})}
        for name in modules:
            with self.subTest(module=name), patch.object(reduced_coord_context, 'require_reduced_checkpoint'), \
                 patch('protomotions.utils.config_utils.load_resolved_configs_from_checkpoint', return_value=fake):
                mod = importlib.import_module('examples.experiments.' + name + '_reduced_coord_obs')
                env = mod.env_config(self.robot, args)
                agent = mod.agent_config(self.robot, env, args)
                self.assertIn('reduced_coord_obs', env.observation_components)
                self.assertGreaterEqual(env.num_state_history_steps, 1)
                self.assertTrue(set(env.observation_components) <= {'reduced_coord_obs', 'mimic_target_poses', 'task_obs', 'historical_reduced_coord_obs'})
                self.assertFalse({'max_coords_obs', 'nearest_surface', 'historical_max_coords_obs'} & set(all_input_keys(agent)))
                if name.endswith('_amp'):
                    self.assertEqual(env.num_state_history_steps, 17)
                    self.assertIn('historical_reduced_coord_obs', agent.reference_obs_components)

    def test_reject_legacy_checkpoint(self):
        from examples.experiments.gpc import reduced_coord_context as context
        with patch.object(context, 'load_resolved_configs_from_checkpoint', return_value={'env': SimpleNamespace(observation_components={})}):
            with self.assertRaisesRegex(ValueError, '71D'):
                context.require_reduced_checkpoint('old.ckpt', 'tracker')


if __name__ == '__main__':
    unittest.main(verbosity=2)
