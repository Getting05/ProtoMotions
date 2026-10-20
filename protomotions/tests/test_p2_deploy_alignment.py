"""Hardware-free deployment contract and state lifecycle checks."""
from __future__ import annotations
import copy
import importlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

from protomotions.envs.obs.p2_deploy_sensors import (
    P2DeploySensorBank, CONTRACT, imu_specific_force, reference_deploy_obs, pack_deploy_sensors,
)
from protomotions.envs.obs.p2_reduced_coord import pack_p2_sensors
from protomotions.envs.component_factories import reduced_coord_obs_factory
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.factory import simulator_config


def state(n=2):
    quat = torch.zeros(n, 4); quat[:, 3] = 1.
    return SimpleNamespace(dof_pos=torch.arange(30.).expand(n, -1).clone(),
        dof_vel=torch.zeros(n, 30), root_rot=quat, root_vel=torch.zeros(n, 3),
        root_ang_vel=torch.zeros(n, 3))


class DeploymentAlignmentTests(unittest.TestCase):
    def test_guard_recovers_before_sensor_read(self):
        from protomotions.envs.base_env.p2_deploy import P2DeployEnv
        events=[]
        guard=SimpleNamespace(invalid=torch.zeros(2,dtype=torch.bool),
            inspect_and_recover=lambda snapshot,substep:events.append('recover'))
        scene=SimpleNamespace(update=lambda dt:events.append('scene'))
        fake=SimpleNamespace(_deploy_resetting=False,_deploy_bank=object(),
            simulator=SimpleNamespace(_scene=scene,_physics_guard=guard),
            _sample_deploy_sensors=lambda invalid=None:events.append('sample'))
        P2DeployEnv._install_deploy_sensor_hook(fake)
        scene.update(.005)
        self.assertEqual(events,['scene'])
        guard.inspect_and_recover(None,0)
        self.assertEqual(events,['scene','recover','sample'])
        events.clear();guard.inspect_and_recover(None,-1)
        self.assertEqual(events,['recover'])

    def test_evaluation_records_policy_action_width(self):
        from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator
        for width in (29, 30):
            fake = SimpleNamespace(env=SimpleNamespace(get_action_size=lambda: width),
                                   device='cpu', _add_robot_state_metrics=lambda *args: None)
            metrics = MimicEvaluator._create_metrics(fake, 2, torch.tensor([3,3]), 3)
            metrics['actions'].update(torch.tensor([0,1]), torch.ones(2,width))
            self.assertEqual(metrics['actions'].data.shape, (2,3,width))

    def test_specific_force_at_imu_lever_arm(self):
        s = state()
        r = torch.tensor(CONTRACT['imu_position_base'])
        s.root_ang_vel[:, 1] = 5.
        a = imu_specific_force(s.root_rot, s.root_vel, s.root_ang_vel, s.root_vel,
                               s.root_ang_vel, .002, r)
        torch.testing.assert_close(a, torch.tensor([[0., 0., 9.81 + 25*.04775]]).expand(2, -1))
        previous_omega = s.root_ang_vel.clone()
        s.root_ang_vel[:, 1] += .02  # alpha_y = 10 rad/s²
        a = imu_specific_force(s.root_rot, s.root_vel, s.root_ang_vel, s.root_vel,
                               previous_omega, .002, r)
        torch.testing.assert_close(a[:, 0], torch.full((2,), -10*.04775), atol=1e-5, rtol=1e-5)

    def test_raw_dq_without_filter_or_lag(self):
        bank = P2DeploySensorBank(2, 17, 'cpu')
        self.assertEqual(bank.dt, .005)
        self.assertFalse(CONTRACT['velocity_filter_enabled'])
        s = state(); bank.seed(s, torch.arange(2))
        for value in (1., -3., 0., 10.):
            s.dof_vel = torch.arange(60.).reshape(2,30) * value
            bank.sample(s)
            torch.testing.assert_close(bank.current[:,29:58], s.dof_vel[:,:29], rtol=0, atol=0)
        self.assertEqual(bank.sample_count, 4)
        self.assertNotIn('filtered_dq', bank.state_dict())

    def test_filtered_v2_sensor_checkpoint_rejected(self):
        bank = P2DeploySensorBank(2, 17, 'cpu')
        old = bank.state_dict(); old['version'] = 2
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            bank.load_state_dict(old)

    def test_layout_partial_reset_and_exact_resume(self):
        bank = P2DeploySensorBank(2, 17, 'cpu'); s = state(); bank.seed(s, torch.arange(2))
        self.assertEqual(bank.current.shape, (2, 71))
        torch.testing.assert_close(bank.current[:, :29], s.dof_pos[:, :29])
        torch.testing.assert_close(bank.current[:, 65:68], torch.tensor([[0.,0.,9.81]]).expand(2,-1))
        s.dof_vel.fill_(2.); bank.sample(s); bank.commit()
        restored = P2DeploySensorBank(2, 17, 'cpu'); restored.load_state_dict(bank.state_dict())
        bank.sample(s); restored.sample(s)
        torch.testing.assert_close(bank.current, restored.current)
        second = bank.current[1].clone()
        reset = state(1); reset.root_vel.fill_(100.)
        bank.seed(reset, torch.tensor([0]))
        torch.testing.assert_close(bank.current[1], second)
        torch.testing.assert_close(bank.current[0,65:68], torch.tensor([0.,0.,9.81]))
        self.assertTrue(torch.equal(bank.history[0,0],bank.history[0,-1]))

    def test_factory_hardware_packing(self):
        r = robot_config('p2'); s = state()
        q = s.root_rot * -1.
        obs = pack_deploy_sensors(s.dof_pos, s.dof_vel, q, s.root_ang_vel,
                                 torch.tensor([[0.,0.,9.81]]).expand(2,-1))
        factory = reduced_coord_obs_factory(r)
        ctx = SimpleNamespace(p2_deploy=SimpleNamespace(sensor_obs=obs))
        torch.testing.assert_close(factory.compute(ctx), obs)
        self.assertTrue((obs[:,61]<0).all())
        self.assertEqual(reduced_coord_obs_factory(r,False).compute(ctx).shape[-1],68)

    def test_all_configs_use_deployment_contract(self):
        from examples.experiments.gpc import reduced_coord_context
        args = SimpleNamespace(batch_size=16, training_max_steps=32, motion_file='unused.pt',
            scenes_file=None, tracker_checkpoint='tracker.ckpt', prior_checkpoint='prior.ckpt', headless=True)
        modules = ['mimic.fsq','mimic.fsq_lift_foot_smooth'] + ['gpc.'+x for x in (
            'prior','sft_target_prior_peft','target_prior_peft','steering_prior_peft',
            'steering_prior_peft_stopfix','task_target_prior_peft','task_steering_headvel_prior_peft',
            'task_target_prior_peft_amp','task_steering_headvel_prior_peft_amp')]
        fake={'env':SimpleNamespace(control_components={'mimic':SimpleNamespace(future_steps=[1,2])})}
        for name in modules:
            with self.subTest(module=name),patch.object(reduced_coord_context,'require_reduced_checkpoint'), \
                patch('protomotions.utils.config_utils.load_resolved_configs_from_checkpoint',return_value=fake):
                r=robot_config('p2'); sim=simulator_config('isaaclab',r,True,2,'test')
                mod=importlib.import_module('examples.experiments.'+name+'_reduced_coord_obs')
                mod.configure_robot_and_simulator(r,sim,args)
                env=mod.env_config(r,args); agent=mod.agent_config(r,env,args)
                self.assertTrue(env._target_.endswith('P2DeployEnv'))
                self.assertEqual((sim.sim.fps,sim.sim.decimation),(200,4))
                torch.testing.assert_close(r.kinematic_info.dof_limits_lower,
                    torch.tensor([v[0] for v in CONTRACT['joint_ranges']]))
                for i,joint in enumerate(CONTRACT['joint_names']):
                    self.assertEqual(r.control.control_info[joint].stiffness,CONTRACT['kp'][i])
                    self.assertEqual(r.control.control_info[joint].damping,CONTRACT['kd'][i])
                if name.startswith('mimic.'):
                    self.assertEqual(agent.model.actor.num_out,29)
                    self.assertEqual(agent.model.actor.mu_model.decoder.num_out,29)

    def test_legacy_reduced_checkpoint_rejected(self):
        from examples.experiments.gpc import reduced_coord_context as context
        from protomotions.envs.mdp_component import MdpComponent
        from protomotions.envs.obs.p2_reduced_coord import compute_p2_reduced_coord_obs
        from protomotions.envs.obs.p2_deploy_sensors import current_deploy_obs
        for compute, version in ((compute_p2_reduced_coord_obs, 1), (current_deploy_obs, 2)):
            component=MdpComponent(compute,{},static_params={
                'motor_dof_ids':list(range(29)), 'contract_version':version,
                'quaternion_sign':'preserve', 'include_projected_gravity':True})
            with self.subTest(version=version), patch.object(context,'load_resolved_configs_from_checkpoint',return_value={
                    'env':SimpleNamespace(observation_components={'reduced_coord_obs':component})}):
                with self.assertRaises(ValueError):context.require_reduced_checkpoint('old.ckpt','tracker')

    def test_reference_sensor_pipeline_finite(self):
        n=2
        def get(ids,t):
            s=state(len(ids)); s.dof_pos=t[:,None].expand(-1,30)
            s.dof_vel=t[:,None].expand(-1,30)
            quat=s.root_rot[:,None,:]
            return SimpleNamespace(dof_pos=s.dof_pos,dof_vel=s.dof_vel,
                rigid_body_rot=quat,rigid_body_vel=s.root_vel[:,None,:],
                rigid_body_ang_vel=s.root_ang_vel[:,None,:])
        lib=SimpleNamespace(get_motion_state=get,motion_lengths=torch.ones(1)*3.)
        obs=reference_deploy_obs(lib,torch.zeros(n,dtype=torch.long),torch.ones(n),17,.02,
                                 list(range(29)),[1,2,4,8,16])
        self.assertEqual(obs.shape,(2,355)); self.assertTrue(torch.isfinite(obs).all())
        for i, step in enumerate([1,2,4,8,16]):
            expected = torch.full((2,29), 1. - step*.02)
            torch.testing.assert_close(obs[:,i*71:i*71+29], expected)
            torch.testing.assert_close(obs[:,i*71+29:i*71+58], expected)


if __name__=='__main__':unittest.main(verbosity=2)
