"""P2 reference tuning contract across configuration, MJCF and action consumers."""
import math
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from protomotions.robot_configs.factory import robot_config
from protomotions.robot_configs.astro_p2_pose import P2_DEFAULT_DOF_POS
from protomotions.envs.action import make_pd_action_config, make_bm_pd_action_config
from protomotions.simulator.isaaclab.utils.actuator_groups import resolve_actuator_specs_for_control_type


class P2ParametersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = robot_config('p2')
        cls.asset = Path(__file__).resolve().parents[1] / 'data/assets/astro_p2/mjcf/astro_p2_protomotions.xml'

    def test_reference_motor_groups(self):
        for name, info in self.robot.control.control_info.items():
            if name == 'head_joint':
                self.assertEqual(info.armature, 0.002387)
                continue
            if '_hip_' in name or '_knee_' in name or name == 'waist_yaw_joint':
                armature, frequency = 0.04416342683946, 10.0
            elif '_wrist_pitch_' in name or '_wrist_yaw_' in name:
                armature, frequency = 0.002387232, 12.0
            else:
                armature, frequency = 0.0087362778817, 10.0
            if '_ankle_' in name or name in ('waist_pitch_joint', 'waist_roll_joint'):
                armature *= 2
            omega = 2 * math.pi * frequency
            self.assertAlmostEqual(info.armature, armature, places=14, msg=name)
            self.assertAlmostEqual(info.stiffness, armature * omega**2, places=10, msg=name)
            self.assertAlmostEqual(info.damping, 4 * armature * omega, places=10, msg=name)
        self.assertEqual(len(self.robot.control.control_info), 30)

    def test_runtime_mjcf_armature_and_no_passive_pd(self):
        model = mujoco.MjModel.from_xml_path(str(self.asset))
        for name, info in self.robot.control.control_info.items():
            joint = model.joint(name)
            np.testing.assert_allclose(model.dof_armature[joint.dofadr], info.armature, rtol=1e-12)
            np.testing.assert_array_equal(joint.stiffness, 0)
            np.testing.assert_array_equal(model.dof_damping[joint.dofadr], 0)
            np.testing.assert_array_equal(model.dof_frictionloss[joint.dofadr], 0)
        urdf = ET.parse(self.asset.parent.parent / 'urdf/astro_p2_30dof_primitive_collision.urdf')
        for joint in urdf.findall('joint'):
            if joint.get('type') == 'fixed':
                continue
            info = self.robot.control.control_info[joint.get('name')]
            self.assertEqual(info.effort_limit, float(joint.find('limit').get('effort')))
            self.assertEqual(info.velocity_limit, float(joint.find('limit').get('velocity')))

    def test_pose_and_action_consumers(self):
        robot = self.robot
        self.assertEqual(robot.default_root_height, 0.645)
        expected = {'hip_pitch': -.312, 'knee': .669, 'ankle_pitch': -.363,
                    'elbow': .6, 'shoulder_pitch': .2}
        for side in ('left', 'right'):
            for joint, value in expected.items():
                self.assertEqual(P2_DEFAULT_DOF_POS[f'{side}_{joint}_joint'], value)
        self.assertEqual(P2_DEFAULT_DOF_POS['left_shoulder_roll_joint'], .2)
        self.assertEqual(P2_DEFAULT_DOF_POS['right_shoulder_roll_joint'], -.2)
        config = make_pd_action_config(robot)
        bm = make_bm_pd_action_config(robot)
        self.assertTrue(torch.equal(config['pd_action_offset'], robot.default_dof_pos))
        self.assertTrue(torch.equal(bm['pd_action_offset'], robot.default_dof_pos))
        for i, name in enumerate(robot.kinematic_info.dof_names):
            ci = robot.control.control_info[name]
            self.assertAlmostEqual(config['stiffness'][i].item(), ci.stiffness, places=4)
            self.assertAlmostEqual(config['damping'][i].item(), ci.damping, places=4)
            self.assertAlmostEqual(bm['action_scale'][i].item(), ci.effort_limit / ci.stiffness, places=6)
        self.assertTrue(torch.all(robot.default_dof_pos >= robot.kinematic_info.dof_limits_lower))
        self.assertTrue(torch.all(robot.default_dof_pos <= robot.kinematic_info.dof_limits_upper))
        alias = robot_config('astro_p2')
        self.assertEqual(alias.control.control_info, robot.control.control_info)

    def test_isaaclab_actuator_consumer(self):
        (spec,) = resolve_actuator_specs_for_control_type(
            self.robot.control.control_info, self.robot.control.control_type)
        for name, info in self.robot.control.control_info.items():
            for field, attr in [('stiffness', 'stiffness'), ('damping', 'damping'),
                                ('armature', 'armature'), ('effort_limit_sim', 'effort_limit'),
                                ('velocity_limit_sim', 'velocity_limit')]:
                self.assertEqual(spec.params[field][name], getattr(info, attr))


if __name__ == '__main__':
    unittest.main()
