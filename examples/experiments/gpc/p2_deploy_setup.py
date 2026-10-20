"""Frozen rl_deploy Astro P2 contract for the independent reduced experiments."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

CONTRACT = json.loads((Path(__file__).resolve().parents[3] /
    'protomotions/envs/obs/p2_deploy_contract.json').read_text())


def configure_p2_deploy(robot_cfg, simulator_cfg, args):
    if list(robot_cfg.kinematic_info.dof_names) != CONTRACT['joint_names']:
        raise ValueError('Deployment alignment requires the P2 30-joint semantic order')
    overrides = {}
    for i, name in enumerate(CONTRACT['joint_names']):
        overrides[name] = replace(robot_cfg.control.control_info[name],
                                 stiffness=CONTRACT['kp'][i], damping=CONTRACT['kd'][i])
    robot_cfg.control.override_control_info = overrides
    robot_cfg.control.control_info = overrides
    robot_cfg.update_fields(asset=replace(robot_cfg.asset,
        asset_file_name='astro_p2/mjcf/astro_p2_deploy_reduced.xml'))
    # update_fields only remaps contact names; explicitly rebuild asset-derived
    # limits and control tables so simulator verification and action scales agree.
    robot_cfg.__post_init__()
    # Preserve the original training physics rate and 50 Hz policy.
    simulator_cfg.sim.fps = 200
    simulator_cfg.sim.decimation = 4


def deployment_env(config, args):
    lift = getattr(args, 'reset_lift_2mm', False)
    config._target_ = ('protomotions.envs.base_env.p2_deploy.P2DeployLift2mmEnv' if lift
                      else 'protomotions.envs.base_env.p2_deploy.P2DeployEnv')
    return config
