# Independent P2 reduced_coord_obs experiment. See docs/gpc_reduced_coord_obs.md.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in steering run with stronger stop/lateral tracking and fall termination.

Use a new experiment name plus --checkpoint for a warm start. Existing runs
retain their frozen configuration. Moving speeds, PEFT, optimizers, and KL
settings are inherited from steering_prior_peft.

Changes: stop coefficient 0.25 -> 25, moving lateral weight 0.1 -> 1,
with the standard 0.15 m fall termination. These are initial tuning choices,
not claims of validated long-horizon policy quality.
"""

from examples.experiments.gpc import steering_prior_peft_reduced_coord_obs as base
from examples.experiments.gpc.steering_prior_peft_reduced_coord_obs import (  # noqa: F401
    additional_experiment_arguments,
    agent_config,
    apply_inference_overrides,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    # The P2 default is "all"; that would disable contact-based fall detection.
    robot_cfg.update_fields(
        non_termination_contact_bodies=list(robot_cfg.contact_bodies)
    )


def env_config(robot_cfg, args):
    from protomotions.envs.component_factories import (
        fall_termination_factory,
        steering_reward_factory,
    )

    config = base.env_config(robot_cfg, args)
    config.control_components["steering"].stop_probability = 0.05
    config.reward_components["steering"] = steering_reward_factory(
        weight=1.0, tangent_err_w=1.0, stop_vel_err_scale=25.0
    )
    config.termination_components["fall"] = fall_termination_factory(
        termination_height=0.15
    )
    return config


_configure_original = configure_robot_and_simulator


def configure_robot_and_simulator(robot_cfg, simulator_cfg, args):
    from examples.experiments.gpc.p2_deploy_setup import configure_p2_deploy
    _configure_original(robot_cfg, simulator_cfg, args)
    configure_p2_deploy(robot_cfg, simulator_cfg, args)


_env_original = env_config


def env_config(robot_cfg, args):
    from examples.experiments.gpc.p2_deploy_setup import deployment_env
    return deployment_env(_env_original(robot_cfg, args), args)
