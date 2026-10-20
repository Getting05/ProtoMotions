# Independent P2 reduced_coord_obs experiment. See docs/gpc_reduced_coord_obs.md.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility alias for ``examples.experiments.gpc.target_prior_peft``."""

from examples.experiments.gpc.target_prior_peft_reduced_coord_obs import (
    PRIOR_CHECKPOINT,
    additional_experiment_arguments,
    configure_robot_and_simulator,
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    agent_config,
    apply_inference_overrides,
)

__all__ = (
    "PRIOR_CHECKPOINT",
    "additional_experiment_arguments",
    "configure_robot_and_simulator",
    "terrain_config",
    "scene_lib_config",
    "motion_lib_config",
    "env_config",
    "agent_config",
    "apply_inference_overrides",
)
