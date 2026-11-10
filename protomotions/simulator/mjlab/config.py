# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo Warp training through mjlab; no runtime imports at config time."""

from dataclasses import dataclass, field
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class MjLabSimParams(SimParams):
    fps: int = 200
    decimation: int = 4
    iterations: int = 100
    ls_iterations: int = 50
    nconmax: int = 128
    njmax: int = 512
    contact_sensor_maxmatch: int = 128
    integrator: str = "implicitfast"
    cone: str = "pyramidal"


@dataclass
class MjLabSimulatorConfig(SimulatorConfig):
    _target_: str = "protomotions.simulator.mjlab.simulator.MjLabSimulator"
    sim: MjLabSimParams = field(default_factory=MjLabSimParams)
    w_last: bool = False
