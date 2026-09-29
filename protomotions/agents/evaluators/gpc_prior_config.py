"""Bounded, reproducible GPC stage-two evaluation configuration."""

from dataclasses import dataclass
from typing import Optional

from protomotions.agents.evaluators.config import EvaluatorConfig


@dataclass
class GPCPriorEvaluatorConfig(EvaluatorConfig):
    _target_: str = "protomotions.agents.evaluators.gpc_prior_evaluator.GPCPriorEvaluator"
    max_eval_steps: int = 1000
    num_eval_envs: int = 32
    teacher_steps: int = 64
    teacher_metric_stride: int = 4
    seed: int = 20260929
    validation_motion_file: Optional[str] = None
    output_dir: Optional[str] = None
    # Upright-locomotion diagnostic, not a general acrobatics success criterion.
    min_anchor_height: float = 0.30
    min_up_z: float = 0.25
    failure_persistence_steps: int = 5

    def validate(self):
        for name in ("max_eval_steps", "num_eval_envs", "teacher_steps",
                     "teacher_metric_stride", "failure_persistence_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.min_anchor_height < 0 or not -1 <= self.min_up_z <= 1:
            raise ValueError("Invalid upright-survival thresholds")
        if self.validation_motion_file and "slurmrank" in self.validation_motion_file:
            raise ValueError("Prior validation requires one concrete MotionLib .pt file")
