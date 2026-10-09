"""Explicit runtime overrides also work with frozen pre-evaluator checkpoints."""

from dataclasses import replace



def add_prior_eval_arguments(parser):
    parser.add_argument("--gpc-prior-eval", action="store_true",
                        help="Enable GPC teacher-token and autonomous-rollout evaluation")
    parser.add_argument("--prior-validation-motion-file", default=None,
                        help="One held-out MotionLib .pt; evaluated separately from training")
    parser.add_argument("--prior-eval-num-envs", type=int, default=None)
    parser.add_argument("--prior-eval-steps", type=int, default=None)
    parser.add_argument("--prior-eval-teacher-steps", type=int, default=None)
    parser.add_argument("--prior-eval-seed", type=int, default=None)
    parser.add_argument("--prior-eval-output", default=None)
    parser.add_argument("--prior-eval-weights", choices=("ema", "raw"), default=None,
                        help="Primary weights for GPC evaluation and best-model selection")
    parser.add_argument("--prior-eval-compare-raw-ema", action="store_true", default=None,
                        help="Evaluate both weight sets on identical cohorts and RNG seeds")


def explicit_prior_eval_options(args):
    mapping = {
        "prior_validation_motion_file": "validation_motion_file",
        "prior_eval_num_envs": "num_eval_envs",
        "prior_eval_steps": "max_eval_steps",
        "prior_eval_teacher_steps": "teacher_steps",
        "prior_eval_seed": "seed",
        "prior_eval_output": "output_dir",
        "prior_eval_weights": "weights",
        "prior_eval_compare_raw_ema": "compare_raw_ema",
    }
    values = {dst: getattr(args, src) for src, dst in mapping.items()
              if getattr(args, src, None) is not None}
    return values if values or getattr(args, "gpc_prior_eval", False) else None


def apply_prior_eval_options(agent_config, options):
    from protomotions.agents.evaluators.gpc_prior_config import GPCPriorEvaluatorConfig

    if options is None:
        return
    if not agent_config.model._target_.endswith("DiscreteAutoregressiveLatentPriorModel"):
        raise ValueError("GPC prior evaluation requires a discrete autoregressive latent prior")
    existing = agent_config.evaluator
    if isinstance(existing, GPCPriorEvaluatorConfig):
        config = replace(existing, **options)
    else:
        config = GPCPriorEvaluatorConfig(eval_metrics_every=existing.eval_metrics_every, **options)
    config.validate()
    agent_config.evaluator = config
