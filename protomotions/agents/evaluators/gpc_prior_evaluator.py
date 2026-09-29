"""GPC prior evaluation: teacher-distribution fit and autonomous upright survival.

No reference-tracking failure is used for the autonomous policy. All scalar
means are first computed per initial motion, so BaseAgent's item-weighted
aggregation remains valid across ranks. Evaluation never changes curriculum.
"""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import json
import logging
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from protomotions.agents.common.latent import LATENT_LOGITS_KEY, TARGET_LATENT_KEY
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.gpc_prior_config import GPCPriorEvaluatorConfig


log = logging.getLogger(__name__)


def token_metrics(logits, targets):
    """Unsmoothed NLL and exact-match diagnostics, with one result per sample."""
    logits = logits.float()
    nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
    nll = nll.reshape_as(targets)
    correct = logits.argmax(-1).eq(targets)
    top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices.eq(targets[..., None]).any(-1)
    result = {
        "nll": nll.mean(-1),
        "token_accuracy": correct.float().mean(-1),
        "top5_accuracy": top5.float().mean(-1),
        "sequence_accuracy": correct.all(-1).float(),
    }
    for index in range(targets.shape[-1]):
        result[f"token_{index}/nll"] = nll[:, index]
        result[f"token_{index}/accuracy"] = correct[:, index].float()
    return result


def upright_failure(height, rotation_xyzw, min_height, min_up_z):
    """Anchor local +Z projected on world +Z; rotation convention is XYZW."""
    up_z = 1 - 2 * (rotation_xyzw[..., 0].square() + rotation_xyzw[..., 1].square())
    finite = torch.isfinite(height) & torch.isfinite(rotation_xyzw).all(-1)
    return ~finite | (height < min_height) | (up_z < min_up_z)


class SurvivalAccumulator:
    def __init__(self, initial_bad, dt, max_steps, persistence):
        self.alive = ~initial_bad
        self.streak = torch.zeros_like(initial_bad, dtype=torch.long)
        self.seconds = torch.full_like(initial_bad, max_steps * dt, dtype=torch.float)
        self.seconds[initial_bad] = 0
        self.dt, self.persistence = dt, persistence

    def update(self, bad, step):
        self.streak = torch.where(bad, self.streak + 1, 0)
        failed = self.alive & (self.streak >= self.persistence)
        self.seconds[failed] = step * self.dt
        self.alive &= ~failed


def select_motion_ids(count, limit, seed, rank=0, world_size=1, shared=True):
    """Fixed without-replacement cohort; shared validation is disjoint by rank."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randperm(count, generator=generator)
    if shared:
        ids = ids[rank::world_size]
    if ids.numel() == 0:
        raise ValueError("Motion library must contain at least one motion per evaluation rank")
    return ids[:limit]


class GPCPriorEvaluator(BaseEvaluator):
    config: GPCPriorEvaluatorConfig

    def __init__(self, agent, fabric, config):
        config.validate()
        super().__init__(agent, fabric, config)
        self._validation_lib = None

    @contextmanager
    def _evaluation_scope(self):
        env = self.env
        snapshot = env.save_state()
        original_lib, original_manager = env.motion_lib, env.motion_manager
        agent_lib = self.agent.motion_lib
        original_noise = env.robot_config.reset_noise
        original_push = env.simulator._push_enabled
        was_training = self.agent.model.training
        py_rng, np_rng = random.getstate(), np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                seed = self.config.seed + self.fabric.global_rank
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                self.agent.eval()
                env.robot_config.reset_noise = None
                env.simulator._push_enabled = False
                yield
        finally:
            env.motion_lib, env.motion_manager = original_lib, original_manager
            self.agent.motion_lib = agent_lib
            env.robot_config.reset_noise = original_noise
            env.simulator._push_enabled = original_push
            env.restore_state(snapshot)
            self.agent.model.train(was_training)
            random.setstate(py_rng)
            np.random.set_state(np_rng)

    def _install_library(self, lib):
        from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
        from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager

        self.env.motion_lib = self.agent.motion_lib = lib
        self.env.motion_manager = MimicMotionManager(
            MimicMotionManagerConfig(init_start_prob=1.0), self.num_envs,
            self.env.dt, self.device, lib,
        )

    def _reset_cohort(self, ids):
        manager = self.env.motion_manager
        # Inactive environments still participate in physics, but never in metrics.
        manager.motion_ids[:] = ids[0]
        manager.motion_ids[:len(ids)] = ids
        manager.motion_times.zero_()
        obs, _ = self.env.reset(sample_flat=True, disable_motion_resample=True)
        return obs

    def _obs(self, obs, n):
        return self.agent.obs_dict_to_tensordict(obs)[:n].clone()

    def _step(self, action, n):
        actions = torch.zeros((self.num_envs, action.shape[-1]), device=self.device)
        actions[:n] = action
        return self.env.step(actions)

    def _teacher(self, ids):
        n = len(ids)
        obs = self._reset_cohort(ids)
        totals, metric_steps = {}, 0
        model = self.agent.model
        for step in range(self.config.teacher_steps):
            td = model.collect_expert_rollout(self._obs(obs, n))
            expert_action = td["mean_action"].clone()
            if step % self.config.teacher_metric_stride == 0:
                prediction = model(td)
                logits, targets = prediction[LATENT_LOGITS_KEY], prediction[TARGET_LATENT_KEY]
                metrics = token_metrics(logits, targets)
                # This is teacher-forced argmax decoding, not free generation.
                decoded = model.decode_latents(td.clone(), logits.argmax(-1))["mean_action"]
                metrics["tf_argmax_action_mse"] = (decoded - expert_action).square().mean(-1)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0) + value
                metric_steps += 1
                del prediction, logits
            obs, _, dones, _, _ = self._step(expert_action, n)
            # Clip end/tracking reset belongs only to the teacher rollout.
            reset_ids = dones[:n].nonzero(as_tuple=True)[0]
            if reset_ids.numel():
                self.env.motion_manager.motion_times[reset_ids] = 0
                obs, _ = self.env.reset(reset_ids, sample_flat=True, disable_motion_resample=True)
        result = {key: float((value / metric_steps).mean()) for key, value in totals.items()}
        # Mean of per-motion perplexities preserves item-weighted DDP aggregation.
        result["mean_motion_perplexity"] = float((totals["nll"] / metric_steps).exp().mean())
        result["labelled_states_per_motion"] = metric_steps
        return result

    def _physical_state(self, n):
        state = self.env.simulator.get_robot_state()
        anchor = self.env.robot_config.anchor_body_index
        pos = state.rigid_body_pos[:n, anchor]
        ground = self.env.terrain.get_ground_heights(pos).reshape(-1)
        bad = upright_failure(pos[:, 2] - ground, state.rigid_body_rot[:n, anchor],
                              self.config.min_anchor_height, self.config.min_up_z)
        return state, bad

    def _autonomous(self, ids):
        n = len(ids)
        obs = self._reset_cohort(ids)
        _, initial_bad = self._physical_state(n)
        tracker = SurvivalAccumulator(initial_bad, self.env.dt, self.config.max_eval_steps,
                                      self.config.failure_persistence_steps)
        action_delta_sum = torch.zeros(n, device=self.device)
        action_delta_count = torch.zeros_like(action_delta_sum)
        slip_sum, slip_count = torch.zeros_like(action_delta_sum), torch.zeros_like(action_delta_sum)
        feet = self.env.contact_body_ids
        contact_available = feet.numel() > 0
        previous_action = None
        survival = {}
        horizon_steps = {math.ceil(t / self.env.dt): t for t in (5, 10, 20)
                         if math.ceil(t / self.env.dt) <= self.config.max_eval_steps}
        for step in range(1, self.config.max_eval_steps + 1):
            # Explicit generate prevents teacher labels from reaching the policy.
            action = self.agent.model.generate(self._obs(obs, n))[0]
            active = tracker.alive.clone()
            finite_action = torch.isfinite(action).all(-1)
            tracker.seconds[active & ~finite_action] = (step - 1) * self.env.dt
            tracker.alive &= finite_action
            active &= finite_action
            action = torch.nan_to_num(action)
            if previous_action is not None:
                delta = (action - previous_action).square().mean(-1)
                action_delta_sum += torch.where(active, delta, 0)
                action_delta_count += active
            previous_action = action.clone()
            action[~active] = 0
            obs, _, _, _, _ = self._step(action, n)
            state, bad = self._physical_state(n)
            # Actual simulator contact flags; no reference contact labels/heights.
            contact = state.rigid_body_contacts[:n, feet].bool() & active[:, None]
            speed = state.rigid_body_vel[:n, feet, :2].norm(dim=-1)
            slip_sum += torch.where(contact, speed, 0).sum(-1)
            slip_count += contact.sum(-1)
            tracker.update(bad, step)
            if step in horizon_steps:
                survival[f"upright_survival_{horizon_steps[step]}s"] = float(tracker.alive.float().mean())
        survival.update({
            "upright_survival_horizon": float(tracker.alive.float().mean()),
            "initial_upright_fraction": float((~initial_bad).float().mean()),
            "mean_capped_survival_seconds": float(tracker.seconds.mean()),
            "horizon_seconds": self.config.max_eval_steps * self.env.dt,
            "action_delta_rms": float((action_delta_sum / action_delta_count.clamp_min(1)).sqrt().mean()),
            "contact_sensing_available": float(contact_available),
            "contact_observed_fraction": float((slip_count > 0).float().mean()),
        })
        if contact_available:
            survival["mean_contact_body_xy_speed_mps"] = float((slip_sum / slip_count.clamp_min(1)).mean())
        details = {"motion_ids": ids.cpu().tolist(), "initial_upright": (~initial_bad).cpu().tolist(),
                   "survived_horizon": tracker.alive.cpu().tolist(),
                   "capped_survival_seconds": tracker.seconds.cpu().tolist()}
        return survival, details

    @torch.no_grad()
    def evaluate(self):
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        if self.config.validation_motion_file and self._validation_lib is None:
            self._validation_lib = MotionLib(
                MotionLibConfig(motion_file=self.config.validation_motion_file), device=self.device)
        libraries = {"train": self.agent.motion_lib}
        if self._validation_lib is not None:
            libraries["validation"] = self._validation_lib
        rank, world = self.fabric.global_rank, self.fabric.world_size
        cohorts = {}
        for split, lib in libraries.items():
            shared = split == "validation" or not lib.different_motion_files_across_ranks
            cohorts[split] = select_motion_ids(lib.num_motions(), min(self.num_envs, self.config.num_eval_envs),
                                               self.config.seed, rank, world, shared).to(self.device)
        n = min(len(ids) for ids in cohorts.values())
        metrics, details = {}, {}
        with self._evaluation_scope():
            for split, lib in libraries.items():
                self._install_library(lib)
                ids = cohorts[split][:n]
                log.info("GPC %s: teacher evaluation on %d motions", split, n)
                teacher = self._teacher(ids)
                log.info("GPC %s: autonomous rollout for %.2f seconds", split, self.config.max_eval_steps * self.env.dt)
                rollout, record = self._autonomous(ids)
                metrics.update({f"eval_{split}/prior/{key}": value for key, value in teacher.items()})
                metrics.update({f"eval_{split}/rollout/{key}": value for key, value in rollout.items()})
                details[split] = dict(motion_file=lib.motion_file, **record)
        score_split = "validation" if "validation" in libraries else "train"
        score = metrics[f"eval_{score_split}/rollout/upright_survival_horizon"]
        self.eval_count += 1
        output = Path(self.config.output_dir) if self.config.output_dir else Path(self.root_dir) / "gpc_eval"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"epoch_{self.agent.current_epoch}_eval_{self.eval_count}_rank_{rank}.json"
        report = {"checkpoint": getattr(self.agent, "evaluation_checkpoint", None),
                  "agent_epoch": self.agent.current_epoch, "config": asdict(self.config), "rank": rank, "world_size": world,
                  "num_items_per_split": n, "score_split": score_split, "score": score,
                  "temperature": self.agent.model.temperature, "top_p": self.agent.model.top_p,
                  "metrics": metrics, "cohorts": details}
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return metrics, score, n
