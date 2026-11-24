"""Train the terrain-conditioned Stage 3 adapters against the frozen FSQ93 prior."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.autoregressive import prior_constrained_sampling_log_probs
from protomotions.agents.common.latent import TARGET_LATENT_KEY, compute_discrete_latent_ppo_loss
from protomotions.agents.common.supermimic_fsq import GOAL_KEY, OBS_KEY
from protomotions.agents.peft.prior_with_peft import DiscretePriorWithPEFT
from protomotions.agents.supervised.latent_prior_model import DiscreteAutoregressiveLatentPriorModel

from .runtime import TerrainEnv, audit_terrain
from .scene import load_config


def terrain_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.SiLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
        nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(288, 64), nn.SiLU(),
    )


def load_prior(path, device):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    model = DiscreteAutoregressiveLatentPriorModel(checkpoint["model_config"]).eval().to(device)
    warmup = TensorDict({
        OBS_KEY: torch.zeros(1, 930, device=device),
        GOAL_KEY: torch.zeros(1, 730, device=device),
    }, batch_size=[1], device=device)
    with torch.no_grad():
        model.target_latent_encoder(warmup)
        model.prior(model._prior_tensordict(warmup, warmup[TARGET_LATENT_KEY]))
        model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    return model, checkpoint["step"]


class Actor(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.terrain = terrain_encoder()
        self.peft = DiscretePriorWithPEFT(
            base.prior, conditioning_dim=930 + 64 + 2,
            peft_type="dora", rank=config["rank"], alpha=config["alpha"],
            sampling_mode="prior_constraint", prior_top_p=config["prior_top_p"],
            temperature=config["temperature"],
        )
        self.peft.init_peft()
        self.peft.capture_reference()
        self.tokenization = base.latent_tokenization
        self.decoder = base.latent_decoder

    def inputs(self, obs):
        history, grid, goal = obs
        return {OBS_KEY: history, "task_cond": torch.cat((self.terrain(grid), goal), -1)}

    @torch.no_grad()
    def sample(self, obs):
        data = self.inputs(obs)
        tokens, logp = self.peft.generate(data, return_logits=False, return_logprob=True)
        indices = self.tokenization.prior_tokens_to_fsq_indices(tokens)
        codes = self.decoder.indices_to_codes(indices)
        td = TensorDict({OBS_KEY: obs[0]}, batch_size=[len(tokens)], device=tokens.device)
        return self.decoder.decode(td, codes), tokens, logp

    def score(self, obs, tokens):
        data = self.inputs(obs)
        data["tokens"] = tokens
        return self.peft(data), self.peft.forward_prior(data)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.terrain = terrain_encoder()
        # Use the latest native observation instead of absolute world position.
        self.value = nn.Sequential(nn.Linear(93 + 64 + 2, 512), nn.SiLU(),
                                   nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1))

    def forward(self, obs):
        history, grid, goal = obs
        return self.value(torch.cat((history[:, -93:], self.terrain(grid), goal), -1)).squeeze(-1)


@torch.no_grad()
def collect(env, actor, critic, cfg):
    actor.eval()
    critic.eval()
    rollout = []
    successes = failures = invalids = resets = 0
    for _ in range(cfg["rollout_steps"]):
        obs = tuple(x.clone() for x in env.observation())
        action, tokens, logp = actor.sample(obs)
        if not action.isfinite().all() or not logp.isfinite().all():
            raise RuntimeError("Nonfinite Stage 3 action or sampling likelihood")
        values = critic(obs)
        rewards, terminal, timeout, reached, failed, invalid = env.step(action)
        next_values = critic(env.observation())
        done = terminal | timeout
        rollout.append((obs, tokens, logp, values, rewards, next_values, terminal, done))
        successes += int(reached.sum())
        failures += int(failed.sum())
        invalids += int(invalid.sum())
        resets += int(done.sum())
        if done.any():
            env.reset(done.nonzero(as_tuple=True)[0])
    return rollout, dict(successes=successes, terrain_failures=failures,
                         invalid_steps=invalids, resets=resets)


def update(rollout, actor, critic, optimizers, cfg):
    history, grid, goal = map(torch.cat, zip(*(step[0] for step in rollout)))
    tokens, old_logp, values, rewards, next_values, terminal, done = (
        torch.stack(parts) for parts in zip(*(step[1:] for step in rollout))
    )
    adv = torch.empty_like(values)
    gae = torch.zeros_like(values[0])
    for t in reversed(range(len(rollout))):
        delta = rewards[t] + cfg["gamma"] * (~terminal[t]) * next_values[t] - values[t]
        gae = delta + cfg["gamma"] * cfg["gae_lambda"] * (~done[t]) * gae
        adv[t] = gae
    returns = (adv + values).flatten()
    adv = adv.flatten()
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    tokens, old_logp = tokens.flatten(0, 1), old_logp.flatten(0, 1)
    sampling = dict(p=cfg["prior_top_p"], temperature=cfg["temperature"])
    totals = torch.zeros(3, device=values.device)
    batches = 0
    actor.train()
    critic.train()
    replay_error = None
    for _ in range(cfg["ppo_epochs"]):
        for ids in torch.randperm(len(adv), device=adv.device).split(cfg["minibatch"]):
            obs = history[ids], grid[ids], goal[ids]
            logits, frozen = actor.score(obs, tokens[ids])
            lp = prior_constrained_sampling_log_probs(logits, frozen, **sampling)
            with torch.no_grad():
                chosen = lp.gather(-1, tokens[ids, :, None]).squeeze(-1)
                if not chosen.isfinite().all():
                    raise RuntimeError("PPO replay lost support for a sampled token")
                if replay_error is None:
                    replay_error = float((chosen - old_logp[ids]).abs().max())
                    if replay_error > 0.002:
                        raise RuntimeError(f"Sampler/replay log-probability error {replay_error:.5f}")
                ref_lp = prior_constrained_sampling_log_probs(frozen, frozen, **sampling)
            ppo, _ = compute_discrete_latent_ppo_loss(
                logits=logits, selected=tokens[ids], old_neglogp=-old_logp[ids],
                advantages=adv[ids], e_clip=cfg["clip"], temperature=sampling["temperature"],
                top_p=sampling["p"], prior_logits=frozen,
            )
            keep = ref_lp.isfinite()
            kl = (lp.exp() * (lp.masked_fill(~keep, 0) - ref_lp.masked_fill(~keep, 0))).sum(-1).mean()
            value_loss = F.mse_loss(critic(obs), returns[ids])
            losses = (ppo + cfg["kl_coeff"] * kl, value_loss)
            for loss, module, optimizer in zip(losses, (actor, critic), optimizers):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(module.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
            totals += torch.stack((ppo.detach(), value_loss.detach(), kl.detach()))
            batches += 1
    return dict(zip(("ppo_loss", "value_loss", "kl"), (totals / batches).tolist()),
                reward=float(rewards.mean()), first_minibatch_replay_max_abs_logp_error=replay_error)


def train(config, output, iterations, smoke, envs, minibatch, warm_start):
    if min(x for x in (iterations, envs, minibatch) if x is not None) < 1:
        raise ValueError("Expected positive iterations, environment count and minibatch")
    cfg = config["stage3"]
    if envs is not None:
        cfg["envs"] = envs
    if minibatch is not None:
        cfg["minibatch"] = minibatch
    if smoke:
        cfg.update(envs=3, rollout_steps=4, minibatch=6, ppo_epochs=1)
        iterations = 1
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    # The fused eval attention kernel differs from the gradient path.
    # Use the same kernel for rollout and teacher-forced PPO scoring.
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda:0")
    env = TerrainEnv(config, output, device, cfg["envs"])
    try:
        base, step = load_prior(config["assets"]["stage2"], device)
        actor = Actor(base, cfg).to(device)
        del base
        critic = Critic().to(device)
        named_parameters = {name for name, p in actor.named_parameters() if p.requires_grad}
        if warm_start is not None:
            checkpoint = torch.load(warm_start, map_location="cpu", weights_only=True)
            if checkpoint["prior_step"] != step or set(checkpoint["actor"]) != named_parameters:
                raise ValueError("Incompatible Stage 3 warm start")
            actor.load_state_dict(checkpoint["actor"], strict=False)
            critic.load_state_dict(checkpoint["critic"])
        params = [p for p in actor.parameters() if p.requires_grad]
        optimizers = [
            torch.optim.AdamW(params, lr=cfg["lr"]),
            torch.optim.AdamW(critic.parameters(), lr=cfg["lr"]),
        ]
        terrain_checks = audit_terrain(env) if smoke else {}
        if smoke:
            # Exercise masked timeout/reset while the remaining lanes continue.
            env.elapsed[0] = cfg["episode_steps"] - 2
        for iteration in range(1, iterations + 1):
            rollout, stats = collect(env, actor, critic, cfg)
            metrics = dict(iteration=iteration, prior_step=step,
                           layout=config["terrain"]["layout"], target_mode=config["terrain"].get("target_mode", "adjacent_platform"),
                           envs=cfg["envs"], map_samples=config["task"]["map_samples"],
                           steps=cfg["rollout_steps"], **stats, **terrain_checks,
                           **update(rollout, actor, critic, optimizers, cfg))
            print(json.dumps(metrics, allow_nan=False), flush=True)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if not smoke:
                # Frozen weights and frozen initial reference are recoverable
                # from the pinned Stage 2 checkpoint and the saved config/seed.
                state = dict(iteration=iteration, prior_step=step, config=config,
                             actor={k: v for k, v in actor.state_dict().items() if k in named_parameters},
                             critic=critic.state_dict())
                temp = output / "adapter.pt.tmp"
                torch.save(state, temp)
                os.replace(temp, output / "adapter.pt")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--minibatch", type=int)
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train(load_config(args.config), args.output, args.iterations, args.smoke, args.envs, args.minibatch, args.warm_start)


if __name__ == "__main__":
    main()
