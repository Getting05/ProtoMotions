"""Eight-GPU Barrier RLFT with the frozen Physics-Drift OneStep prior."""
import argparse
import copy
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import yaml

from p2drift.task_stage3.peft import (
    OneStepGPCActor, discrete_ppo_loss, reference_kl, sample_log_probs, sample_tokens,
)
from p2drift.task_stage3.trainer import load_prior_and_tracker, setup

from .runtime import TerrainEnv
from .scene import load_config


class TerrainEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
            nn.Linear(32 * 3 * 3, 64), nn.SiLU(),
        )

    def forward(self, grid, goal):
        return torch.cat((self.network(grid), goal), dim=-1)


class TerrainActor(nn.Module):
    def __init__(self, prior, quantizer, config):
        super().__init__()
        self.terrain = TerrainEncoder()
        self.actor = OneStepGPCActor(
            prior, quantizer, task_dim=66, rank=config["rank"], alpha=config["alpha"],
            raw_std=config["raw_std"],
        )

    def forward(self, noise, history, grid, goal):
        return self.actor(noise, history, self.terrain(grid, goal))


class TerrainCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.terrain = TerrainEncoder()
        self.value = nn.Sequential(nn.Linear(93 + 66, 512), nn.SiLU(),
                                   nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1))

    def forward(self, history, grid, goal):
        task = self.terrain(grid, goal)
        return self.value(torch.cat((history[:, -1], task), -1)).squeeze(-1)


class Runner:
    def __init__(self, config, output, rank, world, local, device):
        self.config, self.output, self.rank, self.world = config, output, rank, world
        self.opts, self.device = config["stage3"], device
        opts = self.opts
        self.envs, self.steps = opts["envs_per_gpu"], opts["rollout_steps"]
        self.sampling = dict(mode=opts["sampling_mode"], prior_top_p=opts["prior_top_p"],
                             top_p=opts["top_p"], temperature=opts["temperature"])
        prior, self.tracker = load_prior_and_tracker(config, device)
        self.actor = TerrainActor(prior, self.tracker.quantizer, opts).to(device)
        self.critic = TerrainCritic().to(device)
        if world > 1:
            options = dict(device_ids=[local], broadcast_buffers=False,
                           gradient_as_bucket_view=True, static_graph=True)
            self.actor_train = DistributedDataParallel(self.actor, **options)
            self.critic_train = DistributedDataParallel(self.critic, **options)
        else:
            self.actor_train, self.critic_train = self.actor, self.critic
        # Capture after DDP sync, before any updates, so all ranks use the same
        # frozen reference even though policy noise and terrain vary by rank.
        self.reference = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.optim_actor = torch.optim.AdamW(self.actor_params, lr=opts["actor_lr"], fused=True)
        self.optim_critic = torch.optim.AdamW(self.critic.parameters(), lr=opts["critic_lr"], fused=True)

        rank_config = copy.deepcopy(config)
        rank_config["terrain"]["seed"] += rank
        self.env = TerrainEnv(rank_config, output / "scenes" / f"rank-{rank:02d}",
                              device, self.envs, tracker=self.tracker)
        self.episode_return = torch.zeros(self.envs, device=device)
        t, n = self.steps, self.envs
        self.batch = {
            key: torch.empty((t, n, *shape), device=device, dtype=dtype)
            for key, shape, dtype in (
                ("history", (10, 93), torch.float32),
                ("pose", (7,), torch.float32),
                ("target", (3,), torch.float32),
                ("noise", (40,), torch.float32),
                ("prior_logits", (40, 9), torch.float32),
                ("tokens", (40,), torch.uint8),
                ("logp", (), torch.float32),
                ("reward", (), torch.float32),
                ("value", (), torch.float32),
                ("terminal", (), torch.bool),
                ("done", (), torch.bool),
                ("next_value", (), torch.float32),
            )
        }
        print(f"rank={rank} envs={n} scene_seed={rank_config['terrain']['seed']} "
              f"clearances={[round(float(g[4]), 3) for g in self.env.task.scene.beams]}", flush=True)

    @torch.no_grad()
    def collect(self):
        self.actor.eval()
        self.critic.eval()
        batch, env = self.batch, self.env
        history = env.history.buffer
        grid, goal = env.task.observations(env.state, env.target)
        counters = torch.zeros(7, device=self.device)
        beam_x = env.task.beams[:, 0]
        for t in range(self.steps):
            # Keep compact pose and target in the rollout. Reconstruct height
            # maps from fixed per-rank geometry in the PPO minibatch.
            for name, val in zip(("history", "pose", "target"),
                                 (history, env.state[:, :7], env.target)):
                batch[name][t].copy_(val)
            noise = batch["noise"][t].normal_()
            ref = self.reference(noise, history, grid, goal)
            logits = self.actor(noise, history, grid, goal)
            tokens, logp = sample_tokens(sample_log_probs(
                logits, reference_logits=ref, **self.sampling
            ))
            action = self.tracker.decode(history, self.tracker.quantizer.indices_to_codes(tokens))
            value = self.critic(history, grid, goal)
            previous_x = env.state[:, :1]
            reward, terminal, timeout, reached, failed, invalid = env.step(action)
            done = terminal | timeout
            crossed = ((previous_x < beam_x) & (env.state[:, :1] >= beam_x)).any(dim=-1)
            self.episode_return += reward
            counters += torch.stack((
                reached.sum(), (crossed & ~failed & ~invalid).sum(), done.sum(),
                failed.sum(), invalid.sum(), (self.episode_return * done).sum(),
                (terminal & ~reached & ~failed & ~invalid).sum(),
            ))
            self.episode_return.masked_fill_(done, 0.)
            history = env.history.buffer
            grid, goal = env.task.observations(env.state, env.target)
            next_value = self.critic(history, grid, goal)
            for name, val in zip(
                ("prior_logits", "tokens", "logp", "reward", "value", "terminal", "done", "next_value"),
                (ref, tokens, logp, reward, value, terminal, done, next_value),
            ):
                batch[name][t].copy_(val)
            reset_ids = done.nonzero(as_tuple=True)[0]
            if reset_ids.numel():
                env.reset(reset_ids)
                history = env.history.buffer
                new_grid, new_goal = env.task.observations(env.state[reset_ids], env.target[reset_ids])
                grid[reset_ids] = new_grid
                goal[reset_ids] = new_goal
        return counters

    def learn(self):
        opts, batch = self.opts, self.batch
        rewards, values = batch["reward"], batch["value"]
        advantages = torch.empty_like(values)
        gae = torch.zeros_like(values[0])
        for t in reversed(range(self.steps)):
            delta = (rewards[t] + opts["gamma"] * ~batch["terminal"][t]
                     * batch["next_value"][t] - values[t])
            gae = delta + opts["gamma"] * opts["gae_lambda"] * ~batch["done"][t] * gae
            advantages[t] = gae
        returns = (advantages + values).flatten()
        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        data = {key: batch[key].flatten(0, 1) for key in (
            "history", "pose", "target", "noise", "prior_logits", "tokens", "logp"
        )}
        total = torch.zeros(4, device=self.device)
        replay_error = None
        self.actor.train()
        self.critic.train()
        minibatches = 0
        for _ in range(opts["ppo_epochs"]):
            order = torch.randperm(len(advantages), device=self.device)
            for idx in order.split(opts["minibatch_size"]):
                history, pose, target, noise, ref, tokens, old_logp = (
                    data[key][idx] for key in ("history", "pose", "target", "noise",
                                               "prior_logits", "tokens", "logp")
                )
                grid, goal = self.env.task.observations(pose, target)
                logits = self.actor_train(noise, history, grid, goal)
                lp = sample_log_probs(logits, reference_logits=ref, **self.sampling)
                chosen = lp.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1).sum(-1)
                if replay_error is None:
                    replay_error = (chosen.detach() - old_logp).abs().max()
                policy, clip_fraction = discrete_ppo_loss(chosen, old_logp, advantages[idx], opts["clip"])
                kl = reference_kl(logits, ref, self.sampling["mode"], self.sampling["prior_top_p"])
                loss = policy + opts["kl_coeff"] * kl
                self.optim_actor.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, opts["grad_clip"])
                self.optim_actor.step()

                value_loss = F.mse_loss(self.critic_train(history, grid, goal), returns[idx])
                self.optim_critic.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), opts["grad_clip"])
                self.optim_critic.step()
                total += torch.stack((loss.detach(), kl.detach(), value_loss.detach(), clip_fraction))
                minibatches += 1
        return total / minibatches, replay_error, returns.mean()

    def checkpoint(self, iteration):
        if self.rank:
            return
        tmp = self.output / "stage3-latest.pt.tmp"
        torch.save(dict(
            iteration=iteration, model=self.actor.state_dict(), reference=self.reference.state_dict(),
            critic=self.critic.state_dict(), actor_optimizer=self.optim_actor.state_dict(),
            critic_optimizer=self.optim_critic.state_dict(), stage3_config=self.config,
        ), tmp)
        os.replace(tmp, self.output / "stage3-latest.pt")

    def run(self, benchmark=False):
        try:
            for i in range(1, self.opts["iterations"] + 1):
                if self.world > 1:
                    dist.barrier()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
                counters = self.collect()
                torch.cuda.synchronize()
                collect_s = perf_counter() - start
                train_stats, replay_error, mean_return = self.learn()
                torch.cuda.synchronize()
                elapsed = perf_counter() - start
                means = torch.cat((train_stats, torch.stack((self.batch["reward"].mean(), mean_return))))
                maxima = torch.tensor(
                    [collect_s, elapsed, torch.cuda.max_memory_allocated() / 2**30,
                     torch.cuda.max_memory_reserved() / 2**30, float(replay_error)],
                    device=self.device,
                )
                if self.world > 1:
                    dist.all_reduce(means)
                    means /= self.world
                    dist.all_reduce(counters)
                    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
                if not means.isfinite().all():
                    raise RuntimeError(f"Nonfinite rollout or PPO update: stats={means}, counts={counters}")
                if maxima[-1] > 0.20:
                    raise RuntimeError(f"OneStep sampler/replay error: {maxima[-1].item():.6f}")
                if not self.rank:
                    counts = counters.tolist()
                    transitions = self.envs * self.steps * self.world
                    stats = dict(iteration=i, world_size=self.world, envs_per_gpu=self.envs,
                                 minibatch=self.opts["minibatch_size"], transitions=transitions,
                                 **dict(zip(("loss", "kl", "value_loss", "clip_fraction", "mean_reward", "mean_return"), means.tolist())),
                                 **dict(zip(("collect_s", "seconds", "peak_allocated_gib", "peak_reserved_gib", "replay_error"), maxima.tolist())),
                                 successes=int(counts[0]), beam_crossings=int(counts[1]),
                                 episodes=int(counts[2]), terrain_failures=int(counts[3]),
                                 falls=int(counts[6]), invalid_steps=int(counts[4]), mean_episode_reward=counts[5] / max(1, counts[2]))
                    stats["transitions_per_second"] = transitions / stats["seconds"]
                    line = json.dumps(stats, allow_nan=False)
                    print(line, flush=True)
                    with (self.output / "train.jsonl").open("a") as stream:
                        stream.write(line + "\n")
                if not benchmark and (i == 1 or i % self.opts["save_every"] == 0 or i == self.opts["iterations"]):
                    self.checkpoint(i)
        finally:
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--minibatch", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--tracker-checkpoint", type=Path)
    parser.add_argument("--reset-cases", type=Path)
    parser.add_argument("--robot-xml", type=Path)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    node, slot = divmod(rank, 4)
    first = node * 96 + slot * 24
    cpus = set(range(first, first + 12))
    if os.environ.get("BARRIER_NUMA_PIN_5090") == "1":
        os.sched_setaffinity(0, cpus)
        print(f"NUMA rank={rank} node={node} cpus={first}-{first+11}", flush=True)
    config = yaml.safe_load(args.config.read_text())
    scene = load_config(args.config.with_name(config["terrain_config"]))
    config["terrain"], config["task"] = scene["terrain"], scene["task"]
    config["simulator"] = config["simulation"]
    for key in ("prior_checkpoint", "tracker_checkpoint", "reset_cases", "robot_xml"):
        path = getattr(args, key)
        if path is not None:
            config["assets"][key] = str(path.resolve())
    opts = config["stage3"]
    for key, value in (("envs_per_gpu", args.envs), ("minibatch_size", args.minibatch),
                       ("iterations", args.iterations), ("rollout_steps", args.steps)):
        if value is not None:
            opts[key] = value
    if min(opts[k] for k in ("envs_per_gpu", "minibatch_size", "iterations", "rollout_steps")) < 1:
        raise ValueError("Training sizes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rank, world, local, device = setup()
    torch.manual_seed(config["seed"] + rank)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision(opts["precision"])
    torch.backends.cudnn.allow_tf32 = opts["precision"] != "highest"
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        if rank == 0:
            (args.output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        Runner(config, args.output, rank, world, local, device).run(args.benchmark)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
