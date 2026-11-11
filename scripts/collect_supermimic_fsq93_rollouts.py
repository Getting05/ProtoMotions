#!/usr/bin/env python3
"""Collect genuine 10x93 SuperMimic tracking observations and 730D goals.

Run inside the SuperMimic MJLab physics runtime. Saves only complete .npz shards.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-checkpoint", required=True, type=Path)
    parser.add_argument("--motion-library", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--rows-per-shard", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    if not args.tracker_checkpoint.is_file() or not args.motion_library.is_file():
        parser.error("Tracker checkpoint and packed motion library must both exist.")
    if min(args.num_envs, args.steps, args.rows_per_shard) < 1:
        parser.error("Environment, steps, and shard rows must be positive.")
    if args.rows_per_shard % args.num_envs:
        parser.error("--rows-per-shard must be divisible by --num-envs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("*.npz")):
        parser.error("Output directory already contains .npz shards; use a new directory.")
    policy, run_config = load_policy_from_checkpoint(args.tracker_checkpoint, args.device)
    assert policy.cfg.fsq and policy.cfg.history == 10
    assert not policy.cfg.policy_obs_ref_rot6d and policy.cfg.goal_dim == 730
    assert run_config["env"]["robot"] == "p2"
    policy.eval()

    sim_cfg = SuperMimicEnvCfgArgs(
        npz_files=[str(args.motion_library)],
        num_envs=args.num_envs,
        future_frames=10,
        history_length=10,
        episode_length_s=10.0,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="uniform",
        record_transitions=False,
        seed=args.seed,
        robot="p2",
        p2_advanced_actuators=False,
        anchor_obs=False,
        policy_obs_ref_rot6d=False,
        compact_library=True,
        library_device="gpu",
    )
    env = make_env(sim_cfg, device=args.device)
    obs_parts: list[torch.Tensor] = []
    goal_parts: list[torch.Tensor] = []
    rows = 0
    written = 0

    def flush() -> None:
        nonlocal rows, written
        if not rows:
            return
        observation = torch.cat(obs_parts, dim=0).numpy()
        goal = torch.cat(goal_parts, dim=0).numpy()
        assert observation.shape == (rows, 930), observation.shape
        assert goal.shape == (rows, 730), goal.shape
        if not (np.isfinite(observation).all() and np.isfinite(goal).all()):
            raise ValueError("Non-finite physics observation or goal in rollout")
        dest = args.output_dir / f"physics_{written:05d}.npz"
        temp = args.output_dir / f"physics_{written:05d}.tmp.npz"
        np.savez(temp, obs_hist=observation.astype(np.float32, copy=False),
                 goal=goal.astype(np.float32, copy=False))
        os.replace(temp, dest)
        print(f"shard={written} rows={rows} saved={dest}", flush=True)
        written += 1
        rows = 0
        obs_parts.clear()
        goal_parts.clear()

    try:
        obs, _ = env.reset()
        for step in range(args.steps):
            hist, goal = obs["policy"], obs["goal"]
            if hist.ndim != 2 or hist.shape != (args.num_envs, 930):
                raise ValueError(f"Invalid native policy observation: {tuple(hist.shape)}")
            if goal.shape != (args.num_envs, 730):
                raise ValueError(f"Invalid reference goal: {tuple(goal.shape)}")
            with torch.inference_mode():
                action = policy(hist, goal)
            obs_parts.append(hist.detach().to("cpu"))
            goal_parts.append(goal.detach().to("cpu"))
            rows += args.num_envs
            obs, _, _, _, _ = env.step(action)
            if rows >= args.rows_per_shard:
                flush()
        flush()
    finally:
        env.close()
    print(f"COLLECTED: {written} real-physics shards; {args.steps * args.num_envs} rows", flush=True)


if __name__ == "__main__":
    main()