#!/usr/bin/env python3
"""Single-node DDP Stage2 prior training on genuine SuperMimic FSQ93 rollout shards."""
from __future__ import annotations

import argparse
from datetime import timedelta
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from protomotions.agents.common.latent import LATENT_LOGITS_KEY, TARGET_LATENT_KEY
from protomotions.agents.common.supermimic_fsq import GOAL_KEY, OBS_KEY
from protomotions.agents.supervised.latent_prior_model import DiscreteAutoregressiveLatentPriorModel
from train_gpc_stage2_supermimic_fsq93 import build_config, save


def read_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        if "obs_hist" not in d or "goal" not in d:
            raise ValueError(f"{path}: missing obs_hist or goal")
        obs, goal = d["obs_hist"], d["goal"]
    if not (obs.ndim == goal.ndim == 2 and len(obs) == len(goal) and
            obs.shape[1] == 930 and goal.shape[1] == 730 and len(obs) > 0):
        raise ValueError(f"Invalid real rollout shard {path}: {obs.shape}/{goal.shape}")
    if not (np.isfinite(obs).all() and np.isfinite(goal).all()):
        raise ValueError(f"{path}: non-finite rollout values")
    return obs, goal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gpc-stage2-fsq93-ddp"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64, help="Per GPU batch size")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ff-size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fsq-scalars-per-token", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.tracker_checkpoint.is_file():
        parser.error(f"Missing tracker checkpoint: {args.tracker_checkpoint}")
    if args.d_model % args.num_heads or 40 % args.fsq_scalars_per_token:
        parser.error("Invalid transformer heads or FSQ packing.")
    if args.batch_size < 1 or args.steps_per_epoch < 1:
        parser.error("batch size and steps per epoch must be positive")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("DDP Stage2 requires CUDA.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    try:
        all_files = sorted(args.data_dir.glob("*.npz"))
        if len(all_files) < world_size:
            raise ValueError(f"At least {world_size} completed real .npz shards required; got {len(all_files)}")
        own_files = all_files[rank::world_size]
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed + rank)
        config = build_config(args)
        model = DiscreteAutoregressiveLatentPriorModel(config).to(device).train()
        assert not any(p.requires_grad for p in model.latent_decoder.parameters())
        assert not any(p.requires_grad for p in model.target_latent_encoder.parameters())

        # Materialize possible LazyLinear parameters before wrapping the prior in DDP.
        obs_np, goal_np = read_shard(own_files[0])
        first_batch = TensorDict({
            OBS_KEY: torch.from_numpy(obs_np[:1].copy()).to(device),
            GOAL_KEY: torch.from_numpy(goal_np[:1].copy()).to(device)
        }, batch_size=[1], device=device)
        with torch.no_grad():
            model.target_latent_encoder(first_batch)
            model.prior(model._prior_tensordict(first_batch, first_batch[TARGET_LATENT_KEY]))
        wrapped_prior = DDP(model.prior, device_ids=[local_rank], broadcast_buffers=True)
        optimizer = torch.optim.AdamW(model.optimization_module().parameters(),
                                      lr=args.lr, weight_decay=0.01)
        step = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = int(checkpoint["step"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            print(f"DDP Stage2 world_size={world_size}, shards={len(all_files)}, "
                  f"FSQ tokens={model.latent_tokenization.num_prior_tokens}, "
                  f"vocab={model.latent_tokenization.prior_token_vocab_size}, "
                  f"batch_per_gpu={args.batch_size}, starting_step={step}", flush=True)
        cached_path: Path | None = None
        cache_obs, cache_goal = obs_np, goal_np
        for epoch in range(args.epochs):
            for batch_num in range(args.steps_per_epoch):
                file_idx = ((epoch * args.steps_per_epoch + batch_num) // 32) % len(own_files)
                path = own_files[file_idx]
                if path != cached_path:
                    cache_obs, cache_goal = read_shard(path)
                    cached_path = path
                idx = rng.integers(0, len(cache_obs), size=args.batch_size)
                obs_t = torch.from_numpy(np.ascontiguousarray(cache_obs[idx])).to(device)
                goal_t = torch.from_numpy(np.ascontiguousarray(cache_goal[idx])).to(device)
                batch = TensorDict({OBS_KEY: obs_t, GOAL_KEY: goal_t},
                                   batch_size=[args.batch_size], device=device)
                with torch.no_grad():
                    model.target_latent_encoder(batch)
                target = batch[TARGET_LATENT_KEY].long()
                prior_batch = model._prior_tensordict(batch, target)
                logits = wrapped_prior(prior_batch)[LATENT_LOGITS_KEY]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       target.reshape(-1),
                                       label_smoothing=args.label_smoothing)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"rank={rank}, step={step}: non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.optimization_module().parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                if step == 1 or step % args.log_every == 0:
                    with torch.no_grad():
                        acc = (logits.argmax(-1) == target).float().mean()
                        metrics = torch.stack((loss.detach(), acc))
                        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                        metrics /= world_size
                    if rank == 0:
                        print(f"epoch={epoch} step={step} loss={metrics[0].item():.5f} "
                              f"token_acc={metrics[1].item():.4f}", flush=True)
                if step % args.save_every == 0:
                    if rank == 0:
                        save(args.output_dir, model, optimizer, config, step, args)
                if args.max_steps is not None and step >= args.max_steps:
                    if rank == 0:
                        save(args.output_dir, model, optimizer, config, step, args)
                    dist.barrier()
                    return
        if rank == 0:
            save(args.output_dir, model, optimizer, config, step, args)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()