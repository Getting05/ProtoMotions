#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPC Stage2 teacher-forced prior on native SuperMimic FSQ93 rollout shards.

The source tracker is frozen. Only the autoregressive GPC prior is optimized.
Collect .npz shards with export_supermimic_fsq93_rollouts.py in a SuperMimic
simulator environment. No motion-reference information enters the prior context:
the goal tensor is used *only* to form training labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    MLPLayerConfig,
    MLPWithConcatConfig,
    ModuleContainerConfig,
    PretrainedModelConfig,
)
from protomotions.agents.common.latent import LATENT_KEY, LATENT_LOGITS_KEY, TARGET_LATENT_KEY
from protomotions.agents.common.supermimic_fsq import (
    ACTION_DIM, GOAL_DIM, GOAL_KEY, HISTORY, OBS_DIM, OBS_KEY,
    SuperMimicFSQTrackerConfig,
)
from protomotions.agents.supervised.latent_prior_config import (
    DiscreteAutoregressiveLatentPriorModelConfig,
)
from protomotions.agents.supervised.latent_prior_model import DiscreteAutoregressiveLatentPriorModel


def build_config(args: argparse.Namespace) -> DiscreteAutoregressiveLatentPriorModelConfig:
    embed_dim = args.d_model
    context_encoder = ModuleContainerConfig(
        in_keys=[OBS_KEY],
        models=[
            MLPWithConcatConfig(
                in_keys=[OBS_KEY],
                out_keys=["context_embedding"],
                num_out=embed_dim,
                normalize_obs=True,
                layers=[MLPLayerConfig(units=embed_dim, activation="gelu")],
            )
        ],
    )
    prior = DiscreteAutoregressiveTransformerConfig(
        token_key="prior_tokens",
        logits_key=LATENT_LOGITS_KEY,
        generated_tokens_key=LATENT_KEY,
        context_encoder=context_encoder,
        d_model=embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_size=args.ff_size,
        dropout=args.dropout,
        activation="gelu",
        num_tokens=0,
        vocab_size=0,
    )
    return DiscreteAutoregressiveLatentPriorModelConfig(
        latent_decoder=PretrainedModelConfig(
            checkpoint_path="",
            module_config=SuperMimicFSQTrackerConfig(
                checkpoint_path=str(Path(args.tracker_checkpoint).resolve()),
                trust_checkpoint=True,
            ),
            freeze=True,
        ),
        prior=prior,
        fsq_scalars_per_prior_token=args.fsq_scalars_per_token,
        temperature=1.0,
        top_p=0.9,
    )


def iter_shards(dataset_dir: Path):
    paths = sorted(dataset_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No SuperMimic rollout .npz shards in {dataset_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "obs_hist" not in data or "goal" not in data:
                raise ValueError(f"{path}: expected obs_hist and goal arrays.")
            obs, goal = data["obs_hist"], data["goal"]
            if obs.ndim != 2 or goal.ndim != 2 or obs.shape != (goal.shape[0], HISTORY * OBS_DIM) or goal.shape[1] != GOAL_DIM:
                raise ValueError(f"{path}: invalid native SuperMimic shapes {obs.shape}, {goal.shape}")
            if obs.dtype.kind != "f" or goal.dtype.kind != "f":
                raise ValueError(f"{path}: samples must be floating point")
            if not np.isfinite(obs).all() or not np.isfinite(goal).all():
                raise ValueError(f"{path}: nonfinite observations/goals")
            yield path, obs, goal


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = build_config(args)
    model = DiscreteAutoregressiveLatentPriorModel(config).to(device).train()
    model.latent_decoder.eval()
    model.target_latent_encoder.eval()
    assert not any(p.requires_grad for p in model.latent_decoder.parameters())
    assert not any(p.requires_grad for p in model.target_latent_encoder.parameters())
    print(
        f"FSQ: {model.latent_tokenization.num_fsq_scalars} scalars x "
        f"{model.latent_tokenization.num_fsq_levels} levels -> "
        f"{model.latent_tokenization.num_prior_tokens} prior tokens, "
        f"vocab {model.latent_tokenization.prior_token_vocab_size}"
    )
    optimizer = torch.optim.AdamW(model.optimization_module().parameters(), lr=args.lr, weight_decay=0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0

    for epoch in range(args.epochs):
        for source, obs_cpu, goal_cpu in iter_shards(args.data_dir):
            for start in torch.randperm(len(obs_cpu)).split(args.batch_size):
                ids = start.numpy()
                obs = torch.from_numpy(np.ascontiguousarray(obs_cpu[ids])).to(device)
                goal = torch.from_numpy(np.ascontiguousarray(goal_cpu[ids])).to(device)
                batch = TensorDict({OBS_KEY: obs, GOAL_KEY: goal}, batch_size=[len(ids)], device=device)
                with torch.no_grad():
                    model.target_latent_encoder(batch)
                # No action label is required; avoid needless frozen-decoder rollout
                # during teacher-forced token CE. The decoder is verified separately.
                batch["action"] = torch.zeros(len(ids), ACTION_DIM, device=device)
                pred = model(batch)[LATENT_LOGITS_KEY]
                target = batch[TARGET_LATENT_KEY].long()
                loss = F.cross_entropy(
                    pred.reshape(-1, pred.shape[-1]),
                    target.flatten(),
                    label_smoothing=args.label_smoothing,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss for {source}, step={step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.optimization_module().parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                if step % args.log_every == 0 or step == 1:
                    acc = (pred.detach().argmax(-1) == target).float().mean().item()
                    print(f"epoch={epoch} step={step} loss={loss.item():.5f} token_acc={acc:.4f} data={source.name}", flush=True)
                if step % args.save_every == 0:
                    save(args.output_dir, model, optimizer, config, step, args)
                if args.max_steps is not None and step >= args.max_steps:
                    save(args.output_dir, model, optimizer, config, step, args)
                    return
    save(args.output_dir, model, optimizer, config, step, args)


def save(out_dir, model, optimizer, config, step, args):
    tmp = out_dir / "stage2_last.tmp"
    dest = out_dir / "stage2_last.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": config,
        "step": step,
        "source_tracker": str(args.tracker_checkpoint),
        "format_version": "gpc-supermimic-fsq93-stage2-v1",
    }, tmp)
    tmp.replace(dest)
    print(f"saved {dest} at step {step}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gpc-stage2-supermimic-fsq93"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if not args.tracker_checkpoint.is_file():
        parser.error(f"Missing tracker checkpoint: {args.tracker_checkpoint}")
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    if args.fsq_scalars_per_token < 1 or 40 % args.fsq_scalars_per_token:
        parser.error("--fsq-scalars-per-token must divide 40")
    train(args)


if __name__ == "__main__":
    main()