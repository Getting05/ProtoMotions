# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen SuperMimic FSQ tracker bridge for GPC Stage 2.

Inputs are SuperMimic's *native* tensors, not ProtoMotions max-coords:
  supermimic_obs_hist: [B, 10 * 93], oldest frame first
  supermimic_goal:     [B, 10 * 73], future motion reference frames

The 93 features are joint position offsets (29), joint velocities (29),
base angular velocity (3), projected gravity (3), and previous action (29).
The tracker uses 40 FSQ scalars with 9 centered levels. Its decoder returns
29 normalized P2 joint-position actions; no PD rescaling is applied here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common.latent import LATENT_KEY


OBS_KEY = "supermimic_obs_hist"
GOAL_KEY = "supermimic_goal"
OBS_DIM = 93
HISTORY = 10
GOAL_DIM = 730
FSQ_DIM = 40
FSQ_LEVELS = 9
ACTION_DIM = 29


@dataclass
class SuperMimicFSQTrackerConfig:
    _target_: str = "protomotions.agents.common.supermimic_fsq.SuperMimicFSQTracker"
    checkpoint_path: str = ""
    history: int = HISTORY
    obs_dim: int = OBS_DIM
    goal_dim: int = GOAL_DIM
    # Loading a user-provided SuperMimic training checkpoint requires Python
    # pickle for its config/rank-state metadata. Never load untrusted files.
    trust_checkpoint: bool = False


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Module:
    layers: list[nn.Module] = []
    for width in hidden:
        layers.extend([nn.Linear(input_dim, width), nn.ELU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    container = nn.Module()
    container.net = nn.Sequential(*layers)
    return container


class _RunningNorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var + 1e-4).sqrt()


class SuperMimicFSQQuantizer(nn.Module):
    num_fsq_levels = FSQ_LEVELS
    num_fsq_scalars = FSQ_DIM

    def quantize(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != FSQ_DIM:
            raise ValueError(f"Expected {FSQ_DIM} FSQ scalars, got {latent.shape[-1]}")
        return latent.float().round().clamp(-4, 4)

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantize(codes).long() + 4

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] != FSQ_DIM:
            raise ValueError(f"Expected {FSQ_DIM} FSQ indices, got {indices.shape[-1]}")
        if torch.any((indices < 0) | (indices >= FSQ_LEVELS)):
            raise ValueError("FSQ index must be in [0, 8]")
        return indices.float() - 4


class SuperMimicFSQEncoder(ProtoMotionsTensorDictModule):
    def __init__(self, goal_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.in_keys = [GOAL_KEY]
        self.out_keys = [LATENT_KEY]
        self.mlp = _mlp(goal_dim, hidden, FSQ_DIM)
        self.register_buffer("scale", torch.ones(goal_dim))
        self.goal_norm = _RunningNorm(goal_dim)

    def forward(self, tensordict: TensorDict, log_internals: bool = False) -> TensorDict:
        goal = tensordict[GOAL_KEY]
        if goal.ndim != 2 or goal.shape[-1] != self.scale.numel():
            raise ValueError(f"{GOAL_KEY} must be [B, {self.scale.numel()}], got {tuple(goal.shape)}")
        latent = torch.tanh(self.mlp.net(self.goal_norm(goal) * self.scale))
        # Exact SuperMimic FSQ convention: nine centered levels, eps=1e-4.
        bounded = latent.float() * (4.5 * (1 - 1e-4))
        tensordict[LATENT_KEY] = bounded.round()
        return tensordict


class SuperMimicFSQDecoder(ProtoMotionsTensorDictModule):
    def __init__(self, history: int, obs_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.in_keys = [LATENT_KEY, OBS_KEY]
        self.out_keys = ["action"]
        self.history, self.obs_dim = history, obs_dim
        self.mlp = _mlp(FSQ_DIM + history * obs_dim, hidden, ACTION_DIM)
        self.register_buffer("scale", torch.ones(history * obs_dim))
        self.obs_norm = _RunningNorm(obs_dim)

    def forward(self, tensordict: TensorDict, log_internals: bool = False) -> TensorDict:
        history = tensordict[OBS_KEY]
        codes = tensordict[LATENT_KEY]
        if history.ndim != 2 or history.shape[-1] != self.history * self.obs_dim:
            raise ValueError(
                f"{OBS_KEY} must be [B, {self.history * self.obs_dim}], "
                f"got {tuple(history.shape)}"
            )
        if codes.ndim != 2 or codes.shape[-1] != FSQ_DIM:
            raise ValueError(f"FSQ codes must be [B, {FSQ_DIM}]")
        normalized = self.obs_norm(history.reshape(-1, self.history, self.obs_dim))
        flattened = normalized.flatten(1) * self.scale
        tensordict["action"] = self.mlp.net(torch.cat([codes, flattened], dim=-1))
        return tensordict


class SuperMimicFSQTracker(nn.Module):
    """Expose the original policy as encoder/quantizer/decoder to GPC Stage 2."""

    def __init__(self, config: SuperMimicFSQTrackerConfig):
        super().__init__()
        if (config.history, config.obs_dim, config.goal_dim) != (HISTORY, OBS_DIM, GOAL_DIM):
            raise ValueError("This adapter only supports native SuperMimic P2 10x93/730.")
        hidden = (2048, 1024, 512, 256)
        self.encoder = SuperMimicFSQEncoder(config.goal_dim, hidden)
        self.decoder = SuperMimicFSQDecoder(config.history, config.obs_dim, hidden)
        self.quantizer = SuperMimicFSQQuantizer()
        self.latent_key = LATENT_KEY
        if config.checkpoint_path:
            self.load_supermimic_checkpoint(config.checkpoint_path, trusted=config.trust_checkpoint)

    def load_supermimic_checkpoint(self, path: str, *, trusted: bool) -> None:
        if not trusted:
            raise ValueError("Set trust_checkpoint=True only for your own verified SuperMimic ckpt.")
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        # SuperMimic stores NumPy/pathlib objects in its training-state metadata.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != "vibed-parity-state71-v1":
            raise ValueError("Unsupported SuperMimic checkpoint format.")
        cfg = checkpoint.get("config", {}).get("policy", {})
        if not cfg.get("fsq") or cfg.get("policy_obs_ref_rot6d", True):
            raise ValueError("Expected FSQ with the original 93-dimensional no-ref6d observation.")
        if cfg.get("history") != HISTORY or cfg.get("goal_dim") != GOAL_DIM:
            raise ValueError("Checkpoint history/goal dimensions do not match.")
        if tuple(cfg.get("encoder_hidden", ())) != (2048, 1024, 512, 256):
            raise ValueError("Unsupported encoder MLP architecture.")
        if tuple(cfg.get("decoder_hidden", ())) != (2048, 1024, 512, 256):
            raise ValueError("Unsupported decoder MLP architecture.")
        policy = checkpoint["policy"]

        def _sub(prefix: str, module: nn.Module) -> None:
            subset = {key[len(prefix):]: value for key, value in policy.items()
                      if key.startswith(prefix)}
            module.load_state_dict(subset, strict=True)

        _sub("encoders.motion.mlp.", self.encoder.mlp)
        _sub("decoder.mlp.", self.decoder.mlp)
        _sub("goal_norm.", self.encoder.goal_norm)
        _sub("obs_norm.", self.decoder.obs_norm)
        for name, module, key in (
            ("encoder scale", self.encoder, "encoders.motion.scale"),
            ("decoder scale", self.decoder, "decoder.scale"),
        ):
            if key not in policy:
                raise ValueError(f"Missing {name}: {key}")
            target = module.scale
            if target.shape != policy[key].shape:
                raise ValueError(f"{name} shape mismatch.")
            target.copy_(policy[key])
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)