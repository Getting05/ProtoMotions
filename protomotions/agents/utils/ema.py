# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local, detached EMA state for a materialized optimization module."""
from contextlib import contextmanager

import torch

from protomotions.agents.utils.normalization import RunningMeanStd


class ModelEMA:
    """Average trainable parameters/float buffers; copy counters and RMS state.

    This is deliberately not an nn.Module: shadows are never registered with DDP
    or the optimizer. Construct after Fabric setup has synchronized parameters.
    RunningMeanStd already aggregates across ranks; copying its full state keeps
    mean, variance and count coherent without smoothing sufficient statistics.
    """

    def __init__(self, module, decay=1.0):
        if not 0 <= decay <= 1:
            raise ValueError("EMA decay must be in [0, 1]")
        self.decay = float(decay)
        self.num_updates = 0
        self._active = False
        params = {k: v for k, v in module.named_parameters() if v.requires_grad}
        buffers = dict(module.named_buffers())
        self.names = tuple(dict(params, **buffers))
        self.copy_names = {
            f"{prefix}.{name}" if prefix else name
            for prefix, child in module.named_modules()
            if isinstance(child, RunningMeanStd)
            for name, _ in child.named_buffers()
        }
        self.shadow = {k: v.detach().clone() for k, v in dict(params, **buffers).items()}

    def _values(self, module):
        values = dict(module.named_parameters())
        values.update(module.named_buffers())
        return {k: values[k].detach() for k in self.names}

    @torch.no_grad()
    def update(self, module):
        if self._active:
            raise RuntimeError("Cannot update EMA while evaluation weights are installed")
        values = self._values(module)
        # foreach limits launch overhead for the large Transformer parameter set.
        groups = {}
        for name, value in values.items():
            shadow = self.shadow[name]
            if name in self.copy_names or not value.is_floating_point():
                shadow.copy_(value)
            else:
                dst, src = groups.setdefault((value.device, value.dtype), ([], []))
                dst.append(shadow)
                src.append(value)
        for dst, src in groups.values():
            torch._foreach_lerp_(dst, src, 1.0 - self.decay)
        self.num_updates += 1

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates,
                "shadow": {k: v.detach().clone() for k, v in self.shadow.items()}}

    @torch.no_grad()
    def load_state_dict(self, state):
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA checkpoint keys do not match the current prior")
        for name, value in state["shadow"].items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA checkpoint shape mismatch: {name}")
            self.shadow[name].copy_(value)
        self.num_updates = int(state["num_updates"])
        # The resolved training config controls decay, including explicit warm starts.

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, module):
        if self._active:
            raise RuntimeError("Nested EMA weight swaps are not supported")
        values = self._values(module)
        original = {k: v.clone() for k, v in values.items()}
        modes = {child: child.training for child in module.modules()}
        self._active = True
        try:
            for name, value in values.items():
                value.copy_(self.shadow[name])
            module.eval()
            yield
        finally:
            for name, value in values.items():
                value.copy_(original[name])
            for child, training in modes.items():
                child.training = training
            self._active = False
