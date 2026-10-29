# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import torch

torch.set_num_threads(1)
repo = Path("/data/workspace/ProtoMotions")
base = Path("/data/workspace/mjlab-build/realbench_20261029")
report = {}


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for x in value.values():
            yield from tensors(x)
    elif isinstance(value, (list, tuple)):
        for x in value:
            yield from tensors(x)


for label in (
    "mjlab_astro512_single256_20261029",
    "mjlab_astro512_dual256_budget4_20261029",
):
    path = repo / "results" / label / "last.ckpt"
    if not path.exists():
        continue
    data = torch.load(path, map_location="cpu", weights_only=True)
    for tensor in tensors(data):
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), label
    steps = []
    for state in data["critic_optimizer"]["state"].values():
        if "step" in state:
            steps.append(float(state["step"]))
    assert data["epoch"] == 100 and min(steps) > 0
    inference = path.with_name("inference_last.ckpt")
    inf = torch.load(inference, map_location="cpu", weights_only=True)
    for tensor in tensors(inf):
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all()
    report[label] = dict(
        epoch=data["epoch"],
        step_count=data["step_count"],
        all_checkpoint_tensors_finite=True,
        critic_optimizer_step_min=min(steps),
        critic_optimizer_step_max=max(steps),
        training_checkpoint=str(path),
        training_bytes=path.stat().st_size,
        inference_checkpoint=str(inference),
        inference_bytes=inference.stat().st_size,
    )
(base / "checkpoint_verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
