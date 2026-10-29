# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import statistics
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = Path("/data/workspace/mjlab-build/realbench_20261029")
repo = Path("/data/workspace/ProtoMotions")
report = {}
keys = [
    "info/episode_length",
    "info/episode_reward",
    "rewards/task_rewards",
    "rewards/unnormalized_task_rewards",
    "actor/internals/perplexity",
    "actor/bad_grads_count",
    "critic/bad_grads_count",
]
for label in (
    "mjlab_astro512_single256_20261029",
    "mjlab_astro512_dual256_budget4_20261029",
):
    dirs = sorted(
        {p.parent for p in (repo / "results" / label).rglob("events.out.tfevents.*")}
    )
    values = {k: [] for k in keys}
    for directory in dirs:
        acc = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
        for k in keys:
            if k in tags:
                values[k].extend(
                    dict(step=x.step, value=x.value) for x in acc.Scalars(k)
                )
    report[label] = {
        k: dict(
            count=len(v),
            first=v[0],
            last=v[-1],
            last10_mean=statistics.mean(x["value"] for x in v[-10:]),
        )
        for k, v in values.items()
        if v
    }
(base / "training_metrics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
