# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path("/data/workspace/mjlab-build/realbench_20261029")
summary = json.loads((base / "summary.json").read_text())
series = json.loads((base / "iteration_series.json").read_text())
fig, axs = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")
colors = ["#1873a8", "#d56b24"]
for color, (label, rows) in zip(colors, series.items()):
    r = summary[label]
    name = f"{r['world_size']} GPU, {r['envs_per_rank']} env/GPU"
    frames = r["world_size"] * r["envs_per_rank"] * r["rollout_steps"]
    axs[0].plot(
        [x["epoch"] for x in rows],
        [frames / x["compute_seconds"] for x in rows],
        label=name,
        color=color,
        lw=1.4,
    )
    axs[1].plot(
        [x["epoch"] for x in rows],
        [x["rollout_seconds"] for x in rows],
        color=color,
        label=name + " rollout",
        lw=1.3,
    )
    axs[1].plot(
        [x["epoch"] for x in rows],
        [x["update_seconds"] for x in rows],
        color=color,
        ls="--",
        label=name + " optimizer",
        lw=1.3,
    )
    p = base / label
    telemetry = [json.loads(s) for s in (p / "gpu.jsonl").read_text().splitlines()]
    result = json.loads((p / "result.json").read_text())
    telemetry.append(dict(elapsed=result["seconds"], gpus=result["final_gpus"]))
    launch = json.loads((p / "launch.json").read_text())
    baseline = json.loads((p / "baseline.json").read_text())
    selected = launch["limits"]["gpus"].split(",")
    axs[2].plot(
        [t["elapsed"] for t in telemetry],
        [
            max(
                t.get("estimated_owned_growth_mib", {}).get(
                    g, t["gpus"][g]["used"] - baseline[g]["used"]
                )
                for g in selected
            )
            / 1024
            for t in telemetry
        ],
        color=color,
        label=name + " max per-card growth",
        lw=1.4,
    )
axs[0].axvspan(0, 10, color="grey", alpha=0.12, label="Warmup excluded")
axs[0].set(
    ylabel="Compute transitions / second",
    xlabel="Training iteration",
    title="ASTRO-SUPERMIMIC P2: shared H200 training validation",
)
axs[1].set(ylabel="Seconds / iteration", xlabel="Training iteration")

axs[2].set(
    ylabel="Estimated GPU memory growth (GiB)", xlabel="Seconds since process launch"
)
for ax in axs:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
fig.savefig(base / "benchmark.png", dpi=170)
