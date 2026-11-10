# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import statistics
from pathlib import Path

base = Path("/data/workspace/mjlab-build/realbench_20261029")
labels = [
    "mjlab_astro512_single256_20261029",
    "mjlab_astro512_dual256_budget4_20261029",
]


def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def stats(values):
    s = sorted(values)
    return dict(
        mean=statistics.mean(s),
        median=statistics.median(s),
        min=min(s),
        max=max(s),
        p10=s[int(0.1 * (len(s) - 1))],
        p90=s[int(0.9 * (len(s) - 1))],
    )


results = {}
series = {}
for label in labels:
    p = base / label
    if not (p / "result.json").exists():
        continue
    launch = json.loads((p / "launch.json").read_text())
    result = json.loads((p / "result.json").read_text())
    baseline = json.loads((p / "baseline.json").read_text())
    telemetry = lines(p / "gpu.jsonl")
    rank_logs = [lines(path) for path in sorted(p.glob("rank[0-9]*.jsonl"))]
    begins = [next(r for r in logs if r["event"] == "fit_start") for logs in rank_logs]
    iterations = [
        {r["epoch"]: r for r in logs if r["event"] == "iteration"} for logs in rank_logs
    ]
    walls = [
        {r["epoch"]: r["seconds"] for r in logs if r["event"] == "iteration_wall"}
        for logs in rank_logs
    ]
    epochs = sorted(set.intersection(*(set(r) for r in iterations)))
    steady = [e for e in epochs if e > 10]
    world = len(rank_logs)
    nenv = begins[0]["num_envs"]
    steps = begins[0]["num_steps"]
    frames = world * nenv * steps
    samples = [
        dict(
            epoch=e,
            compute_seconds=max(rank[e]["compute_seconds"] for rank in iterations),
            rollout_seconds=max(rank[e]["rollout_seconds"] for rank in iterations),
            update_seconds=max(rank[e]["update_seconds"] for rank in iterations),
            wall_seconds=max(rank[e] for rank in walls),
        )
        for e in epochs
    ]
    selected = [s for s in samples if s["epoch"] > 10]
    allrows = [row for ranks in iterations for row in ranks.values()]
    metrics = {
        key: [r["losses"][key] for r in allrows if key in r["losses"]]
        for key in [
            "losses/actor_loss",
            "losses/critic_loss",
            "actor/approx_kl",
            "actor/grad_norm_before_clip",
            "critic/grad_norm_before_clip",
            "actor/bad_grads_count",
            "critic/bad_grads_count",
            "actor/update_skipped",
        ]
    }
    gpus = launch["limits"]["gpus"].split(",")
    memory = {
        g: dict(
            peak_total_used_MiB=max(t["gpus"][g]["used"] for t in telemetry),
            baseline_used_MiB=baseline[g]["used"],
            peak_increment_MiB=max(
                t.get("estimated_owned_growth_mib", {}).get(
                    g, t["gpus"][g]["used"] - baseline[g]["used"]
                )
                for t in telemetry
            ),
            growth_method="reference corrected estimate"
            if "estimated_owned_growth_mib" in telemetry[0]
            else "raw baseline difference",
            final_used_MiB=result["final_gpus"][g]["used"],
            minimum_free_MiB=min(t["gpus"][g]["free"] for t in telemetry),
        )
        for g in gpus
    }
    summary = dict(
        label=label,
        world_size=world,
        envs_per_rank=nenv,
        batch_size_per_rank=begins[0]["batch_size"],
        rollout_steps=steps,
        iterations=len(epochs),
        warmup_excluded=10,
        steady_iterations=len(steady),
        global_transitions=frames * len(epochs),
        returncode=result["returncode"],
        abort_reason=result["abort_reason"],
        run_wall_seconds=result["seconds"],
        steady_compute_fps=frames
        * len(selected)
        / sum(s["compute_seconds"] for s in selected),
        steady_wall_fps=frames
        * len(selected)
        / sum(s["wall_seconds"] for s in selected),
        steady_rollout_fps=frames
        * len(selected)
        / sum(s["rollout_seconds"] for s in selected),
        compute_iteration_seconds=stats([s["compute_seconds"] for s in selected]),
        rollout_seconds=stats([s["rollout_seconds"] for s in selected]),
        update_seconds=stats([s["update_seconds"] for s in selected]),
        torch_peak_allocated_GiB=max(r["torch_peak_allocated_bytes"] for r in allrows)
        / 2**30,
        torch_peak_reserved_GiB=max(r["torch_peak_reserved_bytes"] for r in allrows)
        / 2**30,
        memory_by_physical_gpu=memory,
        loss_statistics={k: stats(v) for k, v in metrics.items() if v},
        mean_done_fraction=sum(r["done_count"] for r in allrows)
        / (frames * len(epochs)),
        mean_terminated_fraction=sum(r["terminated_count"] for r in allrows)
        / (frames * len(epochs)),
        max_end_rollout_abs_dof_vel=max(
            r["end_rollout_max_abs_dof_vel"] for r in allrows
        ),
        rank_fit_end=[
            next((r for r in log if r["event"] == "fit_end"), None) for log in rank_logs
        ],
        all_logged_loss_scalars_finite=all(
            math.isfinite(v) for r in allrows for v in r["losses"].values()
        ),
    )
    results[label] = summary
    series[label] = samples
if len(results) == 2:
    results["comparison"] = dict(
        compute_speedup=results[labels[1]]["steady_compute_fps"]
        / results[labels[0]]["steady_compute_fps"],
        wall_speedup=results[labels[1]]["steady_wall_fps"]
        / results[labels[0]]["steady_wall_fps"],
    )
(base / "summary.json").write_text(json.dumps(results, indent=2))
(base / "iteration_series.json").write_text(json.dumps(series, indent=2))
print(json.dumps(results, indent=2))
