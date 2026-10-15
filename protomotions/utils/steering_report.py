# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only media verification and plots for the steering review tool."""

import csv
import json
from pathlib import Path

from protomotions.utils.steering_review import file_sha256


def report(directory):
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw

    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    data = np.load(directory / "telemetry.npz")
    protocol = metrics["protocol"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    repeats = len(metrics["initial_motion_ids"])
    for i, case in enumerate(metrics["cases"]):
        ax = axes.flat[i]
        values = data["speed_along"][:, i * repeats : (i + 1) * repeats].mean(1)
        window = min(len(values), max(1, round(0.4 / metrics["dt"])))
        ax.plot(
            data["time"][window - 1 :],
            np.convolve(values, np.ones(window) / window, mode="valid"),
            label="Actual",
            color="#245bc2",
        )
        ax.plot(
            data["time"],
            data["speed_command"][:, i * repeats],
            "--",
            label="Command",
            color="#cd4c36",
        )
        ax.set(
            title=f"{case['case']} | upright {case['survived']}/{repeats}",
            xlabel="Time (s)",
            ylabel="Speed (m/s)",
        )
        ax.grid(alpha=0.15)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        f"Steering review: epoch {metrics['epoch']}, {metrics['num_envs']} controlled trials"
    )
    fig.savefig(directory / "speed_tracking.png", dpi=160)
    plt.close(fig)
    with (directory / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["cases"][0]))
        writer.writeheader()
        writer.writerows(metrics["cases"])
    verification = []
    for view in metrics["views"]:
        path = directory / f"{view}.mp4"
        selected = set(np.linspace(0, metrics["frames"] - 1, 6, dtype=int).tolist())
        sheet = Image.new("RGB", (1440, 780), (20, 22, 26))
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 8),
            f"Epoch {metrics['epoch']} | view {view} | red length = target speed x {protocol['arrow_scale']} s",
            fill="white",
        )
        count = slot = 0
        with imageio.get_reader(path) as reader:
            for index, frame in enumerate(reader):
                count += 1
                if index in selected:
                    x, y = (slot % 3) * 480, 35 + (slot // 3) * 370
                    sheet.paste(Image.fromarray(frame).resize((480, 350)), (x, y))
                    draw.text(
                        (x + 10, y + 352), f"{index/metrics['fps']:.2f} s", fill="white"
                    )
                    slot += 1
        if count != metrics["frames"]:
            raise RuntimeError(
                f"Video decode count mismatch: {view}: {count} != {metrics['frames']}"
            )
        sheet.save(directory / f"preview_{view}.jpg", quality=92)
        verification.append(
            {
                "file": path.name,
                "decoded_frames": count,
                "fps": metrics["fps"],
                "duration_seconds": count / metrics["fps"],
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    (directory / "media_verification.json").write_text(
        json.dumps(verification, indent=2)
    )
    return verification


def compare(baseline, candidate, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline, candidate, output = Path(baseline), Path(candidate), Path(output)
    a, b = [json.loads((p / "metrics.json").read_text()) for p in [baseline, candidate]]
    # Rendering resolution/FPS/arrow scale can differ; the physical protocol cannot.
    for key in ["initial_motion_ids", "selected_motion_sha256", "prior_sha256", "dt"]:
        if a[key] != b[key]:
            raise ValueError(f"Cannot label this a matched comparison: {key} differs")
    for key in [
        "duration",
        "switch_time",
        "steady_start",
        "low_speed",
        "cruise_speed",
        "high_speed",
        "seed",
        "min_height",
        "min_upright",
        "fall_hold",
    ]:
        if a["protocol"][key] != b["protocol"][key]:
            raise ValueError(
                f"Cannot label this a matched comparison: protocol.{key} differs"
            )
    if [v["case"] for v in a["cases"]] != [v["case"] for v in b["cases"]]:
        raise ValueError("Comparison case order differs")
    output.mkdir(parents=True, exist_ok=True)
    x, y = np.load(baseline / "telemetry.npz"), np.load(candidate / "telemetry.npz")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    repeats = len(a["initial_motion_ids"])
    rows = []
    for i, (old, new) in enumerate(zip(a["cases"], b["cases"])):
        ax = axes.flat[i]
        for data, run, color in [(x, a, "#8a919d"), (y, b, "#245bc2")]:
            values = data["speed_along"][:, i * repeats : (i + 1) * repeats].mean(1)
            window = min(len(values), max(1, round(0.4 / run["dt"])))
            ax.plot(
                data["time"][window - 1 :],
                np.convolve(values, np.ones(window) / window, mode="valid"),
                color=color,
                label=f"Epoch {run['epoch']}",
            )
        ax.plot(
            y["time"],
            y["speed_command"][:, i * repeats],
            "--",
            color="#cd4c36",
            label="Command",
        )
        ax.set(
            title=f"{new['case']} | upright {old['survived']} -> {new['survived']}",
            xlabel="Time (s)",
            ylabel="Speed (m/s)",
        )
        ax.grid(alpha=0.15)
        rows.append({"case": new["case"], "baseline": old, "candidate": new})
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Matched steering comparison: same prior, initial motions, commands and seed"
    )
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    result = {
        "baseline_epoch": a["epoch"],
        "candidate_epoch": b["epoch"],
        "cases": rows,
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    return result
