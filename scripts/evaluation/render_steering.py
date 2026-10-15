#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare, render, verify and compare Stage 3 steering policies."""

import argparse
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from protomotions.utils.steering_review import (  # noqa: E402
    SteeringProtocol,
    VIEWS,
    prepare_snapshot,
    verify_snapshot,
)


def add_prepare_args(parser):
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Full Stage 3 last.ckpt with reference state",
    )
    parser.add_argument(
        "--motion-file",
        required=True,
        type=Path,
        help="Physical packed .pt shard, not a slurmrank placeholder",
    )
    parser.add_argument("--motion-ids", nargs="+", type=int, default=[0, 2, 10, 100])
    parser.add_argument(
        "--prior-checkpoint",
        type=Path,
        help="Explicit override if the frozen prior path has moved",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New snapshot directory; render artifacts go into its render/ child",
    )


def add_render_args(parser):
    parser.add_argument(
        "--gpu",
        type=int,
        help="Physical GPU for the inference child; otherwise inherit CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--views", nargs="+", choices=VIEWS, default=["turn_left", "speed_stop"]
    )
    parser.add_argument(
        "--video-motion-id",
        type=int,
        help="Original motion ID to show; default first --motion-ids entry",
    )
    for field in fields(SteeringProtocol):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default),
            default=field.default,
        )


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="CPU-only portable policy/prior/config/motion snapshot"
    )
    add_prepare_args(prepare)
    run = commands.add_parser(
        "run", help="Prepare a fresh snapshot, render, and verify reports"
    )
    add_prepare_args(run)
    add_render_args(run)
    render = commands.add_parser(
        "render",
        help="Render an existing snapshot, including one copied from another host",
    )
    render.add_argument("--prepared-dir", required=True, type=Path)
    add_render_args(render)
    report = commands.add_parser(
        "report", help="CPU-only regenerate plots and fully decode videos"
    )
    report.add_argument("--render-dir", required=True, type=Path)
    compare = commands.add_parser("compare", help="Compare two matched render/ folders")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)
    return parser


def render_snapshot(directory, args):
    import torch

    protocol = SteeringProtocol(
        **{field.name: getattr(args, field.name) for field in fields(SteeringProtocol)}
    ).validate()
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be nonnegative")
    if len(set(args.views)) != len(args.views):
        raise ValueError("Do not repeat video views")
    directory = Path(directory).resolve()
    manifest = verify_snapshot(directory)
    ids = manifest["initial_motion_ids"]
    video_id = args.video_motion_id if args.video_motion_id is not None else ids[0]
    if video_id not in ids:
        raise ValueError(
            "--video-motion-id must be one of this snapshot's initial motion IDs"
        )
    output = directory / "render"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Render output exists; use a fresh snapshot directory: {output}"
        )
    runtime = output / "runtime"
    runtime.mkdir(parents=True)
    try:
        os.link(directory / "policy/last.ckpt", runtime / "last.ckpt")
    except OSError:
        shutil.copy2(directory / "policy/last.ckpt", runtime / "last.ckpt")
    for name in ["resolved_configs.pt", "resolved_configs_inference.pt"]:
        config = torch.load(
            directory / "policy" / name, map_location="cpu", weights_only=False
        )
        config["agent"].pretrained_modules["prior"].checkpoint_path = str(
            directory / "prior/prior.ckpt"
        )
        torch.save(config, runtime / name)
    request = {
        "kind": "steering",
        "protocol": asdict(protocol),
        "views": args.views,
        "initial_motion_ids": ids,
        "video_motion_index": ids.index(video_id),
        "motion_file": str(directory / "selected_motions.pt"),
        "scene_file": None,
        "render_motion_file": str(directory / "selected_motions.pt"),
        "render_motion_id": 0,
        "wandb_video_motion_id": 0,
        "wandb_video_fps": protocol.fps,
        "wandb_video_duration": protocol.duration,
        "wandb_video_width": protocol.width,
        "wandb_video_height": protocol.height,
        "trigger_epoch": manifest["epoch"],
        "output": str(output / "steering.mp4"),
        "selected_motion_sha256": manifest["files"]["selected_motions.pt"]["sha256"],
        "prior_sha256": manifest["files"]["prior/prior.ckpt"]["sha256"],
    }
    request_file = output / "request.json"
    request_file.write_text(json.dumps(request, indent=2))
    environment = os.environ.copy()
    for key in [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]:
        environment.pop(key, None)
    environment["WANDB_MODE"] = "disabled"
    environment.setdefault("OMP_NUM_THREADS", "4")
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    command = [
        sys.executable,
        "-m",
        "protomotions.inference_agent",
        "--simulator",
        "isaaclab",
        "--headless",
        "--num-envs",
        str(8 * len(ids)),
        "--checkpoint",
        str(runtime / "last.ckpt"),
        "--video-request",
        str(request_file),
    ]
    print("Launching:", " ".join(command), flush=True)
    print("Inference log:", output / "render.log", flush=True)
    with (output / "render.log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    from protomotions.utils.steering_report import report

    print(json.dumps(report(output), indent=2))
    print("Review complete:", output)


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("prepare", "run"):
            if args.command == "run":
                SteeringProtocol(
                    **{
                        field.name: getattr(args, field.name)
                        for field in fields(SteeringProtocol)
                    }
                ).validate()
            manifest = prepare_snapshot(
                args.checkpoint,
                args.motion_file,
                args.motion_ids,
                args.output_dir,
                args.prior_checkpoint,
            )
            print(json.dumps(manifest, indent=2), flush=True)
            if args.command == "run":
                render_snapshot(args.output_dir, args)
        elif args.command == "render":
            render_snapshot(args.prepared_dir, args)
        elif args.command == "report":
            from protomotions.utils.steering_report import report

            print(json.dumps(report(args.render_dir), indent=2))
        elif args.command == "compare":
            from protomotions.utils.steering_report import compare

            print(
                json.dumps(
                    compare(args.baseline, args.candidate, args.output_dir), indent=2
                )
            )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
