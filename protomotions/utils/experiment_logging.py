# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging CLI and resume settings. Safe to import before the simulator/torch."""

import argparse
import json
from pathlib import Path


SWANLAB_DEFAULTS = {
    "use_swanlab": False,
    "swanlab_project": "physical_animation",
    "swanlab_workspace": None,
    "swanlab_mode": None,
}


def add_logging_arguments(parser):
    parser.add_argument(
        "--logging-backend",
        choices=("tensorboard", "wandb", "swanlab", "both"),
        default=argparse.SUPPRESS,
        help="Explicitly select cloud loggers, including on resume; TensorBoard is always kept",
    )
    parser.add_argument(
        "--use-swanlab",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable SwanLab alongside any enabled W&B logger (default: disabled)",
    )
    parser.add_argument("--swanlab-project", default=argparse.SUPPRESS)
    parser.add_argument("--swanlab-workspace", default=argparse.SUPPRESS)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "offline", "local", "disabled"),
        default=argparse.SUPPRESS,
        help="Defaults to SwanLab SDK/environment settings",
    )


def explicit_logging_options(args):
    return {
        key: getattr(args, key)
        for key in (*SWANLAB_DEFAULTS, "logging_backend")
        if hasattr(args, key)
    }


def resolve_logging_options(args, explicit, *, migrate_video=False):
    """Only explicit new flags override saved arguments; legacy W&B stays as-is."""
    old_destination = (
        getattr(args, "swanlab_project", SWANLAB_DEFAULTS["swanlab_project"]),
        getattr(args, "swanlab_workspace", None),
    )
    for key, default in SWANLAB_DEFAULTS.items():
        setattr(args, key, explicit.get(key, getattr(args, key, default)))
    backend = explicit.get("logging_backend")
    if backend is not None:
        args.use_wandb = backend in ("wandb", "both")
        args.use_swanlab = backend in ("swanlab", "both")
        # Keep recording enabled when switching away from a saved backend.
        if migrate_video and (
            getattr(args, "wandb_video", False) or getattr(args, "swanlab_video", False)
        ):
            args.wandb_video = args.use_wandb
            args.swanlab_video = args.use_swanlab
    # The selector is a one-shot CLI override, not another source of saved state.
    vars(args).pop("logging_backend", None)
    if old_destination != (args.swanlab_project, args.swanlab_workspace):
        args.swanlab_id = None


def logging_cli_options(args):
    result = []
    for key, value in explicit_logging_options(args).items():
        if key == "use_swanlab":
            result.append("--use-swanlab" if value else "--no-use-swanlab")
        elif value is not None:
            result.append(f"--{key.replace('_', '-')}={value}")
    return result


def build_swanlab_logger_config(args, save_dir):
    return {
        "_target_": "protomotions.utils.swanlab_logger.SwanLabLogger",
        "experiment_name": args.experiment_name,
        "save_dir": str(save_dir),
        "project": args.swanlab_project,
        "workspace": args.swanlab_workspace,
        "mode": args.swanlab_mode,
        "id": getattr(args, "swanlab_id", None),
    }


def is_swanlab_logger(logger):
    return getattr(logger, "_protomotions_backend", None) == "swanlab"


def capture_swanlab_id(args, loggers):
    for logger in loggers:
        if is_swanlab_logger(logger):
            # Initialize on rank zero before persisting config, even if there are
            # no metrics yet. A resumed run must retain its actual SDK run ID.
            args.swanlab_id = logger.experiment.id


def persist_logging_options(save_dir, args):
    """Update only logging settings, retaining all saved training arguments."""
    path = Path(save_dir) / "config.yaml"
    saved = json.loads(path.read_text())
    for key in (*SWANLAB_DEFAULTS, "use_wandb", "swanlab_id", "wandb_id"):
        if hasattr(args, key):
            saved[key] = getattr(args, key)
    temporary = path.with_suffix(".logging.tmp")
    temporary.write_text(json.dumps(saved, indent=2))
    temporary.replace(path)
