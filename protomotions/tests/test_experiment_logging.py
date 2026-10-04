# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging selection must survive saved configs and SLURM command forwarding."""

import argparse
import json
import shlex
from types import SimpleNamespace as NS

import pytest

from protomotions.utils.experiment_logging import (
    add_logging_arguments,
    build_swanlab_logger_config,
    capture_swanlab_id,
    explicit_logging_options,
    logging_cli_options,
    persist_logging_options,
    resolve_logging_options,
)
from protomotions.utils.wandb_video import (
    add_video_arguments,
    explicit_video_options,
    resolve_video_options,
    video_cli_options,
)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--use-wandb", action="store_true")
    add_logging_arguments(result)
    add_video_arguments(result)
    return result


def test_old_saved_wandb_options_are_unchanged_without_new_flags():
    cli = parser().parse_args([])
    explicit = explicit_logging_options(cli)
    assert explicit == {}
    cli.use_wandb = True  # Saved config was loaded.
    resolve_logging_options(cli, explicit)
    assert cli.use_wandb
    assert not cli.use_swanlab
    assert cli.swanlab_project == "physical_animation"


@pytest.mark.parametrize(
    "backend,wandb,swanlab",
    [
        ("swanlab", False, True),
        ("wandb", True, False),
        ("both", True, True),
        ("tensorboard", False, False),
    ],
)
def test_explicit_backend_overrides_resume_without_changing_training(
    backend, wandb, swanlab
):
    cli = parser().parse_args(["--logging-backend", backend])
    explicit = explicit_logging_options(cli)
    cli.__dict__.update(
        use_wandb=True,
        use_swanlab=False,
        wandb_video=True,
        seed=123,
        checkpoint="last.ckpt",
    )
    resolve_logging_options(cli, explicit, migrate_video=True)
    cli.simulator = "isaaclab"
    resolve_video_options(cli, {})
    assert (cli.use_wandb, cli.use_swanlab) == (wandb, swanlab)
    assert (cli.wandb_video, cli.swanlab_video) == (wandb, swanlab)
    assert (cli.seed, cli.checkpoint) == (123, "last.ckpt")
    assert not hasattr(cli, "logging_backend")


def test_swanlab_additive_flag_and_explicit_off():
    args = NS(use_wandb=True, use_swanlab=False)
    resolve_logging_options(args, {"use_swanlab": True})
    assert args.use_wandb and args.use_swanlab
    resolve_logging_options(args, {"use_swanlab": False})
    assert args.use_wandb and not args.use_swanlab


def test_swanlab_destination_change_starts_new_run():
    args = NS(
        use_wandb=False,
        swanlab_project="old",
        swanlab_workspace="team",
        swanlab_id="old-id",
    )
    resolve_logging_options(args, {"swanlab_project": "new"})
    assert args.swanlab_id is None


def test_run_id_persisted_and_reused_without_rewriting_training_config(tmp_path):
    saved = {
        "seed": 8,
        "checkpoint": "original.ckpt",
        "wandb_id": "wandb-original",
        "use_wandb": True,
    }
    (tmp_path / "config.yaml").write_text(json.dumps(saved))
    args = NS(**saved, experiment_name="same-run", swanlab_id=None)
    resolve_logging_options(args, {"logging_backend": "swanlab"})
    logger = NS(_protomotions_backend="swanlab", experiment=NS(id="swanlab-actual-id"))
    capture_swanlab_id(args, [object(), logger])
    persist_logging_options(tmp_path, args)
    loaded = json.loads((tmp_path / "config.yaml").read_text())
    assert loaded["seed"] == saved["seed"]
    assert loaded["checkpoint"] == saved["checkpoint"]
    assert loaded["wandb_id"] == "wandb-original"
    assert loaded["swanlab_id"] == "swanlab-actual-id"
    assert loaded["use_swanlab"] and not loaded["use_wandb"]
    resumed = NS(**loaded, experiment_name="same-run")
    resolve_logging_options(resumed, {})
    config = build_swanlab_logger_config(resumed, tmp_path)
    assert config["id"] == "swanlab-actual-id"


def test_logging_and_video_arguments_roundtrip_for_slurm():
    cli = parser().parse_args(
        [
            "--logging-backend=both",
            "--swanlab-project=with spaces",
            "--swanlab-workspace=team",
            "--swanlab-mode=offline",
            "--swanlab-video",
            "--swanlab-video-every=25",
            "--swanlab-video-fps=12",
        ]
    )
    forwarded = parser().parse_args(logging_cli_options(cli) + video_cli_options(cli))
    assert vars(forwarded) == vars(cli)
    explicit = explicit_logging_options(cli)
    video = explicit_video_options(cli)
    resolve_logging_options(cli, explicit)
    cli.simulator = "isaaclab"
    resolve_video_options(cli, video)
    assert cli.swanlab_video and cli.wandb_video_every == 25
    assert cli.wandb_video_fps == 12
    assert not cli.wandb_video  # Fresh runs enable only explicitly requested media.


def test_swanlab_video_validates_backend():
    args = NS(use_wandb=False, use_swanlab=False, simulator="isaaclab")
    with pytest.raises(ValueError, match="requires --use-swanlab"):
        resolve_video_options(args, {"swanlab_video": True})


def test_slurm_swanlab_never_checks_wandb_credentials(monkeypatch):
    from protomotions import train_slurm

    cli = train_slurm.create_parser().parse_args(
        [
            "--robot-name=g1",
            "--simulator=isaaclab",
            "--num-envs=4",
            "--batch-size=8",
            "--motion-file=motions.pt",
            "--experiment-path=exp.py",
            "--experiment-name=unit",
            "--user=test",
            "--use-wandb",
            "--logging-backend=swanlab",
            "--swanlab-project=team's project",
            "--swanlab-mode=offline",
            "--swanlab-video",
        ]
    )

    def forbidden(*args):
        pytest.fail("SwanLab-only must not request W&B credentials")

    monkeypatch.setattr(train_slurm, "check_wandb_credentials", forbidden)
    command = train_slurm.build_job_command(cli, "/work", "python")
    assert "WANDB_API_KEY" not in command
    assert "--use-wandb" not in command
    assert "--swanlab-project=team's project" in shlex.split(command)
    assert "--swanlab-video" in shlex.split(command)
    script, _ = train_slurm.generate_slurm_script(cli, "/work", command, "container")
    srun = next(line for line in script.splitlines() if line.startswith("srun "))
    assert shlex.split(srun)[-1] == command
