# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric logger semantics, SDK lifecycle, and optional offline integration."""

import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from lightning.fabric.utilities.rank_zero import rank_zero_only
from protomotions.utils.swanlab_logger import SwanLabLogger


@pytest.fixture
def sdk(monkeypatch):
    run = NS(id="swan-123", config={}, log=Mock(), finish=Mock())
    sdk = NS(init=Mock(return_value=run), Video=Mock(return_value="gif-media"))
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    return sdk


def test_lazy_init_resume_config_metrics_media_and_finish(tmp_path, sdk):
    logger = SwanLabLogger(
        "project",
        "experiment",
        tmp_path,
        workspace="team",
        mode="offline",
        id="saved-id",
    )
    sdk.init.assert_not_called()
    assert logger.version == "saved-id"
    logger.log_hyperparams({"robot": {"name": "g1"}, "learning_rate": 0.001})
    assert sdk.init.call_args.kwargs["id"] == "saved-id"
    assert sdk.init.call_args.kwargs["resume"] == "allow"
    assert sdk.init.call_args.kwargs["mode"] == "offline"
    assert sdk.init.call_args.kwargs["workspace"] == "team"
    assert logger.version == "swan-123"
    assert logger.experiment.config["robot"] == {"name": "g1"}
    metrics = {
        "losses/loss": torch.tensor(2.0),
        "eval_train/score": 0.8,
        "eval_validation/score": 0.7,
    }
    logger.log_metrics(metrics, step=42)
    assert isinstance(metrics["losses/loss"], torch.Tensor)
    logger.experiment.log.assert_called_once_with(
        {"losses/loss": 2.0, "eval_train/score": 0.8, "eval_validation/score": 0.7},
        step=42,
    )
    logger.log_video(
        "videos/best_policy",
        tmp_path / "clip.gif",
        "motion 0",
        {"videos/best_epoch": 40},
        42,
    )
    logger.experiment.log.assert_called_with(
        {"videos/best_policy": "gif-media", "videos/best_epoch": 40}, step=42
    )
    run = logger.experiment
    logger.finalize("success")
    run.finish.assert_called_once_with(state="success")
    logger.finalize("success")
    assert sdk.init.call_count == 1


def test_nonzero_rank_never_initializes_or_logs(tmp_path, sdk, monkeypatch):
    logger = SwanLabLogger("project", "experiment", tmp_path)
    monkeypatch.setattr(rank_zero_only, "rank", 1)
    logger.log_metrics({"loss": 1}, 0)
    logger.log_hyperparams({"seed": 0})
    logger.log_video("video", "test.gif", "", {}, 1)
    logger.finalize("success")
    with pytest.raises(RuntimeError, match="rank zero"):
        _ = logger.experiment
    sdk.init.assert_not_called()


def test_failure_status_and_uninitialized_finalize(tmp_path, sdk):
    logger = SwanLabLogger("project", "experiment", tmp_path)
    logger.finalize("failed")
    sdk.init.assert_not_called()
    logger.log_metrics({"loss": 1}, 1)
    run = logger.experiment
    logger.finalize("failed")
    run.finish.assert_called_once_with(state="crashed")


def test_missing_optional_sdk_has_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    with pytest.raises(ModuleNotFoundError, match=r"pip install.*\[swanlab\]"):
        SwanLabLogger("project", "experiment", tmp_path)


def test_real_sdk_offline_metrics_config_gif_and_finalize(tmp_path, monkeypatch):
    pytest.importorskip("swanlab")
    from lightning.fabric import Fabric
    from moviepy import ColorClip
    from PIL import Image
    from protomotions.utils.policy_video import encode_swanlab_gif

    monkeypatch.setattr(rank_zero_only, "rank", 0)
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    video = tmp_path / "policy.mp4"
    with ColorClip((64, 48), color=(40, 100, 200), duration=0.3) as clip:
        clip.write_videofile(str(video), fps=10, logger=None)
    gif = encode_swanlab_gif(video)
    with Image.open(gif) as image:
        assert image.format == "GIF"
        assert image.size == (64, 48)
    logger = SwanLabLogger("protomotions-test", "offline-smoke", tmp_path)
    fabric = Fabric(accelerator="cpu", devices=1, loggers=[logger])
    fabric.launch()
    try:
        logger.log_hyperparams({"robot": {"name": "g1"}, "seed": 7})
        fabric.log_dict(
            {"losses/loss": torch.tensor(1.25), "eval_validation/score": 0.75}, step=10
        )
        logger.log_video(
            "videos/best_policy", gif, "smoke", {"videos/motion_id": 0}, 10
        )
        run_id = logger.version
        assert run_id
        assert logger.experiment.config["seed"] == 7
    finally:
        logger.finalize("success")
    assert list((tmp_path / "swanlog").rglob("*.gif"))
    assert any(path.is_file() for path in (tmp_path / "swanlog").rglob("*"))
    resumed = SwanLabLogger("protomotions-test", "offline-smoke", tmp_path, id=run_id)
    try:
        resumed.log_metrics({"losses/loss": 0.5}, step=11)
        assert resumed.version == run_id
    finally:
        resumed.finalize("success")
