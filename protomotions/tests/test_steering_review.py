# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol, portable snapshot and rendering-dispatch regression coverage."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from protomotions.utils import policy_video
from protomotions.utils.steering_review import (
    SteeringProtocol,
    command_for_case,
    focus_environment,
    prepare_snapshot,
    select_motions,
    stable_copy,
    velocity_arrow_points,
    verify_snapshot,
)


@pytest.mark.parametrize("speed", [0, 0.02, 1.2, 2, 4])
def test_velocity_arrow_has_exact_linear_scale_and_zero_disappears(speed):
    points = velocity_arrow_points([3.0, -2.0], [3.0, 4.0], 0.12, speed)
    if speed == 0:
        assert points == []
    else:
        assert len(points) == 6
        assert math.dist(points[0], points[1]) == pytest.approx(speed * 0.35)
        # Arrow heads stay ahead of the tail, even for very small speeds.
        direction = [0.6, 0.8]
        for point in points:
            assert (
                sum((point[i] - points[0][i]) * direction[i] for i in range(2))
                >= -1e-12
            )


def test_stop_and_turn_commands_switch_at_exact_control_boundary():
    p = SteeringProtocol().validate()
    assert command_for_case("stop", 5.99, p) == (0, 0, 2)
    assert command_for_case("stop", 6, p) == (0, 0, 0)
    assert command_for_case("turn_left", 6, p) == (math.pi / 2, math.pi / 2, 2)
    assert command_for_case("backward", 0, p) == (math.pi, 0, 1.2)
    assert focus_environment("speed_stop", 7.99, 4, 1, 16) == 17
    assert focus_environment("speed_stop", 8, 4, 1, 16) == 21
    with pytest.raises(ValueError, match="switch_time"):
        SteeringProtocol(duration=4).validate()
    with pytest.raises(ValueError, match="positive"):
        SteeringProtocol(arrow_scale=0).validate()


def fixture_files(tmp_path, references=True):
    policy = tmp_path / "source"
    prior = tmp_path / "prior_source"
    policy.mkdir()
    prior.mkdir()
    key = (
        "_actor.prior_with_peft.reference_prior.weight"
        if references
        else "adapter.weight"
    )
    torch.save(
        {"model": {key: torch.ones(2)}, "epoch": 660, "step_count": 346030080},
        policy / "last.ckpt",
    )
    torch.save({"model": {}}, prior / "prior.ckpt")
    torch.save({"agent": NS()}, prior / "resolved_configs.pt")
    cfg = {
        "env": NS(control_components={"steering": NS()}),
        "agent": NS(
            pretrained_modules={"prior": NS(checkpoint_path=str(prior / "prior.ckpt"))}
        ),
    }
    for name in ["resolved_configs.pt", "resolved_configs_inference.pt"]:
        torch.save(cfg, policy / name)
    library = tmp_path / "motions.pt"
    torch.save(
        {
            "length_starts": torch.tensor([0, 2, 5]),
            "motion_lengths": torch.tensor([0.1, 0.2, 0.3]),
            "motion_num_frames": torch.tensor([2, 3, 4]),
            "motion_dt": torch.full((3,), 0.1),
            "motion_weights": torch.ones(3),
            "motion_files": ("a", "b", "c"),
            "gts": torch.arange(27).reshape(9, 3),
            "contacts": torch.ones(9, 2),
        },
        library,
    )
    return policy / "last.ckpt", library


def test_snapshot_is_portable_integrity_checked_and_keeps_reference(tmp_path):
    checkpoint, library = fixture_files(tmp_path)
    package = tmp_path / "snapshot"
    manifest = prepare_snapshot(checkpoint, library, [0, 2], package)
    assert manifest["epoch"] == 660
    assert manifest["reference_keys"] == 1
    selected = torch.load(package / "selected_motions.pt", weights_only=False)
    assert selected["length_starts"].tolist() == [0, 2]
    assert selected["motion_num_frames"].tolist() == [2, 4]
    assert selected["motion_files"] == ("a", "c")
    assert torch.load(library, weights_only=False)["gts"].shape == (9, 3)
    moved = tmp_path / "other_host"
    shutil.move(package, moved)
    assert verify_snapshot(moved) == manifest
    with (moved / "policy/last.ckpt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_snapshot(moved)


def test_adapter_only_checkpoint_and_invalid_motion_ids_are_rejected(tmp_path):
    checkpoint, library = fixture_files(tmp_path, references=False)
    with pytest.raises(ValueError, match="full Stage 3"):
        prepare_snapshot(checkpoint, library, [0], tmp_path / "snapshot")
    with pytest.raises(ValueError, match="outside"):
        select_motions(library, [3], tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="distinct"):
        select_motions(library, [0, 0], tmp_path / "bad.pt")


def test_changing_checkpoint_is_not_accepted_as_a_stable_snapshot(
    tmp_path, monkeypatch
):
    source, target = tmp_path / "live.ckpt", tmp_path / "snapshot.ckpt"
    source.write_bytes(b"live")
    copy = shutil.copyfile

    def copy_and_change(a, b):
        copy(a, b)
        with Path(a).open("ab") as handle:
            handle.write(b"next")

    monkeypatch.setattr(shutil, "copyfile", copy_and_change)
    with pytest.raises(RuntimeError, match="Source changed"):
        stable_copy(source, target)
    assert not target.exists()
    assert not target.with_suffix(".ckpt.partial").exists()


def test_normal_loader_explicitly_dispatches_steering_without_selecting_one_motion(
    monkeypatch, tmp_path
):
    request = {"kind": "steering", "motion_file": "cohort.pt"}
    policy_video.prepare_video_motion(request, tmp_path)
    assert request["render_motion_file"] == "cohort.pt"
    from protomotions.utils import steering_video

    record = Mock(return_value="steering")
    monkeypatch.setattr(steering_video, "record_steering_video", record)
    assert policy_video.record_policy_video("agent", request, "full.ckpt") == "steering"
    record.assert_called_once_with("agent", request, "full.ckpt")


def test_render_relocates_only_runtime_config_and_isolates_child_environment(
    tmp_path, monkeypatch
):
    from scripts.evaluation import render_steering as cli
    from protomotions.utils import steering_report

    checkpoint, library = fixture_files(tmp_path)
    package = tmp_path / "snapshot"
    prepare_snapshot(checkpoint, library, [0, 2], package)
    args = cli.create_parser().parse_args(
        ["render", "--prepared-dir", str(package), "--gpu", "4"]
    )
    calls = []
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(steering_report, "report", lambda directory: [])
    cli.render_snapshot(package, args)
    env = calls[0][1]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["WANDB_MODE"] == "disabled" and "RANK" not in env
    runtime = torch.load(
        package / "render/runtime/resolved_configs_inference.pt", weights_only=False
    )
    assert runtime["agent"].pretrained_modules["prior"].checkpoint_path == str(
        package / "prior/prior.ckpt"
    )
    assert verify_snapshot(package)["epoch"] == 660  # original configs never mutated
    request = json.loads((package / "render/request.json").read_text())
    assert request["kind"] == "steering"
    assert request["protocol"]["arrow_scale"] == 0.35


def test_compare_rejects_different_physical_protocol_before_plotting(tmp_path):
    from protomotions.utils.steering_report import compare

    paths = [tmp_path / "a", tmp_path / "b"]
    for i, path in enumerate(paths):
        path.mkdir()
        (path / "metrics.json").write_text(
            json.dumps(
                {
                    "initial_motion_ids": [0],
                    "selected_motion_sha256": "a",
                    "prior_sha256": "b",
                    "dt": 0.02,
                    "protocol": asdict(SteeringProtocol(seed=i)),
                }
            )
        )
    with pytest.raises(ValueError, match="protocol.seed"):
        compare(*paths, tmp_path / "comparison")


def test_cli_help_is_usable_without_starting_a_simulator():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluation/render_steering.py"),
            "render",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--arrow-scale" in result.stdout and "--views" in result.stdout
