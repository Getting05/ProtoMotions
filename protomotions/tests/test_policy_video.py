"""Packed motion selection and bounded playback without a simulator."""

import json
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from protomotions.utils.policy_video import prepare_video_motion, record_policy_video


def test_select_motion_only_keeps_its_frames(tmp_path):
    source = tmp_path / "library.pt"
    torch.save(
        {
            "length_starts": torch.tensor([0, 3]),
            "motion_lengths": torch.tensor([0.2, 0.3]),
            "motion_num_frames": torch.tensor([3, 4]),
            "motion_dt": torch.tensor([0.1, 0.1]),
            "motion_weights": torch.tensor([1.0, 2.0]),
            "motion_files": ("a", "b"),
            "gts": torch.arange(21).reshape(7, 3),
            "contacts": torch.ones(7, 2),
        },
        source,
    )
    request = {
        "motion_file": str(source),
        "scene_file": None,
        "wandb_video_motion_id": 1,
    }
    prepare_video_motion(request, tmp_path)
    selected = torch.load(request["render_motion_file"], weights_only=False)
    assert request["render_motion_id"] == 0
    assert request["wandb_video_motion_id"] == 1
    assert selected["length_starts"].tolist() == [0]
    assert selected["gts"].tolist() == torch.arange(9, 21).reshape(4, 3).tolist()
    assert selected["motion_files"] == ("b",)
    assert selected["motion_weights"].tolist() == [2.0]
    assert torch.load(source, weights_only=False)["gts"].shape[0] == 7
    request["wandb_video_motion_id"] = 2
    with pytest.raises(ValueError, match="outside"):
        prepare_video_motion(request, tmp_path)


def test_scene_linked_library_retains_indices(tmp_path):
    request = {
        "motion_file": "not-loaded.pt",
        "scene_file": "scene.pt",
        "wandb_video_motion_id": 2,
    }
    prepare_video_motion(request, tmp_path)
    assert "render_motion_file" not in request


@pytest.mark.parametrize("stop_at,expected_frames", [(99, 6), (1, 3)])
def test_playback_is_deterministic_bounded_and_retains_metadata(
    tmp_path, stop_at, expected_frames
):
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"epoch": 12, "best_evaluated_score": 0.75}, checkpoint)
    output = tmp_path / "best_policy.mp4"
    request = {
        "wandb_video_motion_id": 9,
        "render_motion_id": 0,
        "wandb_video_fps": 60,
        "wandb_video_duration": 0.1,
        "wandb_video_width": 8,
        "wandb_video_height": 8,
        "trigger_epoch": 20,
        "motion_file": "source.pt",
        "output": str(output),
    }
    steps = []

    def step(action):
        assert action.item() == 2  # mean action, never sampled action
        steps.append(action)
        return {}, None, torch.tensor([len(steps) >= stop_at]), None, {}

    env = NS(
        dt=1 / 30,
        motion_lib=NS(get_motion_length=lambda ids: torch.tensor([2.0])),
        motion_manager=NS(
            motion_ids=torch.zeros(1, dtype=torch.long), motion_times=torch.ones(1)
        ),
        reset=Mock(return_value=({}, None)),
        step=step,
        simulator=NS(
            _get_simulator_root_state=lambda i: NS(
                root_pos=torch.tensor([[1.0, 2.0, 3.0]])
            )
        ),
    )
    agent = NS(
        env=env,
        device="cpu",
        eval=Mock(),
        fabric=NS(seed_everything=Mock()),
        evaluator=None,
        pre_collect_step=Mock(),
        add_agent_info_to_obs=lambda obs: obs,
        obs_dict_to_tensordict=lambda obs: obs,
        model=lambda obs: {
            "mean_action": torch.tensor([2.0]),
            "action": torch.tensor([-2.0]),
        },
    )
    writer = Mock()

    def get_writer(path, **kwargs):
        from pathlib import Path

        Path(path).write_bytes(b"encoded")
        context = Mock()
        context.__enter__ = Mock(return_value=writer)
        context.__exit__ = Mock(return_value=False)
        return context

    with ExitStack() as stack:
        camera = stack.enter_context(
            patch("protomotions.utils.policy_video.FollowCamera")
        )
        reference = stack.enter_context(
            patch("protomotions.utils.reference_video.ReferenceRobot")
        )
        stack.enter_context(patch("imageio.v2.get_writer", side_effect=get_writer))
        camera.return_value.frame.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        record_policy_video(agent, request, checkpoint)
        camera.return_value.close.assert_called_once()
        reference.return_value.close.assert_called_once()
        assert reference.return_value.update.call_count == len(steps)
    assert writer.append_data.call_count == expected_frames
    assert env.motion_manager.motion_times.item() == 0
    env.reset.assert_called_once_with(disable_motion_resample=True)
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["motion_id"] == 9
    assert metadata["best_epoch"] == 12
    assert metadata["trigger_epoch"] == 20
    assert metadata["duration"] == expected_frames / 60
