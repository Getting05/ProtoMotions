"""CPU-only regression tests for best-policy video scheduling."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from protomotions.utils.wandb_video import (
    DEFAULTS,
    BestPolicyVideoRecorder,
    add_video_arguments,
    explicit_video_options,
    resolve_video_options,
    video_cli_options,
    worker_environment,
)


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agent = NS(
            root_dir=self.root,
            current_epoch=200,
            fabric=NS(global_rank=0, device=NS(index=0), loggers=[]),
            env=NS(
                motion_lib=NS(motion_file="/data/amass_g1_00.pt"),
                scene_lib=NS(config=NS(scene_file=None)),
            ),
        )
        self.args = NS(**{**DEFAULTS, "wandb_video": True})
        self.recorder = BestPolicyVideoRecorder(self.args, self.agent)
        self.addCleanup(self.recorder.close)

    def best(self):
        (self.root / "score_based.ckpt").write_bytes(b"best model")
        (self.root / "resolved_configs_inference.pt").write_bytes(b"frozen config")

    def test_legacy_defaults_and_validation(self):
        parser = argparse.ArgumentParser()
        add_video_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(explicit_video_options(args), {})
        args.use_wandb, args.simulator = False, "isaaclab"
        resolve_video_options(args, {})
        self.assertFalse(args.wandb_video)
        self.assertEqual(args.wandb_video_every, 200)
        with self.assertRaises(ValueError):
            resolve_video_options(args, {"wandb_video": True})
        args.use_wandb, args.simulator = True, "isaacgym"
        with self.assertRaises(ValueError):
            resolve_video_options(args, {})

    def test_resume_explicit_options_win_and_slurm_forwarding(self):
        parser = argparse.ArgumentParser()
        add_video_arguments(parser)
        args = parser.parse_args(["--wandb-video", "--wandb-video-every=50"])
        explicit = explicit_video_options(args)
        self.assertEqual(
            video_cli_options(args), ["--wandb-video", "--wandb-video-every=50"]
        )
        args.wandb_video, args.wandb_video_every = False, 1000
        args.use_wandb, args.simulator = True, "isaaclab"
        resolve_video_options(args, explicit)
        self.assertTrue(args.wandb_video)
        self.assertEqual(args.wandb_video_every, 50)
        resolve_video_options(args, {"wandb_video": False})
        self.assertFalse(args.wandb_video)
        self.assertEqual(args.wandb_video_every, 50)

    def test_no_best_and_nonzero_rank_and_nonperiod(self):
        with patch.object(self.recorder, "_start") as start:
            self.recorder.tick(self.agent)
            start.assert_not_called()
            self.best()
            self.agent.current_epoch = 400
            self.agent.fabric.global_rank = 1
            self.recorder.tick(self.agent)
            self.agent.fabric.global_rank = 0
            self.agent.current_epoch = 401
            self.recorder.tick(self.agent)
            start.assert_not_called()

    def test_periodic_even_when_best_unchanged_no_duplicate(self):
        self.best()
        with patch.object(self.recorder, "_start") as start:
            self.recorder.tick(self.agent)
            self.recorder.tick(self.agent)
            self.agent.current_epoch = 400
            self.recorder.tick(self.agent)
            self.assertEqual(start.call_count, 2)

    def test_snapshot_and_resolved_motion_and_single_process(self):
        self.best()
        process = Mock()
        process.poll.return_value = None
        with patch(
            "protomotions.utils.wandb_video.subprocess.Popen", return_value=process
        ) as popen:
            self.recorder.tick(self.agent)
            job = self.recorder.job
            request = json.loads((job["scratch"] / "request.json").read_text())
            self.assertEqual(request["motion_file"], "/data/amass_g1_00.pt")
            self.assertIsNone(request["scene_file"])
            (self.root / "score_based.ckpt").write_bytes(b"new best")
            self.assertEqual(
                (job["scratch"] / "score_based.ckpt").read_bytes(), b"best model"
            )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--num-envs") + 1], "1")
            self.assertEqual(popen.call_args.kwargs["env"]["WANDB_MODE"], "disabled")
            self.agent.current_epoch = 400
            self.recorder.tick(self.agent)
            self.assertEqual(popen.call_count, 1)
            process.poll.return_value = 1
            self.recorder.tick(self.agent)
            self.assertIsNone(self.recorder.job)
            self.assertFalse(job["scratch"].exists())

    def test_distributed_environment_removed_gpu_remapped(self):
        env = worker_environment(
            {
                "RANK": "7",
                "LOCAL_RANK": "3",
                "WORLD_SIZE": "8",
                "MASTER_PORT": "12",
                "SLURM_PROCID": "7",
                "TORCHELASTIC_RUN_ID": "abc",
                "PMI_RANK": "7",
                "OMPI_COMM_WORLD_RANK": "7",
                "WANDB_RUN_ID": "trainer",
                "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
                "PATH": "/bin",
            },
            1,
        )
        self.assertEqual(
            env,
            {
                "CUDA_VISIBLE_DEVICES": "GPU-b",
                "PATH": "/bin",
                "WANDB_MODE": "disabled",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
        with self.assertRaises(ValueError):
            worker_environment({"CUDA_VISIBLE_DEVICES": "0"}, 1)

    def test_timeout_kills_group_and_cleans_snapshot(self):
        self.best()
        process = Mock(pid=123)
        process.poll.return_value = None
        with patch(
            "protomotions.utils.wandb_video.subprocess.Popen", return_value=process
        ):
            self.recorder.tick(self.agent)
        scratch = self.recorder.job["scratch"]
        self.recorder.job["started"] = -1000
        with patch("protomotions.utils.wandb_video.os.killpg") as kill:
            self.recorder.tick(self.agent)
            kill.assert_called_once()
        self.assertFalse(scratch.exists())
        self.assertIsNone(self.recorder.job)

    def test_upload_uses_existing_logger_current_epoch_and_metadata(self):
        import sys
        from types import ModuleType

        self.best()
        process = Mock()
        process.poll.return_value = None
        with patch(
            "protomotions.utils.wandb_video.subprocess.Popen", return_value=process
        ):
            self.recorder.tick(self.agent)
        folder = self.recorder.job["folder"]
        (folder / "best_policy.mp4").write_bytes(b"mp4")
        (folder / "best_policy.json").write_text(
            json.dumps(
                {
                    "trigger_epoch": 200,
                    "best_epoch": 100,
                    "best_score": 0.9,
                    "motion_id": 0,
                }
            )
        )

        class Logger:
            experiment = Mock()

        logger = Logger()
        self.agent.fabric.loggers = [logger]
        wandb = ModuleType("wandb")
        wandb.Video = Mock(return_value="encoded-video")
        loggers = ModuleType("lightning.pytorch.loggers")
        loggers.WandbLogger = Logger
        process.poll.return_value = 0
        self.agent.current_epoch = 230
        with patch.dict(
            sys.modules, {"wandb": wandb, "lightning.pytorch.loggers": loggers}
        ):
            self.recorder.tick(self.agent)
        record = logger.experiment.log.call_args.args[0]
        self.assertEqual(record["trainer/global_step"], 230)
        self.assertEqual(record["videos/trigger_epoch"], 200)
        self.assertEqual(record["videos/best_policy"], "encoded-video")
        self.assertIsNone(self.recorder.job)
        self.assertTrue((folder / "best_policy.mp4").exists())

    def test_launch_failure_does_not_escape_or_leave_snapshot(self):
        self.best()
        with patch(
            "protomotions.utils.wandb_video.subprocess.Popen",
            side_effect=OSError("no resources"),
        ):
            self.recorder.tick(self.agent)
        self.assertIsNone(self.recorder.job)
        self.assertFalse(list((self.root / "videos").glob("*/.snapshot-*")))

    def test_timeout_watchdog_runs_without_another_epoch(self):
        self.best()
        process = Mock(pid=123)
        process.poll.return_value = None
        with patch(
            "protomotions.utils.wandb_video.subprocess.Popen", return_value=process
        ), patch("protomotions.utils.wandb_video.threading.Timer") as timer:
            self.recorder.tick(self.agent)
            delay, callback = timer.call_args.args
            self.assertEqual(delay, 300.0)
            with patch("protomotions.utils.wandb_video.os.killpg") as kill:
                callback()
                kill.assert_called_once()
            process.poll.return_value = -9
            self.recorder.tick(self.agent)
            timer.return_value.cancel.assert_called_once()

    def test_repeated_epoch_preserves_previous_video(self):
        self.best()
        folder = self.root / "videos" / "epoch_00000200"
        folder.mkdir(parents=True)
        (folder / "best_policy.mp4").write_bytes(b"previous video")
        process = Mock()
        process.poll.return_value = None
        with patch(
            "protomotions.utils.wandb_video.subprocess.Popen", return_value=process
        ):
            self.recorder.tick(self.agent)
        self.assertNotEqual(self.recorder.job["folder"], folder)
        self.assertEqual((folder / "best_policy.mp4").read_bytes(), b"previous video")
        process.poll.return_value = 1


if __name__ == "__main__":
    unittest.main()
