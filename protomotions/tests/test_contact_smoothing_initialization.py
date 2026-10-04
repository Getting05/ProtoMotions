"""Keep motion preprocessing fast without changing contact-label semantics."""

from types import SimpleNamespace
import unittest

import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.envs.base_env.env import BaseEnv


def fake_env():
    return SimpleNamespace(
        config=SimpleNamespace(ref_contact_smooth_window=7),
        _validate_motion_lib_compatibility=lambda _: None,
        create_motion_manager=lambda: None,
    )


class ContactSmoothingInitializationTests(unittest.TestCase):
    def test_benchmark_disabled_only_during_preprocessing(self):
        previous = torch.backends.cudnn.benchmark
        try:
            for initial in [False, True]:
                torch.backends.cudnn.benchmark = initial
                observed = []
                motion = SimpleNamespace(
                    num_motions=lambda: 1,
                    smooth_contacts=lambda window: observed.append(
                        (window, torch.backends.cudnn.benchmark)
                    ),
                )
                env = fake_env()
                BaseEnv.install_motion_lib(env, motion)
                self.assertEqual(observed, [(7, False)])
                self.assertEqual(torch.backends.cudnn.benchmark, initial)
                self.assertIs(env.motion_lib, motion)
        finally:
            torch.backends.cudnn.benchmark = previous

    def test_benchmark_restored_after_preprocessing_error(self):
        def fail(_):
            raise ValueError("invalid labels")

        previous = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = True
            motion = SimpleNamespace(num_motions=lambda: 1, smooth_contacts=fail)
            with self.assertRaisesRegex(ValueError, "invalid labels"):
                BaseEnv.install_motion_lib(fake_env(), motion)
            self.assertTrue(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cudnn.benchmark = previous

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_cpu_and_preserves_motion_boundaries(self):
        labels = torch.zeros(55, 31, dtype=torch.bool)
        labels[18:] = True
        labels[22:30, 6] = False
        libraries = []
        for device in ["cpu", "cuda"]:
            lib = MotionLib(MotionLibConfig(), device=device)
            lib.contacts = labels.to(device)
            lib.length_starts = torch.tensor([0, 18], device=device)
            lib.motion_num_frames = torch.tensor([18, 37], device=device)
            lib.motion_lengths = torch.tensor([0.6, 37 / 30], device=device)
            BaseEnv.install_motion_lib(fake_env(), lib)
            libraries.append(lib)
        torch.testing.assert_close(libraries[1].contacts.cpu(), libraries[0].contacts)
        self.assertTrue((libraries[1].contacts[:18] == 0).all())
        torch.testing.assert_close(libraries[1].contacts[18, :6].cpu(), torch.ones(6))


if __name__ == "__main__":
    unittest.main()
