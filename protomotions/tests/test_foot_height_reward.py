# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foot-height reward invariants and runtime context binding."""

import math
from types import SimpleNamespace
import unittest

import torch

from protomotions.envs.component_factories import foot_height_rew_factory
from protomotions.envs.rewards.tracking import compute_foot_height_rew


class FootHeightRewardTests(unittest.TestCase):
    def test_stance_and_swing_both_peak_at_reference(self):
        ref = torch.zeros(2, 31, 3)
        ref[:, [6, 12], 2] = torch.tensor([[0.06, 0.06], [0.06, 0.26]])
        ids = torch.tensor([6, 12])
        torch.testing.assert_close(compute_foot_height_rew(ref, ref, ids), torch.ones(2))
        too_high = ref.clone()
        too_high[:, ids, 2] += 0.05
        torch.testing.assert_close(
            compute_foot_height_rew(too_high, ref, ids),
            torch.full((2,), math.exp(-1)),
        )

    def test_lifting_toward_reference_improves_reward(self):
        ref = torch.zeros(4, 31, 3)
        ref[:, 6, 2] = 0.10
        current = torch.zeros_like(ref)
        current[:, 6, 2] = torch.tensor([0.0, 0.025, 0.05, 0.10])
        reward = compute_foot_height_rew(current, ref, torch.tensor([6, 12]))
        self.assertTrue(torch.all(reward[1:] > reward[:-1]))
        self.assertAlmostEqual(reward[-1].item(), 1.0)

    def test_horizontal_and_nonfoot_errors_do_not_dilute_reward(self):
        ref = torch.zeros(1, 31, 3)
        current = torch.full_like(ref, 100.0)
        current[:, [6, 12], 2] = torch.tensor([0.05, 0.0])
        expected = torch.tensor([(math.exp(-1) + 1) / 2])
        torch.testing.assert_close(
            compute_foot_height_rew(current, ref, torch.tensor([6, 12])), expected
        )

    def test_opposite_foot_errors_do_not_cancel(self):
        ref = torch.zeros(1, 31, 3)
        current = ref.clone()
        current[:, [6, 12], 2] = torch.tensor([-0.05, 0.05])
        torch.testing.assert_close(
            compute_foot_height_rew(current, ref, torch.tensor([6, 12])),
            torch.tensor([math.exp(-1)]),
        )

    def test_factory_binds_positions_and_ids_without_contact_labels(self):
        ref = torch.zeros(2, 31, 3)
        ctx = SimpleNamespace(
            current=SimpleNamespace(rigid_body_pos=ref),
            mimic=SimpleNamespace(ref_state=SimpleNamespace(rigid_body_pos=ref)),
            contact_body_ids=torch.tensor([6, 12]),
        )
        component = foot_height_rew_factory(weight=0.15, height_std=0.05)
        torch.testing.assert_close(component.compute(ctx), torch.ones(2))
        self.assertEqual(component.static_params["weight"], 0.15)
        self.assertEqual(component.get_bindings_dict()["foot_body_ids"], "contact_body_ids")

    def test_factory_rejects_invalid_height_scale(self):
        for height_std in [0, -0.05, float("nan"), float("inf")]:
            with self.subTest(height_std=height_std), self.assertRaises(ValueError):
                foot_height_rew_factory(height_std=height_std)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_compiled_matches_eager(self):
        generator = torch.Generator().manual_seed(7)
        ref = torch.randn(4096, 31, 3, generator=generator)
        current = ref + 0.05 * torch.randn(4096, 31, 3, generator=generator)
        ids = torch.tensor([6, 12])
        expected = compute_foot_height_rew(current, ref, ids)
        compiled = torch.compile(compute_foot_height_rew, fullgraph=True)
        actual = compiled(current.cuda(), ref.cuda(), ids.cuda())
        torch.testing.assert_close(actual.cpu(), expected)
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
