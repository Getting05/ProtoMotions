"""Regressions for evidence gating and the stair swept-space constraint."""
import unittest
import numpy as np
from types import SimpleNamespace

from .contacts import InferenceConfig
from .stairs import safe_riser, support_windows, reconstruct_stairs, tread_end, supported_interval


class StairTests(unittest.TestCase):
    def event(self, body=0, start=2, end=10, stop=True, sample=0):
        return {"body_id": body, "sample_id": sample, "start": start, "end": end, "normal": [0, 0, 1],
                "point": [0, 0, .2], "evidence": {"deceleration": stop, "max_approach_speed": .2}}

    def test_stationary_foot_without_approach_is_not_a_step(self):
        events = [self.event(stop=False) for _ in range(10)]
        self.assertEqual(support_windows(events, 0, 20, 1/30), [])

    def test_consensus_requires_multiple_surface_samples_and_duration(self):
        self.assertEqual(support_windows([self.event()], 0, 20, 1/30), [])
        self.assertEqual(support_windows([self.event(sample=s) for s in range(4)], 0, 20, 1/30), [(2, 10)])
        self.assertEqual(support_windows([self.event(end=3, sample=s) for s in range(4)], 0, 20, 1/30), [])

    def test_other_footfall_peak_cannot_erase_short_support(self):
        short = [self.event(start=2, end=5, sample=s) for s in range(3)]
        strong = [self.event(start=12, end=17, sample=s) for s in range(40)]
        self.assertEqual(support_windows(short + strong, 0, 20, 1/30), [(2, 5), (12, 17)])

    def test_duplicate_events_do_not_multiply_sample_votes(self):
        self.assertEqual(support_windows([self.event() for _ in range(40)], 0, 20, 1/30), [])

    def test_overlapping_events_for_two_samples_are_insufficient(self):
        events = [self.event(sample=s, start=a, end=z) for s in (0,1) for a,z in ((2,7),(4,10))]
        self.assertEqual(support_windows(events, 0, 20, 1/30), [])

    def test_missing_level_cannot_extend_tread_across_unknown_gap(self):
        self.assertAlmostEqual(tread_end(.45, 1.0), .48)
        self.assertAlmostEqual(tread_end(.45, .46), .46)
        self.assertAlmostEqual(tread_end(.45), .48)

    def test_front_face_bound_uses_every_lower_body_surface(self):
        points = np.array([[.1, 0, .05], [.4, .2, .1], [.8, 0, .4], [2., 2., .02]])
        self.assertAlmostEqual(safe_riser(points, .2, -.5, .5, .008), .408)
        # An interfering body, not just a foot, pushes the allowable riser forward.
        points = np.vstack((points, [.7, 0., .05]))
        self.assertAlmostEqual(safe_riser(points, .2, -.5, .5, .008), .708)

    def test_long_window_preserves_continuous_partial_sole_support(self):
        contact = np.zeros((23, 4), dtype=bool)
        contact[:10, :3] = True
        self.assertEqual(supported_interval(contact, -np.ones_like(contact, dtype=float),
                                            1/30, .8), (0, 10))

    def test_partial_support_does_not_join_disconnected_short_fragments(self):
        contact = np.zeros((20, 3), dtype=bool)
        contact[[0, 1, 5, 6, 10, 11, 15, 16]] = True
        self.assertIsNone(supported_interval(contact, -np.ones_like(contact, dtype=float), 1/30, .8))

    def test_partial_support_requires_three_downward_facing_samples(self):
        contact = np.ones((10, 3), dtype=bool)
        normals = -np.ones((10, 3))
        normals[:, 2] = 0.
        self.assertIsNone(supported_interval(contact, normals, 1/30, .8))

    def test_partial_support_duration_uses_motion_timebase(self):
        contact = np.ones((8, 3), dtype=bool)
        normals = -np.ones((8, 3))
        self.assertIsNone(supported_interval(contact, normals, 1/120, .8))
        self.assertEqual(supported_interval(contact, normals, 1/60, .8), (0, 8))

    def test_no_lower_approach_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "lower approach"):
            safe_riser(np.array([[0, 0, 1.]]), .2, -.5, .5, .008)

    def test_no_support_evidence_keeps_scene_and_motion_unchanged(self):
        scene = {"objects": [], "rejected_objects": []}
        model = SimpleNamespace(body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                                surface=SimpleNamespace(body=np.array([0, 1])))
        pos = np.zeros((20, 2, 3)); pos[:, :, 0] = np.linspace(0, 1, 20)[:, None]
        motion = SimpleNamespace(pos=pos.copy(), dt=1/30)
        points = pos.copy(); normals = np.zeros_like(points); normals[:, :, 2] = -1
        result = reconstruct_stairs(scene, [], points, normals, model, motion, InferenceConfig())
        self.assertFalse(result["objects"])
        self.assertEqual(result["stairs_inference"]["status"], "insufficient_observed_levels")
        np.testing.assert_array_equal(motion.pos, pos)
        self.assertFalse(result["stairs_inference"]["unobserved_levels_filled"])


if __name__ == "__main__":
    unittest.main()
