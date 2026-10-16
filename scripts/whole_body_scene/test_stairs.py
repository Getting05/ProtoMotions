"""Regressions for evidence gating and the stair swept-space constraint."""
import unittest
import numpy as np
from types import SimpleNamespace

from .contacts import InferenceConfig
from .stairs import safe_riser, support_windows, reconstruct_stairs


class StairTests(unittest.TestCase):
    def event(self, body=0, start=2, end=10, stop=True):
        return {"body_id": body, "start": start, "end": end, "normal": [0, 0, 1],
                "point": [0, 0, .2], "evidence": {"deceleration": stop, "max_approach_speed": .2}}

    def test_stationary_foot_without_approach_is_not_a_step(self):
        events = [self.event(stop=False) for _ in range(10)]
        self.assertEqual(support_windows(events, 0, 20, 1/30), [])

    def test_consensus_requires_multiple_surface_samples_and_duration(self):
        self.assertEqual(support_windows([self.event()], 0, 20, 1/30), [])
        self.assertEqual(support_windows([self.event() for _ in range(4)], 0, 20, 1/30), [(2, 10)])
        self.assertEqual(support_windows([self.event(end=3) for _ in range(4)], 0, 20, 1/30), [])

    def test_front_face_bound_uses_every_lower_body_surface(self):
        points = np.array([[.1, 0, .05], [.4, .2, .1], [.8, 0, .4], [2., 2., .02]])
        self.assertAlmostEqual(safe_riser(points, .2, -.5, .5, .008), .408)
        # An interfering body, not just a foot, pushes the allowable riser forward.
        points = np.vstack((points, [.7, 0., .05]))
        self.assertAlmostEqual(safe_riser(points, .2, -.5, .5, .008), .708)

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
