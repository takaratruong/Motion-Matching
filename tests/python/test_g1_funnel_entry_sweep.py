import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch


class G1FunnelEntrySweepTests(unittest.TestCase):
    def test_builds_exact_trained_object_entry_grid(self):
        from tools.evaluate_g1_funnel_entry_sweep import build_entry_conditions

        base = np.arange(24, dtype=np.float32)
        conditions, descriptors = build_entry_conditions(base)

        self.assertEqual(conditions.shape, (72, 24))
        self.assertEqual(len(descriptors), 72)
        np.testing.assert_array_equal(
            conditions[:, :18], np.broadcast_to(base[:18], (72, 18)))
        np.testing.assert_array_equal(
            conditions[:, 22:24], np.zeros((72, 2), dtype=np.float32))
        self.assertEqual(
            {row["bearing_degrees"] for row in descriptors},
            set(range(0, 360, 45)),
        )
        self.assertEqual(
            {row["radius_m"] for row in descriptors},
            {0.55, 0.75, 0.95},
        )
        self.assertEqual(
            {row["yaw_offset_degrees"] for row in descriptors},
            {-15, 0, 15},
        )
        keys = {
            (row["bearing_degrees"], row["radius_m"],
             row["yaw_offset_degrees"])
            for row in descriptors
        }
        self.assertEqual(len(keys), 72)

        for condition, descriptor in zip(conditions, descriptors):
            x, z, yaw_sin, yaw_cos = condition[18:22]
            self.assertAlmostEqual(
                math.hypot(float(x), float(z)),
                descriptor["radius_m"],
                places=6,
            )
            actual_yaw = math.atan2(float(yaw_sin), float(yaw_cos))
            facing_yaw = math.atan2(-float(x), -float(z))
            offset = math.atan2(
                math.sin(actual_yaw - facing_yaw),
                math.cos(actual_yaw - facing_yaw),
            )
            self.assertAlmostEqual(
                math.degrees(offset),
                descriptor["yaw_offset_degrees"],
                places=4,
            )

    def test_evaluates_projected_proposals_deterministically(self):
        from tools.evaluate_g1_funnel_entry_sweep import evaluate_entry_sweep

        def fake_sampler(_checkpoint, conditions, *, device, seed):
            del device, seed
            raw = conditions.cpu().numpy()
            outward = np.zeros((len(raw), 32, 16, 4), dtype=np.float32)
            for row, condition in enumerate(raw):
                entry = condition[18:22]
                entry_yaw = math.atan2(float(entry[2]), float(entry[3]))
                for proposal in range(32):
                    terminal = np.asarray(
                        [0.15 + proposal * 0.0005, -0.12,
                         math.sin(entry_yaw), math.cos(entry_yaw)],
                        dtype=np.float32,
                    )
                    for knot in range(16):
                        alpha = knot / 15.0
                        outward[row, proposal, knot] = (
                            terminal + alpha * (entry - terminal))
                    outward[row, proposal, -1] = entry
            return torch.from_numpy(outward)

        base = np.zeros(24, dtype=np.float32)
        first = evaluate_entry_sweep(
            Path("checkpoint.pt"), base, sampler=fake_sampler)
        second = evaluate_entry_sweep(
            Path("checkpoint.pt"), base, sampler=fake_sampler)

        self.assertEqual(first, second)
        self.assertEqual(first["condition_count"], 72)
        self.assertEqual(first["proposal_count_per_condition"], 32)
        self.assertEqual(first["passing_condition_count"], 72)
        self.assertEqual(first["failing_condition_count"], 0)
        self.assertEqual(first["minimum_accepted_proposals"], 32)
        self.assertEqual(first["median_accepted_proposals"], 32.0)
        self.assertTrue(all(
            row["entry_boundary_exact"] and
            row["entry_boundary_max_abs_error"] == 0.0 and
            row["failure_reason"] is None
            for row in first["conditions"]
        ))

    def test_entry_boundary_mismatch_fails_closed(self):
        from tools.evaluate_g1_funnel_entry_sweep import evaluate_entry_sweep

        def mismatched_sampler(_checkpoint, conditions, *, device, seed):
            del device, seed
            raw = conditions.cpu().numpy()
            outward = np.zeros((len(raw), 32, 16, 4), dtype=np.float32)
            outward[..., 3] = 1.0
            outward[:, :, -1] = raw[:, None, 18:22]
            outward[0, 0, -1, 0] += np.float32(0.001)
            return torch.from_numpy(outward)

        report = evaluate_entry_sweep(
            Path("checkpoint.pt"),
            np.zeros(24, dtype=np.float32),
            sampler=mismatched_sampler,
        )

        self.assertLess(report["passing_condition_count"], 72)
        self.assertEqual(
            report["conditions"][0]["failure_reason"],
            "entry_boundary_mismatch",
        )
        self.assertFalse(report["conditions"][0]["entry_boundary_exact"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
