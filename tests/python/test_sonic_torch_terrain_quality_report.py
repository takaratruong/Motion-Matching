import unittest

import numpy as np

from mm_sonic.torch_terrain_quality_report import (
    NativeCorpusSample,
    StateClassificationEvidence,
    classify_state,
    derive_native_thresholds,
    selected_source_acceptance,
)
from mm_sonic.torch_terrain_action_quality import NativeActionQuality


def _evidence(**changes):
    values = {
        "state_id": "a" * 64,
        "source_acceptable": True,
        "native_acceptable_count": 1,
        "placed_acceptable_count": 1,
        "composed_acceptable_count": 1,
        "baseline_selected_acceptable": True,
        "two_step_acceptable_count": 1,
        "trigger_metrics": {"landing_error_m": 0.01},
    }
    values.update(changes)
    return StateClassificationEvidence(**values)


class TerrainQualityReportTests(unittest.TestCase):
    def test_failure_classification_is_explicit_and_ordered(self):
        self.assertEqual(classify_state(_evidence(source_acceptable=False)), "source")
        self.assertEqual(classify_state(_evidence(native_acceptable_count=0)), "corpus")
        self.assertEqual(classify_state(_evidence(placed_acceptable_count=0)), "placement")
        self.assertEqual(classify_state(_evidence(composed_acceptable_count=0)), "composition")
        self.assertEqual(
            classify_state(_evidence(baseline_selected_acceptable=False)), "search"
        )
        self.assertEqual(
            classify_state(
                _evidence(
                    baseline_selected_acceptable=False,
                    two_step_acceptable_count=0,
                )
            ),
            "representation",
        )
        self.assertEqual(classify_state(_evidence()), "qualified")

    def test_native_thresholds_use_median_plus_two_mad(self):
        thresholds = derive_native_thresholds(
            (
                NativeCorpusSample(0.01, 1.0),
                NativeCorpusSample(0.02, 2.0),
                NativeCorpusSample(0.03, 3.0),
                NativeCorpusSample(1.00, 100.0),
            )
        )

        self.assertAlmostEqual(thresholds.stance_drift_m, 0.045)
        self.assertAlmostEqual(thresholds.boundary_motion, 4.5)
        self.assertEqual(thresholds.sample_count, 4)

    def test_evidence_owns_trigger_metrics(self):
        metrics = {"landing_error_m": 0.01}
        evidence = _evidence(trigger_metrics=metrics)
        metrics["landing_error_m"] = 99.0

        self.assertEqual(evidence.trigger_metrics["landing_error_m"], 0.01)
        with self.assertRaises(TypeError):
            evidence.trigger_metrics["new"] = 1.0

    def test_source_label_uses_selected_native_action_not_route_disagreement(self):
        def quality(drift):
            return NativeActionQuality(
                action_index=0,
                source_key=(3, 10, 20, False),
                entry_support=(True, False),
                swing_foot=1,
                landing_frame_offset=9,
                natural_landing_local_xyz=np.zeros(3),
                root_displacement_local_xy=np.zeros(2),
                root_yaw_delta_rad=0.0,
                source_stance_drift_m=drift,
                minimum_swing_clearance_m=0.05,
                entry_joint_speed_norm=1.0,
                terminal_joint_speed_norm=1.0,
                exact_successor_index=None,
            )

        thresholds = derive_native_thresholds(
            (NativeCorpusSample(0.01, 1.0), NativeCorpusSample(0.02, 2.0))
        )
        self.assertEqual(
            selected_source_acceptance(
                selected_clip_path="clip.npz",
                selected_source_frame=15,
                clip_paths={3: "clip.npz"},
                qualities=(quality(0.50),),
                thresholds=thresholds,
            ),
            (False, 1),
        )
        self.assertEqual(
            selected_source_acceptance(
                selected_clip_path="other.npz",
                selected_source_frame=15,
                clip_paths={3: "clip.npz"},
                qualities=(quality(0.50),),
                thresholds=thresholds,
            ),
            (True, 0),
        )


if __name__ == "__main__":
    unittest.main()
