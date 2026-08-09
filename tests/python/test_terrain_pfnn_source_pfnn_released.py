from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.terrain_pfnn.source_pfnn_released import (
    PFNNIntervalMetrics,
    discover_released_pfnn_records,
    interval_metrics_from_tracks,
    required_coverage,
    select_vertical_slice,
    vertical_slice_receipt,
)


def _write_record(root: Path, stem: str, *, frames: int = 2400) -> None:
    (root / f"{stem}.bvh").write_text(
        "HIERARCHY\nMOTION\n"
        f"Frames: {frames}\n"
        "Frame Time: 0.008333\n",
        encoding="utf-8",
    )
    (root / f"{stem}.phase").write_text("0\n", encoding="utf-8")
    (root / f"{stem}.gait").write_text("1 0 0 0 0 0 0 0\n", encoding="utf-8")
    (root / f"{stem}_footsteps.txt").write_text("0 L\n120 R\n", encoding="utf-8")


def _metric(
    stem: str,
    *,
    displacement: float = 1.0,
    left: float = 0.0,
    right: float = 0.0,
    ascent: float = 0.0,
    descent: float = 0.0,
    idle_transition: bool = False,
) -> PFNNIntervalMetrics:
    return PFNNIntervalMetrics(
        stem=stem,
        start_frame_120hz=240,
        stop_frame_120hz=1200,
        root_displacement_m=displacement,
        maximum_left_turn_rad=left,
        maximum_right_turn_rad=right,
        maximum_ascent_degrees=ascent,
        minimum_descent_degrees=descent,
        has_idle_transition=idle_transition,
    )


class ReleasedPFNNSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for stem in (
            "LocomotionFlat01_000",
            "LocomotionFlat02_000",
            "WalkingUpSteps01_000",
            "WalkingUpSteps02_000",
        ):
            _write_record(self.root, stem)
            _write_record(self.root, stem + "_mirror")
        self.metrics = {
            "LocomotionFlat01_000": _metric(
                "LocomotionFlat01_000", idle_transition=True
            ),
            "LocomotionFlat02_000": _metric(
                "LocomotionFlat02_000", left=1.2, right=1.1
            ),
            "WalkingUpSteps01_000": _metric(
                "WalkingUpSteps01_000", ascent=9.0, descent=-8.0
            ),
            "WalkingUpSteps02_000": _metric(
                "WalkingUpSteps02_000", ascent=7.0, descent=-6.0
            ),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selects_required_train_and_validation_coverage(self) -> None:
        records = discover_released_pfnn_records(self.root)
        selection = select_vertical_slice(records, tuple(self.metrics.values()))
        self.assertEqual([item.role for item in selection].count("train"), 3)
        self.assertEqual([item.role for item in selection].count("validation"), 1)
        self.assertTrue(all("_mirror" not in item.record.stem for item in selection))
        self.assertEqual(
            required_coverage(selection),
            frozenset(
                {
                    "idle_transition",
                    "straight",
                    "left_turn",
                    "right_turn",
                    "ascent",
                    "descent",
                }
            ),
        )
        self.assertEqual(
            [(item.start_frame_120hz, item.stop_frame_120hz) for item in selection],
            [(120, 1320)] * 4,
        )

    def test_discovery_rejects_missing_sidecar(self) -> None:
        (self.root / "LocomotionFlat01_000.phase").unlink()
        with self.assertRaisesRegex(ValueError, "missing PFNN sidecar"):
            discover_released_pfnn_records(self.root)

    def test_selection_rejects_missing_required_coverage(self) -> None:
        records = discover_released_pfnn_records(self.root)
        metrics = dict(self.metrics)
        metrics["LocomotionFlat02_000"] = _metric("LocomotionFlat02_000", left=1.2)
        with self.assertRaisesRegex(ValueError, "right_turn"):
            select_vertical_slice(records, tuple(metrics.values()))

    def test_receipt_binds_hashes_and_is_order_independent(self) -> None:
        records = discover_released_pfnn_records(self.root)
        metrics = tuple(self.metrics.values())
        selected = select_vertical_slice(records, metrics)
        reversed_selected = select_vertical_slice(tuple(reversed(records)), tuple(reversed(metrics)))
        self.assertEqual(
            vertical_slice_receipt(selected), vertical_slice_receipt(reversed_selected)
        )
        changed_record = replace(selected[0].record, bvh_sha256="f" * 64)
        changed = (replace(selected[0], record=changed_record), *selected[1:])
        self.assertNotEqual(
            vertical_slice_receipt(selected), vertical_slice_receipt(changed)
        )

    def test_selection_rejects_interval_without_one_second_context(self) -> None:
        records = discover_released_pfnn_records(self.root)
        metrics = dict(self.metrics)
        metrics["LocomotionFlat01_000"] = replace(
            metrics["LocomotionFlat01_000"], start_frame_120hz=100
        )
        with self.assertRaisesRegex(ValueError, "one-second context"):
            select_vertical_slice(records, tuple(metrics.values()))

    def test_selection_rejects_metric_stem_mismatch(self) -> None:
        records = discover_released_pfnn_records(self.root)
        metrics = dict(self.metrics)
        metrics["LocomotionFlat01_000"] = replace(
            metrics["LocomotionFlat01_000"], stem="LocomotionFlat09_000"
        )
        with self.assertRaisesRegex(ValueError, "no source record"):
            select_vertical_slice(records, tuple(metrics.values()))

    def test_track_audit_emits_straight_turn_and_signed_grade_candidates(self) -> None:
        frames = 1200
        root = np.zeros((frames, 3), dtype=np.float64)
        root[:, 0] = np.arange(frames) / 120.0
        gait = np.zeros((frames, 8), dtype=np.float64)
        gait[:360, 0] = 1.0
        gait[360:, 1] = 1.0
        straight = interval_metrics_from_tracks(
            "LocomotionFlat01_000", root, np.zeros(frames), gait
        )
        self.assertTrue(any(metric.has_idle_transition for metric in straight))

        gait[:, :] = 0.0
        gait[:, 1] = 1.0
        yaw = 0.8 * np.sin(np.linspace(0.0, 4.0 * np.pi, frames))
        turning = interval_metrics_from_tracks(
            "LocomotionFlat02_000", root, yaw, gait
        )
        self.assertTrue(
            any(
                metric.maximum_left_turn_rad >= 0.5
                and metric.maximum_right_turn_rad >= 0.5
                for metric in turning
            )
        )

        root[:, 2] = 0.25 * np.sin(np.linspace(0.0, 4.0 * np.pi, frames))
        terrain = interval_metrics_from_tracks(
            "WalkingUpSteps01_000", root, np.zeros(frames), gait
        )
        self.assertTrue(
            any(
                metric.maximum_ascent_degrees >= 3.0
                and metric.minimum_descent_degrees <= -3.0
                for metric in terrain
            )
        )


if __name__ == "__main__":
    unittest.main()
