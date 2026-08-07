from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr

from mm_sonic.build_terrain_maneuver_pilots import (
    _direct_terrain_transform,
    _terrain_motion_interval,
)
from mm_sonic.prepare_terrain_maneuver_sonic_bundle import (
    _accepted_manifest_selections,
    _clip_stem,
    _copy_physics_ready_terrain,
)
from mm_sonic.terrain_maneuver_composer import (
    HOLD,
    build_bounce_schedule,
    build_maneuver_schedule,
    choose_support_pivot,
    choose_support_pivots,
    command_labels,
    resample_stitched_motion,
)
from mm_sonic.terrain_oracle.stitch import FrameProvenance, StitchedMotion


def _motion(frame_count: int = 160) -> StitchedMotion:
    root = np.zeros((frame_count, 3), dtype=np.float32)
    root[:, 0] = np.arange(frame_count, dtype=np.float32) * 0.01
    root[:, 2] = 0.8
    quaternion = np.zeros((frame_count, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    joints = np.zeros((frame_count, 29), dtype=np.float32)
    joints[:, 0] = np.arange(frame_count, dtype=np.float32) * 0.002
    return StitchedMotion(
        fps=50.0,
        root_position_world=root,
        root_quaternion_world_wxyz=quaternion,
        joint_position=joints,
        provenance=tuple(
            FrameProvenance(0, frame, "synthetic")
            for frame in range(frame_count)
        ),
        seam_indices=(30, 130),
    )


class TerrainManeuverComposerTests(unittest.TestCase):
    def test_physics_ready_binary_usd_crate_is_copied(self) -> None:
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest("USD Python bindings are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "terrain.usd"
            destination = root / "copy.usd"
            stage = Usd.Stage.CreateNew(str(source))
            model = UsdGeom.Xform.Define(stage, "/model").GetPrim()
            stage.SetDefaultPrim(model)
            UsdPhysics.RigidBodyAPI.Apply(model)
            mesh = UsdGeom.Mesh.Define(stage, "/model/mesh").GetPrim()
            UsdPhysics.CollisionAPI.Apply(mesh)
            stage.Save()

            self.assertTrue(source.read_bytes().startswith(b"PXR-USDC"))
            self.assertEqual(
                _copy_physics_ready_terrain(source, destination),
                "source_physics",
            )
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_bundle_stem_distinguishes_nested_pilot_directories(self) -> None:
        first = Path("bank/clip_076/pilots/manifest.json")
        second = Path("bank/clip_083/pilots/manifest.json")

        self.assertNotEqual(
            _clip_stem(first, "event_00_reverse"),
            _clip_stem(second, "event_00_reverse"),
        )

    def test_manifest_selection_expands_only_accepted_pilots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "pilots": [
                            {
                                "label": "kept",
                                "automatic_gate_accepted": True,
                            },
                            {
                                "label": "rejected",
                                "automatic_gate_accepted": False,
                            },
                        ]
                    }
                )
            )

            selections = _accepted_manifest_selections((manifest, manifest))

            self.assertEqual(selections, (f"{manifest.resolve()}#kept",))

    def test_direct_terrain_transform_uses_archive_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terrain = root / "terrain.usd"
            terrain.write_text("#usda 1.0\n")
            archive_path = root / "archive.zarr"
            archive = zarr.open_group(str(archive_path), mode="w")
            archive.create_dataset(
                "terrain_usd_path",
                data=np.asarray((str(terrain),), dtype="U256"),
            )
            archive.create_dataset(
                "terrain_position_env",
                data=np.asarray(((1.0, 2.0, 3.0),), dtype=np.float32),
            )
            archive.create_dataset(
                "terrain_rotation_env_wxyz",
                data=np.asarray(
                    ((0.70710678, 0.0, 0.0, -0.70710678),),
                    dtype=np.float32,
                ),
            )

            position, quaternion, source = _direct_terrain_transform(
                archive_path=archive_path,
                terrain_usd=terrain,
                explicit_position=None,
                explicit_quaternion_wxyz=None,
            )

            np.testing.assert_allclose(position, (1.0, 2.0, 3.0))
            np.testing.assert_allclose(
                quaternion, (0.70710678, 0.0, 0.0, -0.70710678)
            )
            self.assertEqual(source, "archive_lookup")

    def test_stop_restart_holds_exact_pose_and_finishes_source(self) -> None:
        schedule = build_maneuver_schedule(
            160,
            pivot_source_frame=80,
            mode="stop_restart",
            ramp_frames=20,
            hold_frames=15,
        )
        hold = schedule.source_coordinate[schedule.phase == HOLD]

        np.testing.assert_array_equal(hold, np.full(15, 80.0))
        self.assertAlmostEqual(float(schedule.source_coordinate[0]), 0.0)
        self.assertAlmostEqual(float(schedule.source_coordinate[-1]), 159.0)
        delta = np.diff(schedule.source_coordinate)
        self.assertGreaterEqual(float(np.min(delta)), 0.0)
        self.assertLessEqual(float(np.max(delta)), 1.000001)

    def test_reverse_returns_to_source_start_without_velocity_jump(self) -> None:
        schedule = build_maneuver_schedule(
            160,
            pivot_source_frame=80,
            mode="reverse",
            ramp_frames=24,
            hold_frames=10,
        )
        coordinate = schedule.source_coordinate
        hold_indices = np.flatnonzero(schedule.phase == HOLD)

        self.assertAlmostEqual(float(coordinate[-1]), 0.0)
        self.assertTrue(np.all(np.diff(coordinate[: hold_indices[0] + 1]) >= 0.0))
        self.assertTrue(np.all(np.diff(coordinate[hold_indices[-1] :]) <= 0.0))
        # The integrated quintic speed is effectively zero on both sides of
        # the held pose and never exceeds native playback speed.
        delta = np.abs(np.diff(coordinate))
        self.assertLess(float(delta[hold_indices[0] - 1]), 0.001)
        self.assertLess(float(delta[hold_indices[-1]]), 0.001)
        self.assertLessEqual(float(np.max(delta)), 1.000001)

    def test_short_reverse_schedule_keeps_only_requested_hold(self) -> None:
        schedule = build_maneuver_schedule(
            160,
            pivot_source_frame=80,
            mode="reverse",
            ramp_frames=16,
            hold_frames=4,
        )

        hold = schedule.source_coordinate[schedule.phase == HOLD]
        np.testing.assert_array_equal(hold, np.full(4, 80.0))
        self.assertAlmostEqual(float(schedule.source_coordinate[-1]), 0.0)

    def test_bounce_retreats_between_supported_poses_then_finishes(self) -> None:
        schedule = build_bounce_schedule(
            180,
            lower_pivot_source_frame=55,
            upper_pivot_source_frame=115,
            ramp_frames=16,
            hold_frames=4,
        )
        coordinate = schedule.source_coordinate
        hold = coordinate[schedule.phase == HOLD]

        np.testing.assert_array_equal(
            hold,
            np.asarray((115.0,) * 4 + (55.0,) * 4),
        )
        self.assertAlmostEqual(float(coordinate[0]), 0.0)
        self.assertAlmostEqual(float(coordinate[-1]), 179.0)
        self.assertGreater(np.sum(np.diff(coordinate) < 0.0), 20)
        self.assertLessEqual(float(np.max(np.abs(np.diff(coordinate)))), 1.000001)
        self.assertEqual(schedule.pivot_source_frames, (115, 55))

    def test_fractional_resampling_keeps_commands_synchronized(self) -> None:
        source = _motion()
        schedule = build_maneuver_schedule(
            160,
            pivot_source_frame=80,
            mode="stop_restart",
            ramp_frames=20,
            hold_frames=12,
        )
        output = resample_stitched_motion(source, schedule)
        labels = command_labels(output, schedule)

        np.testing.assert_allclose(
            output.root_position_world[:, 0],
            schedule.source_coordinate * 0.01,
            atol=2.0e-7,
        )
        self.assertEqual(labels["command_velocity_world_xy"].shape, (len(output.root_position_world), 2))
        np.testing.assert_allclose(
            labels["command_velocity_world_xy"][schedule.phase == HOLD],
            0.0,
            atol=1.0e-7,
        )

    def test_pivot_requires_central_double_support(self) -> None:
        motion = _motion()
        clearance = np.full((160, 2), 0.10, dtype=np.float64)
        clearance[65:72] = 0.005
        clearance[115:123] = 0.003
        speed = np.full((160, 2), 0.5, dtype=np.float64)
        speed[65:72] = 0.02
        speed[115:123] = 0.02

        pivot = choose_support_pivot(
            motion,
            clearance,
            per_foot_speed_mps=speed,
            terrain_start_frame=40,
            terrain_stop_frame=130,
        )

        self.assertIn(pivot.frame_index, range(66, 71))
        self.assertLessEqual(pivot.left_clearance_m, 0.005)
        self.assertLessEqual(pivot.right_clearance_m, 0.005)

    def test_height_interval_excludes_flat_approach_and_exit(self) -> None:
        motion = _motion(180)
        root = np.asarray(motion.root_position_world).copy()
        root[:45, 2] = 1.4
        root[45:135, 2] = np.linspace(1.4, 0.8, 90)
        root[135:, 2] = 0.8
        motion = StitchedMotion(
            fps=motion.fps,
            root_position_world=root,
            root_quaternion_world_wxyz=motion.root_quaternion_world_wxyz,
            joint_position=motion.joint_position,
            provenance=motion.provenance,
            seam_indices=(),
        )

        start, stop, source = _terrain_motion_interval(motion)

        self.assertEqual(source, "endpoint_height")
        self.assertGreater(start, 45)
        self.assertLess(stop, 135)

    def test_multiple_pivots_cover_distinct_terrain_progress_bands(self) -> None:
        motion = _motion(180)
        clearance = np.full((180, 2), 0.10, dtype=np.float64)
        speed = np.full((180, 2), 0.5, dtype=np.float64)
        for start in (35, 85, 135):
            clearance[start : start + 6] = 0.003
            speed[start : start + 6] = 0.02

        pivots = choose_support_pivots(
            motion,
            clearance,
            count=3,
            per_foot_speed_mps=speed,
            terrain_start_frame=20,
            terrain_stop_frame=160,
            central_fraction=(0.0, 1.0),
        )

        self.assertEqual(len(pivots), 3)
        self.assertIn(pivots[0].frame_index, range(36, 40))
        self.assertIn(pivots[1].frame_index, range(86, 90))
        self.assertIn(pivots[2].frame_index, range(136, 140))


if __name__ == "__main__":
    unittest.main()
