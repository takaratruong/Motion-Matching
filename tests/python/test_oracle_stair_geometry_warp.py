from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from mm_sonic.terrain_oracle.stair_geometry_warp import (
    _archive_terrain_index,
    build_stair_geometry_warp,
    select_best_geometry_compatible_clip,
    warp_archive_clip_to_stair_geometry,
)
from mm_sonic.terrain_oracle.stair_support_route import (
    StairSupportRoute,
    VisibleTread,
)


_MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
_ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None
_REAL_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)
_G1_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)


def _route(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    boundaries: tuple[float, ...],
    heights: tuple[float, ...],
) -> StairSupportRoute:
    direction = np.asarray(end, dtype=np.float64) - np.asarray(
        start, dtype=np.float64
    )
    direction /= np.linalg.norm(direction)
    levels = []
    for index, height in enumerate(heights):
        lo = boundaries[index]
        hi = boundaries[index + 1]
        levels.append(
            VisibleTread(
                height_m=height,
                route_start_distance_m=lo,
                route_stop_distance_m=hi,
                route_start_xy=np.asarray(start) + lo * direction,
                route_stop_xy=np.asarray(start) + hi * direction,
                visible_in_mesh=True,
                left_foothold_center_xy=None,
                right_foothold_center_xy=None,
            )
        )
    return StairSupportRoute(
        start_xy=np.asarray(start, dtype=np.float32),
        end_xy=np.asarray(end, dtype=np.float32),
        levels=tuple(levels),
    )


class StairGeometryWarpTests(unittest.TestCase):
    def test_route_coordinates_map_step_boundaries_and_preserve_lateral_offset(self):
        """Catches a rigid XY transform that ignores different tread depths."""

        source = _route(
            start=(0.0, 0.0),
            end=(3.0, 0.0),
            boundaries=(0.0, 1.0, 2.0, 3.0),
            heights=(0.0, 0.15, 0.30),
        )
        target = _route(
            start=(10.0, 2.0),
            end=(10.0, 6.5),
            boundaries=(0.0, 1.5, 2.5, 4.5),
            heights=(0.0, 0.20, 0.45),
        )
        warp = build_stair_geometry_warp(source, target)

        points = np.asarray(
            (
                (0.0, 0.10, 0.80),
                (1.0, 0.10, 0.95),
                (2.0, 0.10, 1.10),
                (3.0, 0.10, 1.10),
            )
        )
        warped = warp.warp_points(points)

        np.testing.assert_allclose(
            warped[:, :2],
            (
                (9.90, 2.00),
                (9.90, 3.50),
                (9.90, 4.50),
                (9.90, 6.50),
            ),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            warped[:, 2],
            (0.80, 1.00, 1.25, 1.25),
            atol=1.0e-6,
        )

    @unittest.skipUnless(
        _MUJOCO_AVAILABLE
        and _ZARR_AVAILABLE
        and _REAL_ARCHIVE.is_dir()
        and _G1_MODEL.is_file(),
        "real stairs500 archive, G1 model, MuJoCo, and Zarr are required",
    )
    def test_real_clip_uses_only_source_motion_and_target_mesh_route(self):
        """Catches silently copying the target clip's privileged pose scaffold."""

        import zarr

        archive = zarr.open_group(str(_REAL_ARCHIVE), mode="r")
        target_index = 37
        target_start = int(archive["clip_start_idx"][target_index])
        target_stop = int(archive["clip_end_idx"][target_index])
        target_root = np.asarray(
            archive["body_pos_w"][target_start:target_stop, 0]
        )

        result = warp_archive_clip_to_stair_geometry(
            _REAL_ARCHIVE,
            source_clip_index=13,
            source_start_frame=0,
            source_stop_frame=499,
            target_mesh=_archive_terrain_index(archive, target_index),
            target_route_start_xy=target_root[0, :2],
            target_route_end_xy=target_root[-1, :2],
            model_path=_G1_MODEL,
            maximum_joint_correction_rad=0.36,
            maximum_foot_target_error_m=0.001,
        )

        self.assertEqual(len(result.motion.root_position_world), 499)
        self.assertTrue(
            all(
                item.archive_clip_index == 13
                for item in result.motion.provenance
            )
        )
        self.assertEqual(len(result.source_route.levels), 7)
        self.assertEqual(len(result.target_route.levels), 7)
        self.assertLess(result.maximum_joint_correction_rad, 0.36)
        self.assertLess(result.maximum_foot_target_error_m, 0.001)
        self.assertLessEqual(result.maximum_sole_penetration_m, 0.0025 + 1e-7)
        self.assertLessEqual(
            result.maximum_triangle_sphere_penetration_m,
            0.0025 + 1e-7,
        )
        self.assertLess(result.maximum_root_clearance_lift_m, 0.005)
        np.testing.assert_allclose(
            result.motion.root_position_world[[0, -1], :2],
            target_root[[0, -1], :2],
            atol=1.0e-4,
        )

        # The target motion is not the output: arms are copied exactly from
        # source clip 13, while clip 37 has materially different arm motion.
        source_start = int(archive["clip_start_idx"][13])
        source_joints = np.asarray(
            archive["joint_pos"][source_start : source_start + 499]
        )
        target_joints = np.asarray(
            archive["joint_pos"][target_start : target_start + 499]
        )
        joint_names = tuple(str(value) for value in archive["joint_names"][:])
        non_leg = np.asarray(
            [
                index
                for index, name in enumerate(joint_names)
                if not any(
                    token in name
                    for token in ("hip_", "knee_", "ankle_")
                )
            ]
        )
        np.testing.assert_allclose(
            result.motion.joint_position[:, non_leg],
            source_joints[:, non_leg],
            atol=1.0e-6,
        )
        self.assertGreater(
            float(
                np.mean(
                    np.linalg.norm(
                        result.motion.joint_position[:, non_leg]
                        - target_joints[:, non_leg],
                        axis=1,
                    )
                )
            ),
            0.10,
        )

    @unittest.skipUnless(
        _MUJOCO_AVAILABLE
        and _ZARR_AVAILABLE
        and _REAL_ARCHIVE.is_dir()
        and _G1_MODEL.is_file(),
        "real stairs500 archive, G1 model, MuJoCo, and Zarr are required",
    )
    def test_selector_retrieves_geometry_compatible_source_without_clip_rule(self):
        import zarr

        archive = zarr.open_group(str(_REAL_ARCHIVE), mode="r")
        target_start = int(archive["clip_start_idx"][37])
        target_stop = int(archive["clip_end_idx"][37])
        target_root = np.asarray(
            archive["body_pos_w"][target_start:target_stop, 0]
        )
        selected = select_best_geometry_compatible_clip(
            _REAL_ARCHIVE,
            target_clip_index=37,
            target_route_start_xy=target_root[0, :2],
            target_route_end_xy=target_root[-1, :2],
            model_path=_G1_MODEL,
            candidate_limit=1,
            maximum_joint_correction_rad=0.36,
        )

        self.assertEqual(selected.source_clip_index, 13)
        self.assertEqual(selected.target_clip_index, 37)
        self.assertEqual(selected.evaluated_candidate_count, 1)
        self.assertEqual(selected.rejected_candidate_count, 0)
        self.assertEqual(
            {item.archive_clip_index for item in selected.result.motion.provenance},
            {13},
        )

    @unittest.skipUnless(
        _MUJOCO_AVAILABLE
        and _ZARR_AVAILABLE
        and _REAL_ARCHIVE.is_dir()
        and _G1_MODEL.is_file(),
        "real stairs500 archive, G1 model, MuJoCo, and Zarr are required",
    )
    def test_strict_stance_support_rejects_edge_hanging_foot(self):
        """Catches a collision-free stance sole hanging beyond the stair mesh."""

        import zarr

        archive = zarr.open_group(str(_REAL_ARCHIVE), mode="r")
        target_start = int(archive["clip_start_idx"][37])
        target_stop = int(archive["clip_end_idx"][37])
        target_root = np.asarray(
            archive["body_pos_w"][target_start:target_stop, 0]
        )

        with self.assertRaisesRegex(ValueError, "unsupported stance foot"):
            warp_archive_clip_to_stair_geometry(
                _REAL_ARCHIVE,
                source_clip_index=13,
                source_start_frame=0,
                source_stop_frame=499,
                target_clip_index=37,
                target_route_start_xy=target_root[0, :2],
                target_route_end_xy=target_root[-1, :2],
                model_path=_G1_MODEL,
                maximum_joint_correction_rad=0.36,
                maximum_foot_target_error_m=0.001,
                minimum_stance_support_points=2,
            )


if __name__ == "__main__":
    unittest.main()
