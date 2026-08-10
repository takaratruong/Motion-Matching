from __future__ import annotations

import math
import unittest

import numpy as np

from mm_sonic.terrain_foothold_planner import (
    FootholdIntent,
    TerrainFootholdPlannerConfig,
    plan_terrain_footholds,
)
from mm_sonic.terrain_oracle.canonical import CanonicalTerrainMesh
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.stair_foothold_anchors import (
    NominalFootSolePose,
    StanceSpan,
)
from mm_sonic.terrain_oracle.terrain_mesh import (
    TerrainMeshIndex,
    mesh_from_heightfield,
)


def _identity() -> RigidTransform:
    return RigidTransform(
        translation_world=np.zeros(3, dtype=np.float32),
        quaternion_world_from_local_wxyz=np.asarray(
            (1.0, 0.0, 0.0, 0.0), dtype=np.float32
        ),
    )


def _heightfield_index(
    height_function,
    *,
    lower: float = -1.0,
    upper: float = 1.0,
    spacing: float = 0.10,
) -> TerrainMeshIndex:
    axis = np.arange(lower, upper + 0.5 * spacing, spacing)
    grid_x, grid_y = np.meshgrid(axis, axis)
    height = np.asarray(height_function(grid_x, grid_y), dtype=np.float64)
    if height.shape == ():
        height = np.full(grid_x.shape, float(height), dtype=np.float64)
    mesh = mesh_from_heightfield(
        height,
        np.ones_like(height, dtype=np.bool_),
        spacing_m=spacing,
        origin_xy=(lower, lower),
    )
    return TerrainMeshIndex(mesh, _identity())


def _quad_mesh(
    quads: tuple[tuple[float, float, float, float, float], ...]
) -> TerrainMeshIndex:
    """Build disjoint horizontal quads as (x0, x1, y0, y1, z)."""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for x0, x1, y0, y1, height in quads:
        start = len(vertices)
        vertices.extend(
            (
                (x0, y0, height),
                (x1, y0, height),
                (x1, y1, height),
                (x0, y1, height),
            )
        )
        faces.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))
    mesh = CanonicalTerrainMesh(
        vertices_local=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        valid_faces=np.ones(len(faces), dtype=np.bool_),
        source_asset_sha256="f" * 64,
    )
    return TerrainMeshIndex(mesh, _identity())


def _flat_sole_pose(
    center_xy: tuple[float, float],
    *,
    height_m: float = 0.0,
    yaw_rad: float = 0.0,
    half_length_m: float = 0.09,
    half_width_m: float = 0.05,
) -> NominalFootSolePose:
    center = np.asarray((center_xy[0], center_xy[1], height_m), dtype=np.float64)
    local = np.asarray(
        (
            (-half_length_m, -half_width_m, 0.0),
            (half_length_m, -half_width_m, 0.0),
            (half_length_m, half_width_m, 0.0),
            (-half_length_m, half_width_m, 0.0),
        ),
        dtype=np.float64,
    )
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    rotation = np.asarray(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    local[:, :2] = local[:, :2] @ rotation.T
    return NominalFootSolePose(
        sole_center_world=center,
        sole_support_points_world=center + local,
        yaw_rad=yaw_rad,
    )


def _fixed_config(**overrides) -> TerrainFootholdPlannerConfig:
    values = dict(
        maximum_longitudinal_adjustment_m=0.0,
        maximum_lateral_adjustment_m=0.0,
        maximum_yaw_adjustment_rad=0.0,
        longitudinal_samples=1,
        lateral_samples=1,
        yaw_samples=1,
        footprint_sample_spacing_m=0.025,
        pelvis_smoothing_iterations=30,
    )
    values.update(overrides)
    return TerrainFootholdPlannerConfig(**values)


class TerrainFootholdPlannerTests(unittest.TestCase):
    def test_zero_only_fast_path_preserves_nominal_world_xy(self):
        terrain = _heightfield_index(lambda x, y: 0.0)
        intents = (
            FootholdIntent(
                StanceSpan(0, 0, 7),
                _flat_sole_pose((0.0, 0.10), yaw_rad=math.radians(60.0)),
            ),
            FootholdIntent(
                StanceSpan(1, 6, 13),
                _flat_sole_pose((0.25, -0.10), yaw_rad=math.radians(-35.0)),
            ),
        )

        result = plan_terrain_footholds(
            terrain,
            intents,
            np.full(13, 0.78),
            config=TerrainFootholdPlannerConfig(
                longitudinal_samples=5,
                lateral_samples=5,
                yaw_samples=5,
                pelvis_smoothing_iterations=10,
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.message)
        assert result.plan is not None
        self.assertEqual(result.diagnostics.sequence_search_stage, "zero_only")
        self.assertEqual(result.diagnostics.candidate_evaluation_count, 2)
        self.assertEqual(result.diagnostics.candidate_cache_hit_count, 0)
        self.assertEqual(result.diagnostics.candidate_pool_sizes, (1, 1))
        for intent, foothold in zip(intents, result.plan.footholds, strict=True):
            np.testing.assert_allclose(
                foothold.sole_center_world[:2],
                intent.nominal_pose.sole_center_world[:2],
                atol=1.0e-12,
            )
        np.testing.assert_allclose(
            result.plan.pelvis_planar_offset_world_m, 0.0, atol=1.0e-12
        )

    def test_shifted_support_sequence_uses_a_coherent_world_frame_path(self):
        nominal_x = (0.0, 0.40, 0.80)
        required_shift_x = (0.06, 0.0, -0.06)
        terrain = _quad_mesh(
            tuple(
                (
                    center + shift - 0.05,
                    center + shift + 0.05,
                    -0.08,
                    0.08,
                    0.0,
                )
                for center, shift in zip(
                    nominal_x, required_shift_x, strict=True
                )
            )
        )
        intents = tuple(
            FootholdIntent(
                StanceSpan(index % 2, index * 8, index * 8 + 4),
                _flat_sole_pose(
                    (center, 0.0),
                    half_length_m=0.04,
                    half_width_m=0.04,
                ),
            )
            for index, center in enumerate(nominal_x)
        )

        result = plan_terrain_footholds(
            terrain,
            intents,
            np.full(20, 0.78),
            config=_fixed_config(
                maximum_longitudinal_adjustment_m=0.06,
                longitudinal_samples=3,
                maximum_planar_reach_change_m=0.09,
                maximum_pelvis_planar_adjustment_step_m=0.015,
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.message)
        assert result.plan is not None
        self.assertEqual(
            result.diagnostics.sequence_search_stage, "longitudinal_only"
        )
        selected_shift = np.asarray(
            [
                foothold.sole_center_world[0]
                - intent.nominal_pose.sole_center_world[0]
                for intent, foothold in zip(
                    intents, result.plan.footholds, strict=True
                )
            ]
        )
        np.testing.assert_allclose(selected_shift, required_shift_x, atol=1.0e-9)
        self.assertLessEqual(
            result.diagnostics.maximum_selected_world_xy_shift_change_m,
            0.09 + 1.0e-12,
        )
        path_step = np.linalg.norm(
            np.diff(result.plan.pelvis_planar_offset_world_m, axis=0), axis=1
        )
        self.assertLessEqual(float(np.max(path_step)), 0.015 + 1.0e-12)
        for intent, shift in zip(intents, required_shift_x, strict=True):
            np.testing.assert_allclose(
                result.plan.pelvis_planar_offset_world_m[
                    intent.span.start_frame : intent.span.stop_frame, 0
                ],
                shift,
                atol=1.0e-12,
            )
        self.assertGreater(result.diagnostics.candidate_cache_hit_count, 0)

    def test_reports_when_supported_footholds_have_no_coherent_sequence(self):
        terrain = _quad_mesh(
            (
                (0.07, 0.17, -0.08, 0.08, 0.0),
                (0.33, 0.43, -0.08, 0.08, 0.0),
            )
        )
        intents = (
            FootholdIntent(
                StanceSpan(0, 0, 7),
                _flat_sole_pose(
                    (0.0, 0.0), half_length_m=0.04, half_width_m=0.04
                ),
            ),
            FootholdIntent(
                StanceSpan(1, 6, 13),
                _flat_sole_pose(
                    (0.50, 0.0), half_length_m=0.04, half_width_m=0.04
                ),
            ),
        )

        result = plan_terrain_footholds(
            terrain,
            intents,
            np.full(13, 0.78),
            config=_fixed_config(
                maximum_longitudinal_adjustment_m=0.12,
                longitudinal_samples=3,
                maximum_planar_reach_change_m=0.09,
                maximum_pelvis_planar_adjustment_step_m=0.015,
            ),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.diagnostics.reason_code,
            "foothold_sequence_is_infeasible",
        )
        self.assertEqual(result.diagnostics.candidate_pool_sizes, (1, 1))
        self.assertGreater(
            result.diagnostics.sequence_transition_evaluation_count, 0
        )

    def test_exact_rate_fallback_handles_nonadjacent_support_handoff(self):
        # Chronological order is A(right), B(left), C(left).  B is a short
        # plant nested inside A, so after B ends the unsupported gap to C is
        # anchored by A, not by the adjacent B entry.  The cheap first-order
        # selector prefers A's zero shift and would create a 20 mm frame step;
        # exact fallback must choose A's 30 mm alternative and stay at 10 mm.
        terrain = _quad_mesh(
            (
                (-0.015, 0.015, -0.03, 0.03, 0.0),
                (0.015, 0.045, -0.03, 0.03, 0.0),
                (0.385, 0.415, -0.03, 0.03, 0.0),
                (0.845, 0.875, -0.03, 0.03, 0.0),
            )
        )
        intents = (
            FootholdIntent(
                StanceSpan(1, 0, 8),
                _flat_sole_pose(
                    (0.0, 0.0), half_length_m=0.01, half_width_m=0.01
                ),
            ),
            FootholdIntent(
                StanceSpan(0, 3, 6),
                _flat_sole_pose(
                    (0.40, 0.0), half_length_m=0.01, half_width_m=0.01
                ),
            ),
            FootholdIntent(
                StanceSpan(0, 10, 14),
                _flat_sole_pose(
                    (0.80, 0.0), half_length_m=0.01, half_width_m=0.01
                ),
            ),
        )

        result = plan_terrain_footholds(
            terrain,
            intents,
            np.full(14, 0.78),
            config=_fixed_config(
                maximum_longitudinal_adjustment_m=0.06,
                longitudinal_samples=5,
                maximum_planar_reach_change_m=0.09,
                maximum_pelvis_planar_adjustment_step_m=0.015,
                require_alternating_feet=False,
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.message)
        assert result.plan is not None
        self.assertTrue(
            result.diagnostics.sequence_search_stage.endswith("_exact_rate")
        )
        selected_shift = np.asarray(
            [
                foothold.sole_center_world[0]
                - intent.nominal_pose.sole_center_world[0]
                for intent, foothold in zip(
                    intents, result.plan.footholds, strict=True
                )
            ]
        )
        np.testing.assert_allclose(selected_shift, (0.03, 0.0, 0.06), atol=1e-9)
        maximum_step = float(
            np.max(
                np.linalg.norm(
                    np.diff(result.plan.pelvis_planar_offset_world_m, axis=0),
                    axis=1,
                )
            )
        )
        self.assertLessEqual(maximum_step, 0.015 + 1.0e-12)

    def test_sequence_continuity_compares_world_shifts_across_local_yaws(self):
        terrain = _quad_mesh(
            (
                (0.01, 0.11, -0.05, 0.05, 0.0),
                (0.41, 0.51, -0.05, 0.05, 0.0),
            )
        )
        intents = (
            FootholdIntent(
                StanceSpan(0, 0, 7),
                _flat_sole_pose(
                    (0.0, 0.0),
                    yaw_rad=0.0,
                    half_length_m=0.04,
                    half_width_m=0.04,
                ),
            ),
            FootholdIntent(
                StanceSpan(1, 6, 13),
                _flat_sole_pose(
                    (0.40, 0.0),
                    yaw_rad=math.pi / 2.0,
                    half_length_m=0.04,
                    half_width_m=0.04,
                ),
            ),
        )

        result = plan_terrain_footholds(
            terrain,
            intents,
            np.full(13, 0.78),
            config=_fixed_config(
                maximum_longitudinal_adjustment_m=0.06,
                maximum_lateral_adjustment_m=0.06,
                longitudinal_samples=3,
                lateral_samples=3,
                maximum_planar_reach_change_m=0.03,
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.message)
        assert result.plan is not None
        self.assertEqual(result.diagnostics.sequence_search_stage, "planar")
        np.testing.assert_allclose(
            [
                foothold.sole_center_world[:2]
                - intent.nominal_pose.sole_center_world[:2]
                for intent, foothold in zip(
                    intents, result.plan.footholds, strict=True
                )
            ],
            ((0.06, 0.0), (0.06, 0.0)),
            atol=1.0e-9,
        )
        self.assertLessEqual(
            result.diagnostics.maximum_selected_world_xy_shift_change_m,
            1.0e-9,
        )

    def test_accepts_rigid_omnidirectional_footholds_on_a_slope(self):
        slope = 0.15
        terrain = _heightfield_index(lambda x, y: slope * x)
        intents = (
            FootholdIntent(
                StanceSpan(0, 0, 6),
                _flat_sole_pose((0.10, 0.12), yaw_rad=math.radians(35.0)),
            ),
            FootholdIntent(
                StanceSpan(1, 5, 11),
                _flat_sole_pose((0.40, -0.12), yaw_rad=math.radians(-20.0)),
            ),
            FootholdIntent(
                StanceSpan(0, 10, 16),
                _flat_sole_pose((0.70, 0.12), yaw_rad=math.radians(75.0)),
            ),
        )

        result = plan_terrain_footholds(
            terrain,
            intents,
            np.full(16, 0.78, dtype=np.float64),
            config=_fixed_config(
                maximum_surface_slope_rad=math.radians(20.0),
                maximum_vertical_adjustment_m=0.20,
                maximum_pelvis_height_step_m=0.025,
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.message)
        assert result.plan is not None
        self.assertEqual(len(result.plan.footholds), 3)
        expected_normal = np.asarray((-slope, 0.0, 1.0), dtype=np.float64)
        expected_normal /= np.linalg.norm(expected_normal)
        for intent, foothold in zip(intents, result.plan.footholds, strict=True):
            np.testing.assert_allclose(
                foothold.surface_normal_world, expected_normal, atol=2.0e-6
            )
            nominal = intent.nominal_pose.sole_support_points_world
            target = foothold.sole_support_points_world
            np.testing.assert_allclose(
                np.linalg.norm(nominal[:, None] - nominal[None, :], axis=2),
                np.linalg.norm(target[:, None] - target[None, :], axis=2),
                atol=1.0e-10,
            )
            for point in foothold.dense_support_points_world:
                hit = terrain.raycast(
                    point + np.asarray((0.0, 0.0, 1.0)),
                    np.asarray((0.0, 0.0, -1.0)),
                )
                self.assertIsNotNone(hit)
                self.assertAlmostEqual(
                    float(point[2]), float(hit.position_world[2]), places=6
                )
        corridor = result.plan.pelvis_height_corridor
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(corridor.selected_height_world_m)))),
            0.025 + 1.0e-9,
        )
        self.assertTrue(
            np.all(
                corridor.selected_height_world_m
                >= corridor.minimum_height_world_m - 1.0e-10
            )
        )

    def test_rejects_a_narrow_ledge_even_when_the_sole_center_is_supported(self):
        terrain = _quad_mesh(((-0.5, 0.5, -0.035, 0.035, 0.0),))
        intent = FootholdIntent(
            StanceSpan(0, 0, 8),
            _flat_sole_pose((0.0, 0.0), half_width_m=0.05),
        )

        result = plan_terrain_footholds(
            terrain,
            (intent,),
            np.full(8, 0.78),
            config=_fixed_config(),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.diagnostics.reason_code,
            "foothold_has_no_rigid_support_patch",
        )
        counts = dict(result.diagnostics.foothold_searches[-1].rejection_counts)
        self.assertEqual(counts.get("missing_surface_support"), 1)
        center_hit = terrain.raycast(
            np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 0.0, -1.0))
        )
        self.assertIsNotNone(center_hit)

    def test_rejects_a_sole_that_straddles_a_step_discontinuity(self):
        terrain = _quad_mesh(
            (
                (-0.5, 0.0, -0.3, 0.3, 0.0),
                (0.0, 0.5, -0.3, 0.3, 0.16),
            )
        )
        intent = FootholdIntent(
            StanceSpan(0, 0, 8),
            _flat_sole_pose((0.0, 0.0), half_length_m=0.10),
        )

        result = plan_terrain_footholds(
            terrain,
            (intent,),
            np.full(8, 0.78),
            config=_fixed_config(
                maximum_vertical_adjustment_m=0.30,
                allow_partial_rigid_support=True,
            ),
        )

        self.assertFalse(result.accepted)
        counts = dict(result.diagnostics.foothold_searches[-1].rejection_counts)
        self.assertEqual(counts.get("surface_too_steep"), 1)

    def test_partial_rigid_support_accepts_spatially_stable_rough_contacts(self):
        terrain = _heightfield_index(
            lambda x, y: (
                0.008
                * np.cos(np.pi * x / 0.09)
                * np.cos(np.pi * y / 0.05)
            ),
            lower=-0.30,
            upper=0.30,
            spacing=0.01,
        )
        intent = FootholdIntent(
            StanceSpan(0, 0, 8), _flat_sole_pose((0.0, 0.0))
        )
        strict = plan_terrain_footholds(
            terrain,
            (intent,),
            np.full(8, 0.78),
            config=_fixed_config(),
        )
        self.assertFalse(strict.accepted)

        partial = plan_terrain_footholds(
            terrain,
            (intent,),
            np.full(8, 0.78),
            config=_fixed_config(allow_partial_rigid_support=True),
        )

        self.assertTrue(partial.accepted, partial.diagnostics.message)
        assert partial.plan is not None
        foothold = partial.plan.footholds[0]
        self.assertTrue(foothold.diagnostics.partial_rigid_support)
        self.assertGreaterEqual(
            foothold.diagnostics.support_contact_point_count, 3
        )
        self.assertIsNotNone(foothold.sole_support_contact_mask)
        self.assertEqual(
            int(np.count_nonzero(foothold.sole_support_contact_mask)),
            foothold.diagnostics.support_contact_point_count,
        )
        self.assertLessEqual(
            foothold.diagnostics.maximum_surface_residual_m, 0.030
        )

    def test_bounded_search_moves_a_full_footprint_off_an_edge(self):
        terrain = _quad_mesh(((-0.5, 0.5, -0.25, 0.25, 0.0),))
        intent = FootholdIntent(
            StanceSpan(0, 0, 8),
            _flat_sole_pose((0.46, 0.0), half_length_m=0.09),
        )

        result = plan_terrain_footholds(
            terrain,
            (intent,),
            np.full(8, 0.78),
            config=_fixed_config(
                maximum_longitudinal_adjustment_m=0.12,
                longitudinal_samples=5,
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.message)
        assert result.plan is not None
        foothold = result.plan.footholds[0]
        self.assertLess(foothold.diagnostics.longitudinal_adjustment_m, 0.0)
        self.assertLessEqual(
            float(np.max(foothold.dense_support_points_world[:, 0])),
            0.5 + 1.0e-9,
        )

    def test_rejects_a_support_plane_that_over_rotates_the_source_sole(self):
        terrain = _heightfield_index(lambda x, y: 0.20 * x)
        intent = FootholdIntent(
            StanceSpan(0, 0, 8),
            _flat_sole_pose((0.0, 0.0)),
        )

        result = plan_terrain_footholds(
            terrain,
            (intent,),
            np.full(8, 0.78),
            config=_fixed_config(
                maximum_sole_tilt_adjustment_rad=math.radians(5.0)
            ),
        )

        self.assertFalse(result.accepted)
        counts = dict(result.diagnostics.foothold_searches[-1].rejection_counts)
        self.assertEqual(counts.get("sole_tilt_adjustment_exceeds_bound"), 1)

    def test_rejects_supported_sole_when_the_full_foot_envelope_hits_a_riser(self):
        terrain = _quad_mesh(
            (
                (-0.5, 0.0, -0.3, 0.3, 0.0),
                (0.0, 0.5, -0.3, 0.3, 0.15),
            )
        )
        base = _flat_sole_pose((-0.12, 0.0), half_length_m=0.07)
        envelope = np.asarray(
            (
                (-0.19, -0.06, 0.0),
                (-0.19, 0.06, 0.0),
                (0.05, -0.06, 0.05),
                (0.05, 0.06, 0.05),
            ),
            dtype=np.float64,
        )
        intent = FootholdIntent(
            StanceSpan(0, 0, 8),
            NominalFootSolePose(
                sole_center_world=base.sole_center_world,
                sole_support_points_world=base.sole_support_points_world,
                yaw_rad=0.0,
                collision_envelope_points_world=envelope,
            ),
        )

        result = plan_terrain_footholds(
            terrain,
            (intent,),
            np.full(8, 0.78),
            config=_fixed_config(),
        )

        self.assertFalse(result.accepted)
        counts = dict(result.diagnostics.foothold_searches[-1].rejection_counts)
        self.assertEqual(counts.get("foot_envelope_collision"), 1)

    def test_returns_diagnostic_when_double_support_has_no_pelvis_height(self):
        terrain = _quad_mesh(
            (
                (-0.55, -0.05, -0.25, 0.25, 0.0),
                (0.05, 0.55, -0.25, 0.25, 0.30),
            )
        )
        intents = (
            FootholdIntent(
                StanceSpan(0, 0, 8),
                _flat_sole_pose((-0.30, 0.08), half_length_m=0.07),
            ),
            FootholdIntent(
                StanceSpan(1, 0, 8),
                _flat_sole_pose((0.30, -0.08), half_length_m=0.07),
            ),
        )

        result = plan_terrain_footholds(
            terrain,
            intents,
            np.full(8, 0.78),
            config=_fixed_config(
                maximum_vertical_adjustment_m=0.35,
                maximum_pelvis_height_adjustment_m=0.50,
                pelvis_reach_slack_m=0.05,
            ),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.diagnostics.reason_code,
            "pelvis_reach_intervals_do_not_overlap",
        )
        self.assertIsNotNone(result.diagnostics.rejected_frame)
        self.assertEqual(len(result.diagnostics.foothold_searches), 2)
        self.assertTrue(
            all(
                diagnostic.accepted_candidate_count == 1
                for diagnostic in result.diagnostics.foothold_searches
            )
        )


if __name__ == "__main__":
    unittest.main()
