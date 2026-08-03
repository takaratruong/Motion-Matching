import unittest

import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_terrain_conformal_swing import (
    TerrainConformalSwingConfig,
    optimize_terrain_conformal_swing,
    terrain_conformal_swing_cost,
)


def _raw_path():
    return torch.tensor(
        (
            (-0.20, 0.00, 0.05),
            (-0.10, 0.00, 0.12),
            (0.00, 0.00, 0.18),
            (0.10, 0.00, 0.12),
            (0.20, 0.00, 0.05),
        ),
        dtype=torch.float32,
    )


class TerrainConformalSwingCostTests(unittest.TestCase):
    def test_config_rejects_nonpositive_clearance_margin(self):
        with self.assertRaisesRegex(ContractError, "clearance margin"):
            TerrainConformalSwingConfig(clearance_margin_m=0.0)

    def test_clear_flat_swing_has_zero_clearance_and_edge_cost(self):
        raw = _raw_path()
        candidate = raw.clone()
        candidate[:, 2] += 0.20

        cost = terrain_conformal_swing_cost(
            candidate[None],
            raw,
            sample_surface=lambda points: torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            ),
            toe_offset_xy=torch.tensor((0.08, 0.0)),
            heel_offset_xy=torch.tensor((-0.05, 0.0)),
            config=TerrainConformalSwingConfig(),
        )

        torch.testing.assert_close(cost.clearance, torch.zeros(1))
        torch.testing.assert_close(cost.edge, torch.zeros(1))
        self.assertGreater(float(cost.reference[0]), 0.0)

    def test_toe_crossing_a_riser_has_clearance_and_edge_cost(self):
        raw = _raw_path()
        candidate = raw.clone()

        def stair(points):
            return torch.where(
                points[..., 0] >= 0.0,
                torch.full_like(points[..., 0], 0.18),
                torch.zeros_like(points[..., 0]),
            )

        cost = terrain_conformal_swing_cost(
            candidate[None],
            raw,
            sample_surface=stair,
            toe_offset_xy=torch.tensor((0.08, 0.0)),
            heel_offset_xy=torch.tensor((-0.05, 0.0)),
            config=TerrainConformalSwingConfig(),
        )

        self.assertGreater(float(cost.clearance[0]), 0.0)
        self.assertGreater(float(cost.edge[0]), 0.0)

    def test_kinked_path_has_larger_smoothness_cost(self):
        raw = _raw_path()
        smooth = raw.clone()
        kinked = raw.clone()
        kinked[2, 2] += 0.20
        paths = torch.stack((smooth, kinked))
        cost = terrain_conformal_swing_cost(
            paths,
            raw,
            sample_surface=lambda points: torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            ),
            toe_offset_xy=torch.tensor((0.08, 0.0)),
            heel_offset_xy=torch.tensor((-0.05, 0.0)),
            config=TerrainConformalSwingConfig(),
        )

        self.assertGreater(float(cost.smoothness[1]), float(cost.smoothness[0]))

    def test_all_mid_toe_heel_queries_are_batched_once(self):
        calls = []

        def surface(points):
            calls.append(tuple(points.shape))
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        raw = _raw_path()
        terrain_conformal_swing_cost(
            torch.stack((raw, raw)),
            raw,
            sample_surface=surface,
            toe_offset_xy=torch.tensor((0.08, 0.0)),
            heel_offset_xy=torch.tensor((-0.05, 0.0)),
            config=TerrainConformalSwingConfig(),
        )

        self.assertEqual(calls, [(2, 5, 5, 2)])

    def test_frame_varying_foot_offsets_follow_turning_geometry(self):
        sampled = []

        def surface(points):
            sampled.append(points.clone())
            return torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            )

        raw = _raw_path()
        toe = torch.tensor(
            ((0.08, 0.0), (0.06, 0.04), (0.0, 0.08), (-0.04, 0.06), (-0.08, 0.0))
        )
        heel = -toe
        terrain_conformal_swing_cost(
            raw[None],
            raw,
            sample_surface=surface,
            toe_offset_xy=toe,
            heel_offset_xy=heel,
            config=TerrainConformalSwingConfig(),
        )

        queries = sampled[0][0]
        torch.testing.assert_close(queries[:, 1] - raw[:, :2], toe)
        torch.testing.assert_close(queries[:, 2] - raw[:, :2], heel)

    def test_invalid_surface_result_fails_closed(self):
        raw = _raw_path()
        with self.assertRaisesRegex(ContractError, "surface sampler"):
            terrain_conformal_swing_cost(
                raw[None],
                raw,
                sample_surface=lambda _points: torch.zeros(1),
                toe_offset_xy=torch.tensor((0.08, 0.0)),
                heel_offset_xy=torch.tensor((-0.05, 0.0)),
                config=TerrainConformalSwingConfig(),
            )


class TerrainConformalSwingOptimizerTests(unittest.TestCase):
    def _optimizer_config(self):
        return TerrainConformalSwingConfig(
            reference_weight=0.25,
            smoothness_weight=1.0,
            clearance_weight=500.0,
            edge_weight=25.0,
            sample_count=256,
            iteration_count=10,
            temperature=0.03,
            perturbation_xy_std_m=0.01,
            perturbation_z_std_m=0.05,
            maximum_xy_deformation_m=0.04,
            maximum_z_deformation_m=0.18,
        )

    def test_optimizer_preserves_non_swing_and_swing_endpoints_bitwise(self):
        raw = torch.tensor(
            tuple((float(x), 0.0, 0.08) for x in torch.linspace(-0.3, 0.3, 7))
        )
        swing = torch.tensor((False, True, True, True, True, True, False))

        result = optimize_terrain_conformal_swing(
            raw,
            swing,
            sample_surface=lambda points: torch.where(
                torch.abs(points[..., 0]) < 0.07,
                torch.full_like(points[..., 0], 0.18),
                torch.zeros_like(points[..., 0]),
            ),
            toe_offset_xy=torch.tensor((0.08, 0.0)),
            heel_offset_xy=torch.tensor((-0.05, 0.0)),
            config=self._optimizer_config(),
            seed=41,
        )

        self.assertTrue(torch.equal(result.path[~swing], raw[~swing]))
        self.assertTrue(torch.equal(result.path[1], raw[1]))
        self.assertTrue(torch.equal(result.path[5], raw[5]))

    def test_optimizer_strictly_improves_a_riser_collision(self):
        raw = torch.tensor(
            tuple((float(x), 0.0, 0.08) for x in torch.linspace(-0.3, 0.3, 7))
        )
        swing = torch.tensor((False, True, True, True, True, True, False))

        result = optimize_terrain_conformal_swing(
            raw,
            swing,
            sample_surface=lambda points: torch.where(
                torch.abs(points[..., 0]) < 0.07,
                torch.full_like(points[..., 0], 0.18),
                torch.zeros_like(points[..., 0]),
            ),
            toe_offset_xy=torch.tensor((0.08, 0.0)),
            heel_offset_xy=torch.tensor((-0.05, 0.0)),
            config=self._optimizer_config(),
            seed=41,
        )

        self.assertTrue(result.improved)
        self.assertLess(
            float(result.optimized_cost.total[0]),
            float(result.raw_cost.total[0]),
        )
        self.assertLess(
            float(result.optimized_cost.clearance[0]),
            float(result.raw_cost.clearance[0]),
        )

    def test_optimizer_is_exactly_deterministic_for_one_seed(self):
        raw = torch.tensor(
            tuple((float(x), 0.0, 0.08) for x in torch.linspace(-0.3, 0.3, 7))
        )
        swing = torch.tensor((False, True, True, True, True, True, False))
        arguments = dict(
            raw_path=raw,
            swing_mask=swing,
            sample_surface=lambda points: torch.where(
                torch.abs(points[..., 0]) < 0.07,
                torch.full_like(points[..., 0], 0.18),
                torch.zeros_like(points[..., 0]),
            ),
            toe_offset_xy=torch.tensor((0.08, 0.0)),
            heel_offset_xy=torch.tensor((-0.05, 0.0)),
            config=self._optimizer_config(),
            seed=19,
        )

        first = optimize_terrain_conformal_swing(**arguments)
        second = optimize_terrain_conformal_swing(**arguments)

        self.assertTrue(torch.equal(first.path, second.path))
        self.assertTrue(
            torch.equal(first.optimized_cost.total, second.optimized_cost.total)
        )

    def test_optimizer_returns_raw_path_when_flat_path_cost_is_zero(self):
        raw = torch.stack(
            (
                torch.linspace(-0.3, 0.3, 7),
                torch.zeros(7),
                torch.full((7,), 0.30),
            ),
            dim=1,
        )
        result = optimize_terrain_conformal_swing(
            raw,
            torch.tensor((False, True, True, True, True, True, False)),
            sample_surface=lambda points: torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            ),
            toe_offset_xy=torch.tensor((0.08, 0.0)),
            heel_offset_xy=torch.tensor((-0.05, 0.0)),
            config=self._optimizer_config(),
            seed=7,
        )

        self.assertFalse(result.improved)
        self.assertTrue(torch.equal(result.path, raw))

    def test_optimizer_rejects_disjoint_swing_mask(self):
        with self.assertRaisesRegex(ContractError, "contiguous"):
            optimize_terrain_conformal_swing(
                _raw_path(),
                torch.tensor((True, True, False, True, True)),
                sample_surface=lambda points: torch.zeros(
                    points.shape[:-1], dtype=points.dtype, device=points.device
                ),
                toe_offset_xy=torch.tensor((0.08, 0.0)),
                heel_offset_xy=torch.tensor((-0.05, 0.0)),
                config=self._optimizer_config(),
                seed=0,
            )

    def test_optimizer_rejects_invalid_surface_samples(self):
        with self.assertRaisesRegex(ContractError, "surface sampler"):
            optimize_terrain_conformal_swing(
                _raw_path(),
                torch.ones(5, dtype=torch.bool),
                sample_surface=lambda _points: torch.full((1,), float("nan")),
                toe_offset_xy=torch.tensor((0.08, 0.0)),
                heel_offset_xy=torch.tensor((-0.05, 0.0)),
                config=self._optimizer_config(),
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
