import unittest

import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_terrain_conformal_swing import (
    TerrainConformalSwingConfig,
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


if __name__ == "__main__":
    unittest.main()
