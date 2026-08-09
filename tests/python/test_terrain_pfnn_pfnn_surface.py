from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.pfnn_terrain_fit import (
    PFNNTerrainFit,
    save_terrain_fit,
    terrain_height_g1,
)
from mm_sonic.terrain_pfnn.pfnn_surface import (
    PFNN_G1_SCALE,
    PFNN_G1_Z_OFFSET_M,
    PlacedPFNNSurface,
    load_placed_pfnn_surface,
)


def _fit() -> PFNNTerrainFit:
    return PFNNTerrainFit(
        patch=np.arange(16, dtype=np.float64).reshape(4, 4),
        patch_coord=np.array([2.0, 3.0, 4.0, 5.0]),
        contact_center_xz=np.array([0.0, 0.0]),
        patch_height_mean=22.5,
        stance_height_mean=100.0,
        rbf_centers_xz=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        rbf_epsilon=np.array([0.5, 0.5]),
        rbf_weights=np.array([[0.25, -0.25]]),
        source_contacts=np.ones((8, 4), dtype=np.bool_),
        source_start_frame=120,
        source_frame_count=8,
        cycle_start_frame=100,
        cycle_stop_frame=200,
        selected_patch_index=7,
        fitting_error=0.125,
        source_sha256="a" * 64,
        patches_sha256="b" * 64,
    )


class PlacedPFNNSurfaceTest(unittest.TestCase):
    def test_uses_the_frozen_morphology_transform_and_gradient(self) -> None:
        fit = _fit()
        surface = PlacedPFNNSurface(fit)
        source_xy_m = np.array(
            [[-0.01, 0.0], [0.0, 0.01], [0.01, -0.01]], dtype=np.float64
        )
        query = source_xy_m * PFNN_G1_SCALE
        np.testing.assert_allclose(
            surface.height_at(query),
            terrain_height_g1(
                fit,
                query,
                scale=PFNN_G1_SCALE,
                z_offset=PFNN_G1_Z_OFFSET_M,
            ),
            atol=1.0e-12,
            rtol=0.0,
        )
        epsilon = 1.0e-5
        expected_x = (
            surface.height_at(query + [epsilon, 0.0])
            - surface.height_at(query - [epsilon, 0.0])
        ) / (2.0 * epsilon)
        expected_y = (
            surface.height_at(query + [0.0, epsilon])
            - surface.height_at(query - [0.0, epsilon])
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            surface.gradient_at(query, epsilon=epsilon),
            np.column_stack((expected_x, expected_y)),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_loads_safe_fit_and_rejects_transform_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.npz"
            save_terrain_fit(path, _fit())
            loaded = load_placed_pfnn_surface(path)
            self.assertEqual(loaded.scale, PFNN_G1_SCALE)
            self.assertEqual(loaded.z_offset, PFNN_G1_Z_OFFSET_M)
        with self.assertRaisesRegex(ValueError, "frozen"):
            PlacedPFNNSurface(_fit(), scale=1.0)
        with self.assertRaisesRegex(ValueError, "frozen"):
            PlacedPFNNSurface(_fit(), z_offset=0.0)


if __name__ == "__main__":
    unittest.main()
