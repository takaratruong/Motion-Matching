import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest import mock

import numpy as np

from resources.g1_torch_stair_builder import surface as surface_module
from resources.g1_torch_stair_builder.surface import (
    ZUpHeightGrid,
    ZUpTriangleSurface,
    build_source_height_grid,
    rasterize_zup_surface,
    transform_object_vertices,
)
from resources.g1_torch_stair_builder.corpus import PinnedStairSource


def _two_level_surface() -> ZUpTriangleSurface:
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.2],
            [1.0, -1.0, 0.2],
            [0.0, 1.0, 0.2],
            [1.0, 1.0, 0.2],
        ],
        np.float64,
    )
    triangles = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 5, 7], [4, 7, 6]], np.int32
    )
    return ZUpTriangleSurface(vertices, triangles)


class ObjectTransformTests(unittest.TestCase):
    def test_applies_scale_xyzw_rotation_then_translation(self):
        vertices = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, -0.5]])
        transformed = transform_object_vertices(
            vertices,
            position_world=np.array([0.2, -0.3, 1.0]),
            quaternion_world_xyzw=np.array(
                [0.0, 0.0, 2**-0.5, 2**-0.5]
            ),
            scale=np.array([2.0, 3.0, 4.0]),
        )
        np.testing.assert_allclose(
            transformed,
            np.array([[0.2, 1.7, 3.0], [-5.8, -0.3, -1.0]]),
            atol=1e-7,
        )


class ZUpSurfaceTests(unittest.TestCase):
    def test_rasterizer_does_not_query_every_grid_row_through_surface(self):
        surface = _two_level_surface()
        with mock.patch.object(
            surface,
            "height_xy",
            side_effect=AssertionError("row-wise surface query used"),
        ):
            try:
                grid = rasterize_zup_surface(
                    surface,
                    bounds_xy=(-1.0, 1.0, -1.0, 1.0),
                    cell_size_m=0.25,
                )
            except AssertionError as error:
                self.fail(str(error))
        self.assertEqual(grid.height_z.shape, (9, 9))

    def test_triangle_bounded_rasterizer_is_bitwise_reference_equivalent(self):
        self.assertTrue(
            hasattr(surface_module, "_rasterize_zup_surface_reference")
        )
        random = np.random.default_rng(4812)
        vertices = random.uniform(
            [-0.9, -0.8, -0.2], [0.9, 0.8, 0.8], size=(36, 3)
        )
        triangles = np.arange(36, dtype=np.int32).reshape(-1, 3)
        # Include one reversed winding and one projected-degenerate triangle.
        triangles[1] = triangles[1, ::-1]
        vertices[6:9, :2] = [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]
        surface = ZUpTriangleSurface(vertices, triangles)
        expected = surface_module._rasterize_zup_surface_reference(
            surface,
            bounds_xy=(-1.0, 1.0, -1.0, 1.0),
            cell_size_m=0.05,
        )
        actual = rasterize_zup_surface(
            surface,
            bounds_xy=(-1.0, 1.0, -1.0, 1.0),
            cell_size_m=0.05,
        )
        np.testing.assert_array_equal(actual.origin_xy, expected.origin_xy)
        self.assertEqual(actual.cell_size_m, expected.cell_size_m)
        np.testing.assert_array_equal(actual.height_z, expected.height_z)

    def test_source_grid_default_padding_covers_dense_landing_lookahead(self):
        source = PinnedStairSource(
            base="synthetic",
            robot_path=Path("/robot"),
            object_path=Path("/object"),
            usd_path=Path("/mesh"),
            robot_qpos_mujoco=np.zeros((250, 36), np.float32),
            object_position_world=np.zeros(3, np.float32),
            object_quaternion_world_xyzw=np.array(
                [0.0, 0.0, 0.0, 1.0], np.float32
            ),
            object_scale=np.ones(3, np.float32),
            source_fps=25.0,
            source_sha256=MappingProxyType({}),
        )
        with mock.patch(
            "resources.g1_torch_stair_builder.surface.load_source_surface",
            return_value=_two_level_surface(),
        ):
            grid = build_source_height_grid(source, cell_size_m=1.0)
        np.testing.assert_array_equal(grid.origin_xy, [-3.5, -3.5])
        np.testing.assert_allclose(grid.maximum_xy, [3.5, 3.5])

    def test_vertical_triangle_surface_returns_highest_z_and_flat_exterior(self):
        surface = _two_level_surface()
        values = surface.height_xy(
            np.array([[-0.5, 0.0], [0.5, 0.0], [4.0, 4.0]])
        )
        np.testing.assert_allclose(values, [0.0, 0.2, 0.0], atol=1e-7)

    def test_rasterized_grid_uses_fixed_diagonal_and_rejects_outside(self):
        grid = rasterize_zup_surface(
            _two_level_surface(),
            bounds_xy=(-1.0, 1.0, -1.0, 1.0),
            cell_size_m=0.5,
        )
        np.testing.assert_allclose(
            grid.sample_xy(
                np.array([[-0.5, 0.0], [0.5, 0.0]], np.float32)
            ),
            [0.0, 0.2],
            atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            grid.sample_xy(np.array([[100.0, 100.0]], np.float32))

    def test_npz_fields_round_trip_with_strict_validation(self):
        grid = rasterize_zup_surface(
            _two_level_surface(),
            bounds_xy=(-1.0, 1.0, -1.0, 1.0),
            cell_size_m=0.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "terrain.npz"
            np.savez(path, **grid.as_npz_fields())
            loaded = ZUpHeightGrid.load(path)
        np.testing.assert_array_equal(loaded.origin_xy, grid.origin_xy)
        np.testing.assert_array_equal(loaded.height_z, grid.height_z)
        self.assertEqual(loaded.cell_size_m, grid.cell_size_m)

    def test_rejects_nonfinite_degenerate_and_invalid_grid_contracts(self):
        with self.assertRaisesRegex(ValueError, "vertices"):
            ZUpTriangleSurface(
                np.array([[float("nan"), 0.0, 0.0]]),
                np.array([[0, 0, 0]], np.int32),
            )
        with self.assertRaisesRegex(ValueError, "triangle"):
            ZUpTriangleSurface(
                np.zeros((3, 3)), np.array([[0, 1, 3]], np.int32)
            )
        with self.assertRaisesRegex(ValueError, "cell"):
            ZUpHeightGrid(
                np.array([0.0, 0.0], np.float32),
                0.0,
                np.zeros((2, 2), np.float32),
            )


if __name__ == "__main__":
    unittest.main()
