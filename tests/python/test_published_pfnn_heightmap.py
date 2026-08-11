from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np


class PublishedHeightmapMeshTests(unittest.TestCase):
    def test_reproduces_released_loader_grid_and_coordinate_transform(self) -> None:
        from mm_sonic.published_pfnn_heightmap import build_mesh

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "height.txt"
            path.write_text("1 2\n3 4\n")
            vertices, faces = build_mesh(path)

        self.assertEqual(vertices.shape, (6, 3))
        self.assertEqual(faces.shape, (4, 3))
        source_values = np.asarray(((1.0, 2.0, 2.0), (3.0, 4.0, 4.0)))
        expected_height = 3.0 * (source_values - source_values.mean()) / 100.0
        np.testing.assert_allclose(
            vertices[:, 2].reshape(3, 2), expected_height.T, atol=1e-12
        )
        self.assertLess(vertices[-1, 1], vertices[0, 1])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "height.txt"
            path.write_text("1 2\n3 4\n")
            scaled_vertices, scaled_faces = build_mesh(path, uniform_scale=0.875)
        np.testing.assert_allclose(scaled_vertices, vertices * 0.875, atol=1e-12)
        np.testing.assert_array_equal(scaled_faces, faces)


if __name__ == "__main__":
    unittest.main()
