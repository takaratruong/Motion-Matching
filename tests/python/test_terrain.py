import os
import struct
import tempfile
import unittest

import numpy as np

from resources.g1_terrain_builder.terrain import (
    GrailTerrain,
    StepTerrain,
    _densify_faces,
    build_facing_centerline,
    export_heightfield,
    sample_terrain_features,
)


class TerrainTests(unittest.TestCase):
    def test_stationary_centerline_extends_current_heading(self):
        root = np.array([0.0, 0.0])
        headings = np.array([[1.0, 0.0], [1.0, 0.0]])
        path = np.array([[0.0, 0.0], [0.0, 0.0]])

        line = build_facing_centerline(root, headings, path)

        np.testing.assert_allclose(line[-1], [1.0, 0.0], atol=1e-6)

    def test_step_features_are_ground_relative(self):
        terrain = StepTerrain(edge_x=0.5, height=0.29)
        line = np.array([
            [0.0, 0.0],
            [0.25, 0.0],
            [0.50, 0.0],
            [0.75, 0.0],
            [1.00, 0.0],
        ])

        features = sample_terrain_features(terrain, line)

        np.testing.assert_allclose(
            features, [0.0, 0.29, 0.29, 0.29], atol=1e-6)

    def test_heightfield_binary_contract(self):
        terrain = StepTerrain(edge_x=0.5, height=0.29)
        header_size = struct.calcsize("<4sIII4f")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "terrain.bin")
            meta = export_heightfield(
                terrain, (0.0, 1.0, 0.0, 1.0), 0.5, path)
            with open(path, "rb") as stream:
                payload = stream.read()

        magic, version, nx, nz, ox, oz, cell, exterior = struct.unpack(
            "<4sIII4f", payload[:header_size])
        values = np.frombuffer(payload[header_size:], dtype="<f4")
        self.assertEqual((magic, version, nx, nz), (b"G1HF", 1, 3, 3))
        self.assertEqual(
            meta,
            {
                "nx": 3,
                "nz": 3,
                "origin_x": 0.0,
                "origin_z": 0.0,
                "cell_size": 0.5,
                "exterior_height": 0.0,
            },
        )
        self.assertEqual(len(payload), header_size + nx * nz * 4)
        np.testing.assert_allclose(
            [ox, oz, cell, exterior], [0.0, 0.0, 0.5, 0.0])
        np.testing.assert_allclose(
            values.reshape(nz, nx)[0], [0.0, 0.29, 0.29], atol=1e-6)

    def test_face_densification_is_deterministic_for_triangles(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        counts = np.array([3], np.int32)
        indices = np.array([0, 1, 2], np.int32)

        first = _densify_faces(vertices, counts, indices, resolution=3)
        second = _densify_faces(vertices, counts, indices, resolution=3)

        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.any(np.all(np.isclose(first, [1 / 3, 1 / 3, 0]), axis=1)))

    def test_real_grail_curb_has_expected_height_and_stable_obj(self):
        terrain = GrailTerrain.from_base("terrain_curbs__curb_000__000")
        footprint = terrain.footprint()
        self.assertGreater(footprint["height"], 0.1)
        self.assertLess(footprint["height"], 0.5)
        cx = 0.5 * (footprint["x"][0] + footprint["x"][1])
        cz = 0.5 * (footprint["z"][0] + footprint["z"][1])
        self.assertAlmostEqual(
            terrain.height(cx, cz), footprint["height"], places=2)
        self.assertEqual(terrain.height(cx, cz + 10.0), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.obj")
            second = os.path.join(tmp, "second.obj")
            terrain.export_obj(first)
            terrain.export_obj(second)
            with open(first, "rb") as a, open(second, "rb") as b:
                first_bytes = a.read()
                self.assertEqual(first_bytes, b.read())

        text = first_bytes.decode("utf-8")
        vertices = [line for line in text.splitlines() if line.startswith("v ")]
        faces = [line for line in text.splitlines() if line.startswith("f ")]
        self.assertEqual((len(vertices), len(faces)), (120, 30))
        self.assertEqual(faces[0], "f 1 2 3 4")
        self.assertAlmostEqual(
            max(float(line.split()[2]) for line in vertices),
            footprint["height"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
