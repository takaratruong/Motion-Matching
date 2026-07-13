import os
import pickle
import struct
import tempfile
import unittest
import warnings

import numpy as np
from pxr import Usd, UsdGeom

from resources import build_g1_terrain_database as builder
from resources import quat as holden_quat
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    convert_source_clip,
)
from resources.g1_terrain_builder.sources import load_grail
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    GrailTerrain,
    StepTerrain,
    _densify_faces,
    build_facing_centerline,
    export_heightfield,
    sample_terrain_features,
)


GRAIL_USD_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/object_usd"
GRAIL_RECON_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/recon"
GRAIL_ROBOT_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/robot"


class TerrainTests(unittest.TestCase):
    def test_stationary_centerline_extends_current_heading(self):
        root = np.array([0.0, 0.0])
        headings = np.array([[1.0, 0.0], [1.0, 0.0]])
        path = np.array([[0.0, 0.0], [0.0, 0.0]])

        line = build_facing_centerline(root, headings, path)

        np.testing.assert_allclose(line[-1], [1.0, 0.0], atol=1e-6)

    def test_step_features_are_ground_relative(self):
        class ElevatedStepTerrain:
            def height(self, x, z):
                return 1.0 + (0.29 if x >= 0.5 else 0.0)

        terrain = ElevatedStepTerrain()
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

    def test_curved_centerline_features_follow_geometric_arc_distance(self):
        class LinearTerrain:
            def height(self, x, z):
                return x + 10.0 * z

        root = np.array([0.0, 0.0])
        headings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        path = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])

        line = build_facing_centerline(root, headings, path)
        features = sample_terrain_features(LinearTerrain(), line)

        np.testing.assert_allclose(line, path, atol=1e-7)
        np.testing.assert_allclose(features, [0.25, 0.5, 3.0, 5.5], atol=1e-6)

    def test_nonfinite_centerline_is_rejected_before_flat_terrain_query(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            with self.subTest(bad_value=bad_value):
                line = np.array([[0.0, 0.0], [bad_value, 0.0]])
                with self.assertRaisesRegex(ValueError, "centerline.*finite"):
                    sample_terrain_features(FlatTerrain(), line)

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

    def test_heightfield_rejects_non_strict_bounds_without_writing(self):
        cases = (
            ((0.0, 0.0, 0.0, 1.0), "xmax must be greater than xmin"),
            ((0.0, 1.0, 0.0, 0.0), "zmax must be greater than zmin"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (bounds, message) in enumerate(cases):
                with self.subTest(bounds=bounds):
                    path = os.path.join(tmp, f"invalid-{index}.bin")
                    with self.assertRaisesRegex(ValueError, message):
                        export_heightfield(FlatTerrain(), bounds, 0.5, path)
                    self.assertFalse(os.path.exists(path))

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

    def test_densification_rejects_invalid_face_indices(self):
        vertices = np.eye(3)
        counts = np.array([3], np.int32)
        for bad_index in (-1, len(vertices)):
            with self.subTest(bad_index=bad_index):
                indices = np.array([0, 1, bad_index], np.int32)
                with self.assertRaisesRegex(
                    ValueError, r"face indices must be in \[0, 3\)"
                ):
                    _densify_faces(vertices, counts, indices, resolution=3)

    def test_terrain_rejects_topology_that_could_export_invalid_obj_indices(self):
        vertices = np.eye(3)
        counts = np.array([3], np.int32)
        for bad_index in (-1, len(vertices)):
            with self.subTest(bad_index=bad_index):
                indices = np.array([0, 1, bad_index], np.int32)
                with self.assertRaisesRegex(
                    ValueError, r"face indices must be in \[0, 3\)"
                ):
                    GrailTerrain(vertices, counts, indices, vertices)

    def test_full_xz_bounds_include_lower_curb_and_preserve_top_footprint(self):
        vertices = np.array([
            [-1.0, 0.10, 0.0],
            [1.0, 0.10, 0.0],
            [1.0, 0.10, 1.0],
            [-1.0, 0.10, 1.0],
            [-0.5, 0.30, 3.0],
            [0.5, 0.30, 3.0],
            [0.5, 0.30, 4.0],
            [-0.5, 0.30, 4.0],
        ])
        counts = np.array([4, 4], np.int32)
        indices = np.arange(8, dtype=np.int32)
        query_points = np.concatenate((
            vertices,
            np.array([[-1.25, 0.10, -0.25], [1.25, 0.10, 1.25]]),
        ))
        terrain = GrailTerrain(
            vertices, counts, indices, query_points, radius=0.2)

        first = terrain.xz_bounds()
        second = terrain.xz_bounds()
        footprint = terrain.footprint()
        contract, _ = builder._heightfield_contract(terrain)

        self.assertEqual(first, second)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertEqual(first, (-1.25, 1.25, -0.25, 4.0))
        self.assertEqual(footprint["x"], (-0.5, 0.5))
        self.assertEqual(footprint["z"], (3.0, 4.0))
        self.assertEqual(footprint["height"], 0.3)
        self.assertEqual(contract, (-3.25, 3.25, -2.25, 6.0))

    def test_grail_terrain_rejects_empty_or_malformed_geometry(self):
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        nonfinite_vertices = triangle.copy()
        nonfinite_vertices[0, 0] = np.nan
        nonfinite_points = triangle.copy()
        nonfinite_points[0, 2] = np.inf
        cases = (
            (
                np.empty((0, 3)), np.empty(0, np.int32),
                np.empty(0, np.int32), triangle,
                "vertices must have non-empty shape",
            ),
            (
                triangle.ravel(), np.array([3], np.int32),
                np.array([0, 1, 2], np.int32), triangle,
                "vertices must have non-empty shape",
            ),
            (
                triangle, np.empty(0, np.int32), np.empty(0, np.int32),
                triangle, "faces must be non-empty",
            ),
            (
                triangle, np.array([3], np.int32),
                np.array([0, 1, 2], np.int32), np.empty((0, 3)),
                "query points must have non-empty shape",
            ),
            (
                triangle, np.array([3], np.int32),
                np.array([0, 1, 2], np.int32), np.ones((3, 2)),
                "query points must have non-empty shape",
            ),
            (
                nonfinite_vertices, np.array([3], np.int32),
                np.array([0, 1, 2], np.int32), triangle,
                "points must be finite",
            ),
            (
                triangle, np.array([3], np.int32),
                np.array([0, 1, 2], np.int32), nonfinite_points,
                "points must be finite",
            ),
        )
        for vertices, counts, indices, points, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    GrailTerrain(vertices, counts, indices, points)

    def test_real_grail_curb_has_expected_height_and_stable_obj(self):
        base = "terrain_curbs__curb_000__000"
        terrain = GrailTerrain.from_base(base)
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

        stage = Usd.Stage.Open(os.path.join(GRAIL_USD_DIR, base + ".usd"))
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                raw_points = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
                raw_counts = np.asarray(
                    mesh.GetFaceVertexCountsAttr().Get(), np.int32)
                raw_indices = np.asarray(
                    mesh.GetFaceVertexIndicesAttr().Get(), np.int32)
                break
        else:
            self.fail("real GRAIL fixture has no USD mesh")
        with open(os.path.join(GRAIL_RECON_DIR, base + ".pkl"), "rb") as stream:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                reconstruction = pickle.load(stream)
        object_data = reconstruction["obj_data"]
        rotation = np.asarray(object_data["obj_R"], np.float64)[0]
        translation = np.asarray(object_data["obj_t"], np.float64)[0]
        mujoco_world = (rotation @ raw_points.T).T + translation
        expected_vertices = np.column_stack((
            mujoco_world[:, 0], mujoco_world[:, 2], -mujoco_world[:, 1]))
        actual_vertices = np.array([
            [float(value) for value in line.split()[1:]] for line in vertices
        ])
        np.testing.assert_allclose(actual_vertices, expected_vertices, atol=1e-8)

        expected_faces = []
        offset = 0
        for count in raw_counts:
            expected_faces.append(
                (raw_indices[offset:offset + count] + 1).tolist())
            offset += count
        actual_faces = [
            [int(value) for value in line.split()[1:]] for line in faces
        ]
        self.assertEqual(actual_faces, expected_faces)
        self.assertAlmostEqual(actual_vertices[:, 1].max(), footprint["height"], places=6)

    def test_runtime_heightfield_covers_obj_and_motion_lookahead(self):
        base = builder.DEFAULTS["runtime_terrain"]
        terrain = GrailTerrain.from_base(base)
        requested, metadata = builder._heightfield_contract(terrain)
        domain = (
            metadata["origin_x"],
            metadata["origin_x"]
            + (metadata["nx"] - 1) * metadata["cell_size"],
            metadata["origin_z"],
            metadata["origin_z"]
            + (metadata["nz"] - 1) * metadata["cell_size"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            obj_path = os.path.join(tmp, "terrain.obj")
            terrain.export_obj(obj_path)
            with open(obj_path, encoding="utf-8") as stream:
                vertices = np.array([
                    [float(value) for value in line.split()[1:]]
                    for line in stream
                    if line.startswith("v ")
                ])

        target = (
            vertices[:, 0].min() - builder.HEIGHTFIELD_BORDER,
            vertices[:, 0].max() + builder.HEIGHTFIELD_BORDER,
            vertices[:, 2].min() - builder.HEIGHTFIELD_BORDER,
            vertices[:, 2].max() + builder.HEIGHTFIELD_BORDER,
        )
        self.assertAlmostEqual(requested[0], target[0], places=7)
        self.assertAlmostEqual(requested[1], target[1], places=7)
        self.assertAlmostEqual(requested[2], target[2], places=7)
        self.assertAlmostEqual(requested[3], target[3], places=7)
        obj_rounding = 1e-7
        self.assertLessEqual(domain[0], target[0] + obj_rounding)
        self.assertGreaterEqual(domain[1], target[1] - obj_rounding)
        self.assertLessEqual(domain[2], target[2] + obj_rounding)
        self.assertGreaterEqual(domain[3], target[3] - obj_rounding)
        self.assertLess(
            domain[1] - target[1], metadata["cell_size"] + obj_rounding)
        self.assertLess(
            domain[3] - target[3], metadata["cell_size"] + obj_rounding)

        source = load_grail(os.path.join(GRAIL_ROBOT_DIR, base + ".pkl"))
        kinematics = G1Kinematics(builder.DEFAULTS["g1_xml"])
        clip, _, _ = convert_source_clip(source, kinematics, builder.OUTPUT_FPS)
        roots = clip.positions[:, 0][:, [0, 2]].astype(np.float64)
        headings = holden_quat.mul_vec(
            clip.rotations[:, 0].astype(np.float64),
            np.array([0.0, 0.0, 1.0]),
        )[:, [0, 2]]
        runtime_queries = np.concatenate((roots, roots + headings), axis=0)
        self.assertTrue(np.all(runtime_queries[:, 0] >= domain[0]))
        self.assertTrue(np.all(runtime_queries[:, 0] <= domain[1]))
        self.assertTrue(np.all(runtime_queries[:, 1] >= domain[2]))
        self.assertTrue(np.all(runtime_queries[:, 1] <= domain[3]))


if __name__ == "__main__":
    unittest.main()
