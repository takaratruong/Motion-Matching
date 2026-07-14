import json
import os
import pickle
import struct
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
from pxr import Usd, UsdGeom

from resources import build_g1_terrain_database as builder
from resources import quat as holden_quat
from resources.g1_terrain_builder import terrain as terrain_module
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    convert_source_clip,
)
from resources.g1_terrain_builder.sources import load_grail
from resources.g1_terrain_builder.scenes import (
    GRAIL_DEFAULT_BASE,
    build_scene,
    grail_scene_definition,
)
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    GrailTerrain,
    HeightGrid,
    StepTerrain,
    VerticalTriangleSurface,
    build_facing_centerline,
    export_heightfield,
    export_heightfield_obj,
    grail_surface_parity,
    rasterize_heightfield,
    sample_terrain_features,
    surface_semantics,
    surface_semantics_signature,
    triangulate_faces,
)


GRAIL_USD_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/object_usd"
GRAIL_RECON_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/recon"
GRAIL_ROBOT_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/robot"
GRAIL_PARITY_BASES = (
    GRAIL_DEFAULT_BASE,
    "terrain_curbs__curb_186__004",
    "terrain_curbs__curb_022__001",
    "terrain_curbs__curb_165__006",
)


class TerrainTests(unittest.TestCase):
    def test_selected_grail_grids_meet_all_surface_parity_tolerances(self):
        for base in GRAIL_PARITY_BASES:
            with self.subTest(base=base):
                terrain = GrailTerrain.from_base(base)
                xmin, xmax, zmin, zmax = terrain.xz_bounds()
                grid = rasterize_heightfield(
                    terrain, (xmin - 1.0, xmax + 1.0,
                              zmin - 1.0, zmax + 1.0), 0.02)
                report = grail_surface_parity(terrain, grid)
                self.assertLessEqual(report["node_error_m"], 1e-6)
                self.assertLessEqual(report["within_cell_error_m"], 1e-4)
                self.assertLessEqual(
                    report["source_away_edge_error_m"], 0.005)
                self.assertLessEqual(
                    report["top_edge_movement_m"], 0.02 + 1e-9)
                self.assertGreater(report["away_edge_probe_count"], 0)

    def test_real_grail_obj_vertices_and_faces_are_the_grid(self):
        terrain = GrailTerrain.from_base(GRAIL_DEFAULT_BASE)
        xmin, xmax, zmin, zmax = terrain.xz_bounds()
        grid = rasterize_heightfield(
            terrain, (xmin - 0.04, xmax + 0.04,
                      zmin - 0.04, zmax + 0.04), 0.02)
        lines = grid.obj_bytes().decode("utf-8").splitlines()
        vertex_lines = lines[:grid.nx * grid.nz]
        face_lines = lines[grid.nx * grid.nz:]
        vertices = np.array([
            [float(value) for value in line.split()[1:]]
            for line in vertex_lines
        ], dtype=np.float32)

        def runtime_coordinate(value):
            encoded = np.float32(value)
            return np.float32(0.0) if encoded == 0.0 else encoded

        expected = []
        for iz in range(grid.nz):
            for ix in range(grid.nx):
                expected.append([
                    runtime_coordinate(
                        grid.origin_x + ix * grid.cell_size),
                    grid.heights[iz, ix],
                    runtime_coordinate(
                        grid.origin_z + iz * grid.cell_size),
                ])
        expected = np.asarray(expected, dtype=np.float32)
        np.testing.assert_array_equal(
            vertices.view(np.uint32), expected.view(np.uint32))
        expected_faces = []
        for iz in range(grid.nz - 1):
            for ix in range(grid.nx - 1):
                p00 = iz * grid.nx + ix + 1
                p10, p01, p11 = p00 + 1, p00 + grid.nx, p00 + grid.nx + 1
                expected_faces.extend(
                    [f"f {p00} {p11} {p10}", f"f {p00} {p01} {p11}"])
        self.assertEqual(face_lines, expected_faces)

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

    def test_legacy_export_heightfield_remains_g1hf_v1(self):
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

    def test_height_grid_uses_one_fixed_diagonal_for_height_and_normal(self):
        grid = HeightGrid(
            np.array([[0.0, 2.0], [4.0, 10.0]]),
            1.0, -2.0, 2.0, -3.0,
        )

        first = (grid.origin_x + 0.75 * grid.cell_size,
                 grid.origin_z + 0.25 * grid.cell_size)
        second = (grid.origin_x + 0.25 * grid.cell_size,
                  grid.origin_z + 0.75 * grid.cell_size)
        diagonal = (grid.origin_x + 0.5 * grid.cell_size,
                    grid.origin_z + 0.5 * grid.cell_size)

        self.assertAlmostEqual(grid.height(*first), 3.5)
        self.assertAlmostEqual(grid.height(*second), 4.5)
        self.assertAlmostEqual(grid.height(*diagonal), 5.0)
        expected_first = np.array([-1.0, 1.0, -4.0])
        expected_first /= np.linalg.norm(expected_first)
        expected_second = np.array([-3.0, 1.0, -2.0])
        expected_second /= np.linalg.norm(expected_second)
        np.testing.assert_allclose(grid.normal(*first), expected_first)
        np.testing.assert_allclose(grid.normal(*diagonal), expected_first)
        np.testing.assert_allclose(grid.normal(*second), expected_second)
        self.assertGreater(grid.normal(*first)[1], 0.0)
        self.assertGreater(grid.normal(*second)[1], 0.0)
        self.assertEqual(grid.height(grid.origin_x - 1.0, grid.origin_z), -3.0)
        np.testing.assert_array_equal(
            grid.normal(grid.origin_x - 1.0, grid.origin_z), [0.0, 1.0, 0.0])

    def test_height_grid_normal_uses_scale_safe_binary64_normalization(self):
        grid = HeightGrid(
            np.array([
                [0.0, 0.0],
                [-4.124587893450382e-10, 67052.109375],
            ]),
            0.0, 0.0, 1.0, 0.0,
        )
        h00 = float(grid.heights[0, 0])
        h01 = float(grid.heights[1, 0])
        h11 = float(grid.heights[1, 1])
        gradient_normal = np.array([
            -(h11 - h01),
            1.0,
            -(h01 - h00),
        ])
        scaled = gradient_normal / np.max(np.abs(gradient_normal))
        expected = scaled / np.linalg.norm(scaled)

        np.testing.assert_array_equal(grid.normal(0.25, 0.75), expected)

    def test_height_grid_nodes_grid_lines_and_world_edges_are_inclusive(self):
        heights = np.array([
            [0.0, 1.0, 4.0],
            [2.0, 8.0, 16.0],
            [3.0, 12.0, 25.0],
        ])
        grid = HeightGrid(heights, -0.31, 0.73, 0.19, -100.0)

        for iz in range(grid.nz):
            for ix in range(grid.nx):
                with self.subTest(node=(ix, iz)):
                    x = grid.origin_x + ix * grid.cell_size
                    z = grid.origin_z + iz * grid.cell_size
                    self.assertEqual(grid.height(x, z), grid.heights[iz, ix])

        x_line = grid.origin_x + grid.cell_size
        z_line = grid.origin_z + grid.cell_size
        self.assertEqual(grid.height(x_line, z_line), grid.heights[1, 1])
        expected = np.array([
            -(grid.heights[1, 2] - grid.heights[1, 1]) / grid.cell_size,
            1.0,
            -(grid.heights[2, 2] - grid.heights[1, 2]) / grid.cell_size,
        ])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(grid.normal(x_line, z_line), expected)
        self.assertEqual(grid.height(grid.max_x, grid.max_z), heights[-1, -1])

        edge_queries = (
            (grid.origin_x, grid.origin_z + 0.5 * grid.cell_size),
            (grid.max_x, grid.origin_z + 0.5 * grid.cell_size),
            (grid.origin_x + 0.5 * grid.cell_size, grid.origin_z),
            (grid.origin_x + 0.5 * grid.cell_size, grid.max_z),
        )
        for query in edge_queries:
            with self.subTest(edge=query):
                self.assertNotEqual(grid.height(*query), grid.exterior_height)

    def test_height_grid_nextafter_classifies_each_world_edge_explicitly(self):
        grid = HeightGrid(
            np.arange(12, dtype=np.float64).reshape(3, 4) + 5.0,
            0.1000000009, -0.3000000041, 0.0700000007, -99.0,
        )
        center_x = 0.5 * (grid.origin_x + grid.max_x)
        center_z = 0.5 * (grid.origin_z + grid.max_z)
        edges = (
            (grid.origin_x, center_z, np.inf, -np.inf),
            (grid.max_x, center_z, -np.inf, np.inf),
            (center_x, grid.origin_z, np.inf, -np.inf),
            (center_x, grid.max_z, -np.inf, np.inf),
        )
        for x, z, inward_direction, outward_direction in edges:
            if x in (grid.origin_x, grid.max_x):
                inward = (np.nextafter(x, inward_direction), z)
                outward = (np.nextafter(x, outward_direction), z)
            else:
                inward = (x, np.nextafter(z, inward_direction))
                outward = (x, np.nextafter(z, outward_direction))
            with self.subTest(edge=(x, z)):
                self.assertNotEqual(grid.height(*inward), grid.exterior_height)
                self.assertEqual(grid.height(*outward), grid.exterior_height)
                np.testing.assert_array_equal(
                    grid.normal(*outward), [0.0, 1.0, 0.0])

    def test_height_grid_binary_header_is_float32_coordinate_authority(self):
        source = np.array([
            [0.125000001, 1.25000003, 2.50000006],
            [3.75000009, 5.00000012, 6.25000015],
        ], np.float64)
        grid = HeightGrid(
            source, 0.1000000009, -0.3000000041, 0.0700000007,
            -1.70000004,
        )
        payload = grid.g1hf_bytes()
        header = struct.Struct("<4sIII4f")
        magic, version, nx, nz, ox, oz, cell, exterior = \
            header.unpack_from(payload)
        values = np.frombuffer(payload, dtype="<f4", offset=header.size)

        self.assertEqual((magic, version, nx, nz), (b"G1HF", 2, 3, 2))
        self.assertEqual(len(payload), header.size + nx * nz * 4)
        np.testing.assert_array_equal(values.reshape(nz, nx), source.astype("<f4"))
        self.assertEqual(
            (grid.origin_x, grid.origin_z, grid.cell_size, grid.exterior_height),
            (ox, oz, cell, exterior),
        )
        self.assertEqual(grid.height(ox + 2 * cell, oz + cell), values[-1])
        self.assertEqual(grid.metadata(), {
            "schema": "G1HF/v2",
            "version": 2,
            "nx": nx,
            "nz": nz,
            "origin_x": ox,
            "origin_z": oz,
            "cell_size_m": cell,
            "exterior_height_m": exterior,
            "interpolation": "fixed-diagonal-triangles",
            "diagonal": "min-x-min-z_to_max-x-max-z",
        })
        obj_vertices = [
            line for line in grid.obj_bytes().decode("ascii").splitlines()
            if line.startswith("v ")
        ]
        decoded_vertices = np.array([
            [float(component) for component in line.split()[1:]]
            for line in obj_vertices
        ])
        expected_coordinates = np.array([
            [float(np.float32(ox + ix * cell)),
             float(values[iz * nx + ix]),
             float(np.float32(oz + iz * cell))]
            for iz in range(nz) for ix in range(nx)
        ])
        np.testing.assert_allclose(
            decoded_vertices, expected_coordinates, rtol=0.0, atol=5e-9)

    def test_height_grid_canonicalizes_every_serialized_zero_to_positive(self):
        grid = HeightGrid(
            np.full((2, 2), -0.0, dtype=np.float32),
            -0.0, -0.0, 0.5, -0.0,
        )
        payload = grid.g1hf_bytes()
        header = struct.Struct("<4sIII4f")
        _, _, nx, nz, ox, oz, _, exterior = header.unpack_from(payload)
        scalar_bits = [
            struct.unpack("<I", struct.pack("<f", value))[0]
            for value in (ox, oz, exterior)
        ]
        height_bits = np.frombuffer(
            payload, "<u4", nx * nz, offset=header.size)

        self.assertEqual(scalar_bits, [0, 0, 0])
        np.testing.assert_array_equal(height_bits, 0)
        self.assertEqual(
            struct.unpack("<I", struct.pack("<f", grid.metadata()[
                "origin_x"]))[0],
            0,
        )
        self.assertNotIn(b"-0", grid.obj_bytes())

    def test_height_grid_owns_read_only_little_endian_height_bytes(self):
        caller = np.arange(6, dtype=np.float64).reshape(2, 3)
        grid = HeightGrid(caller, 0.1, -0.2, 0.3, -4.0)
        query = (grid.max_x, grid.max_z)
        before = (
            grid.height(*query),
            grid.g1hf_bytes(),
            grid.obj_bytes(),
            json.dumps(grid.metadata(), sort_keys=True).encode("ascii"),
        )

        caller[:] = 1000.0

        after = (
            grid.height(*query),
            grid.g1hf_bytes(),
            grid.obj_bytes(),
            json.dumps(grid.metadata(), sort_keys=True).encode("ascii"),
        )
        self.assertEqual(before, after)
        self.assertTrue(grid.heights.flags.owndata)
        self.assertTrue(grid.heights.flags.c_contiguous)
        self.assertEqual(grid.heights.dtype.str, "<f4")
        self.assertFalse(grid.heights.flags.writeable)
        with self.assertRaises(ValueError):
            grid.heights[0, 0] = 1.0

    def test_height_grid_obj_bytes_lock_order_winding_format_and_newline(self):
        grid = HeightGrid(
            np.array([[0.0, 1.25, -2.5], [3.75, 5.125, 6.5]]),
            0.1000000009, -0.3000000041, 0.0700000007, -1.0,
        )
        binary = grid.g1hf_bytes()
        header = struct.Struct("<4sIII4f")
        _, _, nx, nz, ox, oz, cell, _ = header.unpack_from(binary)
        heights = np.frombuffer(binary, "<f4", nx * nz, header.size).reshape(nz, nx)
        expected = []
        for iz in range(nz):
            z = float(np.float32(oz + iz * cell))
            for ix in range(nx):
                x = float(np.float32(ox + ix * cell))
                expected.append(
                    f"v {x:.9g} {float(heights[iz, ix]):.9g} {z:.9g}\n")
        expected.extend((
            "f 1 5 2\n", "f 1 4 5\n",
            "f 2 6 3\n", "f 2 5 6\n",
        ))
        expected_bytes = "".join(expected).encode("ascii")

        self.assertEqual(grid.obj_bytes(), expected_bytes)
        self.assertTrue(grid.obj_bytes().endswith(b"\n"))
        vertices = np.array([
            [float(value) for value in line.split()[1:]]
            for line in expected[:nx * nz]
        ])
        for face in ((1, 5, 2), (1, 4, 5), (2, 6, 3), (2, 5, 6)):
            a, b, c = (vertices[index - 1] for index in face)
            self.assertGreater(np.cross(b - a, c - a)[1], 0.0)

    def test_height_grid_obj_coordinates_round_trip_to_runtime_float32_bits(self):
        grid = HeightGrid(
            np.zeros((2, 155)),
            7.073084831237793,
            -0.3000000041,
            0.02,
            0.0,
        )
        lines = grid.obj_bytes().decode("ascii").splitlines()

        for iz, ix in ((0, 0), (0, 153), (1, 154)):
            fields = lines[iz * grid.nx + ix].split()
            parsed_x = np.float32(fields[1])
            parsed_z = np.float32(fields[3])
            expected_x = np.float32(
                grid.origin_x + ix * grid.cell_size)
            expected_z = np.float32(
                grid.origin_z + iz * grid.cell_size)
            with self.subTest(node=(ix, iz), axis="x"):
                self.assertEqual(
                    int(parsed_x.view(np.uint32)),
                    int(expected_x.view(np.uint32)),
                )
            with self.subTest(node=(ix, iz), axis="z"):
                self.assertEqual(
                    int(parsed_z.view(np.uint32)),
                    int(expected_z.view(np.uint32)),
                )

    def test_height_grid_rejects_invalid_shape_values_metadata_and_sizes(self):
        invalid_heights = (
            np.array([0.0, 1.0]),
            np.zeros((1, 2)),
            np.zeros((2, 1)),
            np.array([[0.0, 1.0], [2.0, np.nan]]),
            np.array([[0.0, 1.0], [2.0, np.inf]]),
            np.array([[0.0, 1.0], [2.0, 1e100]]),
        )
        for heights in invalid_heights:
            with self.subTest(shape=heights.shape):
                with self.assertRaises(ValueError):
                    HeightGrid(heights, 0.0, 0.0, 0.02, 0.0)

        invalid_scalars = (
            (np.nan, 0.0, 0.02, 0.0),
            (0.0, np.inf, 0.02, 0.0),
            (0.0, 0.0, 1e100, 0.0),
            (0.0, 0.0, 1e-100, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, -0.02, 0.0),
            (0.0, 0.0, 0.02, -1e100),
        )
        for origin_x, origin_z, cell, exterior in invalid_scalars:
            with self.subTest(scalars=(origin_x, origin_z, cell, exterior)):
                with self.assertRaises(ValueError):
                    HeightGrid(
                        np.zeros((2, 2)), origin_x, origin_z, cell, exterior)

        limit = np.iinfo(np.int32).max
        one = np.zeros(1, dtype=np.float32)
        too_wide = np.lib.stride_tricks.as_strided(
            one, shape=(2, limit + 1), strides=(0, 0))
        too_many = np.lib.stride_tricks.as_strided(
            one, shape=(2, limit // 2 + 1), strides=(0, 0))
        for heights in (too_wide, too_many):
            with self.subTest(shape=heights.shape):
                with self.assertRaisesRegex(ValueError, "INT_MAX"):
                    HeightGrid(heights, 0.0, 0.0, 0.02, 0.0)

    def test_height_grid_rejects_nonfinite_queries(self):
        grid = HeightGrid(np.zeros((2, 2)), 0.0, 0.0, 0.02, 0.0)
        for query in ((np.nan, 0.0), (0.0, np.inf), (-np.inf, 0.0)):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "finite"):
                    grid.height(*query)
                with self.assertRaisesRegex(ValueError, "finite"):
                    grid.normal(*query)

    def test_height_grid_rejects_complex_heights_without_lossy_cast(self):
        heights = np.array(
            [[0.0, 1.0], [2.0, 3.0 + 4.0j]], np.complex128)

        with self.assertRaisesRegex(ValueError, "real-valued"):
            HeightGrid(heights, 0.0, 0.0, 0.02, 0.0)

    def test_height_grid_requires_runtime_distinguishable_axis_nodes(self):
        cases = (
            ("x binary64 collapse", (2, 2), 1e20, 0.0, 1e-20, "X"),
            ("z binary64 collapse", (2, 2), 0.0, 1e20, 1e-20, "Z"),
            ("x binary32 collapse", (2, 2), 1e8, 0.0, 0.02, "X"),
            ("z binary32 collapse", (2, 2), 0.0, 1e8, 0.02, "Z"),
            ("x interior collapse", (2, 4), 1e8, 0.0, 5.0, "X"),
            ("z interior collapse", (4, 2), 0.0, 1e8, 5.0, "Z"),
            (
                "x unaligned equal-spacing collapse",
                (2, 5),
                float(np.nextafter(np.float32(2.0), np.float32(-np.inf))),
                0.0,
                float(np.spacing(np.float32(2.0))),
                "X",
            ),
        )
        for label, shape, origin_x, origin_z, cell, axis in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{axis}.*runtime-distinguishable",
                ):
                    HeightGrid(
                        np.zeros(shape), origin_x, origin_z, cell,
                        exterior_height=0.0)

    def test_height_grid_rejects_binary32_subnormal_scalars_and_samples(self):
        minimum_subnormal = float(np.nextafter(
            np.float32(0.0), np.float32(np.inf)))
        normal_cell = float(np.finfo(np.float32).tiny)
        valid_heights = np.zeros((2, 2))
        cases = (
            ("origin_x", valid_heights, minimum_subnormal, 0.0,
             normal_cell, 0.0, "origin_x"),
            ("origin_z", valid_heights, 0.0, -minimum_subnormal,
             normal_cell, 0.0, "origin_z"),
            ("cell", valid_heights, 0.0, 0.0,
             minimum_subnormal, 0.0, "cell_size"),
            ("exterior", valid_heights, 0.0, 0.0,
             normal_cell, minimum_subnormal, "exterior_height"),
            (
                "height",
                np.array([[0.0, minimum_subnormal], [0.0, 0.0]]),
                0.0, 0.0, normal_cell, 0.0, "heights",
            ),
        )
        for label, heights, ox, oz, cell, exterior, field in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError, f"{field}.*normal-or-zero|{field}.*normal"
                ):
                    HeightGrid(heights, ox, oz, cell, exterior)

        underflowed_zero = HeightGrid(
            np.full((2, 2), 1e-100),
            1e-100,
            -1e-100,
            normal_cell,
            -1e-100,
        )
        self.assertEqual(underflowed_zero.origin_x, 0.0)
        self.assertEqual(underflowed_zero.origin_z, -0.0)
        self.assertEqual(underflowed_zero.exterior_height, -0.0)
        np.testing.assert_array_equal(underflowed_zero.heights, 0.0)

        aligned_cell = float(np.spacing(np.float32(2.0)))
        aligned = HeightGrid(
            np.zeros((2, 5)), 2.0, 0.0, aligned_cell, 0.0)
        self.assertEqual(aligned.cell_size, aligned_cell)

    def test_height_grid_rejects_derived_subnormal_runtime_node(self):
        minimum_normal = np.float32(np.finfo(np.float32).tiny)
        minimum_subnormal = np.nextafter(
            np.float32(0.0), np.float32(np.inf))
        next_normal = np.nextafter(minimum_normal, np.float32(np.inf))
        derived = np.float32(-minimum_normal + next_normal)
        self.assertNotEqual(derived, 0.0)
        self.assertLess(abs(derived), minimum_normal)

        with self.assertRaisesRegex(
            ValueError, "X.*normal-or-zero"
        ):
            HeightGrid(
                np.zeros((2, 2)),
                float(-minimum_normal),
                0.0,
                float(next_normal),
                0.0,
            )

        interior_origin = np.float32(
            float(minimum_subnormal) - 3.0 * float(next_normal))
        interior_nodes = np.array([
            np.float32(float(interior_origin) + index * float(next_normal))
            for index in range(7)
        ])
        self.assertTrue(all(
            terrain_module._is_normal_or_zero_binary32(value)
            for value in interior_nodes[[0, 1, -2, -1]]
        ))
        self.assertNotEqual(interior_nodes[3], 0.0)
        self.assertLess(abs(interior_nodes[3]), minimum_normal)
        with self.assertRaisesRegex(ValueError, "X.*normal-or-zero"):
            HeightGrid(
                np.zeros((2, 7)),
                float(interior_origin),
                0.0,
                float(next_normal),
                0.0,
            )

    def test_height_grid_huge_axis_validation_is_constant_time(self):
        limit = np.iinfo(np.int32).max
        backing = np.zeros(1, np.float32)
        heights = np.lib.stride_tricks.as_strided(
            backing,
            shape=(2, limit // 2),
            strides=(0, 0),
        )
        calls = []
        implementation = terrain_module._runtime_node_coordinate

        def counted_coordinate(origin, index, cell_size):
            calls.append(index)
            if len(calls) > 8:
                raise AssertionError("axis validation iterated over the grid")
            return implementation(origin, index, cell_size)

        with mock.patch.object(
            terrain_module,
            "_runtime_node_coordinate",
            side_effect=counted_coordinate,
        ):
            with self.assertRaisesRegex(ValueError, "runtime-distinguishable"):
                HeightGrid(heights, 0.0, 0.0, 1.0, 0.0)

        self.assertLessEqual(len(calls), 8)

    def test_constant_time_axis_validation_has_no_seeded_false_accepts(self):
        rng = np.random.default_rng(20260713)
        accepted = 0
        for _ in range(4000):
            origin_bits = np.uint32(rng.integers(0, 2**32, dtype=np.uint64))
            cell_bits = np.uint32(rng.integers(
                0x00800000, 0x7f800000, dtype=np.uint64))
            origin = float(np.asarray(origin_bits).view(np.float32))
            cell = float(np.asarray(cell_bits).view(np.float32))
            count = int(rng.integers(2, 33))
            if not np.isfinite(origin) or not np.isfinite(cell):
                continue
            if not terrain_module._is_normal_or_zero_binary32(origin):
                continue
            try:
                terrain_module._validate_runtime_axis(
                    origin, count, cell, "X")
            except ValueError:
                continue
            accepted += 1
            nodes = [
                terrain_module._runtime_node_coordinate(origin, index, cell)
                for index in range(count)
            ]
            source = [node[0] for node in nodes]
            runtime = [node[1] for node in nodes]
            self.assertTrue(np.isfinite(source).all())
            self.assertTrue(np.isfinite(runtime).all())
            self.assertTrue(all(
                terrain_module._is_normal_or_zero_binary32(value)
                for value in runtime
            ))
            self.assertTrue(all(
                source[index] > source[index - 1]
                and runtime[index] > runtime[index - 1]
                for index in range(1, count)
            ))
        self.assertGreater(accepted, 100)

    def test_rasterize_heightfield_rounds_minima_down_and_ceil_covers_maxima(self):
        class RecordingTerrain:
            def __init__(self):
                self.queries = []

            def height(self, x, z):
                self.queries.append((x, z))
                return x + 2.0 * z

        terrain = RecordingTerrain()
        bounds = (0.1000000009, 0.490000009, -0.3000000041, 0.110000003)
        requested_cell = 0.0700000007
        grid = rasterize_heightfield(terrain, bounds, requested_cell)
        encoded_cell = float(np.float32(requested_cell))

        def expected_minimum(value):
            encoded = np.float32(value)
            if float(encoded) > value:
                encoded = np.nextafter(encoded, np.float32(-np.inf))
            return float(encoded)

        self.assertEqual(grid.origin_x, expected_minimum(bounds[0]))
        self.assertEqual(grid.origin_z, expected_minimum(bounds[2]))
        self.assertEqual(grid.cell_size, encoded_cell)
        self.assertLessEqual(grid.origin_x, bounds[0])
        self.assertLessEqual(grid.origin_z, bounds[2])
        self.assertGreaterEqual(grid.max_x, bounds[1])
        self.assertGreaterEqual(grid.max_z, bounds[3])
        self.assertEqual(
            grid.nx,
            int(np.ceil((bounds[1] - grid.origin_x) / encoded_cell)) + 1,
        )
        self.assertEqual(
            grid.nz,
            int(np.ceil((bounds[3] - grid.origin_z) / encoded_cell)) + 1,
        )
        self.assertEqual(terrain.queries[0], (grid.origin_x, grid.origin_z))
        self.assertEqual(terrain.queries[-1], (grid.max_x, grid.max_z))
        expected_heights = np.array([
            [grid.origin_x + ix * grid.cell_size
             + 2.0 * (grid.origin_z + iz * grid.cell_size)
             for ix in range(grid.nx)]
            for iz in range(grid.nz)
        ], dtype="<f4")
        np.testing.assert_array_equal(grid.heights, expected_heights)

    def test_rasterize_heightfield_rejects_invalid_or_impossible_sizes_early(self):
        class CountingTerrain:
            def __init__(self):
                self.calls = 0

            def height(self, x, z):
                self.calls += 1
                return 0.0

        invalid = (
            ((0.0, 0.0, 0.0, 1.0), 0.02),
            ((0.0, 1.0, 0.0, 0.0), 0.02),
            ((0.0, np.inf, 0.0, 1.0), 0.02),
            ((0.0, 1.0, 0.0, 1.0), 1e-100),
            ((0.0, 1.0, 0.0, 1.0), 1e100),
            ((0.0, float(np.iinfo(np.int32).max), 0.0, 1.0), 0.02),
            ((0.0, 50000.0, 0.0, 50000.0), 1.0),
        )
        for bounds, cell in invalid:
            terrain = CountingTerrain()
            with self.subTest(bounds=bounds, cell=cell):
                with self.assertRaises(ValueError):
                    rasterize_heightfield(terrain, bounds, cell)
                self.assertEqual(terrain.calls, 0)

    def test_rasterize_rejects_runtime_collapsed_nodes_before_queries(self):
        class CountingTerrain:
            def __init__(self):
                self.calls = 0

            def height(self, x, z):
                self.calls += 1
                return 0.0

        terrain = CountingTerrain()

        with self.assertRaisesRegex(
            ValueError, "X.*runtime-distinguishable"
        ):
            rasterize_heightfield(
                terrain,
                (1e8, 1e8 + 0.04, 0.0, 0.04),
                0.02,
            )

        self.assertEqual(terrain.calls, 0)

    def test_rasterize_rejects_subnormal_runtime_node_before_queries(self):
        class CountingTerrain:
            def __init__(self):
                self.calls = 0

            def height(self, x, z):
                self.calls += 1
                return 0.0

        minimum_normal = np.float32(np.finfo(np.float32).tiny)
        next_normal = np.nextafter(minimum_normal, np.float32(np.inf))
        maximum_x = float(np.float32(-minimum_normal + next_normal))
        terrain = CountingTerrain()

        with self.assertRaisesRegex(ValueError, "X.*normal-or-zero"):
            rasterize_heightfield(
                terrain,
                (float(-minimum_normal), maximum_x,
                 0.0, float(next_normal)),
                float(next_normal),
            )

        self.assertEqual(terrain.calls, 0)

    def test_export_heightfield_obj_serializes_before_opening_and_writes_once(self):
        class BrokenGrid(HeightGrid):
            def obj_bytes(self):
                raise RuntimeError("serialization failed")

        grid = HeightGrid(
            np.array([[0.0, 1.0], [2.0, 3.0]]), 0.0, 0.0, 0.5, 0.0)
        broken = BrokenGrid(
            np.array([[0.0, 1.0], [2.0, 3.0]]), 0.0, 0.0, 0.5, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "terrain.obj")
            with open(path, "wb") as stream:
                stream.write(b"sentinel")
            with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                export_heightfield_obj(broken, path)
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), b"sentinel")

            export_heightfield_obj(grid, path)
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), grid.obj_bytes())

            with self.assertRaises(TypeError):
                export_heightfield_obj(object(), path)
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), grid.obj_bytes())

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

    def test_vertical_triangle_query_interpolates_exact_top(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.5, 1.0],
        ])
        surface = VerticalTriangleSurface(
            vertices, np.array([[0, 1, 2]], np.int32), exterior_height=-2.0)
        self.assertAlmostEqual(surface.height(0.25, 0.25), 0.375, places=12)
        self.assertEqual(surface.height(0.75, 0.75), -2.0)

    def test_vertical_query_is_winding_independent_and_boundary_closed(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.5, 1.0],
        ])
        queries = (
            ((0.25, 0.25), 0.375),
            ((0.50, 0.00), 0.500),
            ((0.50, 0.50), 0.750),
            ((0.00, 1.00), 0.500),
        )
        for winding in ([0, 1, 2], [0, 2, 1]):
            surface = VerticalTriangleSurface(
                vertices, np.array([winding], np.int32),
                exterior_height=-2.0)
            for query, expected in queries:
                with self.subTest(winding=winding, query=query):
                    self.assertAlmostEqual(
                        surface.height(*query), expected, places=12)

    def test_vertical_query_uses_highest_overlapping_triangle(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 0, 1],
            [0, 0.4, 0], [1, 0.4, 0], [0, 0.4, 1],
        ], np.float64)
        surface = VerticalTriangleSurface(
            vertices, np.array([[0, 1, 2], [3, 4, 5]], np.int32))
        self.assertAlmostEqual(surface.height(0.2, 0.2), 0.4, places=12)

    def test_vertical_query_ignores_xz_degenerate_triangles(self):
        vertices = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 10.0, 0.0],
            [2.0, 3.0, 0.0],
        ])
        exterior_height = -2.0
        for winding in ([0, 1, 2], [0, 2, 1]):
            surface = VerticalTriangleSurface(
                vertices,
                np.array([winding], np.int32),
                exterior_height=exterior_height,
            )

            height = surface.height(1.0, 0.0)

            with self.subTest(winding=winding):
                self.assertTrue(np.isfinite(height))
                self.assertEqual(height, exterior_height)

    def test_exact_query_does_not_dilate_a_top_by_fourteen_centimetres(self):
        vertices = np.array([
            [0.0, 0.2, 0.0], [1.0, 0.2, 0.0],
            [1.0, 0.2, 1.0], [0.0, 0.2, 1.0],
        ])
        triangles = triangulate_faces(
            vertices, np.array([4], np.int32),
            np.array([0, 1, 2, 3], np.int32))
        surface = VerticalTriangleSurface(vertices, triangles)
        self.assertEqual(surface.height(-0.001, 0.5), 0.0)
        self.assertEqual(surface.height(1.001, 0.5), 0.0)
        self.assertAlmostEqual(surface.height(0.001, 0.5), 0.2, places=12)

    def test_polygon_triangulation_is_a_stable_first_vertex_fan(self):
        vertices = np.zeros((5, 3), np.float64)
        triangles = triangulate_faces(
            vertices, np.array([5], np.int32),
            np.array([4, 2, 0, 1, 3], np.int32))
        np.testing.assert_array_equal(
            triangles, [[4, 2, 0], [4, 0, 1], [4, 1, 3]])

    def test_mesh_topology_rejects_negative_and_past_end_indices(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        counts = np.array([3], np.int32)
        for bad_index in (-1, len(vertices)):
            indices = np.array([0, 1, bad_index], np.int32)
            for boundary, constructor in (
                ("triangulate", lambda: triangulate_faces(
                    vertices, counts, indices)),
                ("grail", lambda: GrailTerrain(
                    vertices, counts, indices)),
            ):
                with self.subTest(
                    bad_index=bad_index,
                    boundary=boundary,
                ):
                    with self.assertRaisesRegex(
                        ValueError, r"face indices must be in \[0, 3\)"
                    ):
                        constructor()
            with self.subTest(bad_index=bad_index, boundary="surface"):
                with self.assertRaisesRegex(
                    ValueError, "surface triangle index is outside"
                ):
                    VerticalTriangleSurface(
                        vertices,
                        np.array([[0, 1, bad_index]], np.int32),
                    )

    def test_mesh_topology_rejects_lossy_casts_at_public_boundaries(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        valid_counts = np.array([3], np.int32)
        valid_indices = np.array([0, 1, 2], np.int32)
        cases = (
            (
                "triangulate float counts",
                lambda: triangulate_faces(
                    vertices, np.array([3.0]), valid_indices),
                "face counts must use an integer dtype",
            ),
            (
                "triangulate float indices",
                lambda: triangulate_faces(
                    vertices, valid_counts, np.array([0.0, 1.0, 2.0])),
                "face indices must use an integer dtype",
            ),
            (
                "triangulate overflowing counts",
                lambda: triangulate_faces(
                    vertices, np.array([2**32 + 3], np.uint64),
                    valid_indices),
                "face counts must fit int32",
            ),
            (
                "triangulate overflowing indices",
                lambda: triangulate_faces(
                    vertices, valid_counts,
                    np.array([0, 1, 2**32 + 2], np.uint64)),
                "face indices must fit int32",
            ),
            (
                "grail float counts",
                lambda: GrailTerrain(
                    vertices, np.array([3.0]), valid_indices),
                "face counts must use an integer dtype",
            ),
            (
                "grail float indices",
                lambda: GrailTerrain(
                    vertices, valid_counts, np.array([0.0, 1.0, 2.0])),
                "face indices must use an integer dtype",
            ),
            (
                "grail overflowing counts",
                lambda: GrailTerrain(
                    vertices, np.array([2**32 + 3], np.uint64),
                    valid_indices),
                "face counts must fit int32",
            ),
            (
                "grail overflowing indices",
                lambda: GrailTerrain(
                    vertices, valid_counts,
                    np.array([0, 1, 2**32 + 2], np.uint64)),
                "face indices must fit int32",
            ),
        )
        for label, operation, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    operation()

    def test_vertical_surface_rejects_lossy_triangle_index_casts(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        cases = (
            (
                np.array([[0.0, 1.0, 2.0]]),
                "surface triangle indices must use an integer dtype",
            ),
            (
                np.array([[0, 1, 2**32 + 2]], np.uint64),
                "surface triangle indices must fit int32",
            ),
        )
        for triangles, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    VerticalTriangleSurface(vertices, triangles)

    def test_vertical_surface_copies_and_freezes_caller_arrays(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.5, 1.0],
        ])
        triangles = np.array([[0, 1, 2]], np.int32)
        surface = VerticalTriangleSurface(vertices, triangles)

        vertices[:] = 100.0
        triangles[:] = 0

        np.testing.assert_array_equal(surface.triangles, [[0, 1, 2]])
        np.testing.assert_allclose(surface.vertices, [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.5, 1.0],
        ])
        self.assertAlmostEqual(surface.height(0.25, 0.25), 0.375, places=12)
        self.assertFalse(surface.vertices.flags.writeable)
        self.assertFalse(surface.triangles.flags.writeable)

    def test_surface_signature_covers_every_query_semantic(self):
        semantics = surface_semantics()
        self.assertEqual(semantics, {
            "schema": "g1-terrain-surface/v1",
            "coordinate_signature":
                "holden-y-up-right-handed-forward-plus-z",
            "source_query": "vertical-triangle-top",
            "polygon_triangulation": "fan-from-first-index",
            "overlap_height_policy": "maximum-y",
            "projected_boundary_policy": "closed",
            "triangle_winding_policy": "orientation-independent",
            "degenerate_projected_triangle_policy": "ignore",
            "projected_area_measure": "absolute-two-times-area",
            "bbox_tolerance_m": 1e-12,
            "projected_area_epsilon_m2": 1e-12,
            "barycentric_tolerance": 1e-10,
            "heightfield_schema": "G1HF/v2",
            "heightfield_version": 2,
            "heightfield_interpolation": "fixed-diagonal-triangles",
            "heightfield_diagonal":
                "min-x-min-z_to_max-x-max-z",
            "heightfield_scalar_encoding":
                "ieee754-binary32-little-endian",
            "heightfield_domain_policy":
                "inclusive-authoritative-node-rectangle",
            "heightfield_grid_line_policy":
                "positive-index-cell-except-maximum-edge",
            "heightfield_diagonal_tie_policy":
                "tx-greater-or-equal-tz-uses-p00-p10-p11",
            "heightfield_exterior_normal": [0.0, 1.0, 0.0],
            "heightfield_source_node_encoding":
                "binary32-header-values-promoted-to-binary64-arithmetic",
            "heightfield_runtime_query_encoding":
                "normal-or-zero-binary32-canonicalized-positive-and-"
                "promoted-to-binary64",
            "heightfield_scalar_domain": "normal-or-zero-binary32",
            "heightfield_cell_domain": "positive-normal-binary32",
            "heightfield_runtime_node_domain":
                "normal-or-zero-binary32",
            "heightfield_runtime_query_domain":
                "normal-or-zero-binary32-coordinates",
            "heightfield_runtime_parity_domain":
                "normal-or-zero-binary32-coordinates",
            "heightfield_denormal_policy":
                "reject-nonzero-binary32-subnormals",
            "heightfield_evaluation_precision":
                "binary64-from-binary32-samples-and-promoted-node-weights",
            "heightfield_runtime_height_output":
                "finite-binary64-interpolation-rounded-to-binary32",
            "heightfield_normal_evaluation":
                "selected-triangle-binary64-gradient-scale-safe-"
                "unit-normalization",
            "heightfield_runtime_normal_output":
                "unit-normal-components-rounded-to-binary32",
            "heightfield_runtime_output_ftz_policy":
                "binary32-subnormals-and-signed-zero-canonicalized-to-"
                "positive-zero",
            "heightfield_zero_encoding": "canonical-positive-zero",
            "heightfield_runtime_node_distinguishability_policy":
                "normal-or-positive-zero-strictly-increasing-proven-by-"
                "endpoints-near-zero-candidates-max-binary32-spacing-and-"
                "aligned-equality",
            "heightfield_obj_coordinate_quantization":
                "binary32-round-of-promoted-origin-plus-index-times-cell",
            "heightfield_raster_bounds_policy":
                "float32-minimum-rounded-down-and-maximum-ceil-covered",
            "heightfield_obj_vertex_order": "z-major-x-minor",
            "heightfield_obj_face_order":
                "p00-p11-p10_then_p00-p01-p11",
            "heightfield_obj_float_format": ".9g-final-newline",
            "cell_size_m": 0.02,
            "exterior_height_m": 0.0,
        })
        self.assertRegex(surface_semantics_signature(), r"^[0-9a-f]{64}$")

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
        terrain = GrailTerrain(
            render_vertices=vertices,
            face_counts=counts,
            face_indices=indices,
        )

        first = terrain.xz_bounds()
        second = terrain.xz_bounds()
        footprint = terrain.footprint()

        self.assertEqual(first, second)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertEqual(first, (
            float(vertices[:, 0].min()), float(vertices[:, 0].max()),
            float(vertices[:, 2].min()), float(vertices[:, 2].max()),
        ))
        self.assertEqual(footprint["x"], (-0.5, 0.5))
        self.assertEqual(footprint["z"], (3.0, 4.0))
        self.assertEqual(footprint["height"], 0.3)
        self.assertAlmostEqual(terrain.height(0.0, 0.5), 0.1)
        self.assertAlmostEqual(terrain.height(0.0, 3.5), 0.3)

    def test_grail_terrain_copies_and_freezes_caller_mesh(self):
        vertices = np.array([
            [0.0, 0.2, 0.0],
            [1.0, 0.2, 0.0],
            [1.0, 0.2, 1.0],
            [0.0, 0.2, 1.0],
        ])
        counts = np.array([4], np.int32)
        indices = np.array([0, 1, 2, 3], np.int32)
        terrain = GrailTerrain(vertices, counts, indices)
        original_bounds = terrain.xz_bounds()
        original_footprint = terrain.footprint()
        original_height = terrain.height(0.25, 0.25)

        with tempfile.TemporaryDirectory() as tmp:
            before = os.path.join(tmp, "before.obj")
            after = os.path.join(tmp, "after.obj")
            terrain.export_obj(before)
            vertices[:] = 100.0
            counts[:] = 3
            indices[:] = [3, 2, 1, 0]
            terrain.export_obj(after)
            with open(before, "rb") as first, open(after, "rb") as second:
                self.assertEqual(first.read(), second.read())

        self.assertEqual(terrain.xz_bounds(), original_bounds)
        self.assertEqual(terrain.footprint(), original_footprint)
        self.assertEqual(terrain.height(0.25, 0.25), original_height)
        for array in (
            terrain.vertices,
            terrain.triangles,
            terrain._vertices,
            terrain._face_counts,
            terrain._face_indices,
        ):
            self.assertFalse(array.flags.writeable)

    def test_grail_terrain_rejects_empty_or_malformed_geometry(self):
        triangle = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        nonfinite_vertices = triangle.copy()
        nonfinite_vertices[0, 0] = np.nan
        cases = (
            (
                np.empty((0, 3)), np.empty(0, np.int32),
                np.empty(0, np.int32),
                "vertices must have non-empty shape",
            ),
            (
                triangle.ravel(), np.array([3], np.int32),
                np.array([0, 1, 2], np.int32),
                "vertices must have non-empty shape",
            ),
            (
                triangle, np.empty(0, np.int32), np.empty(0, np.int32),
                "faces must be non-empty",
            ),
            (
                nonfinite_vertices, np.array([3], np.int32),
                np.array([0, 1, 2], np.int32),
                "vertices.*finite",
            ),
        )
        for vertices, counts, indices, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    GrailTerrain(vertices, counts, indices)

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

    def test_selected_grail_scene_heightfield_covers_mesh_and_motion_lookahead(
            self):
        base = GRAIL_DEFAULT_BASE
        source = load_grail(os.path.join(GRAIL_ROBOT_DIR, base + ".pkl"))
        kinematics = G1Kinematics(builder.DEFAULTS["g1_xml"])
        clip, _, _ = convert_source_clip(
            source, kinematics, builder.OUTPUT_FPS)
        definition = grail_scene_definition(
            "grail-curb-default", base, clip, None)
        built = build_scene(definition)
        metadata = built.metadata

        magic, version, nx, nz, origin_x, origin_z, cell_size, exterior = \
            struct.unpack_from("<4sIII4f", built.terrain_bin)
        self.assertEqual((magic, version), (b"G1HF", 2))
        self.assertEqual(len(built.terrain_bin), 32 + nx * nz * 4)
        heights = np.frombuffer(
            built.terrain_bin, "<f4", nx * nz, 32).reshape(nz, nx)
        grid = HeightGrid(
            heights, origin_x, origin_z, cell_size, exterior)
        for key, value in grid.metadata().items():
            self.assertEqual(metadata["heightfield"][key], value)

        domain = (
            grid.origin_x, grid.max_x, grid.origin_z, grid.max_z,
        )
        bounds = metadata["bounds"]
        lookahead = (
            bounds["lookahead_min_xz"][0],
            bounds["lookahead_max_xz"][0],
            bounds["lookahead_min_xz"][1],
            bounds["lookahead_max_xz"][1],
        )
        self.assertLessEqual(domain[0], lookahead[0])
        self.assertGreaterEqual(domain[1], lookahead[1])
        self.assertLessEqual(domain[2], lookahead[2])
        self.assertGreaterEqual(domain[3], lookahead[3])

        mesh_bounds = definition.surface.xz_bounds()
        self.assertLessEqual(domain[0], mesh_bounds[0])
        self.assertGreaterEqual(domain[1], mesh_bounds[1])
        self.assertLessEqual(domain[2], mesh_bounds[2])
        self.assertGreaterEqual(domain[3], mesh_bounds[3])

        roots = clip.positions[:, 0][:, [0, 2]].astype(np.float64)
        headings = holden_quat.mul_vec(
            clip.rotations[:, 0].astype(np.float64),
            np.array([0.0, 0.0, 1.0]),
        )[:, [0, 2]]
        maximum_lookahead = max(metadata["terrain_feature_distances_m"])
        runtime_queries = np.concatenate(
            (roots, roots + maximum_lookahead * headings), axis=0)
        route_points = np.concatenate([
            np.asarray(route["waypoints_xz"], np.float64)
            for route in metadata["routes"]
        ])
        covered_queries = np.concatenate((runtime_queries, route_points))
        self.assertTrue(np.all(covered_queries[:, 0] >= lookahead[0]))
        self.assertTrue(np.all(covered_queries[:, 0] <= lookahead[1]))
        self.assertTrue(np.all(covered_queries[:, 1] >= lookahead[2]))
        self.assertTrue(np.all(covered_queries[:, 1] <= lookahead[3]))


if __name__ == "__main__":
    unittest.main()
