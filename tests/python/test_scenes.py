import copy
import hashlib
import json
import os
import struct
import tempfile
import unittest

import numpy as np

from resources.g1_terrain_builder.artifacts import read_walkability
from resources.g1_terrain_builder.scenes import (
    REQUIRED_SCENE_IDS,
    SCENE_CELL_SIZE,
    WALKABILITY_CLASSIFICATION_HALO,
    BuiltScene,
    SceneDefinition,
    SceneRoute,
    build_scene,
    build_scene_pack,
    canonical_json_bytes,
    sha256_hex,
    _region_cell_indices,
    _route_samples,
    _route_cell_covers,
    _validate_region_classes,
    _validate_route_classes,
    _walkability_at,
)
from resources.g1_terrain_builder.terrain import FlatTerrain, HeightGrid, \
    surface_semantics_signature


def flat_definition(scene_id="grail-curb-default"):
    return SceneDefinition(
        scene_id=scene_id,
        label=scene_id.replace("-", " ").title(),
        provenance={
            "kind": "procedural", "source_ids": [],
            "parameters": {"fixture": True},
        },
        surface=FlatTerrain(),
        heightfield_bounds_xz=(-1.0, 1.0, -1.0, 3.0),
        playable_bounds_xz=(-0.5, 0.5, 0.0, 2.0),
        lookahead_bounds_xz=(-1.0, 1.0, -1.0, 3.0),
        spawn_position=(0.0, 0.0, 0.0),
        spawn_yaw_radians=0.0,
        regions={
            "certified": ({"id": "route", "bounds_xz": [-0.5, 0.5, 0, 2]},),
            "stress": (), "blocked": (),
        },
        routes=(SceneRoute(
            route_id="forward",
            waypoints_xz=(
                (0.0, 0.0), (0.0, 0.5),
                (0.0, 1.0), (0.0, 2.0)),
            expected_outcome="traverse",
            walkability_class=1,
            landing_hold_seconds=2.0,
        ),),
        # Query x=-0.5 selects node 25, whose promoted classifier coordinate
        # is -0.5000000111758709. Apply the shared classification-only halo on
        # all sides without moving the published playable/region boundary.
        walkability=lambda x, z: (
            1 if -0.5 - WALKABILITY_CLASSIFICATION_HALO <= x
            <= 0.5 + WALKABILITY_CLASSIFICATION_HALO
            and -WALKABILITY_CLASSIFICATION_HALO <= z
            <= 2.0 + WALKABILITY_CLASSIFICATION_HALO else 0),
    )


class SceneSchemaTests(unittest.TestCase):
    def test_canonical_json_hash_is_over_exact_written_bytes(self):
        value = {"z": 1, "a": [2, 3], "zero": -0.0}
        payload = canonical_json_bytes(value)
        self.assertEqual(
            payload,
            b'{\n  "a": [\n    2,\n    3\n  ],\n'
            b'  "z": 1,\n  "zero": 0.0\n}\n')
        self.assertEqual(
            sha256_hex(payload),
            hashlib.sha256(canonical_json_bytes(value)).hexdigest())
        self.assertEqual(struct.pack("<d", json.loads(payload)["zero"]),
                         struct.pack("<d", 0.0))

    def test_built_scene_has_locked_keys_paths_schemas_and_hashes(self):
        scene = build_scene(flat_definition())
        metadata = scene.metadata
        self.assertEqual(set(metadata), {
            "schema", "id", "label", "provenance",
            "coordinate_signature", "surface_signature",
            "terrain_feature_distances_m", "heightfield", "mesh",
            "walkability", "bounds", "spawn", "regions", "routes",
        })
        self.assertEqual(metadata["schema"], "g1-terrain-scene/v1")
        self.assertEqual(set(metadata["provenance"]), {
            "kind", "source_ids", "parameters"})
        self.assertEqual(
            metadata["coordinate_signature"],
            "holden-y-up-right-handed-forward-plus-z")
        self.assertEqual(
            metadata["surface_signature"], surface_semantics_signature())
        self.assertEqual(
            metadata["terrain_feature_distances_m"],
            [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(set(metadata["heightfield"]), {
            "path", "schema", "version", "nx", "nz",
            "origin_x", "origin_z", "cell_size_m", "exterior_height_m",
            "interpolation", "diagonal", "sha256",
        })
        self.assertEqual(metadata["heightfield"]["path"], "terrain.bin")
        self.assertEqual(metadata["heightfield"]["schema"], "G1HF/v2")
        self.assertEqual(metadata["heightfield"]["version"], 2)
        self.assertEqual(
            metadata["heightfield"]["cell_size_m"],
            struct.unpack("<f", struct.pack("<f", SCENE_CELL_SIZE))[0])
        self.assertEqual(
            metadata["heightfield"]["interpolation"],
            "fixed-diagonal-triangles")
        self.assertEqual(
            metadata["heightfield"]["diagonal"],
            "min-x-min-z_to_max-x-max-z")
        self.assertEqual(metadata["mesh"], {
            "path": "terrain.obj", "schema": "obj/v1",
            "sha256": hashlib.sha256(scene.terrain_obj).hexdigest(),
        })
        self.assertEqual(set(metadata["walkability"]), {
            "path", "schema", "version", "nx", "nz", "classes", "sha256",
        })
        self.assertEqual(metadata["walkability"]["path"], "walkability.bin")
        self.assertEqual(metadata["walkability"]["schema"], "G1WM/v1")
        self.assertEqual(metadata["walkability"]["version"], 1)
        self.assertEqual(metadata["walkability"]["classes"], {
            "blocked": 0, "certified": 1, "stress": 2,
        })
        self.assertEqual(
            metadata["heightfield"]["sha256"],
            hashlib.sha256(scene.terrain_bin).hexdigest())
        self.assertEqual(
            metadata["walkability"]["sha256"],
            hashlib.sha256(scene.walkability_bin).hexdigest())
        magic, version, nx, nz, ox, oz, cell, _ = struct.unpack_from(
            "<4sIII4f", scene.terrain_bin)
        self.assertEqual((magic, version), (b"G1HF", 2))
        self.assertGreaterEqual(nx, 2)
        self.assertGreaterEqual(nz, 2)
        self.assertEqual(
            (metadata["heightfield"]["nx"], metadata["heightfield"]["nz"]),
            (nx, nz))
        self.assertEqual(
            (metadata["walkability"]["nx"], metadata["walkability"]["nz"]),
            (nx, nz))
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "walkability.bin")
            with open(path, "wb") as stream:
                stream.write(scene.walkability_bin)
            decoded = read_walkability(path)
        self.assertEqual(decoded.shape, (nz, nx))
        expected_classes = np.empty((nz, nx), np.uint8)
        for iz in range(nz):
            z = float(oz) + iz * float(cell)
            for ix in range(nx):
                x = float(ox) + ix * float(cell)
                expected_classes[iz, ix] = 1 if (
                    -0.5 - WALKABILITY_CLASSIFICATION_HALO <= x
                    <= 0.5 + WALKABILITY_CLASSIFICATION_HALO
                    and -WALKABILITY_CLASSIFICATION_HALO <= z
                    <= 2.0 + WALKABILITY_CLASSIFICATION_HALO) else 0
        np.testing.assert_array_equal(decoded, expected_classes)
        alias_x = float(ox) + 25 * float(cell)
        self.assertEqual(alias_x, -0.5000000111758709)
        self.assertLess(alias_x, -0.5)
        self.assertEqual(int(decoded[50, 25]), 1)
        heightfield_min = [float(ox), 0.0, float(oz)]
        heightfield_max = [
            float(ox) + (nx - 1) * float(cell), 0.0,
            float(oz) + (nz - 1) * float(cell),
        ]
        vertex_lines = scene.terrain_obj.decode("ascii").splitlines()[:nx * nz]
        first = vertex_lines[0].split()
        last = vertex_lines[-1].split()
        mesh_min = [
            float(np.float32(first[1])), 0.0,
            float(np.float32(first[3])),
        ]
        mesh_max = [
            float(np.float32(last[1])), 0.0,
            float(np.float32(last[3])),
        ]
        self.assertEqual(metadata["bounds"]["heightfield_min_xyz"],
                         heightfield_min)
        self.assertEqual(metadata["bounds"]["heightfield_max_xyz"],
                         heightfield_max)
        self.assertEqual(metadata["bounds"]["mesh_min_xyz"], mesh_min)
        self.assertEqual(metadata["bounds"]["mesh_max_xyz"], mesh_max)
        self.assertEqual(set(metadata["bounds"]), {
            "mesh_min_xyz", "mesh_max_xyz",
            "heightfield_min_xyz", "heightfield_max_xyz",
            "playable_min_xz", "playable_max_xz",
            "lookahead_min_xz", "lookahead_max_xz",
        })
        self.assertNotEqual(mesh_max, heightfield_max)
        self.assertEqual(set(metadata["spawn"]), {"position", "yaw_radians"})
        self.assertEqual(
            set(metadata["regions"]), {"certified", "stress", "blocked"})
        for entries in metadata["regions"].values():
            for entry in entries:
                self.assertEqual(set(entry), {"id", "bounds_xz"})
        for route in metadata["routes"]:
            self.assertEqual(set(route), {
                "id", "waypoints_xz", "expected_outcome",
                "walkability_class", "landing_hold_seconds",
            })
        self.assertEqual(metadata["routes"], [{
            "id": "forward",
            "waypoints_xz": [
                [0.0, 0.0], [0.0, 0.5],
                [0.0, 1.0], [0.0, 2.0]],
            "expected_outcome": "traverse",
            "walkability_class": 1,
            "landing_hold_seconds": 2.0,
        }])

    def test_route_outcomes_and_classes_are_locked(self):
        valid = (
            ("traverse", 1),
            ("safe-stop", 0),
            ("traverse-or-safe-stop", 2),
        )
        for outcome, walkability_class in valid:
            SceneRoute(
                "route", ((0, 0), (0, 1)), outcome,
                walkability_class, 0.0).validate()
        with self.assertRaisesRegex(ValueError, "outcome.*class"):
            SceneRoute(
                "bad", ((0, 0), (0, 1)), "traverse", 2, 0.0).validate()
        with self.assertRaisesRegex(ValueError, "hold.*waypoint 2"):
            SceneRoute(
                "bad-hold", ((0, 0), (0, 1), (0, 2)),
                "traverse", 1, 2.0).validate()

        stop_grid = HeightGrid(
            heights=np.zeros((2, 4), np.float32),
            origin_x=0.0, origin_z=0.0, cell_size=1.0,
            exterior_height=0.0,
        )
        _validate_route_classes(({
            "waypoints_xz": [[0.0, 0.0], [3.0, 0.0]],
            "walkability_class": 0,
        },), stop_grid,
            np.array([[1, 1, 0, 0], [1, 1, 0, 0]], np.uint8),
            (0.0, 3.0, 0.0, 1.0))

        first_cover_grid = HeightGrid(
            heights=np.zeros((2, 2), np.float32),
            origin_x=0.0, origin_z=0.0, cell_size=1.0,
            exterior_height=0.0,
        )
        with self.assertRaisesRegex(ValueError, "safe-stop"):
            _validate_route_classes(({
                "waypoints_xz": [[0.0, 0.0], [1.0, 0.0]],
                "walkability_class": 0,
            },), first_cover_grid,
                np.array([[1, 0], [1, 0]], np.uint8),
                (0.0, 1.0, 0.0, 1.0))

        reentry_grid = HeightGrid(
            heights=np.zeros((2, 4), np.float32),
            origin_x=0.0, origin_z=0.0, cell_size=1.0,
            exterior_height=0.0,
        )
        with self.assertRaisesRegex(ValueError, "safe-stop"):
            _validate_route_classes(({
                "waypoints_xz": [[0.0, 0.0], [3.0, 0.0]],
                "walkability_class": 0,
            },), reentry_grid,
                np.array([[1, 0, 1, 0], [1, 0, 1, 0]], np.uint8),
                (0.0, 3.0, 0.0, 1.0))

    def test_runtime_json_scalars_match_float32_and_positive_zero(self):
        definition = flat_definition()
        definition.spawn_position = (-0.0, 0.0, -0.0)
        definition.routes = (SceneRoute(
            "forward", ((-0.0, -0.0), (0.0, 0.1), (0.0, 1.0)),
            "traverse", 1, 0.0),)
        scene = build_scene(definition)
        metadata = scene.metadata
        expected = struct.unpack("<f", struct.pack("<f", 0.1))[0]
        self.assertEqual(metadata["routes"][0]["waypoints_xz"][1][1], expected)
        for value in (
            metadata["spawn"]["position"][0],
            metadata["spawn"]["position"][2],
            metadata["routes"][0]["waypoints_xz"][0][0],
            metadata["routes"][0]["waypoints_xz"][0][1],
        ):
            self.assertEqual(struct.pack("<f", value), struct.pack("<I", 0))

    def test_walkability_lookup_locks_floorf_half_cell_nextafter_behavior(self):
        grid = HeightGrid(
            heights=np.zeros((2, 3), np.float32),
            origin_x=0.0, origin_z=0.0, cell_size=1.0,
            exterior_height=0.0,
        )
        classes = np.array([[0, 1, 2], [0, 1, 2]], np.uint8)
        half = np.float32(0.5)
        one_below = np.nextafter(
            half, np.float32(-np.inf), dtype=np.float32)
        two_below = np.nextafter(
            one_below, np.float32(-np.inf), dtype=np.float32)
        one_above = np.nextafter(
            half, np.float32(np.inf), dtype=np.float32)
        # floorf(f32(x + 0.5f)) rounds one ULP below the mathematical tie
        # back to 1.0f; two ULPs below remains below the boundary.
        self.assertEqual(_walkability_at(grid, classes, two_below, 0.0), 0)
        self.assertEqual(_walkability_at(grid, classes, one_below, 0.0), 1)
        self.assertEqual(_walkability_at(grid, classes, half, 0.0), 1)
        self.assertEqual(_walkability_at(grid, classes, one_above, 0.0), 1)
        samples = _route_samples(
            ((0.0, 0.0), (float(one_above), 0.0)), np.float32(0.25))
        one_third = np.float32(np.float32(1.0) / np.float32(3.0))
        two_thirds = np.float32(np.float32(2.0) / np.float32(3.0))
        self.assertEqual(
            [struct.pack("<f", point[0]) for point in samples],
            [struct.pack("<f", value) for value in (
                0.0, np.float32(one_third * one_above),
                np.float32(two_thirds * one_above), one_above,
            )])

    def test_route_interpolation_uses_separate_mul_add_on_both_axes(self):
        def from_bits(bits):
            return struct.unpack("<f", struct.pack("<I", bits))[0]

        def bits(value):
            return struct.unpack("<I", struct.pack("<f", value))[0]

        start = from_bits(0x3eb95abd)
        stop = from_bits(0xbfd6c975)
        delta = np.float32(np.float32(stop) - np.float32(start))
        self.assertEqual(bits(delta), 0xc0029012)
        samples = _route_samples(
            ((start, start), (stop, stop)), np.float32(1.0))
        # The exact f32 squared-distance/sqrt/divide path yields count=3.
        self.assertEqual(len(samples), 4)
        self.assertEqual((bits(samples[0][0]), bits(samples[-1][0])),
                         (0x3eb95abd, 0xbfd6c975))
        for axis in (0, 1):
            self.assertEqual(bits(samples[1][axis]), 0xbea2d01f)
        alpha = np.float32(np.float32(1.0) / np.float32(3.0))
        fused_once = np.float32(
            float(start) + float(alpha) * float(delta))
        self.assertEqual(bits(fused_once), 0xbea2d01e)

    def test_region_supercover_rejects_offset_boundary_alias(self):
        grid = HeightGrid(
            heights=np.zeros((2, 3), np.float32),
            origin_x=0.0, origin_z=0.0, cell_size=1.0,
            exterior_height=0.0,
        )
        classes = np.array([[1, 0, 0], [1, 0, 0]], np.uint8)
        bounds = (0.0, 0.51, 0.0, 1.0)
        # The old node-center oracle saw only x=0 and incorrectly passed.
        old_observed = [
            int(classes[iz, ix])
            for iz in range(grid.nz)
            for ix in range(grid.nx)
            if bounds[0] <= grid.origin_x + ix * grid.cell_size <= bounds[1]
            and bounds[2] <= grid.origin_z + iz * grid.cell_size <= bounds[3]
        ]
        self.assertEqual(set(old_observed), {1})
        self.assertEqual(set(_region_cell_indices(grid, bounds)), {
            (0, 0), (0, 1), (1, 0), (1, 1),
        })
        self.assertEqual(
            set(_region_cell_indices(grid, (0.0, 2.0, 0.0, 1.0))),
            {(iz, ix) for iz in range(2) for ix in range(3)})
        regions = {
            "certified": [{"id": "offset", "bounds_xz": list(bounds)}],
            "stress": [], "blocked": [],
        }
        with self.assertRaisesRegex(ValueError, "region disagrees with G1WM"):
            _validate_region_classes(regions, grid, classes)

    def test_route_supercover_rejects_diagonal_corner_graze(self):
        grid = HeightGrid(
            heights=np.zeros((3, 3), np.float32),
            origin_x=0.0, origin_z=0.0, cell_size=1.0,
            exterior_height=0.0,
        )
        classes = np.ones((3, 3), np.uint8)
        classes[0, 1] = 0
        points = ((0.25, 0.25), (0.75, 0.75))
        samples = _route_samples(points, np.float32(0.5))
        self.assertEqual(
            [_walkability_at(grid, classes, *point) for point in samples],
            [1, 1, 1])
        covers = _route_cell_covers(grid, points)
        self.assertIn(
            {(0, 0), (0, 1), (1, 0), (1, 1)},
            [set(cover) for cover in covers])
        routes = [{
            "waypoints_xz": [list(point) for point in points],
            "walkability_class": 1,
        }]
        with self.assertRaisesRegex(
                ValueError, "expected route enters wrong walkability class"):
            _validate_route_classes(
                routes, grid, classes, (0.0, 2.0, 0.0, 2.0))

    def test_scene_pack_requires_exact_order_and_stable_index_fields(self):
        definitions = [flat_definition(scene_id) for scene_id in REQUIRED_SCENE_IDS]
        pack = build_scene_pack(definitions)
        descriptors = [{
            "id": scene.scene_id,
            "path": f"scenes/{scene.scene_id}/scene.json",
            "sha256": hashlib.sha256(scene.scene_json).hexdigest(),
        } for scene in pack.scenes]
        self.assertEqual(set(pack.index), {
            "schema", "default_scene_id", "scene_ids", "scenes",
            "coordinate_signature", "surface_signature",
        })
        for descriptor in pack.index["scenes"]:
            self.assertEqual(set(descriptor), {"id", "path", "sha256"})
        self.assertEqual(pack.index, {
            "schema": "g1-terrain-scene-index/v1",
            "default_scene_id": "grail-curb-default",
            "scene_ids": list(REQUIRED_SCENE_IDS),
            "scenes": descriptors,
            "coordinate_signature":
                "holden-y-up-right-handed-forward-plus-z",
            "surface_signature": surface_semantics_signature(),
        })
        self.assertEqual(pack.index_json, canonical_json_bytes(pack.index))
        with self.assertRaisesRegex(ValueError, "required scene order"):
            build_scene_pack(list(reversed(definitions)))

    def test_pack_materializes_generator_once_and_caches_deep_normalized_bytes(self):
        class MutatingSurface:
            def __init__(self, target):
                self.target = target

            def height(self, x, z):
                self.target.scene_id = "../callback-escape"
                self.target.label = False
                self.target.surface = object()
                self.target.walkability = lambda query_x, query_z: 0
                return 0.0

        definition = flat_definition()
        expected_label = definition.label
        definition.surface = MutatingSurface(definition)
        built = build_scene(definition)
        self.assertEqual(built.scene_id, "grail-curb-default")
        self.assertEqual(
            (built.metadata["id"], built.metadata["label"]),
            ("grail-curb-default", expected_label))

        yielded = []
        source_definitions = [
            flat_definition(scene_id) for scene_id in REQUIRED_SCENE_IDS]
        expected_second_label = source_definitions[1].label
        source_definitions[0].surface = MutatingSurface(source_definitions[1])

        def definitions():
            for definition in source_definitions:
                yielded.append(definition.scene_id)
                yield definition

        pack = build_scene_pack(definitions())
        self.assertEqual(yielded, list(REQUIRED_SCENE_IDS))
        self.assertEqual(
            tuple(scene.scene_id for scene in pack.scenes),
            REQUIRED_SCENE_IDS)
        self.assertEqual(
            (pack.scenes[1].metadata["id"], pack.scenes[1].metadata["label"]),
            (REQUIRED_SCENE_IDS[1], expected_second_label))
        first_scene_json = pack.scenes[0].scene_json
        first_index_json = pack.index_json
        first_assets = (
            pack.scenes[0].terrain_bin, pack.scenes[0].terrain_obj,
            pack.scenes[0].walkability_bin,
        )
        source_definitions[0].provenance["parameters"]["fixture"] = False
        source_definitions[0].regions["certified"][0]["bounds_xz"][0] = -0.25
        leaked_metadata = pack.scenes[0].metadata
        leaked_metadata["label"] = "mutated caller copy"
        leaked_index = pack.index
        leaked_index["scene_ids"].reverse()
        self.assertEqual(pack.scenes[0].scene_json, first_scene_json)
        self.assertEqual(pack.index_json, first_index_json)
        self.assertEqual((
            pack.scenes[0].terrain_bin, pack.scenes[0].terrain_obj,
            pack.scenes[0].walkability_bin,
        ), first_assets)
        self.assertEqual(
            pack.index["scenes"][0]["sha256"],
            hashlib.sha256(first_scene_json).hexdigest())
        self.assertNotEqual(
            pack.index["scenes"][0]["sha256"],
            hashlib.sha256(first_scene_json + b"mutated").hexdigest())
        rebuilt = build_scene_pack(
            flat_definition(scene_id) for scene_id in REQUIRED_SCENE_IDS)
        self.assertEqual(rebuilt.index_json, pack.index_json)
        self.assertEqual(
            tuple(scene.scene_json for scene in rebuilt.scenes),
            tuple(scene.scene_json for scene in pack.scenes))
        self.assertEqual(
            tuple((scene.terrain_bin, scene.terrain_obj, scene.walkability_bin)
                  for scene in rebuilt.scenes),
            tuple((scene.terrain_bin, scene.terrain_obj, scene.walkability_bin)
                  for scene in pack.scenes))

    def test_scene_builder_rejects_strict_bounds_regions_routes_and_classes(self):
        cases = []

        definition = flat_definition()
        definition.spawn_position = (True, 0.0, 0.0)
        cases.append((definition, "spawn.*real scalar"))

        definition = flat_definition()
        definition.playable_bounds_xz = (-0.5, -0.5, 0.0, 2.0)
        cases.append((definition, "playable.*strict"))

        definition = flat_definition()
        definition.lookahead_bounds_xz = (-2.0, 1.0, -1.0, 3.0)
        cases.append((definition, "lookahead.*heightfield"))

        definition = flat_definition()
        definition.scene_id = 7
        cases.append((definition, "scene ID"))

        definition = flat_definition()
        definition.label = False
        cases.append((definition, "scene label"))

        definition = flat_definition()
        definition.regions["stress"] = ({
            "id": "route", "bounds_xz": [-0.2, 0.2, 0.2, 0.4],
        },)
        cases.append((definition, "duplicate region ID"))

        definition = flat_definition()
        definition.routes = (SceneRoute(
            "forward", ((0.0, 0.25), (0.0, 1.0)),
            "traverse", 1, 0.0),)
        cases.append((definition, "route.*start.*spawn"))

        definition = flat_definition()
        definition.routes = (SceneRoute(
            "forward", ((0.0, 0.0), (0.0, 4.0)),
            "traverse", 1, 0.0),)
        cases.append((definition, "route leaves lookahead"))

        definition = flat_definition()
        definition.routes = (SceneRoute(
            "forward", ((0.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
            "traverse", 1, 0.0),)
        cases.append((definition, "nonzero segment"))

        definition = flat_definition()
        definition.walkability = lambda x, z: 1 if z < 0.5 else 0
        definition.regions["certified"] = ({
            "id": "approach", "bounds_xz": [-0.5, 0.5, 0.0, 0.4],
        },)
        cases.append((definition, "wrong walkability class"))

        definition = flat_definition()
        definition.walkability = lambda x, z: True
        cases.append((definition, "walkability.*integer class"))

        for bad_value in ("0.25", 0.25 + 0.0j):
            definition = flat_definition()
            definition.routes = (SceneRoute(
                "forward", ((0.0, 0.0), (0.0, bad_value)),
                "traverse", 1, 0.0),)
            cases.append((definition, "real scalar"))

        definition = flat_definition()
        definition.routes = (SceneRoute(
            "forward", ((0.0, 0.0), (0.0, 1.0)),
            "traverse", True, 0.0),)
        cases.append((definition, "walkability class.*not bool"))

        for definition, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    build_scene(copy.deepcopy(definition))


if __name__ == "__main__":
    unittest.main()
