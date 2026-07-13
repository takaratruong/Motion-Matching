import copy
import glob
import hashlib
import json
import os
import struct
import tempfile
import unittest

import numpy as np

from resources.g1_terrain_builder.artifacts import read_walkability
from resources.g1_terrain_builder.scenes import (
    GRAIL_DEFAULT_BASE,
    REQUIRED_SCENE_IDS,
    SCENE_CELL_SIZE,
    WALKABILITY_CLASSIFICATION_HALO,
    BuiltScene,
    SceneDefinition,
    SceneRoute,
    build_scene,
    build_scene_pack,
    canonical_json_bytes,
    grail_scene_definition,
    grail_scene_definitions,
    all_scene_definitions,
    procedural_scene_definitions,
    select_grail_scene_bases,
    sha256_hex,
    _region_cell_indices,
    _root_route_and_yaw,
    _route_samples,
    _route_cell_covers,
    _validate_region_classes,
    _validate_route_classes,
    _walkability_at,
)
from resources.g1_terrain_builder.schema import HoldenClip
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    GrailTerrain,
    HeightGrid,
    surface_semantics_signature,
)


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


def decoded_scene_grid_and_walkability(scene):
    magic, version, nx, nz, ox, oz, cell, exterior = struct.unpack_from(
        "<4sIII4f", scene.terrain_bin)
    header = (magic, version, nx, nz)
    if header[:2] != (b"G1HF", 2):
        raise AssertionError(f"bad test G1HF header {header}")
    heights = np.frombuffer(
        scene.terrain_bin, "<f4", nx * nz, 32).reshape(nz, nx).copy()
    grid = HeightGrid(
        heights=heights, origin_x=float(ox), origin_z=float(oz),
        cell_size=float(cell), exterior_height=float(exterior),
    )
    wm_magic, wm_version, wm_nx, wm_nz = struct.unpack_from(
        "<4sIII", scene.walkability_bin)
    if (wm_magic, wm_version, wm_nx, wm_nz) \
            != (b"G1WM", 1, nx, nz):
        raise AssertionError("test G1WM header differs from G1HF")
    if len(scene.walkability_bin) != 16 + nx * nz:
        raise AssertionError("test G1WM byte size changed")
    walkability = np.frombuffer(
        scene.walkability_bin, np.uint8, nx * nz, 16,
    ).reshape(nz, nx).copy()
    return grid, walkability


def decoded_footprint_classes(
        grid, walkability, x, z, radius=0.20):
    # Reconstruct promoted binary32 source nodes from the decoded G1HF header,
    # then inspect exactly the circular node footprint consumed by runtime.
    origin_x = np.float32(grid.origin_x)
    origin_z = np.float32(grid.origin_z)
    cell = np.float32(grid.cell_size)
    xs = np.asarray([
        np.float32(origin_x + np.float32(ix) * cell)
        for ix in range(grid.nx)
    ], dtype=np.float32)
    zs = np.asarray([
        np.float32(origin_z + np.float32(iz) * cell)
        for iz in range(grid.nz)
    ], dtype=np.float32)
    dx = np.float32(xs[np.newaxis, :] - np.float32(x))
    dz = np.float32(zs[:, np.newaxis] - np.float32(z))
    radius_squared = np.float32(np.float32(radius) * np.float32(radius))
    # Match the C++ runtime's squared-radius boundary allowance exactly.
    inside = np.float32(dx * dx + dz * dz) <= np.float32(
        radius_squared + np.float32(1e-8))
    if not np.any(inside):
        raise AssertionError("decoded footprint did not contain a grid node")
    return tuple(int(value) for value in walkability[inside])


def assert_native_json_value(test, value, label="provenance"):
    if type(value) is dict:
        test.assertTrue(all(type(key) is str for key in value), label)
        for key, child in value.items():
            assert_native_json_value(test, child, f"{label}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            assert_native_json_value(test, child, f"{label}[{index}]")
        return
    test.assertIn(type(value), (str, int, float, bool, type(None)), label)


GRAIL_ROBOT_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/robot"
LOCKED_GRAIL_BASES = {
    "grail-curb-default": GRAIL_DEFAULT_BASE,
    "grail-curb-low": "terrain_curbs__curb_186__004",
    "grail-curb-medium": "terrain_curbs__curb_022__001",
    "grail-curb-high": "terrain_curbs__curb_165__006",
}
LOCKED_GRAIL_HEIGHTS = {
    GRAIL_DEFAULT_BASE: 0.2921024334377573,
    "terrain_curbs__curb_186__004": 0.12238701526200782,
    "terrain_curbs__curb_022__001": 0.24007104328948528,
    "terrain_curbs__curb_165__006": 0.3599740964554129,
}


def fake_grail_clip(base):
    clip = HoldenClip.empty(frames=5, bones=31)
    clip.name = base + "-clip"
    clip.terrain_id = base
    clip.positions[:, 0, 0] = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    clip.positions[:, 0, 2] = np.array([0.0, 0.4, 0.8, 1.2, 1.6])
    return clip


class GrailSceneTests(unittest.TestCase):
    def test_nearest_height_selection_uses_lexical_tie_break(self):
        measured = {
            GRAIL_DEFAULT_BASE: 0.29,
            "a-low-tie": 0.13,
            "z-low-tie": 0.13,
            "medium": 0.241,
            "high": 0.358,
        }
        selected = select_grail_scene_bases(measured)
        self.assertEqual(selected, {
            "grail-curb-default": GRAIL_DEFAULT_BASE,
            "grail-curb-low": "a-low-tie",
            "grail-curb-medium": "medium",
            "grail-curb-high": "high",
        })

    def test_selection_rejects_nonexact_or_empty_base_names(self):
        class BaseName(str):
            pass

        for base in (BaseName(GRAIL_DEFAULT_BASE), "", 7):
            with self.subTest(base=repr(base)):
                with self.assertRaisesRegex(
                        ValueError,
                        "^GRAIL base names must be non-empty strings$"):
                    select_grail_scene_bases({base: 0.29})

    def test_selection_rejects_bool_and_non_scalar_heights(self):
        for invalid in (True, "0.12", [0.12], np.array(0.12)):
            measured = {
                GRAIL_DEFAULT_BASE: 0.29,
                "invalid": invalid,
            }
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(TypeError, "real scalar"):
                    select_grail_scene_bases(measured)

    def test_grail_scene_uses_matching_converted_root_path_and_facing(self):
        base = GRAIL_DEFAULT_BASE
        clip = fake_grail_clip(base)
        scene = grail_scene_definition(
            "grail-curb-default", base, clip, None)
        self.assertEqual(scene.spawn_position, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(scene.spawn_yaw_radians, 0.0)
        self.assertEqual(scene.routes[0].route_id, "curb-forward")
        expected_first = tuple(
            float(value) for value in clip.positions[0, 0, (0, 2)])
        expected_last = tuple(
            float(value) for value in clip.positions[-1, 0, (0, 2)])
        self.assertEqual(scene.routes[0].waypoints_xz[0], expected_first)
        self.assertEqual(
            scene.routes[0].waypoints_xz[-1], expected_last)
        self.assertEqual(scene.provenance["source_ids"], [base, clip.name])
        wrong = fake_grail_clip("different-base")
        with self.assertRaisesRegex(ValueError, "matching converted clip"):
            grail_scene_definition(
                "grail-curb-default", base, wrong, None)

    def test_root_route_rejects_bad_rotation_shape_and_quaternion_norm(self):
        bad_shape = fake_grail_clip(GRAIL_DEFAULT_BASE)
        bad_shape.rotations = np.zeros(
            bad_shape.positions.shape[:2], np.float32)
        with self.subTest(case="rotation-shape"):
            with self.assertRaisesRegex(ValueError, "root rotations"):
                _root_route_and_yaw(bad_shape)

        zero = fake_grail_clip(GRAIL_DEFAULT_BASE)
        zero.rotations[0, 0] = 0.0
        with self.subTest(case="zero-quaternion"):
            with self.assertRaisesRegex(ValueError, "root quaternion"):
                _root_route_and_yaw(zero)

        nonunit = fake_grail_clip(GRAIL_DEFAULT_BASE)
        nonunit.rotations[0, 0] = np.array([2.0, 0.0, 0.0, 0.0])
        with self.subTest(case="materially-nonunit-quaternion"):
            with self.assertRaisesRegex(ValueError, "root quaternion"):
                _root_route_and_yaw(nonunit)

        near_boundary = fake_grail_clip(GRAIL_DEFAULT_BASE)
        near_boundary.rotations[0, 0] *= np.float32(1.0002)
        with self.subTest(case="near-boundary-nonunit-quaternion"):
            with self.assertRaisesRegex(ValueError, "root quaternion"):
                _root_route_and_yaw(near_boundary)

        roundoff = fake_grail_clip(GRAIL_DEFAULT_BASE)
        yaw = 0.6
        unit = np.array([
            np.cos(0.5 * yaw), 0.0, np.sin(0.5 * yaw), 0.0,
        ], np.float32)
        roundoff.rotations[0, 0] = unit * np.float32(1.00005)
        _, _, _, observed_yaw = _root_route_and_yaw(roundoff)
        self.assertAlmostEqual(observed_yaw, yaw, places=5)

    def test_real_corpus_selection_is_stable(self):
        paths = sorted(glob.glob(os.path.join(GRAIL_ROBOT_DIR, "*.pkl")))
        self.assertEqual(len(paths), 1769)
        measured = {}
        for path in paths:
            base = os.path.splitext(os.path.basename(path))[0]
            measured[base] = GrailTerrain.from_base(base).footprint()["height"]
        for base, expected in LOCKED_GRAIL_HEIGHTS.items():
            self.assertAlmostEqual(measured[base], expected, delta=1e-12)
        self.assertEqual(
            select_grail_scene_bases(measured), LOCKED_GRAIL_BASES)

    def test_full_definition_catalog_has_locked_order_and_classes(self):
        clips = {
            base: fake_grail_clip(base)
            for base in LOCKED_GRAIL_BASES.values()
        }
        definitions = all_scene_definitions(LOCKED_GRAIL_HEIGHTS, clips)
        self.assertEqual(
            tuple(scene.scene_id for scene in definitions), REQUIRED_SCENE_IDS)
        grail = definitions[:4]
        self.assertEqual(
            [(scene.routes[0].expected_outcome,
              scene.routes[0].walkability_class) for scene in grail],
            [("traverse-or-safe-stop", 2), ("traverse", 1),
             ("traverse-or-safe-stop", 2),
             ("traverse-or-safe-stop", 2)])

    def test_grail_g1wm_regions_routes_and_endpoint_footprints_are_class_pure(
            self):
        clips = {
            base: fake_grail_clip(base)
            for base in LOCKED_GRAIL_BASES.values()
        }
        definitions = grail_scene_definitions(
            LOCKED_GRAIL_HEIGHTS, clips)
        self.assertEqual(
            tuple(scene.scene_id for scene in definitions),
            REQUIRED_SCENE_IDS[:4])
        expected_classes = {"blocked": 0, "certified": 1, "stress": 2}
        for definition in definitions:
            expected = definition.routes[0].walkability_class
            halo = np.float32(WALKABILITY_CLASSIFICATION_HALO)
            xmin, xmax, zmin, zmax = (
                np.float32(value)
                for value in definition.playable_bounds_xz
            )
            expanded = tuple(float(value) for value in (
                np.float32(xmin - halo), np.float32(xmax + halo),
                np.float32(zmin - halo), np.float32(zmax + halo),
            ))
            center_x = float(np.float32(
                np.float32(xmin + xmax) / np.float32(2.0)))
            center_z = float(np.float32(
                np.float32(zmin + zmax) / np.float32(2.0)))
            edge_queries = (
                (expanded[0], center_z, 0, -np.inf),
                (expanded[1], center_z, 0, np.inf),
                (center_x, expanded[2], 1, -np.inf),
                (center_x, expanded[3], 1, np.inf),
            )
            for x, z, axis, outward in edge_queries:
                with self.subTest(
                        scene=definition.scene_id, halo_edge=(x, z)):
                    self.assertEqual(definition.walkability(x, z), expected)
                    outside = [x, z]
                    outside[axis] = np.nextafter(outside[axis], outward)
                    self.assertEqual(
                        definition.walkability(*outside), 0)

            built = build_scene(definition)
            grid, walkability = decoded_scene_grid_and_walkability(built)
            metadata = built.metadata
            normalized_playable = tuple(
                float(np.float32(value))
                for value in definition.playable_bounds_xz
            )
            published_playable = (
                metadata["bounds"]["playable_min_xz"][0],
                metadata["bounds"]["playable_max_xz"][0],
                metadata["bounds"]["playable_min_xz"][1],
                metadata["bounds"]["playable_max_xz"][1],
            )
            self.assertEqual(published_playable, normalized_playable)
            region_class = "certified" if expected == 1 else "stress"
            self.assertEqual(len(metadata["regions"][region_class]), 1)
            self.assertEqual(
                tuple(metadata["regions"][region_class][0]["bounds_xz"]),
                normalized_playable)
            self.assertEqual(metadata["regions"]["blocked"], [])
            self.assertEqual(
                metadata["regions"][
                    "stress" if region_class == "certified" else "certified"],
                [])
            with self.subTest(scene=definition.scene_id, contract="source"):
                self.assertEqual(
                    metadata["provenance"]["source_ids"][0],
                    LOCKED_GRAIL_BASES[definition.scene_id])
            for class_name, regions in metadata["regions"].items():
                for region in regions:
                    with self.subTest(
                            scene=definition.scene_id,
                            region=region["id"]):
                        cells = _region_cell_indices(
                            grid, region["bounds_xz"])
                        self.assertTrue(cells)
                        self.assertEqual(
                            {int(walkability[iz, ix]) for iz, ix in cells},
                            {expected_classes[class_name]})
            for route in metadata["routes"]:
                points = tuple(
                    tuple(point) for point in route["waypoints_xz"])
                expected = route["walkability_class"]
                covers = _route_cell_covers(grid, points)
                self.assertTrue(covers)
                with self.subTest(
                        scene=definition.scene_id, route=route["id"]):
                    self.assertTrue(all(
                        {int(walkability[iz, ix]) for iz, ix in cover}
                        == {expected}
                        for cover in covers))
                for endpoint in (points[0], points[-1]):
                    with self.subTest(
                            scene=definition.scene_id, endpoint=endpoint):
                        self.assertEqual(
                            set(decoded_footprint_classes(
                                grid, walkability, *endpoint)),
                            {expected})


class ProceduralSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = {
            scene.scene_id: scene
            for scene in procedural_scene_definitions()
        }
        cls.built = {
            scene_id: build_scene(definition)
            for scene_id, definition in cls.definitions.items()
        }

    def test_procedural_catalog_order_and_route_contracts(self):
        definitions = procedural_scene_definitions()
        self.assertEqual(
            tuple(scene.scene_id for scene in definitions),
            REQUIRED_SCENE_IDS[4:])
        expected = {
            "stairs-shallow": ("ascent-landing-descent", "traverse", 1),
            "stairs-standard": ("ascent-landing-descent", "traverse", 1),
            "stairs-unseen-variable": (
                "ascent-landing-descent", "traverse", 1),
            "ramp-05-up-down": ("up-landing-down", "traverse", 1),
            "ramp-10-up-down": ("up-landing-down", "traverse", 1),
            "ramp-15-stress": (
                "up-landing-down", "traverse-or-safe-stop", 2),
            "cross-slope-05": ("forward-cross-slope", "traverse", 1),
            "cross-slope-10": ("forward-cross-slope", "traverse", 1),
            "mixed-multilevel": ("full-course", "traverse", 1),
        }
        for scene_id, contract in expected.items():
            route = self.definitions[scene_id].routes[0]
            self.assertEqual(
                (route.route_id, route.expected_outcome,
                 route.walkability_class), contract)
        blocked = self.definitions["blocked-course"].routes
        self.assertEqual(
            [(route.route_id, route.expected_outcome, route.walkability_class)
             for route in blocked],
            [("wall-safe-stop", "safe-stop", 0),
             ("ramp-safe-stop", "safe-stop", 0)])

    def test_stair_dimensions_landings_and_return_to_base_are_exact(self):
        expected = {
            "stairs-shallow": ([0.08] * 4, [0.30] * 4),
            "stairs-standard": ([0.12] * 3, [0.32] * 3),
            "stairs-unseen-variable": (
                [0.06, 0.10, 0.08, 0.12],
                [0.24, 0.34, 0.28, 0.38]),
        }
        for scene_id, (rises, runs) in expected.items():
            scene = self.definitions[scene_id]
            parameters = scene.provenance["parameters"]
            self.assertEqual(parameters["rises_m"], rises)
            self.assertEqual(parameters["runs_m"], runs)
            self.assertEqual(parameters["width_m"], 1.2)
            self.assertEqual(parameters["landing_length_m"], 2.0)
            ascent_end = 2.0 + sum(runs)
            self.assertAlmostEqual(
                scene.surface.height(0.0, ascent_end + 1.0), sum(rises))
            self.assertAlmostEqual(
                scene.surface.height(0.0, 2.0 + 2 * sum(runs) + 2.01), 0.0)

    def test_ramp_run_is_exact_formula_and_profile_is_continuous(self):
        for degrees in (5, 10, 15):
            scene_id = f"ramp-{degrees:02d}-" + (
                "stress" if degrees == 15 else "up-down")
            scene = self.definitions[scene_id]
            run = 0.36 / np.tan(np.deg2rad(degrees))
            self.assertAlmostEqual(
                scene.provenance["parameters"]["run_m"], run, places=12)
            self.assertAlmostEqual(
                scene.surface.height(0.0, 2.0 + run / 2.0), 0.18,
                places=12)
            self.assertAlmostEqual(
                scene.surface.height(0.0, 2.0 + run + 1.0), 0.36,
                places=12)
            self.assertAlmostEqual(
                scene.surface.height(0.0, 2.0 + 2 * run + 2.0), 0.0,
                places=12)

    def test_cross_slopes_have_four_metre_grade_and_flat_entry_exit(self):
        for degrees in (5, 10):
            scene = self.definitions[f"cross-slope-{degrees:02d}"]
            parameters = scene.provenance["parameters"]
            self.assertEqual(parameters["grade_length_m"], 4.0)
            self.assertEqual(parameters["flat_spawn_length_m"], 2.0)
            self.assertEqual(parameters["flat_entry_length_m"], 1.0)
            self.assertEqual(parameters["flat_exit_length_m"], 1.0)
            self.assertEqual(scene.surface.height(0.6, 2.5), 0.0)
            self.assertAlmostEqual(
                scene.surface.height(0.6, 5.0),
                0.6 * np.tan(np.deg2rad(degrees)), places=12)
            self.assertEqual(scene.surface.height(0.6, 7.5), 0.0)

    def test_mixed_course_signed_blocks_and_ramp_return_to_zero(self):
        scene = self.definitions["mixed-multilevel"]
        parameters = scene.provenance["parameters"]
        self.assertEqual(parameters["stair_rises_m"], [0.08] * 4)
        self.assertEqual(parameters["elevated_walk_length_m"], 3.0)
        self.assertEqual(parameters["block_height_changes_m"], [0.08, -0.12, 0.04])
        self.assertEqual(parameters["block_top_length_m"], 0.60)
        starts = parameters["block_starts_z_m"]
        for start, expected_height in zip(starts, (0.40, 0.28, 0.32)):
            self.assertAlmostEqual(
                scene.surface.height(0.0, start + 0.30), expected_height)
        self.assertAlmostEqual(
            scene.surface.height(0.0, parameters["course_end_z_m"]), 0.0,
            places=12)

    def test_spawn_margin_corridor_and_walkability_are_locked(self):
        for scene in procedural_scene_definitions():
            self.assertEqual(scene.surface.height(0.0, 0.0), 0.0)
            self.assertGreaterEqual(
                scene.provenance["parameters"]["flat_spawn_length_m"], 2.0)
            xmin, xmax, zmin, zmax = scene.heightfield_bounds_xz
            for route in scene.routes:
                if route.expected_outcome != "traverse":
                    continue
                for x, z in route.waypoints_xz:
                    self.assertGreaterEqual(x - xmin, 1.0)
                    self.assertGreaterEqual(xmax - x, 1.0)
                    self.assertGreaterEqual(z - zmin, 1.0)
                    self.assertGreaterEqual(zmax - z, 1.0)
                    self.assertEqual(scene.walkability(x, z), 1)

    def test_blocked_course_marks_both_obstacle_approaches_blocked(self):
        scene = self.definitions["blocked-course"]
        parameters = scene.provenance["parameters"]
        self.assertEqual(parameters["wall_height_m"], 0.45)
        self.assertEqual(parameters["wall_top_length_m"], 0.50)
        self.assertEqual(parameters["ramp_rise_m"], 0.36)
        self.assertEqual(parameters["ramp_angle_degrees"], 25.0)
        self.assertEqual(scene.walkability(-0.8, 1.5), 1)
        self.assertEqual(scene.walkability(-0.8, 2.01), 0)
        self.assertEqual(scene.walkability(0.8, 2.01), 0)
        self.assertAlmostEqual(scene.surface.height(-0.8, 2.25), 0.45)
        self.assertGreater(scene.surface.height(0.8, 2.25), 0.0)

    def test_binary32_halo_expands_only_classification_outer_bounds(self):
        halo = np.float32(WALKABILITY_CLASSIFICATION_HALO)

        def expanded(bounds):
            xmin, xmax, zmin, zmax = (np.float32(value) for value in bounds)
            return tuple(float(value) for value in (
                np.float32(xmin - halo), np.float32(xmax + halo),
                np.float32(zmin - halo), np.float32(zmax + halo),
            ))

        for scene_id, scene in self.definitions.items():
            if scene_id == "blocked-course":
                continue
            expected = scene.routes[0].walkability_class
            xmin, xmax, zmin, zmax = expanded(scene.playable_bounds_xz)
            self.assertEqual(scene.walkability(xmin, 0.0), expected)
            self.assertEqual(scene.walkability(xmax, 0.0), expected)
            self.assertEqual(scene.walkability(0.0, zmin), expected)
            self.assertEqual(scene.walkability(0.0, zmax), expected)
            self.assertEqual(
                scene.walkability(np.nextafter(xmin, -np.inf), 0.0), 0)
            self.assertEqual(
                scene.walkability(np.nextafter(xmax, np.inf), 0.0), 0)
            self.assertEqual(
                scene.walkability(0.0, np.nextafter(zmin, -np.inf)), 0)
            self.assertEqual(
                scene.walkability(0.0, np.nextafter(zmax, np.inf)), 0)

        blocked = self.definitions["blocked-course"]
        xmin, xmax, zmin, _ = expanded(blocked.playable_bounds_xz)
        self.assertEqual(blocked.walkability(xmin, 0.0), 1)
        self.assertEqual(blocked.walkability(xmax, 0.0), 1)
        self.assertEqual(blocked.walkability(0.0, zmin), 1)
        self.assertEqual(
            blocked.walkability(np.nextafter(xmin, -np.inf), 0.0), 0)
        threshold = blocked.surface.obstacle_start_z - SCENE_CELL_SIZE
        self.assertEqual(blocked.walkability(0.0, threshold), 1)
        self.assertEqual(
            blocked.walkability(
                0.0, np.nextafter(threshold, np.inf)),
            0)

    def test_decoded_routes_and_endpoint_footprints_match_g1wm(self):
        expected_only_routes = 0
        safe_stop_routes = 0
        for scene_id, scene in self.built.items():
            with self.subTest(scene=scene_id):
                grid, walkability = decoded_scene_grid_and_walkability(scene)
                metadata = scene.metadata
                spawn_xz = (
                    metadata["spawn"]["position"][0],
                    metadata["spawn"]["position"][2],
                )
                for route in metadata["routes"]:
                    points = tuple(
                        tuple(point) for point in route["waypoints_xz"])
                    self.assertEqual(points[0], spawn_xz)
                    observed = tuple(
                        frozenset(
                            int(walkability[iz, ix])
                            for iz, ix in cover)
                        for cover in _route_cell_covers(grid, points)
                    )
                    expected = route["walkability_class"]
                    if expected in (1, 2):
                        expected_only_routes += 1
                        self.assertTrue(all(
                            values == {expected} for values in observed))
                        for endpoint in (points[0], points[-1]):
                            self.assertEqual(
                                set(decoded_footprint_classes(
                                    grid, walkability, *endpoint)),
                                {expected})
                    else:
                        safe_stop_routes += 1
                        boundary = [
                            index for index, values in enumerate(observed)
                            if values == {0, 1}
                        ]
                        self.assertEqual(len(boundary), 1)
                        split = boundary[0]
                        self.assertGreater(split, 0)
                        self.assertLess(split, len(observed) - 1)
                        self.assertTrue(all(
                            values == {1} for values in observed[:split]))
                        self.assertTrue(all(
                            values == {0} for values in observed[split + 1:]))
                        self.assertEqual(
                            set(decoded_footprint_classes(
                                grid, walkability, *points[0])),
                            {1})
                        self.assertEqual(
                            set(decoded_footprint_classes(
                                grid, walkability, *points[-1])),
                            {0})
        self.assertEqual(expected_only_routes, 9)
        self.assertEqual(safe_stop_routes, 2)

    def test_decoded_regions_are_nonempty_and_class_pure(self):
        expected_classes = {"blocked": 0, "certified": 1, "stress": 2}
        for scene_id, scene in self.built.items():
            grid, walkability = decoded_scene_grid_and_walkability(scene)
            for class_name, regions in scene.metadata["regions"].items():
                for region in regions:
                    with self.subTest(
                            scene=scene_id, region=region["id"]):
                        cells = _region_cell_indices(
                            grid, region["bounds_xz"])
                        self.assertTrue(cells)
                        observed = tuple(
                            int(walkability[iz, ix]) for iz, ix in cells)
                        self.assertEqual(
                            set(observed),
                            {expected_classes[class_name]})

    def test_serialized_routes_hold_landings_and_derived_margins(self):
        for scene_id, built in self.built.items():
            metadata = built.metadata
            definition = self.definitions[scene_id]
            grid, _ = decoded_scene_grid_and_walkability(built)
            minimum = metadata["bounds"]["heightfield_min_xyz"]
            maximum = metadata["bounds"]["heightfield_max_xyz"]
            spawn_xz = (
                metadata["spawn"]["position"][0],
                metadata["spawn"]["position"][2],
            )
            course_end = metadata["provenance"]["parameters"][
                "course_end_z_m"]
            self.assertEqual(
                struct.unpack("<f", struct.pack("<f", course_end))[0],
                course_end)
            self.assertEqual(
                metadata["bounds"]["playable_max_xz"][1], course_end)
            self.assertGreaterEqual(maximum[2] - course_end, 1.0)
            for route in metadata["routes"]:
                points = tuple(
                    tuple(point) for point in route["waypoints_xz"])
                self.assertEqual(points[0], spawn_xz)
                self.assertTrue(all(
                    start != stop
                    for start, stop in zip(points, points[1:])))
                for point in points:
                    for coordinate in point:
                        packed = struct.pack("<f", coordinate)
                        self.assertEqual(
                            struct.unpack("<f", packed)[0], coordinate)
                        if coordinate == 0.0:
                            self.assertEqual(
                                struct.unpack("<I", packed)[0], 0)
                if route["expected_outcome"] != "safe-stop":
                    for x, z in points:
                        self.assertGreaterEqual(x - minimum[0], 1.0)
                        self.assertGreaterEqual(maximum[0] - x, 1.0)
                        self.assertGreaterEqual(z - minimum[2], 1.0)
                        self.assertGreaterEqual(maximum[2] - z, 1.0)
                if route["landing_hold_seconds"] > 0.0:
                    x, z = points[2]
                    landing_height = definition.surface.height(x, z)
                    self.assertGreater(
                        landing_height,
                        definition.surface.height(*spawn_xz))
                    self.assertAlmostEqual(
                        definition.surface.height(x, z - grid.cell_size),
                        landing_height, places=9)
                    self.assertAlmostEqual(
                        definition.surface.height(x, z + grid.cell_size),
                        landing_height, places=9)

    def test_provenance_is_native_json_and_builds_repeat_byte_exactly(self):
        for scene_id, definition in self.definitions.items():
            with self.subTest(scene=scene_id):
                assert_native_json_value(self, definition.provenance)
                canonical_json_bytes(definition.provenance)
                rebuilt = build_scene(definition)
                original = self.built[scene_id]
                self.assertEqual(rebuilt.scene_json, original.scene_json)
                self.assertEqual(rebuilt.terrain_bin, original.terrain_bin)
                self.assertEqual(rebuilt.terrain_obj, original.terrain_obj)
                self.assertEqual(
                    rebuilt.walkability_bin, original.walkability_bin)
        blocked = self.definitions["blocked-course"].surface
        self.assertIs(type(blocked.ramp_run), float)
        self.assertIs(type(blocked.height(0.8, 2.1)), float)
        for scene_id in ("cross-slope-05", "cross-slope-10"):
            self.assertIs(
                type(self.definitions[scene_id].surface.height(0.6, 5.0)),
                float)

    def test_piecewise_continuous_joins_and_declared_vertical_edges(self):
        def assert_continuous(scene, z, x=0.0):
            center = scene.surface.height(x, z)
            self.assertAlmostEqual(
                scene.surface.height(x, np.nextafter(z, -np.inf)),
                center, places=9)
            self.assertAlmostEqual(
                center,
                scene.surface.height(x, np.nextafter(z, np.inf)),
                places=9)

        for degrees in (5, 10, 15):
            scene_id = f"ramp-{degrees:02d}-" + (
                "stress" if degrees == 15 else "up-down")
            scene = self.definitions[scene_id]
            p = scene.provenance["parameters"]
            for join in (
                    p["flat_spawn_length_m"], p["ascent_end_z_m"],
                    p["descent_start_z_m"],
                    p["descent_start_z_m"] + p["run_m"]):
                assert_continuous(scene, join)

        for degrees in (5, 10):
            scene = self.definitions[f"cross-slope-{degrees:02d}"]
            p = scene.provenance["parameters"]
            start = p["flat_spawn_length_m"] + p["flat_entry_length_m"]
            length = p["grade_length_m"]
            transition = p["transition_length_m"]
            for join in (
                    start, start + transition,
                    start + length - transition, start + length):
                assert_continuous(scene, join, x=0.6)

        mixed = self.definitions["mixed-multilevel"]
        p = mixed.provenance["parameters"]
        ramp_start = p["block_starts_z_m"][-1] + p["block_top_length_m"]
        assert_continuous(mixed, ramp_start)
        assert_continuous(mixed, ramp_start + p["return_ramp_run_m"])
        mixed_ascent_faces = tuple(
            p["flat_spawn_length_m"] + sum(p["stair_runs_m"][:index])
            for index in range(len(p["stair_runs_m"]))
        )
        for face, rise in zip(mixed_ascent_faces, p["stair_rises_m"]):
            before = mixed.surface.height(
                0.0, np.nextafter(face, -np.inf))
            self.assertAlmostEqual(
                mixed.surface.height(0.0, face) - before, rise, places=9)
        for start, change in zip(
                p["block_starts_z_m"], p["block_height_changes_m"]):
            before = mixed.surface.height(0.0, np.nextafter(start, -np.inf))
            self.assertAlmostEqual(
                mixed.surface.height(0.0, start) - before, change, places=9)

        for scene_id in (
                "stairs-shallow", "stairs-standard",
                "stairs-unseen-variable"):
            stairs = self.definitions[scene_id]
            p = stairs.provenance["parameters"]
            ascent_faces = tuple(
                p["flat_spawn_length_m"] + sum(p["runs_m"][:index])
                for index in range(len(p["runs_m"]))
            )
            for face, rise in zip(ascent_faces, p["rises_m"]):
                before = stairs.surface.height(
                    0.0, np.nextafter(face, -np.inf))
                self.assertAlmostEqual(
                    stairs.surface.height(0.0, face) - before,
                    rise, places=9)
            reversed_runs = tuple(reversed(p["runs_m"]))
            descent_faces = tuple(
                p["descent_start_z_m"]
                + sum(reversed_runs[:index + 1])
                for index in range(len(reversed_runs))
            )
            for face, rise in zip(
                    descent_faces, reversed(p["rises_m"])):
                before = stairs.surface.height(
                    0.0, np.nextafter(face, -np.inf))
                self.assertAlmostEqual(
                    stairs.surface.height(0.0, face) - before,
                    -rise, places=9)

        blocked = self.definitions["blocked-course"].surface
        self.assertEqual(
            blocked.height(-0.8, blocked.obstacle_start_z),
            blocked.wall_height)
        self.assertEqual(
            blocked.height(
                -0.8,
                np.nextafter(blocked.obstacle_start_z, -np.inf)),
            0.0)
        wall_end = blocked.obstacle_start_z + blocked.wall_top_length
        self.assertEqual(blocked.height(-0.8, wall_end), blocked.wall_height)
        self.assertEqual(
            blocked.height(-0.8, np.nextafter(wall_end, np.inf)), 0.0)
        ramp_ascent_end = blocked.obstacle_start_z + blocked.ramp_run
        self.assertAlmostEqual(
            blocked.height(
                0.8, np.nextafter(ramp_ascent_end, -np.inf)),
            blocked.ramp_rise, places=9)
        self.assertEqual(
            blocked.height(0.8, ramp_ascent_end), blocked.ramp_rise)
        ramp_end = ramp_ascent_end + blocked.ramp_top_length
        self.assertEqual(blocked.height(0.8, ramp_end), blocked.ramp_rise)
        self.assertEqual(
            blocked.height(0.8, np.nextafter(ramp_end, np.inf)), 0.0)
        wall_top_z = blocked.obstacle_start_z + 0.25
        wall_x_edges = (
            (blocked.wall_center_x - blocked.lane_half_width, -np.inf),
            (blocked.wall_center_x + blocked.lane_half_width, np.inf),
        )
        for edge, outward in wall_x_edges:
            self.assertEqual(
                blocked.height(edge, wall_top_z), blocked.wall_height)
            self.assertEqual(
                blocked.height(np.nextafter(edge, outward), wall_top_z),
                0.0)

        ramp_top_z = ramp_ascent_end + 0.25
        ramp_x_edges = (
            (blocked.ramp_center_x - blocked.lane_half_width, -np.inf),
            (blocked.ramp_center_x + blocked.lane_half_width, np.inf),
        )
        for edge, outward in ramp_x_edges:
            self.assertEqual(
                blocked.height(edge, ramp_top_z), blocked.ramp_rise)
            self.assertEqual(
                blocked.height(np.nextafter(edge, outward), ramp_top_z),
                0.0)


if __name__ == "__main__":
    unittest.main()
