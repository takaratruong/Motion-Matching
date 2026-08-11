from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from mm_sonic import hybrid_terrain_lmm_viewer as viewer_module
from mm_sonic.hybrid_terrain_lmm_runtime import CommandState, TerrainAuthority
from mm_sonic.hybrid_terrain_lmm_viewer import (
    DEFAULT_G1_XML,
    EvdevCommandSource,
    FixedRateAccumulator,
    KeyboardCommandSource,
    build_parser,
    handle_key_press,
    load_scene_terrain,
    main,
    optional_evdev_source,
    overlay_text,
    reset_without_mutation_on_failure,
    run_mujoco_headless_smoke,
)


class _Axes:
    def __init__(self, values):
        self.values = values

    def read_axes(self):
        if isinstance(self.values, BaseException):
            raise self.values
        return self.values


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _write_test_pfnn_authority(root: Path) -> dict[str, object]:
    root.mkdir()
    payload = _canonical_json(
        {"schema": "pfnn-terrain-lmm-supplement/v1", "status": "accepted"}
    )
    path = root / "manifest.json"
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "schema": "pfnn-terrain-lmm-supplement/v1",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_authenticated_scene_pack(root: Path) -> SimpleNamespace:
    scene_id = "ramp-10-up-down"
    scene_root = root / "scenes" / scene_id
    scene_root.mkdir(parents=True)
    authority = TerrainAuthority.multi_hill()
    xs, ys, sampled = authority.height_grid(
        bounds=(-6.0, 6.0, -8.0, 2.0), shape=(101, 121)
    )
    heights = np.ascontiguousarray(sampled[::-1], dtype="<f4")
    terrain_payload = (
        struct.pack(
            "<4sIIIffff",
            b"G1HF",
            2,
            heights.shape[1],
            heights.shape[0],
            float(xs[0]),
            float(-ys[-1]),
            float(xs[1] - xs[0]),
            0.0,
        )
        + heights.tobytes()
    )
    (scene_root / "terrain.bin").write_bytes(terrain_payload)
    surface_signature = "a" * 64
    scene_payload = _canonical_json(
        {
            "schema": "g1-terrain-scene/v1",
            "id": scene_id,
            "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
            "surface_signature": surface_signature,
            "heightfield": {
                "path": "terrain.bin",
                "sha256": hashlib.sha256(terrain_payload).hexdigest(),
            },
            "spawn": {"position": [0.0, 0.0, 0.0], "yaw_radians": 0.0},
        }
    )
    (scene_root / "scene.json").write_bytes(scene_payload)
    index_payload = _canonical_json(
        {
            "schema": "g1-terrain-scene-index/v1",
            "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
            "surface_signature": surface_signature,
            "default_scene_id": scene_id,
            "scene_ids": [scene_id],
            "scenes": [
                {
                    "id": scene_id,
                    "path": f"scenes/{scene_id}/scene.json",
                    "sha256": hashlib.sha256(scene_payload).hexdigest(),
                }
            ],
        }
    )
    (root / "scenes" / "index.json").write_bytes(index_payload)
    manifest_payload = _canonical_json(
        {
            "schema": "g1-terrain-artifacts/v3",
            "scene_index": {
                "path": "scenes/index.json",
                "schema": "g1-terrain-scene-index/v1",
                "sha256": hashlib.sha256(index_payload).hexdigest(),
            },
        }
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(manifest_payload)
    return SimpleNamespace(
        source_root=root,
        manifest_receipt={
            "schema": "g1-hybrid-terrain-lmm-source-receipt/v1",
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "size_bytes": len(manifest_payload),
        },
    )


def _write_full_walking_scene_pack(root: Path) -> SimpleNamespace:
    _write_authenticated_scene_pack(root)
    scene_id = "ramp-10-up-down"
    member_names = (
        "scenes/index.json",
        f"scenes/{scene_id}/scene.json",
        f"scenes/{scene_id}/terrain.bin",
    )
    members = {}
    for name in member_names:
        payload = (root / name).read_bytes()
        members[name] = {
            "path": name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest_payload = _canonical_json(
        {
            "schema": "g1-full-walking-terrain-lmm-corpus/v1",
            "scene_index": {
                "path": "scenes/index.json",
                "scene_ids": [scene_id],
                "sha256": members["scenes/index.json"]["sha256"],
            },
            "members": members,
        }
    )
    (root / "manifest.json").write_bytes(manifest_payload)
    return SimpleNamespace(
        root=root,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )


def _bind_formal_model_and_cache_authorities(
    root: Path, corpus: object, generator: object
) -> Path:
    primary_root = root / "primary-cache"
    primary_root.mkdir()
    primary_payload = _canonical_json(
        {
            "schema": "g1-hybrid-terrain-lmm-corpus/v2-strict",
            "source_root": str(corpus.source_root),
            "source_manifest": corpus.manifest_receipt,
        }
    )
    primary_path = primary_root / "manifest.json"
    primary_path.write_bytes(primary_payload)
    primary_descriptor = {
        "path": str(primary_path.resolve()),
        "schema": "g1-hybrid-terrain-lmm-corpus/v2-strict",
        "size_bytes": len(primary_payload),
        "sha256": hashlib.sha256(primary_payload).hexdigest(),
    }
    pfnn_descriptor = _write_test_pfnn_authority(root / "pfnn-supplement")
    authorities = {
        "primary_cache": primary_descriptor,
        "pfnn_supplement": pfnn_descriptor,
    }
    cache_root = root / "cache"
    cache_root.mkdir()
    cache_payload = _canonical_json(
        {
            "schema": "g1-hybrid-terrain-lmm-combined-cache/v2-strict",
            "rows": len(corpus.artifacts.positions),
            "authorities": authorities,
        }
    )
    cache_path = cache_root / "manifest.json"
    cache_path.write_bytes(cache_payload)
    cache_sha256 = hashlib.sha256(cache_payload).hexdigest()
    corpus.cache_manifest_sha256 = cache_sha256
    corpus.source_root = cache_root
    corpus.manifest_receipt = {
        "schema": "g1-hybrid-terrain-lmm-combined-receipt/v2-strict",
        "path": str(cache_path.resolve()),
        "size_bytes": len(cache_payload),
        "sha256": cache_sha256,
        "authorities": authorities,
    }

    generator.config = SimpleNamespace(fit_all_rows=True)
    generator.selection_provenance_verified = True
    generator.canonical_selection_verified = True
    generator.evaluation_receipt = {
        "schema": "g1-hybrid-terrain-lmm-evaluation/v1",
        "evaluation_scope": "post-selection-all-row-refit-diagnostic",
        "native_joint_limits_evaluated": False,
    }
    generator.manifest = {
        "schema": "g1-hybrid-terrain-lmm-manifest/v1",
        "corpus_manifest_sha256": cache_sha256,
        "canonical_selection_accepted": True,
        "selection_evaluation_sha256": "e" * 64,
        "selection_model_manifest_sha256": "s" * 64,
        "artifacts": {
            "evaluation.json": {
                "path": "evaluation.json",
                "sha256": "a" * 64,
                "size_bytes": 123,
            }
        },
    }
    model_root = root / "model"
    model_root.mkdir()
    model_payload = _canonical_json(generator.manifest)
    (model_root / "manifest.json").write_bytes(model_payload)
    generator.root = model_root
    generator.manifest_sha256 = hashlib.sha256(model_payload).hexdigest()
    return cache_path


class HybridTerrainViewerTests(unittest.TestCase):
    def test_diagnostic_collision_oracle_is_separate_from_render_model(self):
        import mujoco

        terrains = {
            "flat": load_scene_terrain("flat"),
            "hills": load_scene_terrain("hills"),
            "ramp": load_scene_terrain(
                viewer_module.DEFAULT_TERRAIN_ROOT / "ramp-10-up-down"
            ),
        }
        render_model = viewer_module.build_viewer_model(
            DEFAULT_G1_XML, terrains["hills"]
        )

        render_geom = mujoco.mj_name2id(
            render_model, mujoco.mjtObj.mjOBJ_GEOM, "hybrid_lmm_authoritative_terrain"
        )
        self.assertGreaterEqual(render_geom, 0)
        self.assertEqual(int(render_model.geom_contype[render_geom]), 0)
        self.assertEqual(int(render_model.geom_conaffinity[render_geom]), 0)
        for name, terrain in terrains.items():
            with self.subTest(scene=name):
                collision_model = viewer_module.build_diagnostic_collision_model(
                    DEFAULT_G1_XML, terrain
                )
                collision_geom = mujoco.mj_name2id(
                    collision_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "terrain_hybrid_lmm_authoritative",
                )
                floor = mujoco.mj_name2id(
                    collision_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
                )
                self.assertGreaterEqual(collision_geom, 0)
                self.assertEqual(int(collision_model.geom_contype[collision_geom]), 1)
                self.assertEqual(
                    int(collision_model.geom_conaffinity[collision_geom]), 1
                )
                self.assertEqual(int(collision_model.geom_contype[floor]), 0)
                self.assertEqual(int(collision_model.geom_conaffinity[floor]), 0)
                expected_type = (
                    mujoco.mjtGeom.mjGEOM_PLANE
                    if name == "flat"
                    else mujoco.mjtGeom.mjGEOM_MESH
                )
                self.assertEqual(
                    int(collision_model.geom_type[collision_geom]), int(expected_type)
                )

    def test_interactive_visuals_brighten_only_compiled_render_fields(self):
        import mujoco

        terrain = load_scene_terrain("hills")
        canonical = viewer_module.build_viewer_model(DEFAULT_G1_XML, terrain)

        def skybox_pixels(model):
            texture_ids = np.flatnonzero(
                model.tex_type == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
            )
            self.assertEqual(len(texture_ids), 1)
            texture_id = int(texture_ids[0])
            start = int(model.tex_adr[texture_id])
            channels = int(model.tex_nchannel[texture_id])
            size = int(
                model.tex_width[texture_id] * model.tex_height[texture_id] * channels
            )
            return np.asarray(model.tex_data[start : start + size]).reshape(
                -1, channels
            )

        canonical_skybox = skybox_pixels(canonical)
        self.assertEqual(int(canonical_skybox.min()), 0)
        self.assertEqual(int(canonical_skybox.max()), 0)
        np.testing.assert_allclose(
            canonical.vis.headlight.ambient, (0.1, 0.1, 0.1), atol=1.0e-7
        )

        try:
            interactive = viewer_module.build_viewer_model(
                DEFAULT_G1_XML, terrain, interactive_visuals=True
            )
        except TypeError as error:
            self.fail(f"interactive viewer visuals are unavailable: {error}")

        interactive_skybox = skybox_pixels(interactive)
        self.assertGreater(int(interactive_skybox.min()), 0)
        self.assertGreater(
            np.unique(interactive_skybox, axis=0).shape[0],
            1,
        )
        self.assertTrue(np.all(np.diff(interactive_skybox.min(axis=0)) > 0))
        self.assertTrue(np.all(np.diff(interactive_skybox.max(axis=0)) > 0))
        np.testing.assert_allclose(
            interactive.vis.headlight.ambient, (0.3, 0.3, 0.3), atol=1.0e-7
        )
        for field in (
            "qpos0",
            "jnt_type",
            "jnt_range",
            "geom_type",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_contype",
            "geom_conaffinity",
        ):
            np.testing.assert_array_equal(
                getattr(canonical, field),
                getattr(interactive, field),
                err_msg=field,
            )

    def test_file_scene_domain_is_the_exact_inclusive_native_grid_rectangle(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "finite"
            scene.mkdir()
            heights = np.zeros((2, 3), "<f4")
            (scene / "terrain.bin").write_bytes(
                struct.pack("<4sIIIffff", b"G1HF", 2, 3, 2, -1.0, 2.0, 0.5, -0.2)
                + heights.tobytes()
            )

            authority = load_scene_terrain(scene).authority

            for point in ((-1.0, -2.0), (0.0, -2.5), (-0.5, -2.25)):
                self.assertTrue(authority.contains(point), point)
            for point in (
                (-1.0 - 1e-9, -2.0),
                (1e-9, -2.5),
                (-0.5, -2.0 + 1e-9),
                (-0.5, -2.5 - 1e-9),
            ):
                self.assertFalse(authority.contains(point), point)

    def test_generated_hills_and_flat_scene_grids_are_finite_domains(self):
        hills_adapter = load_scene_terrain("hills")
        hills = hills_adapter.authority
        rows, columns = hills_adapter.source_heights.shape
        x_max = hills_adapter.origin_x + (columns - 1) * hills_adapter.cell_size_m
        native_y_min = -(
            hills_adapter.origin_z + (rows - 1) * hills_adapter.cell_size_m
        )
        native_y_max = -hills_adapter.origin_z
        self.assertTrue(hills.contains((hills_adapter.origin_x, native_y_min)))
        self.assertTrue(hills.contains((x_max, native_y_max)))
        self.assertFalse(hills.contains((x_max + 1e-9, 0.0)))
        self.assertFalse(hills.contains((0.0, native_y_max + 1e-9)))

        flat_grid = load_scene_terrain("flat").authority
        self.assertTrue(flat_grid.contains((0.0, 0.0)))
        self.assertFalse(flat_grid.contains((20.0, 0.0)))

    def test_file_scene_preview_exterior_is_visible_but_bypasses_learned_decode(self):
        from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher

        from tests.python.test_hybrid_terrain_lmm_runtime import (
            _corpus,
            _Generator,
            _pose_converter,
        )

        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "finite"
            scene.mkdir()
            heights = np.zeros((2, 3), "<f4")
            (scene / "terrain.bin").write_bytes(
                struct.pack("<4sIIIffff", b"G1HF", 2, 3, 2, -1.0, 2.0, 0.5, 0.5)
                + heights.tobytes()
            )
            adapter = load_scene_terrain(scene)
            generator = _Generator()
            matcher = HybridMatcher(
                _corpus(),
                generator,
                adapter.authority,
                pose_converter=_pose_converter,
                initial_root_xy=(-0.5, -2.0),
            )

            state = matcher.step(CommandState(), dt=0.04, force_search=True)

            self.assertEqual(state.support_status, "OUT OF TRAINED SUPPORT")
            self.assertEqual(state.pose_source, "canonical-fallback")
            self.assertEqual(state.fallback_count, 1)
            self.assertEqual(generator.inputs, [])
            self.assertAlmostEqual(state.support_height, 0.0)
            self.assertTrue(np.any(np.isclose(state.terrain_features, 0.5)))

    def test_generated_hills_preserve_analytic_coordinates_on_square_grid(self):
        adapter = load_scene_terrain("hills")
        analytic = TerrainAuthority.multi_hill()

        rows, columns = adapter.source_heights.shape
        self.assertAlmostEqual((columns - 1) * adapter.cell_size_m, 12.0)
        self.assertAlmostEqual((rows - 1) * adapter.cell_size_m, 10.0)
        for native_xy in ((0.0, -1.8), (-0.8, -4.0), (0.7, -3.1), (0.0, 0.0)):
            self.assertAlmostEqual(
                adapter.authority.height_at(native_xy),
                analytic.height_at(native_xy),
                places=6,
            )

    def test_only_corpus_indexed_scene_bytes_are_authenticated_for_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = _write_authenticated_scene_pack(root)

            indexed = load_scene_terrain("ramp-10-up-down", corpus=corpus)
            indexed_path = load_scene_terrain(
                root / "scenes" / "ramp-10-up-down", corpus=corpus
            )
            generated = load_scene_terrain("hills", corpus=corpus)
            arbitrary_root = root / "arbitrary"
            arbitrary_root.mkdir()
            (arbitrary_root / "terrain.bin").write_bytes(
                struct.pack("<4sIIIffff", b"G1HF", 2, 2, 2, 0.0, 0.0, 1.0, 0.0)
                + np.zeros((2, 2), "<f4").tobytes()
            )
            arbitrary = load_scene_terrain(arbitrary_root, corpus=corpus)

            self.assertTrue(indexed.scene_authenticated)
            self.assertTrue(indexed_path.scene_authenticated)
            self.assertEqual(indexed.scene_evidence_status, "authenticated-indexed")
            self.assertEqual(len(indexed.source_manifest_sha256), 64)
            self.assertEqual(len(indexed.scene_index_sha256), 64)
            self.assertEqual(len(indexed.scene_index_scene_sha256), 64)
            self.assertFalse(generated.scene_authenticated)
            self.assertEqual(generated.scene_evidence_status, "diagnostic-generated")
            self.assertFalse(arbitrary.scene_authenticated)
            self.assertEqual(arbitrary.scene_evidence_status, "diagnostic-unindexed")

            self.assertTrue(
                viewer_module.scene_authentication_is_current(corpus, indexed)
            )
            indexed_payload = indexed.source_path.read_bytes()
            indexed.source_path.write_bytes(indexed_payload[:-1] + b"\x01")
            self.assertFalse(
                viewer_module.scene_authentication_is_current(corpus, indexed)
            )

    def test_full_walking_corpus_scene_pack_is_authenticated_directly(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = _write_full_walking_scene_pack(Path(temporary))

            indexed = load_scene_terrain("ramp-10-up-down", corpus=corpus)

            self.assertTrue(indexed.scene_authenticated)
            self.assertEqual(indexed.scene_evidence_status, "authenticated-indexed")
            self.assertEqual(indexed.source_manifest_sha256, corpus.manifest_sha256)
            self.assertTrue(
                viewer_module.scene_authentication_is_current(corpus, indexed)
            )

    def test_generated_scene_mujoco_smoke_is_always_acceptance_ineligible(self):
        import mujoco
        from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher

        from tests.python.test_hybrid_terrain_lmm_runtime import (
            _corpus,
            _KeywordGenerator,
            _pose_converter,
        )

        values = np.zeros((8, 31), np.float32)
        values[:, 21:27] = np.tile((0.0, 1.0), 3)
        values[4:, 15:21] = np.asarray((0.0, 0.45 * 0.32, 0.0, 0.45 * 0.68, 0.0, 0.45))
        values[[0, 4], 27:31] = -1.0
        values[[1, 5], 27:31] = 1.0
        corpus = _corpus(values)
        corpus.cache_manifest_sha256 = "c" * 64
        generator = _KeywordGenerator()
        generator.root = None
        generator.manifest = {
            "schema": "hybrid-terrain-lmm-model/v1",
            "canonical_selection_accepted": True,
        }
        terrain = load_scene_terrain("hills")
        native_model = mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML))

        def valid_converter(*arguments):
            qpos = _pose_converter(*arguments)
            qpos[7:] = 0.0
            return qpos

        matcher = HybridMatcher(
            corpus,
            generator,
            terrain.authority,
            native_model=native_model,
            pose_converter=valid_converter,
            transition_penalty=0.100000001,
        )

        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_viewer._search_audit",
            return_value={
                "samples": 2,
                "failures": 0,
                "exact_feature_matches": 1,
            },
        ):
            receipt = run_mujoco_headless_smoke(
                matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1_000
            )

        self.assertFalse(receipt["accepted"])
        self.assertIn("DIAGNOSTIC", receipt["label"])
        self.assertIn(
            "authenticated-indexed-scene-required", receipt["acceptance_failures"]
        )
        self.assertIn("transition-penalty-exactly-0.1", receipt["acceptance_failures"])
        self.assertIn(
            "exact-feature-retrieval-required", receipt["acceptance_failures"]
        )
        self.assertIn("all-row-refit-required", receipt["acceptance_failures"])
        self.assertIn(
            "selection-provenance-verified-required",
            receipt["acceptance_failures"],
        )
        self.assertIn("strict-combined-corpus-required", receipt["acceptance_failures"])
        self.assertEqual(
            receipt["identity"]["scene_evidence_status"], "diagnostic-generated"
        )

    def test_combined_corpus_scene_authentication_resolves_primary_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_authenticated_scene_pack(root / "source")
            primary_root = root / "primary"
            primary_root.mkdir()
            primary_payload = _canonical_json(
                {
                    "schema": "g1-hybrid-terrain-lmm-corpus/v2-strict",
                    "source_root": str(source.source_root),
                    "source_manifest": source.manifest_receipt,
                }
            )
            primary_manifest = primary_root / "manifest.json"
            primary_manifest.write_bytes(primary_payload)
            primary_descriptor = {
                "path": str(primary_manifest.resolve()),
                "schema": "g1-hybrid-terrain-lmm-corpus/v2-strict",
                "size_bytes": len(primary_payload),
                "sha256": hashlib.sha256(primary_payload).hexdigest(),
            }
            pfnn_descriptor = _write_test_pfnn_authority(root / "pfnn")
            authorities = {
                "primary_cache": primary_descriptor,
                "pfnn_supplement": pfnn_descriptor,
            }
            combined_root = root / "combined"
            combined_root.mkdir()
            combined_payload = _canonical_json(
                {
                    "schema": "g1-hybrid-terrain-lmm-combined-cache/v2-strict",
                    "authorities": authorities,
                }
            )
            combined_manifest = combined_root / "manifest.json"
            combined_manifest.write_bytes(combined_payload)
            combined = SimpleNamespace(
                source_root=combined_root,
                manifest_receipt={
                    "schema": "g1-hybrid-terrain-lmm-combined-receipt/v2-strict",
                    "path": str(combined_manifest.resolve()),
                    "size_bytes": len(combined_payload),
                    "sha256": hashlib.sha256(combined_payload).hexdigest(),
                    "authorities": authorities,
                },
            )

            terrain = load_scene_terrain("ramp-10-up-down", corpus=combined)

            self.assertTrue(terrain.scene_authenticated)
            self.assertTrue(
                viewer_module.scene_authentication_is_current(combined, terrain)
            )

            incomplete_root = root / "incomplete-combined"
            incomplete_root.mkdir()
            incomplete_payload = _canonical_json(
                {
                    "schema": "g1-hybrid-terrain-lmm-combined-cache/v2-strict",
                    "authorities": {"primary_cache": primary_descriptor},
                }
            )
            incomplete_manifest = incomplete_root / "manifest.json"
            incomplete_manifest.write_bytes(incomplete_payload)
            incomplete = SimpleNamespace(
                source_root=incomplete_root,
                manifest_receipt={
                    "schema": "g1-hybrid-terrain-lmm-combined-receipt/v2-strict",
                    "path": str(incomplete_manifest.resolve()),
                    "size_bytes": len(incomplete_payload),
                    "sha256": hashlib.sha256(incomplete_payload).hexdigest(),
                    "authorities": {"primary_cache": primary_descriptor},
                },
            )
            with self.assertRaisesRegex(ValueError, "authority inventory"):
                load_scene_terrain("ramp-10-up-down", corpus=incomplete)

            for authority_name in ("primary_cache", "pfnn_supplement"):
                relative_authorities = {
                    name: dict(descriptor) for name, descriptor in authorities.items()
                }
                relative_authorities[authority_name]["path"] = os.path.relpath(
                    relative_authorities[authority_name]["path"], Path.cwd()
                )
                relative_root = root / f"relative-{authority_name}"
                relative_root.mkdir()
                relative_payload = _canonical_json(
                    {
                        "schema": "g1-hybrid-terrain-lmm-combined-cache/v2-strict",
                        "authorities": relative_authorities,
                    }
                )
                relative_manifest = relative_root / "manifest.json"
                relative_manifest.write_bytes(relative_payload)
                relative_corpus = SimpleNamespace(
                    source_root=relative_root,
                    manifest_receipt={
                        "schema": ("g1-hybrid-terrain-lmm-combined-receipt/v2-strict"),
                        "path": str(relative_manifest.resolve()),
                        "size_bytes": len(relative_payload),
                        "sha256": hashlib.sha256(relative_payload).hexdigest(),
                        "authorities": relative_authorities,
                    },
                )
                with (
                    self.subTest(authority=authority_name),
                    self.assertRaisesRegex(ValueError, "absolute"),
                ):
                    load_scene_terrain("ramp-10-up-down", corpus=relative_corpus)

            combined_manifest.write_bytes(combined_payload + b" ")
            self.assertFalse(
                viewer_module.scene_authentication_is_current(combined, terrain)
            )

    def test_smoke_cli_dispatches_to_real_mujoco_gate(self):
        matcher, terrain = object(), object()
        receipt = {"accepted": True, "mujoco_forward_count": 1_000}
        stdout = io.StringIO()
        with (
            mock.patch.object(
                viewer_module,
                "configure_single_gpu_visibility",
                create=True,
            ) as configure_visibility,
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_viewer._load_matcher",
                return_value=(matcher, terrain),
            ),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_viewer.run_mujoco_headless_smoke",
                return_value=receipt,
            ) as smoke,
            contextlib.redirect_stdout(stdout),
        ):
            result = main(
                ["smoke", "--cache", "cache", "--model", "model", "--frames", "1000"]
            )

        self.assertEqual(result, 0)
        smoke.assert_called_once_with(
            matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1_000
        )
        configure_visibility.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue()), receipt)

    def test_gpu_view_configures_visibility_before_loading(self):
        events = []
        matcher, terrain = object(), object()

        def configure(device):
            events.append(("configure", device))

        def load(arguments):
            events.append(("load", arguments.search_device))
            return matcher, terrain

        def run(*_arguments, **_keywords):
            events.append("run")
            return {"accepted": True}

        with (
            mock.patch.object(
                viewer_module,
                "configure_single_gpu_visibility",
                side_effect=configure,
                create=True,
            ),
            mock.patch.object(viewer_module, "_load_matcher", side_effect=load),
            mock.patch.object(viewer_module, "run_interactive", side_effect=run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = main(
                [
                    "view",
                    "--cache",
                    "cache",
                    "--model",
                    "model",
                    "--search-device",
                    "cuda:5",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(events, [("configure", "cuda:5"), ("load", "cuda:5"), "run"])

    def test_gpu_view_rejects_capped_search_before_visibility_or_loading(self):
        with (
            mock.patch.object(
                viewer_module, "configure_single_gpu_visibility"
            ) as configure_visibility,
            mock.patch.object(viewer_module, "_load_matcher") as load,
            self.assertRaisesRegex(ValueError, "search-device.*search-rows"),
        ):
            main(
                [
                    "view",
                    "--cache",
                    "cache",
                    "--model",
                    "model",
                    "--search-device",
                    "cuda:5",
                    "--search-rows",
                    "2",
                ]
            )

        configure_visibility.assert_not_called()
        load.assert_not_called()

    def test_cpu_view_does_not_configure_gpu_visibility(self):
        matcher, terrain = object(), object()
        with (
            mock.patch.object(
                viewer_module,
                "configure_single_gpu_visibility",
                create=True,
            ) as configure_visibility,
            mock.patch.object(
                viewer_module,
                "_load_matcher",
                return_value=(matcher, terrain),
            ) as load,
            mock.patch.object(
                viewer_module,
                "run_interactive",
                return_value={"accepted": True},
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = main(["view", "--cache", "cache", "--model", "model"])

        self.assertEqual(result, 0)
        configure_visibility.assert_not_called()
        self.assertIsNone(load.call_args.args[0].search_device)

    def test_gpu_view_rejects_conflicting_visibility_before_loading(self):
        from mm_sonic.hybrid_terrain_lmm_gpu_search import (
            configure_single_gpu_visibility,
        )

        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}),
            mock.patch.object(
                viewer_module,
                "configure_single_gpu_visibility",
                configure_single_gpu_visibility,
                create=True,
            ),
            mock.patch.object(
                viewer_module,
                "_load_matcher",
                side_effect=AssertionError(
                    "visibility conflict must fail before matcher loading"
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "visibility conflicts"),
        ):
            main(
                [
                    "view",
                    "--cache",
                    "cache",
                    "--model",
                    "model",
                    "--search-device",
                    "cuda:5",
                ]
            )

    def test_load_matcher_passes_search_device_to_runtime(self):
        corpus = SimpleNamespace()
        generator = object()
        adapter = SimpleNamespace(
            authority=object(),
            spawn_native_xy=np.zeros(2, np.float64),
            spawn_heading=0.0,
        )
        data_module = SimpleNamespace(load_hybrid_cache=lambda _path: corpus)
        training_module = SimpleNamespace(
            load_hybrid_generator=lambda _path, *, corpus: generator
        )

        def import_module(name):
            if name == "mm_sonic.hybrid_terrain_lmm_data":
                return data_module
            if name == "mm_sonic.hybrid_terrain_lmm_training":
                return training_module
            raise AssertionError(f"unexpected module import: {name}")

        arguments = SimpleNamespace(
            cache=Path("cache"),
            model=Path("model"),
            scene="ramp-10-up-down",
            g1_xml=DEFAULT_G1_XML,
            search_rows=None,
            transition_penalty=0.1,
            search_device="cuda:5",
        )
        built = object()
        fake_mujoco = SimpleNamespace(
            MjModel=SimpleNamespace(from_xml_path=lambda _path: object())
        )
        with (
            mock.patch.object(
                viewer_module.importlib, "import_module", side_effect=import_module
            ),
            mock.patch.object(
                viewer_module, "load_scene_terrain", return_value=adapter
            ),
            mock.patch.object(
                viewer_module, "HybridMatcher", return_value=built
            ) as constructor,
            mock.patch.dict(sys.modules, {"mujoco": fake_mujoco}),
        ):
            matcher, loaded_adapter = viewer_module._load_matcher(arguments)

        self.assertIs(matcher, built)
        self.assertIs(loaded_adapter, adapter)
        self.assertEqual(constructor.call_args.kwargs["search_device"], "cuda:5")

    def test_real_mujoco_smoke_runs_1000_forwards_and_binds_every_identity(self):
        import mujoco
        from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher

        from tests.python.test_hybrid_terrain_lmm_runtime import (
            _corpus,
            _KeywordGenerator,
            _pose_converter,
        )

        values = np.zeros((8, 31), np.float32)
        values[:, 21:27] = np.tile((0.0, 1.0), 3)
        values[4:, 15:21] = np.asarray((0.0, 0.45 * 0.32, 0.0, 0.45 * 0.68, 0.0, 0.45))
        values[[0, 4], 27:31] = -1.0
        values[[1, 5], 27:31] = 1.0
        corpus = _corpus(values)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        authority = _write_authenticated_scene_pack(Path(temporary.name))
        corpus.source_root = authority.source_root
        corpus.manifest_receipt = authority.manifest_receipt
        generator = _KeywordGenerator()
        cache_manifest = _bind_formal_model_and_cache_authorities(
            Path(temporary.name), corpus, generator
        )
        terrain = load_scene_terrain("ramp-10-up-down", corpus=corpus)
        native_model = mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML))

        reject_next_learned = [False]

        def valid_converter(*arguments):
            qpos = _pose_converter(*arguments)
            qpos[7:] = 0.0
            decoded = np.asarray(arguments[0], dtype=np.float64)
            if reject_next_learned[0] and float(decoded[0]) >= 100.0:
                qpos[7] = 100.0
                reject_next_learned[0] = False
            return qpos

        matcher = HybridMatcher(
            corpus,
            generator,
            terrain.authority,
            native_model=native_model,
            pose_converter=valid_converter,
            initial_root_xy=terrain.spawn_native_xy,
            initial_heading=terrain.spawn_heading,
            cache_manifest_path=cache_manifest,
        )
        reject_next_learned[0] = True

        receipt = run_mujoco_headless_smoke(
            matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1_000
        )

        self.assertEqual(
            receipt["schema"],
            "hybrid-terrain-lmm-mujoco-smoke/v2",
        )
        self.assertEqual(
            receipt["acceptance_contract"],
            "strict-combined-frozen-authorities-scripted-runtime/v2",
        )
        self.assertFalse(receipt["accepted"])
        self.assertIn(
            "frozen-artifact-authorities-required",
            receipt["acceptance_failures"],
        )
        self.assertEqual(receipt["mujoco_forward_count"], 1_000)
        self.assertGreaterEqual(len(receipt["selected_ranges"]), 2)
        self.assertGreaterEqual(len(receipt["terrain_classes"]), 2)
        self.assertEqual(receipt["fallback_count"], 0)
        self.assertEqual(receipt["canonical_fallback_count"], 0)
        self.assertEqual(receipt["clamped_fallback_count"], 0)
        self.assertEqual(receipt["joint_clamp_count"], 0)
        self.assertEqual(receipt["max_joint_clamp_magnitude"], 0.0)
        self.assertEqual(receipt["candidate_limit_rejection_count"], 1)
        self.assertIsInstance(receipt["first_candidate_limit_rejection_row"], int)
        self.assertEqual(receipt["max_candidate_limit_rejections_per_step"], 1)
        self.assertEqual(receipt["crash_count"], 0)
        self.assertTrue(receipt["completed_without_exception"])
        self.assertIsNone(receipt["failure_frame"])
        self.assertIsNone(receipt["failure_reason"])
        self.assertTrue(receipt["canonical_selection_accepted"])
        self.assertTrue(receipt["fit_all_rows"])
        self.assertTrue(receipt["selection_provenance_verified"])
        self.assertTrue(receipt["strict_combined_corpus"])
        self.assertGreater(receipt["terrain_variance"], 0.0)
        self.assertGreater(receipt["tree_brute_parity_samples"], 0)
        self.assertEqual(receipt["tree_brute_parity_failures"], 0)
        self.assertGreater(receipt["retrieval_audit_samples"], 0)
        self.assertEqual(receipt["retrieval_audit_failures"], 0)
        self.assertEqual(
            receipt["retrieval_exact_feature_matches"],
            receipt["retrieval_audit_samples"],
        )
        self.assertGreaterEqual(receipt["range_diversity_count"], 2)
        self.assertEqual(receipt["search_scope"], "full-range-safe-corpus")
        self.assertEqual(receipt["transition_penalty"], 0.1)
        self.assertNotIn(
            "cpu-exact-search-backend-required", receipt["acceptance_failures"]
        )
        identity = receipt["identity"]
        self.assertEqual(identity["search_backend_identity"], "cpu-ckdtree-exact")
        self.assertIsNone(identity["last_search_elapsed_ms"])
        self.assertIsNone(identity["first_runtime_search_elapsed_ms"])
        self.assertEqual(
            identity["cache_manifest_sha256"],
            corpus.cache_manifest_sha256,
        )
        for name in (
            "model_manifest_sha256",
            "scene_index_sha256",
            "source_manifest_sha256",
            "g1_xml_sha256",
            "g1_asset_inventory_sha256",
            "search_view_sha256",
        ):
            self.assertEqual(len(identity[name]), 64, name)
        self.assertIsNone(identity["scene_generated_grid_sha256"])
        self.assertEqual(identity["scene_evidence_status"], "authenticated-indexed")
        self.assertTrue(identity["scene_authentication_current"])
        self.assertTrue(identity["cache_manifest_authority_current"])
        self.assertTrue(identity["model_manifest_authority_current"])
        self.assertTrue(identity["search_identity_current"])
        self.assertTrue(receipt["evidence_authority_unchanged"])
        self.assertEqual(receipt["evidence_authority_pre"], identity)
        self.assertEqual(receipt["evidence_authority_post"], identity)
        self.assertFalse(receipt["full_dataset_native_limit_audit_performed"])
        self.assertEqual(
            receipt["native_limit_audit_scope"],
            "scripted-runtime-mujoco-forward-only",
        )
        self.assertEqual(
            receipt["all_row_canonical_evaluation_identity"][
                "evaluation_artifact_sha256"
            ],
            "a" * 64,
        )
        self.assertGreaterEqual(len(identity["g1_asset_inventory"]), 30)
        for descriptor in identity["g1_asset_inventory"].values():
            self.assertEqual(len(descriptor["sha256"]), 64)
            self.assertGreater(descriptor["size_bytes"], 0)

        before = viewer_module._runtime_identity(matcher, terrain, DEFAULT_G1_XML)
        gpu_identity = dict(before)
        gpu_identity["search_backend_identity"] = "single-gpu-full-row-fp32:cuda:5"
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_viewer._runtime_identity",
            side_effect=(gpu_identity, gpu_identity),
        ) as identity_check:
            gpu_snapshot = run_mujoco_headless_smoke(
                matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1
            )
        self.assertEqual(identity_check.call_count, 2)
        self.assertFalse(gpu_snapshot["accepted"])
        self.assertIn(
            "cpu-exact-search-backend-required",
            gpu_snapshot["acceptance_failures"],
        )

        after = dict(before)
        after["model_manifest_sha256"] = "0" * 64
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_viewer._runtime_identity",
            side_effect=(before, after),
        ) as identity_check:
            changed = run_mujoco_headless_smoke(
                matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1
            )
        self.assertEqual(identity_check.call_count, 2)
        self.assertFalse(changed["accepted"])
        self.assertFalse(changed["evidence_authority_unchanged"])
        self.assertIn(
            "evidence-authority-unchanged",
            changed["acceptance_failures"],
        )

        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_viewer._runtime_identity",
            side_effect=(before, OSError("model manifest disappeared")),
        ) as identity_check:
            disappeared = run_mujoco_headless_smoke(
                matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1
            )
        self.assertEqual(identity_check.call_count, 2)
        self.assertFalse(disappeared["accepted"])
        self.assertFalse(disappeared["evidence_authority_current"])
        self.assertIn(
            "evidence-authorities-current",
            disappeared["acceptance_failures"],
        )
        self.assertFalse(
            disappeared["evidence_authority_post"]["identity_capture_succeeded"]
        )

        rejected_status = dict(before["generator_acceptance_status"])
        rejected_status["accepted"] = False
        rejected_status["fit_all_rows"] = False
        rejected_identity = dict(before)
        rejected_identity["generator_acceptance_status"] = rejected_status
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_viewer._runtime_identity",
            side_effect=(rejected_identity, rejected_identity),
        ):
            rejected_snapshot = run_mujoco_headless_smoke(
                matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1
            )
        self.assertFalse(rejected_snapshot["accepted"])
        self.assertIn(
            "all-row-refit-required",
            rejected_snapshot["acceptance_failures"],
        )
        self.assertFalse(rejected_snapshot["fit_all_rows"])

        wrong_penalty_identity = dict(before)
        wrong_penalty_identity["transition_penalty"] = 0.2
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_viewer._runtime_identity",
            side_effect=(wrong_penalty_identity, wrong_penalty_identity),
        ):
            rejected_penalty_snapshot = run_mujoco_headless_smoke(
                matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1
            )
        self.assertIn(
            "transition-penalty-exactly-0.1",
            rejected_penalty_snapshot["acceptance_failures"],
        )
        self.assertEqual(rejected_penalty_snapshot["transition_penalty"], 0.2)

    def test_real_mujoco_smoke_completes_1000_joint_clamped_forwards_as_rejected(self):
        import mujoco
        from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher

        from tests.python.test_hybrid_terrain_lmm_runtime import (
            _corpus,
            _Generator,
            _pose_converter,
        )

        corpus = _corpus()
        corpus.cache_manifest_sha256 = "c" * 64
        generator = _Generator()
        generator.root = None
        generator.manifest = {
            "schema": "hybrid-terrain-lmm-model/v1",
            "canonical_selection_accepted": True,
        }
        terrain = load_scene_terrain("hills")
        native_model = mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML))

        def joint_invalid_converter(*arguments):
            qpos = _pose_converter(*arguments)
            for joint_id in range(1, int(native_model.njnt)):
                address = int(native_model.jnt_qposadr[joint_id])
                lower, upper = np.asarray(native_model.jnt_range[joint_id], np.float64)
                qpos[address] = 0.5 * (float(lower) + float(upper))
            first_hinge = int(native_model.jnt_qposadr[1])
            qpos[first_hinge] = float(native_model.jnt_range[1, 1]) + 0.25
            return qpos

        matcher = HybridMatcher(
            corpus,
            generator,
            terrain.authority,
            native_model=native_model,
            pose_converter=joint_invalid_converter,
            initial_root_xy=terrain.spawn_native_xy,
            initial_heading=terrain.spawn_heading,
        )

        receipt = run_mujoco_headless_smoke(
            matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1_000
        )

        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["mujoco_forward_count"], 1_000)
        self.assertEqual(receipt["fallback_count"], 1_000)
        self.assertEqual(receipt["canonical_fallback_count"], 0)
        self.assertEqual(receipt["clamped_fallback_count"], 1_000)
        self.assertEqual(receipt["joint_clamp_count"], 1_000)
        self.assertAlmostEqual(receipt["max_joint_clamp_magnitude"], 0.25)
        self.assertEqual(
            receipt["last_pose_source"], "canonical-fallback-joint-clamped"
        )
        self.assertIsNone(receipt["failure_frame"])
        self.assertIsNone(receipt["failure_reason"])
        self.assertEqual(receipt["crash_count"], 0)
        self.assertTrue(receipt["completed_without_exception"])
        self.assertTrue(receipt["finite"])
        self.assertTrue(receipt["native_joint_limits"])

    def test_mujoco_smoke_receipt_counts_exact_canonical_fallbacks(self):
        import mujoco
        from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher

        from tests.python.test_hybrid_terrain_lmm_runtime import (
            _corpus,
            _Generator,
            _pose_converter,
        )

        corpus = _corpus()
        corpus.cache_manifest_sha256 = "c" * 64
        generator = _Generator()
        generator.invalid = True
        generator.root = None
        generator.manifest = {
            "schema": "hybrid-terrain-lmm-model/v1",
            "canonical_selection_accepted": True,
        }
        terrain = load_scene_terrain("hills")
        native_model = mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML))

        def canonical_valid_converter(*arguments):
            qpos = _pose_converter(*arguments)
            qpos[7:] = 0.0
            return qpos

        matcher = HybridMatcher(
            corpus,
            generator,
            terrain.authority,
            native_model=native_model,
            pose_converter=canonical_valid_converter,
        )

        receipt = run_mujoco_headless_smoke(
            matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=10
        )

        self.assertFalse(receipt["accepted"])
        self.assertIn("minimum-1000-frames", receipt["acceptance_failures"])
        self.assertEqual(receipt["learned_decode_count"], 0)
        self.assertEqual(receipt["canonical_fallback_count"], 10)
        self.assertEqual(receipt["clamped_fallback_count"], 0)
        self.assertEqual(receipt["mujoco_forward_count"], 10)

    def test_mujoco_smoke_catches_step_exception_and_returns_rejected_receipt(self):
        import mujoco
        from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher

        from tests.python.test_hybrid_terrain_lmm_runtime import (
            _corpus,
            _KeywordGenerator,
            _pose_converter,
        )

        corpus = _corpus()
        corpus.cache_manifest_sha256 = "c" * 64
        generator = _KeywordGenerator()
        generator.root = None
        generator.manifest = {
            "schema": "hybrid-terrain-lmm-model/v1",
            "canonical_selection_accepted": True,
        }
        terrain = load_scene_terrain("hills")
        native_model = mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML))
        fail = [False]

        def converter(*arguments):
            if fail[0]:
                raise ValueError("forced canonical root failure")
            qpos = _pose_converter(*arguments)
            qpos[7:] = 0.0
            return qpos

        matcher = HybridMatcher(
            corpus,
            generator,
            terrain.authority,
            native_model=native_model,
            pose_converter=converter,
        )
        fail[0] = True

        receipt = run_mujoco_headless_smoke(
            matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1_000
        )

        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["mujoco_forward_count"], 0)
        self.assertEqual(receipt["failure_frame"], 0)
        self.assertEqual(
            receipt["failure_reason"], "ValueError: forced canonical root failure"
        )
        self.assertEqual(receipt["crash_count"], 1)
        self.assertFalse(receipt["completed_without_exception"])

    def test_file_scene_spawn_yaw_and_hashes_initialize_and_reset_matcher(self):
        from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher

        from tests.python.test_hybrid_terrain_lmm_runtime import (
            _corpus,
            _Generator,
            _pose_converter,
        )

        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "spawned"
            scene.mkdir()
            heights = np.zeros((2, 2), "<f4")
            terrain_payload = (
                struct.pack("<4sIIIffff", b"G1HF", 2, 2, 2, 0.0, 0.0, 1.0, 0.0)
                + heights.tobytes()
            )
            (scene / "terrain.bin").write_bytes(terrain_payload)
            scene_payload = (
                json.dumps(
                    {
                        "schema": "g1-terrain-scene/v1",
                        "id": "spawned",
                        "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
                        "heightfield": {
                            "path": "terrain.bin",
                            "sha256": hashlib.sha256(terrain_payload).hexdigest(),
                        },
                        "spawn": {
                            "position": [1.25, 0.0, 3.5],
                            "yaw_radians": 0.4,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            (scene / "scene.json").write_bytes(scene_payload)

            adapter = load_scene_terrain(scene)
            matcher = HybridMatcher(
                _corpus(),
                _Generator(),
                adapter.authority,
                pose_converter=_pose_converter,
                initial_root_xy=adapter.spawn_native_xy,
                initial_heading=adapter.spawn_heading,
            )

            np.testing.assert_allclose(
                matcher.state.root_position_world[:2], (1.25, -3.5)
            )
            self.assertAlmostEqual(matcher.state.heading, 0.4)
            matcher.step(CommandState(1.0, 0.5), dt=0.04)
            reset = matcher.reset()
            np.testing.assert_allclose(reset.root_position_world[:2], (1.25, -3.5))
            self.assertAlmostEqual(reset.heading, 0.4)
            self.assertEqual(
                adapter.scene_json_sha256, hashlib.sha256(scene_payload).hexdigest()
            )
            self.assertEqual(
                adapter.terrain_sha256, hashlib.sha256(terrain_payload).hexdigest()
            )

    def test_xml_asset_identity_binds_nested_include_files_and_their_assets(self):
        import mujoco

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir(parents=True)
            (nested / "part.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n",
                encoding="utf-8",
            )
            included = nested / "part.xml"
            included.write_text(
                '<mujocoinclude><asset><mesh name="part" file="part.obj"/>'
                '</asset><worldbody><geom type="mesh" mesh="part"/></worldbody>'
                "</mujocoinclude>",
                encoding="utf-8",
            )
            main_xml = root / "main.xml"
            main_xml.write_text(
                '<mujoco><include file="nested/part.xml"/></mujoco>',
                encoding="utf-8",
            )

            model = mujoco.MjModel.from_xml_path(str(main_xml))
            before, inventory = viewer_module._g1_xml_asset_identity(main_xml)

            self.assertGreater(model.ngeom, 0)
            self.assertIn("nested/part.xml", inventory)
            self.assertIn("nested/part.obj", inventory)
            included.write_text(
                '<mujocoinclude><asset><mesh name="renamed" file="part.obj"/>'
                '</asset><worldbody><geom type="mesh" mesh="renamed"/></worldbody>'
                "</mujocoinclude>",
                encoding="utf-8",
            )
            after, _ = viewer_module._g1_xml_asset_identity(main_xml)
            self.assertNotEqual(before, after)

    def test_included_compiler_meshdir_resolves_from_main_xml_directory(self):
        import mujoco

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            meshes = root / "meshes"
            nested.mkdir()
            meshes.mkdir()
            mesh_payload = (
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n"
            )
            (meshes / "part.obj").write_text(mesh_payload, encoding="utf-8")
            (nested / "part.xml").write_text(
                '<mujocoinclude><compiler meshdir="meshes"/>'
                '<asset><mesh name="part" file="part.obj"/></asset>'
                '<worldbody><geom type="mesh" mesh="part"/></worldbody>'
                "</mujocoinclude>",
                encoding="utf-8",
            )
            main_xml = root / "main.xml"
            main_xml.write_text(
                '<mujoco><include file="nested/part.xml"/></mujoco>',
                encoding="utf-8",
            )

            model = mujoco.MjModel.from_xml_path(str(main_xml))
            _, inventory = viewer_module._g1_xml_asset_identity(main_xml)

            self.assertGreater(model.ngeom, 0)
            self.assertIn("nested/part.xml", inventory)
            self.assertIn("meshes/part.obj", inventory)

    def test_scene_adapter_uses_one_heightfield_for_query_and_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "synthetic"
            scene.mkdir()
            heights = np.asarray(((0.0, 0.1, 0.2), (0.3, 0.8, 0.5)), "<f4")
            (scene / "terrain.bin").write_bytes(
                struct.pack("<4sIIIffff", b"G1HF", 2, 3, 2, -1.0, 2.0, 0.5, -0.2)
                + heights.tobytes()
            )

            adapter = load_scene_terrain(scene)

            self.assertAlmostEqual(adapter.authority.height_at((-0.5, -2.0)), 0.1)
            # The query uses the same fixed a-d cell diagonal as native_mesh,
            # rather than a bilinear surface the renderer does not contain.
            self.assertAlmostEqual(
                adapter.authority.height_at((-0.625, -2.125)), 0.25, places=6
            )
            np.testing.assert_array_equal(adapter.source_heights, heights)
            rendered = adapter.render_heights()
            np.testing.assert_array_equal(rendered, heights)
            vertices, faces = adapter.native_mesh()
            normals = np.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]],
                vertices[faces[:, 2]] - vertices[faces[:, 0]],
            )
            self.assertTrue(np.all(normals[:, 2] > 0.0))

    def test_keyboard_wasd_space_and_reset_are_level_safe(self):
        keys = KeyboardCommandSource()
        keys.press("w")
        keys.press("a")
        self.assertEqual(keys.snapshot()[0], CommandState(1.0, 1.0))
        keys.release("a")
        keys.press("s")
        self.assertEqual(keys.snapshot()[0], CommandState())
        keys.press(" ")
        self.assertEqual(keys.snapshot()[0], CommandState())
        keys.release(" ")
        keys.press("r")
        command, reset, stopped = keys.snapshot()
        self.assertEqual(command, CommandState())
        self.assertTrue(reset)
        self.assertFalse(stopped)
        self.assertFalse(keys.snapshot()[1], "reset must be edge-triggered")
        keys.press("r")
        self.assertFalse(keys.snapshot()[1], "key repeat must not retrigger reset")
        keys.release("r")
        keys.press("r")
        self.assertTrue(keys.snapshot()[1], "a new physical press must reset")

    def test_arrow_keys_map_to_level_safe_drive_commands(self):
        keyboard = SimpleNamespace(
            Key=SimpleNamespace(
                up=object(),
                down=object(),
                left=object(),
                right=object(),
                space=object(),
                esc=object(),
            )
        )
        keys = KeyboardCommandSource()

        for special, expected in (
            (keyboard.Key.up, CommandState(speed=1.0)),
            (keyboard.Key.down, CommandState(speed=-1.0)),
            (keyboard.Key.left, CommandState(steering=1.0)),
            (keyboard.Key.right, CommandState(steering=-1.0)),
        ):
            token = viewer_module._keyboard_command_token(special, keyboard)
            keys.press(token)
            self.assertEqual(keys.snapshot()[0], expected)
            keys.release(token)
            self.assertEqual(keys.snapshot()[0], CommandState())

        up_token = viewer_module._keyboard_command_token(keyboard.Key.up, keyboard)
        down_token = viewer_module._keyboard_command_token(keyboard.Key.down, keyboard)
        self.assertNotEqual(up_token, "w")
        self.assertNotEqual(down_token, "s")
        keys.press(up_token)
        keys.press(down_token)
        self.assertEqual(keys.snapshot()[0], CommandState())
        keys.release(up_token)
        keys.release(down_token)

        left_token = viewer_module._keyboard_command_token(keyboard.Key.left, keyboard)
        right_token = viewer_module._keyboard_command_token(
            keyboard.Key.right, keyboard
        )
        keys.press(left_token)
        keys.press(right_token)
        self.assertEqual(keys.snapshot()[0], CommandState())
        keys.release(left_token)
        keys.release(right_token)

        keys.press(up_token)
        keys.press("w")
        self.assertEqual(keys.snapshot()[0], CommandState(speed=1.0))
        keys.release(up_token)
        self.assertEqual(keys.snapshot()[0], CommandState(speed=1.0))
        keys.press(up_token)
        keys.release("w")
        self.assertEqual(keys.snapshot()[0], CommandState(speed=1.0))
        keys.release(up_token)
        self.assertEqual(keys.snapshot()[0], CommandState())

        keys.press(up_token)
        keys.press(up_token)
        keys.release(up_token)
        self.assertEqual(keys.snapshot()[0], CommandState())

    def test_space_hard_stop_overrides_an_active_gamepad(self):
        keyboard = SimpleNamespace(
            Key=SimpleNamespace(
                up=object(),
                down=object(),
                left=object(),
                right=object(),
                space=object(),
            )
        )
        keys = KeyboardCommandSource()
        space_token = viewer_module._keyboard_command_token(
            keyboard.Key.space, keyboard
        )
        self.assertEqual(space_token, " ")
        keys.press(space_token)

        command = viewer_module._select_control_command(
            keyboard_command=keys.snapshot()[0],
            gamepad_command=CommandState(speed=1.0, steering=0.7),
            gamepad_connected=True,
            hard_stop=keys.hard_stop_active(),
        )

        self.assertEqual(command, CommandState())

    def test_listener_press_does_not_consume_reset_edge(self):
        keys = KeyboardCommandSource()

        self.assertIsNone(handle_key_press(keys, "r", escape=False))

        self.assertTrue(keys.snapshot()[1])

    def test_evdev_deadzone_clamping_and_disconnect_neutral_fallback(self):
        source = EvdevCommandSource(_Axes((-1.2, 0.19)), deadzone=0.2)
        self.assertEqual(source.snapshot(), CommandState(speed=1.0, steering=0.0))
        source = EvdevCommandSource(_Axes((-0.5, 0.6)), deadzone=0.2)
        self.assertEqual(source.snapshot(), CommandState(speed=0.5, steering=0.6))
        source = EvdevCommandSource(_Axes(OSError("disconnected")), deadzone=0.2)
        self.assertEqual(source.snapshot(), CommandState())
        self.assertFalse(source.connected)

    def test_absent_evdev_device_opens_as_neutral_without_error(self):
        def unavailable(_path):
            raise OSError("device is gone")

        source = optional_evdev_source("/dev/input/event99", factory=unavailable)

        self.assertEqual(source.snapshot(), CommandState())
        self.assertFalse(source.connected)

    def test_overlay_exposes_required_runtime_fields(self):
        state = SimpleNamespace(
            family="slope",
            range_index=12,
            row=345,
            search_distance=0.125,
            terrain_features=(0.0, 0.1, 0.2, 0.3),
            decode_count=91,
            fallback_count=2,
            joint_clamp_count=1,
            max_joint_clamp_magnitude=0.0375,
            candidate_limit_rejection_count=3,
            first_candidate_limit_rejection_row=168,
            max_candidate_limit_rejections_per_step=2,
            support_height=0.18,
            support_status="SUPPORTED",
            pose_source="learned",
            searchable_row_count=500_000,
            searchable_family_counts=(("flat", 125_000), ("slope", 125_000)),
            total_searchable_row_count=3_955_117,
            search_scope="diagnostic-stratified-cap",
            transition_penalty=0.1,
        )
        title, body = overlay_text(
            state,
            paused=False,
            scene_evidence_status="diagnostic-generated",
            display_postprocessor_identity={
                "diagnostic_display_postprocessor": (
                    "existing-pose-inertializer-repair-foot-lock/v1"
                ),
                "inertialization_halflife_s": 0.10,
                "pose_repair_count": 7,
                "foot_lock_accept_count": 5,
                "foot_lock_bypass_count": 2,
                "raw_repair_failure_count": 3,
                "last_reason": "raw source pose repair rejected",
            },
        )
        self.assertIn("HYBRID TERRAIN LMM POC", title)
        self.assertNotIn("EXACT SEARCH", title)
        self.assertIn(
            "PoseInertializer + G1TerrainPoseRepair + G1TerrainFootLock", title
        )
        for value in (
            "family slope",
            "range 12",
            "row 345",
            "distance 0.125000",
            "terrain [0.000, 0.100, 0.200, 0.300]",
            "decode 91",
            "fallback 2",
            "canonical 1",
            "clamped 1",
            "max 0.037500",
            "native candidate rejects 3",
            "first 168",
            "max/step 2",
            "support SUPPORTED",
            "height 0.180",
            "diagnostic capped rows 500000/3955117",
            "flat=125000",
            "slope=125000",
            "transition penalty 0.100",
            "PoseInertializer + G1TerrainPoseRepair + G1TerrainFootLock",
            "0.10s half-life",
            "repair accepted 7",
            "lock accept/bypass 5/2",
            "raw repair failures 3",
            "last reason raw source pose repair rejected",
        ):
            self.assertIn(value, body)

        state.search_scope = "full-range-safe-corpus"
        state.searchable_row_count = state.total_searchable_row_count
        title, body = overlay_text(
            state, paused=False, scene_evidence_status="diagnostic-generated"
        )
        self.assertIn("DIAGNOSTIC", title)
        self.assertNotIn("EXACT SEARCH", title)

        title, body = overlay_text(
            state,
            paused=False,
            scene_evidence_status="authenticated-indexed",
            generator_accepted=True,
        )
        self.assertIn("EXACT SEARCH", title)
        self.assertIn("full exact rows 3955117/3955117", body)
        title, body = overlay_text(
            state,
            paused=False,
            scene_evidence_status="authenticated-indexed",
            generator_accepted=True,
            search_backend_identity="single-gpu-full-row-fp32:cuda:5",
            last_search_elapsed_ms=7.25,
            first_runtime_search_elapsed_ms=11.5,
        )
        self.assertIn("DIAGNOSTIC", title)
        self.assertNotIn("EXACT SEARCH", title)
        self.assertIn("search backend single-gpu-full-row-fp32:cuda:5", body)
        self.assertIn("last search 7.250 ms", body)
        self.assertIn("first runtime search 11.500 ms", body)
        title, _ = overlay_text(
            state,
            paused=False,
            scene_evidence_status="authenticated-indexed",
            generator_accepted=False,
        )
        self.assertIn("DIAGNOSTIC", title)
        self.assertNotIn("EXACT SEARCH", title)

    def test_overlay_distinguishes_complete_mechanical_view_from_cap(self):
        state = SimpleNamespace(
            family="slope",
            range_index=12,
            row=345,
            search_distance=0.125,
            terrain_features=(0.0, 0.1, 0.2, 0.3),
            decode_count=91,
            fallback_count=0,
            joint_clamp_count=0,
            diagnostic_pose_rejection_count=3,
            unsupported_hold_count=4,
            max_joint_clamp_magnitude=0.0,
            candidate_limit_rejection_count=0,
            first_candidate_limit_rejection_row=None,
            max_candidate_limit_rejections_per_step=0,
            support_height=0.18,
            support_status="SUPPORTED",
            pose_source="learned",
            searchable_row_count=90,
            searchable_family_counts=(("slope", 90),),
            total_searchable_row_count=100,
            search_scope="diagnostic-mechanically-filtered-corpus",
            transition_penalty=0.1,
        )
        overlay_arguments = {
            "paused": False,
            "scene_evidence_status": "authenticated-indexed",
            "generator_accepted": True,
            "search_backend_identity": "cpu-ckdtree-mechanically-filtered-exact",
            "diagnostic_mechanical_retained_searchable_row_count": 90,
            "diagnostic_mechanical_clearance_bounds_m": (0.4, 1.2),
        }

        def render_overlay():
            try:
                return overlay_text(state, **overlay_arguments)
            except TypeError as error:
                self.fail(f"mechanical overlay metadata is unavailable: {error}")

        title, body = render_overlay()

        self.assertIn("DIAGNOSTIC", title)
        self.assertNotIn("diagnostic capped rows", body)
        self.assertIn("mechanically filtered rows 90/100", body)
        self.assertIn("clearance 0.400..1.200 m", body)
        self.assertIn("diagnostic pose rejects 3", body)
        self.assertIn("unsupported holds 4", body)

        state.search_scope = "diagnostic-mechanically-filtered-cap"
        state.searchable_row_count = 25
        title, body = render_overlay()

        self.assertIn("DIAGNOSTIC", title)
        self.assertIn("diagnostic capped rows 25/90", body)
        self.assertIn("mechanically filtered rows 90/100", body)

    def test_generator_acceptance_requires_verified_all_row_refit(self):
        selection = SimpleNamespace(
            config=SimpleNamespace(fit_all_rows=False),
            selection_provenance_verified=False,
            canonical_selection_verified=True,
            manifest={"canonical_selection_accepted": True},
        )
        final = SimpleNamespace(
            config=SimpleNamespace(fit_all_rows=True),
            selection_provenance_verified=True,
            canonical_selection_verified=True,
            manifest={"canonical_selection_accepted": True},
        )

        self.assertFalse(
            viewer_module._generator_acceptance_status(selection)["accepted"]
        )
        self.assertTrue(viewer_module._generator_acceptance_status(final)["accepted"])
        matcher = SimpleNamespace(
            search_acceptance_eligible=True,
            generator=selection,
            search_backend_identity="cpu-ckdtree-exact",
        )
        terrain = SimpleNamespace(scene_authenticated=True)
        self.assertIn("DIAGNOSTIC", viewer_module._runtime_label(matcher, terrain))
        matcher.generator = final
        matcher.search_backend_identity = "single-gpu-full-row-fp32:cuda:5"
        self.assertIn("DIAGNOSTIC", viewer_module._runtime_label(matcher, terrain))

    def test_formal_authority_gate_is_frozen_to_delivered_artifacts(self):
        identity = {
            "cache_manifest_sha256": (
                "084c168b473e730ec24419a49f4be1526226e5f95a2dd81ae4d367c729889cdb"
            ),
            "model_manifest_sha256": (
                "f200db342dd514df6deb6203f3be014054efd4f9038dc0486cc3a4b274a6d826"
            ),
            "g1_xml_sha256": (
                "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376"
            ),
            "g1_asset_inventory_sha256": (
                "47dcad0d533434233e7769486ae79074751be9eeb583af39f38e219c96c71a28"
            ),
            "corpus_row_count": 3_973_057,
            "corpus_range_count": 15_833,
            "corpus_train_row_count": 3_575_813,
            "corpus_evaluation_row_count": 397_244,
            "generator_loader_authority": True,
            "total_safe_row_count": 3_957_224,
            "total_safe_family_counts": {
                "flat": 18_174,
                "curb": 440_481,
                "slope": 462_393,
                "stair": 3_036_176,
            },
        }
        self.assertTrue(viewer_module._formal_artifact_authorities(identity))
        for key in tuple(identity):
            with self.subTest(key=key):
                changed = dict(identity)
                changed[key] = "0" * 64
                self.assertFalse(viewer_module._formal_artifact_authorities(changed))

    def test_generator_loader_authority_requires_exact_loaded_class(self):
        from mm_sonic.hybrid_terrain_lmm_training import HybridGenerator

        model_sha = "f" * 64
        loaded = object.__new__(HybridGenerator)
        loaded.root = Path("model")
        loaded.manifest_sha256 = model_sha

        spoof_type = type(
            "HybridGenerator",
            (),
            {"__module__": "mm_sonic.hybrid_terrain_lmm_training"},
        )
        spoof = spoof_type()
        spoof.root = Path("model")
        spoof.manifest_sha256 = model_sha

        self.assertTrue(
            viewer_module._generator_has_loader_authority(loaded, model_sha)
        )
        self.assertFalse(
            viewer_module._generator_has_loader_authority(spoof, model_sha)
        )

    def test_fixed_accumulator_steps_matcher_at_25_hz(self):
        accumulator = FixedRateAccumulator(step_hz=25.0)
        ticks = []
        self.assertEqual(accumulator.advance(1.0 / 60.0, ticks.append), 0)
        self.assertEqual(accumulator.advance(1.0 / 60.0, ticks.append), 0)
        self.assertEqual(accumulator.advance(1.0 / 60.0, ticks.append), 1)
        self.assertEqual(ticks, [0.04])

    def test_interactive_receipt_binds_identity_and_is_not_acceptance_evidence(self):
        import mujoco

        state = SimpleNamespace(
            qpos=np.zeros(36, np.float64),
            family="slope",
            range_index=1,
            row=2,
            search_distance=0.0,
            terrain_features=np.zeros(4),
            decode_count=3,
            fallback_count=0,
            joint_clamp_count=0,
            max_joint_clamp_magnitude=0.0,
            support_height=0.0,
            support_status="SUPPORTED",
            pose_source="learned",
            searchable_row_count=6,
            searchable_family_counts=(("slope", 6),),
            total_searchable_row_count=6,
            search_scope="full-range-safe-corpus",
            transition_penalty=0.1,
        )
        matcher = SimpleNamespace(
            corpus=object(),
            state=state,
            step=lambda _command, dt: state,
            reset=lambda: state,
            search_scope="full-range-safe-corpus",
            search_acceptance_eligible=True,
            search_backend_identity="single-gpu-full-row-fp32:cuda:5",
            last_search_elapsed_ms=7.25,
            warm_search_elapsed_ms=11.5,
            diagnostic_mechanical_retained_searchable_row_count=5,
            diagnostic_mechanical_clearance_bounds_m=(0.4, 1.2),
        )
        terrain = replace(
            load_scene_terrain("hills"),
            scene_authenticated=True,
            scene_evidence_status="authenticated-indexed",
        )
        listener = SimpleNamespace(
            start=lambda: None, stop=lambda: None, join=lambda timeout: None
        )
        keyboard = SimpleNamespace(
            Key=SimpleNamespace(esc=object()),
            Listener=lambda **_kwargs: listener,
        )

        overlay_texts = []

        class FakeViewer:
            def __init__(self):
                self.cam = SimpleNamespace(
                    distance=0.0,
                    azimuth=0.0,
                    elevation=0.0,
                    lookat=np.zeros(3, np.float64),
                )
                self.user_scn = SimpleNamespace(flags=np.zeros(1_000, np.int32))

            def __enter__(self):
                return self

            def __exit__(self, *_arguments):
                return False

            def is_running(self):
                return True

            def lock(self):
                return contextlib.nullcontext()

            def set_texts(self, texts):
                overlay_texts.append((texts[2], texts[3]))

            def sync(self):
                return None

        data = SimpleNamespace(qpos=np.zeros(36, np.float64))
        identity = {
            "cache_manifest_sha256": "c" * 64,
            "scene_authentication_current": True,
            "search_backend_identity": "single-gpu-full-row-fp32:cuda:5",
            "last_search_elapsed_ms": None,
            "first_runtime_search_elapsed_ms": None,
        }
        with (
            mock.patch.dict(os.environ, {"DISPLAY": ":99"}),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_viewer._load_keyboard_module",
                return_value=keyboard,
            ),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_viewer.optional_evdev_source",
                return_value=SimpleNamespace(
                    connected=False, snapshot=lambda: CommandState()
                ),
            ),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_viewer.build_viewer_model",
                return_value=object(),
            ) as build_model,
            mock.patch.object(mujoco, "MjData", return_value=data),
            mock.patch.object(mujoco, "mj_forward"),
            mock.patch("mujoco.viewer.launch_passive", return_value=FakeViewer()),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_viewer.overlay_text",
                wraps=viewer_module.overlay_text,
            ) as rendered_overlay,
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_viewer._runtime_identity",
                return_value=identity,
            ) as runtime_identity,
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_viewer.scene_authentication_is_current",
                return_value=False,
            ) as scene_current,
        ):
            receipt = viewer_module.run_interactive(
                matcher, terrain, g1_xml=DEFAULT_G1_XML, max_render_frames=1
            )

        runtime_identity.assert_called_once_with(matcher, terrain, DEFAULT_G1_XML)
        build_model.assert_called_once_with(
            DEFAULT_G1_XML, terrain, interactive_visuals=True
        )
        self.assertEqual(
            rendered_overlay.call_args.kwargs.get(
                "diagnostic_mechanical_retained_searchable_row_count"
            ),
            5,
        )
        self.assertEqual(
            rendered_overlay.call_args.kwargs.get(
                "diagnostic_mechanical_clearance_bounds_m"
            ),
            (0.4, 1.2),
        )
        scene_current.assert_called_once_with(matcher.corpus, terrain)
        self.assertNotIn("EXACT SEARCH", overlay_texts[0][0])
        self.assertIn(
            "search backend single-gpu-full-row-fp32:cuda:5", overlay_texts[0][1]
        )
        self.assertIn("last search 7.250 ms", overlay_texts[0][1])
        self.assertIn("first runtime search 11.500 ms", overlay_texts[0][1])
        self.assertEqual(receipt["identity"], identity)
        self.assertEqual(identity["last_search_elapsed_ms"], 7.25)
        self.assertEqual(identity["first_runtime_search_elapsed_ms"], 11.5)
        self.assertEqual(
            receipt["overlay_scene_evidence_status"],
            "diagnostic-authentication-not-current",
        )
        self.assertEqual(
            receipt["evidence_status"], "interactive-diagnostic-not-acceptance"
        )
        self.assertFalse(receipt["acceptance_eligible"])

    def test_interactive_postprocessor_runs_on_fixed_ticks_and_resets_in_order(self):
        import mujoco

        events: list[str] = []
        state = SimpleNamespace(
            qpos=np.arange(36, dtype=np.float64),
            family="flat",
            range_index=1,
            row=2,
            search_distance=0.0,
            terrain_features=np.zeros(4),
            decode_count=0,
            fallback_count=0,
            joint_clamp_count=0,
            max_joint_clamp_magnitude=0.0,
            support_height=0.0,
            support_status="SUPPORTED",
            pose_source="canonical",
            searchable_row_count=3,
            searchable_family_counts=(("flat", 3),),
            total_searchable_row_count=3,
            search_scope="diagnostic",
            transition_penalty=0.1,
        )
        matcher = SimpleNamespace(
            artifacts=SimpleNamespace(contacts=np.asarray([[False, False]] * 3)),
            corpus=object(),
            state=state,
            step=lambda _command, dt: events.append("matcher.step") or state,
            reset=lambda: events.append("matcher.reset") or state,
            search_scope="diagnostic",
            search_acceptance_eligible=False,
        )
        terrain = load_scene_terrain("hills")
        postprocessor_identity = {
            "diagnostic_display_postprocessor": (
                "existing-pose-inertializer-repair-foot-lock/v1"
            ),
            "inertialization_halflife_s": 0.10,
            "pose_repair_count": 3,
            "foot_lock_accept_count": 2,
            "foot_lock_bypass_count": 1,
            "raw_repair_failure_count": 4,
            "last_reason": "lock rejected",
        }
        postprocessor = SimpleNamespace(
            reset=lambda: events.append("postprocessor.reset"),
            step=lambda qpos, **_kwargs: (
                events.append("postprocessor.step") or np.asarray(qpos) + 1.0
            ),
            identity=lambda: dict(postprocessor_identity),
        )
        keys = SimpleNamespace(
            snapshot=lambda: (CommandState(), True, False),
            hard_stop_active=lambda: False,
            press=lambda _key: None,
            release=lambda _key: None,
        )
        listener = SimpleNamespace(
            start=lambda: None, stop=lambda: None, join=lambda timeout: None
        )
        keyboard = SimpleNamespace(
            Key=SimpleNamespace(esc=object()),
            Listener=lambda **_kwargs: listener,
        )

        class FakeAccumulator:
            def __init__(self, *, step_hz):
                self.step_hz = step_hz

            def advance(self, _elapsed, callback):
                callback(1.0 / self.step_hz)
                return 1

        overlay_texts: list[tuple[str, str]] = []

        class FakeViewer:
            cam = SimpleNamespace(
                distance=0.0,
                azimuth=0.0,
                elevation=0.0,
                lookat=np.zeros(3),
            )
            user_scn = SimpleNamespace(flags=np.zeros(1_000, np.int32))

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def is_running(self):
                return True

            def lock(self):
                return contextlib.nullcontext()

            def set_texts(self, texts):
                overlay_texts.append((texts[2], texts[3]))

            def sync(self):
                return None

        data = SimpleNamespace(qpos=np.zeros(36, dtype=np.float64))
        model = object()
        identity = {
            "scene_authentication_current": False,
            "generator_acceptance_status": {"accepted": False},
        }
        factory_calls: list[tuple[object, object, object]] = []

        def factory(*args):
            factory_calls.append(args)
            return postprocessor

        with (
            mock.patch.dict(os.environ, {"DISPLAY": ":99"}),
            mock.patch.object(
                viewer_module, "KeyboardCommandSource", return_value=keys
            ),
            mock.patch.object(viewer_module, "FixedRateAccumulator", FakeAccumulator),
            mock.patch.object(
                viewer_module, "_load_keyboard_module", return_value=keyboard
            ),
            mock.patch.object(
                viewer_module,
                "optional_evdev_source",
                return_value=SimpleNamespace(connected=False, snapshot=CommandState),
            ),
            mock.patch.object(viewer_module, "build_viewer_model", return_value=model),
            mock.patch.object(mujoco, "MjData", return_value=data),
            mock.patch.object(
                mujoco,
                "mj_forward",
                side_effect=lambda *_args: events.append("mj_forward"),
            ),
            mock.patch("mujoco.viewer.launch_passive", return_value=FakeViewer()),
            mock.patch.object(
                viewer_module, "_runtime_identity", return_value=identity
            ),
        ):
            receipt = viewer_module.run_interactive(
                matcher,
                terrain,
                max_render_frames=1,
                display_postprocessor_factory=factory,
            )

        self.assertEqual(factory_calls, [(model, matcher, terrain)])
        self.assertEqual(
            events,
            [
                "matcher.reset",
                "postprocessor.reset",
                "matcher.step",
                "postprocessor.step",
                "mj_forward",
            ],
        )
        np.testing.assert_array_equal(data.qpos, state.qpos + 1.0)
        self.assertEqual(receipt["identity"]["pose_repair_count"], 3)
        self.assertIn(
            "PoseInertializer + G1TerrainPoseRepair + G1TerrainFootLock",
            overlay_texts[0][0],
        )
        self.assertIn("0.10s half-life", overlay_texts[0][0])
        self.assertIn("repair accepted 3", overlay_texts[0][1])
        self.assertIn("lock accept/bypass 2/1", overlay_texts[0][1])
        self.assertIn("raw repair failures 4", overlay_texts[0][1])
        self.assertIn("last reason lock rejected", overlay_texts[0][1])

    def test_failed_reset_returns_original_runtime_unchanged(self):
        runtime = SimpleNamespace(marker=object())

        def fail():
            raise RuntimeError("model unavailable")

        returned, error = reset_without_mutation_on_failure(runtime, fail)
        self.assertIs(returned, runtime)
        self.assertIsInstance(error, RuntimeError)

    def test_cli_has_smoke_and_view_with_scene(self):
        parser = build_parser()
        smoke = parser.parse_args(
            [
                "smoke",
                "--cache",
                "cache",
                "--model",
                "model",
                "--frames",
                "1000",
                "--receipt",
                "smoke.json",
            ]
        )
        view = parser.parse_args(
            [
                "view",
                "--cache",
                "cache",
                "--model",
                "model",
                "--scene",
                "hills",
                "--search-device",
                "cuda:5",
            ]
        )
        self.assertEqual(smoke.command, "smoke")
        self.assertEqual(smoke.frames, 1000)
        self.assertEqual(smoke.receipt, Path("smoke.json"))
        self.assertEqual(smoke.scene, "ramp-10-up-down")
        self.assertIsNone(smoke.search_rows)
        self.assertEqual(smoke.transition_penalty, 0.1)
        self.assertFalse(hasattr(smoke, "search_device"))
        self.assertEqual(view.command, "view")
        self.assertEqual(view.scene, "hills")
        self.assertEqual(view.search_device, "cuda:5")
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(
                [
                    "smoke",
                    "--cache",
                    "cache",
                    "--model",
                    "model",
                    "--search-device",
                    "cuda:5",
                ]
            )

    def test_receipt_output_is_exclusive_and_never_overwrites_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "receipt.json"
            target.write_bytes(b"immutable prior evidence\n")
            with (
                mock.patch(
                    "mm_sonic.hybrid_terrain_lmm_viewer._load_matcher",
                    return_value=(object(), object()),
                ),
                mock.patch(
                    "mm_sonic.hybrid_terrain_lmm_viewer.run_mujoco_headless_smoke",
                    return_value={"accepted": True},
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(FileExistsError),
            ):
                main(
                    [
                        "smoke",
                        "--cache",
                        "cache",
                        "--model",
                        "model",
                        "--receipt",
                        str(target),
                    ]
                )
            self.assertEqual(target.read_bytes(), b"immutable prior evidence\n")

    def test_receipt_output_is_canonical_read_only_and_fsyncs_file_and_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "receipt.json"
            receipt = {"accepted": True, "nested": {"value": 1}}
            stdout = io.StringIO()
            events = []
            real_fchmod = os.fchmod
            real_fsync = os.fsync

            def record_fchmod(descriptor, mode):
                events.append(("fchmod", descriptor, mode))
                return real_fchmod(descriptor, mode)

            def record_fsync(descriptor):
                events.append(("fsync", descriptor))
                return real_fsync(descriptor)

            with (
                mock.patch(
                    "mm_sonic.hybrid_terrain_lmm_viewer._load_matcher",
                    return_value=(object(), object()),
                ),
                mock.patch(
                    "mm_sonic.hybrid_terrain_lmm_viewer.run_mujoco_headless_smoke",
                    return_value=receipt,
                ),
                mock.patch.object(os, "fchmod", side_effect=record_fchmod),
                mock.patch.object(os, "fsync", side_effect=record_fsync) as fsync,
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(
                    main(
                        [
                            "smoke",
                            "--cache",
                            "cache",
                            "--model",
                            "model",
                            "--receipt",
                            str(target),
                        ]
                    ),
                    0,
                )

            expected = _canonical_json(receipt)
            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode) & 0o222, 0)
            self.assertGreaterEqual(fsync.call_count, 2)
            self.assertEqual(events[0][0], "fchmod")
            self.assertEqual(
                [path.name for path in target.parent.iterdir()], [target.name]
            )
            self.assertEqual(json.loads(stdout.getvalue()), receipt)


if __name__ == "__main__":
    unittest.main()
