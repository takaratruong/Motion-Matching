from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from mm_sonic.joints import ContractError
from mm_sonic.terrain_oracle.audit import (
    structural_model_sha256,
    terrain_query_sha256,
)
from mm_sonic.terrain_oracle.canonical import (
    CanonicalTerrainMesh,
    TerrainBinding,
)
from mm_sonic.terrain_oracle.contact import CanonicalMeshQuery
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.storage import read_clip, write_clip, write_mesh
from tests.python.terrain_oracle_test_utils import synthetic_canonical_clip


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/g1_29dof_rev_1_0.xml"
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _model_authority(model_path: Path, model_root: Path) -> dict[str, object]:
    """Build independent exact authority for one synthetic model VFS."""

    import mujoco

    root = model_root.resolve()
    path = model_path.resolve()
    directories: list[str] = []
    files: list[dict[str, object]] = []
    for entry in sorted(
        root.rglob("*"),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative_path = entry.relative_to(root).as_posix()
        if entry.is_dir():
            directories.append(relative_path)
            continue
        payload = entry.read_bytes()
        files.append(
            {
                "relative_path": relative_path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    tree_evidence = {"directories": directories, "files": files}
    root_payload = path.read_bytes()
    model = mujoco.MjModel.from_xml_path(str(path))
    return {
        "path": str(path),
        "size_bytes": len(root_payload),
        "sha256": hashlib.sha256(root_payload).hexdigest(),
        "structural_sha256": structural_model_sha256(model),
        "dependency_vfs": {
            "root_path": str(root),
            "root_relative_path": path.relative_to(root).as_posix(),
            "directory_count": len(directories),
            "file_count": len(files),
            "size_bytes": sum(int(item["size_bytes"]) for item in files),
            "tree_sha256": hashlib.sha256(
                _canonical_json(tree_evidence)
            ).hexdigest(),
        },
    }


@unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
class RenderRequestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        import mujoco
        from mm_sonic.terrain_oracle.render_media import FIXED_RENDER_CONFIG

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        clips = self.root / "clips"
        meshes = self.root / "meshes"
        clips.mkdir()
        meshes.mkdir()
        transform = RigidTransform(
            translation_world=np.zeros(3, dtype=np.float32),
            quaternion_world_from_local_wxyz=np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
        )
        vertices = np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [-2.0, -2.0, -0.05],
                [2.0, -2.0, -0.05],
                [2.0, 2.0, -0.05],
                [-2.0, 2.0, -0.05],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 6, 5], [4, 7, 6],
                [0, 4, 5], [0, 5, 1],
                [1, 5, 6], [1, 6, 2],
                [2, 6, 7], [2, 7, 3],
                [3, 7, 4], [3, 4, 0],
            ],
            dtype=np.int32,
        )
        mesh = CanonicalTerrainMesh(
            vertices_local=vertices,
            faces=faces,
            valid_faces=np.ones(len(faces), dtype=np.bool_),
            source_asset_sha256="a" * 64,
        )
        mesh_record = write_mesh(meshes, mesh)
        terrain = TerrainBinding(
            asset_path="/audited/source/terrain.obj",
            asset_size_bytes=123,
            asset_sha256=mesh.source_asset_sha256,
            asset_license_id="CC0-1.0",
            mesh_sha256=mesh_record.sha256,
            world_from_terrain=transform,
            validity_mask_path=None,
        )
        original = synthetic_canonical_clip(frames=4)
        root_position = np.array(original.root_position_world, copy=True)
        root_position[:, 2] = 0.793
        contact = np.array(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        clip = replace(
            original,
            root_position_world=root_position,
            contact=contact,
            terrain=terrain,
            action_tags=("walk", "flat"),
        )
        clip_record = write_clip(clips, clip)
        clip_path = (clips / f"{clip_record.sha256}.npz").resolve()
        mesh_path = (meshes / f"{mesh_record.sha256}.npz").resolve()
        interval_key = f"{clip_record.sha256}:0:4"
        self.request = {
            "schema": "terrain-oracle-render-request/v1",
            "kind": "accepted_interval",
            "interval_keys": [interval_key],
            "inputs": [
                {
                    "interval_key": interval_key,
                    "interval": [0, 4],
                    "clip": {
                        "path": str(clip_path),
                        "size_bytes": clip_path.stat().st_size,
                        "sha256": clip_record.sha256,
                        "clip_id": clip.clip_id,
                        "source_sha256": clip.source.source_sha256,
                        "frame_count": clip.frame_count,
                    },
                    "terrain_mesh": {
                        "path": str(mesh_path),
                        "size_bytes": mesh_path.stat().st_size,
                        "sha256": mesh_record.sha256,
                        "source_asset_sha256": mesh.source_asset_sha256,
                    },
                    "terrain_query": {
                        "mesh_sha256": mesh_record.sha256,
                        "query_sha256": terrain_query_sha256(
                            CanonicalMeshQuery(mesh, transform)
                        ),
                        "world_from_terrain": {
                            "translation_world": [0.0, 0.0, 0.0],
                            "quaternion_world_from_local_wxyz": [
                                1.0, 0.0, 0.0, 0.0
                            ],
                        },
                    },
                }
            ],
            "model": _model_authority(MODEL_PATH, MODEL_PATH.parent),
            "render_config": FIXED_RENDER_CONFIG,
        }
        self.request_path = self.root / "request.json"
        self.request_path.write_bytes(_canonical_json(self.request))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _restore_request(self, request: dict[str, object] | None = None) -> None:
        self.request_path.write_bytes(_canonical_json(request or self.request))

    def test_request_binds_exact_canonical_authorities_and_complete_inputs(self):
        """Catches accepting stale bytes, unsafe paths, or unbound intervals."""

        from mm_sonic.terrain_oracle.render_media import load_render_request

        loaded = load_render_request(self.request_path)
        self.assertEqual(loaded["interval_keys"], self.request["interval_keys"])
        self.assertEqual(loaded["inputs"][0]["clip"]["clip_id"], "synthetic-forward")

        mutations: list[tuple[str, dict[str, object]]] = []
        extra = json.loads(json.dumps(self.request))
        extra["extra"] = False
        mutations.append(("extra field", extra))
        stale = json.loads(json.dumps(self.request))
        stale["inputs"][0]["clip"]["sha256"] = "f" * 64
        mutations.append(("stale clip hash", stale))
        missing = json.loads(json.dumps(self.request))
        missing["interval_keys"] = []
        mutations.append(("unbound accepted interval", missing))
        bad_interval = json.loads(json.dumps(self.request))
        bad_interval["inputs"][0]["interval"] = [0, 5]
        mutations.append(("out-of-bounds interval", bad_interval))
        wrong_query = json.loads(json.dumps(self.request))
        wrong_query["inputs"][0]["terrain_query"]["query_sha256"] = "e" * 64
        mutations.append(("stale terrain query", wrong_query))
        wrong_model = json.loads(json.dumps(self.request))
        wrong_model["model"]["structural_sha256"] = "d" * 64
        mutations.append(("stale model structure", wrong_model))
        wrong_vfs = json.loads(json.dumps(self.request))
        wrong_vfs["model"]["dependency_vfs"]["tree_sha256"] = "c" * 64
        mutations.append(("stale model dependency VFS", wrong_vfs))
        for label, document in mutations:
            with self.subTest(label=label):
                self.request_path.write_bytes(_canonical_json(document))
                with self.assertRaises(ContractError):
                    load_render_request(self.request_path)

        symlink = self.root / "clip-link.npz"
        symlink.symlink_to(self.request["inputs"][0]["clip"]["path"])
        linked = json.loads(json.dumps(self.request))
        linked["inputs"][0]["clip"]["path"] = str(symlink)
        self.request_path.write_bytes(_canonical_json(linked))
        with self.assertRaises(ContractError):
            load_render_request(self.request_path)

    def test_request_file_itself_must_be_canonical_regular_and_nonsymlinked(self):
        """Catches JSON aliases and request-path authority substitution."""

        from mm_sonic.terrain_oracle.render_media import load_render_request

        self.request_path.write_text(json.dumps(self.request), encoding="ascii")
        with self.assertRaises(ContractError):
            load_render_request(self.request_path)
        self.request_path.write_bytes(_canonical_json(self.request))
        link = self.root / "request-link.json"
        link.symlink_to(self.request_path)
        with self.assertRaises(ContractError):
            load_render_request(link)

    def test_validated_model_snapshot_survives_model_and_dependency_swaps(self):
        """Catches render-time reopening of authenticated XML or asset files."""

        import mujoco
        import mm_sonic.terrain_oracle.render_media as renderer

        model_root = self.root / "snapshot-model"
        model_root.mkdir()
        mesh_path = model_root / "visual.obj"
        include_path = model_root / "body.xml"
        model_path = model_root / "model.xml"
        mesh_path.write_text(
            "\n".join(
                (
                    "v 0 0 0",
                    "v 1 0 0",
                    "v 0 1 0",
                    "v 0 0 1",
                    "f 1 2 3",
                    "f 1 4 2",
                    "f 1 3 4",
                    "f 2 4 3",
                    "",
                )
            ),
            encoding="ascii",
        )
        include_path.write_text(
            '<mujoco><worldbody><body name="authenticated_visual">'
            '<geom type="mesh" mesh="visual"/>'
            "</body></worldbody></mujoco>\n",
            encoding="ascii",
        )
        model_path.write_text(
            '<mujoco model="authenticated">'
            '<asset><mesh name="visual" file="visual.obj"/></asset>'
            '<include file="body.xml"/>'
            "</mujoco>\n",
            encoding="ascii",
        )
        document = json.loads(json.dumps(self.request))
        document["model"] = _model_authority(model_path, model_root)
        self._restore_request(document)

        loaded = renderer.load_render_request(self.request_path)
        item = loaded["_loaded_inputs"][0]
        baseline, baseline_mesh_count = renderer._build_scene_model(
            loaded["_model_spec"], item["mesh"], item["transform"]
        )

        include_path.write_text(
            '<mujoco><worldbody><body name="swapped_dependency"/>'
            "</worldbody></mujoco>\n",
            encoding="ascii",
        )
        mesh_path.write_text("not an authenticated mesh\n", encoding="ascii")
        after_dependency_swap, dependency_mesh_count = renderer._build_scene_model(
            loaded["_model_spec"], item["mesh"], item["transform"]
        )
        self.assertEqual(
            structural_model_sha256(after_dependency_swap),
            structural_model_sha256(baseline),
        )
        self.assertEqual(dependency_mesh_count, baseline_mesh_count)

        model_path.write_text(
            '<mujoco model="swapped_model"><worldbody/></mujoco>\n',
            encoding="ascii",
        )
        after_model_swap, model_mesh_count = renderer._build_scene_model(
            loaded["_model_spec"], item["mesh"], item["transform"]
        )
        self.assertEqual(
            structural_model_sha256(after_model_swap),
            structural_model_sha256(baseline),
        )
        self.assertEqual(model_mesh_count, baseline_mesh_count)
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                after_model_swap,
                mujoco.mjtObj.mjOBJ_BODY,
                "authenticated_visual",
            ),
            0,
        )
        self.assertEqual(
            mujoco.mj_name2id(
                after_model_swap,
                mujoco.mjtObj.mjOBJ_BODY,
                "swapped_dependency",
            ),
            -1,
        )

    def test_root_xml_swap_at_parse_boundary_cannot_enter_retained_spec(self):
        """Catches authenticating root bytes before reopening the model path."""

        import mujoco
        import mm_sonic.terrain_oracle.render_media as renderer

        model_root = self.root / "root-parse-boundary"
        model_root.mkdir()
        model_path = model_root / "model.xml"
        model_path.write_text(
            '<mujoco model="authenticated"><worldbody>'
            '<body name="authenticated_body"/>'
            "</worldbody></mujoco>\n",
            encoding="ascii",
        )
        authority = _model_authority(model_path, model_root)
        original_from_string = mujoco.MjSpec.from_string

        def swap_then_parse(xml, include=None, assets=None):
            model_path.write_text(
                '<mujoco model="swapped"><worldbody>'
                '<body name="swapped_body"/>'
                "</worldbody></mujoco>\n",
                encoding="ascii",
            )
            return original_from_string(xml, include=include, assets=assets)

        with mock.patch.object(
            mujoco.MjSpec,
            "from_string",
            side_effect=swap_then_parse,
        ):
            spec, actual_vfs = renderer._authenticated_model_spec(authority)
        model = spec.compile()
        self.assertEqual(actual_vfs, authority["dependency_vfs"])
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "authenticated_body",
            ),
            0,
        )
        self.assertEqual(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "swapped_body",
            ),
            -1,
        )

    def test_visual_include_and_asset_swap_cannot_bypass_structural_hash(self):
        """Catches visual-only dependency swaps omitted by the mechanics hash."""

        import mujoco
        import mm_sonic.terrain_oracle.render_media as renderer

        model_root = self.root / "visual-parse-boundary"
        model_root.mkdir()
        model_path = model_root / "model.xml"
        include_path = model_root / "body.xml"
        texture_path = model_root / "visual.png"
        model_path.write_text(
            '<mujoco model="authenticated_visual">'
            '<asset><texture name="visual_texture" type="2d" '
            'file="visual.png"/>'
            '<material name="visual_material" texture="visual_texture"/>'
            "</asset><include file=\"body.xml\"/></mujoco>\n",
            encoding="ascii",
        )
        authenticated_include = (
            '<mujoco><worldbody><body name="visual_body">'
            '<geom name="visual_include_geom" type="sphere" size=".05" '
            'rgba=".1 .2 .3 1"/>'
            '<geom name="visual_asset_geom" type="box" size=".1 .1 .1" '
            'material="visual_material"/>'
            "</body></worldbody></mujoco>\n"
        )
        swapped_include = authenticated_include.replace(
            'rgba=".1 .2 .3 1"',
            'rgba=".8 .1 .6 1"',
        )

        def png_bytes(rgb: tuple[int, int, int]) -> bytes:
            buffer = io.BytesIO()
            Image.new("RGB", (2, 2), rgb).save(buffer, format="PNG")
            return buffer.getvalue()

        authenticated_texture = png_bytes((12, 34, 56))
        swapped_texture = png_bytes((210, 25, 150))
        include_path.write_text(authenticated_include, encoding="ascii")
        texture_path.write_bytes(authenticated_texture)
        root_xml = model_path.read_text(encoding="ascii")
        authenticated_model = mujoco.MjSpec.from_string(
            root_xml,
            include={"body.xml": authenticated_include.encode("ascii")},
            assets={"visual.png": authenticated_texture},
        ).compile()
        authority = _model_authority(model_path, model_root)

        swapped_model = mujoco.MjSpec.from_string(
            root_xml,
            include={"body.xml": swapped_include.encode("ascii")},
            assets={"visual.png": swapped_texture},
        ).compile()
        self.assertEqual(
            structural_model_sha256(swapped_model),
            structural_model_sha256(authenticated_model),
        )
        self.assertFalse(
            np.array_equal(
                swapped_model.geom("visual_include_geom").rgba,
                authenticated_model.geom("visual_include_geom").rgba,
            )
        )
        self.assertFalse(
            np.array_equal(swapped_model.tex_data, authenticated_model.tex_data)
        )
        include_path.write_text(authenticated_include, encoding="ascii")
        texture_path.write_bytes(authenticated_texture)
        original_from_string = mujoco.MjSpec.from_string

        def swap_then_parse(xml, include=None, assets=None):
            include_path.write_text(swapped_include, encoding="ascii")
            texture_path.write_bytes(swapped_texture)
            return original_from_string(xml, include=include, assets=assets)

        with mock.patch.object(
            mujoco.MjSpec,
            "from_string",
            side_effect=swap_then_parse,
        ):
            spec, actual_vfs = renderer._authenticated_model_spec(authority)
        retained_model = spec.compile()
        self.assertEqual(actual_vfs, authority["dependency_vfs"])
        np.testing.assert_array_equal(
            retained_model.geom("visual_include_geom").rgba,
            authenticated_model.geom("visual_include_geom").rgba,
        )
        np.testing.assert_array_equal(
            retained_model.tex_data,
            authenticated_model.tex_data,
        )

    def test_repeated_authenticated_records_are_decoded_once(self):
        """Catches reloading the same clip and mesh for every interval."""

        import mm_sonic.terrain_oracle.render_media as renderer

        document = json.loads(json.dumps(self.request))
        digest = document["inputs"][0]["clip"]["sha256"]
        first = json.loads(json.dumps(document["inputs"][0]))
        second = json.loads(json.dumps(document["inputs"][0]))
        first["interval_key"] = f"{digest}:0:2"
        first["interval"] = [0, 2]
        second["interval_key"] = f"{digest}:2:4"
        second["interval"] = [2, 4]
        document["kind"] = "contact_sheet"
        document["interval_keys"] = [first["interval_key"], second["interval_key"]]
        document["inputs"] = [first, second]
        self._restore_request(document)

        with (
            mock.patch.object(
                renderer,
                "_decode_authenticated_clip_payload",
                wraps=renderer._decode_authenticated_clip_payload,
            ) as read_clip_spy,
            mock.patch.object(
                renderer,
                "_decode_authenticated_mesh_payload",
                wraps=renderer._decode_authenticated_mesh_payload,
            ) as read_mesh_spy,
        ):
            loaded = renderer.load_render_request(self.request_path)
        self.assertEqual(read_clip_spy.call_count, 1)
        self.assertEqual(read_mesh_spy.call_count, 1)
        self.assertIs(
            loaded["_loaded_inputs"][0]["clip"],
            loaded["_loaded_inputs"][1]["clip"],
        )
        self.assertIs(
            loaded["_loaded_inputs"][0]["mesh"],
            loaded["_loaded_inputs"][1]["mesh"],
        )

    def test_unique_decoded_artifact_bytes_are_bounded(self):
        """Catches accepting decoded clip/mesh state beyond the fixed budget."""

        import mm_sonic.terrain_oracle.render_media as renderer

        tiny_config = json.loads(json.dumps(renderer.FIXED_RENDER_CONFIG))
        tiny_config["max_decoded_unique_bytes"] = 1
        document = json.loads(json.dumps(self.request))
        document["render_config"] = tiny_config
        self._restore_request(document)
        with mock.patch.dict(
            renderer.FIXED_RENDER_CONFIG, tiny_config, clear=True
        ):
            with self.assertRaisesRegex(ContractError, "decoded artifact"):
                renderer.load_render_request(self.request_path)

    def test_repeated_records_charge_once_before_second_unique_budget_rejection(
        self,
    ):
        """Catches charging duplicates repeatedly or admitting a unique overflow."""

        import mm_sonic.terrain_oracle.render_media as renderer

        document = json.loads(json.dumps(self.request))
        original_input = document["inputs"][0]
        original_clip_path = Path(original_input["clip"]["path"])
        original_clip = read_clip(original_clip_path)
        unique_clip = replace(original_clip, clip_id="decoded-budget-unique")
        unique_record = write_clip(original_clip_path.parent, unique_clip)
        unique_clip_path = original_clip_path.parent / (
            f"{unique_record.sha256}.npz"
        )
        unique_input = json.loads(json.dumps(original_input))
        unique_input["clip"] = {
            "path": str(unique_clip_path),
            "size_bytes": unique_clip_path.stat().st_size,
            "sha256": unique_record.sha256,
            "clip_id": unique_clip.clip_id,
            "source_sha256": unique_clip.source.source_sha256,
            "frame_count": unique_clip.frame_count,
        }
        unique_input["interval"] = [0, 2]
        unique_input["interval_key"] = f"{unique_record.sha256}:0:2"

        first = json.loads(json.dumps(original_input))
        repeated = json.loads(json.dumps(original_input))
        digest = first["clip"]["sha256"]
        first["interval"] = [0, 2]
        first["interval_key"] = f"{digest}:0:2"
        repeated["interval"] = [2, 4]
        repeated["interval_key"] = f"{digest}:2:4"
        first_payload = original_clip_path.read_bytes()
        mesh_payload = Path(first["terrain_mesh"]["path"]).read_bytes()
        clip_decoded_bound = renderer._npz_decoded_size_upper_bound(
            first_payload,
            "clip",
        )
        mesh_decoded_bound = renderer._npz_decoded_size_upper_bound(
            mesh_payload,
            "terrain mesh",
        )
        exact_first_unique_budget = max(
            clip_decoded_bound,
            len(first_payload),
        ) + max(
            mesh_decoded_bound,
            len(mesh_payload),
        )
        tiny_config = json.loads(json.dumps(renderer.FIXED_RENDER_CONFIG))
        tiny_config["max_decoded_unique_bytes"] = exact_first_unique_budget
        document["kind"] = "contact_sheet"
        document["render_config"] = tiny_config

        repeated_only = sorted(
            (first, repeated),
            key=lambda item: item["interval_key"],
        )
        document["inputs"] = repeated_only
        document["interval_keys"] = [
            item["interval_key"] for item in repeated_only
        ]
        self._restore_request(document)
        with mock.patch.dict(
            renderer.FIXED_RENDER_CONFIG,
            tiny_config,
            clear=True,
        ):
            loaded = renderer.load_render_request(self.request_path)
        self.assertIs(
            loaded["_loaded_inputs"][0]["clip"],
            loaded["_loaded_inputs"][1]["clip"],
        )

        with_unique = sorted(
            (*repeated_only, unique_input),
            key=lambda item: item["interval_key"],
        )
        document["inputs"] = with_unique
        document["interval_keys"] = [
            item["interval_key"] for item in with_unique
        ]
        self._restore_request(document)
        with mock.patch.dict(
            renderer.FIXED_RENDER_CONFIG,
            tiny_config,
            clear=True,
        ):
            with self.assertRaisesRegex(ContractError, "decoded artifact"):
                renderer.load_render_request(self.request_path)

    def test_clip_and_mesh_decode_never_reopen_swapped_authority_paths(self):
        """Catches an unbounded second read after bounded byte authentication."""

        import mm_sonic.terrain_oracle.render_media as renderer

        original_match = renderer._match_file_record
        for record_name, label in (
            ("clip", "clip"),
            ("terrain_mesh", "terrain_mesh"),
        ):
            with self.subTest(record=record_name):
                document = json.loads(json.dumps(self.request))
                authority_path = Path(
                    document["inputs"][0][record_name]["path"]
                )
                authenticated_payload = authority_path.read_bytes()
                swapped = False

                def swap_after_authentication(
                    value,
                    fields,
                    current_label,
                    *,
                    max_bytes=None,
                ):
                    nonlocal swapped
                    path, payload = original_match(
                        value,
                        fields,
                        current_label,
                        max_bytes=max_bytes,
                    )
                    if current_label == label and not swapped:
                        path.write_bytes(
                            b"unauthenticated replacement outside the bound"
                            * 1024
                        )
                        swapped = True
                    return path, payload

                self._restore_request(document)
                try:
                    with mock.patch.object(
                        renderer,
                        "_match_file_record",
                        side_effect=swap_after_authentication,
                    ):
                        loaded = renderer.load_render_request(self.request_path)
                    self.assertTrue(swapped)
                    self.assertEqual(len(loaded["_loaded_inputs"]), 1)
                finally:
                    authority_path.write_bytes(authenticated_payload)

    def test_real_media_passes_exact_probe_decode_and_png_coverage_validation(self):
        """Catches arbitrary bytes, invented contact display, or partial output."""

        from mm_sonic.terrain_oracle.render_media import (
            render_media,
            validate_png,
            validate_video,
        )

        self._restore_request()
        video = self.root / "accepted.mp4"
        overlay = self.root / "accepted.contact.png"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(ROOT / "sonic/python"), str(ROOT))
        )
        invocation = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "mm_sonic.terrain_oracle.render_media",
                "--request",
                str(self.request_path),
                "--video",
                str(video),
                "--overlay",
                str(overlay),
            ],
            check=False,
            capture_output=True,
            timeout=60.0,
            env=environment,
        )
        self.assertEqual(invocation.returncode, 0, invocation.stderr.decode())
        self.assertTrue(invocation.stdout, "renderer emitted no canonical result")
        result = json.loads(invocation.stdout.decode("ascii"))
        self.assertEqual(invocation.stdout, _canonical_json(result))
        video_metadata = validate_video(
            video,
            width=320,
            height=240,
            fps=50,
            frame_count=4,
        )
        png_metadata = validate_png(
            overlay,
            width=320,
            height=240,
            interval_keys=self.request["interval_keys"],
        )
        self.assertEqual(video_metadata["codec_name"], "h264")
        self.assertEqual(video_metadata["pix_fmt"], "yuv420p")
        self.assertEqual(video_metadata["nb_read_frames"], 4)
        self.assertEqual(png_metadata["interval_keys"], self.request["interval_keys"])
        self.assertEqual(result["request_sha256"], hashlib.sha256(
            self.request_path.read_bytes()
        ).hexdigest())
        self.assertTrue(result["completed"])
        self.assertEqual(result["video"]["sha256"], hashlib.sha256(
            video.read_bytes()
        ).hexdigest())
        self.assertEqual(result["contact_overlay"]["sha256"], hashlib.sha256(
            overlay.read_bytes()
        ).hexdigest())

        with Image.open(overlay) as image:
            pixels = np.asarray(image.convert("RGB"))
        self.assertGreater(np.unique(pixels.reshape(-1, 3), axis=0).shape[0], 16)
        self.assertTrue(np.any(np.all(pixels == [40, 220, 100], axis=-1)))
        self.assertTrue(np.any(np.all(pixels == [255, 155, 35], axis=-1)))
        self.assertGreater(result["render_evidence"]["visual_mesh_geom_count"], 20)
        self.assertEqual(result["render_evidence"]["terrain_face_count"], 12)
        self.assertEqual(
            result["render_evidence"]["model_dependency_vfs"],
            self.request["model"]["dependency_vfs"],
        )

        fake_video = self.root / "fake.mp4"
        fake_video.write_bytes(b"not a video")
        with self.assertRaises(ContractError):
            validate_video(
                fake_video, width=320, height=240, fps=50, frame_count=4
            )
        fake_png = self.root / "fake.png"
        fake_png.write_bytes(b"nonempty but not png")
        with self.assertRaises(ContractError):
            validate_png(
                fake_png,
                width=320,
                height=240,
                interval_keys=self.request["interval_keys"],
            )
        truncated = self.root / "truncated.mp4"
        truncated.write_bytes(video.read_bytes()[: len(video.read_bytes()) // 2])
        with self.assertRaises(ContractError):
            validate_video(
                truncated, width=320, height=240, fps=50, frame_count=4
            )

    def test_rerender_is_byte_deterministic(self):
        """Catches nondeterministic cameras, pixels, PNGs, or encoder settings."""

        from mm_sonic.terrain_oracle.render_media import render_media

        self._restore_request()
        first_video = self.root / "first.mp4"
        first_overlay = self.root / "first.png"
        second_video = self.root / "second.mp4"
        second_overlay = self.root / "second.png"
        render_media(self.request_path, first_video, first_overlay)
        render_media(self.request_path, second_video, second_overlay)
        self.assertEqual(first_video.read_bytes(), second_video.read_bytes())
        self.assertEqual(first_overlay.read_bytes(), second_overlay.read_bytes())

    def test_composite_overlay_covers_every_declared_interval(self):
        """Catches silently dropped composite inputs or forged tile manifests."""

        from mm_sonic.terrain_oracle.render_media import render_media, validate_png

        composite = json.loads(json.dumps(self.request))
        digest = composite["inputs"][0]["clip"]["sha256"]
        first = json.loads(json.dumps(composite["inputs"][0]))
        second = json.loads(json.dumps(composite["inputs"][0]))
        first["interval_key"] = f"{digest}:0:2"
        first["interval"] = [0, 2]
        second["interval_key"] = f"{digest}:2:4"
        second["interval"] = [2, 4]
        composite["kind"] = "contact_sheet"
        composite["interval_keys"] = [first["interval_key"], second["interval_key"]]
        composite["inputs"] = [first, second]
        self._restore_request(composite)
        video = self.root / "sheet.mp4"
        overlay = self.root / "sheet.png"
        render_media(self.request_path, video, overlay)
        metadata = validate_png(
            overlay,
            width=320,
            height=240,
            interval_keys=composite["interval_keys"],
        )
        self.assertEqual(metadata["tile_count"], 2)
        with self.assertRaises(ContractError):
            validate_png(
                overlay,
                width=320,
                height=240,
                interval_keys=[first["interval_key"]],
            )

    def test_runtime_identity_binds_code_tools_executable_and_dependencies(self):
        """Catches a renderer receipt detached from its executable runtime."""

        from mm_sonic.terrain_oracle import render_media

        identity = render_media.runtime_identity()
        self.assertEqual(
            set(identity),
            {
                "schema",
                "module",
                "config",
                "python",
                "ffmpeg",
                "ffprobe",
                "dependencies",
                "content_sha256",
            },
        )
        module_bytes = Path(render_media.__file__).read_bytes()
        self.assertEqual(
            identity["module"]["sha256"],
            hashlib.sha256(module_bytes).hexdigest(),
        )
        self.assertEqual(identity["config"]["bytes_sha256"], hashlib.sha256(
            _canonical_json(render_media.FIXED_RENDER_CONFIG)
        ).hexdigest())
        self.assertEqual(
            set(identity["dependencies"]), {"mujoco", "numpy", "Pillow"}
        )
        without_hash = dict(identity)
        content_sha256 = without_hash.pop("content_sha256")
        self.assertEqual(
            content_sha256,
            hashlib.sha256(_canonical_json(without_hash)).hexdigest(),
        )


class RenderMediaCliTests(unittest.TestCase):
    def test_fixed_module_invocation_exposes_request_video_and_overlay(self):
        """Catches a missing or incompatible package-owned renderer CLI."""

        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(ROOT / "sonic/python"), str(ROOT))
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "mm_sonic.terrain_oracle.render_media",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--request", result.stdout)
        self.assertIn("--video", result.stdout)
        self.assertIn("--overlay", result.stdout)


class RenderStreamingTests(unittest.TestCase):
    def test_exact_overlay_grid_rejects_first_zero_height_layout(self):
        """Catches accepting a square grid whose integer tile height is zero."""

        import mm_sonic.terrain_oracle.render_media as renderer

        self.assertEqual(renderer._overlay_grid(57_840), (241, 240, 1, 1))
        with self.assertRaises(ContractError):
            renderer._overlay_grid(57_841)

    def test_resource_profile_admits_corpus_and_rejects_next_input(self):
        """Catches a profile too small for 720 clips or unbounded retained frames."""

        import mm_sonic.terrain_oracle.render_media as renderer

        corpus_inputs = [{"start": 0, "end": 2}] * 720
        renderer._validate_render_resource_budget("contact_sheet", corpus_inputs)
        self.assertGreaterEqual(
            renderer.FIXED_RENDER_CONFIG["max_decoded_unique_bytes"],
            1_600_000_000,
        )
        maximum_inputs = renderer.FIXED_RENDER_CONFIG["max_inputs"]
        maximum = [{"start": 0, "end": 2}] * maximum_inputs
        renderer._validate_render_resource_budget("contact_sheet", maximum)
        with self.assertRaises(ContractError):
            renderer._validate_render_resource_budget(
                "contact_sheet",
                maximum + [{"start": 0, "end": 2}],
            )

    def test_contact_sheet_budget_is_one_authenticated_frame_per_input(self):
        """Catches rejecting or redundantly encoding the 720-clip contact sheet."""

        import mm_sonic.terrain_oracle.render_media as renderer

        inputs = [
            {"start": 0, "end": 80_000},
            {"start": 100, "end": 90_100},
        ]
        self.assertEqual(
            renderer._encoded_frame_count("contact_sheet", inputs),
            2,
        )
        self.assertEqual(
            renderer._encoded_frame_count("full_video", inputs),
            170_000,
        )
        self.assertGreaterEqual(
            renderer.FIXED_RENDER_CONFIG["max_frames"],
            170_000,
        )

    def test_ten_thousand_frames_are_streamed_one_at_a_time(self):
        """Catches reintroducing an O(frame_count) RGB-frame buffer."""

        import mm_sonic.terrain_oracle.render_media as renderer

        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        class Sink:
            def __init__(self) -> None:
                self.write_count = 0
                self.largest_write = 0
                self.closed = False

            def write(self, payload: bytes) -> int:
                self.write_count += 1
                self.largest_write = max(self.largest_write, len(payload))
                return len(payload)

            def close(self) -> None:
                self.closed = True

        class ErrorPipe:
            def __init__(self) -> None:
                self.closed = False

            def read(self) -> bytes:
                return b""

            def close(self) -> None:
                self.closed = True

        class Process:
            def __init__(self) -> None:
                self.stdin = Sink()
                self.stderr = ErrorPipe()

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None:
                raise AssertionError("successful streaming must not kill encoder")

        process = Process()

        def fake_render(
            item,
            model_spec,
            consume,
            *,
            representative_only,
        ):
            self.assertFalse(representative_only)
            for _ in range(10_000):
                consume(frame)
            return frame.copy(), {
                "visual_mesh_geom_count": 35,
                "terrain_face_count": 2,
            }, 10_000

        with (
            mock.patch.object(renderer, "_tool_path", return_value=Path("/ffmpeg")),
            mock.patch.object(renderer.subprocess, "Popen", return_value=process),
            mock.patch.object(renderer, "_render_input_frames", side_effect=fake_render),
        ):
            representatives, evidence, frame_count = renderer._encode_video(
                Path("/unused.mp4"), [{"interval_key": "fixture"}], object()
            )
        self.assertEqual(frame_count, 10_000)
        self.assertEqual(process.stdin.write_count, 10_000)
        self.assertEqual(process.stdin.largest_write, frame.nbytes)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stderr.closed)
        self.assertEqual(len(representatives), 1)
        self.assertEqual(evidence["visual_mesh_geom_count"], 35)


if __name__ == "__main__":
    unittest.main()
