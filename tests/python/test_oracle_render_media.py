from __future__ import annotations

from dataclasses import replace
import hashlib
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
from mm_sonic.terrain_oracle.storage import write_clip, write_mesh
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
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        model_bytes = MODEL_PATH.read_bytes()
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
            "model": {
                "path": str(MODEL_PATH.resolve()),
                "size_bytes": len(model_bytes),
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "structural_sha256": structural_model_sha256(model),
            },
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

        def fake_render(item, model_path, consume):
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
                Path("/unused.mp4"), [{"interval_key": "fixture"}], Path("/model.xml")
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
