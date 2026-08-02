from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.terrain_oracle.audit import (
    ClipAudit,
    audit_clip,
    structural_model_sha256,
)
from mm_sonic.terrain_oracle.canonical import CanonicalTerrainMesh
from mm_sonic.terrain_oracle.contact import (
    CONTACT_RECONSTRUCTION_TAG,
    CanonicalMeshQuery,
)
from mm_sonic.terrain_oracle.storage import (
    load_corpus,
    publish_corpus,
    read_clip,
    read_mesh,
    write_clip,
    write_mesh,
)
from tests.python.test_oracle_audit import (
    MODEL_PATH,
    _plane_query,
    _real_clean_clip,
)
from tests.python.terrain_oracle_test_utils import synthetic_canonical_clip


def _write_fixture_bundle(root: Path) -> tuple[Path, str, str]:
    bundle = root / "fixture"
    bundle.mkdir()
    clip = synthetic_canonical_clip(frames=8)
    mesh = CanonicalTerrainMesh(
        vertices_local=np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        ),
        faces=np.array(((0, 1, 2),), dtype=np.int32),
        valid_faces=np.array((True,), dtype=np.bool_),
        source_asset_sha256="a" * 64,
    )
    clip_record = write_clip(bundle / "clips", clip)
    mesh_record = write_mesh(bundle / "meshes", mesh)
    manifest_only = publish_corpus(
        root / "fixture-manifest",
        (clip_record,),
        {
            "fixture": "terrain-oracle-cli",
            "mesh_records": [mesh_record.to_dict()],
        },
    )
    (bundle / "manifest.json").write_bytes(
        (manifest_only / "manifest.json").read_bytes()
    )
    return bundle, clip_record.sha256, mesh_record.sha256


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _canonical_json_bytes(value: object) -> bytes:
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


def _rewrite_hashed_document(path: Path, value: dict[str, object]) -> None:
    without_hash = dict(value)
    without_hash.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(without_hash)
    ).hexdigest()
    path.write_bytes(_canonical_json_bytes(value))


def _write_audit_fixture(root: Path) -> Path:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    query = _plane_query()
    clip = _real_clean_clip(model, query, frames=40)
    joint_position = np.array(clip.joint_position, copy=True)
    joint = model.joint(clip.joint_names[0]).id
    joint_position[20, 0] = model.jnt_range[joint, 1] + 0.1
    clip = replace(
        clip,
        clip_id="fixture-removable-defect",
        joint_position=joint_position,
        action_tags=("walk", CONTACT_RECONSTRUCTION_TAG),
    )
    bundle = root / "audit-fixture"
    bundle.mkdir()
    clip_record = write_clip(bundle / "clips", clip)
    mesh_record = write_mesh(bundle / "meshes", query.mesh)
    manifest_only = publish_corpus(
        root / "audit-fixture-manifest",
        (clip_record,),
        {
            "fixture": "terrain-oracle-audit-cli",
            "mesh_records": [mesh_record.to_dict()],
        },
    )
    (bundle / "manifest.json").write_bytes(
        (manifest_only / "manifest.json").read_bytes()
    )
    return bundle


def _write_interval_renderer(path: Path) -> Path:
    path.write_text(
        """from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--kind", required=True)
parser.add_argument("--clip-sha256")
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--interval-key", action="append", default=[])
parser.add_argument("--stratum")
parser.add_argument("--video", required=True, type=Path)
parser.add_argument("--overlay", required=True, type=Path)
arguments = parser.parse_args()
semantic = json.dumps(
    {
        "kind": arguments.kind,
        "clip_sha256": arguments.clip_sha256,
        "start": arguments.start,
        "end": arguments.end,
        "interval_keys": arguments.interval_key,
        "stratum": arguments.stratum,
    },
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
arguments.video.parent.mkdir(parents=True, exist_ok=True)
arguments.overlay.parent.mkdir(parents=True, exist_ok=True)
arguments.video.write_bytes(b"deterministic-video-v1\\0" + semantic)
arguments.overlay.write_bytes(b"deterministic-contact-overlay-v1\\0" + semantic)
"""
    )
    return path


def _write_broken_renderer(path: Path, mode: str) -> Path:
    path.write_text(
        """from __future__ import annotations
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, type=Path)
parser.add_argument("--overlay", required=True, type=Path)
arguments, _ = parser.parse_known_args()
mode = "__MODE__"
if mode == "nonzero":
    raise SystemExit(9)
if mode == "missing":
    raise SystemExit(0)
arguments.video.parent.mkdir(parents=True, exist_ok=True)
arguments.overlay.parent.mkdir(parents=True, exist_ok=True)
arguments.video.symlink_to("/dev/null")
arguments.overlay.symlink_to("/dev/null")
""".replace("__MODE__", mode)
    )
    return path


def _write_huggingface_metadata(
    release: Path,
    relative_path: str,
) -> None:
    from mm_sonic.terrain_oracle.source_lafan import (
        LAFAN1_UPSTREAM_REVISION,
    )

    path = (
        release
        / ".cache"
        / "huggingface"
        / "download"
        / f"{relative_path}.metadata"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{LAFAN1_UPSTREAM_REVISION}\n"
        f"etag-{relative_path}\n"
        "1720000000.0\n"
    )


class CorpusCliTests(unittest.TestCase):
    def test_fixture_import_is_deterministic_and_publishes_a_complete_corpus(self):
        """Catches partial, path-dependent, or nondeterministic publication."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, clip_sha256, mesh_sha256 = _write_fixture_bundle(root)
            first = root / "first"
            second = root / "second"
            try:
                from mm_sonic.terrain_oracle.corpus_cli import main
            except ModuleNotFoundError:
                self.assertFalse(first.exists())
                self.assertFalse(second.exists())
                raise

            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(first),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(second),
                    ]
                ),
                0,
            )

            self.assertEqual(_tree_bytes(first), _tree_bytes(second))
            manifest = load_corpus(first)
            self.assertEqual(
                [record.sha256 for record in manifest.clips],
                [clip_sha256],
            )
            self.assertEqual(
                [record.sha256 for record in manifest.meshes],
                [mesh_sha256],
            )
            clip_path = first / manifest.clips[0].relative_path
            mesh_path = first / manifest.meshes[0].relative_path
            self.assertEqual(
                hashlib.sha256(clip_path.read_bytes()).hexdigest(),
                clip_sha256,
            )
            self.assertEqual(
                hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                mesh_sha256,
            )
            read_clip(clip_path).validate()
            self.assertEqual(
                read_mesh(mesh_path).source_asset_sha256,
                "a" * 64,
            )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_audit_publishes_exact_closed_evidence_deterministically(self):
        """Catches stale hashes, noncanonical paths, or partial audit output."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            first = root / "first-audit"
            second = root / "second-audit"
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(raw),
                    ]
                ),
                0,
            )

            command = [
                "audit",
                "--corpus",
                str(raw),
                "--model",
                str(MODEL_PATH),
                "--output",
            ]
            try:
                first_status = main([*command, str(first)])
            except BaseException:
                self.assertFalse(first.exists())
                raise
            self.assertEqual(first_status, 0)
            self.assertEqual(main([*command, str(second)]), 0)
            self.assertEqual(_tree_bytes(first), _tree_bytes(second))

            manifest = load_corpus(first)
            self.assertEqual(len(manifest.clips), 1)
            self.assertEqual(len(manifest.meshes), 1)
            clip_path = first / manifest.clips[0].relative_path
            mesh_path = first / manifest.meshes[0].relative_path
            clip = read_clip(clip_path)
            mesh = read_mesh(mesh_path)
            query = CanonicalMeshQuery(
                mesh,
                clip.terrain.world_from_terrain,
            )
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
            expected_report = audit_clip(clip, model, query)
            self.assertEqual(
                expected_report.accepted_intervals,
                ((0, 17), (24, 40)),
            )

            index_path = first / "audit-index.json"
            index = json.loads(index_path.read_text("ascii"))
            self.assertEqual(
                set(index),
                {
                    "schema",
                    "source_corpus_manifest_sha256",
                    "corpus_manifest_sha256",
                    "model",
                    "reports",
                    "content_sha256",
                },
            )
            self.assertEqual(index["schema"], "terrain-oracle-audit-index/v1")
            without_hash = dict(index)
            content_sha256 = without_hash.pop("content_sha256")
            self.assertEqual(
                content_sha256,
                hashlib.sha256(
                    _canonical_json_bytes(without_hash)
                ).hexdigest(),
            )
            self.assertEqual(
                index["source_corpus_manifest_sha256"],
                hashlib.sha256(
                    (raw / "manifest.json").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                index["corpus_manifest_sha256"],
                hashlib.sha256(
                    (first / "manifest.json").read_bytes()
                ).hexdigest(),
            )
            model_bytes = MODEL_PATH.read_bytes()
            self.assertEqual(
                index["model"],
                {
                    "asset_path": str(MODEL_PATH.resolve()),
                    "asset_size_bytes": len(model_bytes),
                    "asset_sha256": hashlib.sha256(model_bytes).hexdigest(),
                    "structural_sha256": structural_model_sha256(model),
                },
            )
            self.assertEqual(len(index["reports"]), 1)
            entry = index["reports"][0]
            self.assertEqual(
                set(entry),
                {
                    "clip_id",
                    "clip_sha256",
                    "source_sha256",
                    "model_sha256",
                    "terrain_sha256",
                    "status",
                    "accepted_intervals",
                    "reasons",
                    "relative_path",
                    "sha256",
                },
            )
            self.assertEqual(
                entry["relative_path"],
                f"audits/{manifest.clips[0].sha256}.json",
            )
            self.assertEqual(
                entry["accepted_intervals"],
                [[0, 17], [24, 40]],
            )
            report_path = first / entry["relative_path"]
            self.assertEqual(
                entry["sha256"],
                hashlib.sha256(report_path.read_bytes()).hexdigest(),
            )
            report = ClipAudit.from_dict(
                json.loads(report_path.read_text("ascii"))
            )
            self.assertEqual(report, expected_report)
            self.assertEqual(entry["clip_sha256"], report.clip_sha256)
            self.assertEqual(entry["source_sha256"], report.source_sha256)
            self.assertEqual(entry["model_sha256"], report.model_sha256)
            self.assertEqual(entry["terrain_sha256"], report.terrain_sha256)
            self.assertEqual(entry["status"], report.status)
            self.assertEqual(
                entry["reasons"],
                [reason.to_dict() for reason in report.reasons],
            )

            before = _tree_bytes(first)
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(main([*command, str(first)]), 2)
            self.assertIn("exist", errors.getvalue().lower())
            self.assertEqual(_tree_bytes(first), before)

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_render_audit_records_exact_interval_subprocess_evidence(self):
        """Catches unbound renderer argv or receipts unrelated to output bytes."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            renders = audited / "render-audit"
            renderer = _write_interval_renderer(root / "renderer.py")
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(raw),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(raw),
                        "--model",
                        str(MODEL_PATH),
                        "--output",
                        str(audited),
                    ]
                ),
                0,
            )

            try:
                status = main(
                    [
                        "render-audit",
                        "--corpus",
                        str(audited),
                        "--renderer",
                        sys.executable,
                        "--renderer-arg",
                        str(renderer),
                        "--output",
                        str(renders),
                    ]
                )
            except BaseException:
                self.assertFalse(renders.exists())
                raise
            self.assertEqual(status, 0)

            audit_index = json.loads(
                (audited / "audit-index.json").read_text("ascii")
            )
            audit_entry = audit_index["reports"][0]
            manifest = load_corpus(audited)
            clip = read_clip(
                audited / manifest.clips[0].relative_path
            )
            receipts = [
                json.loads(path.read_text("ascii"))
                for path in sorted((renders / "receipts").glob("*.json"))
            ]
            self.assertEqual(len(receipts), 2)
            self.assertEqual(
                {tuple(receipt["interval"]) for receipt in receipts},
                {(0, 17), (24, 40)},
            )
            for receipt in receipts:
                self.assertEqual(
                    set(receipt),
                    {
                        "schema",
                        "kind",
                        "clip_id",
                        "clip_sha256",
                        "model",
                        "terrain_sha256",
                        "terrain_mesh_sha256",
                        "interval",
                        "renderer",
                        "video",
                        "contact_overlay",
                        "completed",
                    },
                )
                self.assertEqual(
                    receipt["schema"],
                    "terrain-oracle-render-receipt/v1",
                )
                self.assertEqual(receipt["kind"], "accepted_interval")
                self.assertEqual(receipt["clip_id"], clip.clip_id)
                self.assertEqual(
                    receipt["clip_sha256"],
                    manifest.clips[0].sha256,
                )
                self.assertEqual(receipt["model"], audit_index["model"])
                self.assertEqual(
                    receipt["terrain_sha256"],
                    audit_entry["terrain_sha256"],
                )
                self.assertEqual(
                    receipt["terrain_mesh_sha256"],
                    clip.terrain.mesh_sha256,
                )
                self.assertIs(receipt["completed"], True)
                self.assertEqual(
                    set(receipt["renderer"]),
                    {
                        "argv",
                        "executable_sha256",
                        "invocation_sha256",
                        "shell",
                        "returncode",
                    },
                )
                self.assertIs(receipt["renderer"]["shell"], False)
                self.assertEqual(receipt["renderer"]["returncode"], 0)
                normalized_argv = receipt["renderer"]["argv"]
                self.assertEqual(
                    normalized_argv[:8],
                    [
                        str(Path(sys.executable).resolve()),
                        str(renderer.resolve()),
                        "--kind",
                        "accepted_interval",
                        "--clip-sha256",
                        manifest.clips[0].sha256,
                        "--start",
                        str(receipt["interval"][0]),
                    ],
                )
                self.assertEqual(
                    normalized_argv[8:12],
                    [
                        "--end",
                        str(receipt["interval"][1]),
                        "--video",
                        receipt["video"]["relative_path"],
                    ],
                )
                self.assertEqual(
                    normalized_argv[12:],
                    [
                        "--overlay",
                        receipt["contact_overlay"]["relative_path"],
                    ],
                )
                self.assertEqual(
                    receipt["renderer"]["invocation_sha256"],
                    hashlib.sha256(
                        _canonical_json_bytes(normalized_argv)
                    ).hexdigest(),
                )
                self.assertEqual(
                    receipt["renderer"]["executable_sha256"],
                    hashlib.sha256(
                        Path(sys.executable).resolve().read_bytes()
                    ).hexdigest(),
                )
                for field in ("video", "contact_overlay"):
                    artifact = receipt[field]
                    self.assertEqual(
                        set(artifact),
                        {"relative_path", "sha256", "size_bytes"},
                    )
                    relative = Path(artifact["relative_path"])
                    self.assertFalse(relative.is_absolute())
                    self.assertNotIn("..", relative.parts)
                    path = renders / relative
                    payload = path.read_bytes()
                    self.assertEqual(len(payload), artifact["size_bytes"])
                    self.assertEqual(
                        hashlib.sha256(payload).hexdigest(),
                        artifact["sha256"],
                    )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_render_index_covers_contact_sheet_and_recomputed_strata(self):
        """Catches interval-only evidence or receipt-declared strata."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            renders = audited / "render-audit"
            renderer = _write_interval_renderer(root / "renderer.py")
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(raw),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(raw),
                        "--model",
                        str(MODEL_PATH),
                        "--output",
                        str(audited),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "render-audit",
                        "--corpus",
                        str(audited),
                        "--renderer",
                        sys.executable,
                        "--renderer-arg",
                        str(renderer),
                        "--output",
                        str(renders),
                    ]
                ),
                0,
            )

            manifest = load_corpus(audited)
            clip_record = manifest.clips[0]
            accepted_keys = [
                f"{clip_record.sha256}:0:17",
                f"{clip_record.sha256}:24:40",
            ]
            index_path = renders / "render-index.json"
            index = json.loads(index_path.read_text("ascii"))
            self.assertEqual(
                set(index),
                {
                    "schema",
                    "corpus_manifest_sha256",
                    "audit_index_sha256",
                    "accepted_interval_keys",
                    "renderer",
                    "interval_receipts",
                    "contact_sheet_receipt",
                    "stratum_receipts",
                    "content_sha256",
                },
            )
            self.assertEqual(
                index["schema"],
                "terrain-oracle-render-index/v1",
            )
            self.assertEqual(index["accepted_interval_keys"], accepted_keys)
            self.assertEqual(
                index["corpus_manifest_sha256"],
                hashlib.sha256(
                    (audited / "manifest.json").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                index["audit_index_sha256"],
                hashlib.sha256(
                    (audited / "audit-index.json").read_bytes()
                ).hexdigest(),
            )
            without_hash = dict(index)
            content_sha256 = without_hash.pop("content_sha256")
            self.assertEqual(
                content_sha256,
                hashlib.sha256(
                    _canonical_json_bytes(without_hash)
                ).hexdigest(),
            )
            self.assertEqual(
                index["renderer"],
                {
                    "argv_prefix": [
                        str(Path(sys.executable).resolve()),
                        str(renderer.resolve()),
                    ],
                    "executable_sha256": hashlib.sha256(
                        Path(sys.executable).resolve().read_bytes()
                    ).hexdigest(),
                },
            )
            self.assertEqual(len(index["interval_receipts"]), 2)
            self.assertEqual(len(index["stratum_receipts"]), 1)

            def load_receipt(reference):
                self.assertEqual(
                    set(reference),
                    {"relative_path", "sha256"},
                )
                relative = Path(reference["relative_path"])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                payload = (renders / relative).read_bytes()
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    reference["sha256"],
                )
                return json.loads(payload.decode("ascii"))

            sheet = load_receipt(index["contact_sheet_receipt"])
            self.assertEqual(
                set(sheet),
                {
                    "schema",
                    "kind",
                    "inputs",
                    "interval_keys",
                    "renderer",
                    "video",
                    "contact_overlay",
                    "completed",
                },
            )
            self.assertEqual(sheet["kind"], "contact_sheet")
            self.assertEqual(sheet["interval_keys"], accepted_keys)
            self.assertEqual(
                [item["interval_key"] for item in sheet["inputs"]],
                accepted_keys,
            )

            stratum = load_receipt(index["stratum_receipts"][0])
            expected_stratum = {
                "source": "synthetic",
                "action_class": "walk",
                "terrain": "other",
                "direction": "forward",
            }
            self.assertEqual(
                set(stratum),
                {
                    "schema",
                    "kind",
                    "stratum",
                    "inputs",
                    "interval_keys",
                    "renderer",
                    "video",
                    "contact_overlay",
                    "completed",
                },
            )
            self.assertEqual(stratum["kind"], "full_video")
            self.assertEqual(stratum["stratum"], expected_stratum)
            self.assertEqual(stratum["interval_keys"], accepted_keys)
            self.assertEqual(
                [item["interval_key"] for item in stratum["inputs"]],
                accepted_keys,
            )

            for receipt in (sheet, stratum):
                self.assertEqual(
                    receipt["schema"],
                    "terrain-oracle-render-receipt/v1",
                )
                self.assertIs(receipt["completed"], True)
                argv = receipt["renderer"]["argv"]
                self.assertEqual(
                    argv[:4],
                    [
                        str(Path(sys.executable).resolve()),
                        str(renderer.resolve()),
                        "--kind",
                        receipt["kind"],
                    ],
                )
                self.assertEqual(
                    receipt["renderer"]["invocation_sha256"],
                    hashlib.sha256(
                        _canonical_json_bytes(argv)
                    ).hexdigest(),
                )
                self.assertIs(receipt["renderer"]["shell"], False)
                self.assertEqual(receipt["renderer"]["returncode"], 0)
                if receipt["kind"] == "contact_sheet":
                    self.assertNotIn("--stratum", argv)
                else:
                    position = argv.index("--stratum")
                    self.assertEqual(
                        json.loads(argv[position + 1]),
                        expected_stratum,
                    )
                declared_keys = [
                    argv[position + 1]
                    for position, value in enumerate(argv)
                    if value == "--interval-key"
                ]
                self.assertEqual(declared_keys, accepted_keys)
                for field in ("video", "contact_overlay"):
                    artifact = receipt[field]
                    self.assertEqual(
                        set(artifact),
                        {"relative_path", "sha256", "size_bytes"},
                    )
                    path = renders / artifact["relative_path"]
                    payload = path.read_bytes()
                    self.assertEqual(len(payload), artifact["size_bytes"])
                    self.assertEqual(
                        hashlib.sha256(payload).hexdigest(),
                        artifact["sha256"],
                    )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_render_evidence_rejects_missing_extra_duplicate_or_false_sets(self):
        """Catches hash-consistent evidence that is not semantically exhaustive."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _load_render_evidence,
            main,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            renders = audited / "render-audit"
            renderer = _write_interval_renderer(root / "renderer.py")
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(raw),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(raw),
                        "--model",
                        str(MODEL_PATH),
                        "--output",
                        str(audited),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "render-audit",
                        "--corpus",
                        str(audited),
                        "--renderer",
                        sys.executable,
                        "--renderer-arg",
                        str(renderer),
                        "--output",
                        str(renders),
                    ]
                ),
                0,
            )
            _load_render_evidence(audited)

            def index_at(render_root):
                return json.loads(
                    (render_root / "render-index.json").read_text("ascii")
                )

            def missing_interval(render_root):
                index = index_at(render_root)
                (render_root / index["interval_receipts"][0]["relative_path"]).unlink()

            def extra_interval(render_root):
                index = index_at(render_root)
                original = (
                    render_root
                    / index["interval_receipts"][0]["relative_path"]
                )
                (render_root / "receipts" / "extra.json").write_bytes(
                    original.read_bytes()
                )

            def duplicate_interval(render_root):
                index = index_at(render_root)
                index["interval_receipts"].append(
                    dict(index["interval_receipts"][0])
                )
                _rewrite_hashed_document(
                    render_root / "render-index.json",
                    index,
                )

            def missing_sheet(render_root):
                index = index_at(render_root)
                (
                    render_root
                    / index["contact_sheet_receipt"]["relative_path"]
                ).unlink()

            def extra_sheet(render_root):
                index = index_at(render_root)
                original = (
                    render_root
                    / index["contact_sheet_receipt"]["relative_path"]
                )
                (render_root / "contact-sheet" / "extra.json").write_bytes(
                    original.read_bytes()
                )

            def missing_stratum(render_root):
                index = index_at(render_root)
                (
                    render_root
                    / index["stratum_receipts"][0]["relative_path"]
                ).unlink()

            def extra_stratum(render_root):
                index = index_at(render_root)
                original = (
                    render_root
                    / index["stratum_receipts"][0]["relative_path"]
                )
                (render_root / "strata" / "extra.json").write_bytes(
                    original.read_bytes()
                )

            def duplicate_stratum(render_root):
                index = index_at(render_root)
                index["stratum_receipts"].append(
                    dict(index["stratum_receipts"][0])
                )
                _rewrite_hashed_document(
                    render_root / "render-index.json",
                    index,
                )

            def false_declared_stratum(render_root):
                index = index_at(render_root)
                reference = index["stratum_receipts"][0]
                receipt_path = render_root / reference["relative_path"]
                receipt = json.loads(receipt_path.read_text("ascii"))
                receipt["stratum"]["direction"] = "backward"
                payload = _canonical_json_bytes(receipt)
                receipt_path.write_bytes(payload)
                reference["sha256"] = hashlib.sha256(payload).hexdigest()
                _rewrite_hashed_document(
                    render_root / "render-index.json",
                    index,
                )

            mutations = {
                "missing interval": missing_interval,
                "extra interval": extra_interval,
                "duplicate interval": duplicate_interval,
                "missing sheet": missing_sheet,
                "extra sheet": extra_sheet,
                "missing stratum": missing_stratum,
                "extra stratum": extra_stratum,
                "duplicate stratum": duplicate_stratum,
                "false declared stratum": false_declared_stratum,
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    variant = root / name.replace(" ", "-")
                    shutil.copytree(audited, variant)
                    mutate(variant / "render-audit")
                    with self.assertRaises(ContractError):
                        _load_render_evidence(variant)

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_renderer_failures_leave_no_output_or_staging_debris(self):
        """Catches partial evidence after process or artifact failure."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            renders = audited / "render-audit"
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(raw),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(raw),
                        "--model",
                        str(MODEL_PATH),
                        "--output",
                        str(audited),
                    ]
                ),
                0,
            )
            before = _tree_bytes(audited)
            for mode in ("nonzero", "missing", "symlink"):
                with self.subTest(mode=mode):
                    renderer = _write_broken_renderer(
                        root / f"{mode}.py",
                        mode,
                    )
                    errors = io.StringIO()
                    with redirect_stderr(errors):
                        status = main(
                            [
                                "render-audit",
                                "--corpus",
                                str(audited),
                                "--renderer",
                                sys.executable,
                                "--renderer-arg",
                                str(renderer),
                                "--output",
                                str(renders),
                            ]
                        )
                    self.assertEqual(status, 2)
                    self.assertTrue(errors.getvalue().strip())
                    self.assertFalse(renders.exists())
                    self.assertEqual(_tree_bytes(audited), before)
                    self.assertEqual(
                        list(audited.glob(".render-audit.*")),
                        [],
                    )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_coverage_rebuilds_public_task8_manifest_without_replacement(self):
        """Catches CLI-specific coverage inference or overwrite behavior."""

        from mm_sonic.terrain_oracle.corpus_cli import main
        from mm_sonic.terrain_oracle.coverage import (
            AcceptedClipRecord,
            CoverageManifest,
            build_coverage,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            coverage_path = audited / "coverage.json"
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(raw),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(raw),
                        "--model",
                        str(MODEL_PATH),
                        "--output",
                        str(audited),
                    ]
                ),
                0,
            )
            try:
                status = main(
                    [
                        "coverage",
                        "--corpus",
                        str(audited),
                        "--output",
                        str(coverage_path),
                    ]
                )
            except BaseException:
                self.assertFalse(coverage_path.exists())
                raise
            self.assertEqual(status, 0)

            manifest = load_corpus(audited)
            clip_record = manifest.clips[0]
            clip = read_clip(audited / clip_record.relative_path)
            mesh_record = manifest.meshes[0]
            mesh = read_mesh(audited / mesh_record.relative_path)
            audit_index = json.loads(
                (audited / "audit-index.json").read_text("ascii")
            )
            report_entry = audit_index["reports"][0]
            report = ClipAudit.from_dict(
                json.loads(
                    (
                        audited / report_entry["relative_path"]
                    ).read_text("ascii")
                )
            )
            query = CanonicalMeshQuery(
                mesh,
                clip.terrain.world_from_terrain,
            )
            expected = build_coverage(
                (
                    AcceptedClipRecord(
                        clip=clip,
                        clip_record=clip_record,
                        audit=report,
                        accepted_intervals=report.accepted_intervals,
                        terrain_mesh=mesh,
                        terrain_query=query,
                        model_sha256=report.model_sha256,
                        action_class="walk",
                    ),
                )
            )
            payload = coverage_path.read_bytes()
            self.assertEqual(
                payload,
                _canonical_json_bytes(expected.to_dict()),
            )
            self.assertEqual(
                CoverageManifest.from_dict(
                    json.loads(payload.decode("ascii"))
                ),
                expected,
            )
            before = _tree_bytes(audited)
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(
                    main(
                        [
                            "coverage",
                            "--corpus",
                            str(audited),
                            "--output",
                            str(coverage_path),
                        ]
                    ),
                    2,
                )
            self.assertIn("exist", errors.getvalue().lower())
            self.assertEqual(_tree_bytes(audited), before)

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_freeze_requires_exact_audit_render_and_coverage_evidence(self):
        """Catches partial, stale, externally dependent, or replaceable releases."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            renders = audited / "render-audit"
            coverage_path = audited / "coverage.json"
            frozen = root / "frozen"
            renderer = _write_interval_renderer(root / "renderer.py")
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(raw),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(raw),
                        "--model",
                        str(MODEL_PATH),
                        "--output",
                        str(audited),
                    ]
                ),
                0,
            )

            for missing in ("renders and coverage", "coverage"):
                with self.subTest(missing=missing):
                    before = _tree_bytes(audited)
                    errors = io.StringIO()
                    with redirect_stderr(errors):
                        status = main(
                            [
                                "freeze",
                                "--corpus",
                                str(audited),
                                "--output",
                                str(frozen),
                            ]
                        )
                    self.assertEqual(status, 2)
                    self.assertTrue(errors.getvalue().strip())
                    self.assertFalse(frozen.exists())
                    self.assertEqual(_tree_bytes(audited), before)
                    if missing == "renders and coverage":
                        self.assertEqual(
                            main(
                                [
                                    "render-audit",
                                    "--corpus",
                                    str(audited),
                                    "--renderer",
                                    sys.executable,
                                    "--renderer-arg",
                                    str(renderer),
                                    "--output",
                                    str(renders),
                                ]
                            ),
                            0,
                        )
            self.assertEqual(
                main(
                    [
                        "coverage",
                        "--corpus",
                        str(audited),
                        "--output",
                        str(coverage_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "freeze",
                        "--corpus",
                        str(audited),
                        "--output",
                        str(frozen),
                    ]
                ),
                0,
            )

            source_manifest = load_corpus(audited)
            frozen_manifest = load_corpus(frozen)
            self.assertEqual(frozen_manifest, source_manifest)
            for record in frozen_manifest.clips:
                self.assertEqual(
                    (frozen / record.relative_path).read_bytes(),
                    (audited / record.relative_path).read_bytes(),
                )
            for record in frozen_manifest.meshes:
                self.assertEqual(
                    (frozen / record.relative_path).read_bytes(),
                    (audited / record.relative_path).read_bytes(),
                )
            freeze_index_path = frozen / "freeze-index.json"
            freeze_index = json.loads(
                freeze_index_path.read_text("ascii")
            )
            self.assertEqual(
                set(freeze_index),
                {
                    "schema",
                    "source",
                    "model",
                    "files",
                    "content_sha256",
                },
            )
            self.assertEqual(
                freeze_index["schema"],
                "terrain-oracle-frozen-corpus/v1",
            )
            without_hash = dict(freeze_index)
            content_sha256 = without_hash.pop("content_sha256")
            self.assertEqual(
                content_sha256,
                hashlib.sha256(
                    _canonical_json_bytes(without_hash)
                ).hexdigest(),
            )
            self.assertEqual(
                freeze_index["source"],
                {
                    "corpus_manifest_sha256": hashlib.sha256(
                        (audited / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "audit_index_sha256": hashlib.sha256(
                        (audited / "audit-index.json").read_bytes()
                    ).hexdigest(),
                    "render_index_sha256": hashlib.sha256(
                        (
                            renders / "render-index.json"
                        ).read_bytes()
                    ).hexdigest(),
                    "coverage_sha256": hashlib.sha256(
                        coverage_path.read_bytes()
                    ).hexdigest(),
                },
            )
            model_artifact = freeze_index["model"]
            self.assertEqual(
                set(model_artifact),
                {
                    "relative_path",
                    "sha256",
                    "size_bytes",
                    "original_model_file_sha256",
                    "structural_sha256",
                },
            )
            model_path = frozen / model_artifact["relative_path"]
            self.assertEqual(model_path.suffix, ".mjb")
            self.assertNotEqual(model_path.read_bytes(), MODEL_PATH.read_bytes())
            self.assertEqual(
                model_artifact["sha256"],
                hashlib.sha256(model_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                model_artifact["original_model_file_sha256"],
                hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                model_artifact["structural_sha256"],
                json.loads(
                    (audited / "audit-index.json").read_text("ascii")
                )["model"]["structural_sha256"],
            )
            import mujoco

            release_model = mujoco.MjModel.from_binary_path(
                str(model_path)
            )
            self.assertEqual(
                structural_model_sha256(release_model),
                model_artifact["structural_sha256"],
            )
            declared_files = freeze_index["files"]
            self.assertEqual(
                [item["relative_path"] for item in declared_files],
                sorted(item["relative_path"] for item in declared_files),
            )
            self.assertEqual(
                {item["relative_path"] for item in declared_files},
                {
                    path.relative_to(frozen).as_posix()
                    for path in frozen.rglob("*")
                    if path.is_file() and path != freeze_index_path
                },
            )
            for item in declared_files:
                self.assertEqual(
                    set(item),
                    {"relative_path", "sha256", "size_bytes"},
                )
                payload = (frozen / item["relative_path"]).read_bytes()
                self.assertEqual(len(payload), item["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    item["sha256"],
                )

            before = _tree_bytes(frozen)
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(
                    main(
                        [
                            "freeze",
                            "--corpus",
                            str(audited),
                            "--output",
                            str(frozen),
                        ]
                    ),
                    2,
                )
            self.assertIn("exist", errors.getvalue().lower())
            self.assertEqual(_tree_bytes(frozen), before)

            from mm_sonic.terrain_oracle.corpus_cli import (
                _load_frozen_corpus,
            )

            for mode in ("missing", "tampered"):
                with self.subTest(model_artifact=mode):
                    variant = root / f"frozen-model-{mode}"
                    shutil.copytree(frozen, variant)
                    variant_model = (
                        variant / model_artifact["relative_path"]
                    )
                    if mode == "missing":
                        variant_model.unlink()
                    else:
                        variant_model.write_bytes(
                            variant_model.read_bytes() + b"tampered"
                        )
                    with self.assertRaises(ContractError):
                        _load_frozen_corpus(variant)

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_freeze_rejects_missing_extra_stale_mismatched_and_malformed_evidence(
        self,
    ):
        """Catches permissive freeze validation at any evidence layer."""

        from mm_sonic.terrain_oracle.corpus_cli import main
        from mm_sonic.terrain_oracle.coverage import build_coverage

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            renderer = _write_interval_renderer(root / "renderer.py")
            self.assertEqual(
                main(
                    [
                        "import",
                        "--fixture",
                        str(fixture),
                        "--output",
                        str(raw),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(raw),
                        "--model",
                        str(MODEL_PATH),
                        "--output",
                        str(audited),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "render-audit",
                        "--corpus",
                        str(audited),
                        "--renderer",
                        sys.executable,
                        "--renderer-arg",
                        str(renderer),
                        "--output",
                        str(audited / "render-audit"),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "coverage",
                        "--corpus",
                        str(audited),
                        "--output",
                        str(audited / "coverage.json"),
                    ]
                ),
                0,
            )

            def audit_index_at(corpus):
                return json.loads(
                    (corpus / "audit-index.json").read_text("ascii")
                )

            def render_index_at(corpus):
                return json.loads(
                    (
                        corpus / "render-audit" / "render-index.json"
                    ).read_text("ascii")
                )

            def missing_audit(corpus):
                entry = audit_index_at(corpus)["reports"][0]
                (corpus / entry["relative_path"]).unlink()

            def extra_audit(corpus):
                entry = audit_index_at(corpus)["reports"][0]
                source = corpus / entry["relative_path"]
                (corpus / "audits" / "extra.json").write_bytes(
                    source.read_bytes()
                )

            def malformed_audit(corpus):
                (corpus / "audit-index.json").write_bytes(b"{")

            def missing_receipt(corpus):
                entry = render_index_at(corpus)["interval_receipts"][0]
                (
                    corpus / "render-audit" / entry["relative_path"]
                ).unlink()

            def extra_receipt(corpus):
                entry = render_index_at(corpus)["interval_receipts"][0]
                source = corpus / "render-audit" / entry["relative_path"]
                (
                    corpus / "render-audit" / "receipts" / "extra.json"
                ).write_bytes(source.read_bytes())

            def mismatched_receipt(corpus):
                render_root = corpus / "render-audit"
                index = render_index_at(corpus)
                reference = index["interval_receipts"][0]
                path = render_root / reference["relative_path"]
                receipt = json.loads(path.read_text("ascii"))
                receipt["clip_id"] = "hash-consistent-wrong-clip"
                payload = _canonical_json_bytes(receipt)
                path.write_bytes(payload)
                reference["sha256"] = hashlib.sha256(payload).hexdigest()
                _rewrite_hashed_document(
                    render_root / "render-index.json",
                    index,
                )

            def missing_artifact(corpus):
                render_root = corpus / "render-audit"
                entry = render_index_at(corpus)["interval_receipts"][0]
                receipt = json.loads(
                    (render_root / entry["relative_path"]).read_text("ascii")
                )
                (render_root / receipt["video"]["relative_path"]).unlink()

            def extra_artifact(corpus):
                (
                    corpus
                    / "render-audit"
                    / "artifacts"
                    / "extra.mp4"
                ).write_bytes(b"unreviewed")

            def stale_coverage(corpus):
                (corpus / "coverage.json").write_bytes(
                    _canonical_json_bytes(build_coverage(()).to_dict())
                )

            mutations = {
                "missing audit": missing_audit,
                "extra audit": extra_audit,
                "malformed audit": malformed_audit,
                "missing receipt": missing_receipt,
                "extra receipt": extra_receipt,
                "mismatched receipt": mismatched_receipt,
                "missing artifact": missing_artifact,
                "extra artifact": extra_artifact,
                "stale coverage": stale_coverage,
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    corpus = root / f"corpus-{name.replace(' ', '-')}"
                    output = root / f"frozen-{name.replace(' ', '-')}"
                    shutil.copytree(audited, corpus)
                    mutate(corpus)
                    before = _tree_bytes(corpus)
                    errors = io.StringIO()
                    with redirect_stderr(errors):
                        status = main(
                            [
                                "freeze",
                                "--corpus",
                                str(corpus),
                                "--output",
                                str(output),
                            ]
                        )
                    self.assertEqual(status, 2)
                    self.assertTrue(errors.getvalue().strip())
                    self.assertFalse(output.exists())
                    self.assertEqual(_tree_bytes(corpus), before)
                    self.assertEqual(
                        list(root.glob(f".{output.name}.*")),
                        [],
                    )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_inventory_validates_adapters_and_pinned_lafan_provenance(self):
        """Catches glob counts, folder-name provenance, or mutable output."""

        from mm_sonic.terrain_oracle.corpus_cli import main
        from mm_sonic.terrain_oracle.source_lafan import (
            LAFAN1_UPSTREAM_REVISION,
        )
        from tests.python.test_oracle_source_lafan import _write_lafan_csv
        from tests.python.torch_motion_test_utils import write_takara_clip

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flat = root / "flat"
            flat_source = write_takara_clip(
                flat / "walk",
                frames=60,
            )
            release = root / "lafan-release"
            lafan = release / "g1"
            lafan.mkdir(parents=True)
            lafan_source = _write_lafan_csv(
                lafan / "walk1_subject1.csv",
                frames=4,
            )
            (release / "README.md").write_text(
                "LAFAN1 is licensed under Creative Commons "
                "Attribution-NonCommercial-NoDerivatives 4.0 "
                "International Public License.\n"
            )
            (release / "LICENSE").write_text(
                "BSD 3-Clause License\n\n"
                "Redistribution and use in source and binary forms are "
                "permitted.\n"
            )
            for relative_path in (
                "README.md",
                "LICENSE",
                "g1/walk1_subject1.csv",
            ):
                _write_huggingface_metadata(release, relative_path)
            output = root / "inventory.json"

            try:
                status = main(
                    [
                        "inventory",
                        "--flat",
                        str(flat),
                        "--lafan",
                        str(lafan),
                        "--output",
                        str(output),
                    ]
                )
            except BaseException:
                self.assertFalse(output.exists())
                raise
            self.assertEqual(status, 0)
            payload = output.read_bytes()
            inventory = json.loads(payload.decode("ascii"))
            self.assertEqual(
                set(inventory),
                {
                    "schema",
                    "sources",
                    "summary",
                    "content_sha256",
                },
            )
            self.assertEqual(
                inventory["schema"],
                "terrain-oracle-source-inventory/v1",
            )
            without_hash = dict(inventory)
            content_sha256 = without_hash.pop("content_sha256")
            self.assertEqual(
                content_sha256,
                hashlib.sha256(
                    _canonical_json_bytes(without_hash)
                ).hexdigest(),
            )
            self.assertEqual(set(inventory["sources"]), {"flat", "lafan"})
            flat_inventory = inventory["sources"]["flat"]
            self.assertEqual(flat_inventory["count"], 1)
            self.assertEqual(
                flat_inventory["counts_by_format"],
                {"takara-motion-npz-v1": 1},
            )
            self.assertEqual(
                flat_inventory["records"][0]["source_sha256"],
                hashlib.sha256(flat_source.read_bytes()).hexdigest(),
            )
            lafan_inventory = inventory["sources"]["lafan"]
            self.assertEqual(lafan_inventory["count"], 1)
            self.assertEqual(
                lafan_inventory["repository"],
                "lvhaidong/LAFAN1_Retargeting_Dataset",
            )
            self.assertEqual(
                lafan_inventory["revision"],
                LAFAN1_UPSTREAM_REVISION,
            )
            self.assertEqual(
                lafan_inventory["license_id"],
                "CC-BY-NC-ND-4.0",
            )
            self.assertEqual(
                lafan_inventory["counts_by_format"],
                {
                    "lafan1-retargeted-g1-csv-30hz@"
                    + LAFAN1_UPSTREAM_REVISION: 1
                },
            )
            self.assertEqual(
                lafan_inventory["records"][0]["source_sha256"],
                hashlib.sha256(lafan_source.read_bytes()).hexdigest(),
            )
            self.assertEqual(inventory["summary"]["clip_count"], 2)
            self.assertEqual(
                payload,
                _canonical_json_bytes(inventory),
            )

            before = output.read_bytes()
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(
                    main(
                        [
                            "inventory",
                            "--flat",
                            str(flat),
                            "--lafan",
                            str(lafan),
                            "--output",
                            str(output),
                        ]
                    ),
                    2,
                )
            self.assertIn("exist", errors.getvalue().lower())
            self.assertEqual(output.read_bytes(), before)

    def test_main_returns_parse_status_without_leaking_system_exit(self):
        """Catches programmatic CLI parse failures escaping as exceptions."""

        from mm_sonic.terrain_oracle import corpus_main
        from mm_sonic.terrain_oracle.corpus_cli import main

        output = io.StringIO()
        with redirect_stderr(output):
            self.assertEqual(main([]), 2)
            self.assertEqual(main(["unknown-command"]), 2)
            self.assertEqual(main(["import", "--fixture"]), 2)
        self.assertIn("error", output.getvalue().lower())
        help_output = io.StringIO()
        from contextlib import redirect_stdout

        with redirect_stdout(help_output):
            self.assertEqual(main(["--help"]), 0)
            self.assertEqual(corpus_main(["--help"]), 0)
        self.assertIn("inventory", help_output.getvalue())
        self.assertIn("freeze", help_output.getvalue())

    def test_import_rejects_symlink_aliases_and_parent_traversal(self):
        """Catches resolving unsafe paths before checking their spelling."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, _, _ = _write_fixture_bundle(root)

            fixture_alias = root / "fixture-alias"
            fixture_alias.symlink_to(fixture, target_is_directory=True)
            alias_output = root / "alias-output"
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(
                    main(
                        [
                            "import",
                            "--fixture",
                            str(fixture_alias),
                            "--output",
                            str(alias_output),
                        ]
                    ),
                    2,
                )
            self.assertFalse(alias_output.exists())

            real_parent = root / "real-parent"
            real_parent.mkdir()
            parent_alias = root / "parent-alias"
            parent_alias.symlink_to(
                real_parent,
                target_is_directory=True,
            )
            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "import",
                            "--fixture",
                            str(fixture),
                            "--output",
                            str(parent_alias / "published"),
                        ]
                    ),
                    2,
                )
            self.assertFalse((real_parent / "published").exists())

            traversal_output = root / "narrow" / ".." / "escaped"
            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "import",
                            "--fixture",
                            str(fixture),
                            "--output",
                            str(traversal_output),
                        ]
                    ),
                    2,
                )
            self.assertFalse((root / "escaped").exists())


if __name__ == "__main__":
    unittest.main()
