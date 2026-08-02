from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import errno
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

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
    COMPLETION_MARKER,
    _seal_directory,
    load_corpus,
    mesh_digest,
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
    _seal_directory(
        bundle,
        excluded_top_level=("coverage.json", "render-audit"),
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


def _write_source_inventory(
    path: Path,
    sources: dict[str, dict[str, object]],
) -> dict[str, object]:
    counts_by_source = {
        name: int(source["count"])
        for name, source in sorted(sources.items())
    }
    counts_by_format: dict[str, int] = {}
    for source in sources.values():
        for name, count in source["counts_by_format"].items():
            counts_by_format[name] = counts_by_format.get(name, 0) + int(
                count
            )
    without_hash = {
        "schema": "terrain-oracle-source-inventory/v1",
        "sources": sources,
        "summary": {
            "clip_count": sum(counts_by_source.values()),
            "counts_by_source": counts_by_source,
            "counts_by_format": dict(sorted(counts_by_format.items())),
        },
    }
    document = {
        **without_hash,
        "content_sha256": hashlib.sha256(
            _canonical_json_bytes(without_hash)
        ).hexdigest(),
    }
    path.write_bytes(_canonical_json_bytes(document))
    return document


def _reseal_corpus_directory(path: Path) -> None:
    (path / COMPLETION_MARKER).unlink()
    _seal_directory(
        path,
        excluded_top_level=("coverage.json", "render-audit"),
    )


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
    _seal_directory(
        bundle,
        excluded_top_level=("coverage.json", "render-audit"),
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


def _write_hanging_renderer(path: Path) -> Path:
    path.write_text(
        """from pathlib import Path
import subprocess
import sys
import time

marker = Path(sys.argv[1])
child = (
    "from pathlib import Path; import sys, time; "
    "time.sleep(0.7); Path(sys.argv[1]).write_text('orphan')"
)
subprocess.Popen([sys.executable, "-c", child, str(marker)])
time.sleep(60.0)
"""
    )
    return path


def _write_early_exit_renderer(path: Path) -> Path:
    path.write_text(
        """from pathlib import Path
import subprocess
import sys

marker = Path(sys.argv[1])
child = (
    "from pathlib import Path; import sys, time; "
    "time.sleep(0.7); Path(sys.argv[1]).write_text('orphan')"
)
subprocess.Popen([sys.executable, "-c", child, str(marker)])
raise SystemExit(7)
"""
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


def _grail_record_fixture(
    root: Path,
    *,
    frame_count: int = 4,
) -> tuple[object, dict[str, object]]:
    robot = root / "robot.pkl"
    terrain = root / "terrain.usd"
    metadata = root / "clips.json"
    robot.write_bytes(b"trusted-pickle")
    terrain.write_bytes(b"trusted-terrain")
    metadata.write_bytes(b"{}")
    pose_source = "clips.json:test"
    record = mock.Mock(
        family="c490_stair_p1",
        robot_path=robot,
        usd_path=terrain,
        shard_path=root,
        stem="verified-record",
        n_frames=frame_count,
        terrain_position_env=np.array(
            (1.0, 2.0, 3.0),
            dtype=np.float32,
        ),
        terrain_rotation_env_wxyz=np.array(
            (1.0, 0.0, 0.0, 0.0),
            dtype=np.float32,
        ),
        pose_source=pose_source,
    )

    def identity(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "relative_path": path.name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    return record, {
        "family": record.family,
        "stem": record.stem,
        "frame_count": frame_count,
        "robot": {
            **identity(robot),
            "path": str(robot.resolve()),
            "license_id": "UNRECORDED",
        },
        "terrain": {
            **identity(terrain),
            "path": str(terrain.resolve()),
            "license_id": "UNRECORDED",
            "world_from_terrain": {
                "translation_world": [1.0, 2.0, 3.0],
                "quaternion_world_from_local_wxyz": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            },
        },
        "shard_metadata": identity(metadata),
        "pose_source": pose_source,
    }


class CorpusCliTests(unittest.TestCase):
    def test_trusted_media_rejects_self_consistent_unpinned_model_vfs(self):
        """Catches accepting an attacker-chosen model tree and matching digest."""

        import mujoco
        from mm_sonic.terrain_oracle.corpus_cli import _trusted_media_model

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "attacker.xml"
            model_path.write_text(
                "<mujoco><worldbody/></mujoco>\n",
                encoding="ascii",
            )
            payload = model_path.read_bytes()
            model = mujoco.MjModel.from_xml_path(str(model_path))
            record = {
                "asset_path": str(model_path),
                "asset_size_bytes": len(payload),
                "asset_sha256": hashlib.sha256(payload).hexdigest(),
                "structural_sha256": structural_model_sha256(model),
            }
            with self.assertRaisesRegex(ContractError, "pinned exact G1"):
                _trusted_media_model(record)

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_trusted_media_publishes_pinned_dependency_vfs_authority(self):
        """Catches omitting the fixed G1 visual dependency tree authority."""

        import mujoco
        from mm_sonic.terrain_oracle.corpus_cli import _trusted_media_model

        payload = MODEL_PATH.read_bytes()
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        trusted = _trusted_media_model(
            {
                "asset_path": str(MODEL_PATH.resolve()),
                "asset_size_bytes": len(payload),
                "asset_sha256": hashlib.sha256(payload).hexdigest(),
                "structural_sha256": structural_model_sha256(model),
            }
        )
        self.assertEqual(
            trusted["dependency_vfs"],
            {
                "root_path": str(MODEL_PATH.parent.resolve()),
                "root_relative_path": MODEL_PATH.name,
                "directory_count": 2,
                "file_count": 88,
                "size_bytes": 60_248_622,
                "tree_sha256": (
                    "e41a1311012dd9eaf3d5e733d5aefc82"
                    "62f6024dd5c2f8e55dc617ab1db3977d"
                ),
            },
        )

    def test_real_import_snapshots_mutable_source_trees_before_adapters(self):
        """Catches adapters reopening live flat, Justin, or LAFAN source bytes."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _snapshot_real_import_roots,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flat = root / "flat"
            justin = root / "justin.zarr"
            release = root / "lafan-release"
            lafan = release / "g1"
            for directory in (flat, justin, lafan):
                directory.mkdir(parents=True)
            (flat / "motion.npz").write_bytes(b"flat-before")
            (justin / "zarr.json").write_bytes(b"justin-before")
            (lafan / "walk.csv").write_bytes(b"lafan-before")
            (release / "README.md").write_bytes(b"readme-before")

            with _snapshot_real_import_roots(
                flat,
                justin,
                lafan,
            ) as (flat_snapshot, justin_snapshot, lafan_snapshot):
                (flat / "motion.npz").write_bytes(b"flat-after!")
                (justin / "zarr.json").write_bytes(b"justin-after!")
                (lafan / "walk.csv").write_bytes(b"lafan-after!")
                self.assertEqual(
                    (flat_snapshot / "motion.npz").read_bytes(),
                    b"flat-before",
                )
                self.assertEqual(
                    (justin_snapshot / "zarr.json").read_bytes(),
                    b"justin-before",
                )
                self.assertEqual(
                    (lafan_snapshot / "walk.csv").read_bytes(),
                    b"lafan-before",
                )
                self.assertEqual(
                    (lafan_snapshot.parent / "README.md").read_bytes(),
                    b"readme-before",
                )
                self.assertFalse(
                    flat_snapshot.is_relative_to(root)
                )

    def test_real_import_keeps_primary_and_lafan_inventory_authority_distinct(
        self,
    ):
        """Catches silently replacing or merging away the pinned LAFAN manifest."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _load_import_inventories,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path = root / "inventory.json"
            lafan_path = root / "lafan-inventory.json"
            primary = _write_source_inventory(
                primary_path,
                {
                    name: {
                        "count": count,
                        "counts_by_format": {f"{name}-format": count},
                    }
                    for name, count in (
                        ("flat", 173),
                        ("justin", 18),
                        ("grail", 489),
                    )
                },
            )
            lafan = _write_source_inventory(
                lafan_path,
                {
                    "lafan": {
                        "count": 40,
                        "counts_by_format": {"lafan-format": 40},
                    }
                },
            )

            with self.assertRaisesRegex(
                ContractError,
                "trusted phase-1 inventory",
            ):
                _load_import_inventories(primary_path, lafan_path)
            loaded_primary, loaded_lafan, sources = _load_import_inventories(
                primary_path,
                lafan_path,
                primary_sha256=hashlib.sha256(
                    primary_path.read_bytes()
                ).hexdigest(),
                lafan_sha256=hashlib.sha256(
                    lafan_path.read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(loaded_primary, primary)
            self.assertEqual(loaded_lafan, lafan)
            self.assertEqual(
                set(sources),
                {"flat", "justin", "grail", "lafan"},
            )
            self.assertNotEqual(
                loaded_primary["content_sha256"],
                loaded_lafan["content_sha256"],
            )

            wrong_lafan = root / "wrong-lafan.json"
            _write_source_inventory(
                wrong_lafan,
                {
                    "flat": {
                        "count": 40,
                        "counts_by_format": {"lafan-format": 40},
                    }
                },
            )
            with self.assertRaises(ContractError):
                _load_import_inventories(
                    primary_path,
                    wrong_lafan,
                    primary_sha256=hashlib.sha256(
                        primary_path.read_bytes()
                    ).hexdigest(),
                    lafan_sha256=hashlib.sha256(
                        wrong_lafan.read_bytes()
                    ).hexdigest(),
                )

    def test_terrain_recipe_composes_world_plane_in_obstacle_local_frame(self):
        """Catches GRAIL meshes that omit runout floor or double-apply pose."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _bind_terrain_recipe,
        )
        from mm_sonic.terrain_oracle.math3d import RigidTransform

        model_record = {
            "asset_path": "/authority/g1.xml",
            "asset_size_bytes": 123,
            "asset_sha256": "a" * 64,
            "structural_sha256": "b" * 64,
        }
        obstacle = {
            "kind": "grail-usd",
            "path": "/authority/terrain.usd",
            "size_bytes": 456,
            "sha256": "c" * 64,
            "license_id": "UNRECORDED",
        }
        obstacle_mesh = CanonicalTerrainMesh(
            vertices_local=np.array(
                ((0.0, 0.0, 0.2), (1.0, 0.0, 0.2), (0.0, 1.0, 0.2)),
                dtype=np.float32,
            ),
            faces=np.array(((0, 1, 2),), dtype=np.int32),
            valid_faces=np.array((True,), dtype=np.bool_),
            source_asset_sha256="c" * 64,
        )
        transform = RigidTransform(
            np.array((2.0, -3.0, 0.5), dtype=np.float32),
            np.array(
                (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
                dtype=np.float32,
            ),
        )

        mesh, binding, recipe = _bind_terrain_recipe(
            model_record,
            obstacle,
            obstacle_mesh,
            transform,
        )
        repeated = _bind_terrain_recipe(
            model_record,
            obstacle,
            obstacle_mesh,
            transform,
        )

        self.assertEqual(
            binding.asset_sha256,
            mesh.source_asset_sha256,
        )
        self.assertEqual(binding.mesh_sha256, mesh_digest(mesh))
        self.assertEqual(binding.asset_path, f"recipe://{binding.asset_sha256}")
        self.assertEqual(recipe["schema"], "terrain-oracle-terrain-recipe/v1")
        self.assertEqual(recipe["obstacle"], obstacle)
        self.assertEqual(mesh.vertices_local.shape[0], 7)
        plane_world = transform.apply_points(mesh.vertices_local[-4:])
        self.assertTrue(
            np.array_equal(
                plane_world,
                np.array(
                    (
                        (-16.0, -16.0, 0.0),
                        (16.0, -16.0, 0.0),
                        (16.0, 16.0, 0.0),
                        (-16.0, 16.0, 0.0),
                    ),
                    dtype=np.float32,
                ),
            )
        )
        self.assertEqual(
            binding.mesh_sha256,
            repeated[1].mesh_sha256,
        )
        self.assertEqual(binding.asset_sha256, repeated[1].asset_sha256)

    def test_justin_urdf_boxes_are_hashed_and_triangulated_strictly(self):
        """Catches folder-name provenance or a visually implied stair surface."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _load_justin_obstacle,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stairs.urdf"
            collisions = "\n".join(
                (
                    "<link name='box-{index}'><collision>"
                    "<origin xyz='{x} 0 {z}' rpy='0 0 0'/>"
                    "<geometry><box size='1 2 0.2'/></geometry>"
                    "</collision></link>"
                ).format(index=index, x=index, z=0.1 + index * 0.2)
                for index in range(4)
            )
            path.write_text(f"<robot name='stairs'>{collisions}</robot>")

            obstacle, mesh = _load_justin_obstacle(path)

            payload = path.read_bytes()
            self.assertEqual(
                obstacle,
                {
                    "kind": "justin-urdf",
                    "path": str(path.resolve()),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "license_id": "UNRECORDED",
                },
            )
            self.assertEqual(mesh.vertices_local.shape, (32, 3))
            self.assertEqual(mesh.faces.shape, (48, 3))
            self.assertTrue(mesh.valid_faces.all())
            self.assertEqual(
                mesh.source_asset_sha256,
                obstacle["sha256"],
            )
            normals = np.cross(
                mesh.vertices_local[mesh.faces[:, 1]]
                - mesh.vertices_local[mesh.faces[:, 0]],
                mesh.vertices_local[mesh.faces[:, 2]]
                - mesh.vertices_local[mesh.faces[:, 0]],
            )
            self.assertEqual(int(np.count_nonzero(normals[:, 2] > 0.0)), 8)

    def test_grail_same_size_swap_is_rejected_before_joblib_execution(self):
        """Catches an inventory-to-deserialization PKL substitution window."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _load_verified_grail_source,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, inventory_record = _grail_record_fixture(root)
            robot = record.robot_path
            robot.write_bytes(b"hostile-pickle")

            with mock.patch("joblib.load") as unsafe_load:
                with self.assertRaises(ContractError):
                    _load_verified_grail_source(
                        record,
                        inventory_record,
                        mock.sentinel.fk,
                        grail_root=root,
                    )
            unsafe_load.assert_not_called()

    def test_grail_semantic_swap_is_rejected_before_joblib_execution(self):
        """Catches clips.json changing frames or terrain pose after inventory."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _load_verified_grail_source,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, inventory_record = _grail_record_fixture(root)
            inventory_record["frame_count"] = 5
            with mock.patch("joblib.load") as unsafe_load:
                with self.assertRaisesRegex(
                    ContractError,
                    "metadata|semantic|inventory",
                ):
                    _load_verified_grail_source(
                        record,
                        inventory_record,
                        mock.sentinel.fk,
                        grail_root=root,
                    )
            unsafe_load.assert_not_called()

    def test_import_accepts_explicit_real_source_authority(self):
        """Catches a fixture-only importer that cannot publish audited sources."""

        from mm_sonic.terrain_oracle import corpus_cli

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = [
                "import",
                "--inventory",
                str(root / "inventory.json"),
                "--lafan-inventory",
                str(root / "lafan-inventory.json"),
                "--flat",
                str(root / "flat"),
                "--justin",
                str(root / "justin.zarr"),
                "--grail-root",
                str(root / "grail"),
                "--grail-families",
                "c490_stair_p1,c490_curb",
                "--lafan",
                str(root / "lafan" / "g1"),
                "--model",
                str(root / "g1.xml"),
                "--justin-terrain",
                str(root / "stairs.urdf"),
                "--output",
                str(root / "corpus"),
            ]
            with mock.patch.object(
                corpus_cli,
                "_import_real_sources",
                create=True,
            ) as import_real_sources:
                self.assertEqual(corpus_cli.main(arguments), 0)

            import_real_sources.assert_called_once()
            namespace = import_real_sources.call_args.args[0]
            self.assertIsNone(namespace.fixture)
            self.assertEqual(namespace.inventory, root / "inventory.json")
            self.assertEqual(
                namespace.lafan_inventory,
                root / "lafan-inventory.json",
            )
            self.assertEqual(namespace.flat, root / "flat")
            self.assertEqual(namespace.justin, root / "justin.zarr")
            self.assertEqual(namespace.grail_root, root / "grail")
            self.assertEqual(
                namespace.grail_families,
                "c490_stair_p1,c490_curb",
            )
            self.assertEqual(namespace.lafan, root / "lafan" / "g1")
            self.assertEqual(namespace.model, root / "g1.xml")
            self.assertEqual(
                namespace.justin_terrain,
                root / "stairs.urdf",
            )
            self.assertEqual(namespace.output, root / "corpus")

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
    def test_audit_publishes_unbound_clip_as_terrain_registration_rejection(
        self,
    ):
        """Catches aborting the corpus instead of retaining rejection evidence."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "unbound"
            source.mkdir()
            record = write_clip(
                source / "clips",
                synthetic_canonical_clip(frames=8),
            )
            manifest_only = publish_corpus(
                root / "unbound-manifest",
                (record,),
                {"mesh_records": []},
            )
            (source / "manifest.json").write_bytes(
                (manifest_only / "manifest.json").read_bytes()
            )
            _seal_directory(
                source,
                excluded_top_level=("coverage.json", "render-audit"),
            )
            audited = root / "audited"

            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(source),
                        "--model",
                        str(MODEL_PATH),
                        "--output",
                        str(audited),
                    ]
                ),
                0,
            )
            index = json.loads(
                (audited / "audit-index.json").read_text("ascii")
            )
            report_path = (
                audited / index["reports"][0]["relative_path"]
            )
            report = json.loads(report_path.read_text("ascii"))
            self.assertEqual(report["status"], "rejected")
            self.assertEqual(report["terrain_sha256"], "0" * 64)
            self.assertEqual(report["accepted_intervals"], [])
            self.assertEqual(
                [reason["code"] for reason in report["reasons"]],
                ["terrain_registration"],
            )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_directory_publication_uses_nfs_completion_protocol_on_einval(
        self,
    ):
        """Catches renameat2-only import, audit, render, or freeze publication."""

        from mm_sonic.terrain_oracle import (
            coverage as oracle_coverage,
            storage as oracle_storage,
        )
        from mm_sonic.terrain_oracle.corpus_cli import (
            _load_frozen_corpus,
            _load_render_evidence,
            main,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            rendered = audited / "render-audit"
            frozen = root / "frozen"
            forced_einval = OSError(
                errno.EINVAL,
                "forced renameat2 RENAME_NOREPLACE rejection",
            )
            with mock.patch.object(
                oracle_storage,
                "_rename_noreplace",
                side_effect=forced_einval,
            ), mock.patch.object(
                oracle_coverage,
                "_rename_noreplace",
                side_effect=forced_einval,
            ):
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
                            "--output",
                            str(rendered),
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

            for published in (raw, audited, rendered, frozen):
                marker = published / COMPLETION_MARKER
                self.assertTrue(marker.is_file())
                document = json.loads(marker.read_text("ascii"))
                self.assertEqual(
                    set(document),
                    {
                        "schema",
                        "excluded_top_level",
                        "file_count",
                        "tree_sha256",
                    },
                )
                self.assertEqual(
                    document["schema"],
                    "terrain-oracle-complete/v1",
                )
                self.assertEqual(
                    document["excluded_top_level"],
                    (
                        ["coverage.json", "render-audit"]
                        if published in (raw, audited)
                        else []
                    ),
                )
            load_corpus(raw)
            _load_render_evidence(audited)
            _load_frozen_corpus(frozen)

    def test_completion_marker_rejects_missing_tampered_and_extra_content(self):
        """Catches readers treating an incomplete or changed tree as published."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, _, _ = _write_fixture_bundle(root)

            missing = root / "missing-marker"
            shutil.copytree(fixture, missing)
            (missing / COMPLETION_MARKER).unlink()

            tampered = root / "tampered-file"
            shutil.copytree(fixture, tampered)
            with (tampered / "manifest.json").open("ab") as stream:
                stream.write(b" ")

            extra = root / "unexpected-file"
            shutil.copytree(fixture, extra)
            (extra / "untrusted.bin").write_bytes(b"not in the sealed tree")

            for case in (missing, tampered, extra):
                with self.subTest(case=case.name):
                    with self.assertRaisesRegex(
                        ContractError,
                        "completion|digest",
                    ):
                        load_corpus(case)

    def test_nfs_copy_claim_preserves_preexisting_and_foreign_destinations(self):
        """Catches fallback publication overwriting or deleting another writer."""

        from mm_sonic.terrain_oracle import storage as oracle_storage

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, _, _ = _write_fixture_bundle(root)
            forced_einval = OSError(
                errno.EINVAL,
                "forced renameat2 RENAME_NOREPLACE rejection",
            )

            preexisting = root / "preexisting"
            preexisting.mkdir()
            (preexisting / "sentinel").write_bytes(b"owned elsewhere")
            before = _tree_bytes(preexisting)
            with mock.patch.object(
                oracle_storage,
                "_rename_noreplace",
                side_effect=forced_einval,
            ):
                with self.assertRaises(FileExistsError):
                    oracle_storage._publish_directory_no_replace(
                        fixture,
                        preexisting,
                    )
            self.assertEqual(_tree_bytes(preexisting), before)
            self.assertTrue(fixture.is_dir())

            original_write = oracle_storage._write_no_replace
            owned_failure = root / "owned-failure"

            def fail_owned_copy(path: Path, payload: bytes) -> None:
                if (
                    owned_failure in path.parents
                    and path.name != oracle_storage._OWNER_MARKER
                ):
                    raise OSError(errno.EIO, "forced copy failure")
                original_write(path, payload)

            with mock.patch.object(
                oracle_storage,
                "_write_no_replace",
                side_effect=fail_owned_copy,
            ):
                with self.assertRaisesRegex(OSError, "forced copy failure"):
                    oracle_storage._copy_directory_claim_noreplace(
                        fixture,
                        owned_failure,
                    )
            self.assertFalse(owned_failure.exists())
            self.assertTrue(fixture.is_dir())

            foreign_failure = root / "foreign-failure"
            foreign_owner = b"terrain-oracle-owner-v1:foreign-writer\n"

            def fail_after_owner_change(path: Path, payload: bytes) -> None:
                if (
                    foreign_failure in path.parents
                    and path.name != oracle_storage._OWNER_MARKER
                ):
                    (
                        foreign_failure / oracle_storage._OWNER_MARKER
                    ).write_bytes(foreign_owner)
                    raise OSError(errno.EIO, "forced foreign copy failure")
                original_write(path, payload)

            with mock.patch.object(
                oracle_storage,
                "_write_no_replace",
                side_effect=fail_after_owner_change,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "forced foreign copy failure",
                ):
                    oracle_storage._copy_directory_claim_noreplace(
                        fixture,
                        foreign_failure,
                    )
            self.assertTrue(foreign_failure.is_dir())
            self.assertEqual(
                (
                    foreign_failure / oracle_storage._OWNER_MARKER
                ).read_bytes(),
                foreign_owner,
            )

            inode_changed = root / "inode-changed"
            inode_changed.mkdir()
            owner_payload = b"terrain-oracle-owner-v1:original-writer\n"
            owner_path = inode_changed / oracle_storage._OWNER_MARKER
            owner_path.write_bytes(owner_payload)
            metadata = inode_changed.stat()
            identity = (metadata.st_dev, metadata.st_ino)
            inode_changed.rename(root / "displaced-owned-directory")
            inode_changed.mkdir()
            (inode_changed / oracle_storage._OWNER_MARKER).write_bytes(
                owner_payload
            )
            oracle_storage._cleanup_owned_destination(
                inode_changed,
                identity=identity,
                owner_payload=owner_payload,
            )
            self.assertTrue(inode_changed.is_dir())

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
    def test_audit_loader_rejects_rehashed_forged_model_and_terrain_authority(
        self,
    ):
        """Catches internally consistent reports detached from exact authority."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _load_audit_evidence,
            main,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
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
            _load_audit_evidence(audited)

            for field, forged in (
                ("model_sha256", "f" * 64),
                ("terrain_sha256", "e" * 64),
            ):
                with self.subTest(field=field):
                    variant = root / f"forged-{field}"
                    shutil.copytree(audited, variant)
                    index_path = variant / "audit-index.json"
                    index = json.loads(index_path.read_text("ascii"))
                    entry = index["reports"][0]
                    report_path = variant / entry["relative_path"]
                    report = json.loads(report_path.read_text("ascii"))
                    report[field] = forged
                    report_payload = _canonical_json_bytes(report)
                    report_path.write_bytes(report_payload)
                    entry[field] = forged
                    entry["sha256"] = hashlib.sha256(
                        report_payload
                    ).hexdigest()
                    _rewrite_hashed_document(index_path, index)
                    _reseal_corpus_directory(variant)

                    with self.assertRaisesRegex(
                        ContractError,
                        "model|terrain|authority|binding",
                    ):
                        _load_audit_evidence(variant)

            for case, mutate_audit_metadata in (
                (
                    "source-corpus",
                    lambda audit: audit.__setitem__(
                        "source_corpus_manifest_sha256",
                        "d" * 64,
                    ),
                ),
                (
                    "model",
                    lambda audit: audit["model"].__setitem__(
                        "structural_sha256",
                        "c" * 64,
                    ),
                ),
                (
                    "extra-field",
                    lambda audit: audit.__setitem__("untrusted", True),
                ),
            ):
                with self.subTest(manifest_audit=case):
                    variant = root / f"forged-manifest-{case}"
                    shutil.copytree(audited, variant)
                    manifest_path = variant / "manifest.json"
                    manifest = json.loads(manifest_path.read_text("ascii"))
                    mutate_audit_metadata(manifest["metadata"]["audit"])
                    manifest_payload = _canonical_json_bytes(manifest)
                    manifest_path.write_bytes(manifest_payload)
                    index_path = variant / "audit-index.json"
                    index = json.loads(index_path.read_text("ascii"))
                    index["corpus_manifest_sha256"] = hashlib.sha256(
                        manifest_payload
                    ).hexdigest()
                    _rewrite_hashed_document(index_path, index)
                    _reseal_corpus_directory(variant)

                    with self.assertRaisesRegex(
                        ContractError,
                        "manifest|model|source|metadata|authority|binding",
                    ):
                        _load_audit_evidence(variant)

            coordinated = root / "forged-coordinated-source"
            shutil.copytree(audited, coordinated)
            manifest_path = coordinated / "manifest.json"
            manifest = json.loads(manifest_path.read_text("ascii"))
            forged_source_sha256 = "b" * 64
            manifest["metadata"]["audit"][
                "source_corpus_manifest_sha256"
            ] = forged_source_sha256
            manifest_payload = _canonical_json_bytes(manifest)
            manifest_path.write_bytes(manifest_payload)
            index_path = coordinated / "audit-index.json"
            index = json.loads(index_path.read_text("ascii"))
            index[
                "source_corpus_manifest_sha256"
            ] = forged_source_sha256
            index["corpus_manifest_sha256"] = hashlib.sha256(
                manifest_payload
            ).hexdigest()
            _rewrite_hashed_document(index_path, index)
            _reseal_corpus_directory(coordinated)

            with self.assertRaisesRegex(
                ContractError,
                "source.*manifest|manifest.*source|authority|binding",
            ):
                _load_audit_evidence(coordinated)

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_external_renderer_cannot_publish_acceptable_evidence(self):
        """Catches restoring an arbitrary-byte production renderer escape."""

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
                        str(Path(sys.executable).resolve()),
                        "--renderer-arg",
                        str(renderer),
                        "--output",
                        str(renders),
                    ]
                )
            except BaseException:
                self.assertFalse(renders.exists())
                raise
            self.assertEqual(status, 2)
            self.assertFalse(renders.exists())
            self.assertEqual(
                list(audited.glob(".render-audit.*")),
                [],
            )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_default_render_audit_uses_authenticated_package_media(self):
        """Catches production render evidence backed only by arbitrary bytes."""

        from mm_sonic.terrain_oracle.corpus_cli import (
            _load_render_evidence,
            main,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_audit_fixture(root)
            audited = root / "audited"
            rendered = audited / "render-audit"
            self.assertEqual(
                main(
                    [
                        "audit",
                        "--corpus",
                        str(source),
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
                        "--output",
                        str(rendered),
                    ]
                ),
                0,
            )

            index = _load_render_evidence(audited)
            self.assertEqual(
                index["schema"],
                "terrain-oracle-render-index/v2",
            )
            self.assertEqual(
                index["renderer"]["mode"],
                "trusted-package-media-v1",
            )
            receipt_path = (
                rendered
                / index["interval_receipts"][0]["relative_path"]
            )
            receipt = json.loads(receipt_path.read_text("ascii"))
            self.assertEqual(
                receipt["renderer"]["result"]["schema"],
                "terrain-oracle-render-result/v1",
            )
            self.assertTrue(
                receipt["renderer"]["result"]["completed"]
            )
            request_path = (
                rendered
                / receipt["renderer"]["request"]["relative_path"]
            )
            request = json.loads(request_path.read_text("ascii"))
            self.assertEqual(
                receipt["renderer"]["result"]["render_evidence"][
                    "model_dependency_vfs"
                ],
                request["model"]["dependency_vfs"],
            )
            self.assertEqual(
                request["model"]["dependency_vfs"]["tree_sha256"],
                (
                    "e41a1311012dd9eaf3d5e733d5aefc82"
                    "62f6024dd5c2f8e55dc617ab1db3977d"
                ),
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
                "terrain-oracle-render-index/v2",
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
                index["renderer"]["argv_prefix"],
                [
                    str(Path(sys.executable).resolve()),
                    "-B",
                    "-m",
                    "mm_sonic.terrain_oracle.render_media",
                ],
            )
            self.assertEqual(
                index["renderer"]["mode"],
                "trusted-package-media-v1",
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
                        "-B",
                        "-m",
                        "mm_sonic.terrain_oracle.render_media",
                    ],
                )
                self.assertEqual(argv[4], "--request")
                self.assertEqual(
                    receipt["renderer"]["invocation_sha256"],
                    hashlib.sha256(
                        _canonical_json_bytes(argv)
                    ).hexdigest(),
                )
                self.assertIs(receipt["renderer"]["shell"], False)
                self.assertEqual(receipt["renderer"]["returncode"], 0)
                self.assertEqual(
                    receipt["renderer"]["result"]["interval_keys"],
                    accepted_keys,
                )
                self.assertEqual(
                    set(receipt["renderer"]["request"]),
                    {"relative_path", "sha256", "size_bytes"},
                )
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
                                str(Path(sys.executable).resolve()),
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
    def test_renderer_path_rejects_symlink_before_resolution(self):
        """Catches direct and ancestor symlinks erased by eager resolution."""

        from mm_sonic.terrain_oracle.corpus_cli import main

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
            executable = Path(sys.executable).resolve()
            direct_alias = root / "python-alias"
            direct_alias.symlink_to(executable)
            ancestor_alias = root / "bin-alias"
            ancestor_alias.symlink_to(
                executable.parent,
                target_is_directory=True,
            )
            for case, renderer_path in (
                ("direct", direct_alias),
                ("ancestor", ancestor_alias / executable.name),
            ):
                with self.subTest(case=case):
                    corpus = root / f"audited-{case}"
                    shutil.copytree(audited, corpus)
                    output = corpus / "render-audit"
                    with redirect_stderr(io.StringIO()):
                        status = main(
                            [
                                "render-audit",
                                "--corpus",
                                str(corpus),
                                "--renderer",
                                str(renderer_path),
                                "--renderer-arg",
                                str(renderer),
                                "--output",
                                str(output),
                            ]
                        )
                    self.assertEqual(status, 2)
                    self.assertFalse(output.exists())
                    self.assertEqual(
                        list(corpus.glob(".render-audit.*")),
                        [],
                    )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_renderer_timeout_kills_process_group_and_cleans_stage(self):
        """Catches an unbounded renderer or a surviving renderer child."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            output = audited / "render-audit"
            marker = root / "orphan-marker"
            renderer = _write_hanging_renderer(root / "hang.py")
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
            errors = io.StringIO()
            started = time.monotonic()
            with redirect_stderr(errors):
                status = main(
                    [
                        "render-audit",
                        "--corpus",
                        str(audited),
                        "--renderer",
                        str(Path(sys.executable).resolve()),
                        "--renderer-arg",
                        str(renderer),
                        "--renderer-arg",
                        str(marker),
                        "--renderer-timeout-seconds",
                        "0.1",
                        "--output",
                        str(output),
                    ]
                )
            elapsed = time.monotonic() - started

            self.assertEqual(status, 2)
            self.assertIn("timed out", errors.getvalue().lower())
            self.assertLess(elapsed, 3.0)
            time.sleep(1.0)
            self.assertFalse(marker.exists())
            self.assertFalse(output.exists())
            self.assertEqual(
                list(audited.glob(".render-audit.*")),
                [],
            )

    @unittest.skipUnless(MODEL_PATH.is_file(), "real G1 model unavailable")
    def test_renderer_early_exit_kills_process_group_and_cleans_stage(self):
        """Catches a renderer exiting while its delayed child survives."""

        from mm_sonic.terrain_oracle.corpus_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            output = audited / "render-audit"
            marker = root / "early-exit-orphan"
            renderer = _write_early_exit_renderer(root / "early-exit.py")
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
            with redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "render-audit",
                        "--corpus",
                        str(audited),
                        "--renderer",
                        str(Path(sys.executable).resolve()),
                        "--renderer-arg",
                        str(renderer),
                        "--renderer-arg",
                        str(marker),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 2)
            time.sleep(1.0)
            self.assertFalse(marker.exists())
            self.assertFalse(output.exists())
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
                    if (
                        path.is_file()
                        and path != freeze_index_path
                        and path != frozen / COMPLETION_MARKER
                    )
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
    def test_freeze_snapshots_before_validation_and_ignores_later_source_swap(
        self,
    ):
        """Catches a post-validation symlink swap entering the frozen release."""

        from mm_sonic.terrain_oracle import corpus_cli

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_audit_fixture(root)
            raw = root / "raw"
            audited = root / "audited"
            rendered = audited / "render-audit"
            frozen = root / "frozen"
            for command in (
                [
                    "import",
                    "--fixture",
                    str(fixture),
                    "--output",
                    str(raw),
                ],
                [
                    "audit",
                    "--corpus",
                    str(raw),
                    "--model",
                    str(MODEL_PATH),
                    "--output",
                    str(audited),
                ],
                [
                    "render-audit",
                    "--corpus",
                    str(audited),
                    "--output",
                    str(rendered),
                ],
                [
                    "coverage",
                    "--corpus",
                    str(audited),
                    "--output",
                    str(audited / "coverage.json"),
                ],
            ):
                self.assertEqual(corpus_cli.main(command), 0)

            render_index = json.loads(
                (rendered / "render-index.json").read_text("ascii")
            )
            receipt_path = (
                rendered
                / render_index["interval_receipts"][0]["relative_path"]
            )
            receipt = json.loads(receipt_path.read_text("ascii"))
            artifact_relative_path = receipt["video"]["relative_path"]
            source_artifact = rendered / artifact_relative_path
            original_payload = source_artifact.read_bytes()
            replacement = root / "replacement.mp4"
            replacement.write_bytes(b"X" * len(original_payload))
            original_freeze_source_paths = corpus_cli._freeze_source_paths
            swapped = False

            def swap_after_validation(
                snapshot_root,
                manifest,
                audit_index,
            ):
                nonlocal swapped
                result = original_freeze_source_paths(
                    snapshot_root,
                    manifest,
                    audit_index,
                )
                if not swapped:
                    source_artifact.unlink()
                    source_artifact.symlink_to(replacement)
                    swapped = True
                return result

            with mock.patch.object(
                corpus_cli,
                "_freeze_source_paths",
                side_effect=swap_after_validation,
            ):
                self.assertEqual(
                    corpus_cli.main(
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
            self.assertTrue(swapped)
            self.assertEqual(
                (
                    frozen
                    / "render-audit"
                    / artifact_relative_path
                ).read_bytes(),
                original_payload,
            )

    def test_freeze_model_uses_authenticated_visual_vfs_at_parse_boundary(self):
        """Catches visual-only include/texture swaps entering the frozen MJB."""

        import mujoco
        from PIL import Image
        from mm_sonic.terrain_oracle.corpus_cli import (
            _snapshot_freeze_model,
        )
        from tests.python.test_oracle_render_media import _model_authority

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            corpus.mkdir()
            model_root = root / "model-vfs"
            model_root.mkdir()
            model_path = model_root / "model.xml"
            include_path = model_root / "body.xml"
            texture_path = model_root / "visual.png"
            model_path.write_text(
                '<mujoco model="freeze_visual">'
                '<asset><texture name="visual_texture" type="2d" '
                'file="visual.png"/>'
                '<material name="visual_material" '
                'texture="visual_texture"/>'
                "</asset><include file=\"body.xml\"/></mujoco>\n",
                encoding="ascii",
            )
            authenticated_include = (
                '<mujoco><worldbody><body name="visual_body">'
                '<geom name="visual_include_geom" type="sphere" size=".05" '
                'rgba=".1 .2 .3 1"/>'
                '<geom name="visual_asset_geom" type="box" '
                'size=".1 .1 .1" material="visual_material"/>'
                "</body></worldbody></mujoco>\n"
            )
            swapped_include = authenticated_include.replace(
                'rgba=".1 .2 .3 1"',
                'rgba=".8 .1 .6 1"',
            )

            def png_bytes(rgb):
                buffer = io.BytesIO()
                Image.new("RGB", (2, 2), rgb).save(
                    buffer,
                    format="PNG",
                )
                return buffer.getvalue()

            authenticated_texture = png_bytes((12, 34, 56))
            swapped_texture = png_bytes((210, 25, 150))
            include_path.write_text(
                authenticated_include,
                encoding="ascii",
            )
            texture_path.write_bytes(authenticated_texture)
            trusted_model = _model_authority(model_path, model_root)
            root_payload = model_path.read_bytes()
            authenticated_model = mujoco.MjSpec.from_string(
                model_path.read_text(encoding="ascii"),
                include={
                    "body.xml": authenticated_include.encode("ascii")
                },
                assets={"visual.png": authenticated_texture},
            ).compile()
            swapped_model = mujoco.MjSpec.from_string(
                model_path.read_text(encoding="ascii"),
                include={"body.xml": swapped_include.encode("ascii")},
                assets={"visual.png": swapped_texture},
            ).compile()
            self.assertEqual(
                structural_model_sha256(authenticated_model),
                structural_model_sha256(swapped_model),
            )
            audit_index = {
                "model": {
                    "asset_path": str(model_path),
                    "asset_size_bytes": len(root_payload),
                    "asset_sha256": hashlib.sha256(
                        root_payload
                    ).hexdigest(),
                    "structural_sha256": structural_model_sha256(
                        authenticated_model
                    ),
                }
            }
            (corpus / "audit-index.json").write_bytes(
                _canonical_json_bytes(audit_index)
            )
            original_from_path = mujoco.MjModel.from_xml_path
            original_from_string = mujoco.MjSpec.from_string
            swapped = False

            def swap_dependencies_once():
                nonlocal swapped
                if not swapped:
                    include_path.write_text(
                        swapped_include,
                        encoding="ascii",
                    )
                    texture_path.write_bytes(swapped_texture)
                    swapped = True

            def swap_then_parse_path(path):
                swap_dependencies_once()
                return original_from_path(path)

            def swap_then_parse_string(xml, include=None, assets=None):
                swap_dependencies_once()
                return original_from_string(
                    xml,
                    include=include,
                    assets=assets,
                )

            with (
                mock.patch(
                    "mm_sonic.terrain_oracle.corpus_cli._trusted_media_model",
                    return_value=trusted_model,
                ),
                mock.patch.object(
                    mujoco.MjModel,
                    "from_xml_path",
                    side_effect=swap_then_parse_path,
                ),
                mock.patch.object(
                    mujoco.MjSpec,
                    "from_string",
                    side_effect=swap_then_parse_string,
                ),
            ):
                snapshot = _snapshot_freeze_model(corpus, root)
            self.assertTrue(swapped)
            np.testing.assert_array_equal(
                snapshot.model.geom("visual_include_geom").rgba,
                authenticated_model.geom("visual_include_geom").rgba,
            )
            np.testing.assert_array_equal(
                snapshot.model.tex_data,
                authenticated_model.tex_data,
            )

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

    def test_grail_inventory_hashes_pair_bytes_pose_and_license_identity(self):
        """Catches same-size PKL/USD substitution outside inventory authority."""

        from mm_sonic.terrain_oracle.corpus_cli import main
        from tests.python.test_oracle_source_grail import (
            _write_grail_fixture,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grail, expected_position, expected_rotation = (
                _write_grail_fixture(root / "grail")
            )
            stem = "terrain_stairs__fixture__0000"
            robot_path = (
                grail
                / "c490_stair_p1_0000"
                / "robot"
                / f"{stem}.pkl"
            )
            terrain_path = (
                grail
                / "c490_stair_p1_0000"
                / "object_usd"
                / f"{stem}.usd"
            )
            original_robot = robot_path.read_bytes()
            original_terrain = terrain_path.read_bytes()

            def publish(name: str) -> dict[str, object]:
                output = root / f"{name}.json"
                self.assertEqual(
                    main(
                        [
                            "inventory",
                            "--grail-root",
                            str(grail),
                            "--grail-families",
                            "c490_stair_p1",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
                return json.loads(output.read_text("ascii"))

            original = publish("original")
            robot_path.write_bytes(
                bytes((original_robot[0] ^ 1,)) + original_robot[1:]
            )
            robot_substitution = publish("robot-substitution")
            robot_path.write_bytes(original_robot)
            terrain_path.write_bytes(
                bytes((original_terrain[0] ^ 1,)) + original_terrain[1:]
            )
            terrain_substitution = publish("terrain-substitution")

            self.assertNotEqual(
                original["content_sha256"],
                robot_substitution["content_sha256"],
            )
            self.assertNotEqual(
                original["content_sha256"],
                terrain_substitution["content_sha256"],
            )
            record = next(
                item
                for item in original["sources"]["grail"]["records"]
                if item["stem"] == stem
            )
            self.assertEqual(
                record["robot"],
                {
                    "relative_path": robot_path.relative_to(grail).as_posix(),
                    "path": str(robot_path.resolve()),
                    "size_bytes": len(original_robot),
                    "sha256": hashlib.sha256(original_robot).hexdigest(),
                    "license_id": "UNRECORDED",
                },
            )
            self.assertEqual(
                record["terrain"],
                {
                    "relative_path": terrain_path.relative_to(grail).as_posix(),
                    "path": str(terrain_path.resolve()),
                    "size_bytes": len(original_terrain),
                    "sha256": hashlib.sha256(original_terrain).hexdigest(),
                    "license_id": "UNRECORDED",
                    "world_from_terrain": {
                        "translation_world": np.asarray(
                            expected_position,
                            dtype=np.float32,
                        ).tolist(),
                        "quaternion_world_from_local_wxyz": np.asarray(
                            expected_rotation,
                            dtype=np.float32,
                        ).tolist(),
                    },
                },
            )

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
