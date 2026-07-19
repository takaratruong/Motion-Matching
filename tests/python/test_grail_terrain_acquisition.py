import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pxr import Usd, UsdGeom

from resources import grail_terrain_acquisition as acquisition


REPOSITORY_ID = "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
REVISION = "943946a972d5de2eb0d2ff214b236d0e43575fd7"
ROBOT_INVENTORY_SHA256 = (
    "595b1276c9191e86bc6101a68929810680ac6790a795c35e692f2f7e8a7f97e2"
)
OBJECTS_INVENTORY_SHA256 = (
    "461a6ff0a533dac1bfa069694d491d3382e9a04cab11f1c3268bbf0a94d9f667"
)
OBJECT_USD_INVENTORY_SHA256 = (
    "6bc4a91cee21aa9a98214d557e27c1b86c556f585fe9673f222dd66eae767846"
)
SLOPE_BASENAME_SHA256 = (
    "73cd3ec78289aa70caad2cd2df05ba0cea0301478c7654409d2c6557306b4b57"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKED_MANIFEST = REPOSITORY_ROOT / "resources" / "grail_terrain_inputs.json"


def _entry(path, payload):
    return {
        "bytes": len(payload),
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _small_manifest(
        entries, *, allowed_globs=None, partitions=None,
        source_coverage=None):
    entries = sorted(entries, key=lambda value: value["path"])
    if allowed_globs is None:
        allowed_globs = ["data/stair_p1/robot/*.pkl"]
    if partitions is None:
        partitions = {
            "stair_p1": {
                "path_prefix": "data/stair_p1/robot/",
                "file_count": len(entries),
                "byte_count": sum(value["bytes"] for value in entries),
            }
        }
    return {
        "schema": acquisition.MANIFEST_SCHEMA,
        "repository": {
            "id": REPOSITORY_ID,
            "revision": REVISION,
            "type": "dataset",
        },
        "modalities": {
            "robot": {
                "allowed_globs": allowed_globs,
                "partitions": partitions,
                "file_count": len(entries),
                "byte_count": sum(value["bytes"] for value in entries),
                "canonical_inventory_sha256":
                    acquisition.canonical_inventory_sha256(entries),
            }
        },
        "source_coverage": {} if source_coverage is None else source_coverage,
    }


def _modality(entries, allowed_globs, partitions):
    entries = sorted(entries, key=lambda value: value["path"])
    return {
        "allowed_globs": allowed_globs,
        "partitions": partitions,
        "file_count": len(entries),
        "byte_count": sum(value["bytes"] for value in entries),
        "canonical_inventory_sha256":
            acquisition.canonical_inventory_sha256(entries),
    }


def _manifest(modalities, source_coverage=None):
    return {
        "schema": acquisition.MANIFEST_SCHEMA,
        "repository": {
            "id": REPOSITORY_ID,
            "revision": REVISION,
            "type": "dataset",
        },
        "modalities": modalities,
        "source_coverage": {} if source_coverage is None else source_coverage,
    }


def _slope_source_coverage(basenames):
    return {
        "slope": {
            "canonical_basename_sha256":
                acquisition.canonical_basenames_sha256(basenames),
            "file_count": len(basenames),
            "modalities": {
                "object_usd": {
                    "path_prefix": "data/slope/object_usd/",
                    "suffix": ".usd",
                },
                "objects": {
                    "path_prefix": "data/slope/objects/",
                    "suffix": ".pkl",
                },
                "robot": {
                    "path_prefix": "data/slope/robot/",
                    "suffix": ".pkl",
                },
            },
        }
    }


def _inventory(entries, modalities=("robot",)):
    return {
        "schema": acquisition.INVENTORY_SCHEMA,
        "repository": {
            "id": REPOSITORY_ID,
            "revision": REVISION,
            "type": "dataset",
        },
        "modalities": list(modalities),
        "files": sorted(entries, key=lambda value: value["path"]),
    }


class GrailTerrainAcquisitionTests(unittest.TestCase):
    @staticmethod
    def _write_usd(path, with_mesh=True):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.Xform.Define(stage, "/model")
        if with_mesh:
            mesh = UsdGeom.Mesh.Define(stage, "/model/mesh")
            mesh.CreatePointsAttr([
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        stage.GetRootLayer().Save()
        del stage

    def _source_coverage_fixture(self, root):
        root = Path(root)
        seed_usd = root / "seed.usd"
        self._write_usd(seed_usd)
        usd_payload = seed_usd.read_bytes()
        seed_usd.unlink()

        payloads = {
            "data/slope/object_usd/a.usd": usd_payload,
            "data/slope/objects/a.pkl": b"slope object",
            "data/stair_p1/object_usd/stair.usd": usd_payload,
            "data/stair_p1/objects/stair.pkl": b"stair object",
            "data/stair_p1/robot/stair.pkl": b"stair robot",
        }
        entries = {
            path: _entry(path, payload)
            for path, payload in payloads.items()
        }
        manifest = _manifest({
            "object_usd": _modality([
                entries["data/slope/object_usd/a.usd"],
                entries["data/stair_p1/object_usd/stair.usd"],
            ], [
                "data/slope/object_usd/*.usd",
                "data/stair_p1/object_usd/*.usd",
            ], {
                "slope": {
                    "path_prefix": "data/slope/object_usd/",
                    "file_count": 1,
                    "byte_count": len(usd_payload),
                },
                "stair_p1": {
                    "path_prefix": "data/stair_p1/object_usd/",
                    "file_count": 1,
                    "byte_count": len(usd_payload),
                },
            }),
            "objects": _modality([
                entries["data/slope/objects/a.pkl"],
                entries["data/stair_p1/objects/stair.pkl"],
            ], [
                "data/slope/objects/*.pkl",
                "data/stair_p1/objects/*.pkl",
            ], {
                "slope": {
                    "path_prefix": "data/slope/objects/",
                    "file_count": 1,
                    "byte_count": len(payloads[
                        "data/slope/objects/a.pkl"]),
                },
                "stair_p1": {
                    "path_prefix": "data/stair_p1/objects/",
                    "file_count": 1,
                    "byte_count": len(payloads[
                        "data/stair_p1/objects/stair.pkl"]),
                },
            }),
            "robot": _modality([
                entries["data/stair_p1/robot/stair.pkl"],
            ], ["data/stair_p1/robot/*.pkl"], {
                "stair_p1": {
                    "path_prefix": "data/stair_p1/robot/",
                    "file_count": 1,
                    "byte_count": len(payloads[
                        "data/stair_p1/robot/stair.pkl"]),
                },
            }),
        }, source_coverage=_slope_source_coverage(["a"]))
        return manifest, entries, payloads

    @staticmethod
    def _fake_download(payloads):
        def download(**kwargs):
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payloads[kwargs["filename"]])
            return str(target)
        return download

    def test_checked_manifest_pins_exact_complete_robot_corpus(self):
        payload = CHECKED_MANIFEST.read_bytes()
        decoded = json.loads(payload)
        self.assertEqual(payload, acquisition.canonical_json_bytes(decoded))
        manifest = acquisition.load_manifest(CHECKED_MANIFEST)

        self.assertEqual(manifest["schema"], acquisition.MANIFEST_SCHEMA)
        self.assertEqual(manifest["repository"], {
            "id": REPOSITORY_ID,
            "revision": REVISION,
            "type": "dataset",
        })
        self.assertEqual(
            tuple(manifest["modalities"]),
            ("object_usd", "objects", "robot"),
        )
        robot = manifest["modalities"]["robot"]
        self.assertEqual(robot["allowed_globs"], [
            "data/stair_p1/robot/*.pkl",
            "data/stair_p2/robot/*.pkl",
        ])
        self.assertEqual(robot["partitions"], {
            "stair_p1": {
                "byte_count": 1_210_648_890,
                "file_count": 6_094,
                "path_prefix": "data/stair_p1/robot/",
            },
            "stair_p2": {
                "byte_count": 1_210_377_370,
                "file_count": 6_094,
                "path_prefix": "data/stair_p2/robot/",
            },
        })
        self.assertEqual(robot["file_count"], 12_188)
        self.assertEqual(robot["byte_count"], 2_421_026_260)
        self.assertEqual(
            robot["canonical_inventory_sha256"], ROBOT_INVENTORY_SHA256)

    def test_checked_manifest_pins_only_matching_object_and_geometry_inputs(self):
        manifest = acquisition.load_manifest(CHECKED_MANIFEST)
        expected = {
            "objects": {
                "allowed_globs": [
                    "data/slope/objects/*.pkl",
                    "data/stair_p1/objects/*.pkl",
                    "data/stair_p2/objects/*.pkl",
                ],
                "byte_count": 252_587_908,
                "canonical_inventory_sha256": OBJECTS_INVENTORY_SHA256,
                "file_count": 14_068,
                "partitions": {
                    "slope": {
                        "byte_count": 36_377_080,
                        "file_count": 1_880,
                        "path_prefix": "data/slope/objects/",
                    },
                    "stair_p1": {
                        "byte_count": 114_180_950,
                        "file_count": 6_094,
                        "path_prefix": "data/stair_p1/objects/",
                    },
                    "stair_p2": {
                        "byte_count": 102_029_878,
                        "file_count": 6_094,
                        "path_prefix": "data/stair_p2/objects/",
                    },
                },
            },
            "object_usd": {
                "allowed_globs": [
                    "data/slope/object_usd/*.usd",
                    "data/stair_p1/object_usd/*.usd",
                    "data/stair_p2/object_usd/*.usd",
                ],
                "byte_count": 5_357_565_995,
                "canonical_inventory_sha256": OBJECT_USD_INVENTORY_SHA256,
                "file_count": 14_068,
                "partitions": {
                    "slope": {
                        "byte_count": 15_515_596,
                        "file_count": 1_880,
                        "path_prefix": "data/slope/object_usd/",
                    },
                    "stair_p1": {
                        "byte_count": 2_962_501_357,
                        "file_count": 6_094,
                        "path_prefix": "data/stair_p1/object_usd/",
                    },
                    "stair_p2": {
                        "byte_count": 2_379_549_042,
                        "file_count": 6_094,
                        "path_prefix": "data/stair_p2/object_usd/",
                    },
                },
            },
        }
        for name, config in expected.items():
            with self.subTest(modality=name):
                self.assertEqual(manifest["modalities"][name], config)
                self.assertFalse(any(
                    "texture" in pattern or pattern.endswith((".jpg", ".png"))
                    for pattern in config["allowed_globs"]
                ))

    def test_checked_manifest_locks_slope_robot_source_coverage_separately(self):
        manifest = acquisition.load_manifest(CHECKED_MANIFEST)
        self.assertEqual(manifest["source_coverage"], {
            "slope": {
                "canonical_basename_sha256": SLOPE_BASENAME_SHA256,
                "file_count": 1_880,
                "modalities": {
                    "object_usd": {
                        "path_prefix": "data/slope/object_usd/",
                        "suffix": ".usd",
                    },
                    "objects": {
                        "path_prefix": "data/slope/objects/",
                        "suffix": ".pkl",
                    },
                    "robot": {
                        "path_prefix": "data/slope/robot/",
                        "suffix": ".pkl",
                    },
                },
            },
        })
        robot = manifest["modalities"]["robot"]
        self.assertEqual(robot["file_count"], 12_188)
        self.assertEqual(robot["byte_count"], 2_421_026_260)
        self.assertEqual(
            robot["canonical_inventory_sha256"], ROBOT_INVENTORY_SHA256)

    def test_inventory_requires_exact_object_geometry_basename_coverage(self):
        robot = _entry("data/stair_p1/robot/a.pkl", b"robot")
        objects = _entry("data/stair_p1/objects/b.pkl", b"objects")
        geometry = _entry("data/stair_p1/object_usd/b.usd", b"usd")
        entries = [robot, objects, geometry]
        modalities = {
            "robot": _modality([robot], ["data/stair_p1/robot/*.pkl"], {
                "stair_p1": {
                    "path_prefix": "data/stair_p1/robot/",
                    "file_count": 1,
                    "byte_count": robot["bytes"],
                },
            }),
            "objects": _modality([objects], ["data/stair_p1/objects/*.pkl"], {
                "stair_p1": {
                    "path_prefix": "data/stair_p1/objects/",
                    "file_count": 1,
                    "byte_count": objects["bytes"],
                },
            }),
            "object_usd": _modality(
                [geometry], ["data/stair_p1/object_usd/*.usd"], {
                    "stair_p1": {
                        "path_prefix": "data/stair_p1/object_usd/",
                        "file_count": 1,
                        "byte_count": geometry["bytes"],
                    },
                }),
        }
        with self.assertRaisesRegex(ValueError, "basename coverage"):
            acquisition.validate_inventory_document(
                _manifest(modalities),
                _inventory(entries, ("object_usd", "objects", "robot")),
                ("object_usd", "objects", "robot"),
            )

    def test_inventory_digest_has_exact_sorted_canonical_encoding(self):
        entries = [
            {"path": "z.pkl", "bytes": 2, "sha256": "b" * 64},
            {"path": "a.pkl", "bytes": 1, "sha256": "a" * 64},
        ]
        expected = (
            '[\n'
            '  {\n'
            '    "bytes": 1,\n'
            '    "path": "a.pkl",\n'
            '    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
            '  },\n'
            '  {\n'
            '    "bytes": 2,\n'
            '    "path": "z.pkl",\n'
            '    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"\n'
            '  }\n'
            ']\n'
        ).encode("utf-8")
        self.assertEqual(
            acquisition.canonical_inventory_bytes(entries), expected)
        self.assertEqual(
            acquisition.canonical_inventory_sha256(entries),
            hashlib.sha256(expected).hexdigest())
        self.assertEqual(
            acquisition.canonical_inventory_sha256(list(reversed(entries))),
            hashlib.sha256(expected).hexdigest())

    def test_basename_digest_has_exact_sorted_canonical_encoding(self):
        expected = '[\n  "a",\n  "z"\n]\n'.encode("utf-8")
        self.assertEqual(
            acquisition.canonical_basenames_bytes(["z", "a"]), expected)
        self.assertEqual(
            acquisition.canonical_basenames_sha256(["z", "a"]),
            hashlib.sha256(expected).hexdigest(),
        )
        for invalid in (["a", "a"], ["../a"], ["a.pkl"], [1]):
            with self.subTest(invalid=invalid), self.assertRaises(
                    (TypeError, ValueError)):
                acquisition.canonical_basenames_sha256(invalid)

    def test_inventory_requires_locked_slope_source_basename_digest(self):
        objects = _entry("data/slope/objects/b.pkl", b"objects")
        geometry = _entry("data/slope/object_usd/b.usd", b"usd")
        modalities = {
            "objects": _modality([objects], ["data/slope/objects/*.pkl"], {
                "slope": {
                    "path_prefix": "data/slope/objects/",
                    "file_count": 1,
                    "byte_count": objects["bytes"],
                },
            }),
            "object_usd": _modality(
                [geometry], ["data/slope/object_usd/*.usd"], {
                    "slope": {
                        "path_prefix": "data/slope/object_usd/",
                        "file_count": 1,
                        "byte_count": geometry["bytes"],
                    },
                }),
        }
        manifest = _manifest(
            modalities, source_coverage=_slope_source_coverage(["a"]))
        with self.assertRaisesRegex(ValueError, "source coverage"):
            acquisition.validate_inventory_document(
                manifest,
                _inventory(
                    [objects, geometry], ("object_usd", "objects")),
                ("object_usd", "objects"),
            )

    def test_inventory_rejects_duplicates_traversal_and_unexpected_modality(self):
        good = _entry("data/stair_p1/robot/good.pkl", b"good")
        manifest = _small_manifest([good])
        cases = (
            ("duplicate", [good, dict(good)]),
            ("relative", [dict(good, path="../escape.pkl")]),
            ("relative", [dict(good, path="/absolute.pkl")]),
            ("relative", [dict(good, path="data\\stair_p1\\robot\\x.pkl")]),
            ("unexpected", [dict(
                good, path="data/stair_p1/video/good.mp4")]),
        )
        for message, entries in cases:
            with self.subTest(entries=entries), self.assertRaisesRegex(
                    ValueError, message):
                acquisition.validate_inventory_document(
                    manifest, _inventory(entries), ("robot",))
        with self.assertRaisesRegex(ValueError, "unknown modalit"):
            acquisition.validate_inventory_document(
                manifest, _inventory([good]), ("objects",))

    def test_inventory_requires_lexical_order_and_exact_file_identity_types(self):
        first = _entry("data/stair_p1/robot/a.pkl", b"a")
        second = _entry("data/stair_p1/robot/b.pkl", b"bb")
        manifest = _small_manifest([first, second])
        unsorted_inventory = _inventory([first, second])
        unsorted_inventory["files"] = [second, first]
        with self.assertRaisesRegex(ValueError, "lexical"):
            acquisition.validate_inventory_document(
                manifest, unsorted_inventory, ("robot",))

        bad_entries = (
            dict(first, bytes=True),
            dict(first, bytes=-1),
            dict(first, sha256="A" * 64),
            dict(first, extra="field"),
        )
        for entry in bad_entries:
            with self.subTest(entry=entry), self.assertRaises((TypeError, ValueError)):
                acquisition.canonical_inventory_bytes([entry])

    def test_streaming_sha256_uses_only_bounded_reads(self):
        payload = b"x" * (acquisition.HASH_CHUNK_BYTES * 2 + 17)

        class TrackingReader(io.BytesIO):
            def __init__(self, value):
                super().__init__(value)
                self.read_sizes = []

            def read(self, size=-1):
                self.read_sizes.append(size)
                if size <= 0 or size > acquisition.HASH_CHUNK_BYTES:
                    raise AssertionError(f"unbounded read size: {size}")
                return super().read(size)

        stream = TrackingReader(payload)
        self.assertEqual(
            acquisition.sha256_stream(stream), hashlib.sha256(payload).hexdigest())
        self.assertEqual(
            stream.read_sizes,
            [acquisition.HASH_CHUNK_BYTES] * 4)

    def test_offline_verifier_rejects_missing_size_hash_and_unexpected_files(self):
        expected = _entry("data/stair_p1/robot/good.pkl", b"good")
        manifest = _small_manifest([expected])
        inventory = _inventory([expected])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / expected["path"]
            path.parent.mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                acquisition.verify_local(manifest, inventory, root, ("robot",))

            path.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "byte size"):
                acquisition.verify_local(manifest, inventory, root, ("robot",))

            path.write_bytes(b"baad")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                acquisition.verify_local(manifest, inventory, root, ("robot",))

            path.write_bytes(b"good")
            unexpected = path.with_name("unexpected.pkl")
            unexpected.write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected local"):
                acquisition.verify_local(manifest, inventory, root, ("robot",))

    def test_robot_only_download_and_verify_skip_unselected_source_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, entries, payloads = self._source_coverage_fixture(root)
            inventory = _inventory([
                entries["data/stair_p1/robot/stair.pkl"],
            ])
            inventory_path = root / "g1_mm_inventory.json"

            downloaded = acquisition.download_inventory(
                manifest, inventory, root, ("robot",), inventory_path,
                hf_download=self._fake_download(payloads))
            verified = acquisition.verify_local(
                manifest, inventory, root, ("robot",))

            self.assertEqual(downloaded["file_count"], 1)
            self.assertEqual(verified, downloaded)
            self.assertFalse((root / "data/slope").exists())

    def test_incremental_terrain_download_and_verify_skip_slope_robot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, entries, payloads = self._source_coverage_fixture(root)
            robot_inventory = _inventory([
                entries["data/stair_p1/robot/stair.pkl"],
            ])
            terrain_inventory = _inventory([
                entries["data/slope/object_usd/a.usd"],
                entries["data/slope/objects/a.pkl"],
                entries["data/stair_p1/object_usd/stair.usd"],
                entries["data/stair_p1/objects/stair.pkl"],
            ], ("object_usd", "objects"))
            inventory = acquisition.merge_inventory_documents(
                manifest, robot_inventory, terrain_inventory)
            inventory_path = root / "g1_mm_inventory.json"

            downloaded = acquisition.download_inventory(
                manifest, inventory, root, ("object_usd", "objects"),
                inventory_path, hf_download=self._fake_download(payloads))
            verified = acquisition.verify_local(
                manifest, inventory, root, ("object_usd", "objects"))

            self.assertEqual(downloaded["file_count"], 4)
            self.assertEqual(verified, downloaded)
            self.assertFalse((root / "data/slope/robot").exists())
            self.assertFalse((root / "data/stair_p1/robot").exists())

    def test_full_selection_enforces_external_slope_source_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, entries, payloads = self._source_coverage_fixture(root)
            inventory = _inventory(
                list(entries.values()),
                ("object_usd", "objects", "robot"))
            for path, payload in payloads.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            robot_path = root / "data/slope/robot/a.pkl"
            robot_path.parent.mkdir(parents=True)
            robot_path.write_bytes(b"external slope robot")

            summary = acquisition.verify_local(
                manifest, inventory, root,
                ("object_usd", "objects", "robot"))
            self.assertEqual(summary["file_count"], 5)

            robot_path.unlink()
            with self.assertRaisesRegex(ValueError, "source coverage"):
                acquisition.verify_local(
                    manifest, inventory, root,
                    ("object_usd", "objects", "robot"))
            robot_path.write_bytes(b"external slope robot")

            extra = robot_path.with_name("b.pkl")
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "source coverage"):
                acquisition.verify_local(
                    manifest, inventory, root,
                    ("object_usd", "objects", "robot"))
            extra.unlink()

            renamed = robot_path.with_name("renamed.pkl")
            robot_path.rename(renamed)
            with self.assertRaisesRegex(ValueError, "source coverage"):
                acquisition.verify_local(
                    manifest, inventory, root,
                    ("object_usd", "objects", "robot"))

    def test_offline_verifier_rejects_symlinks_directories_and_nonregular_files(self):
        expected = _entry("data/stair_p1/robot/good.pkl", b"good")
        manifest = _small_manifest([expected])
        inventory = _inventory([expected])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / expected["path"]
            path.parent.mkdir(parents=True)
            outside = root / "outside"
            outside.write_bytes(b"good")

            os.symlink(outside, path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                acquisition.verify_local(manifest, inventory, root, ("robot",))
            path.unlink()

            path.mkdir()
            with self.assertRaisesRegex(ValueError, "regular file"):
                acquisition.verify_local(manifest, inventory, root, ("robot",))
            path.rmdir()

            os.mkfifo(path)
            with self.assertRaisesRegex(ValueError, "regular file"):
                acquisition.verify_local(manifest, inventory, root, ("robot",))
            path.unlink()

            parent = path.parent
            parent.rmdir()
            os.symlink(root, parent)
            with self.assertRaisesRegex(ValueError, "symlink"):
                acquisition.verify_local(manifest, inventory, root, ("robot",))

    def test_remote_inventory_uses_only_pinned_allowed_prefixes(self):
        one_payload = b"one"
        two_payload = b"two!"
        entries = [
            _entry("data/stair_p1/robot/a.pkl", one_payload),
            _entry("data/stair_p2/robot/b.pkl", two_payload),
        ]
        manifest = _small_manifest(
            entries,
            allowed_globs=[
                "data/stair_p1/robot/*.pkl",
                "data/stair_p2/robot/*.pkl",
            ],
            partitions={
                "stair_p1": {
                    "path_prefix": "data/stair_p1/robot/",
                    "file_count": 1,
                    "byte_count": len(one_payload),
                },
                "stair_p2": {
                    "path_prefix": "data/stair_p2/robot/",
                    "file_count": 1,
                    "byte_count": len(two_payload),
                },
            })

        def remote(entry):
            return SimpleNamespace(
                path=entry["path"], size=entry["bytes"],
                lfs=SimpleNamespace(
                    size=entry["bytes"], sha256=entry["sha256"]))

        class FakeApi:
            def __init__(self):
                self.calls = []

            def list_repo_tree(self, repository_id, path_in_repo, **kwargs):
                self.calls.append((repository_id, path_in_repo, kwargs))
                selected = [
                    entry for entry in reversed(entries)
                    if entry["path"].startswith(path_in_repo + "/")
                ]
                return [remote(entry) for entry in selected]

        api = FakeApi()
        observed = acquisition.build_remote_inventory(
            manifest, ("robot",), api=api)
        self.assertEqual(observed, _inventory(entries))
        self.assertEqual(api.calls, [
            (REPOSITORY_ID, "data/stair_p1/robot", {
                "recursive": True,
                "expand": False,
                "revision": REVISION,
                "repo_type": "dataset",
            }),
            (REPOSITORY_ID, "data/stair_p2/robot", {
                "recursive": True,
                "expand": False,
                "revision": REVISION,
                "repo_type": "dataset",
            }),
        ])

    def test_remote_inventory_rejects_changed_byte_size(self):
        expected = _entry("data/stair_p1/robot/a.pkl", b"good")
        manifest = _small_manifest([expected])
        changed = SimpleNamespace(
            path=expected["path"], size=5,
            lfs=SimpleNamespace(size=5, sha256=expected["sha256"]))
        api = SimpleNamespace(
            list_repo_tree=lambda *args, **kwargs: [changed])
        with self.assertRaisesRegex(ValueError, "byte count"):
            acquisition.build_remote_inventory(manifest, ("robot",), api=api)

    def test_remote_inventory_stream_hashes_regular_git_geometry(self):
        payload = b"ordinary git USD blob"
        expected = _entry("data/stair_p1/object_usd/a.usd", payload)
        manifest = _manifest({
            "object_usd": _modality(
                [expected], ["data/stair_p1/object_usd/*.usd"], {
                    "stair_p1": {
                        "path_prefix": "data/stair_p1/object_usd/",
                        "file_count": 1,
                        "byte_count": len(payload),
                    },
                }),
        })
        remote = SimpleNamespace(
            path=expected["path"], size=len(payload), lfs=None)
        api = SimpleNamespace(list_repo_tree=lambda *args, **kwargs: [remote])
        with tempfile.TemporaryDirectory() as temporary:
            downloaded = Path(temporary) / "asset.usd"
            downloaded.write_bytes(payload)
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                return str(downloaded)

            observed = acquisition.build_remote_inventory(
                manifest, ("object_usd",), api=api,
                hf_download=fake_download)

        self.assertEqual(observed, _inventory([expected], ("object_usd",)))
        self.assertEqual(calls, [{
            "repo_id": REPOSITORY_ID,
            "filename": expected["path"],
            "repo_type": "dataset",
            "revision": REVISION,
        }])

    def test_remote_inventory_ignores_nested_texture_tree_entries(self):
        payload = b"ordinary git USD blob"
        expected = _entry("data/stair_p1/object_usd/a.usd", payload)
        manifest = _manifest({
            "object_usd": _modality(
                [expected], ["data/stair_p1/object_usd/*.usd"], {
                    "stair_p1": {
                        "path_prefix": "data/stair_p1/object_usd/",
                        "file_count": 1,
                        "byte_count": len(payload),
                    },
                }),
        })
        remotes = [
            SimpleNamespace(
                path=expected["path"], size=len(payload), lfs=None),
            SimpleNamespace(
                path="data/stair_p1/object_usd/textures/a/model.jpg",
                size=123,
                lfs=None,
            ),
        ]
        api = SimpleNamespace(list_repo_tree=lambda *args, **kwargs: remotes)
        with tempfile.TemporaryDirectory() as temporary:
            downloaded = Path(temporary) / "asset.usd"
            downloaded.write_bytes(payload)
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs["filename"])
                return str(downloaded)

            observed = acquisition.build_remote_inventory(
                manifest, ("object_usd",), api=api,
                hf_download=fake_download)

        self.assertEqual(observed, _inventory([expected], ("object_usd",)))
        self.assertEqual(calls, [expected["path"]])

    def test_remote_inventory_rejects_unlisted_nested_geometry(self):
        payload = b"ordinary git USD blob"
        expected = _entry("data/stair_p1/object_usd/a.usd", payload)
        manifest = _manifest({
            "object_usd": _modality(
                [expected], ["data/stair_p1/object_usd/*.usd"], {
                    "stair_p1": {
                        "path_prefix": "data/stair_p1/object_usd/",
                        "file_count": 1,
                        "byte_count": len(payload),
                    },
                }),
        })
        remotes = [
            SimpleNamespace(
                path=expected["path"], size=len(payload), lfs=None),
            SimpleNamespace(
                path="data/stair_p1/object_usd/nested/unlisted.usd",
                size=10,
                lfs=None,
            ),
        ]
        api = SimpleNamespace(list_repo_tree=lambda *args, **kwargs: remotes)
        with tempfile.TemporaryDirectory() as temporary:
            downloaded = Path(temporary) / "asset.usd"
            downloaded.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "unexpected remote"):
                acquisition.build_remote_inventory(
                    manifest, ("object_usd",), api=api,
                    hf_download=lambda **kwargs: str(downloaded))

    def test_offline_verifier_ignores_textures_but_rejects_unlisted_usd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usd_path = root / "data/stair_p1/object_usd/a.usd"
            usd_path.parent.mkdir(parents=True)
            self._write_usd(usd_path)
            expected = _entry(
                "data/stair_p1/object_usd/a.usd", usd_path.read_bytes())
            manifest = _manifest({
                "object_usd": _modality(
                    [expected], ["data/stair_p1/object_usd/*.usd"], {
                        "stair_p1": {
                            "path_prefix": "data/stair_p1/object_usd/",
                            "file_count": 1,
                            "byte_count": expected["bytes"],
                        },
                    }),
            })
            inventory = _inventory([expected], ("object_usd",))
            texture = usd_path.parent / "textures/a/model.jpg"
            texture.parent.mkdir(parents=True)
            texture.write_bytes(b"excluded render texture")

            summary = acquisition.verify_local(
                manifest, inventory, root, ("object_usd",))
            self.assertEqual(summary["file_count"], 1)

            self._write_usd(usd_path.with_name("unlisted.usd"))
            with self.assertRaisesRegex(ValueError, "unexpected local"):
                acquisition.verify_local(
                    manifest, inventory, root, ("object_usd",))
            usd_path.with_name("unlisted.usd").unlink()

            nested = usd_path.parent / "nested/unlisted.usd"
            nested.parent.mkdir()
            self._write_usd(nested)
            with self.assertRaisesRegex(ValueError, "unexpected local"):
                acquisition.verify_local(
                    manifest, inventory, root, ("object_usd",))

    def test_offline_verifier_requires_openable_mesh_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usd_path = root / "data/stair_p1/object_usd/a.usd"
            usd_path.parent.mkdir(parents=True)
            self._write_usd(usd_path, with_mesh=False)
            expected = _entry(
                "data/stair_p1/object_usd/a.usd", usd_path.read_bytes())
            manifest = _manifest({
                "object_usd": _modality(
                    [expected], ["data/stair_p1/object_usd/*.usd"], {
                        "stair_p1": {
                            "path_prefix": "data/stair_p1/object_usd/",
                            "file_count": 1,
                            "byte_count": expected["bytes"],
                        },
                    }),
            })
            inventory = _inventory([expected], ("object_usd",))

            with self.assertRaisesRegex(ValueError, "openable mesh geometry"):
                acquisition.verify_local(
                    manifest, inventory, root, ("object_usd",))

    def test_download_is_resumable_and_publishes_inventory_atomically(self):
        present_payload = b"present"
        missing_payload = b"missing"
        present = _entry("data/stair_p1/robot/present.pkl", present_payload)
        missing = _entry("data/stair_p1/robot/missing.pkl", missing_payload)
        entries = sorted([present, missing], key=lambda value: value["path"])
        manifest = _small_manifest(entries)
        remote_inventory = _inventory(entries)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            present_path = root / present["path"]
            present_path.parent.mkdir(parents=True)
            present_path.write_bytes(present_payload)
            inventory_path = root / "g1_mm_inventory.json"
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                target = Path(kwargs["local_dir"]) / kwargs["filename"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(missing_payload)
                return str(target)

            acquisition.download_inventory(
                manifest, remote_inventory, root, ("robot",),
                inventory_path, hf_download=fake_download)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], {
                "repo_id": REPOSITORY_ID,
                "filename": missing["path"],
                "repo_type": "dataset",
                "revision": REVISION,
                "local_dir": str(root),
                "force_download": False,
            })
            self.assertEqual(
                inventory_path.read_bytes(),
                acquisition.canonical_json_bytes(remote_inventory))
            self.assertEqual(
                list(root.glob(".g1_mm_inventory.json.*.tmp")), [])

    def test_atomic_publish_preserves_old_inventory_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "inventory.json"
            target.write_bytes(b"old\n")
            with (
                mock.patch.object(
                    acquisition.os, "replace", side_effect=OSError("replace")),
                self.assertRaisesRegex(OSError, "replace"),
            ):
                acquisition.atomic_write_json(target, {"new": True})
            self.assertEqual(target.read_bytes(), b"old\n")
            self.assertEqual(list(target.parent.glob(".inventory.json.*.tmp")), [])

    def test_verify_command_is_offline_and_never_executes_pickle_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "executed"
            payload = (
                b"cos\nsystem\n(S'touch " + str(marker).encode("utf-8") +
                b"'\ntR."
            )
            entry = _entry("data/stair_p1/robot/hostile.pkl", payload)
            manifest = _small_manifest([entry])
            inventory = _inventory([entry])
            manifest_path = root / "manifest.json"
            inventory_path = root / "inventory.json"
            path = root / entry["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            acquisition.atomic_write_json(manifest_path, manifest)
            acquisition.atomic_write_json(inventory_path, inventory)

            with (
                mock.patch.object(
                    acquisition, "build_remote_inventory",
                    side_effect=AssertionError("network path used")),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                result = acquisition.main([
                    "verify", "--manifest", str(manifest_path),
                    "--dataset-root", str(root), "--modalities", "robot",
                    "--inventory", str(inventory_path),
                ])
            self.assertEqual(result, 0)
            self.assertIn("verified 1 files", stdout.getvalue())
            self.assertFalse(marker.exists())

    def test_acquisition_source_contains_no_pickle_or_joblib_execution(self):
        source = Path(acquisition.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "joblib.load", "pickle.load", "pickle.loads", "runpy.run_path",
            "exec(", "eval(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
