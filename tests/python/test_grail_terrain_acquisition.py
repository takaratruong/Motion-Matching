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

from resources import grail_terrain_acquisition as acquisition


REPOSITORY_ID = "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
REVISION = "943946a972d5de2eb0d2ff214b236d0e43575fd7"
ROBOT_INVENTORY_SHA256 = (
    "595b1276c9191e86bc6101a68929810680ac6790a795c35e692f2f7e8a7f97e2"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKED_MANIFEST = REPOSITORY_ROOT / "resources" / "grail_terrain_inputs.json"


def _entry(path, payload):
    return {
        "bytes": len(payload),
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _small_manifest(entries, *, allowed_globs=None, partitions=None):
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
        self.assertEqual(tuple(manifest["modalities"]), ("robot",))
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
