from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mm_sonic.build_terrain_pfnn_dataset import (
    _height_at,
    _deterministic_npz_bytes,
    _select_families,
    canonical_json_sha256,
    source_root_records,
    write_pfnn_dataset,
)
from mm_sonic.terrain_oracle.canonical import ISAACLAB_JOINT_NAMES
from mm_sonic.terrain_oracle.canonical import CanonicalTerrainMesh
from mm_sonic.terrain_oracle.contact import CanonicalMeshQuery
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_pfnn.dataset import PFNNShardDataset, normalize_pfnn_input
from mm_sonic.terrain_pfnn.features import PFNNTrainingWindow
from mm_sonic.terrain_pfnn.layout import (
    CONTACT_ORDER,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
    TRAJECTORY_TIMES_S,
)
from mm_sonic.terrain_pfnn.provenance import DATASET_SCHEMA, source_set_payload
from mm_sonic.terrain_pfnn.splits import split_identity, terrain_identity
from mm_sonic.terrain_pfnn.sources import GrailSlopeRecord, _GRAIL_SOURCE_LICENSE_ID


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _window(
    clip_id: str, value: float, frame: int, *, sequence_lane: str
) -> PFNNTrainingWindow:
    x = np.arange(INPUT_LAYOUT.size, dtype=np.float32) + value
    y = np.arange(OUTPUT_LAYOUT.size, dtype=np.float32) - value
    y[OUTPUT_LAYOUT["contact_logit"]] = np.array((0, 1, 1, 0), np.float32)
    return PFNNTrainingWindow(
        x=x,
        y=y,
        phase=0.25 + frame * 0.01,
        clip_id=clip_id,
        split_identity=terrain_identity(clip_id),
        split=split_identity(clip_id),
        sequence_lane=sequence_lane,
        terrain_class="flat" if clip_id.startswith("walk") else "ascent",
        center_frame=frame,
        motion_sha256=_DIGEST_A,
        terrain_sha256=None if clip_id.startswith("walk") else _DIGEST_B,
    )


def _source_roots() -> dict[str, dict[str, str]]:
    return {
        "synthetic": {
            "path": "/synthetic",
            "license_id": "test-only",
            "license_manifest_path": "/synthetic/LICENSE",
            "license_manifest_sha256": _DIGEST_A,
        }
    }


def _source_records(windows: list[PFNNTrainingWindow]) -> dict[str, dict[str, str]]:
    return {
        window.clip_id: {
            "source_kind": "lafan" if window.clip_id.startswith("walk") else "grail",
            "motion_sha256": window.motion_sha256,
            "terrain_sha256": window.terrain_sha256 or "",
            "license_id": "test-only",
            "split_identity": window.split_identity,
            "split": window.split,
        }
        for window in windows
    }


class TerrainPFNNDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.windows = [
            _window("walk1_subject1", 1.0, 30, sequence_lane="motion"),
            _window("walk1_subject1", 3.0, 31, sequence_lane="motion"),
            _window("walk2_subject3", 5.0, 32, sequence_lane="motion"),
            _window("walk2_subject1", 7.0, 33, sequence_lane="motion"),
        ]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, output: Path, windows: list[PFNNTrainingWindow]) -> dict[str, object]:
        return write_pfnn_dataset(
            output,
            windows,
            source_roots=_source_roots(),
            source_records=_source_records(windows),
            rejection_counts={"invalid_phase": 2},
            max_windows_per_shard=2,
        )

    def test_round_trip_preserves_shapes_provenance_and_exact_manifest(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)

        self.assertEqual(manifest["schema"], "mm-sonic-terrain-pfnn-dataset/v2")
        self.assertEqual(DATASET_SCHEMA, "mm-sonic-terrain-pfnn-dataset/v2")
        self.assertEqual(manifest["status"], "accepted")
        self.assertEqual(manifest["fps"], 30.0)
        self.assertEqual(manifest["input_size"], 288)
        self.assertEqual(manifest["output_size"], 268)
        self.assertEqual(manifest["joint_order"], list(ISAACLAB_JOINT_NAMES))
        self.assertEqual(manifest["trajectory_times_s"], TRAJECTORY_TIMES_S.tolist())
        self.assertEqual(manifest["contact_order"], list(CONTACT_ORDER))
        self.assertEqual(manifest["rejections"], {"invalid_phase": 2})
        self.assertEqual(
            manifest["source_set_digest_sha256"],
            canonical_json_sha256(
                source_set_payload(
                    _source_roots(),
                    _source_records(self.windows),
                    manifest["split_identities"],
                    manifest["build_options"],
                )
            ),
        )
        self.assertEqual(len(manifest["shards"]), 3)
        identities = manifest["split_identities"]
        self.assertEqual(identities["train"], ["walk1_subject1"])
        self.assertEqual(identities["validation"], ["walk2_subject3"])
        self.assertEqual(identities["test"], ["walk2_subject1"])
        self.assertEqual(
            len(set(identities["train"]) & set(identities["validation"])), 0
        )
        self.assertEqual(
            len(set(identities["train"]) & set(identities["test"])), 0
        )

        for record in manifest["shards"]:
            shard = output / record["path"]
            self.assertEqual(hashlib.sha256(shard.read_bytes()).hexdigest(), record["sha256"])
            self.assertLessEqual(record["count"], 2)
            with np.load(shard, allow_pickle=False) as arrays:
                count = record["count"]
                self.assertEqual(arrays["x"].shape, (count, 288))
                self.assertEqual(arrays["y"].shape, (count, 268))
                self.assertEqual(arrays["phase"].shape, (count,))
                self.assertEqual(arrays["clip_id"].shape, (count,))
                self.assertEqual(arrays["split_identity"].shape, (count,))
                self.assertEqual(arrays["sequence_lane"].shape, (count,))
                self.assertEqual(arrays["terrain_class"].shape, (count,))
                self.assertEqual(arrays["center_frame"].shape, (count,))
                self.assertEqual(arrays["motion_sha256"].shape, (count,))
                self.assertEqual(arrays["terrain_sha256"].shape, (count,))
                self.assertEqual(arrays["x"].dtype, np.dtype(np.float32))
                self.assertEqual(arrays["y"].dtype, np.dtype(np.float32))
                self.assertEqual(arrays["phase"].dtype, np.dtype(np.float32))
                self.assertEqual(arrays["center_frame"].dtype, np.dtype(np.int32))
                self.assertEqual(arrays["clip_id"].dtype, np.dtype("<U128"))
                self.assertEqual(arrays["split_identity"].dtype, np.dtype("<U128"))
                self.assertEqual(arrays["sequence_lane"].dtype, np.dtype("<U16"))
                self.assertEqual(arrays["terrain_class"].dtype, np.dtype("<U10"))
                self.assertEqual(arrays["motion_sha256"].dtype, np.dtype("<U64"))
                self.assertEqual(arrays["terrain_sha256"].dtype, np.dtype("<U64"))
                self.assertTrue(np.isfinite(arrays["x"]).all())
                self.assertTrue(np.isfinite(arrays["y"]).all())
                self.assertEqual(set(map(str, arrays["sequence_lane"])), {"motion"})

        persisted = json.loads((output / "manifest.json").read_text())
        self.assertEqual(persisted, manifest)
        self.assertFalse(list(output.rglob("*.tmp")))

    def test_canonical_digest_is_stable_across_mapping_order(self) -> None:
        left = {"z": [3, 2, 1], "a": {"y": 2, "x": "é"}}
        right = {"a": {"x": "é", "y": 2}, "z": [3, 2, 1]}
        self.assertEqual(canonical_json_sha256(left), canonical_json_sha256(right))

    def test_dataset_digest_and_duplicate_identity_bind_sequence_lane(self) -> None:
        motion = replace(self.windows[0], phase=0.0, sequence_lane="motion")
        idle = replace(self.windows[0], phase=0.0, sequence_lane="idle_phase_0")
        motion_output = self.root / "motion"
        idle_output = self.root / "idle"
        motion_manifest = self._write(motion_output, [motion])
        idle_manifest = self._write(idle_output, [idle])
        self.assertEqual(
            (motion_output / "normalization.npz").read_bytes(),
            (idle_output / "normalization.npz").read_bytes(),
        )
        self.assertEqual(
            motion_manifest["source_set_digest_sha256"],
            idle_manifest["source_set_digest_sha256"],
        )
        self.assertNotEqual(
            motion_manifest["dataset_digest_sha256"],
            idle_manifest["dataset_digest_sha256"],
        )

        both_manifest = self._write(self.root / "both", [motion, idle])
        self.assertEqual(sum(row["count"] for row in both_manifest["shards"]), 2)

    def test_duplicate_identity_is_clip_lane_frame_for_input_load_and_resume(self) -> None:
        motion = replace(self.windows[0], phase=0.0, sequence_lane="motion")
        changed_phase = replace(motion, phase=0.5)
        with self.assertRaisesRegex(ValueError, "duplicate windows"):
            self._write(self.root / "input-duplicate", [motion, changed_phase])

        idle = replace(motion, phase=0.5, sequence_lane="idle_phase_0")
        output = self.root / "persisted-duplicate"
        manifest = self._write(output, [motion, idle])
        record = manifest["shards"][0]
        shard = output / record["path"]
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        arrays["sequence_lane"][1] = "motion"
        shard.write_bytes(_deterministic_npz_bytes(arrays))
        record["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))

        with self.assertRaisesRegex(ValueError, "duplicate window"):
            PFNNShardDataset(output, "train")
        with self.assertRaisesRegex(ValueError, "duplicate window"):
            write_pfnn_dataset(
                output,
                [motion, idle],
                source_roots=_source_roots(),
                source_records=_source_records([motion, idle]),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_lane_tampering_fails_loader_and_resume(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        record = manifest["shards"][0]
        shard = output / record["path"]
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        arrays["sequence_lane"][0] = "idle_phase_7"
        shard.write_bytes(_deterministic_npz_bytes(arrays))

        with self.assertRaisesRegex(ValueError, "shard hash"):
            PFNNShardDataset(output, "train")
        with self.assertRaisesRegex(ValueError, "shard hash"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_missing_lane_field_fails_loader_and_resume(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        record = manifest["shards"][0]
        shard = output / record["path"]
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        del arrays["sequence_lane"]
        shard.write_bytes(_deterministic_npz_bytes(arrays))
        record["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))

        with self.assertRaisesRegex(ValueError, "shard fields"):
            PFNNShardDataset(output, "train")
        with self.assertRaisesRegex(ValueError, "invalid shard"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_v1_schema_fails_before_shards_are_opened(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        manifest["schema"] = "mm-sonic-terrain-pfnn-dataset/v1"
        (output / "manifest.json").write_text(json.dumps(manifest))

        with patch(
            "mm_sonic.terrain_pfnn.dataset.np.load",
            side_effect=AssertionError("old schema opened a shard"),
        ) as load:
            with self.assertRaisesRegex(ValueError, "schema"):
                PFNNShardDataset(output, "train")
            load.assert_not_called()
        with patch(
            "mm_sonic.build_terrain_pfnn_dataset._validate_resume_files",
            side_effect=AssertionError("old schema opened a shard"),
        ) as validate_files:
            with self.assertRaisesRegex(ValueError, "schema"):
                write_pfnn_dataset(
                    output,
                    self.windows,
                    source_roots=_source_roots(),
                    source_records=_source_records(self.windows),
                    rejection_counts={"invalid_phase": 2},
                    max_windows_per_shard=2,
                    resume=True,
                )
            validate_files.assert_not_called()

    def test_source_set_digest_is_stable_across_absolute_root_relocation(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        first_manifest = self._write(first, self.windows)
        relocated = _source_roots()
        relocated["synthetic"]["path"] = "/relocated/source"
        relocated["synthetic"]["license_manifest_path"] = "/relocated/LICENSE"
        second_manifest = write_pfnn_dataset(
            second,
            self.windows,
            source_roots=relocated,
            source_records=_source_records(self.windows),
            rejection_counts={"invalid_phase": 2},
            max_windows_per_shard=2,
        )
        self.assertEqual(
            first_manifest["source_set_digest_sha256"],
            second_manifest["source_set_digest_sha256"],
        )

    def test_source_root_schema_rejects_non_string_values(self) -> None:
        roots = _source_roots()
        roots["synthetic"]["license_id"] = 7  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "string"):
            write_pfnn_dataset(
                self.root / "dataset",
                self.windows,
                source_roots=roots,
                source_records=_source_records(self.windows),
                rejection_counts={},
            )

    def test_source_roots_validate_license_manifests_and_lafan_inventory(self) -> None:
        grail = self.root / "GRAIL"
        lafan = self.root / "lafan" / "g1"
        grail.mkdir()
        lafan.mkdir(parents=True)
        (grail / "README.md").write_text("---\nlicense: apache-2.0\n---\n")
        motion = lafan / "walk1_subject1.csv"
        motion.write_text("0,0,0\n")
        source_manifest = {
            "license": "CC BY-NC-ND 4.0",
            "source_fps": 30.0,
            "files": [
                {
                    "path": "g1/walk1_subject1.csv",
                    "sha256": hashlib.sha256(motion.read_bytes()).hexdigest(),
                }
            ],
        }
        (lafan.parent / "source_manifest.json").write_text(json.dumps(source_manifest))

        roots = source_root_records(grail, lafan)
        self.assertEqual(_GRAIL_SOURCE_LICENSE_ID, "Apache-2.0")
        self.assertEqual(roots["grail"]["license_id"], "Apache-2.0")
        self.assertEqual(roots["lafan"]["license_id"], "CC-BY-NC-ND-4.0")
        self.assertEqual(roots["grail"]["path"], str(grail.resolve()))
        self.assertEqual(
            roots["lafan"]["license_manifest_path"],
            str((lafan.parent / "source_manifest.json").resolve()),
        )

        source_manifest["source_fps"] = 60.0
        (lafan.parent / "source_manifest.json").write_text(json.dumps(source_manifest))
        with self.assertRaisesRegex(ValueError, "source_fps"):
            source_root_records(grail, lafan)

    def test_finite_ramp_height_extends_flat_runup_and_plateau(self) -> None:
        mesh = CanonicalTerrainMesh(
            vertices_local=np.asarray(
                ((-1, 0, 0), (1, 0, 0), (1, 2, 1), (-1, 2, 1)),
                dtype=np.float32,
            ),
            faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
            valid_faces=np.ones(2, dtype=bool),
            source_asset_sha256=_DIGEST_A,
        )
        query = CanonicalMeshQuery(
            mesh,
            RigidTransform(
                np.zeros(3, dtype=np.float32),
                np.asarray((1, 0, 0, 0), dtype=np.float32),
            ),
        )
        np.testing.assert_allclose(
            _height_at(query, np.asarray(((0, -2), (0, 1), (0, 4)))),
            (0.0, 0.5, 1.0),
            atol=1.0e-6,
        )

    def test_piecewise_hill_uses_nearest_upward_surface(self) -> None:
        mesh = CanonicalTerrainMesh(
            vertices_local=np.asarray(
                ((-1, 0, 0), (1, 0, 0), (-1, 1, 1), (1, 1, 1),
                 (-1, 2, 0), (1, 2, 0)),
                dtype=np.float32,
            ),
            faces=np.asarray(
                ((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)),
                dtype=np.int32,
            ),
            valid_faces=np.ones(4, dtype=bool),
            source_asset_sha256=_DIGEST_A,
        )
        query = CanonicalMeshQuery(
            mesh,
            RigidTransform(
                np.zeros(3, dtype=np.float32),
                np.asarray((1, 0, 0, 0), dtype=np.float32),
            ),
        )
        np.testing.assert_allclose(
            _height_at(
                query,
                np.asarray(((0, -1), (0, 0.5), (0, 1), (0, 1.5), (0, 3))),
            ),
            (0.0, 0.5, 1.0, 0.5, 0.0),
            atol=1.0e-6,
        )

    def test_family_selection_excludes_and_records_failed_variants(self) -> None:
        records = tuple(
            GrailSlopeRecord(
                stem=f"terrain_slopes__slope_000__{index:03d}",
                terrain_id="slope_000",
                robot_path=self.root / f"motion-{index}.pkl",
                terrain_path=self.root / f"terrain-{index}.usd",
            )
            for index in range(3)
        )

        def grade(record: GrailSlopeRecord) -> float:
            if record.stem.endswith("__001"):
                raise ValueError("broken terrain")
            return 10.0 + int(record.stem[-1])

        with patch(
            "mm_sonic.build_terrain_pfnn_dataset._traversed_grade",
            side_effect=grade,
        ):
            selected, audit = _select_families(records, 1)

        self.assertEqual(
            [record.stem for record in selected],
            [records[0].stem, records[2].stem],
        )
        self.assertEqual(
            audit["variant_rejections"],
            [{"clip_id": records[1].stem, "reason": "ValueError: broken terrain"}],
        )

    def test_test_values_cannot_change_normalization_bytes(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        self._write(first, self.windows)
        changed = [
            _window(
                window.clip_id,
                1.0e6,
                window.center_frame,
                sequence_lane=window.sequence_lane,
            )
            if window.split == "test"
            else window
            for window in self.windows
        ]
        self._write(second, changed)
        self.assertEqual(
            (first / "normalization.npz").read_bytes(),
            (second / "normalization.npz").read_bytes(),
        )

        with np.load(first / "normalization.npz", allow_pickle=False) as normal:
            self.assertEqual(normal["x_mean"].shape, (288,))
            self.assertEqual(normal["x_std"].shape, (288,))
            self.assertEqual(normal["y_mean"].shape, (268,))
            self.assertEqual(normal["y_std"].shape, (268,))
            np.testing.assert_array_equal(
                normal["y_mean"][OUTPUT_LAYOUT["contact_logit"]], 0.0
            )
            np.testing.assert_array_equal(
                normal["y_std"][OUTPUT_LAYOUT["contact_logit"]], 1.0
            )
            self.assertEqual(int(normal["training_sample_count"]), 2)
            np.testing.assert_array_equal(normal["split_counts"], (2, 1, 1))
            self.assertEqual(normal["split_identity_digest"].shape, ())

    def test_writer_canonicalizes_near_wrap_phase_after_float32_cast(self) -> None:
        window = replace(
            self.windows[0], phase=np.nextafter(2.0 * math.pi, 0.0)
        )
        output = self.root / "dataset"
        manifest = write_pfnn_dataset(
            output,
            [window],
            source_roots=_source_roots(),
            source_records=_source_records([window]),
            rejection_counts={},
        )
        with np.load(output / manifest["shards"][0]["path"], allow_pickle=False) as data:
            self.assertEqual(float(data["phase"][0]), 0.0)
            self.assertLess(float(data["phase"][0]), 2.0 * math.pi)

    def test_writer_revalidates_phase_instead_of_trusting_window_object(self) -> None:
        for invalid in (2.0 * math.pi, -1.0e-9, float("nan")):
            with self.subTest(invalid=invalid):
                window = self.windows[0]
                object.__setattr__(window, "phase", invalid)
                with self.assertRaisesRegex(ValueError, "phase"):
                    write_pfnn_dataset(
                        self.root / f"invalid-{len(str(invalid))}",
                        [window],
                        source_roots=_source_roots(),
                        source_records=_source_records([window]),
                        rejection_counts={},
                    )

    def test_loader_normalizes_and_scales_only_previous_body_inputs(self) -> None:
        output = self.root / "dataset"
        self._write(output, self.windows)
        dataset = PFNNShardDataset(output, "train")

        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        np.testing.assert_array_equal(
            sample["x"],
            normalize_pfnn_input(
                self.windows[0].x, dataset.x_mean, dataset.x_std
            ),
        )
        self.assertEqual(sample["x"].shape, (288,))
        self.assertEqual(sample["y"].shape, (268,))
        self.assertEqual(sample["phase"].shape, ())
        self.assertEqual(sample["clip_id"], "walk1_subject1")
        self.assertEqual(sample["split_identity"], "walk1_subject1")
        self.assertEqual(sample["split"], "train")
        self.assertEqual(sample["sequence_lane"], "motion")
        self.assertEqual(sample["terrain_class"], "flat")
        np.testing.assert_array_equal(
            sample["y"][OUTPUT_LAYOUT["contact_logit"]], (0.0, 1.0, 1.0, 0.0)
        )
        for field in ("previous_body_position", "previous_body_velocity"):
            np.testing.assert_allclose(sample["x"][INPUT_LAYOUT[field]], -0.1)
        np.testing.assert_allclose(sample["x"][INPUT_LAYOUT["trajectory_position"]], -1.0)

    def test_refuses_nonempty_destination(self) -> None:
        output = self.root / "dataset"
        output.mkdir()
        (output / "keep.txt").write_text("owned")
        with self.assertRaisesRegex(FileExistsError, "nonempty"):
            self._write(output, self.windows)
        self.assertEqual((output / "keep.txt").read_text(), "owned")

    def test_complete_resume_is_noop_and_corruption_or_mismatch_fails_closed(self) -> None:
        output = self.root / "dataset"
        original = self._write(output, self.windows)
        files = sorted(path for path in output.rglob("*") if path.is_file())
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
        resumed = write_pfnn_dataset(
            output,
            self.windows,
            source_roots=_source_roots(),
            source_records=_source_records(self.windows),
            rejection_counts={"invalid_phase": 2},
            max_windows_per_shard=2,
            resume=True,
        )
        self.assertEqual(resumed, original)
        self.assertEqual(
            {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}, before
        )

        mismatched_roots = _source_roots()
        mismatched_roots["synthetic"]["license_manifest_sha256"] = _DIGEST_B
        with self.assertRaisesRegex(ValueError, "source set digest"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=mismatched_roots,
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

        shard = output / original["shards"][0]["path"]
        shard.write_bytes(shard.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "shard hash"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_incomplete_resume_appends_only_missing_windows(self) -> None:
        output = self.root / "dataset"
        original = self._write(output, self.windows)
        retained_record = original["shards"][0]
        retained_path = output / retained_record["path"]
        retained_state = (retained_path.read_bytes(), retained_path.stat().st_mtime_ns)
        missing_record = original["shards"][-1]
        (output / missing_record["path"]).unlink()
        (output / "normalization.npz").unlink()
        incomplete = dict(original)
        incomplete["status"] = "building"
        incomplete["shards"] = original["shards"][:-1]
        incomplete.pop("normalization")
        incomplete.pop("dataset_digest_sha256")
        (output / "manifest.json").write_text(json.dumps(incomplete))

        resumed = write_pfnn_dataset(
            output,
            self.windows,
            source_roots=_source_roots(),
            source_records=_source_records(self.windows),
            rejection_counts={"invalid_phase": 2},
            max_windows_per_shard=2,
            resume=True,
        )
        self.assertEqual(resumed["status"], "accepted")
        self.assertEqual(sum(record["count"] for record in resumed["shards"]), 4)
        self.assertEqual(
            (retained_path.read_bytes(), retained_path.stat().st_mtime_ns),
            retained_state,
        )

    def test_incomplete_resume_deduplicates_canonicalized_wrap_phase(self) -> None:
        output = self.root / "dataset"
        windows = [
            replace(
                self.windows[0], phase=np.nextafter(2.0 * math.pi, 0.0)
            ),
            self.windows[1],
        ]
        original = write_pfnn_dataset(
            output,
            windows,
            source_roots=_source_roots(),
            source_records=_source_records(windows),
            rejection_counts={},
            max_windows_per_shard=1,
        )
        missing = original["shards"].pop()
        (output / missing["path"]).unlink()
        (output / "normalization.npz").unlink()
        original["status"] = "building"
        original.pop("normalization")
        original.pop("dataset_digest_sha256")
        (output / "manifest.json").write_text(json.dumps(original))

        resumed = write_pfnn_dataset(
            output,
            windows,
            source_roots=_source_roots(),
            source_records=_source_records(windows),
            rejection_counts={},
            max_windows_per_shard=1,
            resume=True,
        )

        self.assertEqual(sum(record["count"] for record in resumed["shards"]), 2)

    def test_loader_rejects_corrupt_shard_before_returning_samples(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        shard = output / manifest["shards"][0]["path"]
        shard.write_bytes(shard.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "shard hash"):
            PFNNShardDataset(output, "train")

    def test_loader_rejects_incompatible_manifest_constants(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        manifest["joint_order"] = list(reversed(manifest["joint_order"]))
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest contract"):
            PFNNShardDataset(output, "train")

    def test_resume_rejects_manifest_provenance_tampering(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        manifest["source_records"]["walk1_subject1"]["license_id"] = "tampered"
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "source set digest"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_resume_rejects_artifact_paths_outside_destination(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        record = manifest["shards"][0]
        outside = self.root / "outside.npz"
        outside.write_bytes((output / record["path"]).read_bytes())
        record["path"] = "../outside.npz"
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "escapes|sealed grammar"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_resume_rejects_unknown_shard_split_even_with_matching_hashes(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        record = manifest["shards"][0]
        shard = output / record["path"]
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        arrays["split_identity"] = np.full(
            record["count"], "evil", dtype="<U128"
        )
        shard.write_bytes(_deterministic_npz_bytes(arrays))
        record["split"] = "evil"
        record["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "split"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_loader_rejects_duplicate_shard_paths(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        manifest["shards"].append(dict(manifest["shards"][0]))
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "duplicate shard path"):
            PFNNShardDataset(output, "train")

    def test_recomputed_hashes_cannot_hide_row_provenance_tampering(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        record = manifest["shards"][0]
        shard = output / record["path"]
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        arrays["motion_sha256"] = np.full(
            record["count"], _DIGEST_B, dtype="<U64"
        )
        shard.write_bytes(_deterministic_npz_bytes(arrays))
        record["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "row provenance"):
            PFNNShardDataset(output, "train")
        with self.assertRaisesRegex(ValueError, "row provenance"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_recomputed_hashes_cannot_hide_unknown_shard_clip(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        record = manifest["shards"][0]
        shard = output / record["path"]
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        arrays["clip_id"] = np.full(
            record["count"], "unknown_subject", dtype="<U128"
        )
        shard.write_bytes(_deterministic_npz_bytes(arrays))
        record["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))

        with self.assertRaisesRegex(ValueError, "unknown shard clip_id"):
            PFNNShardDataset(output, "train")
        with self.assertRaisesRegex(ValueError, "unknown shard clip_id"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_recomputed_hashes_cannot_hide_negative_center_frame(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        record = manifest["shards"][0]
        shard = output / record["path"]
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        arrays["center_frame"] = np.full(
            record["count"], -1, dtype=np.int32
        )
        shard.write_bytes(_deterministic_npz_bytes(arrays))
        record["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))

        with self.assertRaisesRegex(ValueError, "center_frame"):
            PFNNShardDataset(output, "train")
        with self.assertRaisesRegex(ValueError, "center_frame"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_normalization_metadata_tamper_fails_loader_and_resume(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        normalization = output / "normalization.npz"
        with np.load(normalization, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        arrays["training_sample_count"] = np.asarray(999, dtype=np.int64)
        normalization.write_bytes(_deterministic_npz_bytes(arrays))
        manifest["normalization"]["sha256"] = hashlib.sha256(
            normalization.read_bytes()
        ).hexdigest()
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "normalization metadata"):
            PFNNShardDataset(output, "train")
        with self.assertRaisesRegex(ValueError, "normalization metadata"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

    def test_resume_requires_exact_shard_path_grammar(self) -> None:
        output = self.root / "dataset"
        manifest = self._write(output, self.windows)
        record = manifest["shards"][0]
        old_path = output / record["path"]
        new_path = old_path.with_name("renamed_00000.npz")
        old_path.rename(new_path)
        record["path"] = new_path.relative_to(output).as_posix()
        manifest["dataset_digest_sha256"] = canonical_json_sha256(
            {
                "source_set_digest_sha256": manifest["source_set_digest_sha256"],
                "shards": sorted(manifest["shards"], key=lambda item: item["path"]),
                "normalization_sha256": manifest["normalization"]["sha256"],
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "shard path"):
            write_pfnn_dataset(
                output,
                self.windows,
                source_roots=_source_roots(),
                source_records=_source_records(self.windows),
                rejection_counts={"invalid_phase": 2},
                max_windows_per_shard=2,
                resume=True,
            )

if __name__ == "__main__":
    unittest.main()
