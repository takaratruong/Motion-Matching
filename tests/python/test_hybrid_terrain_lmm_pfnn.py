from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path

import mm_sonic.hybrid_terrain_lmm_pfnn as pfnn_bridge
import numpy as np
from mm_sonic.hybrid_terrain_lmm_pfnn import (
    PFNNSupplement,
    PlaneSurface,
    annotate_holden_clip_terrain,
    build_pfnn_supplement,
    load_pfnn_supplement,
    publish_pfnn_supplement,
    resample_contact_timeline,
    select_half_open_segments,
    summarize_admission,
)

from resources.g1_terrain_builder.artifacts import canonical_json_bytes
from resources.g1_terrain_builder.database import combine_clips, derive_velocities
from resources.g1_terrain_builder.features import build_matching_features
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
    FeatureSet,
    HoldenClip,
    SkeletonSpec,
)

_PINNED_INPUT_FACTS = {
    "g1_xml": (
        26_914,
        "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
    ),
    "patches": (
        620_692_714,
        "344ec49b3aab3c93cd3f618c613e7d8fd8dfaf07964583a7461f0fb7593aae79",
    ),
    "retarget_manifest": (
        11_405,
        "6a477a223d38375124bc34e19d2fdd7eee795b3ef6339b44c21ce26b873d186b",
    ),
    "source_selection": (
        12_086,
        "a1babd74388c5b23593c562213d7116fc280c85bb1abd540bac7c6d28e0e3596",
    ),
    "vertical_dataset_manifest": (
        3_040,
        "eaea027337fa39c53464f15ad69e8946a381e44d08b28c1b027ef5add41311e9",
    ),
}
_EXPECTED_COVERAGE = {
    "LocomotionFlat02_000": ["left_turn", "right_turn"],
    "LocomotionFlat06_000": ["idle_transition", "straight"],
    "LocomotionFlat08_000": ["left_turn", "right_turn"],
    "LocomotionFlat10_000": ["left_turn", "right_turn"],
    "LocomotionFlat11_000": ["idle_transition", "straight"],
    "LocomotionFlat12_000": ["idle_transition", "straight"],
}
_VERTICAL_FIT_COUNTS = (5, 4, 1, 5, 4, 5, 5, 6, 5, 7, 6)
_CONTACT_MAP_SHA256 = (
    "3a60a0b2396a362ff97b4f9fc6c4c0db538ae1efdde58e0f9db94572e1063a39"
)
_FULL_TERRAIN_RECEIPT_SET_SHA256 = (
    "17ccb203f59a7cfc5440bdb1cc2813828a1b8e2c07fd0f37fc1d320846a002b3"
)


def _synthetic_descriptor(label: str) -> dict[str, object]:
    payload = label.encode("utf-8")
    return {
        "path": f"/synthetic/{label}",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _synthetic_supplement() -> PFNNSupplement:
    skeleton = SkeletonSpec(
        G1_SKELETON_NAMES,
        np.asarray(G1_SKELETON_PARENTS, dtype=np.int32),
    )
    stem_families = (
        ("LocomotionFlat02_000", "flat"),
        ("LocomotionFlat06_000", "flat"),
        ("LocomotionFlat08_000", "flat"),
        ("LocomotionFlat10_000", "flat"),
        ("LocomotionFlat11_000", "flat"),
        ("LocomotionFlat12_000", "flat"),
        ("WalkingUpSteps01_000", "stair"),
        ("WalkingUpSteps02_000", "stair"),
        ("WalkingUpSteps03_000", "stair"),
        ("WalkingUpSteps04_000", "stair"),
        ("WalkingUpSteps05_000", "stair"),
        ("WalkingUpSteps06_000", "stair"),
        ("WalkingUpSteps07_000", "stair"),
        ("WalkingUpSteps08_000", "stair"),
        ("WalkingUpSteps09_000", "stair"),
        ("WalkingUpSteps11_000", "stair"),
        ("WalkingUpSteps12_000", "stair"),
    )
    clips = []
    ranges = []
    vertical_index = 0
    for index, (stem, family) in enumerate(stem_families):
        clip = HoldenClip.empty(125, 31)
        clip.name = stem
        clip.positions[:, 0, 2] = index + np.arange(125, dtype=np.float32) * np.float32(
            0.04
        )
        if family == "stair":
            clip.terrain_features[:] = np.array(
                [0.025, 0.05, 0.075, 0.1], dtype=np.float32
            )
            clip.terrain_support[:] = np.float32(0.1)
        with np.errstate(divide="ignore", invalid="ignore"):
            clip.velocities, clip.angular_velocities = derive_velocities(
                clip.positions, clip.rotations, 25.0
            )
        clips.append(clip)
        source_start = index * 1_000
        core_start = source_start + 120
        core_stop = source_start + 720
        retarget = _synthetic_descriptor(f"{stem}__retarget.npz")
        retarget_receipt = _synthetic_descriptor(
            f"{stem}__retarget.receipt.json"
        )
        source_bvh = _synthetic_descriptor(f"{stem}.bvh")
        terrain_fits = []
        if family == "stair":
            fit_count = _VERTICAL_FIT_COUNTS[vertical_index]
            vertical_index += 1
            boundaries = np.linspace(
                core_start, core_stop, fit_count + 1, dtype=np.int64
            )
            for fit_index, (fit_start, fit_stop) in enumerate(
                itertools.pairwise(boundaries)
            ):
                artifact = _synthetic_descriptor(
                    f"{stem}__fit-{fit_index:02d}.npz"
                )
                fit_receipt = _synthetic_descriptor(
                    f"{stem}__fit-{fit_index:02d}.receipt.json"
                )
                terrain_fits.append(
                    {
                        "artifact": artifact,
                        "receipt": fit_receipt,
                        "source_start_frame_120hz": int(fit_start),
                        "source_stop_frame_120hz": int(fit_stop),
                    }
                )
        ranges.append(
            {
                "contact_source_rows_sha256": _CONTACT_MAP_SHA256,
                "core_start_120hz": core_start,
                "core_stop_120hz": core_stop,
                "coverage": ["ascent", "descent"]
                if family == "stair"
                else _EXPECTED_COVERAGE[stem],
                "duration_error_s": 0.0,
                "fk_max_error_m": 0.0,
                "quaternion_norm_max_error": 0.0,
                "retarget": retarget,
                "retarget_receipt": retarget_receipt,
                "source_bvh": source_bvh,
                "source_interval_start_120hz": source_start,
                "source_interval_stop_120hz": source_start + 840,
                "terrain_fits": terrain_fits,
                "stem": stem,
                "role": "train",
                "family": family,
                "start": index * 125,
                "stop": (index + 1) * 125,
                "source_rows_120hz": 840,
                "core_rows_120hz": 600,
                "reference_rows_60hz": 300,
                "output_rows_25hz": 125,
            }
        )
    artifacts = combine_clips(clips, skeleton)
    features = build_matching_features(artifacts, 25.0, (8, 17, 25))
    input_descriptors = {
        name: {
            "path": f"/synthetic/{name}",
            "size_bytes": size,
            "sha256": digest,
        }
        for name, (size, digest) in _PINNED_INPUT_FACTS.items()
    }
    receipt = {
        "schema": "pfnn-terrain-lmm-supplement/v1",
        "status": "accepted",
        "output_fps": 25.0,
        "trajectory_horizons": [8, 17, 25],
        "ranges": ranges,
        "counts": {
            "excluded_clips": 4,
            "excluded_core_rows_120hz": 2_400,
            "excluded_output_rows_25hz": 500,
            "excluded_reference_rows_60hz": 1_200,
            "selected_clips": 17,
            "selected_core_rows_120hz": 10_200,
            "selected_flat_clips": 6,
            "selected_output_rows_25hz": 2_125,
            "selected_reference_rows_60hz": 5_100,
            "selected_source_rows_120hz": 14_280,
            "selected_vertical_clips": 11,
            "terrain_fit_artifacts_excluded": 3,
            "terrain_fit_artifacts_selected": 53,
            "terrain_fit_artifacts_total": 56,
        },
        "inputs": input_descriptors,
        "terrain": {
            "semantics": "PFNN fitted surface lookahead relative to current support",
            "lookahead_m": [0.25, 0.5, 0.75, 1.0],
            "support_columns": [
                "source_root_height_m",
                "source_left_toe_height_m",
                "source_right_toe_height_m",
            ],
            "receipt_set_sha256": _FULL_TERRAIN_RECEIPT_SET_SHA256,
        },
        "validation": {
            "range_local": True,
            "all_source_derivatives_range_local": True,
            "maximum_fk_error_m": 0.0,
            "maximum_duration_error_s": 0.0,
            "maximum_quaternion_norm_error": 0.0,
            "terrain_feature_max_abs_m": float(
                np.max(np.abs(artifacts.terrain_features))
            ),
            "terrain_feature_std_m": float(np.std(artifacts.terrain_features)),
            "support_min_m": float(np.min(artifacts.terrain_support)),
            "support_max_m": float(np.max(artifacts.terrain_support)),
        },
        "skeleton": {
            "names": list(G1_SKELETON_NAMES),
            "parents": list(G1_SKELETON_PARENTS),
            "signature": G1_SKELETON_SIGNATURE,
        },
    }
    return PFNNSupplement(artifacts, features, receipt)


class PFNNSupplementTest(unittest.TestCase):
    @unittest.skipUnless(
        Path("sonic/runs/native-g1-pfnn/expanded/retarget-manifest.json").is_file()
        and Path("/home/ubuntu/datasets/pfnn/pfnn/data/animations").is_dir(),
        "authenticated PFNN overnight inputs are unavailable",
    )
    def test_retarget_manifest_receipt_must_be_the_output_companion(self) -> None:
        source_root = Path("sonic/runs/native-g1-pfnn/expanded").resolve()
        selection = json.loads((source_root / "source-selection.json").read_text())
        manifest = json.loads((source_root / "retarget-manifest.json").read_text())

        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary)
            (copied_root / "source-selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            for item in manifest["items"]:
                for field in ("output", "receipt"):
                    relative = Path(item[field])
                    destination = copied_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source_root / relative, destination)

            first, second = manifest["items"][:2]
            first["receipt"] = second["receipt"]
            first["receipt_sha256"] = second["receipt_sha256"]
            (copied_root / "retarget-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "companion"):
                pfnn_bridge._authenticate_source_documents(
                    copied_root, Path("/home/ubuntu/datasets/pfnn/pfnn")
                )

    @unittest.skipUnless(
        Path("sonic/runs/native-g1-pfnn/expanded/retarget-manifest.json").is_file()
        and Path("/home/ubuntu/datasets/pfnn/pfnn/data/animations").is_dir(),
        "authenticated PFNN overnight inputs are unavailable",
    )
    def test_retarget_companion_receipt_cannot_resolve_to_an_alias(self) -> None:
        source_root = Path("sonic/runs/native-g1-pfnn/expanded").resolve()
        selection = json.loads((source_root / "source-selection.json").read_text())
        manifest = json.loads((source_root / "retarget-manifest.json").read_text())

        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary)
            (copied_root / "source-selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            for item in manifest["items"]:
                for field in ("output", "receipt"):
                    relative = Path(item[field])
                    destination = copied_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source_root / relative, destination)

            first, second = manifest["items"][:2]
            first_receipt = copied_root / first["receipt"]
            first_receipt.unlink()
            first_receipt.symlink_to(copied_root / second["receipt"])
            first["receipt_sha256"] = second["receipt_sha256"]
            (copied_root / "retarget-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "companion"):
                pfnn_bridge._authenticate_source_documents(
                    copied_root, Path("/home/ubuntu/datasets/pfnn/pfnn")
                )

    def test_audited_train_roles_produce_exact_overnight_counts(self) -> None:
        items = []
        for index in range(6):
            items.append(
                {
                    "stem": f"LocomotionFlat{index:02d}_000",
                    "role": "train",
                    "coverage": ["straight"],
                    "start_frame_120hz": 1000 * index,
                    "stop_frame_120hz": 1000 * index + 840,
                }
            )
        for index in range(11):
            items.append(
                {
                    "stem": f"WalkingUpSteps{index:02d}_000",
                    "role": "train",
                    "coverage": ["ascent", "descent"],
                    "start_frame_120hz": 20000 + 1000 * index,
                    "stop_frame_120hz": 20840 + 1000 * index,
                }
            )
        for index in range(4):
            items.append(
                {
                    "stem": f"HeldOut{index:02d}",
                    "role": "validation",
                    "coverage": ["straight"],
                    "start_frame_120hz": 40000 + 1000 * index,
                    "stop_frame_120hz": 40840 + 1000 * index,
                }
            )

        receipt = summarize_admission(items, terrain_fit_count=56)

        self.assertEqual(receipt["selected_clips"], 17)
        self.assertEqual(receipt["excluded_clips"], 4)
        self.assertEqual(receipt["selected_flat_clips"], 6)
        self.assertEqual(receipt["selected_vertical_clips"], 11)
        self.assertEqual(receipt["selected_core_rows_120hz"], 10_200)
        self.assertEqual(receipt["selected_reference_rows_60hz"], 5_100)
        self.assertEqual(receipt["selected_output_rows_25hz"], 2_125)
        self.assertEqual(receipt["terrain_fit_artifacts_total"], 56)

    def test_segment_selection_is_half_open_and_rejects_gaps(self) -> None:
        positions = np.array([120.0, 159.999, 160.0, 219.5, 220.0])
        selected = select_half_open_segments(
            positions,
            starts=np.array([120, 160, 220]),
            stops=np.array([160, 220, 240]),
        )
        np.testing.assert_array_equal(selected, [0, 0, 1, 1, 2])

        with self.assertRaisesRegex(ValueError, "gap"):
            select_half_open_segments(
                np.array([165.0]),
                starts=np.array([120, 170]),
                stops=np.array([160, 200]),
            )

    def test_contact_resampling_is_range_local_and_has_exact_25hz_count(self) -> None:
        contacts = np.zeros((600, 4), dtype=np.bool_)
        contacts[:300, 0] = True
        contacts[300:, 3] = True

        bilateral, selected_source_rows = resample_contact_timeline(
            contacts, source_fps=120.0, target_fps=25.0
        )

        self.assertEqual(bilateral.shape, (125, 2))
        self.assertEqual(bilateral.dtype, np.uint8)
        self.assertEqual(selected_source_rows.shape, (125,))
        self.assertGreaterEqual(int(selected_source_rows.min()), 0)
        self.assertLess(int(selected_source_rows.max()), 600)
        self.assertTrue(np.all(bilateral[:63, 0] == 1))
        self.assertTrue(np.all(bilateral[63:, 1] == 1))

    def test_terrain_channels_are_support_relative(self) -> None:
        clip = HoldenClip.empty(8, 31)
        clip.name = "plane"
        clip.positions[:, 0, 2] = np.arange(8, dtype=np.float32) * 0.04
        surface = PlaneSurface(offset=0.3, slope_x=0.0, slope_z=0.2)

        annotate_holden_clip_terrain(
            clip,
            surfaces=(surface,),
            surface_indices=np.zeros(8, dtype=np.int32),
            fps=25.0,
        )

        expected_support = 0.3 + 0.2 * clip.positions[:, 0, 2]
        np.testing.assert_allclose(
            clip.terrain_support[:, 0], expected_support, atol=1.0e-6
        )
        np.testing.assert_allclose(
            clip.terrain_features,
            np.tile(np.array([0.05, 0.10, 0.15, 0.20]), (8, 1)),
            atol=1.0e-6,
        )

    def test_published_supplement_round_trips_and_detects_tampering(self) -> None:
        supplement = _synthetic_supplement()

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "supplement"
            manifest = publish_pfnn_supplement(supplement, output)
            manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
            loaded = load_pfnn_supplement(
                manifest.parent, expected_manifest_sha256=manifest_sha256
            )
            np.testing.assert_array_equal(
                loaded.artifacts.positions, supplement.artifacts.positions
            )
            np.testing.assert_array_equal(
                loaded.artifacts.terrain_features,
                supplement.artifacts.terrain_features,
            )
            np.testing.assert_array_equal(
                loaded.features.values, supplement.features.values
            )

            with (output / "terrain.bin").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_pfnn_supplement(
                    output, expected_manifest_sha256=manifest_sha256
                )

    def test_supplement_rejects_receipt_claims_that_differ_from_binary(self) -> None:
        accepted = _synthetic_supplement()

        def wrong_count(receipt: dict[str, object]) -> None:
            receipt["counts"]["selected_output_rows_25hz"] = 2_124

        def wrong_range(receipt: dict[str, object]) -> None:
            receipt["ranges"][1]["stop"] = 249

        def duplicate_stem(receipt: dict[str, object]) -> None:
            receipt["ranges"][1]["stem"] = receipt["ranges"][0]["stem"]

        def missing_range_count(receipt: dict[str, object]) -> None:
            del receipt["ranges"][0]["source_rows_120hz"]

        def missing_top_level_field(receipt: dict[str, object]) -> None:
            del receipt["validation"]

        for label, mutate in (
            ("count", wrong_count),
            ("range", wrong_range),
            ("stem", duplicate_stem),
            ("range count", missing_range_count),
            ("top-level field", missing_top_level_field),
        ):
            with self.subTest(label=label):
                receipt = copy.deepcopy(accepted.receipt)
                mutate(receipt)
                with self.assertRaisesRegex(ValueError, "receipt.*binary"):
                    PFNNSupplement(accepted.artifacts, accepted.features, receipt)

    def test_supplement_rejects_unbound_nested_provenance(self) -> None:
        accepted = _synthetic_supplement()

        def empty_inputs(receipt: dict[str, object]) -> None:
            receipt["inputs"] = {}

        def false_locality(receipt: dict[str, object]) -> None:
            receipt["validation"]["range_local"] = False

        def wrong_terrain_aggregate(receipt: dict[str, object]) -> None:
            receipt["validation"]["terrain_feature_std_m"] = 123.0

        def fit_gap(receipt: dict[str, object]) -> None:
            receipt["ranges"][6]["terrain_fits"][0][
                "source_stop_frame_120hz"
            ] -= 1

        def wrong_receipt_set(receipt: dict[str, object]) -> None:
            receipt["terrain"]["receipt_set_sha256"] = "0" * 64

        for label, mutate in (
            ("inputs", empty_inputs),
            ("locality", false_locality),
            ("aggregate", wrong_terrain_aggregate),
            ("fit topology", fit_gap),
            ("receipt set", wrong_receipt_set),
        ):
            with self.subTest(label=label):
                receipt = copy.deepcopy(accepted.receipt)
                mutate(receipt)
                with self.assertRaisesRegex(ValueError, "PFNN supplement"):
                    PFNNSupplement(accepted.artifacts, accepted.features, receipt)

    def test_loader_requires_expected_manifest_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "supplement"
            manifest = publish_pfnn_supplement(_synthetic_supplement(), output)
            expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaisesRegex(TypeError, "expected manifest"):
                load_pfnn_supplement(output)
            loaded = load_pfnn_supplement(
                output, expected_manifest_sha256=expected
            )
            self.assertEqual(len(loaded.artifacts.positions), 2_125)
            with self.assertRaisesRegex(ValueError, "expected manifest"):
                load_pfnn_supplement(
                    output, expected_manifest_sha256="0" * 64
                )

    def test_supplement_rejects_noncanonical_binary_skeleton(self) -> None:
        accepted = _synthetic_supplement()
        artifacts = copy.deepcopy(accepted.artifacts)
        artifacts.parents[-1] = 0
        features = build_matching_features(artifacts, 25.0, (8, 17, 25))

        with self.assertRaisesRegex(ValueError, "canonical G1 skeleton"):
            PFNNSupplement(artifacts, features, accepted.receipt)

    def test_supplement_rejects_features_that_differ_from_exact_recompute(self) -> None:
        accepted = _synthetic_supplement()
        values = accepted.features.values.copy()
        values[0, 0] += np.float32(0.25)
        features = FeatureSet(
            values, accepted.features.offset.copy(), accepted.features.scale.copy()
        )

        with self.assertRaisesRegex(ValueError, "recomputed 31-D features"):
            PFNNSupplement(accepted.artifacts, features, accepted.receipt)

    def test_publish_revalidates_mutable_receipt_before_writing(self) -> None:
        supplement = _synthetic_supplement()
        supplement.receipt["counts"]["selected_output_rows_25hz"] = 2_124

        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "receipt.*binary"),
        ):
            publish_pfnn_supplement(supplement, Path(temporary) / "must-not-publish")

    def test_loader_rejects_mutated_receipt_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "supplement"
            manifest_path = publish_pfnn_supplement(_synthetic_supplement(), output)
            receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
            receipt["counts"]["selected_output_rows_25hz"] = 2_124
            manifest_path.write_bytes(canonical_json_bytes(receipt))
            with self.assertRaisesRegex(ValueError, "receipt.*binary"):
                load_pfnn_supplement(
                    output,
                    expected_manifest_sha256=hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                )

    def test_loader_rejects_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "supplement"
            manifest = publish_pfnn_supplement(_synthetic_supplement(), output)
            manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (output / "unexpected.txt").write_text("not part of the publication")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_pfnn_supplement(
                    output, expected_manifest_sha256=manifest_sha256
                )

    @unittest.skipUnless(
        Path("sonic/runs/native-g1-pfnn/expanded/retarget-manifest.json").is_file()
        and Path("/home/ubuntu/datasets/pfnn/pfnn/patches.npz").is_file(),
        "authenticated PFNN overnight inputs are unavailable",
    )
    def test_real_bridge_authenticates_all_inputs_and_builds_exact_train_split(
        self,
    ) -> None:
        supplement = build_pfnn_supplement(
            retarget_root=Path("sonic/runs/native-g1-pfnn/expanded"),
            pfnn_root=Path("/home/ubuntu/datasets/pfnn/pfnn"),
            g1_xml=Path(
                "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
            ),
        )

        self.assertEqual(supplement.artifacts.positions.shape, (2_125, 31, 3))
        self.assertEqual(supplement.features.values.shape, (2_125, 31))
        np.testing.assert_array_equal(
            supplement.artifacts.range_starts,
            np.arange(17, dtype=np.int32) * 125,
        )
        np.testing.assert_array_equal(
            supplement.artifacts.range_stops,
            np.arange(1, 18, dtype=np.int32) * 125,
        )
        counts = supplement.receipt["counts"]
        self.assertEqual(counts["selected_clips"], 17)
        self.assertEqual(counts["excluded_clips"], 4)
        self.assertEqual(counts["terrain_fit_artifacts_total"], 56)
        self.assertEqual(counts["terrain_fit_artifacts_selected"], 53)
        self.assertEqual(counts["terrain_fit_artifacts_excluded"], 3)
        self.assertTrue(np.isfinite(supplement.artifacts.terrain_support).all())
        self.assertGreater(
            float(np.max(np.abs(supplement.artifacts.terrain_features))), 1.0e-3
        )
        self.assertTrue(
            all(value["role"] == "train" for value in supplement.receipt["ranges"])
        )


if __name__ == "__main__":
    unittest.main()
