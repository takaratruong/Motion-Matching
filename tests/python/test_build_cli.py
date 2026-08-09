import gc
import copy
import hashlib
import io
import json
import os
import struct
import subprocess
import tempfile
import unittest
import weakref
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from resources import build_g1_terrain_database as builder
from resources import validate_g1_terrain_database as validator
from resources.g1_terrain_builder.artifacts import (
    read_support_sidecar,
    read_terrain_sidecar,
    read_walkability,
)
from resources.g1_terrain_builder.database import read_holden_database
from resources.g1_terrain_builder.scenes import REQUIRED_SCENE_IDS
from resources.g1_terrain_builder.schema import HoldenClip, SkeletonSpec


PYTHON = "/home/ubuntu/miniconda3/envs/diffsim/bin/python"


class BuildCliTests(unittest.TestCase):
    @staticmethod
    def _write_xml(directory, payload, name="g1.xml"):
        path = os.path.join(directory, name)
        with open(path, "wb") as stream:
            stream.write(payload)
        return path

    @staticmethod
    def _noncanonical_flat_skeletons():
        reordered_names = list(validator.G1_SKELETON_NAMES)
        reordered_names[2], reordered_names[3] = (
            reordered_names[3], reordered_names[2])
        reordered = SkeletonSpec(
            tuple(reordered_names),
            np.asarray(validator.G1_SKELETON_PARENTS, np.int32),
        )
        changed_parents = list(validator.G1_SKELETON_PARENTS)
        changed_parents[3] = 1
        reparented = SkeletonSpec(
            tuple(validator.G1_SKELETON_NAMES),
            np.asarray(changed_parents, np.int32),
        )
        return (("reordered", reordered), ("reparented", reparented))

    def test_flat_builder_rejects_noncanonical_skeleton_order_and_topology(self):
        args = SimpleNamespace(
            output_fps=60.0,
            retarget_npz="flat.npz",
            retarget_receipt="flat.receipt.json",
            g1_xml="g1.xml",
        )
        source = SimpleNamespace(
            fps=120.0,
            terrain_id="flat",
            qpos=np.zeros((2, 36), np.float32),
        )
        preliminary = SimpleNamespace(
            positions=np.zeros((256, 31, 3), np.float32),
        )
        for name, skeleton in self._noncanonical_flat_skeletons():
            with (
                self.subTest(name=name),
                mock.patch.object(builder, "_require_file"),
                mock.patch.object(
                    builder, "_read_canonical_flat_g1_xml",
                    return_value=b"<mujoco/>",
                ),
                mock.patch.object(
                    builder, "load_mujoco_xml_assets", return_value={},
                ),
                mock.patch.object(
                    builder, "load_retarget_npz", return_value=source),
                mock.patch.object(
                    builder, "_require_canonical_flat_retarget"),
                mock.patch.object(
                    builder.G1Kinematics, "from_xml_bytes",
                    return_value=mock.sentinel.kinematics,
                ),
                mock.patch.object(
                    builder, "convert_source_clip",
                    return_value=(preliminary, skeleton, {}),
                ),
                mock.patch.object(
                    builder, "resample_vectors",
                    side_effect=AssertionError(
                        "builder continued past a noncanonical skeleton"),
                ),
                self.assertRaisesRegex(ValueError, "canonical G1 skeleton"),
            ):
                builder._assemble_flat_candidate(args)

    def test_flat_validator_rejects_self_consistent_noncanonical_skeletons(self):
        for name, skeleton in self._noncanonical_flat_skeletons():
            receipt = {
                "names": list(skeleton.names),
                "parents": skeleton.parents.tolist(),
                "basis": "holden-y-up-right-handed-forward-plus-z",
                "signature": skeleton.signature(),
            }
            database = SimpleNamespace(parents=skeleton.parents.copy())
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "canonical G1 skeleton"),
            ):
                validator._validate_flat_skeleton_receipt(receipt, database)

    def test_v1_and_v2_flat_manifests_are_rejected_before_tree_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            for version in ("v1", "v2"):
                with open(
                    os.path.join(temporary, "manifest.json"), "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(
                        {"schema": f"g1-lmm-flat-data/{version}"}, stream,
                        indent=2, sort_keys=True,
                    )
                    stream.write("\n")
                with (
                    self.subTest(version=version),
                    self.assertRaisesRegex(ValueError, "v3"),
                ):
                    validator.validate_artifact_directory(temporary)

    def test_stale_flat_retarget_rejects_before_kinematics(self):
        args = builder._parser().parse_args([
            "--output-fps", "60", "--flat-only",
            "--retarget-npz",
            "sonic/runs/native-g1-pfnn/sample-retarget/"
            "LocomotionFlat01_000-120hz.npz",
            "--retarget-receipt",
            "sonic/runs/native-g1-pfnn/sample-retarget/"
            "LocomotionFlat01_000-120hz.receipt.json",
        ])
        with (
            mock.patch.object(builder, "G1Kinematics") as kinematics,
            self.assertRaisesRegex(ValueError, "receipt fields"),
        ):
            builder._assemble_flat_candidate(args)
        kinematics.assert_not_called()

    def test_flat_builder_pins_canonical_receipt_before_kinematics(self):
        args = SimpleNamespace(
            output_fps=60.0,
            retarget_npz="flat.npz",
            retarget_receipt="flat.receipt.json",
            g1_xml="g1.xml",
        )
        with open(
            "sonic/runs/native-g1-pfnn/sample-retarget/"
            "LocomotionFlat01_000-walk-only-7659-8171-120hz.receipt.json",
            encoding="utf-8",
        ) as stream:
            original = json.load(stream)
        cases = (
            ("source_sha256", "0" * 64),
            ("prepared_sha256", "0" * 64),
            ("output_sha256", "0" * 64),
            ("start_frame", 7658),
            ("frame_count", 511),
            ("warmup_frames", 119),
            ("grounding", "source"),
            ("grounding_offset_m", 0.0),
            ("pfnn_position_scale", 1.0),
            ("aliases", [["Spine1", "Spine2"],
                         ["LeftToeBase", "LeftToe"]]),
            ("gmr_commit", "0" * 40),
            ("retarget_project_commit", "0" * 40),
        )
        for key, value in cases:
            receipt = copy.deepcopy(original)
            receipt[key] = value
            source = SimpleNamespace(
                name="LocomotionFlat01_000-walk-only-7659-8171-120hz",
                fps=120.0,
                terrain_id="flat",
                qpos=np.zeros((512, 36), np.float32),
                source_frames=np.arange(7659, 8171, dtype=np.int32),
                provenance={
                    "sha256": receipt["output_sha256"],
                    "receipt": receipt,
                },
            )
            with (
                self.subTest(key=key),
                mock.patch.object(builder, "_require_file"),
                mock.patch.object(
                    builder, "load_retarget_npz", return_value=source),
                mock.patch.object(builder, "G1Kinematics") as kinematics,
                self.assertRaisesRegex(ValueError, "canonical flat retarget"),
            ):
                builder._assemble_flat_candidate(args)
            kinematics.assert_not_called()

    def test_flat_builder_rejects_alternate_g1_xml_before_kinematics(self):
        with tempfile.TemporaryDirectory() as temporary:
            alternate = self._write_xml(
                temporary, b"<mujoco><worldbody/></mujoco>\n")
            args = SimpleNamespace(
                output_fps=60.0, g1_xml=alternate,
                retarget_npz="unused.npz",
                retarget_receipt="unused.receipt.json",
            )
            with (
                mock.patch.object(builder, "_require_file"),
                mock.patch.object(
                    builder, "load_retarget_npz",
                    return_value=mock.sentinel.source,
                ),
                mock.patch.object(builder, "_require_canonical_flat_retarget"),
                mock.patch.object(
                    builder, "G1Kinematics", wraps=builder.G1Kinematics,
                ) as kinematics,
                self.assertRaisesRegex(ValueError, "canonical G1 XML"),
            ):
                builder._assemble_flat_candidate(args)
            kinematics.assert_not_called()

    def test_flat_builder_parses_the_authenticated_xml_bytes(self):
        class ParsedAuthenticatedBytes(Exception):
            pass

        observed = {}

        def parse_bytes(xml_bytes, assets):
            observed["xml_bytes"] = xml_bytes
            observed["assets"] = assets
            with open(selected, "ab") as stream:
                stream.write(b"\n<!-- swapped after authentication -->\n")
            raise ParsedAuthenticatedBytes

        with tempfile.TemporaryDirectory() as temporary:
            payload = b"<mujoco><worldbody/></mujoco>\n"
            selected = self._write_xml(temporary, payload)
            descriptor = {
                "asset": "g1.xml",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            args = SimpleNamespace(
                output_fps=60.0, g1_xml=selected,
                retarget_npz="unused.npz",
                retarget_receipt="unused.receipt.json",
            )
            with (
                mock.patch.object(builder, "_require_file"),
                mock.patch.object(
                    builder, "load_retarget_npz",
                    return_value=mock.sentinel.source,
                ),
                mock.patch.object(builder, "_require_canonical_flat_retarget"),
                mock.patch.object(
                    builder, "CANONICAL_FLAT_KINEMATICS_MODEL", descriptor,
                ),
                mock.patch.object(
                    builder, "load_mujoco_xml_assets",
                    return_value={"fixture.bin": b"asset"},
                ),
                mock.patch.object(
                    builder.G1Kinematics, "from_xml_bytes", create=True,
                    side_effect=parse_bytes,
                ),
                self.assertRaises(ParsedAuthenticatedBytes),
            ):
                builder._assemble_flat_candidate(args)
        self.assertEqual(
            hashlib.sha256(observed["xml_bytes"]).hexdigest(),
            descriptor["sha256"],
        )
        self.assertEqual(observed["assets"], {"fixture.bin": b"asset"})

    def test_flat_validator_authenticates_selected_g1_xml_before_payloads(self):
        descriptor = {
            "asset": "g1_29dof.xml",
            "sha256":
                "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
            "size_bytes": 26914,
        }
        with tempfile.TemporaryDirectory() as temporary:
            alternate = self._write_xml(
                temporary, b"<mujoco><worldbody/></mujoco>\n")
            with (
                mock.patch.object(
                    validator, "load_mujoco_xml_assets",
                    side_effect=AssertionError("assets loaded before XML auth"),
                ),
                mock.patch.object(
                    validator, "G1Kinematics", wraps=validator.G1Kinematics,
                ) as kinematics,
                self.assertRaisesRegex(ValueError, "canonical G1 XML"),
            ):
                validator._load_flat_kinematics(
                    {"kinematics_model": descriptor},
                    {"g1_xml": alternate},
                )
            kinematics.assert_not_called()

    def test_flat_validator_parses_authenticated_bytes_after_path_swap(self):
        class ParsedAuthenticatedBytes(Exception):
            pass

        observed = {}
        with tempfile.TemporaryDirectory() as temporary:
            payload = b"<mujoco><worldbody/></mujoco>\n"
            selected = self._write_xml(temporary, payload)
            descriptor = {
                "asset": "g1.xml",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }

            def parse_bytes(xml_bytes, assets):
                observed["xml_bytes"] = xml_bytes
                observed["assets"] = assets
                with open(selected, "ab") as stream:
                    stream.write(b"\n<!-- swapped after authentication -->\n")
                raise ParsedAuthenticatedBytes

            with (
                mock.patch.object(
                    validator, "CANONICAL_FLAT_KINEMATICS_MODEL", descriptor,
                ),
                mock.patch.object(
                    validator, "load_mujoco_xml_assets",
                    return_value={"fixture.bin": b"asset"},
                ),
                mock.patch.object(
                    validator.G1Kinematics, "from_xml_bytes", create=True,
                    side_effect=parse_bytes,
                ),
                self.assertRaises(ParsedAuthenticatedBytes),
            ):
                validator._load_flat_kinematics(
                    {"kinematics_model": descriptor},
                    {"g1_xml": selected},
                )
        self.assertEqual(
            hashlib.sha256(observed["xml_bytes"]).hexdigest(),
            descriptor["sha256"],
        )
        self.assertEqual(observed["assets"], {"fixture.bin": b"asset"})

    def test_flat_validator_requires_exact_kinematics_model_descriptor(self):
        descriptor = {
            "asset": "g1_29dof.xml",
            "sha256":
                "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
            "size_bytes": 26914,
        }
        cases = [("missing", {})]
        for key, value in (
            ("asset", "other-g1.xml"),
            ("sha256", "0" * 64),
            ("size_bytes", 26913),
            ("size_bytes", 26914.0),
        ):
            manifest = {"kinematics_model": copy.deepcopy(descriptor)}
            manifest["kinematics_model"][key] = value
            cases.append((key, manifest))
        for name, manifest in cases:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "kinematics model"),
            ):
                validator._load_flat_kinematics(
                    manifest, {"g1_xml": "/does/not/exist"})

    def test_flat_validator_independently_pins_canonical_receipt(self):
        source = builder.load_retarget_npz(
            "sonic/runs/native-g1-pfnn/sample-retarget/"
            "LocomotionFlat01_000-walk-only-7659-8171-120hz.npz",
            "sonic/runs/native-g1-pfnn/sample-retarget/"
            "LocomotionFlat01_000-walk-only-7659-8171-120hz.receipt.json",
        )
        validator._require_canonical_flat_retarget(source)
        for key, value in (
            ("source_sha256", "0" * 64),
            ("prepared_sha256", "0" * 64),
            ("start_frame", 7658),
            ("frame_count", 511),
            ("warmup_frames", 119),
            ("grounding", "source"),
            ("pfnn_position_scale", 1.0),
            ("aliases", [["Spine1", "Spine2"],
                         ["RightToeBase", "RightToe"]]),
            ("gmr_commit", "0" * 40),
            ("retarget_project_commit", "0" * 40),
        ):
            changed = copy.deepcopy(source)
            changed.provenance["receipt"][key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "canonical flat retarget"),
            ):
                validator._require_canonical_flat_retarget(changed)

    def test_flat_contact_receipt_rejects_one_sided_or_nonmeaningful_runs(self):
        one_sided = np.zeros((20, 2), np.uint8)
        one_sided[:4, 0] = 1
        short = np.zeros((20, 2), np.uint8)
        short[:4, 0] = 1
        short[10:14, 1] = 1
        for name, contacts in (("one-sided", one_sided), ("short", short)):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "bilateral contact"),
            ):
                builder._flat_contact_receipt(
                    contacts,
                    np.array([0], np.int32),
                    np.array([20], np.int32),
                    median_filter_frames=6,
                )

    def test_flat_contact_receipt_reports_alternating_runs(self):
        contacts = np.zeros((24, 2), np.uint8)
        contacts[1:9, 0] = 1
        contacts[14:23, 1] = 1
        self.assertEqual(
            builder._flat_contact_receipt(
                contacts,
                np.array([0], np.int32),
                np.array([24], np.int32),
                median_filter_frames=6,
            ),
            {
                "schema": "g1-lmm-bilateral-contact/v1",
                "left_contact_frames": 8,
                "right_contact_frames": 9,
                "left_run_count": 1,
                "right_run_count": 1,
                "left_max_run_frames": 8,
                "right_max_run_frames": 9,
                "alternating_run_transition_count": 1,
            },
        )

    def test_flat_validator_recomputes_bilateral_contact_observations(self):
        contacts = np.zeros((24, 2), np.uint8)
        contacts[1:9, 0] = 1
        contacts[14:23, 1] = 1
        ranges = (np.array([0], np.int32), np.array([24], np.int32))
        expected = builder._flat_contact_receipt(
            contacts, *ranges, median_filter_frames=6)
        self.assertEqual(
            validator._flat_contact_receipt(
                contacts, *ranges, median_filter_frames=6),
            expected,
        )
        old_labels = np.zeros((24, 2), np.uint8)
        old_labels[:4, 0] = 1
        with self.assertRaisesRegex(ValueError, "bilateral contact"):
            validator._flat_contact_receipt(
                old_labels, *ranges, median_filter_frames=6)

    def test_real_flat_candidate_has_exact_continuity_safe_v3_receipt(self):
        args = builder._parser().parse_args([
            "--output-fps", "60", "--flat-only",
            "--retarget-npz",
            "sonic/runs/native-g1-pfnn/sample-retarget/"
            "LocomotionFlat01_000-walk-only-7659-8171-120hz.npz",
            "--retarget-receipt",
            "sonic/runs/native-g1-pfnn/sample-retarget/"
            "LocomotionFlat01_000-walk-only-7659-8171-120hz.receipt.json",
        ])

        artifacts, features, manifest = builder._assemble_flat_candidate(args)

        self.assertEqual(manifest["schema"], "g1-lmm-flat-data/v3")
        self.assertEqual(manifest["kinematics_model"], {
            "asset": "g1_29dof.xml",
            "sha256":
                "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
            "size_bytes": 26914,
        })
        self.assertEqual(len(artifacts.positions), 256)
        self.assertEqual(features.values.shape, (256, 31))
        self.assertEqual(manifest["source_count"], 1)
        self.assertEqual(manifest["range_count"], 1)
        self.assertEqual(len(manifest["ranges"]), 1)
        self.assertEqual(
            {len(entry) for entry in manifest["ranges"]}, {7})
        self.assertEqual(set(manifest["ranges"][0]), {
            "start", "stop", "source", "source_first_frame",
            "source_last_frame", "motion_class", "terrain_class",
        })
        self.assertEqual(
            max(entry["stop"] - entry["start"]
                for entry in manifest["ranges"]),
            256,
        )
        continuity = manifest["continuity"]
        self.assertEqual(continuity["schema"], "g1-lmm-continuity/v1")
        self.assertEqual(continuity["source_native_rejected_edge_count"], 0)
        self.assertEqual(continuity["database_local_rejected_edge_count"], 0)
        self.assertEqual(continuity["union_rejected_edge_count"], 0)
        self.assertEqual(continuity["dropped_fragment_count"], 0)
        self.assertEqual(continuity["dropped_frame_count"], 0)
        self.assertEqual(continuity["published_range_count"], 1)
        self.assertEqual(continuity["published_frame_count"], 256)
        self.assertAlmostEqual(
            continuity["maximum_admitted_native_step_rad"],
            0.12608182430267334,
        )
        self.assertAlmostEqual(
            continuity["maximum_admitted_local_rotation_step_rad"],
            0.12608181972804253,
        )
        digest_rows = np.asarray([[
            entry["start"], entry["stop"], entry["source_first_frame"],
            entry["source_last_frame"],
        ] for entry in manifest["ranges"]], dtype="<i4")
        self.assertEqual(
            continuity["range_digest_sha256"],
            hashlib.sha256(digest_rows.tobytes(order="C")).hexdigest(),
        )
        source = manifest["sources"][0]
        self.assertEqual(len(source["left_source_index"]), 256)
        map_payload = b"".join((
            np.asarray(source["left_source_index"], dtype="<i4").tobytes(),
            np.asarray(source["right_source_index"], dtype="<i4").tobytes(),
            np.asarray(source["source_alpha"], dtype="<f4").tobytes(),
        ))
        self.assertEqual(
            continuity["source_map_digest_sha256"],
            hashlib.sha256(map_payload).hexdigest(),
        )
        self.assertEqual(manifest["contact_observations"], {
            "schema": "g1-lmm-bilateral-contact/v1",
            "left_contact_frames": 116,
            "right_contact_frames": 117,
            "left_run_count": 3,
            "right_run_count": 4,
            "left_max_run_frames": 49,
            "right_max_run_frames": 45,
            "alternating_run_transition_count": 6,
        })
        self.assertEqual(manifest["contact"], {
            "semantics": "bundled-orange-duck-global-toe-speed-only",
            "speed_threshold": 0.15,
            "median_filter_frames": 6,
            "median_filter_mode": "nearest",
        })

    def test_flat_parser_requires_receipt_bound_60hz_inputs(self):
        args = builder._parser().parse_args([
            "--output-fps", "60", "--flat-only",
            "--retarget-npz", "/motion/flat.npz",
            "--retarget-receipt", "/motion/flat.receipt.json",
            "--output", "/artifacts/flat",
        ])
        self.assertEqual(args.output_fps, 60.0)
        self.assertTrue(args.flat_only)
        self.assertEqual(args.retarget_npz, "/motion/flat.npz")
        self.assertEqual(
            args.retarget_receipt, "/motion/flat.receipt.json")

    def test_flat_cli_writes_60hz_31d_feature_receipt(self):
        args = SimpleNamespace(
            output_fps=60.0, flat_only=True,
            retarget_npz="flat.npz", retarget_receipt="flat.receipt.json",
            output="published", g1_xml="g1.xml", grail_limit=None,
            grail_glob="unused", takara="unused", remap="unused",
        )
        artifacts = object()
        features = object()
        manifest_base = {
            "schema": "g1-lmm-flat-data/v3", "output_fps": 60.0,
            "trajectory_horizons": [20, 40, 60],
            "feature_dimensions": 31,
        }
        finalized = dict(manifest_base, status="accepted")
        with (
            mock.patch.object(
                builder, "_assemble_flat_candidate",
                return_value=(artifacts, features, manifest_base)),
            mock.patch.object(
                builder, "publish_flat_artifacts",
                return_value=finalized) as publish,
        ):
            manifest = builder.build_artifacts(args)
        self.assertEqual(manifest["output_fps"], 60.0)
        self.assertEqual(manifest["trajectory_horizons"], [20, 40, 60])
        self.assertEqual(manifest["feature_dimensions"], 31)
        publish.assert_called_once()

    def test_grail_limit_measures_complete_corpus_before_motion_selection(self):
        args = SimpleNamespace(grail_glob="clips/*.pkl", grail_limit=1)
        paths = [
            "clips/e.pkl", "clips/c.pkl", "clips/a.pkl",
            "clips/d.pkl", "clips/b.pkl",
        ]
        heights = {
            "a": 0.10, "b": 0.20, "c": 0.30, "d": 0.40, "e": 0.50,
        }
        selected = {
            "grail-curb-default": "a",
            "grail-curb-low": "b",
            "grail-curb-medium": "c",
            "grail-curb-high": "d",
        }
        measured = []

        class Terrain:
            def __init__(self, base):
                self.base = base

            def footprint(self):
                measured.append(self.base)
                return {"height": heights[self.base]}

        with (
            mock.patch.object(builder.glob, "glob", return_value=paths),
            mock.patch.object(builder, "_require_file") as require_file,
            mock.patch.object(
                builder.GrailTerrain, "from_base",
                side_effect=lambda base: Terrain(base)),
            mock.patch.object(
                builder, "select_grail_scene_bases",
                return_value=selected, create=True) as select,
        ):
            observed = builder._inspect_grail_corpus(args)

        all_paths, motion_paths, path_by_base, observed_heights, observed_selected = \
            observed
        self.assertEqual(all_paths, tuple(sorted(paths)))
        self.assertEqual(motion_paths, ("clips/a.pkl",))
        self.assertEqual(tuple(path_by_base), ("a", "b", "c", "d", "e"))
        self.assertEqual(measured, ["a", "b", "c", "d", "e"])
        self.assertEqual(observed_heights, heights)
        self.assertEqual(observed_selected, selected)
        self.assertEqual(
            [call.args for call in require_file.call_args_list],
            [(path, "GRAIL clip") for path in sorted(paths)])
        select.assert_called_once_with(heights)

    def test_duplicate_grail_basename_fails_before_surface_measurement(self):
        args = SimpleNamespace(grail_glob="clips/*.pkl", grail_limit=1)
        with (
            mock.patch.object(
                builder.glob, "glob",
                return_value=["clips/a/same.pkl", "clips/b/same.pkl"]),
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(
                builder.GrailTerrain, "from_base") as from_base,
            self.assertRaisesRegex(ValueError, "duplicate GRAIL.*base"),
        ):
            builder._inspect_grail_corpus(args)
        from_base.assert_not_called()

    def test_empty_grail_corpus_is_rejected_even_at_limit_zero(self):
        args = SimpleNamespace(grail_glob="missing/*.pkl", grail_limit=0)
        with (
            mock.patch.object(builder.glob, "glob", return_value=[]),
            self.assertRaisesRegex(FileNotFoundError, "matched no clips"),
        ):
            builder._inspect_grail_corpus(args)

    def test_assembly_streams_sources_and_excludes_route_only_clips(self):
        args = SimpleNamespace(
            output="unused", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        bases = ("motion", "route-a", "route-b", "route-c")
        all_paths = tuple(f"clips/{base}.pkl" for base in bases)
        path_by_base = dict(zip(bases, all_paths))
        measured_heights = {
            "motion": 0.10, "route-a": 0.20,
            "route-b": 0.30, "route-c": 0.40,
        }
        selected = {
            "grail-curb-default": "motion",
            "grail-curb-low": "route-a",
            "grail-curb-medium": "route-b",
            "grail-curb-high": "route-c",
        }
        skeleton = SkeletonSpec(
            tuple(f"bone-{index}" for index in range(31)),
            np.arange(-1, 30, dtype=np.int32))
        source_refs = []
        clip_refs = []
        load_events = []

        class Source:
            pass

        def source(name, terrain_id):
            value = Source()
            value.name = name
            value.terrain_id = terrain_id
            value.fps = 25.0
            value.qpos = np.zeros((2, 36), np.float32)
            value.source_frames = np.arange(2, dtype=np.int64)
            source_refs.append(weakref.ref(value))
            return value

        def load_takara(path, remap):
            load_events.append("takara_walk_50hz")
            return source("takara_walk_50hz", "flat")

        def load_grail(path):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in source_refs))
            base = os.path.splitext(os.path.basename(path))[0]
            load_events.append(base)
            return source(base, base)

        def clip_for(value):
            clip = HoldenClip.empty(2, 31)
            clip.name = value.name
            clip.terrain_id = value.terrain_id
            clip_refs.append((clip.name, weakref.ref(clip)))
            return clip

        report = {
            "fk_max_error_m": 0.0,
            "duration_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        }

        def finalize(value, terrain, kinematics):
            return clip_for(value), skeleton, dict(report)

        def convert(value, kinematics, output_fps):
            return clip_for(value), skeleton, dict(report)

        def all_definitions(observed_heights, clips_by_terrain):
            self.assertEqual(observed_heights, measured_heights)
            self.assertEqual(set(clips_by_terrain), set(bases))
            return ("all-definitions",)

        scene_pack = SimpleNamespace(scenes=tuple(range(14)))

        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(
                builder, "_inspect_grail_corpus",
                return_value=(
                    all_paths, all_paths[:1], path_by_base,
                    measured_heights, selected)),
            mock.patch.object(builder, "G1Kinematics", return_value=object()),
            mock.patch.object(builder, "load_takara", side_effect=load_takara),
            mock.patch.object(builder, "load_grail", side_effect=load_grail),
            mock.patch.object(builder, "FlatTerrain", return_value=object()),
            mock.patch.object(
                builder.GrailTerrain, "from_base", return_value=object()),
            mock.patch.object(builder, "finalize_clip", new=finalize),
            mock.patch.object(
                builder, "convert_source_clip", new=convert),
            mock.patch.object(
                builder, "all_scene_definitions",
                new=all_definitions, create=True),
            mock.patch.object(
                builder, "build_scene_pack",
                new=lambda definitions: scene_pack, create=True),
        ):
            artifacts, manifest, observed_pack = \
                builder._assemble_candidate(args)

        gc.collect()
        self.assertIs(observed_pack, scene_pack)
        self.assertEqual(load_events, [
            "takara_walk_50hz", "motion", "route-a", "route-b", "route-c",
        ])
        self.assertEqual(
            [entry["name"] for entry in manifest["sources"]],
            ["takara_walk_50hz", "motion"])
        self.assertEqual(manifest["total_clips"], 2)
        self.assertEqual(manifest["grail_clips"], 1)
        self.assertEqual(set(manifest), {
            "schema", "output_fps", "feature_dimensions",
            "terrain_dimensions", "support_dimensions",
            "terrain_feature_distances_m", "total_clips", "grail_clips",
            "skipped_clips", "database_frames", "diagnostic_mode",
            "sources", "skeleton", "contact", "surface", "validation",
        })
        self.assertEqual(manifest["schema"], "g1-terrain-artifacts/v2")
        self.assertEqual(manifest["support_dimensions"], 3)
        self.assertEqual(
            manifest["terrain_feature_distances_m"], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(set(manifest["surface"]), {"semantics", "signature"})
        self.assertEqual(set(manifest["validation"]), {
            "schema", "fk_max_error_m", "duration_error_s",
            "quaternion_norm_max_error",
        })
        self.assertEqual(
            manifest["validation"]["schema"],
            "g1-terrain-validation/v1")
        self.assertEqual(len(artifacts.range_starts), 2)
        self.assertEqual(len(artifacts.positions), 4)
        self.assertTrue(all(reference() is None for reference in source_refs))
        self.assertTrue(all(reference() is None for _, reference in clip_refs))

    def test_loaded_grail_identity_must_match_validated_basename(self):
        valid = SimpleNamespace(name="terrain-a", terrain_id="terrain-a")
        builder._require_loaded_grail_source(valid, "terrain-a")
        cases = (
            (SimpleNamespace(name="wrong", terrain_id="terrain-a"), "name"),
            (SimpleNamespace(name="terrain-a", terrain_id="wrong"), "terrain"),
        )
        for source, field in cases:
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "loaded source name/terrain identity changed",
            ):
                builder._require_loaded_grail_source(source, "terrain-a")

    def test_candidate_validator_argv_is_exact_in_both_modes(self):
        base = dict(
            output="unused", grail_glob="/data/grail/*.pkl",
            g1_xml="/models/g1.xml", takara="/motion/takara.npz",
            remap="/motion/remap.npy",
        )
        validator = os.path.join(
            builder.REPOSITORY_ROOT,
            "resources", "validate_g1_terrain_database.py")
        diagnostic_argv = [
            builder.sys.executable, validator, "/tmp/candidate"]
        cases = (
            (0, diagnostic_argv),
            (1, diagnostic_argv),
            (None, [
                builder.sys.executable, validator, "/tmp/candidate",
                "--full-source-validation",
                "--grail-glob", "/data/grail/*.pkl",
                "--g1-xml", "/models/g1.xml",
                "--takara", "/motion/takara.npz",
                "--remap", "/motion/remap.npy",
            ]),
        )
        for limit, expected in cases:
            args = SimpleNamespace(grail_limit=limit, **base)
            with self.subTest(limit=limit), mock.patch.object(
                subprocess, "run",
            ) as run:
                callback = builder._candidate_validation_policy(args)
                callback("/tmp/candidate")
                run.assert_called_once_with(expected, check=True)

    def test_candidate_validator_nonzero_is_a_value_error(self):
        args = SimpleNamespace(
            output="unused", grail_glob="/data/grail/*.pkl", grail_limit=1,
            g1_xml="/models/g1.xml", takara="/motion/takara.npz",
            remap="/motion/remap.npy",
        )
        failure = subprocess.CalledProcessError(7, ["validator"])
        with mock.patch.object(
            subprocess, "run", side_effect=failure,
        ), self.assertRaisesRegex(
            ValueError, "candidate validator failed.*status 7",
        ):
            builder._run_candidate_validator("/tmp/candidate", args)

    def test_build_publishes_assembled_candidate_with_five_arguments(self):
        args = SimpleNamespace(
            output="published", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        artifacts = object()
        manifest_base = {"schema": "g1-terrain-artifacts/v2"}
        scene_pack = object()
        callback = lambda staging: None
        finalized = {"schema": "g1-terrain-artifacts/v2"}
        with (
            mock.patch.object(
                builder, "_assemble_candidate",
                return_value=(artifacts, manifest_base, scene_pack)) as assemble,
            mock.patch.object(
                builder, "_candidate_validation_policy",
                return_value=callback, create=True) as policy,
            mock.patch.object(
                builder, "publish_artifacts",
                return_value=finalized) as publish,
        ):
            observed = builder.build_artifacts(args)

        self.assertIs(observed, finalized)
        assemble.assert_called_once_with(args)
        policy.assert_called_once_with(args)
        publish.assert_called_once_with(
            "published", artifacts, manifest_base, scene_pack, callback)

    def test_parser_removes_the_obsolete_runtime_terrain_option(self):
        parser = builder._parser()
        self.assertNotIn("runtime_terrain", vars(parser.parse_args([])))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--runtime-terrain", "obsolete"])

    def test_success_summary_reports_the_complete_scene_count(self):
        manifest = {
            "schema": "g1-terrain-artifacts/v2",
            "database_frames": 7,
            "total_clips": 2,
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(builder, "build_artifacts", return_value=manifest),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = builder.main(["--output", "/tmp/candidate"])
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            "BUILT g1-terrain-artifacts/v2 frames=7 clips=2 scenes=14 "
            "output=/tmp/candidate\n")

    def test_one_grail_clip_builds_and_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "g1_terrain")
            built = subprocess.run([
                PYTHON, "resources/build_g1_terrain_database.py",
                "--output", output, "--grail-limit", "1",
            ], check=True, text=True, capture_output=True)
            validated = subprocess.run([
                PYTHON, "resources/validate_g1_terrain_database.py",
                output,
            ], check=True, text=True, capture_output=True)
            with open(
                os.path.join(output, "manifest.json"), encoding="utf-8",
            ) as stream:
                manifest = json.load(stream)
            with open(
                os.path.join(output, "scenes", "index.json"),
                encoding="utf-8",
            ) as stream:
                index = json.load(stream)
            database = read_holden_database(
                os.path.join(output, "database.bin"))
            terrain_features = read_terrain_sidecar(
                os.path.join(output, "terrain_features.bin"))
            terrain_support = read_support_sidecar(
                os.path.join(output, "terrain_support.bin"))
            scene_headers = {}
            walkability_shapes = {}
            for scene_id in REQUIRED_SCENE_IDS:
                scene_root = os.path.join(output, "scenes", scene_id)
                with open(os.path.join(scene_root, "terrain.bin"), "rb") as stream:
                    scene_headers[scene_id] = struct.unpack(
                        "<4sIII4f", stream.read(32))
                walkability_shapes[scene_id] = read_walkability(
                    os.path.join(scene_root, "walkability.bin")).shape
            grail_provenance = {}
            for scene_id in REQUIRED_SCENE_IDS[:4]:
                with open(
                    os.path.join(output, "scenes", scene_id, "scene.json"),
                    encoding="utf-8",
                ) as stream:
                    grail_provenance[scene_id] = json.load(stream)[
                        "provenance"]["source_ids"][0]

        self.assertEqual(manifest["schema"], "g1-terrain-artifacts/v2")
        self.assertEqual(built.stderr, "")
        self.assertIn(
            "VALID g1-terrain-artifacts/v2", built.stdout)
        self.assertIn(
            "BUILT g1-terrain-artifacts/v2", built.stdout)
        self.assertIn("clips=2 scenes=14", built.stdout)
        self.assertEqual(manifest["output_fps"], 25.0)
        self.assertEqual(manifest["feature_dimensions"], 31)
        self.assertEqual(manifest["terrain_dimensions"], 4)
        self.assertEqual(manifest["support_dimensions"], 3)
        self.assertEqual(manifest["grail_clips"], 1)
        self.assertEqual(manifest["total_clips"], 2)
        self.assertEqual(manifest["skipped_clips"], 0)
        self.assertTrue(manifest["diagnostic_mode"])
        self.assertEqual(set(manifest["validation"]), {
            "schema", "duration_error_s", "fk_max_error_m",
            "quaternion_norm_max_error",
        })
        self.assertEqual(len(manifest["skeleton"]["names"]), 31)
        self.assertEqual(database.positions.shape[1], 31)
        self.assertEqual(len(database.positions), len(terrain_features))
        self.assertEqual(len(database.positions), len(terrain_support))
        self.assertFalse(os.path.exists(os.path.join(output, "terrain.bin")))
        self.assertFalse(os.path.exists(os.path.join(output, "terrain.obj")))
        self.assertEqual(index["scene_ids"], list(REQUIRED_SCENE_IDS))
        self.assertEqual(
            [descriptor["id"] for descriptor in index["scenes"]],
            list(REQUIRED_SCENE_IDS))
        for scene_id in REQUIRED_SCENE_IDS:
            self.assertEqual(scene_headers[scene_id][:2], (b"G1HF", 2))
            self.assertGreaterEqual(walkability_shapes[scene_id][0], 2)
            self.assertGreaterEqual(walkability_shapes[scene_id][1], 2)
        self.assertEqual(grail_provenance, {
            "grail-curb-default": "terrain_curbs__curb_000__000",
            "grail-curb-low": "terrain_curbs__curb_186__004",
            "grail-curb-medium": "terrain_curbs__curb_022__001",
            "grail-curb-high": "terrain_curbs__curb_165__006",
        })
        takara_stop = manifest["sources"][0]["range_stop"]
        np.testing.assert_array_equal(
            terrain_support[:takara_stop],
            np.zeros((takara_stop, 3), np.float32))
        self.assertEqual(validated.stderr, "")
        self.assertEqual(
            validated.stdout,
            f"VALID g1-terrain-artifacts/v2 frames={len(database.positions)} "
            "clips=2 bones=31 terrain_dims=4 support_dims=3 scenes=14 "
            "source_rows=0\n",
        )
        cursor = 0
        for source in manifest["sources"]:
            self.assertEqual(source["range_start"], cursor)
            self.assertEqual(
                source["range_stop"] - source["range_start"],
                source["output_frames"])
            self.assertEqual(
                len(source["source_frame_map"]), source["output_frames"])
            self.assertEqual(source["range_stop"], cursor + source["output_frames"])
            cursor = source["range_stop"]
        self.assertEqual(cursor, len(database.positions))

    def test_invalid_build_arguments_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "missing")
            cases = (
                ("negative_limit", ["--grail-limit", "-1"], "non-negative"),
                ("empty_positive_glob", [
                    "--grail-glob", os.path.join(td, "none", "*.pkl"),
                    "--grail-limit", "1",
                ], "matched no clips"),
                ("empty_zero_glob", [
                    "--grail-glob", os.path.join(td, "none", "*.pkl"),
                    "--grail-limit", "0",
                ], "matched no clips"),
                ("empty_full_glob", [
                    "--grail-glob", os.path.join(td, "none", "*.pkl"),
                ], "matched no clips"),
                ("missing_g1_xml", [
                    "--g1-xml", missing, "--grail-limit", "0",
                ], "missing G1 XML"),
            )
            for name, arguments, message in cases:
                output = os.path.join(td, name)
                result = subprocess.run([
                    PYTHON, "resources/build_g1_terrain_database.py",
                    "--output", output, *arguments,
                ], text=True, capture_output=True)
                with self.subTest(name=name):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(os.path.exists(output))

    def test_duplicate_source_names_are_rejected_before_conversion(self):
        args = SimpleNamespace(
            output="unused", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        collision = "clips/takara_walk_50hz.pkl"
        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(
                builder, "_inspect_grail_corpus", return_value=(
                    (collision,), (collision,),
                    {"takara_walk_50hz": collision},
                    {"takara_walk_50hz": 0.1},
                    {"grail-curb-default": "takara_walk_50hz"},
                )),
            mock.patch.object(builder, "finalize_clip") as finalize,
        ):
            with self.assertRaisesRegex(ValueError, "duplicate source names"):
                builder._assemble_candidate(args)
        finalize.assert_not_called()

    def test_skeleton_changes_are_rejected_before_combination(self):
        args = SimpleNamespace(
            output="unused", grail_glob="clips/*.pkl", grail_limit=1,
            g1_xml="g1.xml", takara="takara.npz", remap="remap.npy",
        )
        takara = SimpleNamespace(
            name="takara_walk_50hz", terrain_id="flat", fps=25.0,
            qpos=np.zeros((3, 36), np.float32),
        )
        grail = SimpleNamespace(
            name="grail", terrain_id="grail", fps=25.0,
            qpos=np.zeros((3, 36), np.float32),
        )
        clip_a = HoldenClip.empty(3, 2)
        clip_b = HoldenClip.empty(3, 2)
        skeleton_a = SkeletonSpec(
            ("Simulation", "Hips"), np.array([-1, 0], np.int32))
        skeleton_b = SkeletonSpec(
            ("Simulation", "Changed"), np.array([-1, 0], np.int32))
        report = {
            "fk_max_error_m": 0.0,
            "duration_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        }
        grail_path = "clips/grail.pkl"
        with (
            mock.patch.object(builder, "_require_file"),
            mock.patch.object(
                builder, "_inspect_grail_corpus", return_value=(
                    (grail_path,), (grail_path,), {"grail": grail_path},
                    {"grail": 0.1}, {
                        "grail-curb-default": "grail",
                        "grail-curb-low": "grail",
                        "grail-curb-medium": "grail",
                        "grail-curb-high": "grail",
                    },
                )),
            mock.patch.object(builder, "G1Kinematics"),
            mock.patch.object(builder, "load_takara", return_value=takara),
            mock.patch.object(builder, "load_grail", return_value=grail),
            mock.patch.object(
                builder.GrailTerrain, "from_base", return_value=object()),
            mock.patch.object(
                builder, "finalize_clip",
                side_effect=((clip_a, skeleton_a, report),
                             (clip_b, skeleton_b, report))),
            mock.patch.object(builder, "combine_clips") as combine,
        ):
            with self.assertRaisesRegex(ValueError, "grail.*skeleton signature"):
                builder._assemble_candidate(args)
        combine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
