import json
import io
from pathlib import Path
from types import MappingProxyType
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

import numpy as np

from mm_sonic.torch_motion_data import MotionFolder
from mm_sonic.torch_terrain_features import TerrainDataset
from resources.g1_torch_stair_builder.conversion import NativeMotionArrays
from resources.g1_torch_stair_builder.surface import ZUpHeightGrid
from resources.g1_torch_terrain_builder.publish import publish_expanded_corpus
from resources.g1_torch_terrain_builder.registry import (
    ResolvedSource,
    SourceSpec,
)
from resources.g1_torch_terrain_builder.terrain import TerrainEvidence
from tests.python.torch_motion_test_utils import write_takara_clip


def _motion() -> NativeMotionArrays:
    frames = 60
    joint = np.zeros((frames, 29), np.float32)
    body = np.zeros((frames, 30, 3), np.float32)
    body[:, 0, 2] = 0.8
    body[:, (18, 19), 2] = 0.035
    quaternion = np.zeros((frames, 30, 4), np.float32)
    quaternion[..., 0] = 1.0
    return NativeMotionArrays(
        fps=50,
        joint_position=joint,
        joint_velocity=np.zeros_like(joint),
        body_position_world=body,
        body_quaternion_world_wxyz=quaternion,
        body_linear_velocity_world=np.zeros_like(body),
        body_angular_velocity_world=np.zeros_like(body),
    )


def _source(root: Path, name: str, digest: str) -> ResolvedSource:
    path = root / f"{name}.npz"
    path.write_bytes(name.encode("utf-8"))
    spec = SourceSpec(
        logical_name=name,
        family="stair-local",
        source_adapter="native-npz",
        motion_relative_path=path.name,
        motion_sha256=digest,
        terrain_adapter="fixed-staircase",
        geometry_relative_paths=(),
    )
    return ResolvedSource(
        spec=spec,
        motion_path=path,
        geometry_paths=(),
        source_sha256=MappingProxyType({"motion": digest}),
    )


def _terrain() -> TerrainEvidence:
    return TerrainEvidence(
        adapter="fixed-staircase",
        grid=ZUpHeightGrid(
            origin_xy=np.array([-2.0, -2.0], np.float32),
            cell_size_m=1.0,
            height_z=np.zeros((5, 5), np.float32),
        ),
        geometry_sha256=MappingProxyType({}),
        motion_to_terrain_xy_yaw=(0.0, 0.0, 0.0),
    )


class ExpandedTerrainPublicationTests(unittest.TestCase):
    def test_publishes_accepted_and_records_rejected_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = write_takara_clip(root / "flat", frames=60)
            xml = root / "g1.xml"
            xml.write_text("<mujoco/>", encoding="utf-8")
            accepted = _source(root, "side-step", "a" * 64)
            missing = _source(root, "missing", "b" * 64)
            duplicate = _source(root, "duplicate", "a" * 64)
            motion = _motion()

            def terrain_builder(source, _motion):
                return None if source.spec.logical_name == "missing" else _terrain()

            manifest = publish_expanded_corpus(
                output=root / "corpus",
                source_root=root,
                grail_root=root,
                g1_xml=xml,
                flat_motion=flat,
                resolved_sources=(accepted, missing, duplicate),
                motion_loader=lambda _source: motion,
                terrain_builder=terrain_builder,
                fk=lambda _qpos: (
                    motion.body_position_world.copy(),
                    motion.body_quaternion_world_wxyz.copy(),
                ),
            )
            folder = MotionFolder.load(root / "corpus")
            dataset = TerrainDataset.load(root / "corpus", device="cpu")
            stored = json.loads(
                (root / "corpus" / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest, stored)
        self.assertEqual(manifest["schema"], "g1-torch-terrain-corpus/v1")
        self.assertEqual(
            {entry["logical_name"] for entry in manifest["accepted_clips"]},
            {"flat-takara", "side-step"},
        )
        self.assertEqual(
            [entry["reason"] for entry in manifest["rejected_candidates"]],
            ["missing_authoritative_terrain", "exact_duplicate"],
        )
        self.assertEqual(len(folder.clips), 2)
        self.assertEqual(len(dataset.folder.clips), 2)

    def test_precommit_failure_preserves_destination_and_removes_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = write_takara_clip(root / "flat", frames=60)
            xml = root / "g1.xml"
            xml.write_text("<mujoco/>", encoding="utf-8")
            output = root / "corpus"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("original", encoding="utf-8")

            def reject(_path):
                raise RuntimeError("forced precommit failure")

            with self.assertRaisesRegex(RuntimeError, "forced precommit"):
                publish_expanded_corpus(
                    output=output,
                    source_root=root,
                    grail_root=root,
                    g1_xml=xml,
                    flat_motion=flat,
                    resolved_sources=(),
                    motion_loader=lambda _source: _motion(),
                    terrain_builder=lambda _source, _motion: _terrain(),
                    fk=lambda _qpos: (_motion().body_position_world, _motion().body_quaternion_world_wxyz),
                    validate_candidate=reject,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(output.iterdir()), [sentinel])
            self.assertEqual(
                [
                    path
                    for path in root.iterdir()
                    if path.name.startswith(".corpus.staging-")
                ],
                [],
            )

    def test_cli_writes_full_admission_report_and_published_summary(self):
        from resources import build_g1_torch_terrain_corpus as cli

        manifest = {
            "schema": "g1-torch-terrain-corpus/v1",
            "clips": [{}, {}],
            "accepted_clips": [{}, {}],
            "rejected_candidates": [{"reason": "fk_mismatch"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "corpus"
            report = root / "admission.json"
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    cli,
                    "publish_expanded_corpus",
                    return_value=manifest,
                ) as publish,
                redirect_stdout(stdout),
            ):
                result = cli.main(
                    [
                        "--source-root",
                        str(root / "sources"),
                        "--grail-root",
                        str(root / "grail"),
                        "--g1-xml",
                        str(root / "g1.xml"),
                        "--flat-motion",
                        str(root / "flat.npz"),
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )
            summary = json.loads(stdout.getvalue())
            stored = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(stored, manifest)
        self.assertEqual(summary["status"], "PUBLISHED")
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["rejected"], 1)
        publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
