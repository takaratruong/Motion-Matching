import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.torch_motion_data import MotionFolder
from resources.g1_torch_stair_builder.publish import publish_stair_slice
from resources.g1_torch_stair_builder.surface import ZUpHeightGrid
from tests.python.test_torch_stair_conversion import (
    _FakeKinematics,
    write_synthetic_pinned_corpus,
)
from tests.python.torch_motion_test_utils import write_takara_clip


def _flat_grid(_source) -> ZUpHeightGrid:
    return ZUpHeightGrid(
        origin_xy=np.array([-2.0, -2.0], np.float32),
        cell_size_m=1.0,
        height_z=np.zeros((5, 5), np.float32),
    )


class StairSlicePublicationTests(unittest.TestCase):
    def test_publishes_exact_five_clip_native_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grail = root / "grail"
            write_synthetic_pinned_corpus(grail)
            flat_motion = write_takara_clip(root / "flat", frames=60)
            g1_xml = root / "g1.xml"
            g1_xml.write_text("<mujoco/>", encoding="utf-8")
            output = root / "build" / "torch-stair-small"

            manifest = publish_stair_slice(
                output=output,
                grail_root=grail,
                g1_xml=g1_xml,
                flat_motion=flat_motion,
                kinematics=_FakeKinematics(),
                grid_builder=_flat_grid,
            )

            folder = MotionFolder.load(output)
            stored = json.loads((output / "manifest.json").read_text())
            paths = [
                path.relative_to(output).as_posix()
                for path in sorted(output.rglob("motion.npz"))
            ]
            terrain_paths = [
                path.relative_to(output).as_posix()
                for path in sorted(output.rglob("terrain.npz"))
            ]

        self.assertEqual(manifest, stored)
        self.assertEqual(len(folder.clips), 5)
        self.assertEqual(
            paths,
            [
                "flat/motion.npz",
                "stair/0000/motion.npz",
                "stair/0001/motion.npz",
                "stair/0002/motion.npz",
                "stair/updown-0000/motion.npz",
            ],
        )
        self.assertEqual(len(terrain_paths), 4)
        self.assertEqual(stored["schema"], "g1-torch-stair-slice/v1")
        self.assertEqual(stored["output_fps"], 50)
        self.assertEqual(stored["layout"], "g1-29dof-isaaclab-v1")
        self.assertEqual(len(stored["clips"]), 5)
        self.assertEqual(stored["clips"][0]["terrain"]["kind"], "flat")
        self.assertTrue(
            all(
                set(clip["source_sha256"]) == {"robot", "object", "usd"}
                for clip in stored["clips"][1:]
            )
        )

    def test_validation_failure_preserves_existing_destination_and_cleans_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grail = root / "grail"
            write_synthetic_pinned_corpus(grail)
            flat_motion = write_takara_clip(root / "flat", frames=60)
            g1_xml = root / "g1.xml"
            g1_xml.write_text("<mujoco/>", encoding="utf-8")
            output = root / "torch-stair-small"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("original", encoding="utf-8")

            def reject(_path: Path) -> None:
                raise RuntimeError("forced candidate rejection")

            with self.assertRaisesRegex(RuntimeError, "forced candidate"):
                publish_stair_slice(
                    output=output,
                    grail_root=grail,
                    g1_xml=g1_xml,
                    flat_motion=flat_motion,
                    kinematics=_FakeKinematics(),
                    grid_builder=_flat_grid,
                    validate_candidate=reject,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(output.iterdir()), [sentinel])
            leftovers = [
                path.name
                for path in root.iterdir()
                if path.name.startswith(".torch-stair-small.staging-")
            ]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
