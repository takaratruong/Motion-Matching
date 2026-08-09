from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mm_sonic.build_g1_pfnn_vertical_slice import retarget_vertical_slice
from mm_sonic.terrain_pfnn.source_pfnn_released import (
    PFNNSliceRole,
    ReleasedPFNNRecord,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildG1PFNNVerticalSliceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        self.selection = []
        for index, (stem, role, coverage) in enumerate(
            (
                ("LocomotionFlat06_000", "train", ("idle_transition", "straight")),
                ("LocomotionFlat02_000", "train", ("left_turn", "right_turn")),
                ("WalkingUpSteps01_000", "train", ("ascent", "descent")),
                ("WalkingUpSteps02_000", "validation", ("ascent", "descent")),
            )
        ):
            members = []
            for suffix in (".bvh", ".phase", ".gait", "_footsteps.txt"):
                path = source / f"{stem}{suffix}"
                path.write_text(f"{stem}:{suffix}\n", encoding="utf-8")
                members.append(path)
            record = ReleasedPFNNRecord(
                stem=stem,
                bvh_path=members[0],
                phase_path=members[1],
                gait_path=members[2],
                footsteps_path=members[3],
                bvh_sha256=_sha256(members[0]),
                phase_sha256=_sha256(members[1]),
                gait_sha256=_sha256(members[2]),
                footsteps_sha256=_sha256(members[3]),
                frame_count=3000,
                fps=120.0,
            )
            self.selection.append(
                PFNNSliceRole(
                    record=record,
                    role=role,
                    start_frame_120hz=240 + 60 * index,
                    stop_frame_120hz=1080 + 60 * index,
                    coverage=coverage,
                )
            )
        self.selection = tuple(self.selection)
        self.calls = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fake_retarget(self, **arguments):
        self.calls.append(dict(arguments))
        output = Path(arguments["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(
            f"{Path(arguments['source']).stem}:{arguments['start_frame']}:"
            f"{arguments['frame_count']}".encode()
        )
        receipt = output.with_suffix(".receipt.json")
        receipt.write_text(
            json.dumps(
                {
                    "schema": "fixture-retarget/v1",
                    "output_sha256": _sha256(output),
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "accepted",
            "output": str(output.resolve()),
            "receipt": str(receipt.resolve()),
            "output_sha256": _sha256(output),
        }

    def test_build_is_atomic_and_resume_is_a_verified_noop(self) -> None:
        output = self.root / "run"
        manifest = retarget_vertical_slice(
            self.selection,
            output=output,
            gmr_root=self.root,
            retarget_project_root=self.root,
            retarget_one=self._fake_retarget,
        )
        self.assertEqual(manifest["status"], "accepted")
        self.assertEqual(len(manifest["items"]), 4)
        self.assertEqual(len(self.calls), 4)
        before = (output / "retarget-manifest.json").read_bytes()
        self.calls.clear()
        resumed = retarget_vertical_slice(
            self.selection,
            output=output,
            gmr_root=self.root,
            retarget_project_root=self.root,
            resume=True,
            retarget_one=self._fake_retarget,
        )
        self.assertEqual(resumed, manifest)
        self.assertEqual(self.calls, [])
        self.assertEqual((output / "retarget-manifest.json").read_bytes(), before)

    def test_resume_rejects_output_tamper_before_retarget(self) -> None:
        output = self.root / "run"
        manifest = retarget_vertical_slice(
            self.selection,
            output=output,
            gmr_root=self.root,
            retarget_project_root=self.root,
            retarget_one=self._fake_retarget,
        )
        first = output / manifest["items"][0]["output"]
        first.write_bytes(b"tamper")
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "retarget output digest mismatch"):
            retarget_vertical_slice(
                self.selection,
                output=output,
                gmr_root=self.root,
                retarget_project_root=self.root,
                resume=True,
                retarget_one=self._fake_retarget,
            )
        self.assertEqual(self.calls, [])

    def test_resume_rejects_manifest_rewrite_that_blesses_changed_output(self) -> None:
        output = self.root / "run"
        manifest = retarget_vertical_slice(
            self.selection,
            output=output,
            gmr_root=self.root,
            retarget_project_root=self.root,
            retarget_one=self._fake_retarget,
        )
        first = output / manifest["items"][0]["output"]
        first.write_bytes(b"changed motion")
        manifest["items"][0]["output_sha256"] = _sha256(first)
        (output / "retarget-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "receipt output digest mismatch"):
            retarget_vertical_slice(
                self.selection,
                output=output,
                gmr_root=self.root,
                retarget_project_root=self.root,
                resume=True,
                retarget_one=self._fake_retarget,
            )
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
