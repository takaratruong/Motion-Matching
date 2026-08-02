import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)
from mm_sonic.terrain_oracle.source_lafan import (
    LAFAN1_UPSTREAM_REVISION,
    classify_lafan_name,
    load_lafan_csv,
)


MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)


def _write_lafan_csv(path: Path, *, frames: int = 4) -> Path:
    timeline = np.arange(frames, dtype=np.float64)
    rows = np.zeros((frames, 36), dtype=np.float64)
    rows[:, 0] = timeline * 0.03
    rows[:, 1] = timeline * -0.01
    rows[:, 2] = 0.8
    yaw = timeline * 0.2
    rows[:, 5] = np.sin(yaw / 2.0)
    rows[:, 6] = np.cos(yaw / 2.0)
    rows[:, 7:] = timeline[:, None] + np.arange(29)[None, :] * 0.01
    np.savetxt(path, rows, delimiter=",", fmt="%.9f")
    return path


@unittest.skipUnless(MODEL.is_file(), "real G1 MuJoCo model is unavailable")
class LafanSourceAdapterTests(unittest.TestCase):
    def test_30_hz_csv_resamples_to_50_hz_without_duplicate_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_lafan_csv(
                Path(temporary) / "walk4_subject1.csv", frames=4
            )
            source_bytes = path.read_bytes()

            clip = load_lafan_csv(path, MODEL, terrain=None)

            self.assertEqual(clip.frame_count, 6)
            self.assertEqual(clip.fps, 50.0)
            np.testing.assert_allclose(
                clip.root_position_world[:, 0],
                (0.0, 0.018, 0.036, 0.054, 0.072, 0.09),
                rtol=0.0,
                atol=1.0e-7,
            )
            self.assertEqual(clip.joint_names, ISAACLAB_JOINT_NAMES)
            self.assertEqual(clip.body_names, ISAACLAB_BODY_NAMES)
            self.assertEqual(
                clip.source.source_format,
                "lafan1-retargeted-g1-csv-30hz@"
                + LAFAN1_UPSTREAM_REVISION,
            )
            self.assertEqual(
                clip.source.source_license_id, "CC-BY-NC-ND-4.0"
            )
            self.assertEqual(clip.source.source_path, str(path.resolve()))
            self.assertEqual(clip.source.source_size_bytes, len(source_bytes))
            self.assertEqual(
                clip.source.source_sha256,
                hashlib.sha256(source_bytes).hexdigest(),
            )
            self.assertEqual(clip.source.quaternion_convention, "wxyz")
            self.assertEqual(clip.source.pose_origin, "clean-motion-corpus")
            clip.validate()

    def test_crlf_rows_and_one_final_line_ending_remain_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_lafan_csv(
                Path(temporary) / "walk4_subject1.csv", frames=4
            )
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

            clip = load_lafan_csv(path, MODEL, terrain=None)

            self.assertEqual(clip.frame_count, 6)

    def test_shortest_path_slerp_and_exact_mujoco_to_isaaclab_joint_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_lafan_csv(
                Path(temporary) / "walk1_subject1.csv", frames=4
            )
            rows = np.loadtxt(path, delimiter=",")
            rows[1, 3:7] *= -1.0
            np.savetxt(path, rows, delimiter=",", fmt="%.9f")

            clip = load_lafan_csv(path, MODEL, terrain=None)

            yaw = np.unwrap(
                2.0
                * np.arctan2(
                    clip.root_quaternion_world_wxyz[:, 3],
                    clip.root_quaternion_world_wxyz[:, 0],
                )
            )
            np.testing.assert_allclose(
                yaw, np.arange(6) * 0.12, rtol=0.0, atol=1.0e-6
            )
            np.testing.assert_allclose(
                clip.joint_position[0],
                np.arange(29, dtype=np.float32)[
                    [
                        0,
                        6,
                        12,
                        1,
                        7,
                        13,
                        2,
                        8,
                        14,
                        3,
                        9,
                        15,
                        22,
                        4,
                        10,
                        16,
                        23,
                        5,
                        11,
                        17,
                        24,
                        18,
                        25,
                        19,
                        26,
                        20,
                        27,
                        21,
                        28,
                    ]
                ]
                * 0.01,
                rtol=0.0,
                atol=1.0e-7,
            )
            self.assertFalse(clip.joint_position.flags.writeable)

    def test_contacts_are_unreconstructed_and_commands_are_only_inferred(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_lafan_csv(
                Path(temporary) / "run2_subject4.csv", frames=4
            )

            clip = load_lafan_csv(path, MODEL, terrain=None)

            self.assertIsNone(clip.terrain)
            self.assertEqual(
                clip.action_tags,
                ("lafan", "run", "contacts-unreconstructed"),
            )
            self.assertFalse(np.any(clip.contact))
            self.assertFalse(np.any(clip.contact_confidence))
            self.assertFalse(np.any(clip.commands.observed_mask))
            self.assertGreater(
                float(np.max(np.abs(clip.commands.inferred_velocity_local_xy))),
                0.0,
            )

    def test_obstacle_name_does_not_invent_a_terrain_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_lafan_csv(
                Path(temporary) / "obstacle1_subject1.csv", frames=4
            )

            clip = load_lafan_csv(path, MODEL, terrain=None)

            self.assertEqual(
                classify_lafan_name(path.stem),
                ("lafan", "obstacle"),
            )
            self.assertIn("obstacle", clip.action_tags)
            self.assertIsNone(clip.terrain)

    def test_rejects_headers_wrong_width_and_nonfinite_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = {
                "header.csv": "x,y,z\n1,2,3\n",
                "width.csv": ",".join("0" for _ in range(35)) + "\n",
                "nonfinite.csv": ",".join(
                    ["nan", *("0" for _ in range(35))]
                )
                + "\n",
            }
            for name, text in cases.items():
                with self.subTest(name=name):
                    path = root / name
                    path.write_text(text)
                    with self.assertRaisesRegex(
                        ValueError, "36|finite|numeric|frames"
                    ):
                        load_lafan_csv(path, MODEL, terrain=None)

    def test_rejects_comment_and_blank_physical_rows(self):
        """Catches NumPy silently skipping lines absent from the CSV contract."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = _write_lafan_csv(root / "valid.csv").read_text().splitlines()
            cases = {
                "comment-first.csv": ["# undocumented header", *valid],
                "comment-middle.csv": [valid[0], "# hidden row", *valid[1:]],
                "blank-middle.csv": [valid[0], "", *valid[1:]],
            }
            for name, lines in cases.items():
                with self.subTest(name=name):
                    path = root / name
                    path.write_text("\n".join(lines) + "\n")
                    with self.assertRaisesRegex(
                        ValueError, "numeric row|blank|comment"
                    ):
                        load_lafan_csv(path, MODEL, terrain=None)


if __name__ == "__main__":
    unittest.main()
