import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)
from mm_sonic.terrain_oracle.source_flat import iter_flat_clips
from tests.python.torch_motion_test_utils import write_takara_clip


class FlatSourceAdapterTests(unittest.TestCase):
    def test_snapshot_bytes_drive_data_and_hash_while_authority_path_is_retained(self):
        """Catches reparsing mutable authority bytes after a private snapshot."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authority_root = root / "authority"
            snapshot_root = root / "snapshot"
            authority_source = write_takara_clip(
                authority_root / "walk",
                frames=60,
                root_velocity_xy=(0.1, 0.0),
            )
            authority_alias = root / "authority-alias"
            authority_alias.symlink_to(authority_root, target_is_directory=True)
            snapshot_source = write_takara_clip(
                snapshot_root / "walk",
                frames=60,
                root_velocity_xy=(0.7, 0.0),
            )

            try:
                clip = next(
                    iter_flat_clips(
                        snapshot_root,
                        tags=("flat", "walk"),
                        authority_root=authority_alias,
                    )
                )
            except TypeError as error:
                self.fail(f"snapshot authority API is missing: {error}")

            self.assertEqual(
                clip.source.source_path,
                str(
                    authority_alias
                    / authority_source.relative_to(authority_root)
                ),
            )
            self.assertEqual(
                clip.source.source_sha256,
                hashlib.sha256(snapshot_source.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                clip.source.source_size_bytes, snapshot_source.stat().st_size
            )
            np.testing.assert_allclose(
                clip.root_linear_velocity_world[:, 0],
                np.full(60, 0.7, dtype=np.float32),
                rtol=0.0,
                atol=1.0e-6,
            )

    def test_preserves_native_wxyz_50_hz_and_exact_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_takara_clip(root / "walk", frames=60)
            with np.load(source, allow_pickle=False) as archive:
                expected_joint_position = archive["joint_pos"].copy()
                expected_body_quaternion = archive["body_quat_w"].copy()

            clip = next(iter_flat_clips(root, tags=("flat", "walk")))

            self.assertEqual(clip.clip_id, "walk")
            self.assertEqual(clip.fps, 50.0)
            self.assertEqual(clip.joint_names, ISAACLAB_JOINT_NAMES)
            self.assertEqual(clip.body_names, ISAACLAB_BODY_NAMES)
            self.assertEqual(clip.source.source_format, "takara-motion-npz-v1")
            self.assertEqual(clip.source.source_path, str(source.resolve()))
            self.assertEqual(clip.source.source_size_bytes, source.stat().st_size)
            self.assertEqual(
                clip.source.source_sha256,
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(clip.source.source_license_id, "UNRECORDED")
            self.assertEqual(
                clip.source.coordinate_convention, "z-up-right-handed"
            )
            self.assertEqual(clip.source.quaternion_convention, "wxyz")
            self.assertEqual(clip.source.pose_origin, "clean-motion-corpus")
            np.testing.assert_array_equal(
                clip.joint_position, expected_joint_position
            )
            np.testing.assert_allclose(
                clip.body_quaternion_world_wxyz,
                expected_body_quaternion,
                rtol=0.0,
                atol=1.0e-7,
            )
            clip.validate()

    def test_marks_unreconstructed_contacts_and_keeps_commands_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_clip(
                root / "turn",
                frames=60,
                yaw_rate=0.25,
                root_velocity_xy=(0.4, 0.0),
            )

            clip = next(iter_flat_clips(root, tags=("flat", "turn")))

            self.assertEqual(
                clip.action_tags, ("flat", "turn", "contacts-unreconstructed")
            )
            np.testing.assert_array_equal(clip.contact, np.zeros((60, 2)))
            np.testing.assert_array_equal(
                clip.contact_confidence, np.zeros((60, 2))
            )
            np.testing.assert_array_equal(
                clip.sole_position_world,
                clip.body_position_world[:, (18, 19)],
            )
            np.testing.assert_array_equal(
                clip.heel_position_world, clip.sole_position_world
            )
            np.testing.assert_array_equal(
                clip.toe_position_world, clip.sole_position_world
            )
            np.testing.assert_array_equal(
                clip.sole_quaternion_world_wxyz,
                clip.body_quaternion_world_wxyz[:, (18, 19)],
            )
            self.assertFalse(np.any(clip.commands.observed_mask))
            np.testing.assert_array_equal(
                clip.commands.observed_travel_stick_xy,
                np.zeros((60, 2)),
            )
            np.testing.assert_allclose(
                clip.commands.inferred_velocity_local_xy[0],
                (0.4, 0.0),
                rtol=0.0,
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                clip.commands.inferred_yaw_rate_rad_s,
                np.full(60, 0.25),
                rtol=0.0,
                atol=1.0e-6,
            )
            self.assertFalse(clip.joint_position.flags.writeable)
            self.assertFalse(
                clip.commands.inferred_velocity_local_xy.flags.writeable
            )


if __name__ == "__main__":
    unittest.main()
