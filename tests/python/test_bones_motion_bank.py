import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.export_bones_motion_bank import (
    BONES_BODY_NAMES,
    DEFAULT_BONES_BASE_CLIPS,
    export_motion_bank,
    mirror_clip_name,
    resolve_clip_selection,
)
from mm_sonic.joints import ContractError
from mm_sonic.offline_corpus import build_corpus
from mm_sonic.torch_motion_data import MotionFolder
from tests.python.torch_motion_test_utils import (
    build_takara_arrays,
    write_takara_arrays,
)


def _store(*, include_mirror: bool = True, fps: int = 50) -> dict[str, object]:
    names = ["walk_forward_loop_002__A023"]
    if include_mirror:
        names.append("walk_forward_loop_002__A023_M")
    clips = [build_takara_arrays(frames=60, fps=fps) for _ in names]
    result: dict[str, object] = {
        "fps": np.asarray([fps], dtype=np.int32),
        "clip_names": np.asarray(names, dtype=str),
        "clip_start_idx": np.arange(len(names), dtype=np.int64) * 60,
        "clip_end_idx": (np.arange(len(names), dtype=np.int64) + 1) * 60,
        "body_names": np.asarray(BONES_BODY_NAMES, dtype=str),
    }
    for field in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        result[field] = np.concatenate([clip[field] for clip in clips], axis=0)
    return result


class BonesMotionBankTests(unittest.TestCase):
    def test_default_bank_has_dense_balanced_start_stop_support(self):
        transitions = tuple(
            name
            for name in DEFAULT_BONES_BASE_CLIPS
            if "start" in name.lower() or "stop" in name.lower()
        )
        self.assertEqual(len(DEFAULT_BONES_BASE_CLIPS), 86)
        self.assertEqual(len(set(DEFAULT_BONES_BASE_CLIPS)), 86)
        self.assertEqual(len(transitions), 77)

    def test_exact_base_mirror_selection(self):
        available = (
            "walk_forward_loop_002__A023",
            "walk_forward_loop_002__A023_M",
        )
        self.assertEqual(mirror_clip_name(available[0]), available[1])
        self.assertEqual(
            resolve_clip_selection(available, available[:1]),
            available,
        )

    def test_export_is_native_motion_folder_and_keeps_exact_rows(self):
        source = _store()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "bones_bank"
            manifest = export_motion_bank(
                source,
                destination,
                base_names=("walk_forward_loop_002__A023",),
                source_label="synthetic-bones",
            )
            self.assertEqual(manifest["clip_count"], 2)
            folder = MotionFolder.load(destination)
            self.assertEqual(len(folder.clips), 2)
            np.testing.assert_array_equal(
                folder.clips[0].joint_position,
                np.asarray(source["joint_pos"])[:60],
            )
            np.testing.assert_array_equal(
                folder.clips[1].joint_position,
                np.asarray(source["joint_pos"])[60:120],
            )
            self.assertTrue((destination / "manifest.json").is_file())

    def test_export_can_augment_bones_with_legacy_xyzw_takara(self):
        source = _store()
        takara = build_takara_arrays(frames=80)
        expected_wxyz = takara["body_quat_w"].copy()
        takara["body_quat_w"] = np.ascontiguousarray(
            expected_wxyz[..., (1, 2, 3, 0)]
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "combined_bank"
            manifest = export_motion_bank(
                source,
                destination,
                base_names=("walk_forward_loop_002__A023",),
                source_label="synthetic-bones",
                legacy_takara=takara,
                legacy_takara_label="synthetic-takara",
            )
            self.assertEqual(manifest["bones_clip_count"], 2)
            self.assertEqual(manifest["clip_count"], 3)
            folder = MotionFolder.load(destination)
            takara_clip = next(
                clip
                for clip in folder.clips
                if clip.relative_path == "takara_walk/motion.npz"
            )
            np.testing.assert_allclose(
                takara_clip.body_quaternion_world_wxyz,
                expected_wxyz,
                atol=1e-7,
                rtol=1e-7,
            )

    def test_missing_mirror_fails_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "bones_bank"
            with self.assertRaisesRegex(ContractError, "mirror is missing"):
                export_motion_bank(
                    _store(include_mirror=False),
                    destination,
                    base_names=("walk_forward_loop_002__A023",),
                    source_label="synthetic-bones",
                )
            self.assertFalse(destination.exists())

    def test_wrong_rate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ContractError, "fps must equal 50"):
                export_motion_bank(
                    _store(fps=33),
                    Path(tmp) / "bones_bank",
                    base_names=("walk_forward_loop_002__A023",),
                    source_label="synthetic-bones",
                )

    def test_offline_corpus_accepts_exported_native_motion_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "motions"
            write_takara_arrays(
                source / "forward",
                build_takara_arrays(frames=150, yaw_rate=0.3),
            )
            write_takara_arrays(
                source / "turn",
                build_takara_arrays(
                    frames=150,
                    yaw_rate=-0.2,
                    root_velocity_xy=(0.2, 0.25),
                ),
            )
            output = root / "corpus.zarr"
            build_corpus(
                source_motion_dir=source,
                output=output,
                device="cpu",
                frames=100,
                base_start=0,
                base_stop=None,
                base_indices=(0,),
                source_quaternion_convention="wxyz",
            )
            self.assertTrue((output / ".zgroup").is_file())
            self.assertTrue((output / "meta" / "episode_clip").is_dir())


if __name__ == "__main__":
    unittest.main()
