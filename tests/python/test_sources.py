import copy
import glob
import hashlib
import json
import os
import tempfile
import unittest
import numpy as np

from resources.g1_terrain_builder.sources import (
    load_grail,
    load_retarget_npz,
    load_takara,
)

TAKARA = "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz"
REMAP = "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy"
GRAIL_GLOB = "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl"
RETARGET_ROOT = "sonic/runs/native-g1-pfnn/sample-retarget"
CURRENT_RETARGET = os.path.join(
    RETARGET_ROOT, "LocomotionFlat01_000-walk-only-7659-8171-120hz.npz")
CURRENT_RECEIPT = os.path.join(
    RETARGET_ROOT,
    "LocomotionFlat01_000-walk-only-7659-8171-120hz.receipt.json")
STALE_RETARGET = os.path.join(
    RETARGET_ROOT, "LocomotionFlat01_000-120hz.npz")
STALE_RECEIPT = os.path.join(
    RETARGET_ROOT, "LocomotionFlat01_000-120hz.receipt.json")


class SourceTests(unittest.TestCase):
    def test_current_walk_only_retarget_authenticates_exact_receipt(self):
        clip = load_retarget_npz(CURRENT_RETARGET, CURRENT_RECEIPT)

        self.assertEqual(clip.qpos.shape, (512, 36))
        self.assertEqual(clip.fps, 120.0)
        self.assertEqual(clip.terrain_id, "flat")
        np.testing.assert_array_equal(
            clip.source_frames, np.arange(7659, 8171, dtype=np.int32))
        self.assertEqual(
            clip.provenance["sha256"],
            "bbdeb79760950480582ae937e54b913c376caa49f344896a8958476b82f3317f",
        )
        self.assertEqual(len(clip.provenance["receipt"]), 18)

    def test_retarget_start_frame_is_preserved_in_source_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            motion = os.path.join(temporary, "interval.npz")
            receipt_path = os.path.join(temporary, "interval.receipt.json")
            with np.load(CURRENT_RETARGET, allow_pickle=False) as source:
                np.savez_compressed(
                    motion,
                    **{
                        key: source[key][:12]
                        if key in {"root_pos", "root_quat", "dof"}
                        else source[key]
                        for key in source.files
                    },
                )
            with open(CURRENT_RECEIPT, encoding="utf-8") as stream:
                receipt = json.load(stream)
            with open(motion, "rb") as stream:
                output_sha256 = hashlib.sha256(stream.read()).hexdigest()
            receipt.update({
                "output_sha256": output_sha256,
                "source_frame_count": 2000,
                "start_frame": 240,
                "frame_count": 12,
                "warmup_frames": 120,
            })
            with open(receipt_path, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream)

            clip = load_retarget_npz(motion, receipt_path)

        np.testing.assert_array_equal(
            clip.source_frames, np.arange(240, 252, dtype=np.int32))

    def test_stale_full_retarget_receipt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "receipt fields"):
            load_retarget_npz(STALE_RETARGET, STALE_RECEIPT)

    def test_retarget_receipt_rejects_missing_or_tampered_contract_fields(self):
        with open(CURRENT_RECEIPT, encoding="utf-8") as stream:
            original = json.load(stream)
        cases = (
            ("missing-scale", lambda value: value.pop("pfnn_position_scale")),
            ("scale", lambda value: value.__setitem__(
                "pfnn_position_scale", 1.0)),
            ("grounding", lambda value: value.__setitem__(
                "grounding", "source")),
            ("grounding-offset", lambda value: value.__setitem__(
                "grounding_offset_m", float("nan"))),
            ("start-frame", lambda value: value.__setitem__(
                "start_frame", -1)),
            ("source-interval", lambda value: value.__setitem__(
                "source_frame_count", 8170)),
            ("aliases", lambda value: value.__setitem__(
                "aliases", [["Spine1", "Spine2"],
                            ["LeftToeBase", "LeftToe"]])),
            ("gmr-commit", lambda value: value.__setitem__(
                "gmr_commit", "0" * 40)),
            ("retarget-commit", lambda value: value.__setitem__(
                "retarget_project_commit", "0" * 40)),
            ("source-hash", lambda value: value.__setitem__(
                "source_sha256", "not-a-sha")),
            ("prepared-hash", lambda value: value.__setitem__(
                "prepared_sha256", "0" * 63)),
            ("output-hash", lambda value: value.__setitem__(
                "output_sha256", "0" * 64)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for name, mutation in cases:
                receipt = copy.deepcopy(original)
                mutation(receipt)
                path = os.path.join(temporary, f"{name}.json")
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump(receipt, stream)
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "retarget receipt|SHA-256"),
                ):
                    load_retarget_npz(CURRENT_RETARGET, path)

    def test_takara_is_native_g1_qpos(self):
        clip = load_takara(TAKARA, REMAP)
        self.assertEqual(clip.qpos.shape[1], 36)
        self.assertEqual(clip.fps, 50.0)
        np.testing.assert_allclose(
            np.linalg.norm(clip.qpos[:, 3:7], axis=1), 1.0, atol=1e-4,
        )

    def test_grail_is_native_g1_qpos(self):
        path = sorted(glob.glob(GRAIL_GLOB))[0]
        clip = load_grail(path)
        self.assertEqual(clip.qpos.shape, (250, 36))
        self.assertEqual(clip.fps, 25.0)
        self.assertEqual(clip.source_frames[-1], 249)
        np.testing.assert_allclose(
            np.linalg.norm(clip.qpos[:, 3:7], axis=1), 1.0, atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
