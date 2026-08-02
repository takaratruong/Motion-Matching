import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from mm_sonic.joints import ContractError
from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    TerrainBinding,
)
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.source_justin import iter_justin_clips


def _terrain_binding() -> TerrainBinding:
    return TerrainBinding(
        asset_path="synthetic://justin-stairs",
        asset_size_bytes=1,
        asset_sha256="1" * 64,
        asset_license_id="UNRECORDED",
        mesh_sha256="2" * 64,
        world_from_terrain=RigidTransform(
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        validity_mask_path=None,
    )


def _write_justin_zarr(
    path: Path,
    *,
    include_order_names: bool = False,
    contradictory_joint_names: bool = False,
    cross_clip_antipodes: bool = False,
) -> Path:
    frames = 8
    timeline = np.arange(frames, dtype=np.float32) / np.float32(50.0)
    joint_position = np.broadcast_to(
        np.arange(29, dtype=np.float32), (frames, 29)
    ).copy()
    joint_velocity = np.zeros_like(joint_position)
    body_position = np.zeros((frames, 30, 3), dtype=np.float32)
    body_position[:, 0, 0] = 0.5 * timeline
    body_position[:, 18, 0] = 1.0 + timeline
    body_position[:, 19, 0] = 2.0 + timeline
    body_quaternion_xyzw = np.zeros((frames, 30, 4), dtype=np.float32)
    body_quaternion_xyzw[..., 3] = 1.0
    if cross_clip_antipodes:
        second_clip_yaw = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        body_quaternion_xyzw[4:, :, 2] = -np.sin(
            second_clip_yaw[:, None] / np.float32(2.0)
        )
        body_quaternion_xyzw[4:, :, 3] = -np.cos(
            second_clip_yaw[:, None] / np.float32(2.0)
        )
    body_linear_velocity = np.zeros((frames, 30, 3), dtype=np.float32)
    body_linear_velocity[:, 0, 0] = 0.5
    body_angular_velocity = np.zeros((frames, 30, 3), dtype=np.float32)

    root = zarr.open(str(path), mode="w")
    root.attrs["quaternion_convention"] = "xyzw"
    if include_order_names:
        joints = list(ISAACLAB_JOINT_NAMES)
        if contradictory_joint_names:
            joints[0], joints[1] = joints[1], joints[0]
        root.attrs["joint_names"] = joints
        root.attrs["body_names"] = list(ISAACLAB_BODY_NAMES)
    root.create_dataset("fps", data=np.array([50], dtype=np.int32))
    root.create_dataset(
        "clip_names",
        data=np.array(["up_zero", "down_zero"], dtype=object),
        object_codec=zarr.codecs.VLenUTF8(),
    )
    root.create_dataset(
        "clip_start_idx", data=np.array([0, 4], dtype=np.int64)
    )
    root.create_dataset(
        "clip_end_idx", data=np.array([4, 8], dtype=np.int64)
    )
    root.create_dataset("joint_pos", data=joint_position)
    root.create_dataset("joint_vel", data=joint_velocity)
    root.create_dataset("body_pos_w", data=body_position)
    root.create_dataset("body_quat_w", data=body_quaternion_xyzw)
    root.create_dataset("body_lin_vel_w", data=body_linear_velocity)
    root.create_dataset("body_ang_vel_w", data=body_angular_velocity)
    return path


class JustinSourceAdapterTests(unittest.TestCase):
    def test_legacy_adapter_reorders_explicit_xyzw_once_and_slices_exclusively(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _write_justin_zarr(Path(tmp) / "justin.zarr")

            clips = list(iter_justin_clips(source, _terrain_binding()))

            self.assertEqual([clip.clip_id for clip in clips], ["up_zero", "down_zero"])
            self.assertEqual([clip.frame_count for clip in clips], [4, 4])
            self.assertEqual(
                clips[0].source.source_format,
                "justin-zarr-legacy-isaaclab-v1",
            )
            self.assertEqual(clips[0].source.source_license_id, "UNRECORDED")
            self.assertEqual(clips[0].source.source_path, str(source.resolve()))
            expected_size = sum(
                file.stat().st_size for file in source.rglob("*") if file.is_file()
            )
            self.assertEqual(clips[0].source.source_size_bytes, expected_size)
            self.assertRegex(clips[0].source.source_sha256, r"^[0-9a-f]{64}$")
            np.testing.assert_allclose(
                clips[0].root_quaternion_world_wxyz[0],
                (1.0, 0.0, 0.0, 0.0),
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_array_equal(
                clips[1].root_position_world[:, 0],
                np.arange(4, 8, dtype=np.float32) / np.float32(100.0),
            )
            for clip in clips:
                self.assertIs(clip.terrain, clips[0].terrain)
                self.assertIn("contacts-unreconstructed", clip.action_tags)
                self.assertFalse(np.any(clip.contact_confidence))
                self.assertFalse(np.any(clip.commands.observed_mask))
                clip.validate()

    def test_explicit_order_names_must_equal_the_canonical_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            named = _write_justin_zarr(
                root / "named.zarr", include_order_names=True
            )
            named_clips = list(
                iter_justin_clips(named, _terrain_binding())
            )
            self.assertEqual(
                named_clips[0].source.source_format,
                "justin-zarr-isaaclab-v1",
            )
            self.assertEqual(named_clips[0].joint_names, ISAACLAB_JOINT_NAMES)
            self.assertEqual(named_clips[0].body_names, ISAACLAB_BODY_NAMES)

            contradictory = _write_justin_zarr(
                root / "contradictory.zarr",
                include_order_names=True,
                contradictory_joint_names=True,
            )
            with self.assertRaisesRegex(ContractError, "joint_names"):
                list(iter_justin_clips(contradictory, _terrain_binding()))

    def test_present_null_order_attrs_never_enter_the_legacy_fallback(self):
        cases = (
            ("joint-only-null", {"joint_names": None}),
            ("body-only-null", {"body_names": None}),
            (
                "both-null",
                {"joint_names": None, "body_names": None},
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case, attrs in cases:
                with self.subTest(case=case):
                    source = _write_justin_zarr(root / f"{case}.zarr")
                    archive = zarr.open(str(source), mode="a")
                    archive.attrs.update(attrs)

                    with self.assertRaisesRegex(
                        ContractError, "joint_names|body_names"
                    ):
                        list(iter_justin_clips(source, _terrain_binding()))

    def test_quaternion_and_inferred_state_are_independent_at_clip_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _write_justin_zarr(
                Path(tmp) / "antipodes.zarr",
                cross_clip_antipodes=True,
            )

            first, second = iter_justin_clips(source, _terrain_binding())

            np.testing.assert_allclose(
                first.root_quaternion_world_wxyz[-1],
                (1.0, 0.0, 0.0, 0.0),
                rtol=0.0,
                atol=1.0e-7,
            )
            np.testing.assert_allclose(
                second.root_quaternion_world_wxyz[0],
                (
                    -np.cos(0.15),
                    0.0,
                    0.0,
                    -np.sin(0.15),
                ),
                rtol=0.0,
                atol=1.0e-7,
            )
            np.testing.assert_allclose(
                first.commands.inferred_facing_local_xy,
                np.tile(
                    np.array([1.0, 0.0], dtype=np.float32),
                    (4, 1),
                ),
                rtol=0.0,
                atol=1.0e-7,
            )
            np.testing.assert_allclose(
                second.commands.inferred_facing_local_xy[0],
                (np.cos(0.3), np.sin(0.3)),
                rtol=0.0,
                atol=1.0e-6,
            )

    def test_legacy_schema_rejects_nonexclusive_or_noncontiguous_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _write_justin_zarr(Path(tmp) / "bad-ranges.zarr")
            root = zarr.open(str(source), mode="a")
            root["clip_start_idx"][:] = np.array([0, 5], dtype=np.int64)

            with self.assertRaisesRegex(ContractError, "exclusive"):
                list(iter_justin_clips(source, _terrain_binding()))

    def test_clip_names_reject_numeric_null_and_other_non_string_values(self):
        """Catches laundering malformed producer names through ``str(value)``."""

        cases = (
            (
                "numeric",
                np.array([101, 202], dtype=np.int64),
                None,
            ),
            (
                "null",
                np.array([None, "down_zero"], dtype=object),
                zarr.codecs.JSON(),
            ),
            (
                "mixed-object",
                np.array(["up_zero", {"name": "down_zero"}], dtype=object),
                zarr.codecs.JSON(),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root_path = Path(tmp)
            for case, values, object_codec in cases:
                with self.subTest(case=case):
                    source = _write_justin_zarr(
                        root_path / f"{case}.zarr"
                    )
                    archive = zarr.open(str(source), mode="a")
                    del archive["clip_names"]
                    kwargs = {}
                    if object_codec is not None:
                        kwargs["object_codec"] = object_codec
                    archive.create_dataset(
                        "clip_names",
                        data=values,
                        **kwargs,
                    )

                    with self.assertRaisesRegex(
                        ContractError, "clip_names.*strings"
                    ):
                        list(iter_justin_clips(source, _terrain_binding()))


if __name__ == "__main__":
    unittest.main()
