from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import subprocess
import sys
import unittest

import numpy as np

from mm_sonic.terrain_pfnn import features as pfnn_features
from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)
from mm_sonic.terrain_pfnn.features import (
    PFNNTrainingWindow,
    build_clip_windows,
    build_clip_windows_with_audit,
    mirror_window,
)
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from mm_sonic.terrain_pfnn.phase import ContactPhaseTrack
from mm_sonic.terrain_pfnn.sources import PFNNSourceClip
from mm_sonic.terrain_pfnn.splits import (
    LAFAN_BASE_STEMS_BY_DIGEST,
    split_identity,
    terrain_identity,
)


JOINT_MIRROR_PERMUTATION = np.asarray(
    (1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 12, 11, 14, 13,
     16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27),
    dtype=np.int64,
)
BODY_MIRROR_PERMUTATION = np.asarray(
    (0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 13, 12, 15,
     14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28),
    dtype=np.int64,
)
JOINT_MIRROR_SIGNS = np.asarray(
    (1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1,
     -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1),
    dtype=np.float32,
)
_LEFT_HIP_BODY = ISAACLAB_BODY_NAMES.index("left_hip_pitch_link")
_RIGHT_HIP_BODY = ISAACLAB_BODY_NAMES.index("right_hip_pitch_link")
_LEFT_SHOULDER_BODY = ISAACLAB_BODY_NAMES.index("left_shoulder_pitch_link")
_RIGHT_SHOULDER_BODY = ISAACLAB_BODY_NAMES.index("right_shoulder_pitch_link")


def _body_offsets_local() -> np.ndarray:
    offsets = np.zeros((30, 3), dtype=np.float32)
    offsets[:, 0] = np.arange(30, dtype=np.float32) * 0.01
    offsets[:, 1] = np.arange(30, dtype=np.float32) * -0.02
    offsets[:, 2] = np.arange(30, dtype=np.float32) * 0.005
    for left, right, half_width in (
        (_LEFT_HIP_BODY, _RIGHT_HIP_BODY, 0.12),
        (_LEFT_SHOULDER_BODY, _RIGHT_SHOULDER_BODY, 0.20),
    ):
        pair_center = 0.5 * (offsets[left] + offsets[right])
        offsets[left] = pair_center + np.array((0.0, half_width, 0.0))
        offsets[right] = pair_center - np.array((0.0, half_width, 0.0))
    return offsets


def _with_body_facing(clip: PFNNSourceClip, yaw: np.ndarray) -> PFNNSourceClip:
    yaw_trace = np.asarray(yaw, dtype=np.float64)
    if yaw_trace.shape != (clip.frame_count,):
        raise ValueError("yaw must have one value per frame")
    across = np.column_stack((-np.sin(yaw_trace), np.cos(yaw_trace)))
    body = np.array(clip.body_position_world, copy=True)
    root = np.asarray(clip.root_position_world)
    for left, right, half_width, height in (
        (_LEFT_HIP_BODY, _RIGHT_HIP_BODY, 0.12, -0.05),
        (_LEFT_SHOULDER_BODY, _RIGHT_SHOULDER_BODY, 0.20, 0.45),
    ):
        body[:, left, :2] = root[:, :2] + half_width * across
        body[:, right, :2] = root[:, :2] - half_width * across
        body[:, left, 2] = root[:, 2] + height
        body[:, right, 2] = root[:, 2] + height
    return replace(clip, body_position_world=body)


def _holden_reference_facing_yaw(clip: PFNNSourceClip) -> np.ndarray:
    """Literal released PFNN facing recipe, converted from y-up to G1 z-up."""

    g1 = np.asarray(clip.body_position_world, dtype=np.float64)
    source_y_up = np.stack((g1[..., 0], g1[..., 2], -g1[..., 1]), axis=-1)
    across = (
        source_y_up[:, _LEFT_SHOULDER_BODY]
        - source_y_up[:, _RIGHT_SHOULDER_BODY]
        + source_y_up[:, _LEFT_HIP_BODY]
        - source_y_up[:, _RIGHT_HIP_BODY]
    )
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    forward = np.cross(across, np.array((0.0, 1.0, 0.0)))

    # Holden uses sigma=20 at 120 Hz. PFNNSourceClip is sealed at 30 Hz, so
    # sigma=5 preserves the same temporal width. scipy's default radius is
    # round(4*sigma), and mode="nearest" clamps samples at both clip edges.
    sigma = 20.0 * clip.fps / 120.0
    radius = int(4.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= np.sum(weights)
    smoothed = np.empty_like(forward)
    for frame in range(clip.frame_count):
        indices = np.clip(frame + offsets, 0, clip.frame_count - 1)
        smoothed[frame] = weights @ forward[indices]
    smoothed /= np.linalg.norm(smoothed, axis=1, keepdims=True)
    forward_g1 = np.column_stack((smoothed[:, 0], -smoothed[:, 2]))
    return np.arctan2(forward_g1[:, 1], forward_g1[:, 0])


def _quaternion_yaw_roll(yaw: float, roll: float = 0.0) -> np.ndarray:
    yaw_quaternion = np.array(
        (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
        dtype=np.float64,
    )
    roll_quaternion = np.array(
        (math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0),
        dtype=np.float64,
    )
    w0, x0, y0, z0 = yaw_quaternion
    w1, x1, y1, z1 = roll_quaternion
    return np.array(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dtype=np.float32,
    )


class _Plane:
    def __init__(self, x_grade: float = 0.0, y_grade: float = 0.0) -> None:
        self.x_grade = float(x_grade)
        self.y_grade = float(y_grade)

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy)
        return self.x_grade * points[..., 0] + self.y_grade * points[..., 1]


class _RunupRamp:
    def __init__(self, transition_x: float = 0.8) -> None:
        self.transition_x = float(transition_x)

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        x = np.asarray(xy)[..., 0]
        return 0.1 * np.maximum(x - self.transition_x, 0.0)


def _fixture(
    *,
    frames: int = 90,
    yaw: float = 0.0,
    roll: float = 0.2,
    step_m: float = 0.02,
    slope: _Plane | None = None,
    clip_id: str = "terrain_slopes__slope_113__000",
    terrain_id: str = "slope_113",
    phase_valid: bool = True,
) -> tuple[PFNNSourceClip, ContactPhaseTrack, _Plane]:
    surface = slope or _Plane(0.1, 0.2)
    timeline = np.arange(frames, dtype=np.float32)
    forward = np.array((math.cos(yaw), math.sin(yaw)), dtype=np.float32)
    root_position = np.zeros((frames, 3), dtype=np.float32)
    root_position[:, :2] = timeline[:, None] * step_m * forward[None, :]
    root_position[:, 2] = surface.height_at(root_position[:, :2]) + 1.0
    quaternion = np.tile(_quaternion_yaw_roll(yaw, roll), (frames, 1))

    body_offsets_local = _body_offsets_local()
    rotation = np.array(
        ((math.cos(yaw), -math.sin(yaw)),
         (math.sin(yaw), math.cos(yaw))),
        dtype=np.float32,
    )
    body_position = np.empty((frames, 30, 3), dtype=np.float32)
    body_position[:, :, :2] = (
        root_position[:, None, :2]
        + body_offsets_local[None, :, :2] @ rotation.T
    )
    body_position[:, :, 2] = (
        surface.height_at(root_position[:, :2])[:, None]
        + 1.0
        + body_offsets_local[None, :, 2]
    )
    body_velocity = np.zeros((frames, 30, 3), dtype=np.float32)
    body_velocity[:, :, :2] = (step_m * 30.0 * forward)[None, None, :]
    body_velocity[:, :, 2] = (
        np.gradient(surface.height_at(root_position[:, :2])) * 30.0
    )[:, None]
    body_quaternion = np.tile(quaternion[:, None, :], (1, 30, 1))
    joint_position = timeline[:, None] + np.arange(29, dtype=np.float32)[None, :] / 100.0

    source = PFNNSourceClip(
        clip_id=clip_id,
        terrain_id=terrain_id,
        fps=30.0,
        root_position_world=root_position,
        root_quaternion_world_wxyz=quaternion,
        joint_position=joint_position,
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_quaternion,
        body_linear_velocity_world=body_velocity,
        body_angular_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
        # Deliberately wrong: root-motion targets must use t -> t+1 displacement.
        root_linear_velocity_world=np.full((frames, 3), 99.0, dtype=np.float32),
        root_angular_velocity_world=np.full((frames, 3), 99.0, dtype=np.float32),
        joint_velocity=np.zeros((frames, 29), dtype=np.float32),
        terrain_path=None,
        terrain_position_world=np.zeros(3, dtype=np.float32),
        terrain_quaternion_world_from_usd_wxyz=np.array(
            (1.0, 0.0, 0.0, 0.0), dtype=np.float32
        ),
        motion_sha256="a" * 64,
        terrain_sha256=None,
        source_license_id="test",
    )
    contact = (
        (np.arange(frames)[:, None] + np.arange(4)[None, :]) % 2 == 0
    )
    phase = np.remainder(np.arange(frames) * 0.17, 2.0 * math.pi).astype(np.float32)
    phase_advance = (0.01 + np.arange(frames) * 0.001).astype(np.float32)
    track = ContactPhaseTrack(
        contact=contact,
        confidence=np.ones((frames, 4), dtype=np.float32),
        phase=phase,
        phase_advance=phase_advance,
        valid=np.full(frames, phase_valid, dtype=bool),
    )
    return source, track, surface


class SplitIdentityTest(unittest.TestCase):
    def test_all_motion_variants_share_the_terrain_identity_and_split(self) -> None:
        names = [f"terrain_slopes__slope_113__{index:03d}" for index in range(10)]
        self.assertEqual({terrain_identity(name) for name in names}, {"slope_113"})
        splits = {split_identity(name) for name in names}
        self.assertEqual(len(splits), 1)
        self.assertTrue(splits <= {"train", "validation", "test"})

    def test_lafan_assignment_is_an_explicit_digest_sorted_eight_two_two(self) -> None:
        train = {
            "walk1_subject1", "walk4_subject1", "walk1_subject2", "walk3_subject5",
            "walk2_subject4", "walk3_subject2", "walk3_subject3", "walk3_subject1",
        }
        validation = {"walk2_subject3", "walk3_subject4"}
        test = {"walk2_subject1", "walk1_subject5"}
        self.assertEqual(len(LAFAN_BASE_STEMS_BY_DIGEST), 12)
        self.assertEqual(len(set(LAFAN_BASE_STEMS_BY_DIGEST)), 12)
        self.assertEqual(
            LAFAN_BASE_STEMS_BY_DIGEST,
            tuple(sorted(
                LAFAN_BASE_STEMS_BY_DIGEST,
                key=lambda stem: hashlib.sha256(stem.encode("utf-8")).digest(),
            )),
        )
        for expected, stems in (
            ("train", train), ("validation", validation), ("test", test)
        ):
            for stem in stems:
                with self.subTest(stem=stem):
                    self.assertEqual(split_identity(stem), expected)
                    self.assertEqual(split_identity(f"{stem}.csv"), expected)
                    self.assertEqual(split_identity(f"{stem}__mirror"), expected)


class TrainingWindowTest(unittest.TestCase):
    def test_features_module_stays_importable_without_optional_scipy(self) -> None:
        program = """
import builtins
import mm_sonic.terrain_pfnn.layout

original_import = builtins.__import__

def import_without_scipy(name, *args, **kwargs):
    if name == "scipy" or name.startswith("scipy."):
        raise ModuleNotFoundError("scipy is intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_scipy
import mm_sonic.terrain_pfnn.features
"""
        result = subprocess.run(
            (sys.executable, "-B", "-c", program),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_smoothed_body_facing_matches_holden_after_z_up_conversion(self) -> None:
        flat = _Plane()
        clip, _, _ = _fixture(
            slope=flat,
            clip_id="walk1_subject1",
            terrain_id="flat",
        )
        frame = np.arange(clip.frame_count, dtype=np.float64)
        authored_yaw = 0.75 * np.tanh((frame - 45.0) / 3.0)
        facing_clip = _with_body_facing(clip, authored_yaw)
        facing_function = getattr(
            pfnn_features, "smoothed_body_facing_yaw_world", None
        )
        self.assertTrue(
            callable(facing_function),
            "features must expose the Holden-parity smoothed body-facing yaw",
        )

        actual = facing_function(facing_clip)
        expected = _holden_reference_facing_yaw(facing_clip)
        np.testing.assert_allclose(actual, expected, atol=1.0e-10, rtol=0.0)
        self.assertGreater(float(np.max(np.abs(actual - authored_yaw))), 0.1)

        mirrored_body = np.array(facing_clip.body_position_world, copy=True)
        mirrored_body = mirrored_body[:, BODY_MIRROR_PERMUTATION]
        mirrored_body[..., 1] *= -1.0
        mirrored = replace(facing_clip, body_position_world=mirrored_body)
        mirrored_yaw = facing_function(mirrored)
        np.testing.assert_allclose(np.cos(mirrored_yaw), np.cos(actual), atol=1e-10)
        np.testing.assert_allclose(np.sin(mirrored_yaw), -np.sin(actual), atol=1e-10)

    def test_body_facing_not_pelvis_twist_defines_the_feature_frame(self) -> None:
        flat = _Plane()
        clip, track, _ = _fixture(
            slope=flat,
            clip_id="walk1_subject1",
            terrain_id="flat",
        )
        body_facing = _with_body_facing(
            clip, np.zeros(clip.frame_count, dtype=np.float64)
        )
        pelvis_yaw = 0.4 + np.arange(clip.frame_count, dtype=np.float64) * 0.01
        pelvis_quaternion = np.stack(
            (
                np.cos(pelvis_yaw / 2.0),
                np.zeros_like(pelvis_yaw),
                np.zeros_like(pelvis_yaw),
                np.sin(pelvis_yaw / 2.0),
            ),
            axis=1,
        ).astype(np.float32)
        twisted = replace(
            body_facing, root_quaternion_world_wxyz=pelvis_quaternion
        )
        sample = next(
            window
            for window in build_clip_windows(
                twisted, track, height_at=flat.height_at
            )
            if window.center_frame == 40
        )

        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["root_planar_velocity"]],
            (0.6, 0.0),
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["root_yaw_velocity"]], (0.0,), atol=1.0e-6
        )
        np.testing.assert_allclose(
            sample.x[INPUT_LAYOUT["trajectory_position"]].reshape(12, 2)[:, 1],
            0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            sample.x[INPUT_LAYOUT["trajectory_direction"]].reshape(12, 2),
            np.tile((1.0, 0.0), (12, 1)),
            atol=1.0e-6,
        )

    def test_exact_shapes_terrain_order_provenance_and_frame_bounds(self) -> None:
        clip, track, slope = _fixture()
        windows = build_clip_windows(clip, track, height_at=slope.height_at)

        self.assertEqual([window.center_frame for window in windows], list(range(30, 64)))
        self.assertEqual({window.sequence_lane for window in windows}, {"motion"})
        sample = windows[len(windows) // 2]
        self.assertEqual(sample.x.shape, (288,))
        self.assertEqual(sample.y.shape, (268,))
        self.assertFalse(sample.x.flags.writeable)
        self.assertFalse(sample.y.flags.writeable)
        self.assertEqual(sample.clip_id, clip.clip_id)
        self.assertEqual(sample.motion_sha256, clip.motion_sha256)
        self.assertEqual(sample.terrain_sha256, clip.terrain_sha256)
        self.assertEqual(sample.split_identity, terrain_identity(clip.clip_id))
        self.assertEqual(sample.split, split_identity(sample.split_identity))
        self.assertEqual(sample.terrain_class, "ascent")
        terrain = sample.x[INPUT_LAYOUT["terrain_height"]].reshape(12, 3)
        self.assertTrue(np.isfinite(terrain).all())
        self.assertGreater(terrain[-1, 1], terrain[0, 1])
        np.testing.assert_allclose(terrain[:, 0] - terrain[:, 1], 0.05, atol=1e-6)
        np.testing.assert_allclose(terrain[:, 1] - terrain[:, 2], 0.05, atol=1e-6)

    def test_local_frames_field_order_and_t_to_t_plus_one_indexing(self) -> None:
        flat = _Plane()
        clip, track, _ = _fixture(
            yaw=math.pi / 2.0,
            slope=flat,
            clip_id="walk1_subject1",
            terrain_id="flat",
        )
        sample = next(
            window
            for window in build_clip_windows(clip, track, height_at=flat.height_at)
            if window.center_frame == 40
        )

        trajectory_x = sample.x[INPUT_LAYOUT["trajectory_position"]].reshape(12, 2)
        trajectory_y = sample.y[OUTPUT_LAYOUT["trajectory_position"]].reshape(12, 2)
        np.testing.assert_allclose(trajectory_x[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(trajectory_y[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(trajectory_x[6], (0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(trajectory_y[6], (0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(
            sample.x[INPUT_LAYOUT["trajectory_direction"]].reshape(12, 2),
            np.tile((1.0, 0.0), (12, 1)),
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            sample.x[INPUT_LAYOUT["semantic_intent"]].reshape(12, 2),
            np.tile((0.0, 1.0), (12, 1)),
        )

        input_body = sample.x[INPUT_LAYOUT["previous_body_position"]].reshape(30, 3)
        target_body = sample.y[OUTPUT_LAYOUT["body_position"]].reshape(30, 3)
        expected_offsets = _body_offsets_local()
        expected_offsets[:, 2] += 1.0
        np.testing.assert_allclose(input_body, expected_offsets, atol=1e-6)
        np.testing.assert_allclose(target_body, expected_offsets, atol=1e-6)
        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["joint_position"]], clip.joint_position[41]
        )
        np.testing.assert_array_equal(
            sample.y[OUTPUT_LAYOUT["contact_logit"]], track.contact[41].astype(np.float32)
        )
        self.assertTrue(
            set(sample.y[OUTPUT_LAYOUT["contact_logit"]].tolist()) <= {0.0, 1.0}
        )
        self.assertAlmostEqual(
            float(sample.y[OUTPUT_LAYOUT["phase_advance"]][0]),
            float(track.phase_advance[40]),
        )
        # The displacement from t to t+1 is 0.02 m forward in 1/30 s.
        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["root_planar_velocity"]], (0.6, 0.0), atol=1e-5
        )
        self.assertAlmostEqual(
            float(sample.y[OUTPUT_LAYOUT["root_yaw_velocity"]][0]), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(sample.y[OUTPUT_LAYOUT["root_height"]][0]), 1.0, places=6
        )
        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["root_tilt"]], (0.2, 0.0), atol=1e-6
        )
        self.assertAlmostEqual(sample.phase, float(track.phase[40]), places=6)

    def test_invalid_phase_missing_rays_and_unsupported_family_are_rejected(self) -> None:
        clip, track, slope = _fixture(phase_valid=False)
        with self.assertRaisesRegex(ValueError, r"invalid_phase=34"):
            build_clip_windows(clip, track, height_at=slope.height_at)

        clip, track, _ = _fixture(
            clip_id="walk1_subject1", terrain_id="flat"
        )

        def missing_height(xy: np.ndarray) -> np.ndarray:
            return np.full(np.asarray(xy).shape[:-1], np.nan)

        with self.assertRaisesRegex(ValueError, r"terrain_ray_missing=34"):
            build_clip_windows(clip, track, height_at=missing_height)

        slope_clip, slope_track, slope = _fixture()

        def incomplete_family(xy: np.ndarray) -> np.ndarray:
            points = np.asarray(xy)
            height = np.asarray(slope.height_at(points), dtype=np.float64)
            height = np.array(height, copy=True)
            height[np.isclose(points[..., 0], slope_clip.root_position_world[-1, 0])] = np.nan
            return height

        with self.assertRaisesRegex(ValueError, "terrain_family_grade_unmeasurable"):
            build_clip_windows(
                slope_clip, slope_track, height_at=incomplete_family
            )

        low = _Plane(math.tan(math.radians(4.0)))
        low_clip, low_track, _ = _fixture(slope=low)
        with self.assertRaisesRegex(ValueError, "terrain_family_grade_below_5_degrees"):
            build_clip_windows(low_clip, low_track, height_at=low.height_at)

        high = _Plane(math.tan(math.radians(21.0)))
        high_clip, high_track, _ = _fixture(slope=high)
        with self.assertRaisesRegex(ValueError, "terrain_family_grade_above_20_degrees"):
            build_clip_windows(high_clip, high_track, height_at=high.height_at)
        for degrees in (5.0, 20.0):
            boundary = _Plane(math.tan(math.radians(degrees)))
            boundary_clip, boundary_track, _ = _fixture(slope=boundary)
            with self.subTest(degrees=degrees):
                self.assertTrue(build_clip_windows(
                    boundary_clip, boundary_track, height_at=boundary.height_at
                ))

    def test_root_transition_uses_frame_t_for_planar_and_yaw_velocity(self) -> None:
        clip, track, slope = _fixture(roll=0.0)
        yaw = np.arange(clip.frame_count, dtype=np.float64) * 0.01
        quaternion = np.stack(
            (
                np.cos(yaw / 2.0),
                np.zeros_like(yaw),
                np.zeros_like(yaw),
                np.sin(yaw / 2.0),
            ),
            axis=1,
        ).astype(np.float32)
        turning = _with_body_facing(
            replace(clip, root_quaternion_world_wxyz=quaternion), yaw
        )
        sample = next(
            window
            for window in build_clip_windows(
                turning, track, height_at=slope.height_at
            )
            if window.center_frame == 40
        )
        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["root_planar_velocity"]],
            (0.6 * math.cos(0.4), -0.6 * math.sin(0.4)),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["root_yaw_velocity"]], (0.3,), atol=1e-6
        )
        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["trajectory_position"]].reshape(12, 2)[6],
            (0.0, 0.0),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sample.y[OUTPUT_LAYOUT["trajectory_direction"]].reshape(12, 2)[6],
            (1.0, 0.0),
            atol=1e-6,
        )

    def test_every_accepted_window_has_a_stratification_class(self) -> None:
        ascent = _Plane(0.1)
        ascent_clip, track, _ = _fixture(slope=ascent)
        self.assertEqual(
            {w.terrain_class for w in build_clip_windows(
                ascent_clip, track, height_at=ascent.height_at
            )},
            {"ascent"},
        )

        descent = _Plane(-0.1)
        descent_clip, track, _ = _fixture(slope=descent)
        self.assertEqual(
            {w.terrain_class for w in build_clip_windows(
                descent_clip, track, height_at=descent.height_at
            )},
            {"descent"},
        )

        course = _RunupRamp()
        course_clip, track, _ = _fixture(slope=course)
        self.assertEqual(
            {w.terrain_class for w in build_clip_windows(
                course_clip, track, height_at=course.height_at
            )},
            {"flat", "ascent", "transition"},
        )

    def test_stationary_flat_windows_use_deterministic_eight_phase_bins(self) -> None:
        flat = _Plane()
        clip, track, _ = _fixture(
            step_m=0.0,
            slope=flat,
            clip_id="walk2_subject1",
            terrain_id="flat",
            phase_valid=False,
        )
        first = build_clip_windows(clip, track, height_at=flat.height_at)
        second = build_clip_windows(clip, track, height_at=flat.height_at)
        center = first[0].center_frame
        same_center = [window for window in first if window.center_frame == center]
        self.assertEqual(len(same_center), 8)
        self.assertEqual(
            {window.sequence_lane for window in same_center},
            {f"idle_phase_{index}" for index in range(8)},
        )
        np.testing.assert_allclose(
            [window.phase for window in same_center],
            np.arange(8) * (2.0 * math.pi / 8.0),
            atol=1e-6,
        )
        for window in same_center:
            self.assertEqual(window.terrain_class, "flat")
            self.assertEqual(float(window.y[OUTPUT_LAYOUT["phase_advance"]][0]), 0.0)
            np.testing.assert_array_equal(
                window.x[INPUT_LAYOUT["semantic_intent"]].reshape(12, 2),
                np.tile((1.0, 0.0), (12, 1)),
            )
        self.assertEqual(
            [(w.center_frame, w.phase, w.x.tobytes(), w.y.tobytes()) for w in first],
            [(w.center_frame, w.phase, w.x.tobytes(), w.y.tobytes()) for w in second],
        )
        self.assertFalse(np.shares_memory(first[0].x, second[0].x))

    def test_nonbinary_contacts_and_negative_phase_advance_are_rejected(self) -> None:
        clip, track, slope = _fixture()
        with self.assertRaisesRegex(ValueError, "binary"):
            build_clip_windows(
                clip,
                replace(track, contact=np.full(track.contact.shape, 2, dtype=np.int8)),
                height_at=slope.height_at,
            )
        invalid_advance = np.array(track.phase_advance, copy=True)
        invalid_advance[40] = -0.1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            build_clip_windows(
                clip,
                replace(track, phase_advance=invalid_advance),
                height_at=slope.height_at,
            )

    def test_valid_mask_requires_exact_boolean_dtype_without_coercion(self) -> None:
        clip, track, slope = _fixture()
        float_mask = np.ones(clip.frame_count, dtype=np.float32)
        float_mask[40] = np.nan
        integer_mask = np.ones(clip.frame_count, dtype=np.int8)
        integer_mask[40] = 2
        for label, mask in (("float_nan", float_mask), ("integer_two", integer_mask)):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError, r"phase_track\.valid must have boolean dtype"
                ):
                    build_clip_windows(
                        clip,
                        replace(track, valid=mask),
                        height_at=slope.height_at,
                    )

        self.assertEqual(
            len(build_clip_windows(clip, track, height_at=slope.height_at)), 34
        )

    def test_audit_preserves_partial_and_out_of_bounds_rejection_reasons(self) -> None:
        clip, track, slope = _fixture()
        valid = np.array(track.valid, copy=True)
        valid[40] = False
        result = build_clip_windows_with_audit(
            clip, replace(track, valid=valid), height_at=slope.height_at
        )
        self.assertEqual(len(result.windows), 32)
        self.assertEqual(
            [(item.center_frame, item.reason) for item in result.rejections],
            [(39, "invalid_phase"), (40, "invalid_phase")],
        )

        short_clip, short_track, short_slope = _fixture(frames=40)
        short = build_clip_windows_with_audit(
            short_clip, short_track, height_at=short_slope.height_at
        )
        self.assertEqual(short.windows, ())
        self.assertEqual(
            [(item.center_frame, item.reason) for item in short.rejections],
            [(None, "trajectory_knot_outside_clip")],
        )
        with self.assertRaisesRegex(ValueError, "trajectory_knot_outside_clip=1"):
            build_clip_windows(
                short_clip, short_track, height_at=short_slope.height_at
            )


class MirrorWindowTest(unittest.TestCase):
    def test_native_g1_maps_and_complete_window_mirror_are_involutions(self) -> None:
        self.assertEqual(len(ISAACLAB_JOINT_NAMES), 29)
        self.assertEqual(len(ISAACLAB_BODY_NAMES), 30)
        np.testing.assert_array_equal(
            JOINT_MIRROR_PERMUTATION[JOINT_MIRROR_PERMUTATION], np.arange(29)
        )
        np.testing.assert_array_equal(
            BODY_MIRROR_PERMUTATION[BODY_MIRROR_PERMUTATION], np.arange(30)
        )
        np.testing.assert_array_equal(JOINT_MIRROR_SIGNS**2, np.ones(29))

        clip, track, slope = _fixture()
        sample = build_clip_windows(clip, track, height_at=slope.height_at)[10]
        x = np.array(sample.x, copy=True)
        y = np.array(sample.y, copy=True)
        x[INPUT_LAYOUT["trajectory_position"]].reshape(12, 2)[:, 1] = np.arange(12)
        x[INPUT_LAYOUT["trajectory_direction"]].reshape(12, 2)[:, 1] = 0.25
        y[OUTPUT_LAYOUT["trajectory_position"]].reshape(12, 2)[:, 1] = np.arange(12)
        y[OUTPUT_LAYOUT["trajectory_direction"]].reshape(12, 2)[:, 1] = -0.25
        y[OUTPUT_LAYOUT["root_tilt"]] = (0.3, 0.4)
        y[OUTPUT_LAYOUT["root_planar_velocity"]] = (0.5, -0.6)
        y[OUTPUT_LAYOUT["root_yaw_velocity"]] = 0.7
        sample = replace(sample, x=x, y=y)
        mirrored = mirror_window(sample)
        restored = mirror_window(mirrored)
        np.testing.assert_allclose(restored.x, sample.x, atol=1e-6)
        np.testing.assert_allclose(restored.y, sample.y, atol=1e-6)
        self.assertAlmostEqual(restored.phase, sample.phase, places=6)
        self.assertEqual(restored.clip_id, sample.clip_id)
        self.assertEqual(restored.split_identity, sample.split_identity)
        self.assertEqual(restored.center_frame, sample.center_frame)
        self.assertEqual(restored.motion_sha256, sample.motion_sha256)
        self.assertEqual(restored.terrain_sha256, sample.terrain_sha256)
        self.assertEqual(restored.terrain_class, sample.terrain_class)
        self.assertEqual(mirrored.sequence_lane, sample.sequence_lane)
        self.assertNotEqual(mirrored.clip_id, sample.clip_id)

        original_terrain = sample.x[INPUT_LAYOUT["terrain_height"]].reshape(12, 3)
        mirrored_terrain = mirrored.x[INPUT_LAYOUT["terrain_height"]].reshape(12, 3)
        np.testing.assert_allclose(mirrored_terrain, original_terrain[:, ::-1])
        for layout, original, reflected in (
            (INPUT_LAYOUT, sample.x, mirrored.x),
            (OUTPUT_LAYOUT, sample.y, mirrored.y),
        ):
            for field in ("trajectory_position", "trajectory_direction"):
                expected = original[layout[field]].reshape(12, 2).copy()
                expected[:, 1] *= -1.0
                np.testing.assert_allclose(
                    reflected[layout[field]].reshape(12, 2), expected
                )
        np.testing.assert_array_equal(
            mirrored.x[INPUT_LAYOUT["semantic_intent"]],
            sample.x[INPUT_LAYOUT["semantic_intent"]],
        )
        for layout, original, reflected, position_name, velocity_name in (
            (
                INPUT_LAYOUT, sample.x, mirrored.x,
                "previous_body_position", "previous_body_velocity",
            ),
            (
                OUTPUT_LAYOUT, sample.y, mirrored.y,
                "body_position", "body_velocity",
            ),
        ):
            for field in (position_name, velocity_name):
                expected = original[layout[field]].reshape(30, 3)[
                    BODY_MIRROR_PERMUTATION
                ].copy()
                expected[:, 1] *= -1.0
                np.testing.assert_allclose(
                    reflected[layout[field]].reshape(30, 3), expected
                )
        np.testing.assert_array_equal(
            mirrored.y[OUTPUT_LAYOUT["contact_logit"]],
            sample.y[OUTPUT_LAYOUT["contact_logit"]][[2, 3, 0, 1]],
        )
        np.testing.assert_allclose(
            mirrored.y[OUTPUT_LAYOUT["joint_position"]],
            sample.y[OUTPUT_LAYOUT["joint_position"]][JOINT_MIRROR_PERMUTATION]
            * JOINT_MIRROR_SIGNS,
        )
        np.testing.assert_allclose(
            mirrored.y[OUTPUT_LAYOUT["root_tilt"]], (-0.3, 0.4)
        )
        np.testing.assert_allclose(
            mirrored.y[OUTPUT_LAYOUT["root_planar_velocity"]], (0.5, 0.6)
        )
        np.testing.assert_allclose(
            mirrored.y[OUTPUT_LAYOUT["root_yaw_velocity"]], (-0.7,)
        )
        self.assertAlmostEqual(
            mirrored.phase,
            (sample.phase + math.pi) % (2.0 * math.pi),
            places=6,
        )
        self.assertEqual(
            float(mirrored.y[OUTPUT_LAYOUT["phase_advance"]][0]),
            float(sample.y[OUTPUT_LAYOUT["phase_advance"]][0]),
        )

    def test_window_owns_finite_copies(self) -> None:
        x = np.zeros(288, dtype=np.float32)
        y = np.zeros(268, dtype=np.float32)
        window = PFNNTrainingWindow(
            x=x,
            y=y,
            phase=0.0,
            clip_id="walk1_subject1",
            split_identity="walk1_subject1",
            split="train",
            terrain_class="flat",
            sequence_lane="motion",
            center_frame=30,
            motion_sha256="a" * 64,
            terrain_sha256=None,
        )
        x[0] = 1.0
        y[0] = 1.0
        self.assertEqual(float(window.x[0]), 0.0)
        self.assertEqual(float(window.y[0]), 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(window, x=np.full(288, np.nan))
        nonbinary = np.array(window.y, copy=True)
        nonbinary[OUTPUT_LAYOUT["contact_logit"]] = (0.0, 1.0, 2.0, 0.0)
        with self.assertRaisesRegex(ValueError, "binary"):
            replace(window, y=nonbinary)
        with self.assertRaisesRegex(ValueError, "canonical identity"):
            replace(window, split_identity="walk2_subject3")
        with self.assertRaisesRegex(ValueError, "sealed split"):
            replace(window, split="validation")
        for invalid_lane in ("", "idle", "idle_phase_8", "motion_phase_0"):
            with self.subTest(invalid_lane=invalid_lane):
                with self.assertRaisesRegex(ValueError, "sequence_lane"):
                    replace(window, sequence_lane=invalid_lane)


if __name__ == "__main__":
    unittest.main()
