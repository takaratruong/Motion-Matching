from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.terrain_oracle.canonical import (
    CanonicalTerrainMesh,
    ISAACLAB_BODY_NAMES,
)
from mm_sonic.terrain_oracle.contact import CanonicalMeshQuery, SoleGeometry
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_pfnn.phase import (
    phase_from_contacts,
    reconstruct_heel_toe_contacts,
)
from mm_sonic.terrain_pfnn.sources import PFNNSourceClip


MODEL = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/assets/skeletons/g1/g1_29dof.xml"
)
LEFT_ANKLE = ISAACLAB_BODY_NAMES.index("left_ankle_roll_link")
RIGHT_ANKLE = ISAACLAB_BODY_NAMES.index("right_ankle_roll_link")


def contacts_with_strikes(*strikes: tuple[str, int], frames: int = 91) -> np.ndarray:
    contact = np.zeros((frames, 4), dtype=bool)
    for side, frame in strikes:
        channel = 0 if side == "left" else 2
        contact[frame : min(frame + 10, frames), channel] = True
    return contact


def _plane_query() -> CanonicalMeshQuery:
    mesh = CanonicalTerrainMesh(
        vertices_local=np.array(
            ((-2.0, -2.0, 0.0), (2.0, -2.0, 0.0),
             (2.0, 2.0, 0.0), (-2.0, 2.0, 0.0)),
            dtype=np.float32,
        ),
        faces=np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        valid_faces=np.ones(2, dtype=bool),
        source_asset_sha256="f" * 64,
    )
    return CanonicalMeshQuery(
        mesh,
        RigidTransform(
            np.zeros(3, dtype=np.float32),
            np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        ),
    )


def _source_with_ankles(
    left_position: np.ndarray,
    left_quaternion_wxyz: np.ndarray,
    *,
    frames: int = 6,
) -> PFNNSourceClip:
    body_position = np.zeros((frames, 30, 3), dtype=np.float32)
    body_position[:, LEFT_ANKLE] = left_position
    body_position[:, RIGHT_ANKLE, 2] = 1.0
    body_quaternion = np.zeros((frames, 30, 4), dtype=np.float32)
    body_quaternion[..., 0] = 1.0
    body_quaternion[:, LEFT_ANKLE] = left_quaternion_wxyz
    return PFNNSourceClip(
        clip_id="native-heel-only",
        terrain_id="flat",
        fps=30.0,
        root_position_world=np.zeros((frames, 3), dtype=np.float32),
        root_quaternion_world_wxyz=np.tile(
            np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (frames, 1)
        ),
        joint_position=np.zeros((frames, 29), dtype=np.float32),
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_quaternion,
        body_linear_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
        body_angular_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
        root_linear_velocity_world=np.zeros((frames, 3), dtype=np.float32),
        root_angular_velocity_world=np.zeros((frames, 3), dtype=np.float32),
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


class PhaseFromContactsTest(unittest.TestCase):
    def test_alternating_strikes_anchor_zero_pi_and_wrap(self) -> None:
        contact = np.zeros((91, 4), dtype=bool)
        contact[3:13, 0] = True
        contact[33:43, 2] = True
        contact[63:73, 0] = True
        track = phase_from_contacts(contact, fps=30.0)
        self.assertTrue(track.valid[3:64].all())
        self.assertAlmostEqual(track.phase[3], 0.0)
        self.assertAlmostEqual(track.phase[33], np.pi)
        self.assertAlmostEqual(track.phase[63], 0.0)
        self.assertTrue((track.phase_advance[3:63] >= 0.0).all())

    def test_same_foot_twice_rejects_the_ambiguous_span(self) -> None:
        contact = contacts_with_strikes(("left", 3), ("left", 33), ("right", 63))
        track = phase_from_contacts(contact, fps=30.0)
        self.assertFalse(track.valid[3:34].any())
        self.assertTrue(track.valid[34:64].all())

    def test_half_cycle_outside_point_two_to_one_second_is_rejected(self) -> None:
        contact = contacts_with_strikes(("left", 3), ("right", 8), ("left", 43))
        track = phase_from_contacts(contact, fps=30.0)
        self.assertFalse(track.valid[3:9].any())

    def test_unstable_rise_is_a_barrier_to_the_candidate_half_cycle(self) -> None:
        contact = np.zeros((64, 4), dtype=bool)
        contact[3:7, 0] = True
        contact[9:19, 0] = True
        contact[33:43, 2] = True
        track = phase_from_contacts(contact, fps=30.0)
        self.assertFalse(track.valid[3:34].any())

    def test_one_stable_one_unstable_simultaneous_rise_removes_both(self) -> None:
        contact = np.zeros((121, 4), dtype=bool)
        contact[3:32, 0] = True
        contact[33:43, 0] = True
        contact[33:43, 2] = True
        contact[63:73, 2] = True
        contact[90:100, 0] = True
        track = phase_from_contacts(contact, fps=30.0)
        self.assertFalse(track.valid[3:64].any())
        self.assertTrue(track.valid[64:91].all())

    def test_simultaneous_left_right_rises_are_not_phase_anchors(self) -> None:
        contact = np.zeros((64, 4), dtype=bool)
        contact[3:13, 0] = True
        contact[3:13, 2] = True
        contact[33:43, 0] = True
        track = phase_from_contacts(contact, fps=30.0)
        self.assertFalse(track.valid.any())

    def test_confidence_is_preserved_in_channel_order(self) -> None:
        contact = contacts_with_strikes(("left", 3), ("right", 33))
        confidence = np.arange(contact.size, dtype=np.float32).reshape(contact.shape)
        confidence /= float(contact.size)
        track = phase_from_contacts(contact, confidence=confidence, fps=30.0)
        np.testing.assert_array_equal(track.contact, contact)
        np.testing.assert_array_equal(track.confidence, confidence)

    def test_rejected_internal_span_is_not_filled(self) -> None:
        contact = contacts_with_strikes(
            ("left", 3), ("right", 33), ("right", 63), ("left", 90), frames=121
        )
        track = phase_from_contacts(contact, fps=30.0)
        self.assertTrue(track.valid[3:33].all())
        self.assertFalse(track.valid[33:64].any())
        self.assertTrue(track.valid[64:91].all())

    def test_short_rejected_interval_removes_shared_anchors(self) -> None:
        contact = contacts_with_strikes(
            ("left", 3), ("right", 33), ("left", 34), ("right", 64), frames=91
        )
        track = phase_from_contacts(contact, fps=30.0)
        self.assertFalse(track.valid[33:35].any())
        self.assertTrue(track.valid[3:33].all())
        self.assertTrue(track.valid[35:65].all())
        self.assertTrue(np.all(track.phase_advance >= 0.0))

    def test_adjacent_trailing_bilateral_margin_extends_at_zero_advance(self) -> None:
        contact = np.zeros((64, 4), dtype=bool)
        contact[3:, 0] = True
        contact[33:, 2] = True
        track = phase_from_contacts(contact, fps=30.0)
        self.assertTrue(track.valid[3:].all())
        np.testing.assert_allclose(track.phase[33:], math.pi, atol=1.0e-6)
        np.testing.assert_array_equal(
            track.phase_advance[33:], np.zeros(31, dtype=np.float32)
        )

    def test_bilateral_prefix_and_single_support_bridge_extend_leading_phase(self) -> None:
        contact = np.zeros((71, 4), dtype=bool)
        contact[0:4, 0] = True
        contact[0:20, 2] = True
        contact[10:50, 0] = True
        contact[40:50, 2] = True
        track = phase_from_contacts(contact, fps=30.0)
        omega = math.pi / 30.0
        self.assertTrue(track.valid[0:41].all())
        np.testing.assert_allclose(track.phase[10], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(track.phase_advance[0:3], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(track.phase_advance[3:10], omega, atol=1.0e-6)

    def test_leading_gap_without_opposite_support_remains_invalid(self) -> None:
        contact = np.zeros((71, 4), dtype=bool)
        contact[0:4, 0] = True
        contact[0:4, 2] = True
        contact[10:50, 0] = True
        contact[40:50, 2] = True
        track = phase_from_contacts(contact, fps=30.0)
        self.assertFalse(track.valid[:10].any())
        self.assertTrue(track.valid[10:41].all())

    def test_nonadjacent_and_internal_bilateral_support_do_not_bridge(self) -> None:
        contact = np.zeros((121, 4), dtype=bool)
        contact[3:13, 0] = True
        contact[33:42, 2] = True
        contact[45:55, 0] = True
        contact[45:55, 2] = True
        contact[63:73, 0] = True
        contact[90:100, 2] = True
        contact[105:, 0] = True
        contact[105:, 2] = True
        track = phase_from_contacts(contact, fps=30.0)
        self.assertFalse(track.valid[34:63].any())
        self.assertFalse(track.valid[105:].any())

    def test_invalid_shapes_and_nonfinite_confidence_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"contact\[T,4\]"):
            phase_from_contacts(np.zeros((10, 2), dtype=bool), fps=30.0)
        with self.assertRaisesRegex(ValueError, "positive fps"):
            phase_from_contacts(np.zeros((10, 4), dtype=bool), fps=math.nan)
        for fps in (29.999, 60.0):
            with self.subTest(fps=fps):
                with self.assertRaisesRegex(ValueError, "exactly 30 Hz"):
                    phase_from_contacts(np.zeros((10, 4), dtype=bool), fps=fps)
        with self.assertRaisesRegex(ValueError, "confidence"):
            confidence = np.zeros((10, 4), dtype=np.float32)
            confidence[0, 0] = np.nan
            phase_from_contacts(
                np.zeros((10, 4), dtype=bool), confidence=confidence, fps=30.0
            )


@unittest.skipUnless(
    MODEL.is_file() and importlib.util.find_spec("mujoco") is not None,
    "native G1 MuJoCo model is unavailable",
)
class NativeContactGeometryTest(unittest.TestCase):
    def test_native_model_separates_grounded_heel_from_lifted_toe(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MODEL))
        geometry = SoleGeometry.from_model(model)
        heel = np.asarray(geometry.heel_positions_body[0], dtype=np.float64)
        toe = np.asarray(geometry.toe_positions_body[0], dtype=np.float64)
        longitudinal_span = float(toe[0] - heel[0])
        pitch = -math.asin(0.04 / longitudinal_span)
        quaternion = np.array(
            (math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0),
            dtype=np.float64,
        )
        rotated_heel_z = -math.sin(pitch) * heel[0] + math.cos(pitch) * heel[2]
        source = _source_with_ankles(
            np.array((0.0, 0.0, -rotated_heel_z), dtype=np.float32),
            quaternion.astype(np.float32),
        )

        track = reconstruct_heel_toe_contacts(source, _plane_query(), geometry)

        np.testing.assert_array_equal(
            track.contact,
            np.tile(np.array((True, False, False, False)), (source.frame_count, 1)),
        )
        self.assertTrue(np.all(track.confidence[:, 0] > track.confidence[:, 1]))

    def test_paired_terrain_digest_mismatch_is_invalid_geometry(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MODEL))
        geometry = SoleGeometry.from_model(model)
        source = replace(
            _source_with_ankles(
                np.zeros(3, dtype=np.float32),
                np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            ),
            terrain_path=Path("paired.usd"),
            terrain_sha256="a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "invalid geometry.*terrain digest"):
            reconstruct_heel_toe_contacts(source, _plane_query(), geometry)

    def test_nonfinite_surface_result_is_invalid_geometry(self) -> None:
        import mujoco

        class NonfiniteQuery(CanonicalMeshQuery):
            def query(self, points_world: object) -> object:
                count = len(np.asarray(points_world))
                return SimpleNamespace(
                    distance_m=np.full(count, np.nan),
                    surface_normal_world=np.zeros((count, 3)),
                    downward_ray_distance_m=np.full(count, np.inf),
                    downward_ray_normal_world=np.zeros((count, 3)),
                )

        model = mujoco.MjModel.from_xml_path(str(MODEL))
        geometry = SoleGeometry.from_model(model)
        base_query = _plane_query()
        query = NonfiniteQuery(base_query.mesh, base_query.world_from_terrain)
        source = _source_with_ankles(
            np.zeros(3, dtype=np.float32),
            np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "invalid geometry.*surface query"):
            reconstruct_heel_toe_contacts(source, query, geometry)


if __name__ == "__main__":
    unittest.main()
