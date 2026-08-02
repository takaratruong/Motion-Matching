from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.torch_terrain_quality_states import (
    capture_quality_states,
    load_quality_state_corpus,
    save_quality_state_corpus,
)


def _arrays(frame_count: int = 8):
    root = np.zeros((frame_count, 3), dtype=np.float64)
    root[:, 0] = np.arange(frame_count) * 0.01
    feet = np.zeros((frame_count, 2, 3), dtype=np.float64)
    feet[:, :, 2] = 0.035
    surface = np.zeros((frame_count, 2), dtype=np.float64)
    surface[4:, 1] = 0.18
    return {
        "qpos": np.zeros((frame_count, 36), dtype=np.float64),
        "joint_position": np.zeros((frame_count, 29), dtype=np.float64),
        "joint_velocity": np.zeros((frame_count, 29), dtype=np.float64),
        "root_position_world": root,
        "root_yaw_world": np.zeros(frame_count, dtype=np.float64),
        "foot_position_world": feet,
        "foot_surface_height_m": surface,
        "command_velocity_world_xy": np.tile([0.4, 0.0], (frame_count, 1)),
        "command_heading_world_yaw": np.zeros(frame_count, dtype=np.float64),
        "command_segment_index": np.array([0, 0, 0, 1, 1, 1, 1, 1]),
        "selected_clip_path": np.array(["a", "a", "a", "b", "b", "b", "b", "b"]),
        "selected_source_frame": np.array([10, 11, 12, 5, 6, 7, 8, 9]),
    }


def _support(frame_count: int = 8):
    return np.ones((frame_count, 2), dtype=bool)


def _sampler(points: np.ndarray) -> np.ndarray:
    return points[:, 0] + 2.0 * points[:, 1]


class TerrainQualityStateTests(unittest.TestCase):
    def test_captures_command_source_and_split_height_boundaries(self):
        states = capture_quality_states(
            route_name="turn-90-middle-left",
            arrays=_arrays(),
            source_support_mask=_support(),
            terrain_patch_sampler=_sampler,
        )

        identities = {(state.route_frame, state.reason) for state in states}
        self.assertIn((0, "reset"), identities)
        self.assertIn((3, "command-boundary"), identities)
        self.assertIn((3, "source-transition"), identities)
        self.assertIn((4, "split-height-stance"), identities)
        self.assertEqual(len(identities), len(states))

    def test_patch_is_centered_on_root_with_fixed_shape(self):
        arrays = _arrays()
        arrays["root_position_world"][0, :2] = [1.0, -2.0]
        state = capture_quality_states(
            route_name="route",
            arrays=arrays,
            source_support_mask=_support(),
            terrain_patch_sampler=_sampler,
        )[0]

        self.assertEqual(state.terrain_patch_world_xyh.shape, (441, 3))
        np.testing.assert_allclose(
            state.terrain_patch_world_xyh[220], [1.0, -2.0, -3.0], atol=1e-12
        )
        self.assertFalse(state.terrain_patch_world_xyh.flags.writeable)

    def test_state_identity_changes_with_command_but_not_input_mutation(self):
        arrays = _arrays()
        first = capture_quality_states(
            route_name="route",
            arrays=arrays,
            source_support_mask=_support(),
            terrain_patch_sampler=_sampler,
        )[0]
        arrays["command_velocity_world_xy"][0, 0] = 0.2
        second = capture_quality_states(
            route_name="route",
            arrays=arrays,
            source_support_mask=_support(),
            terrain_patch_sampler=_sampler,
        )[0]

        self.assertNotEqual(first.state_id, second.state_id)
        with self.assertRaises(ValueError):
            first.qpos[0] = 1.0

    def test_state_corpus_round_trip_is_byte_stable(self):
        states = capture_quality_states(
            route_name="route",
            arrays=_arrays(),
            source_support_mask=_support(),
            terrain_patch_sampler=_sampler,
        )
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first"
            second_path = Path(temporary) / "second"
            first_hash = save_quality_state_corpus(states, first_path)
            loaded = load_quality_state_corpus(first_path)
            second_hash = save_quality_state_corpus(loaded, second_path)

            self.assertEqual(first_hash, second_hash)
            self.assertEqual(
                (first_path / "manifest.json").read_bytes(),
                (second_path / "manifest.json").read_bytes(),
            )
            self.assertEqual(
                tuple(state.state_id for state in loaded),
                tuple(state.state_id for state in states),
            )

    def test_rejects_bad_sampler_shape(self):
        with self.assertRaisesRegex(ValueError, "terrain patch"):
            capture_quality_states(
                route_name="route",
                arrays=_arrays(),
                source_support_mask=_support(),
                terrain_patch_sampler=lambda points: np.zeros(3),
            )


if __name__ == "__main__":
    unittest.main()
