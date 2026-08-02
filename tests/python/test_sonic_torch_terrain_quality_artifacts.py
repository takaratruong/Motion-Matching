from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mm_sonic.torch_terrain_features import TerrainDataset
from mm_sonic.torch_terrain_quality_artifacts import (
    analyze_saved_route,
    reconstruct_source_support,
    transition_start_mask,
)


def _clip(path: str, support: np.ndarray):
    frame_count = support.shape[0]
    position = np.zeros((frame_count, 3, 3), dtype=np.float32)
    velocity = np.zeros_like(position)
    position[:, 1:, 2] = np.where(support, 0.035, 0.20)
    return SimpleNamespace(
        relative_path=path,
        body_position_world=position,
        body_linear_velocity_world=velocity,
    )


def _dataset():
    clips = (
        _clip("a/motion.npz", np.ones((3, 2), dtype=bool)),
        _clip(
            "b/motion.npz",
            np.array([[False, True], [True, True], [True, False]]),
        ),
    )
    folder = SimpleNamespace(
        clips=clips,
        layout=SimpleNamespace(left_foot_body_index=1, right_foot_body_index=2),
    )
    return TerrainDataset(
        root=Path("."),
        folder=folder,
        clip_grids=(None, None),
        clip_alignments=(None, None),
        manifest_sha256="manifest",
        _manifest={},
        device=torch.device("cpu"),
    )


def _arrays():
    frame_count = 3
    feet = np.zeros((frame_count, 2, 3), dtype=np.float64)
    feet[:, :, 2] = 0.035
    return {
        "foot_position_world": feet,
        "foot_surface_height_m": np.zeros((frame_count, 2), dtype=np.float64),
        "root_position_world": np.zeros((frame_count, 3), dtype=np.float64),
        "command_velocity_world_xy": np.zeros((frame_count, 2), dtype=np.float64),
        "selected_clip_path": np.array(
            ["a/motion.npz", "a/motion.npz", "b/motion.npz"]
        ),
        "selected_source_frame": np.array([0, 1, 0], dtype=np.int64),
        "step_time_ns": np.array([1, 2, 3], dtype=np.int64),
        "search_time_ns": np.array([4, 5, 6], dtype=np.int64),
    }


class TerrainQualityArtifactTests(unittest.TestCase):
    def test_reconstructs_support_by_exact_clip_and_source_frame(self):
        support = reconstruct_source_support(
            _dataset(),
            np.array(["a/motion.npz", "a/motion.npz", "b/motion.npz"]),
            np.array([0, 1, 0]),
        )

        np.testing.assert_array_equal(
            support,
            [[True, True], [True, True], [False, True]],
        )
        self.assertFalse(support.flags.writeable)

    def test_rejects_unknown_clip_or_out_of_range_frame(self):
        with self.assertRaisesRegex(ValueError, "unknown selected clip"):
            reconstruct_source_support(
                _dataset(), np.array(["missing/motion.npz"]), np.array([0])
            )
        with self.assertRaisesRegex(ValueError, "source frame"):
            reconstruct_source_support(
                _dataset(), np.array(["a/motion.npz"]), np.array([10])
            )

    def test_transition_mask_marks_initial_switch_and_nonconsecutive_source(self):
        mask = transition_start_mask(
            np.array(["a", "a", "a", "b", "b"]),
            np.array([10, 11, 20, 3, 4]),
        )
        np.testing.assert_array_equal(mask, [True, False, True, True, False])
        self.assertFalse(mask.flags.writeable)

    def test_saved_route_hash_excludes_timing_arrays(self):
        first = analyze_saved_route(_dataset(), "route", _arrays())
        changed = _arrays()
        changed["step_time_ns"][:] = 99
        changed["search_time_ns"][:] = 101
        second = analyze_saved_route(_dataset(), "route", changed)

        self.assertEqual(first.deterministic_sha256, second.deterministic_sha256)
        np.testing.assert_array_equal(
            first.source_support_mask,
            [[True, True], [True, True], [False, True]],
        )
        np.testing.assert_array_equal(
            first.transition_start_mask,
            [True, False, True],
        )

    def test_saved_route_hash_changes_with_non_timing_motion(self):
        first = analyze_saved_route(_dataset(), "route", _arrays())
        changed = _arrays()
        changed["root_position_world"][2, 0] = 0.1
        second = analyze_saved_route(_dataset(), "route", changed)
        self.assertNotEqual(first.deterministic_sha256, second.deterministic_sha256)


if __name__ == "__main__":
    unittest.main()
