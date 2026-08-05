import tempfile
import unittest
from pathlib import Path
import math

import numpy as np
import torch

from mm_sonic.annotate_rollout_commands import annotate
from mm_sonic.human_command_curriculum import (
    COMMAND_EVENT_ABRUPT,
    COMMAND_EVENT_DUAL_STICK,
    COMMAND_EVENT_HEADING_JUMP,
    COMMAND_EVENT_STOP_OR_RESTART,
    COMMAND_EVENT_VELOCITY_REVERSE,
    build_abrupt_curriculum,
    build_base_curriculum,
    build_omnidirectional_curriculum,
    build_paired_curriculum,
    command_event_mask,
    map_two_stick_commands,
    mirror_command_trace,
)
from mm_sonic.offline_corpus import (
    BODY_NAMES,
    JOINT_NAMES,
    _motion_folder_for_npz,
    _task12_intended,
    generate_clip,
    mirror_clip,
    validate_mirror_pair,
    write_corpus,
)
from mm_sonic.prepare_sonic_bundle import prepare
from mm_sonic.torch_motion_data import MotionFolder
from mm_sonic.torch_motion_matcher import (
    MatcherConfig,
    TorchMotionMatcher,
    critically_damped_position_step,
    critically_damped_yaw_step,
)
from tests.python.torch_motion_test_utils import (
    build_varying_takara_arrays,
    write_takara_arrays,
)


class HumanCommandCurriculumTests(unittest.TestCase):
    def test_curriculum_has_sixty_base_and_exact_command_pairs(self):
        base = build_base_curriculum(frames=100)
        paired = build_paired_curriculum(frames=100)
        self.assertEqual(len(base), 60)
        self.assertEqual(len(paired), 120)
        for original, mirrored in zip(paired[::2], paired[1::2], strict=True):
            expected = mirror_command_trace(original)
            self.assertEqual(mirrored.trace_id, expected.trace_id)
            np.testing.assert_array_equal(mirrored.raw_sticks, expected.raw_sticks)
            np.testing.assert_array_equal(
                mirrored.requested_velocity_local_xy,
                expected.requested_velocity_local_xy,
            )
            np.testing.assert_array_equal(
                mirrored.requested_heading_world_yaw,
                expected.requested_heading_world_yaw,
            )

    def test_centered_facing_stick_holds_last_heading(self):
        raw = np.asarray(
            [
                [0.0, -1.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        buttons = np.ones((3, 3), dtype=np.uint8)
        buttons[:, 2] = 0
        _, heading = map_two_stick_commands(raw, buttons)
        np.testing.assert_allclose(heading, np.pi / 2, atol=1e-6)

    def test_omnidirectional_curriculum_covers_full_two_stick_grid(self):
        base = build_omnidirectional_curriculum(frames=240)
        self.assertEqual(len(base), 102)
        steady = [trace for trace in base if trace.category == "omni_steady"]
        transitions = [
            trace for trace in base if trace.category == "omni_transition"
        ]
        self.assertEqual(len(steady), 34)
        self.assertEqual(len(transitions), 68)

        states = set()
        for trace in steady:
            for command in (trace, mirror_command_trace(trace)):
                velocity = command.requested_velocity_local_xy[-1]
                travel_bin = (
                    round(
                        math.atan2(float(velocity[1]), float(velocity[0]))
                        / (math.pi / 4.0)
                    )
                    % 8
                )
                heading_bin = (
                    round(
                        float(command.requested_heading_world_yaw[-1])
                        / (math.pi / 4.0)
                    )
                    % 8
                )
                states.add((travel_bin, heading_bin))
        self.assertEqual(len(states), 64)
        for trace in transitions:
            speed = np.linalg.norm(
                trace.requested_velocity_local_xy,
                axis=1,
            )
            self.assertLess(float(speed.min()), 1e-6)

    def test_omnidirectional_curriculum_expands_each_steady_speed_grid(self):
        magnitudes = (0.40, 0.70, 0.90)
        base = build_omnidirectional_curriculum(
            frames=240,
            steady_magnitudes=magnitudes,
        )
        steady = [trace for trace in base if trace.category == "omni_steady"]
        transitions = [
            trace for trace in base if trace.category == "omni_transition"
        ]
        self.assertEqual(len(base), 170)
        self.assertEqual(len(steady), 102)
        self.assertEqual(len(transitions), 68)
        observed = {
            round(float(np.linalg.norm(trace.raw_sticks[-1, :2])), 2)
            for trace in steady
        }
        self.assertEqual(observed, set(magnitudes))

    def test_abrupt_curriculum_has_balanced_one_frame_command_changes(self):
        base = build_abrupt_curriculum(frames=240)
        self.assertEqual(len(base), 102)
        self.assertEqual(
            {trace.category for trace in base},
            {
                "abrupt_velocity",
                "abrupt_dual_stick",
                "abrupt_stop_speed_yaw",
            },
        )
        for trace in base:
            raw_delta = np.linalg.norm(np.diff(trace.raw_sticks, axis=0), axis=1)
            self.assertGreater(float(raw_delta.max()), 0.5)
        for trace in base:
            if trace.category == "abrupt_velocity":
                speed = np.linalg.norm(
                    trace.requested_velocity_local_xy, axis=1
                )
                self.assertLess(float(speed.min()), 1e-6)

    def test_abrupt_events_are_explicitly_available_for_weighted_sampling(self):
        masks = np.concatenate(
            [
                command_event_mask(trace.raw_sticks)
                for trace in build_abrupt_curriculum(frames=240)
            ]
        )
        self.assertEqual(int(np.count_nonzero(masks & COMMAND_EVENT_ABRUPT)), 408)
        self.assertEqual(
            int(np.count_nonzero(masks & COMMAND_EVENT_STOP_OR_RESTART)), 204
        )
        self.assertEqual(
            int(np.count_nonzero(masks & COMMAND_EVENT_VELOCITY_REVERSE)), 68
        )
        self.assertEqual(
            int(np.count_nonzero(masks & COMMAND_EVENT_HEADING_JUMP)), 204
        )
        self.assertEqual(
            int(np.count_nonzero(masks & COMMAND_EVENT_DUAL_STICK)), 170
        )


class OfflineCorpusTests(unittest.TestCase):
    def _matcher(self, root: Path) -> tuple[TorchMotionMatcher, Path]:
        source = write_takara_arrays(
            root / "walk", build_varying_takara_arrays(frames=150)
        )
        folder = MotionFolder.load(root)
        matcher = TorchMotionMatcher.from_folder(root, device="cpu")
        self.assertEqual(
            matcher.database._search_features.shape[0],
            sum(clip.valid_frame_stop for clip in folder.clips),
        )
        return matcher, source

    def test_legacy_xyzw_source_is_canonicalized_to_wxyz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _matcher, canonical_source = self._matcher(root / "source")
            with np.load(canonical_source, allow_pickle=False) as archive:
                values = {name: np.asarray(archive[name]) for name in archive.files}
            expected_wxyz = values["body_quat_w"].copy()
            values["body_quat_w"] = expected_wxyz[..., (1, 2, 3, 0)]
            legacy_source = root / "legacy_xyzw.npz"
            np.savez(legacy_source, **values)

            converted = _motion_folder_for_npz(
                legacy_source,
                source_quaternion_convention="xyzw",
            )
            try:
                folder = MotionFolder.load(converted.name)
                np.testing.assert_allclose(
                    folder.clips[0].body_quaternion_world_wxyz,
                    expected_wxyz,
                    atol=1e-7,
                )
            finally:
                converted.cleanup()

    def test_full_body_matcher_output_and_exact_mirror_involution(self):
        with tempfile.TemporaryDirectory() as tmp:
            matcher, _source = self._matcher(Path(tmp))
            trace = build_base_curriculum(frames=100)[0]
            arrays = generate_clip(matcher, trace)
            self.assertEqual(arrays["body_pos_w"].shape, (100, 30, 3))
            self.assertEqual(arrays["body_quat_w"].shape, (100, 30, 4))
            self.assertEqual(arrays["task12_intended"].shape, (100, 12))
            self.assertEqual(len(JOINT_NAMES), 29)
            self.assertEqual(len(BODY_NAMES), 30)
            mirrored = mirror_clip(arrays)
            validate_mirror_pair(arrays, mirrored)
            round_trip = mirror_clip(mirrored)
            for name in arrays:
                np.testing.assert_array_equal(round_trip[name], arrays[name])

    def test_takara_spring_matches_cpp_reference_values(self):
        position, velocity, acceleration = critically_damped_position_step(
            torch.tensor([1.25, -0.4], dtype=torch.float32),
            torch.tensor([0.3, -0.2], dtype=torch.float32),
            torch.tensor([-0.1, 0.05], dtype=torch.float32),
            torch.tensor([0.8, 0.25], dtype=torch.float32),
            halflife_s=0.27,
            dt=0.02,
        )
        np.testing.assert_allclose(
            position.numpy(), [1.2560221, -0.40395463], atol=1e-7
        )
        np.testing.assert_allclose(
            velocity.numpy(), [0.30059367, -0.19693860], atol=1e-7
        )
        np.testing.assert_allclose(
            acceleration.numpy(), [0.15692414, 0.25461093], atol=1e-7
        )
        yaw, yaw_velocity = critically_damped_yaw_step(
            torch.tensor(2.9, dtype=torch.float32),
            torch.tensor(-0.35, dtype=torch.float32),
            torch.tensor(-2.7, dtype=torch.float32),
            halflife_s=0.27,
            dt=0.02,
        )
        self.assertAlmostEqual(float(yaw), 2.8969600, places=6)
        self.assertAlmostEqual(float(yaw_velocity), 0.04162413, places=6)

    def test_takara_ball_clamps_generated_root_and_mirrors_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(
                root / "walk", build_varying_takara_arrays(frames=150)
            )
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(
                    trajectory_model="takara_ball",
                    max_source_joint_step_rad=0.35,
                ),
            )
            trace = build_base_curriculum(frames=100)[20]
            arrays = generate_clip(matcher, trace)
            separation = np.linalg.norm(
                arrays["body_pos_w"][:, 0, :2]
                - arrays["command_ball_position_world_xy"],
                axis=1,
            )
            self.assertLessEqual(float(separation.max()), 0.150001)
            self.assertTrue(
                np.array_equal(
                    arrays["command_path_world_xy"],
                    arrays["command_ball_position_world_xy"],
                )
            )
            mirrored = mirror_clip(arrays)
            validate_mirror_pair(arrays, mirrored)
            np.testing.assert_array_equal(
                mirrored["matcher_root_clamp_distance_m"],
                arrays["matcher_root_clamp_distance_m"],
            )

    def test_source_step_filter_rejects_every_window_crossing_a_jump(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = build_varying_takara_arrays(frames=150)
            arrays["joint_pos"][80:, 0] += np.float32(0.8)
            arrays["joint_vel"][130, 1] = np.float32(100.0)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root,
                device="cpu",
                config=MatcherConfig(max_source_joint_step_rad=0.35),
            )
            self.assertIsNotNone(matcher.database.row_for_source(0, 34))
            for frame in range(35, 80):
                self.assertIsNone(
                    matcher.database.row_for_source(0, frame),
                    f"frame {frame} publishes a window crossing the jump",
                )
            self.assertIsNotNone(matcher.database.row_for_source(0, 80))
            self.assertIsNotNone(matcher.database.row_for_source(0, 84))
            for frame in range(85, 105):
                self.assertIsNone(
                    matcher.database.row_for_source(0, frame),
                    f"frame {frame} publishes a window with unsafe velocity",
                )

    def test_written_zarr_is_multiclip_pipeline_shape(self):
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matcher, source = self._matcher(root / "source")
            trace = build_base_curriculum(frames=100)[0]
            arrays = generate_clip(matcher, trace)
            mirrored = mirror_clip(arrays)
            mirror_trace = mirror_command_trace(trace)
            output = root / "corpus.zarr"
            write_corpus(
                output,
                [
                    (trace, arrays, None),
                    (mirror_trace, mirrored, trace.trace_id),
                ],
                source_npz=source,
                matcher=matcher,
                matcher_config=MatcherConfig(),
            )
            store = zarr.open(str(output), mode="r")
            self.assertEqual(
                np.asarray(store["meta/episode_ends"]).tolist(), [100, 200]
            )
            self.assertEqual(
                np.asarray(store["meta/episode_clip"]).tolist(),
                [trace.trace_id, mirror_trace.trace_id],
            )
            self.assertEqual(
                np.asarray(store["meta/episode_command_seed"]).tolist(),
                [-1, -1],
            )
            self.assertEqual(store["data/joint_pos"].shape, (200, 29))
            self.assertEqual(store["data/body_pos_w"].shape, (200, 30, 3))
            self.assertEqual(store["data/task4_intended"].shape, (200, 4))
            self.assertEqual(store["data/task12_intended"].shape, (200, 12))
            self.assertEqual(
                store["data/matcher_selected_clip_index"].shape,
                (200,),
            )
            self.assertEqual(
                np.asarray(store["meta/source_motion_clip"]).tolist(),
                ["walk/motion.npz"],
            )
            try:
                import joblib  # noqa: F401
            except ModuleNotFoundError:
                self.skipTest("Justin bridge dependencies live in env_isaaclab")
            prepared = root / "prepared"
            prepare(
                output,
                prepared,
                grail_stairs_root=Path(
                    "/move/u/justingu/Projects/grail-stairs"
                ),
            )
            self.assertTrue((prepared / "motion_lib_merged.pkl").is_file())
            self.assertTrue((prepared / "clips.json").is_file())
            self.assertEqual(
                len(list((prepared / "object_usd").glob("*.usd"))), 2
            )

    def test_noisy_command_sidecar_uses_physical_pelvis_frame(self):
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matcher, source = self._matcher(root / "source")
            trace = build_base_curriculum(frames=100)[0]
            arrays = generate_clip(matcher, trace)
            corpus = root / "corpus.zarr"
            write_corpus(
                corpus,
                [(trace, arrays, None)],
                source_npz=source,
                matcher=matcher,
                matcher_config=MatcherConfig(),
            )
            rollout = zarr.open_group(str(root / "rollout.zarr"), mode="w")
            data = rollout.create_group("data")
            meta = rollout.create_group("meta")
            body_rotation = arrays["body_quat_w"][:20].copy()
            yaw = np.pi / 2
            body_rotation[:, 0] = np.asarray(
                [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                dtype=np.float32,
            )
            data.array("body_rot", body_rotation.reshape(20, 120))
            meta.array("episode_ends", np.asarray([20], dtype=np.int64))
            meta.array(
                "episode_clip", np.asarray([trace.trace_id], dtype="<U64")
            )
            meta.array(
                "episode_start_frame", np.asarray([0], dtype=np.int64)
            )
            sidecar = root / "commands.zarr"
            annotate(corpus, root / "rollout.zarr", sidecar)
            labels = zarr.open(str(sidecar), mode="r")
            task = np.asarray(labels["data/task4_intended"])
            shaped = np.asarray(
                labels["data/command_shaped_velocity_world_xy"]
            )
            self.assertEqual(
                np.asarray(labels["meta/episode_command_seed"]).tolist(),
                [-1],
            )
            np.testing.assert_allclose(task[:, 0], shaped[:, 1], atol=1e-6)
            np.testing.assert_allclose(task[:, 1], -shaped[:, 0], atol=1e-6)
            np.testing.assert_array_equal(
                np.asarray(labels["data/task12_intended"]),
                arrays["task12_intended"][:20],
            )

    def test_task12_uses_future_ball_heading_and_repeats_terminal_command(self):
        velocity = np.zeros((30, 2), dtype=np.float32)
        velocity[:, 0] = 1.0
        heading = np.linspace(0.0, np.pi / 2, 30, dtype=np.float32)
        yaw_rate = np.linspace(-0.5, 0.5, 30, dtype=np.float32)
        task = _task12_intended(velocity, heading, yaw_rate).reshape(30, 4, 3)
        future = np.asarray([6, 12, 18, 24])
        np.testing.assert_allclose(
            task[0, :, 0], np.cos(heading[future]), atol=1e-6
        )
        np.testing.assert_allclose(
            task[0, :, 1], -np.sin(heading[future]), atol=1e-6
        )
        np.testing.assert_allclose(task[0, :, 2], yaw_rate[future], atol=1e-7)
        expected_terminal = np.asarray(
            [np.cos(heading[-1]), -np.sin(heading[-1]), yaw_rate[-1]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            task[-1], np.repeat(expected_terminal[None], 4, axis=0), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
