from __future__ import annotations

from contextlib import redirect_stdout
import inspect
from io import StringIO
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from mm_sonic.pfnn_terrain_fit import PFNNTerrainFit
from mm_sonic.hybrid_terrain_lmm_viewer import _scene_adapter_from_grid
from mm_sonic.terrain_pfnn.pfnn_surface import PlacedPFNNSurface
from mm_sonic.terrain_pfnn.runtime import PFNNRuntimeFrame, PFNNTrajectoryState


class TerrainPFNNViewerTests(unittest.TestCase):
    def _sloped_scene_adapter(self, heading: float = math.pi / 2.0) -> object:
        # Native coordinates cover X=[2, 4], Y=[3, 5], with h=2X+3Y.
        return _scene_adapter_from_grid(
            np.asarray(((19.0, 23.0), (13.0, 17.0)), dtype=np.float32),
            origin_x=2.0,
            origin_z=-5.0,
            cell=2.0,
            exterior=-1.0,
            name="two-by-two-slope",
            source_path=None,
            spawn_native_xy=(3.0, 4.0),
            spawn_heading=heading,
        )

    def test_scene_terrain_callback_uses_one_rigid_course_frame_and_native_mesh(
        self,
    ) -> None:
        from mm_sonic.terrain_pfnn_viewer import _ScenePFNNTerrainCallback

        heading = 0.3
        adapter = self._sloped_scene_adapter(heading)
        callback = _ScenePFNNTerrainCallback(adapter)

        origin = callback(np.zeros(2, dtype=np.float64))
        forward = callback(np.asarray((0.25, 0.0), dtype=np.float64))
        self.assertIsNotNone(origin)
        self.assertIsNotNone(forward)
        assert origin is not None and forward is not None
        forward_native = np.asarray(
            (3.0 + 0.25 * math.sin(heading), 4.0 - 0.25 * math.cos(heading))
        )
        self.assertEqual(origin.height_m, adapter.authority.height_at((3.0, 4.0)))
        self.assertEqual(forward.height_m, adapter.authority.height_at(forward_native))
        np.testing.assert_allclose(
            origin.gradient_xy,
            (
                2.0 * math.sin(heading) - 3.0 * math.cos(heading),
                2.0 * math.cos(heading) + 3.0 * math.sin(heading),
            ),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            callback.collision_heights_at(np.asarray(((0.0, 0.0), (0.25, 0.0)))),
            (18.0, adapter.authority.height_at(forward_native)),
            atol=1.0e-12,
        )
        vertices, faces = adapter.native_mesh()
        np.testing.assert_array_equal(callback.vertices, vertices)
        np.testing.assert_array_equal(callback.faces, faces)

    def test_exact_pfnn_surface_drives_runtime_collision_and_render_mesh(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _PFNNTerrainCallback

        fit = PFNNTerrainFit(
            patch=np.arange(16, dtype=np.float64).reshape(4, 4),
            patch_coord=np.zeros(4),
            contact_center_xz=np.zeros(2),
            patch_height_mean=22.5,
            stance_height_mean=100.0,
            rbf_centers_xz=np.array([[-1.0, 0.0], [1.0, 0.0]]),
            rbf_epsilon=np.array([0.5, 0.5]),
            rbf_weights=np.array([[0.25, -0.25]]),
            source_contacts=np.ones((8, 4), dtype=np.bool_),
            source_start_frame=120,
            source_frame_count=8,
            cycle_start_frame=100,
            cycle_stop_frame=200,
            selected_patch_index=0,
            fitting_error=0.0,
            source_sha256="a" * 64,
            patches_sha256="b" * 64,
        )
        surface = PlacedPFNNSurface(fit)
        callback = _PFNNTerrainCallback(
            surface,
            x_samples=np.linspace(-0.25, 0.25, 7),
            y_samples=np.linspace(-0.2, 0.2, 5),
        )
        np.testing.assert_allclose(
            callback.vertices[:, 2],
            callback.collision_heights_at(callback.vertices[:, :2]),
            atol=1.0e-12,
            rtol=0.0,
        )
        sample = callback(np.array([0.0, 0.0]))
        self.assertIsNotNone(sample)
        assert sample is not None
        np.testing.assert_allclose(
            sample.gradient_xy,
            surface.gradient_at(np.array([[0.0, 0.0]]))[0],
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_pfnn_course_has_flat_bootstrap_and_then_the_exact_transferred_surface(
        self,
    ) -> None:
        from mm_sonic.terrain_pfnn_viewer import _PFNNCourseCallback

        fit = PFNNTerrainFit(
            patch=np.arange(16, dtype=np.float64).reshape(4, 4),
            patch_coord=np.zeros(4),
            contact_center_xz=np.array((20.0, -10.0)),
            patch_height_mean=22.5,
            stance_height_mean=100.0,
            rbf_centers_xz=np.array([[-1.0, 0.0], [1.0, 0.0]]),
            rbf_epsilon=np.array([0.5, 0.5]),
            rbf_weights=np.array([[0.25, -0.25]]),
            source_contacts=np.ones((8, 4), dtype=np.bool_),
            source_start_frame=120,
            source_frame_count=8,
            cycle_start_frame=100,
            cycle_stop_frame=200,
            selected_patch_index=0,
            fitting_error=0.0,
            source_sha256="a" * 64,
            patches_sha256="b" * 64,
        )
        surface = PlacedPFNNSurface(fit)
        callback = _PFNNCourseCallback(
            surface,
            x_samples=np.linspace(-1.0, 3.0, 17),
            y_samples=np.linspace(-0.5, 0.5, 9),
        )

        bootstrap = callback(np.array((0.0, 0.0)))
        self.assertIsNotNone(bootstrap)
        assert bootstrap is not None
        self.assertEqual(bootstrap.height_m, 0.0)
        np.testing.assert_array_equal(bootstrap.gradient_xy, np.zeros(2))
        world = np.array((2.0, 0.2))
        source = callback.source_anchor_xy + np.array((1.25, 0.2))
        expected_height = (
            float(surface.height_at(source[None, :])[0]) - callback.source_height_m
        )
        sample = callback(world)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertAlmostEqual(sample.height_m, expected_height)
        np.testing.assert_allclose(
            callback.vertices[:, 2],
            callback.collision_heights_at(callback.vertices[:, :2]),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_viewer_loads_raw_pfnn_runtime(self) -> None:
        import mm_sonic.terrain_pfnn_viewer as viewer_module
        from mm_sonic.terrain_pfnn_viewer import _load_runtime, _parser

        arguments = _parser().parse_args([])
        terrain = object()
        checkpoint = SimpleNamespace(
            dataset_digest="a" * 64,
            kinematic_signature_sha256="b" * 64,
            source_kind="grail",
            vertical_slice_receipt_sha256="0" * 64,
            terrain_receipt_set_sha256="0" * 64,
        )
        kinematics = SimpleNamespace(kinematic_signature_sha256="b" * 64)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                '{"dataset_digest_sha256":"' + ("a" * 64) + '"}',
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    viewer_module, "load_classic_checkpoint", return_value=checkpoint
                ) as loader,
                mock.patch.object(
                    viewer_module.TorchG1ForwardKinematics,
                    "from_mjcf",
                    return_value=kinematics,
                ) as load_kinematics,
                mock.patch.object(
                    viewer_module, "TerrainPFNNRuntime", return_value=object()
                ) as factory,
            ):
                loaded = _load_runtime(
                    arguments,
                    Path("checkpoint.pt"),
                    manifest,
                    Path("g1.xml"),
                    terrain,
                )

        self.assertIsNotNone(loaded)
        loader.assert_called_once_with(Path("checkpoint.pt"))
        load_kinematics.assert_called_once_with(Path("g1.xml"))
        self.assertIs(factory.call_args.kwargs["checkpoint"], checkpoint)
        self.assertIs(factory.call_args.kwargs["kinematics"], kinematics)
        self.assertIs(factory.call_args.kwargs["height_and_grade_at"], terrain)
        self.assertEqual(factory.call_args.kwargs["device"], "cuda")
        self.assertIs(factory.call_args.kwargs["enforce_motion_envelope"], False)
        self.assertIs(factory.call_args.kwargs["command_driven_root"], False)
        self.assertIs(factory.call_args.kwargs["hold_idle_pose"], False)

    def test_vertical_runtime_requires_exact_source_and_terrain_receipts(self) -> None:
        import json
        import mm_sonic.terrain_pfnn_viewer as viewer_module
        from mm_sonic.terrain_pfnn_viewer import _load_runtime, _parser

        arguments = _parser().parse_args([])
        checkpoint = SimpleNamespace(
            dataset_digest="a" * 64,
            kinematic_signature_sha256="b" * 64,
            source_kind="released_pfnn",
            vertical_slice_receipt_sha256="c" * 64,
            terrain_receipt_set_sha256="d" * 64,
        )
        kinematics = SimpleNamespace(kinematic_signature_sha256="b" * 64)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_sha256": "a" * 64,
                        "selection_sha256": "c" * 64,
                        "terrain_receipt_set_sha256": "d" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    viewer_module, "load_classic_checkpoint", return_value=checkpoint
                ),
                mock.patch.object(
                    viewer_module.TorchG1ForwardKinematics,
                    "from_mjcf",
                    return_value=kinematics,
                ),
                mock.patch.object(
                    viewer_module, "TerrainPFNNRuntime", return_value=object()
                ) as factory,
            ):
                _load_runtime(
                    arguments,
                    Path("checkpoint.pt"),
                    manifest,
                    Path("g1.xml"),
                    object(),
                )
        self.assertIs(factory.call_args.kwargs["hold_idle_pose"], False)
        self.assertEqual(factory.call_args.kwargs["maximum_grade_degrees"], 89.0)

    def test_defaults_select_the_classic_g1_artifacts(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _parser

        arguments = _parser().parse_args([])
        self.assertEqual(
            arguments.dataset,
            Path(
                "sonic/runs/native-g1-pfnn/expanded/mixed-corpus-filtered/manifest.json"
            ),
        )
        self.assertEqual(
            arguments.checkpoint,
            Path(
                "sonic/runs/native-g1-pfnn/expanded/"
                "model-mixed-filtered-rollout16-final-v2/best.pt"
            ),
        )
        self.assertEqual(
            arguments.terrain_fit,
            Path(
                "sonic/runs/native-g1-pfnn/expanded/vertical-corpus/terrain/"
                "WalkingUpSteps08_000__01550_01675.npz"
            ),
        )
        self.assertEqual(
            arguments.idle_clips,
            Path(
                "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
                "motionbricks/out/G1-clip.ckpt"
            ),
        )

    def test_scene_options_select_authenticated_terrain_and_reject_explicit_fit(
        self,
    ) -> None:
        from mm_sonic.terrain_pfnn_viewer import _parser, _validate

        arguments = _parser().parse_args(
            [
                "--scene",
                "ramp-10-up-down",
                "--terrain-root",
                "/tmp/terrain",
                "--terrain-fit",
                "/tmp/fit.npz",
            ]
        )
        self.assertEqual(arguments.scene, "ramp-10-up-down")
        self.assertEqual(arguments.terrain_root, Path("/tmp/terrain"))
        with mock.patch.object(Path, "is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "--scene.*--terrain-fit"):
                _validate(arguments)

    def test_scene_validation_ignores_the_absent_default_terrain_fit(self) -> None:
        import mm_sonic.terrain_pfnn_viewer as viewer_module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default_fit = root / "absent-default-fit.npz"
            inputs = tuple(
                root / name
                for name in ("checkpoint", "dataset", "model", "scene", "idle")
            )
            for path in inputs:
                path.touch()
            with mock.patch.object(viewer_module, "DEFAULT_TERRAIN_FIT", default_fit):
                arguments = viewer_module._parser().parse_args(
                    [
                        "--scene",
                        "ramp-10-up-down",
                        "--checkpoint",
                        str(inputs[0]),
                        "--dataset",
                        str(inputs[1]),
                        "--model-path",
                        str(inputs[2]),
                        "--scene-xml",
                        str(inputs[3]),
                        "--idle-clips",
                        str(inputs[4]),
                    ]
                )
                self.assertFalse(arguments.terrain_fit.is_file())
                self.assertEqual(viewer_module._validate(arguments), inputs)

    def test_motionbricks_native_g1_idle_pose_is_loaded_safely(self) -> None:
        import torch
        from mm_sonic.terrain_pfnn_viewer import _motionbricks_idle_mujoco_qpos

        qpos = torch.zeros((2, 3, 36), dtype=torch.float32)
        qpos[..., 2] = 0.8
        qpos[..., 3] = 1.0
        qpos[..., 7:] = torch.arange(29, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "G1-clip.ckpt"
            torch.save(
                {
                    "mujoco_qpos": qpos,
                    "num_frames_per_clip": torch.tensor((3, 2), dtype=torch.int32),
                },
                path,
            )
            loaded = _motionbricks_idle_mujoco_qpos(path)

        np.testing.assert_array_equal(loaded, qpos[0, 0].numpy())
        self.assertEqual(loaded.dtype, np.float64)

    def test_help_and_interactive_defaults_do_not_load_runtime(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _parser, main

        arguments = _parser().parse_args([])
        self.assertGreaterEqual(arguments.max_steps, 1_000_000)
        self.assertEqual(arguments.device, "cuda")
        self.assertFalse(hasattr(arguments, "strict_envelope"))

        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--checkpoint", output.getvalue())
        self.assertIn("--no-viewer", output.getvalue())
        self.assertNotIn("--strict-envelope", output.getvalue())

    def test_wasd_command_is_bounded_and_release_stops(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _command_from_pressed

        self.assertTrue(np.array_equal(_command_from_pressed(set(), 0.8), (0.0, 0.0)))
        self.assertTrue(np.array_equal(_command_from_pressed({"w"}, 0.8), (0.8, 0.0)))
        self.assertTrue(np.array_equal(_command_from_pressed({"s"}, 0.8), (-0.8, 0.0)))
        diagonal = _command_from_pressed({"w", "a"}, 0.8)
        self.assertAlmostEqual(float(np.linalg.norm(diagonal)), 0.8, places=12)

    def test_runtime_callback_uses_the_rendered_triangle_height_and_grade(self) -> None:
        from mm_sonic.terrain_pfnn.hill_map import TerrainPFNNHillMap
        from mm_sonic.terrain_pfnn_viewer import _HillTerrainCallback

        terrain = TerrainPFNNHillMap()
        callback = _HillTerrainCallback(terrain)
        hill = terrain.hills[-1]
        point = np.asarray((hill.start_x + terrain.blend_m + 0.5, 0.0))
        sample = callback(point)

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.height_m, terrain.height_at(point))
        measured = math.degrees(math.atan(float(sample.gradient_xy[0])))
        self.assertAlmostEqual(measured, 18.9, delta=0.05)
        self.assertIsNone(callback((terrain.x_max + 0.1, 0.0)))

    def test_default_viewer_course_has_lateral_recovery_room(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _viewer_terrain_map

        terrain = _viewer_terrain_map()
        self.assertGreaterEqual(terrain.y_max, 10.0)
        x = terrain.hills[1].start_x + terrain.blend_m + 0.25
        self.assertEqual(terrain.height_at((x, -9.0)), terrain.height_at((x, 9.0)))

    def test_camera_defaults_show_the_longitudinal_hill_profile(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _configure_camera

        viewer = SimpleNamespace(
            cam=SimpleNamespace(azimuth=0.0, elevation=0.0, distance=0.0)
        )
        _configure_camera(viewer)
        self.assertEqual(viewer.cam.azimuth, 90.0)
        self.assertEqual(viewer.cam.elevation, -18.0)
        self.assertEqual(viewer.cam.distance, 4.0)

    def test_exact_frame_preserves_full_quaternion_and_all_pfnn_joints(self) -> None:
        from mm_sonic.gear_action import mujoco_to_isaaclab_joint_vector
        from mm_sonic.terrain_pfnn_viewer import _apply_frame

        model = SimpleNamespace(qpos0=np.arange(36, dtype=np.float64))
        data = SimpleNamespace(qpos=np.zeros(36, dtype=np.float64))
        roll, pitch, yaw = 0.3, -0.2, 0.4
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        frame_quaternion = np.asarray(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            )
        )
        frame = PFNNRuntimeFrame(
            root_position_world=np.asarray((1.0, 2.0, 0.8)),
            root_quaternion_world_wxyz=frame_quaternion,
            joint_position_isaaclab=np.arange(29, dtype=np.float64) / 10.0,
            phase=0.0,
            contact_probability=np.zeros(4),
            trajectory=PFNNTrajectoryState(
                position_world_xy=np.zeros((12, 2)),
                direction_world_xy=np.tile((1.0, 0.0), (12, 1)),
                semantic_intent=np.tile((1.0, 0.0), (12, 1)),
            ),
            supported=True,
            diagnostics={},
        )
        adapter = self._sloped_scene_adapter(math.pi)

        _apply_frame(model, data, frame, scene_terrain=adapter)
        np.testing.assert_allclose(data.qpos[:3], (1.0, 5.0, 0.8), atol=1.0e-12)
        scene_yaw = np.asarray(
            (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
        )
        expected_quaternion = np.asarray(
            (
                scene_yaw[0] * frame_quaternion[0] - scene_yaw[3] * frame_quaternion[3],
                scene_yaw[0] * frame_quaternion[1] - scene_yaw[3] * frame_quaternion[2],
                scene_yaw[0] * frame_quaternion[2] + scene_yaw[3] * frame_quaternion[1],
                scene_yaw[0] * frame_quaternion[3] + scene_yaw[3] * frame_quaternion[0],
            )
        )
        expected_quaternion /= np.linalg.norm(expected_quaternion)
        np.testing.assert_allclose(
            data.qpos[3:7], expected_quaternion, atol=1.0e-12, rtol=0.0
        )
        np.testing.assert_array_equal(
            mujoco_to_isaaclab_joint_vector(data.qpos[7:36]),
            frame.joint_position_isaaclab,
        )

    def test_advance_has_no_display_joint_override_or_blend(self) -> None:
        import mm_sonic.terrain_pfnn_viewer as viewer_module

        source = inspect.getsource(viewer_module._run)
        self.assertNotIn("_upright_yaw_quaternion(", source)
        self.assertNotIn("_blend_display_joints(", source)
        self.assertNotIn("display_joints_mujoco=", source)

    def test_interactive_camera_tracks_the_scene_placed_root(self) -> None:
        import mm_sonic.terrain_pfnn_viewer as viewer_module

        source = inspect.getsource(viewer_module._run)
        self.assertIn(
            "viewer.cam.lookat[:] = _scene_root_position(\n"
            "                    frame.root_position_world, scene_terrain\n"
            "                )",
            source,
        )


if __name__ == "__main__":
    unittest.main()
