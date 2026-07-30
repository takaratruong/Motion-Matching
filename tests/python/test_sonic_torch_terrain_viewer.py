import hashlib
import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import mm_sonic.torch_terrain_viewer as viewer_module
from mm_sonic.torch_terrain_rollout import (
    build_rollout_argument_parser,
    load_experiment_config,
    resolve_stair_config,
    run_stair_rollout,
    save_stair_rollout,
)
from mm_sonic.torch_terrain_viewer import (
    PlaybackController,
    build_viewer_argument_parser,
    load_saved_rollout,
    render_saved_frame,
    save_saved_video,
)
from resources.g1_torch_stair_builder.publish import publish_stair_slice
from tests.python.test_sonic_torch_terrain_rollout import (
    CONFIG_PATH,
    _wide_curved_grid,
)
from tests.python.test_torch_stair_conversion import (
    _FakeKinematics,
    write_synthetic_pinned_corpus,
)
from tests.python.torch_motion_test_utils import write_takara_clip


def _minimal_g1_xml(path: Path) -> None:
    opening = [
        '<mujoco model="viewer-test">',
        "<option gravity=\"0 0 0\"/>",
        "<worldbody>",
        '<body name="pelvis" pos="0 0 0">',
        "<freejoint/>",
        '<geom type="sphere" size="0.03" mass="1"/>',
    ]
    closing = []
    for index in range(29):
        opening.extend(
            [
                f'<body name="link_{index}" pos="0.03 0 0">',
                f'<joint name="joint_{index}" type="hinge" axis="0 0 1"/>',
                '<geom type="capsule" fromto="0 0 0 0.03 0 0" '
                'size="0.005" mass="0.01"/>',
            ]
        )
        closing.append("</body>")
    payload = "\n".join(
        opening + list(reversed(closing)) + ["</body>", "</worldbody>", "</mujoco>"]
    )
    path.write_text(payload, encoding="utf-8")


class TerrainViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        grail = cls.root / "grail"
        write_synthetic_pinned_corpus(grail)
        flat_motion = write_takara_clip(cls.root / "flat", frames=80)
        g1_xml = cls.root / "builder.xml"
        g1_xml.write_text("<mujoco/>", encoding="utf-8")
        cls.dataset_root = cls.root / "dataset"
        publish_stair_slice(
            output=cls.dataset_root,
            grail_root=grail,
            g1_xml=g1_xml,
            flat_motion=flat_motion,
            kinematics=_FakeKinematics(),
            grid_builder=_wide_curved_grid,
        )
        config = load_experiment_config(CONFIG_PATH)
        config["duration_s"] = 0.10
        resolved = resolve_stair_config(
            cls.dataset_root, config, device="cpu"
        )
        rollout = run_stair_rollout(resolved, "dense", device="cpu")
        cls.run_root = cls.root / "run"
        save_stair_rollout(rollout, cls.run_root)
        cls.g1_xml = cls.root / "viewer.xml"
        _minimal_g1_xml(cls.g1_xml)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_loader_validates_rollout_and_grid_hashes_before_rendering(self):
        saved = load_saved_rollout(self.run_root)
        self.assertEqual(saved.frame_count, 5)
        self.assertEqual(saved.arrays["joint_position"].shape, (5, 29))
        self.assertEqual(saved.arrays["terrain_patch_position_world"].shape, (5, 91, 3))
        self.assertEqual(len(saved.terrain_sha256), 64)

        corrupt = self.root / "corrupt"
        shutil.copytree(self.run_root, corrupt)
        archive = corrupt / "rollout.npz"
        payload = bytearray(archive.read_bytes())
        payload[len(payload) // 2] ^= 0x01
        archive.write_bytes(payload)
        with self.assertRaisesRegex(Exception, "hash"):
            load_saved_rollout(corrupt)

    def test_loader_defaults_missing_safety_override_for_v1_rollout(self):
        legacy = self.root / "legacy-without-safety-override"
        shutil.copytree(self.run_root, legacy)
        archive_path = legacy / "rollout.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
                if name != "terrain_safety_override"
            }
        np.savez(archive_path, **arrays)
        metrics_path = legacy / "metrics.json"
        metrics = json.loads(metrics_path.read_text("utf-8"))
        metrics["rollout_npz_sha256"] = hashlib.sha256(
            archive_path.read_bytes()
        ).hexdigest()
        metrics_path.write_text(
            json.dumps(
                metrics,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        saved = load_saved_rollout(legacy)

        self.assertIn("terrain_safety_override", saved.arrays)
        self.assertEqual(
            saved.arrays["terrain_safety_override"].dtype,
            np.dtype(np.bool_),
        )
        self.assertFalse(saved.arrays["terrain_safety_override"].any())
        self.assertIn("terrain_safety_override_rank", saved.arrays)
        self.assertEqual(
            saved.arrays["terrain_safety_override_rank"].dtype,
            np.dtype(np.int32),
        )
        self.assertFalse(saved.arrays["terrain_safety_override_rank"].any())

    def test_playback_controls_only_move_through_saved_frames(self):
        playback = PlaybackController(frame_count=3)
        self.assertEqual(playback.frame_index, 0)
        playback.handle_key("right")
        self.assertEqual(playback.frame_index, 1)
        playback.handle_key("right")
        playback.handle_key("right")
        self.assertEqual(playback.frame_index, 2)
        playback.handle_key("left")
        self.assertEqual(playback.frame_index, 1)
        playback.handle_key(" ")
        self.assertTrue(playback.paused)
        playback.handle_key("r")
        self.assertEqual(playback.frame_index, 0)
        playback.handle_key("escape")
        self.assertTrue(playback.closed)

    def test_viewer_has_no_matcher_or_physics_step_dependency(self):
        source = inspect.getsource(viewer_module)
        self.assertNotIn("torch_motion_matcher", source)
        self.assertNotIn("mj_step(", source)
        self.assertIn("--dataset", build_rollout_argument_parser().format_help())
        viewer_help = build_viewer_argument_parser().format_help()
        self.assertIn("--run", viewer_help)
        self.assertIn("--output-mp4", viewer_help)

    def test_headless_frame_contains_saved_skeleton_terrain_and_diagnostics(self):
        saved = load_saved_rollout(self.run_root)
        output = self.root / "frame.png"
        render_saved_frame(
            saved,
            g1_xml=self.g1_xml,
            frame_index=0,
            output_png=output,
        )
        self.assertTrue(output.is_file())
        self.assertGreater(output.stat().st_size, 10_000)
        signature = output.read_bytes()[:8]
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")
        self.assertTrue(np.isfinite(saved.arrays["foot_clearance_m"]).all())

        video = self.root / "rollout.mp4"
        save_saved_video(
            saved, g1_xml=self.g1_xml, output_mp4=video, fps=50
        )
        self.assertTrue(video.is_file())
        self.assertGreater(video.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
