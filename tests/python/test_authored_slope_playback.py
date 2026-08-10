import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from sonic.python.mm_sonic.authored_slope_playback import (
    AUTHORED_PLAYBACK_LABEL,
    AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
    AuthoredPlaybackKeys,
    AuthoredPlaybackState,
    authored_playback_smoke,
    build_playback_model,
    enforce_solid_rendering,
    holden_vertices_to_native,
    load_authored_slope_bundle,
    main,
    overlay_text,
)


class AuthoredSlopePlaybackStateTests(unittest.TestCase):
    def test_release_pauses_the_exact_rendered_row_bitwise(self):
        qpos = np.arange(5 * 36, dtype=np.float32).reshape(5, 36)
        state = AuthoredPlaybackState(qpos)

        self.assertEqual(state.row, 0)
        state.tick(w_down=True)
        self.assertEqual(state.row, 1)
        held = state.current_qpos().tobytes()

        for _ in range(8):
            state.tick(w_down=False)

        self.assertEqual(state.row, 1)
        self.assertEqual(state.current_qpos().tobytes(), held)

    def test_w_advances_one_row_and_terminal_w_holds(self):
        qpos = np.arange(3 * 36, dtype=np.float32).reshape(3, 36)
        state = AuthoredPlaybackState(qpos)

        self.assertEqual(state.tick(w_down=True), 1)
        self.assertEqual(state.tick(w_down=True), 2)
        terminal = hashlib.sha256(state.current_qpos().tobytes()).hexdigest()
        self.assertEqual(state.tick(w_down=True), 2)
        self.assertEqual(
            hashlib.sha256(state.current_qpos().tobytes()).hexdigest(), terminal
        )

    def test_playback_is_unambiguously_labeled_non_learned(self):
        self.assertEqual(
            AUTHORED_PLAYBACK_LABEL,
            "AUTHORED SOURCE PLAYBACK (NOT LEARNED)",
        )

    def test_listener_state_tracks_w_press_and_release_levels(self):
        keys = AuthoredPlaybackKeys()

        keys.press("w")
        self.assertEqual(keys.snapshot(), (True, False))
        keys.release("w")
        self.assertEqual(keys.snapshot(), (False, False))

    def test_listener_state_rejects_other_commands_and_x_stops(self):
        keys = AuthoredPlaybackKeys()

        for character in ("a", "s", "d", "q"):
            keys.press(character)
        self.assertEqual(keys.snapshot(), (False, False))
        keys.press("x")
        self.assertEqual(keys.snapshot(), (False, True))

class AuthoredSlopeTerrainConversionTests(unittest.TestCase):
    def test_holden_terrain_vertices_share_the_inverse_z_up_basis(self):
        holden = np.asarray(
            ((1.0, 2.0, 3.0), (-4.0, -5.0, -6.0)), dtype=np.float64
        )

        native = holden_vertices_to_native(holden)

        np.testing.assert_array_equal(
            native,
            np.asarray(((1.0, -3.0, 2.0), (-4.0, 6.0, -5.0))),
        )


class AuthenticatedAuthoredSlopeBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_authored_slope_bundle()

    def test_exact_pair_becomes_the_595_admitted_60_hz_rows(self):
        self.assertEqual(self.bundle.clip_id, "terrain_slopes__slope_000__000")
        self.assertEqual(self.bundle.fps, 60.0)
        self.assertEqual(self.bundle.qpos.shape, (595, 36))
        self.assertFalse(self.bundle.qpos.flags.writeable)
        self.assertEqual(
            hashlib.sha256(self.bundle.qpos.astype("<f4").tobytes()).hexdigest(),
            "b2abecf5e0423708b6b9330347aab051502158a965e9fb0fa2d7873b9d8ee4be",
        )

    def test_receipt_binds_all_authenticated_inputs_and_terrain_calibration(self):
        self.assertEqual(
            self.bundle.hashes,
            {
                "robot_sha256": "b77480d5f8f3339a3064276d6f9d443ac3a3456f20eb9195d48add176e561ee1",
                "usd_sha256": "8d1e696fb5bd2aecfa17797549bddd001093b060773cb313185a6a46db7eb5a5",
                "reconstruction_sha256": "d05d6c5a7d6a13eff7e69da0e9e96ab5a79f700bb706611699f0b9c44c9b51ec",
                "metadata_sha256": "7d88be608419afd2be127e4ac4c5a8aa1b4863fdf84e86c5ec93356881ee861d",
                "g1_xml_sha256": "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
            },
        )
        self.assertEqual(
            self.bundle.support_calibration_m,
            AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
        )
        self.assertEqual(
            self.bundle.exterior_height_native_m,
            -AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
        )
        self.assertEqual(self.bundle.terrain_vertices_native.shape, (48, 3))
        self.assertEqual(self.bundle.terrain_faces.shape, (24, 3))

    def test_model_renders_exact_mesh_over_explicit_exterior_flat(self):
        import mujoco

        model = build_playback_model(self.bundle)
        terrain_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "authored_slope_exact_mesh"
        )
        exterior_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "authored_slope_exterior_flat"
        )
        original_floor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.assertGreaterEqual(terrain_id, 0)
        self.assertGreaterEqual(exterior_id, 0)
        self.assertEqual(
            float(model.geom_pos[exterior_id, 2]),
            -AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
        )
        self.assertEqual(float(model.geom_rgba[original_floor_id, 3]), 0.0)

    def test_model_uses_captured_xml_and_assets_after_source_path_mutation(self):
        canonical = Path(
            "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied_xml = root / "g1_29dof.xml"
            copied_xml.write_bytes(canonical.read_bytes())
            mesh_link = root / "meshes"
            os.symlink(canonical.parent / "meshes", mesh_link, target_is_directory=True)
            bundle = load_authored_slope_bundle(g1_xml=copied_xml)

            copied_xml.write_bytes(b"<mujoco model='mutated-after-load'/>")
            mesh_link.unlink()
            model = build_playback_model(bundle)

        self.assertEqual(model.nq, 36)

    def test_w_cannot_leave_the_mujoco_scene_in_wireframe_mode(self):
        import mujoco

        flags = np.ones(32, dtype=np.uint8)
        enforce_solid_rendering(flags)

        self.assertEqual(
            int(flags[int(mujoco.mjtRndFlag.mjRND_WIREFRAME)]), 0
        )

    def test_headless_smoke_proves_pause_and_terminal_are_bitwise_holds(self):
        receipt = authored_playback_smoke(self.bundle)

        self.assertEqual(receipt["label"], AUTHORED_PLAYBACK_LABEL)
        self.assertEqual(receipt["status"], "accepted")
        self.assertEqual(receipt["frame_count"], 595)
        self.assertEqual(receipt["advance_ticks"], 594)
        self.assertTrue(receipt["release_pause_bitwise"])
        self.assertTrue(receipt["terminal_hold_bitwise"])
        self.assertEqual(receipt["terminal_row"], 594)

    def test_overlay_keeps_the_non_learned_label_and_key_level_visible(self):
        title, body = overlay_text(row=18, frame_count=595, focused=True, w_down=False)

        self.assertEqual(title, AUTHORED_PLAYBACK_LABEL)
        self.assertIn("row 19/595", body)
        self.assertIn("PAUSED (W released)", body)
        self.assertIn("12.000 mm terrain-only calibration", body)

    def test_headless_cli_prints_an_accepted_machine_readable_receipt(self):
        output = io.StringIO()

        with redirect_stdout(output):
            result = main(["--headless-smoke"])

        receipt = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(receipt["status"], "accepted")
        self.assertEqual(receipt["label"], AUTHORED_PLAYBACK_LABEL)


if __name__ == "__main__":
    unittest.main()
