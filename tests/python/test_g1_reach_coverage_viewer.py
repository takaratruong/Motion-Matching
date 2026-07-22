import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
VIEWER = ROOT / "g1_reach_coverage_viewer.cpp"
PROBE = ROOT / "g1_reach_coverage_probe.cpp"
MAKEFILE = ROOT / "Makefile"


class G1ReachCoverageViewerTests(unittest.TestCase):
    def source(self, path):
        self.assertTrue(path.exists(), f"missing {path.name}")
        return path.read_text(encoding="utf-8")

    def test_flat_viewer_contract(self):
        source = self.source(VIEWER)
        for required in (
            "reach::load_pack(", "reach::select_candidates(",
            "reach::evaluate_candidate(", "KEY_ENTER", "KEY_SLASH",
            "KEY_LEFT_BRACKET", "KEY_RIGHT_BRACKET", "KEY_M", "KEY_G",
            "KEY_Q", "KEY_E", "KEY_R", "KEY_F", "KEY_Z", "KEY_C",
            "KEY_LEFT", "KEY_RIGHT", "KEY_UP", "KEY_DOWN", "KEY_W",
            "KEY_S", "KEY_V", "KEY_BACKSPACE", "DrawCylinderEx(",
            "DrawLine3D(", "CAPTURED LEFT", "MIRRORED RIGHT",
            "POSITION ERROR", "APPROACH AXIS", "FULL ORIENTATION",
            "OBJECT COLLISION", "ENVIRONMENT COLLISION",
            "SEARCH STALE - press Enter", "--object-size",
            "results.options", "selected_evaluation",
            "REJECTED - motion shown for diagnosis",
            "ClearBackground(Color{238, 241, 245, 255})",
            "DrawSphereWires(", "DARKBLUE", "SKYBLUE", "LIME",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "g1_mesh_renderer", "terrain_runtime", "TakeScreenshot(",
            "LoadModel(", "controller.cpp",
        ):
            self.assertNotIn(forbidden, source)

    def test_probe_contract(self):
        source = self.source(PROBE)
        for required in (
            "reach::load_pack(", "reach::shape_candidate(", "--json",
            "zero_retarget", "position_perturbations",
            "orientation_perturbations", "augmentation", "height_band",
            "direction_band", "source", "union", "observed_collisions",
        ):
            self.assertIn(required, source)
        self.assertIn("0.001F", source)
        self.assertIn("0.008726646F", source)
        self.assertIn("std::atomic<size_t> next_clip", source)
        self.assertIn("worker_count", source)
        self.assertIn("workers.emplace_back", source)

    def test_makefile_has_standalone_targets(self):
        makefile = self.source(MAKEFILE)
        self.assertIn("g1_reach_coverage_viewer:", makefile)
        self.assertIn("g1_reach_coverage_probe:", makefile)
        self.assertIn("reach_coverage.cpp", makefile)
        self.assertIn("reach_database.cpp", makefile)
        probe_target = makefile.split("g1_reach_coverage_probe:", 1)[1].split(
            "\n\n", 1
        )[0]
        self.assertIn("-pthread", probe_target)


if __name__ == "__main__":
    unittest.main()
