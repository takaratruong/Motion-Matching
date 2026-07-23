from pathlib import Path
import unittest


SOURCE_PATH = Path("g1_interaction_episode_viewer.cpp")


class InteractionEpisodeViewerContractTest(unittest.TestCase):
    def source(self) -> str:
        return SOURCE_PATH.read_text(encoding="utf-8")

    def test_viewer_uses_flat_episode_runtime(self) -> None:
        source = self.source()
        self.assertIn('#include "interaction_episode.h"', source)
        self.assertNotIn("G1_TERRAIN_DIR", source)
        self.assertNotIn("table-ground-pack", source)

    def test_viewer_uses_episode_pack_g1_walking_and_one_pose(self) -> None:
        source = self.source()
        self.assertIn(
            'options.episode_pack / "walking_database.bin"', source
        )
        self.assertNotIn('"resources/database.bin"', source)
        self.assertNotIn("draw_flat_bones(", source)
        self.assertNotIn("flat_visible", source)

    def test_controls_and_search_are_edge_triggered(self) -> None:
        source = self.source()
        self.assertIn("IsKeyPressed(KEY_F)", source)
        self.assertIn("IsKeyDown(KEY_W)", source)
        self.assertIn("IsKeyDown(KEY_A)", source)
        self.assertIn("IsKeyDown(KEY_S)", source)
        self.assertIn("IsKeyDown(KEY_D)", source)
        self.assertIn("std::future<reach::SearchResult>", source)

    def test_viewer_has_no_terrain_or_production_controller_dependency(
        self,
    ) -> None:
        source = self.source()
        self.assertNotIn('#include "terrain_', source)
        self.assertNotIn('#include "controller', source)
        self.assertIn("resources/g1_mesh/g1_raylib.glb", source)

    def test_viewer_defaults_to_bones_and_full_parallel_search(self) -> None:
        source = self.source()
        self.assertIn("bool show_mesh = false;", source)
        self.assertIn("bool show_bones = true;", source)
        self.assertIn("reach::search_all(", source)
        self.assertNotIn("search_candidates(", source)
        self.assertIn("8U,", source)
        self.assertIn("std::thread::hardware_concurrency()", source)
        self.assertIn("search.has_value() ? 0.0F : dt", source)

    def test_viewer_has_contextual_same_hand_placement(self) -> None:
        source = self.source()
        self.assertIn("SearchPurpose::Place", source)
        self.assertIn("runtime.commit_place(", source)
        self.assertIn("job.required_hand", source)
        self.assertIn("destination_generation", source)
        self.assertIn("destination_support_index", source)
        self.assertIn("std::swap(", source)

    def test_viewer_displays_recorded_return_without_carry_ik(self) -> None:
        source = self.source()
        self.assertIn('case episode::EpisodeState::Return: return "RETURN";', source)
        self.assertNotIn("CARRY IK", source)

    def test_viewer_names_post_return_state_hold(self) -> None:
        source = self.source()
        self.assertIn(
            'case episode::EpisodeState::Neutral: return "HOLD";', source
        )
        self.assertNotIn("CARRY MOTION MATCHING", source)
        self.assertNotIn("WALKING CARRY", source)


if __name__ == "__main__":
    unittest.main()
