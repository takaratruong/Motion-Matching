import pathlib
import re
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]


def _source(name):
    return (REPOSITORY / name).read_text(encoding="utf-8")


def _without_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


class G1BankedControllerIntegrationTests(unittest.TestCase):
    def test_controller_builds_and_snapshots_exactly_one_39d_query(self):
        source = _without_comments(_source("controller.cpp"))
        self.assertIn("motion_match_query_is_finite_39d(query)", source)
        self.assertIn("terrain_descriptor_sample_v2", source)
        self.assertRegex(
            source,
            r"for\s*\([^;]*terrain_feature[^;]*;[^;]*"
            r"TERRAIN_DESCRIPTOR_VALUE_COUNT")
        self.assertRegex(source, r"query\s*\(\s*offset\+\+\s*\)")
        self.assertIn("char query_bits_hex[39 * 8 + 1]", source)
        self.assertNotIn("motion_match_query_is_finite_31d", source)
        self.assertNotIn("31 * 8 + 1", source)

    def test_g1_path_has_one_indexed_search_and_no_global_fallback(self):
        source = _without_comments(_source("controller.cpp"))
        self.assertEqual(source.count("database_search_indexed("), 1)
        self.assertIsNone(re.search(r"\bdatabase_search\s*\(", source))
        for contract in (
                "motion_bank_for_family",
                "motion_bank_state_observe",
                "direction_mask",
                "speed_mask",
                "elevation_mode",
                "DATABASE_INDEXED_SEARCH_EMPTY",
                "empty_compatible_set"):
            self.assertIn(contract, source)

    def test_compatible_alternative_requires_user_authored_switch_margin(self):
        source = _without_comments(_source("controller.cpp"))
        self.assertIn(
            "g1_motion_match_candidate_beats_continuation(", source)
        self.assertRegex(
            source,
            r"g1_motion_match_candidate_beats_continuation\s*\(\s*"
            r"incumbent_compatible\s*,\s*indexed_search_result\.cost\s*,\s*"
            r"state\.incumbent_cost\s*\)")
        self.assertRegex(
            source,
            r"selected_database_frame\s*=\s*prior_index")
        self.assertRegex(
            source,
            r"state\.selected_cost\s*=\s*state\.incumbent_cost")

    def test_incumbent_and_candidate_publishability_are_explicit(self):
        source = _without_comments(_source("controller.cpp"))
        incumbent = re.search(
            r"const bool incumbent_compatible\s*=\s*(?P<body>.*?);",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(incumbent)
        self.assertIn(
            "database_indexed_frame_is_publishable(",
            incumbent.group("body"),
        )
        self.assertIn(
            "G1_MOTION_MATCH_MINIMUM_FUTURE_PUBLISHED_FRAMES",
            source,
        )

    def test_empty_compatible_search_uses_checked_no_ik_safe_stop(self):
        source = _without_comments(_source("controller.cpp"))
        empty_branch = re.search(
            r"DATABASE_INDEXED_SEARCH_EMPTY(?P<body>.{0,5000})",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(empty_branch)
        body = empty_branch.group("body")
        self.assertIn("empty_compatible_set", body)
        self.assertRegex(body, r"force_search\s*=\s*true")
        self.assertRegex(body, r"applied_velocity\s*=\s*vec3\s*\(")
        self.assertIn("skip_frame_advance", body)
        self.assertNotIn("database_search(", body)
        self.assertNotIn("g1_ik_apply", body)

    def test_heading_controls_mesh_and_classic_ui_are_preserved_with_ik_off(self):
        source = _without_comments(_source("controller.cpp"))
        for contract in (
                '"resources/g1_mesh/g1_raylib.glb"',
                '"WASD / left stick - move"',
                '"Arrows / right stick - camera"',
                '"Left trigger - strafe"',
                "static constexpr bool ik_enabled = false",
                "static constexpr bool lmm_enabled = false",
                "(void)desired_velocity",
                "desired_rotation_curr = desired_rotation"):
            self.assertIn(contract, source)
        self.assertIsNone(re.search(
            r"desired_(?:rotation|heading)\s*=\s*"
            r"[^;]*(?:desired_)?velocity",
            source,
        ))

    def test_database_and_log_headers_publish_39d_banked_contracts(self):
        database = _source("database.h")
        log = _source("motion_match_log.h")
        self.assertIn("database_search_indexed", database)
        self.assertIn("eligible_frame_count", database)
        self.assertIn("evaluated_frame_count", database)
        self.assertIn("motion_match_query_is_finite_39d", log)
        for contract in (
                "terrain[12]", "terrain_center_points[8]",
                "terrain_left_points[4]", "terrain_right_points[4]",
                "requested_family", "active_family", "source_family",
                "direction_mask", "speed_mask", "elevation_mode",
                "classifier_confidence", "bank_transition_reason",
                "eligible_frame_count", "evaluated_frame_count",
                "considered_bound_count", "skipped_bound_count",
                "empty_compatible_set"):
            self.assertIn(contract, log)


if __name__ == "__main__":
    unittest.main()
