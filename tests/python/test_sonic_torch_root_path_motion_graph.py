import unittest

from mm_sonic.joints import ContractError
from mm_sonic.torch_root_path_motion_graph import (
    OrderedLevelMotionCandidate,
    RootPathMotionCandidate,
    RootPathMotionTransition,
    root_path_search_anchors,
    root_path_segment_starts,
    select_ordered_level_motion_chain,
    select_root_path_motion_chain,
)


class RootPathMotionGraphTests(unittest.TestCase):
    def candidate(self, candidate_id, start, stop, cost=0.0):
        return RootPathMotionCandidate(
            candidate_id=candidate_id,
            covered_start_m=start,
            covered_stop_m=stop,
            placement_cost=cost,
            frame_count=100,
        )

    def test_selects_connected_chain_covering_the_root_path(self):
        candidates = (
            self.candidate("a", 0.0, 0.9, 0.2),
            self.candidate("b", 0.7, 1.6, 0.3),
            self.candidate("c", 1.4, 2.1, 0.1),
        )
        transitions = (
            RootPathMotionTransition("a", "b", 0.1),
            RootPathMotionTransition("b", "c", 0.2),
        )

        result = select_root_path_motion_chain(
            candidates=candidates,
            transitions=transitions,
            path_length_m=2.0,
        )

        self.assertEqual(result.candidate_ids, ("a", "b", "c"))
        self.assertAlmostEqual(result.covered_stop_m, 2.1)
        self.assertAlmostEqual(result.total_cost, 0.9)

    def test_ordered_selector_rejects_skipped_level_and_requires_exit(self):
        start = self.candidate("start", 0.0, 0.8, 0.1)
        skipped = self.candidate("skipped", 0.7, 1.5, 0.0)
        middle = self.candidate("middle", 0.7, 1.5, 0.2)
        finish = self.candidate("finish", 1.4, 2.1, 0.1)
        candidates = (
            OrderedLevelMotionCandidate(
                start, (0, 0), (0, 1), (10, 70)
            ),
            OrderedLevelMotionCandidate(
                skipped, (0, 2), (0, 1), (10, 70)
            ),
            OrderedLevelMotionCandidate(
                middle, (0, 1, 2), (0, 1, 0), (10, 40, 70)
            ),
            OrderedLevelMotionCandidate(
                finish,
                (2, 3, 3),
                (0, 1, 0),
                (10, 50, 80),
                terminal_double_support=True,
            ),
        )
        transitions = (
            RootPathMotionTransition("start", "skipped", 0.0, 90, 0),
            RootPathMotionTransition("skipped", "finish", 0.0, 90, 0),
            RootPathMotionTransition("start", "middle", 0.1, 90, 0),
            RootPathMotionTransition("middle", "finish", 0.1, 90, 0),
        )

        result = select_ordered_level_motion_chain(
            candidates=candidates,
            transitions=transitions,
            path_length_m=2.0,
            required_level_count=4,
        )

        self.assertEqual(
            result.candidate_ids, ("start", "middle", "finish")
        )

        with self.assertRaisesRegex(
            ContractError, "no complete ordered-level root-path chain"
        ):
            select_ordered_level_motion_chain(
                candidates=candidates[:-1],
                transitions=transitions[:1],
                path_length_m=1.4,
                required_level_count=4,
            )

    def test_rejects_a_geometric_gap_even_when_transition_is_supplied(self):
        candidates = (
            self.candidate("a", 0.0, 0.7),
            self.candidate("b", 0.9, 2.0),
        )

        with self.assertRaisesRegex(ContractError, "no complete root-path chain"):
            select_root_path_motion_chain(
                candidates=candidates,
                transitions=(RootPathMotionTransition("a", "b", 0.0),),
                path_length_m=2.0,
                maximum_coverage_gap_m=0.05,
            )

    def test_prefers_transition_compatible_route_over_cheaper_nodes(self):
        candidates = (
            self.candidate("start", 0.0, 0.8),
            self.candidate("bad", 0.7, 1.5, 0.0),
            self.candidate("good", 0.7, 1.5, 0.3),
            self.candidate("finish", 1.4, 2.1),
        )
        transitions = (
            RootPathMotionTransition("start", "bad", 4.0),
            RootPathMotionTransition("bad", "finish", 4.0),
            RootPathMotionTransition("start", "good", 0.1),
            RootPathMotionTransition("good", "finish", 0.1),
        )

        result = select_root_path_motion_chain(
            candidates=candidates,
            transitions=transitions,
            path_length_m=2.0,
        )

        self.assertEqual(
            result.candidate_ids, ("start", "good", "finish")
        )

    def test_search_prefers_coherent_chain_over_many_equal_cost_fragments(self):
        candidates = (
            self.candidate("long-a", 0.0, 1.1, 0.1),
            self.candidate("long-b", 1.0, 2.0, 0.1),
            self.candidate("s1", 0.0, 0.6, 0.05),
            self.candidate("s2", 0.5, 1.1, 0.05),
            self.candidate("s3", 1.0, 1.6, 0.05),
            self.candidate("s4", 1.5, 2.0, 0.05),
        )
        transitions = (
            RootPathMotionTransition("long-a", "long-b", 0.05),
            RootPathMotionTransition("s1", "s2", 0.05),
            RootPathMotionTransition("s2", "s3", 0.05),
            RootPathMotionTransition("s3", "s4", 0.05),
        )

        result = select_root_path_motion_chain(
            candidates=candidates,
            transitions=transitions,
            path_length_m=2.0,
            fragmentation_cost=0.05,
        )

        self.assertEqual(result.candidate_ids, ("long-a", "long-b"))

    def test_rejects_backward_or_nonadvancing_candidates(self):
        with self.assertRaises(ContractError):
            RootPathMotionCandidate(
                candidate_id="bad",
                covered_start_m=0.5,
                covered_stop_m=0.5,
                placement_cost=0.0,
                frame_count=10,
            )

    def test_segment_starts_include_both_path_boundaries(self):
        self.assertEqual(
            root_path_segment_starts(
                path_length_m=2.54,
                segment_length_m=0.90,
                stride_m=0.20,
            ),
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.64),
        )

    def test_search_anchors_extend_to_short_terminal_windows(self):
        anchors = root_path_search_anchors(
            path_length_m=2.54,
            stride_m=0.20,
            minimum_window_length_m=0.40,
        )

        self.assertEqual(anchors[-2:], (2.0, 2.14))
        self.assertIn(1.8, anchors)

    def test_rejects_chain_whose_outgoing_cut_precedes_incoming_cut(self):
        candidates = (
            self.candidate("start", 0.0, 0.8),
            self.candidate("cheap-middle", 0.7, 1.5),
            self.candidate("valid-middle", 0.7, 1.5, 0.5),
            self.candidate("finish", 1.4, 2.1),
        )
        transitions = (
            RootPathMotionTransition(
                "start", "cheap-middle", 0.0, 70, 80
            ),
            RootPathMotionTransition(
                "cheap-middle", "finish", 0.0, 20, 10
            ),
            RootPathMotionTransition(
                "start", "valid-middle", 0.0, 70, 10
            ),
            RootPathMotionTransition(
                "valid-middle", "finish", 0.0, 80, 10
            ),
        )

        result = select_root_path_motion_chain(
            candidates=candidates,
            transitions=transitions,
            path_length_m=2.0,
            minimum_segment_frames=5,
        )

        self.assertEqual(
            result.candidate_ids,
            ("start", "valid-middle", "finish"),
        )


if __name__ == "__main__":
    unittest.main()
