from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from mm_sonic.hybrid_terrain_lmm_gpu_search import GpuSearchCandidates
from mm_sonic.hybrid_terrain_lmm_runtime import (
    CommandState,
    HybridMatcher,
    SE2Transform,
    SearchResult,
    TerrainAuthority,
    compose_root_delta,
    run_headless_smoke,
)

from resources import quat
from resources.g1_terrain_builder.schema import ArtifactSet, FeatureSet


def _artifacts(rows: int = 8) -> ArtifactSet:
    result = ArtifactSet.empty(rows, 31)
    result.range_starts = np.asarray((0, rows // 2), dtype=np.int32)
    result.range_stops = np.asarray((rows // 2, rows), dtype=np.int32)
    result.positions[:, 0, 1] = 0.9
    result.positions[:, 1, 0] = np.arange(rows, dtype=np.float32)
    result.rotations[..., 0] = 1.0
    result.terrain_support[: rows // 2, 0] = 0.1
    result.terrain_support[rows // 2 :, 0] = 0.3
    return result


def _corpus(values: np.ndarray | None = None) -> SimpleNamespace:
    if values is None:
        values = np.zeros((8, 31), dtype=np.float32)
        values[:, 0] = np.asarray((0, 1, 2, 3, 10, 11, 12, 13), np.float32)
        values[:4, 15:21] = 0.0
        values[4:, 15:21] = 1.0
        values[:4, 27:31] = 0.0
        values[4:, 27:31] = 1.0
    artifacts = _artifacts(len(values))
    feature_set = FeatureSet(
        np.asarray(values, dtype=np.float32),
        np.zeros(31, dtype=np.float32),
        np.ones(31, dtype=np.float32),
    )
    return SimpleNamespace(
        artifacts=artifacts,
        features=feature_set,
        family_ids=np.asarray((0, 1), dtype=np.int32),
        family_names=("flat", "hill"),
    )


def _motion_corpus() -> SimpleNamespace:
    artifacts = ArtifactSet.empty(8, 31)
    artifacts.range_starts = np.asarray((0, 4), dtype=np.int32)
    artifacts.range_stops = np.asarray((4, 8), dtype=np.int32)
    artifacts.rotations[..., 0] = 1.0
    artifacts.positions[:, 0, 1] = 0.9
    artifacts.positions[1, 0, 0] = 0.1
    artifacts.positions[2, 0, 0] = 0.1
    artifacts.positions[2, 0, 2] = 0.2
    yaw = np.pi / 2.0
    artifacts.rotations[2, 0] = (np.cos(yaw / 2.0), 0.0, np.sin(yaw / 2.0), 0.0)
    artifacts.positions[3, 0, 2] = 0.3
    artifacts.positions[4:, 0, 0] = 9.0
    artifacts.contacts[:4] = (False, False)
    artifacts.contacts[4:] = (True, False)
    values = np.full((8, 31), 10.0, dtype=np.float32)
    values[:4] = 0.0
    values[:, 21:27] = np.tile((0.0, 1.0), 3)
    return SimpleNamespace(
        artifacts=artifacts,
        features=FeatureSet(values, np.zeros(31, np.float32), np.ones(31, np.float32)),
        family_ids=np.asarray((0, 1), dtype=np.int32),
        family_names=("flat", "hill"),
        fps=10.0,
        horizons=(2, 5, 10),
    )


class _Generator:
    def __init__(self, rows: int = 8) -> None:
        self.latent = np.arange(rows, dtype=np.float32)[:, None]
        self.invalid = False
        self.inputs: list[np.ndarray] = []

    def decode(self, features: np.ndarray, latent: np.ndarray) -> np.ndarray:
        row = int(np.asarray(latent).reshape(-1)[0])
        self.inputs.append(np.array(features, dtype=np.float32, copy=True).reshape(-1))
        output = np.zeros(458, dtype=np.float32)
        output[0] = np.nan if self.invalid else 100.0 + row
        return output


class _KeywordGenerator(_Generator):
    def decode(self, features: np.ndarray, *, latent: np.ndarray) -> np.ndarray:
        self.inputs.append(np.array(features, dtype=np.float32, copy=True).reshape(-1))
        rows = np.asarray(latent).reshape(len(features), -1)[:, 0].astype(np.int64)
        output = np.zeros((len(features), 458), dtype=np.float32)
        output[:, 0] = 100.0 + rows
        return output


class _FloatingPointGenerator(_Generator):
    def decode_rows(self, features: np.ndarray, rows: np.ndarray) -> np.ndarray:
        raise FloatingPointError("non-finite model output")


class _FakeSingleGpuSearch:
    def __init__(self) -> None:
        self.responses: list[object] = []
        self.calls: list[dict[str, object]] = []

    def match_candidates(self, query: object, **arguments) -> object:
        self.calls.append(
            {
                "query": np.array(query, dtype=np.float64, copy=True),
                **arguments,
            }
        )
        if not self.responses:
            raise AssertionError("fake GPU search has no queued response")
        return self.responses.pop(0)


def _gpu_candidates(
    rows: object,
    *,
    candidate_count: int,
    close_candidate_count: int,
    device_minimum_score: float = 0.0,
    elapsed_ms: float = 2.5,
) -> GpuSearchCandidates:
    return GpuSearchCandidates(
        rows=np.asarray(rows, dtype=np.int64),
        candidate_count=candidate_count,
        device_minimum_score=device_minimum_score,
        close_candidate_count=close_candidate_count,
        elapsed_ms=elapsed_ms,
    )


def _pose_converter(decoded, simulation_position, simulation_rotation, _model):
    qpos = np.zeros(36, dtype=np.float64)
    local = quat.mul_vec(
        np.asarray(simulation_rotation, dtype=np.float64),
        np.asarray(decoded, dtype=np.float64)[:3],
    )
    qpos[:3] = (
        float(simulation_position[0] + local[0]),
        float(-simulation_position[2] - local[2]),
        float(simulation_position[1] + local[1]),
    )
    qpos[3] = 1.0
    qpos[7] = float(np.asarray(decoded)[0])
    return qpos


def _limited_native_model() -> SimpleNamespace:
    return SimpleNamespace(
        nq=36,
        njnt=2,
        jnt_qposadr=np.asarray((0, 7), dtype=np.int32),
        jnt_range=np.asarray(((0.0, 0.0), (-0.5, 0.5)), dtype=np.float64),
        jnt_type=np.asarray((0, 3), dtype=np.int32),
    )


class HybridTerrainRuntimeTests(unittest.TestCase):
    def test_se2_composition_rotates_local_delta_and_wraps_yaw(self):
        world = SE2Transform(np.asarray((2.0, -3.0)), np.pi / 2.0)

        composed = compose_root_delta(
            world, np.asarray((0.25, -0.5)), 3.0 * np.pi / 2.0
        )

        np.testing.assert_allclose(composed.xy, (2.5, -2.75), atol=1e-12)
        self.assertAlmostEqual(composed.yaw, 0.0)
        self.assertFalse(composed.xy.flags.writeable)

    def test_successor_composes_canonical_root_and_yaw_exactly_once(self):
        placements: list[tuple[np.ndarray, np.ndarray]] = []

        def converter(decoded, simulation_position, simulation_rotation, model):
            placements.append(
                (
                    np.array(simulation_position, copy=True),
                    np.array(simulation_rotation, copy=True),
                )
            )
            return _pose_converter(
                decoded, simulation_position, simulation_rotation, model
            )

        matcher = HybridMatcher(
            _motion_corpus(),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=converter,
            initial_root_xy=(2.0, -3.0),
            initial_heading=0.25,
        )
        matcher._elapsed_since_search = 0.0
        matcher._last_command = CommandState()
        matcher._last_terrain_class = "flat"

        first = matcher.step(CommandState(), dt=0.01)
        second = matcher.step(CommandState(), dt=0.01)

        np.testing.assert_allclose(
            first.root_position_world[:2],
            (2.0 + 0.1 * np.cos(0.25), -3.0 + 0.1 * np.sin(0.25)),
            atol=1e-7,
        )
        expected = compose_root_delta(
            SE2Transform(first.root_position_world[:2], first.heading),
            np.asarray((0.0, -0.2)),
            np.pi / 2.0,
        )
        np.testing.assert_allclose(
            second.root_position_world[:2], expected.xy, atol=1e-7
        )
        self.assertAlmostEqual(second.heading, expected.yaw, places=7)
        placed_yaw = 2.0 * np.arctan2(placements[-1][1][2], placements[-1][1][0])
        self.assertAlmostEqual(placed_yaw, second.heading, delta=1e-6)

    def test_search_jump_reanchors_source_without_root_teleport(self):
        corpus = _motion_corpus()
        corpus.artifacts.contacts[4:] = (False, False)
        matcher = HybridMatcher(
            corpus,
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
            initial_root_xy=(1.0, 2.0),
            initial_heading=-0.3,
        )
        before = matcher.state
        query = np.asarray(matcher.features[6], dtype=np.float64)

        selected = matcher.select_query(query)

        self.assertEqual(selected.row, 4)
        np.testing.assert_array_equal(
            selected.root_position_world[:2], before.root_position_world[:2]
        )
        self.assertEqual(selected.heading, before.heading)

    def test_successor_retry_reanchors_retried_row_once_for_pose_and_terrain(self):
        placements: list[np.ndarray] = []

        def converter(decoded, simulation_position, simulation_rotation, model):
            placements.append(np.array(simulation_rotation, copy=True))
            return _pose_converter(
                decoded, simulation_position, simulation_rotation, model
            )

        class RetryGenerator(_Generator):
            def decode(self, features: np.ndarray, latent: np.ndarray) -> np.ndarray:
                row = int(np.asarray(latent).reshape(-1)[0])
                output = np.zeros(458, dtype=np.float32)
                output[0] = 0.75 if row == 1 else 0.25
                return output

        matcher = HybridMatcher(
            _motion_corpus(),
            RetryGenerator(),
            TerrainAuthority(lambda xy: float(np.asarray(xy)[0]), name="x-ramp"),
            pose_converter=converter,
            native_model=_limited_native_model(),
        )
        matcher._elapsed_since_search = 0.0
        matcher._last_command = CommandState()
        matcher._last_terrain_class = "flat"

        with mock.patch.object(
            matcher,
            "match",
            return_value=SimpleNamespace(row=2, distance=1.25),
        ) as patched_match:
            state = matcher.step(CommandState(), dt=0.01)

        expected = compose_root_delta(
            SE2Transform(np.asarray((0.0, 0.0)), 0.0),
            np.asarray((0.1, -0.2)),
            np.pi / 2.0,
        )
        np.testing.assert_allclose(
            state.root_position_world[:2], expected.xy, atol=1e-7
        )
        self.assertAlmostEqual(state.heading, expected.yaw, places=7)
        placed_yaw = 2.0 * np.arctan2(placements[-1][2], placements[-1][0])
        self.assertAlmostEqual(placed_yaw, state.heading, delta=1e-6)
        np.testing.assert_allclose(
            state.terrain_features,
            np.asarray((0.25, 0.5, 0.75, 1.0)),
            atol=1e-7,
        )
        self.assertEqual(state.row, 2)
        self.assertEqual(state.pose_source, "learned")
        self.assertEqual(state.candidate_limit_rejection_count, 1)
        self.assertEqual(state.first_candidate_limit_rejection_row, 1)
        self.assertEqual(state.max_candidate_limit_rejections_per_step, 1)
        self.assertEqual(patched_match.call_count, 1)

    def test_yaw_alignment_is_shared_by_root_articulation_and_terrain_probes(self):
        queried: list[np.ndarray] = []
        placements: list[np.ndarray] = []
        terrain = TerrainAuthority(
            lambda xy: queried.append(np.array(xy, copy=True)) or float(xy[0]),
            name="x-ramp",
        )

        def converter(decoded, simulation_position, simulation_rotation, model):
            placements.append(np.array(simulation_rotation, copy=True))
            return _pose_converter(
                decoded, simulation_position, simulation_rotation, model
            )

        corpus = _motion_corpus()
        corpus.artifacts.positions[1, 0] = corpus.artifacts.positions[0, 0]
        corpus.artifacts.rotations[1, 0] = (
            np.cos(np.pi / 4.0),
            0.0,
            np.sin(np.pi / 4.0),
            0.0,
        )
        matcher = HybridMatcher(
            corpus,
            _Generator(),
            terrain,
            pose_converter=converter,
            initial_heading=0.2,
        )
        matcher._elapsed_since_search = 0.0
        matcher._last_command = CommandState()
        matcher._last_terrain_class = matcher.state.terrain_class
        queried.clear()

        state = matcher.step(CommandState(), dt=0.01)
        points = matcher._preview_points(CommandState())

        self.assertAlmostEqual(state.heading, 0.2 + np.pi / 2.0, places=7)
        placed_yaw = 2.0 * np.arctan2(placements[-1][2], placements[-1][0])
        self.assertAlmostEqual(placed_yaw, state.heading, places=7)
        forward = points[0] - state.root_position_world[:2]
        np.testing.assert_allclose(
            forward / np.linalg.norm(forward),
            (np.sin(state.heading), -np.cos(state.heading)),
            atol=1e-7,
        )
        self.assertTrue(queried)

    def test_stationary_corpus_does_not_move_for_nonzero_command(self):
        corpus = _motion_corpus()
        corpus.artifacts.positions[:, 0, (0, 2)] = 0.0
        matcher = HybridMatcher(
            corpus,
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )
        matcher._elapsed_since_search = 0.0
        matcher._last_command = CommandState(speed=1.0)
        matcher._last_terrain_class = "flat"

        state = matcher.step(CommandState(speed=1.0), dt=matcher.dt)

        np.testing.assert_array_equal(state.root_position_world[:2], (0.0, 0.0))

    def test_rate_horizons_and_command_speed_come_from_corpus(self):
        matcher = HybridMatcher(
            _motion_corpus(),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )

        query = matcher._command_query(CommandState(speed=1.0), np.zeros(4))

        expected_p95 = np.percentile((1.0, 2.0, np.sqrt(2.0), 0.0, 0.0, 0.0), 95)
        self.assertEqual(matcher.fps, 10.0)
        self.assertEqual(matcher.dt, 0.1)
        self.assertEqual(matcher.horizons, (2, 5, 10))
        self.assertAlmostEqual(matcher.walking_speed_p95_mps, expected_p95)
        np.testing.assert_allclose(query[15:21:2], 0.0, atol=1e-7)
        np.testing.assert_allclose(
            query[16:21:2],
            expected_p95 * np.asarray((0.2, 0.5, 1.0)),
            atol=1e-7,
        )

    def test_contact_bits_are_a_hard_exact_match_filter(self):
        corpus = _motion_corpus()
        corpus.features.values[1:4] = 0.1
        corpus.features.values[4] = 0.0
        matcher = HybridMatcher(
            corpus,
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )
        query = np.zeros(31, dtype=np.float64)

        tree = matcher.match(query, excluded_rows={0})
        brute = matcher.brute_force_match(query, excluded_rows={0})

        self.assertEqual(tree.row, 1)
        self.assertEqual(brute.row, 1)
        self.assertAlmostEqual(tree.distance, brute.distance, places=12)

    def test_contact_exclusions_do_not_resort_when_caller_exclusions_are_empty(
        self,
    ):
        matcher = HybridMatcher(
            _motion_corpus(),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )
        with mock.patch("numpy.union1d", side_effect=AssertionError("redundant sort")):
            result = matcher.match(np.zeros(31, np.float64))
        self.assertEqual(result.row, matcher.brute_force_match(np.zeros(31)).row)

    def test_single_gpu_construction_skips_ckdtree_and_requires_canonical_device(
        self,
    ):
        fake = _FakeSingleGpuSearch()
        with (
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_runtime.cKDTree",
                side_effect=AssertionError("GPU mode must not construct cKDTree"),
            ),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch",
                return_value=fake,
            ) as factory,
        ):
            matcher = HybridMatcher(
                _corpus(),
                _Generator(),
                TerrainAuthority.flat(),
                pose_converter=_pose_converter,
                search_device="cuda:5",
            )

        self.assertEqual(
            matcher.search_backend_identity,
            "single-gpu-full-row-fp32:cuda:5",
        )
        self.assertFalse(hasattr(matcher, "_tree"))
        self.assertFalse(hasattr(matcher, "_tree_values"))
        self.assertIsNone(matcher.last_search_elapsed_ms)
        self.assertIsNone(matcher.warm_search_elapsed_ms)
        factory.assert_called_once()
        positional = factory.call_args.args
        np.testing.assert_array_equal(positional[0], matcher.features)
        np.testing.assert_array_equal(positional[1], matcher.searchable_rows)
        np.testing.assert_array_equal(positional[2], matcher.row_ranges)
        np.testing.assert_array_equal(positional[3], matcher.artifacts.contacts)
        self.assertEqual(
            factory.call_args.kwargs,
            {
                "physical_device_index": 5,
                "transition_penalty": 0.1,
                "exclusion_budget": matcher.candidate_retry_budget,
            },
        )

        for invalid in ("5", "cuda:+5", "cuda:05", "cuda:-1", "cpu:5", 5):
            with (
                self.subTest(search_device=invalid),
                self.assertRaisesRegex(ValueError, "canonical cuda"),
            ):
                HybridMatcher(
                    _corpus(),
                    _Generator(),
                    TerrainAuthority.flat(),
                    pose_converter=_pose_converter,
                    search_device=invalid,
                )

    def test_single_gpu_passes_current_contact_range_and_explicit_exclusions(self):
        fake = _FakeSingleGpuSearch()
        fake.responses.append(
            _gpu_candidates((5,), candidate_count=2, close_candidate_count=1)
        )
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch",
            return_value=fake,
        ):
            matcher = HybridMatcher(
                _motion_corpus(),
                _Generator(),
                TerrainAuthority.flat(),
                pose_converter=_pose_converter,
                search_device="cuda:5",
            )
        matcher.state = SimpleNamespace(row=4, range_index=1)
        query = np.zeros(31, dtype=np.float64)

        result = matcher.match(query, excluded_rows=(4, 4, 3))

        self.assertEqual(result.row, 5)
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        np.testing.assert_array_equal(call["query"], query)
        self.assertEqual(call["current_range"], 1)
        self.assertEqual(call["active_contact_code"], 1)
        np.testing.assert_array_equal(call["excluded_rows"], (4,))

    def test_single_gpu_rejects_capped_search_before_backend_construction(self):
        with (
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch"
            ) as factory,
            self.assertRaisesRegex(ValueError, "full-range-safe"),
        ):
            HybridMatcher(
                _corpus(),
                _Generator(),
                TerrainAuthority.flat(),
                pose_converter=_pose_converter,
                max_search_rows=2,
                search_device="cuda:5",
            )

        factory.assert_not_called()

    def test_single_gpu_rescores_complete_top128_and_preserves_stable_ties(self):
        values = np.ones((130, 31), dtype=np.float32)
        values[2] = np.float32(0.25)
        values[6] = np.float32(0.25)
        values[128] = np.float32(0.0)
        fake = _FakeSingleGpuSearch()
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch",
            return_value=fake,
        ):
            matcher = HybridMatcher(
                _corpus(values),
                _Generator(),
                TerrainAuthority.flat(),
                pose_converter=_pose_converter,
                search_device="cuda:5",
            )
        self.assertEqual(len(matcher.searchable_rows), 128)
        fake.responses.extend(
            (
                _gpu_candidates(
                    matcher.searchable_rows,
                    candidate_count=128,
                    close_candidate_count=1,
                    elapsed_ms=3.25,
                ),
                _gpu_candidates(
                    (2, 6),
                    candidate_count=128,
                    close_candidate_count=2,
                    elapsed_ms=7.0,
                ),
            )
        )

        complete = matcher.match(np.zeros(31, dtype=np.float64))
        tied = matcher.match(np.full(31, 0.25, dtype=np.float64))

        self.assertEqual(complete.row, 128)
        self.assertEqual(complete.candidate_count, 128)
        self.assertEqual(tied.row, 2)
        self.assertEqual(matcher.last_search_elapsed_ms, 7.0)
        self.assertEqual(matcher.warm_search_elapsed_ms, 3.25)

    def test_single_gpu_rejects_malformed_candidates_before_timing_or_scoring(self):
        fake = _FakeSingleGpuSearch()
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch",
            return_value=fake,
        ):
            matcher = HybridMatcher(
                _motion_corpus(),
                _Generator(),
                TerrainAuthority.flat(),
                pose_converter=_pose_converter,
                search_device="cuda:5",
            )
        base = {
            "rows": np.asarray((0,), dtype=np.int64),
            "candidate_count": 3,
            "device_minimum_score": 0.0,
            "close_candidate_count": 1,
            "elapsed_ms": 2.5,
        }
        cases = (
            ("negative row", {"rows": np.asarray((-1,), dtype=np.int64)}, ()),
            ("excluded row", {"rows": np.asarray((1,), dtype=np.int64)}, (1,)),
            ("terminal row", {"rows": np.asarray((3,), dtype=np.int64)}, ()),
            ("incompatible row", {"rows": np.asarray((4,), dtype=np.int64)}, ()),
            ("duplicate rows", {"rows": np.asarray((0, 0), dtype=np.int64)}, ()),
            ("unsorted rows", {"rows": np.asarray((1, 0), dtype=np.int64)}, ()),
            ("empty rows", {"rows": np.empty(0, dtype=np.int64)}, ()),
            ("rank-two rows", {"rows": np.asarray(((0,),), dtype=np.int64)}, ()),
            ("float rows", {"rows": np.asarray((0.5,), dtype=np.float64)}, ()),
            ("too many rows", {"rows": np.arange(129, dtype=np.int64)}, ()),
            ("boolean candidate count", {"candidate_count": True}, ()),
            ("fractional candidate count", {"candidate_count": 1.0}, ()),
            ("short candidate count", {"candidate_count": 0}, ()),
            ("inexact candidate count", {"candidate_count": 2}, ()),
            ("large candidate count", {"candidate_count": 7}, ()),
            ("boolean close count", {"close_candidate_count": True}, ()),
            ("fractional close count", {"close_candidate_count": 1.0}, ()),
            ("zero close count", {"close_candidate_count": 0}, ()),
            ("large close count", {"close_candidate_count": 4}, ()),
            ("negative minimum", {"device_minimum_score": -0.1}, ()),
            ("nonfinite minimum", {"device_minimum_score": np.inf}, ()),
            ("negative timing", {"elapsed_ms": -1.0}, ()),
            ("nonfinite timing", {"elapsed_ms": np.nan}, ()),
            ("nonnumeric timing", {"elapsed_ms": "2.5"}, ()),
        )
        query = np.zeros(31, dtype=np.float64)
        for label, changed, exclusions in cases:
            with self.subTest(case=label):
                fake.responses.append(SimpleNamespace(**{**base, **changed}))
                matcher.last_search_elapsed_ms = 19.0
                matcher.warm_search_elapsed_ms = 17.0
                with (
                    mock.patch.object(
                        matcher,
                        "_candidate_score",
                        side_effect=AssertionError(
                            "invalid candidates must not be scored"
                        ),
                    ),
                    self.assertRaisesRegex(ValueError, "single-GPU"),
                ):
                    matcher.match(query, excluded_rows=exclusions)
                self.assertEqual(matcher.last_search_elapsed_ms, 19.0)
                self.assertEqual(matcher.warm_search_elapsed_ms, 17.0)

    def test_single_gpu_rejects_more_than_top128_on_a_large_compatible_view(self):
        values = np.zeros((132, 31), dtype=np.float32)
        fake = _FakeSingleGpuSearch()
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch",
            return_value=fake,
        ):
            matcher = HybridMatcher(
                _corpus(values),
                _Generator(),
                TerrainAuthority.flat(),
                pose_converter=_pose_converter,
                search_device="cuda:5",
            )
        self.assertGreater(len(matcher.searchable_rows), 128)
        fake.responses.append(
            SimpleNamespace(
                rows=np.array(matcher.searchable_rows[:129], copy=True),
                candidate_count=len(matcher.searchable_rows),
                device_minimum_score=0.0,
                close_candidate_count=1,
                elapsed_ms=2.5,
            )
        )
        matcher.last_search_elapsed_ms = 19.0
        matcher.warm_search_elapsed_ms = 17.0

        with self.assertRaisesRegex(ValueError, "single-GPU"):
            matcher.match(np.zeros(31, dtype=np.float64))

        self.assertEqual(matcher.last_search_elapsed_ms, 19.0)
        self.assertEqual(matcher.warm_search_elapsed_ms, 17.0)

    def test_single_gpu_close_set_overflow_invokes_complete_brute_force(self):
        fake = _FakeSingleGpuSearch()
        fake.responses.append(
            _gpu_candidates(
                (0, 2),
                candidate_count=5,
                close_candidate_count=3,
                device_minimum_score=-0.0,
            )
        )
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch",
            return_value=fake,
        ):
            matcher = HybridMatcher(
                _corpus(),
                _Generator(),
                TerrainAuthority.flat(),
                pose_converter=_pose_converter,
                search_device="cuda:5",
            )
        expected = SearchResult(row=2, distance=0.75, candidate_count=5)
        query = np.zeros(31, dtype=np.float64)
        with mock.patch.object(
            matcher, "brute_force_match", return_value=expected
        ) as brute:
            result = matcher.match(query, excluded_rows=(1, 1, 3))

        self.assertEqual(result, expected)
        brute.assert_called_once()
        np.testing.assert_array_equal(brute.call_args.args[0], query)
        np.testing.assert_array_equal(
            brute.call_args.kwargs["excluded_rows"], np.asarray((1,), dtype=np.int64)
        )

    def test_single_gpu_default_keeps_cpu_ckdtree_mode_unchanged(self):
        values = np.zeros((8, 31), dtype=np.float32)
        values[:, 0] = np.arange(8, dtype=np.float32)
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch"
        ) as factory:
            matcher = HybridMatcher(
                _corpus(values),
                _Generator(),
                TerrainAuthority.flat(),
                pose_converter=_pose_converter,
            )

        factory.assert_not_called()
        self.assertEqual(matcher.search_backend_identity, "cpu-ckdtree-exact")
        self.assertTrue(hasattr(matcher, "_tree"))
        self.assertTrue(hasattr(matcher, "_tree_values"))
        self.assertIsNone(matcher.last_search_elapsed_ms)
        self.assertIsNone(matcher.warm_search_elapsed_ms)
        for query in (
            np.zeros(31, dtype=np.float64),
            np.full(31, 2.5, dtype=np.float64),
        ):
            tree = matcher.match(query)
            brute = matcher.brute_force_match(query)
            self.assertEqual(tree.row, brute.row)
            self.assertAlmostEqual(tree.distance, brute.distance, places=12)

    def test_finite_terrain_root_domain_miss_bypasses_learned_decode(self):
        generator = _Generator()
        terrain = TerrainAuthority(
            lambda _xy: 0.0,
            domain_contains=lambda xy: bool(np.linalg.norm(xy) <= 1.0),
            name="finite-flat",
        )
        matcher = HybridMatcher(
            _corpus(),
            generator,
            terrain,
            pose_converter=_pose_converter,
            initial_root_xy=(2.0, 0.0),
        )

        state = matcher.step(CommandState(), dt=0.04, force_search=True)

        self.assertEqual(terrain.height_at((2.0, 0.0)), 0.0)
        self.assertEqual(state.support_status, "OUT OF TRAINED SUPPORT")
        self.assertEqual(state.pose_source, "canonical-fallback")
        self.assertEqual(state.fallback_count, 1)
        self.assertEqual(state.decode_count, 0)
        self.assertEqual(generator.inputs, [])

    def test_finite_terrain_checks_root_and_exact_four_curved_preview_points(self):
        calls: list[np.ndarray] = []

        def domain_contains(xy):
            calls.append(np.array(xy, dtype=np.float64, copy=True))
            return len(calls) != 5

        generator = _Generator()
        matcher = HybridMatcher(
            _corpus(),
            generator,
            TerrainAuthority(
                lambda _xy: 0.0,
                domain_contains=domain_contains,
                name="finite-curved-preview",
            ),
            pose_converter=_pose_converter,
        )
        calls.clear()
        command = CommandState(speed=1.0, steering=1.0)

        state = matcher.step(command, dt=0.04, force_search=True)

        distances = np.asarray((0.25, 0.5, 0.75, 1.0), np.float64)
        yaw = 1.5 * distances / 0.45
        local_x = distances * (1.0 - np.cos(yaw)) / yaw
        local_z = distances * np.sin(yaw) / yaw
        cosine, sine = np.cos(state.heading), np.sin(state.heading)
        expected = np.column_stack(
            (
                state.root_position_world[0] + cosine * local_x + sine * local_z,
                state.root_position_world[1] + sine * local_x - cosine * local_z,
            )
        )
        self.assertEqual(len(calls), 5)
        np.testing.assert_allclose(calls[0], state.root_position_world[:2], atol=1e-12)
        np.testing.assert_allclose(np.stack(calls[1:]), expected, atol=1e-12)
        self.assertEqual(state.support_status, "OUT OF TRAINED SUPPORT")
        self.assertEqual(state.pose_source, "canonical-fallback")
        self.assertEqual(generator.inputs, [])

    def test_flat_terrain_authority_is_explicitly_infinite(self):
        terrain = TerrainAuthority.flat(height=0.3)

        self.assertTrue(terrain.contains((1.0e12, -1.0e12)))
        self.assertEqual(terrain.height_at((1.0e12, -1.0e12)), 0.3)

    def test_capped_search_view_is_deterministically_stratified_by_family(self):
        rows = 16
        artifacts = ArtifactSet.empty(rows, 31)
        artifacts.range_starts = np.asarray((0, 10, 13), np.int32)
        artifacts.range_stops = np.asarray((10, 13, 16), np.int32)
        artifacts.rotations[..., 0] = 1.0
        corpus = SimpleNamespace(
            artifacts=artifacts,
            features=FeatureSet(
                np.arange(rows * 31, dtype=np.float32).reshape(rows, 31),
                np.zeros(31, np.float32),
                np.ones(31, np.float32),
            ),
            range_family_ids=np.asarray((0, 1, 2), np.int16),
            family_ids=np.asarray((0,) * 10 + (1,) * 3 + (2,) * 3, np.int16),
            family_names=("flat", "curb", "slope"),
        )

        first = HybridMatcher(
            corpus,
            _Generator(rows),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
            max_search_rows=6,
        )
        second = HybridMatcher(
            corpus,
            _Generator(rows),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
            max_search_rows=6,
        )

        self.assertEqual(
            first.searchable_family_counts, (("flat", 2), ("curb", 2), ("slope", 2))
        )
        np.testing.assert_array_equal(first.searchable_rows, second.searchable_rows)
        self.assertEqual(len(first.searchable_rows), 6)
        self.assertEqual(first.total_searchable_row_count, 13)
        self.assertEqual(
            first.total_searchable_family_counts,
            (("flat", 9), ("curb", 2), ("slope", 2)),
        )
        self.assertEqual(first.search_scope, "diagnostic-stratified-cap")
        self.assertFalse(first.search_acceptance_eligible)

        full = HybridMatcher(
            corpus,
            _Generator(rows),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )
        self.assertEqual(len(full.searchable_rows), 13)
        self.assertEqual(full.total_searchable_row_count, 13)
        self.assertEqual(full.search_scope, "full-range-safe-corpus")
        self.assertTrue(full.search_acceptance_eligible)

    def test_transition_penalty_defaults_to_point_one_squared_normalized_cost(self):
        values = np.full((8, 31), 10.0, dtype=np.float32)
        values[1] = 0.0
        values[1, 0] = np.sqrt(0.05)
        values[4] = 0.0
        query = np.zeros(31, dtype=np.float64)

        regularized = HybridMatcher(
            _corpus(values),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )
        unregularized = HybridMatcher(
            _corpus(values),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
            transition_penalty=0.0,
        )

        self.assertAlmostEqual(regularized.transition_penalty, 0.1)
        self.assertEqual(regularized.match(query).row, 1)
        self.assertEqual(unregularized.match(query).row, 4)

    def test_exact_match_exclusions_preserve_transition_penalty_ordering(self):
        values = np.full((8, 31), 10.0, dtype=np.float32)
        values[1] = 0.0
        values[1, 0] = np.sqrt(0.05)
        values[4] = 0.0
        query = np.zeros(31, dtype=np.float64)
        matcher = HybridMatcher(
            _corpus(values),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )

        tree = matcher.match(query, excluded_rows={1})
        brute = matcher.brute_force_match(query, excluded_rows={1})

        self.assertEqual(tree.row, 4)
        self.assertEqual(brute.row, 4)
        self.assertAlmostEqual(tree.distance, np.sqrt(0.1), places=12)
        self.assertAlmostEqual(tree.distance, brute.distance, places=12)

    def test_exact_tree_expands_beyond_128_before_chunked_brute_fallback(self):
        rows = 205
        artifacts = ArtifactSet.empty(rows, 31)
        artifacts.range_starts = np.asarray((0, 3), np.int32)
        artifacts.range_stops = np.asarray((3, rows), np.int32)
        artifacts.rotations[..., 0] = 1.0
        values = np.full((rows, 31), 10.0, np.float32)
        values[1] = 0.0
        values[1, 0] = np.sqrt(0.05)
        competing = np.sqrt(np.linspace(0.0, 0.04, rows - 4))
        values[3:-1] = 0.0
        values[3:-1, 0] = competing
        corpus = SimpleNamespace(
            artifacts=artifacts,
            features=FeatureSet(
                values, np.zeros(31, np.float32), np.ones(31, np.float32)
            ),
            range_family_ids=np.asarray((0, 1), np.int16),
            family_names=("flat", "slope"),
        )
        matcher = HybridMatcher(
            corpus,
            _Generator(rows),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )

        with mock.patch.object(
            matcher, "brute_force_match", wraps=matcher.brute_force_match
        ) as brute:
            result = matcher.match(np.zeros(31, np.float64))

        self.assertEqual(result.row, 1)
        self.assertLessEqual(result.candidate_count, 34)
        brute.assert_not_called()

    def test_capped_search_view_water_fills_after_a_family_exhausts(self):
        # Search-safe capacities are [1, 40, 20, 15]. A 40-row equal-family
        # cap first grants ten each, then water-fills nine unused flat slots
        # equally instead of weighting the refill by remaining raw density.
        lengths = np.asarray((2, 41, 21, 16), np.int32)
        stops = np.cumsum(lengths, dtype=np.int32)
        starts = np.concatenate((np.asarray((0,), np.int32), stops[:-1]))
        rows = int(stops[-1])
        artifacts = ArtifactSet.empty(rows, 31)
        artifacts.range_starts = starts
        artifacts.range_stops = stops
        artifacts.rotations[..., 0] = 1.0
        family_ids = np.repeat(np.arange(4, dtype=np.int16), lengths)
        corpus = SimpleNamespace(
            artifacts=artifacts,
            features=FeatureSet(
                np.zeros((rows, 31), np.float32),
                np.zeros(31, np.float32),
                np.ones(31, np.float32),
            ),
            range_family_ids=np.arange(4, dtype=np.int16),
            family_ids=family_ids,
            family_names=("flat", "curb", "slope", "stair"),
        )

        matcher = HybridMatcher(
            corpus,
            _Generator(rows),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
            max_search_rows=40,
        )

        self.assertEqual(
            matcher.searchable_family_counts,
            (("flat", 1), ("curb", 13), ("slope", 13), ("stair", 13)),
        )

    def test_command_shapes_query_without_dragging_stationary_source_root(self):
        terrain = TerrainAuthority(
            lambda xy: -float(np.asarray(xy)[1]), name="native-negative-y-ramp"
        )
        matcher = HybridMatcher(
            _corpus(),
            _Generator(),
            terrain,
            pose_converter=_pose_converter,
        )

        state = matcher.step(CommandState(speed=1.0), dt=0.04, force_search=True)

        self.assertAlmostEqual(state.root_position_world[1], 0.0)
        self.assertAlmostEqual(state.qpos[1], state.root_position_world[1])
        np.testing.assert_allclose(
            state.query[15:21],
            (0.0, 0.45 * 0.32, 0.0, 0.45 * 0.68, 0.0, 0.45),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            state.terrain_features, (0.25, 0.5, 0.75, 1.0), atol=1e-7
        )

    def test_steering_trajectory_and_terrain_follow_the_same_command_arc(self):
        terrain = TerrainAuthority(lambda xy: float(np.asarray(xy)[0]), name="x-ramp")
        matcher = HybridMatcher(
            _corpus(),
            _Generator(),
            terrain,
            pose_converter=_pose_converter,
        )
        command = CommandState(speed=1.0, steering=1.0)

        state = matcher.step(command, dt=0.04, force_search=True)

        times = np.asarray((0.32, 0.68, 1.0))
        expected_trajectory_x = 0.45 / 1.5 * (1.0 - np.cos(1.5 * times))
        np.testing.assert_allclose(
            state.query[15:21:2], expected_trajectory_x, atol=1e-7
        )
        distances = np.asarray((0.25, 0.5, 0.75, 1.0))
        yaw = 1.5 * distances / 0.45
        local_x = distances * (1.0 - np.cos(yaw)) / yaw
        local_z = distances * np.sin(yaw) / yaw
        expected_terrain_x = (
            np.cos(state.heading) * local_x + np.sin(state.heading) * local_z
        )
        np.testing.assert_allclose(
            state.terrain_features, expected_terrain_x, atol=1e-7
        )
        self.assertTrue(np.all(state.query[15:21:2] > 0.0))
        self.assertTrue(np.all(state.terrain_features > 0.0))

    def test_outside_corpus_normalized_terrain_envelope_skips_learned_decode(self):
        values = np.zeros((8, 31), np.float32)
        values[:4, 27:31] = -0.1
        values[4:, 27:31] = 0.1
        generator = _Generator()
        matcher = HybridMatcher(
            _corpus(values),
            generator,
            TerrainAuthority(lambda xy: -float(np.asarray(xy)[1]), name="steep"),
            pose_converter=_pose_converter,
        )

        state = matcher.step(CommandState(speed=1.0), dt=0.04, force_search=True)

        np.testing.assert_allclose(matcher.terrain_feature_min, -0.1)
        np.testing.assert_allclose(matcher.terrain_feature_max, 0.1)
        self.assertEqual(state.support_status, "OUT OF TRAINED SUPPORT")
        self.assertEqual(state.pose_source, "canonical-fallback")
        self.assertEqual(state.decode_count, 0)
        self.assertEqual(state.fallback_count, 1)
        self.assertEqual(generator.inputs, [])

    def test_real_generator_keyword_batch_decode_uses_learned_pose(self):
        generator = _KeywordGenerator()
        matcher = HybridMatcher(
            _corpus(),
            generator,
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )

        state = matcher.step(CommandState(speed=1.0), dt=0.04, force_search=True)

        self.assertEqual(state.pose_source, "learned")
        self.assertEqual(state.fallback_count, 0)
        self.assertEqual(state.decode_count, 1)
        self.assertEqual(generator.inputs[-1].shape, (31,))

    def test_real_generator_floating_point_failure_uses_canonical_fallback(self):
        matcher = HybridMatcher(
            _corpus(),
            _FloatingPointGenerator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )

        state = matcher.step(CommandState(), dt=0.04, force_search=True)

        self.assertEqual(state.pose_source, "canonical-fallback")
        self.assertEqual(state.fallback_count, 1)

    def test_terrain_channels_are_elevation_invariant_relative_heights(self):
        ramp = TerrainAuthority(lambda xy: 0.2 * float(np.asarray(xy)[1]), name="ramp")
        raised = TerrainAuthority(
            lambda xy: 7.5 + 0.2 * float(np.asarray(xy)[1]), name="raised-ramp"
        )

        np.testing.assert_allclose(
            ramp.sample((0.3, 1.2), 0.0),
            raised.sample((0.3, 1.2), 0.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            ramp.sample((0.3, 1.2), 0.0), (-0.05, -0.1, -0.15, -0.2)
        )
        self.assertNotEqual(ramp.support((0.3, 1.2)), raised.support((0.3, 1.2)))

    def test_legacy_full_speed_query_retains_point_45_meter_per_second_scale(self):
        matcher = HybridMatcher(
            _corpus(),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )

        state = matcher.step(CommandState(speed=1.0), dt=0.04, force_search=True)

        self.assertAlmostEqual(state.root_position_world[1], 0.0)
        np.testing.assert_allclose(
            state.query[15:21],
            (0.0, 0.45 * 0.32, 0.0, 0.45 * 0.68, 0.0, 0.45),
            atol=1e-7,
        )

    def test_ckdtree_matches_float64_brute_force_and_stable_ties(self):
        rng = np.random.default_rng(83)
        values = rng.normal(size=(8, 31)).astype(np.float32)
        values[1] = values[0]
        matcher = HybridMatcher(
            _corpus(values),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )

        tied = values[0].astype(np.float64)
        self.assertEqual(matcher.match(tied).row, 0)
        for _ in range(32):
            query = rng.normal(size=31)
            tree = matcher.match(query)
            brute = matcher.brute_force_match(query)
            self.assertEqual(tree.row, brute.row)
            self.assertAlmostEqual(tree.distance, brute.distance, places=12)

    def test_successors_never_cross_a_source_range(self):
        matcher = HybridMatcher(
            _corpus(),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )
        self.assertEqual(matcher.successor(2), 3)
        self.assertEqual(matcher.successor(3), 3)
        self.assertEqual(matcher.successor(6), 7)
        self.assertEqual(matcher.successor(7), 7)
        with self.assertRaisesRegex(IndexError, "row"):
            matcher.successor(8)

    def test_live_terrain_overwrites_generator_input_and_places_root_from_clearance(
        self,
    ):
        generator = _Generator()
        terrain = TerrainAuthority(
            lambda xy: 0.5 + 0.1 * float(np.asarray(xy)[0]),
            name="ramp",
        )
        matcher = HybridMatcher(
            _corpus(),
            generator,
            terrain,
            pose_converter=_pose_converter,
        )

        state = matcher.step(CommandState(speed=0.5), dt=0.04, force_search=True)

        np.testing.assert_allclose(generator.inputs[-1][27:31], state.terrain_features)
        selected_clearance = (
            matcher.corpus.artifacts.positions[state.row, 0, 1]
            - matcher.corpus.artifacts.terrain_support[state.row, 0]
        )
        self.assertAlmostEqual(
            state.root_position_world[2], state.support_height + selected_clearance
        )
        self.assertAlmostEqual(state.qpos[2], state.root_position_world[2])

    def test_search_can_switch_ranges_nonmonotonically(self):
        matcher = HybridMatcher(
            _corpus(),
            _Generator(),
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )
        high = np.array(matcher.corpus.features.values[5], dtype=np.float64)
        low = np.array(matcher.corpus.features.values[1], dtype=np.float64)
        self.assertEqual(matcher.select_query(high).range_index, 1)
        self.assertEqual(matcher.select_query(low).range_index, 0)
        self.assertEqual(matcher.select_query(high).range_index, 1)

    def test_invalid_learned_decode_commits_visible_canonical_fallback_atomically(self):
        generator = _Generator()
        matcher = HybridMatcher(
            _corpus(),
            generator,
            TerrainAuthority.flat(),
            pose_converter=_pose_converter,
        )
        accepted = matcher.step(CommandState(), dt=0.04, force_search=True)
        generator.invalid = True

        fallback = matcher.step(CommandState(speed=1.0), dt=0.04, force_search=True)

        self.assertEqual(fallback.decode_count, accepted.decode_count)
        self.assertEqual(fallback.fallback_count, accepted.fallback_count + 1)
        self.assertEqual(fallback.pose_source, "canonical-fallback")
        self.assertEqual(fallback.qpos[7], float(fallback.row))
        self.assertTrue(np.isfinite(fallback.qpos).all())

    def test_native_invalid_exact_winner_retries_next_exact_candidate_atomically(self):
        values = np.full((8, 31), 10.0, dtype=np.float32)
        values[0] = 0.0
        values[1] = 0.0
        values[1, 0] = 0.1

        class SelectiveGenerator(_Generator):
            def decode(self, features: np.ndarray, latent: np.ndarray) -> np.ndarray:
                row = int(np.asarray(latent).reshape(-1)[0])
                self.inputs.append(
                    np.array(features, dtype=np.float32, copy=True).reshape(-1)
                )
                output = np.zeros(458, dtype=np.float32)
                output[0] = 0.75 if row == 0 else 0.25
                return output

        matcher = HybridMatcher(
            _corpus(values),
            SelectiveGenerator(),
            TerrainAuthority.flat(),
            native_model=_limited_native_model(),
            pose_converter=_pose_converter,
        )

        state = matcher.select_query(np.zeros(31, dtype=np.float64))

        self.assertEqual(state.row, 1)
        self.assertEqual(state.pose_source, "learned")
        self.assertEqual(state.decode_count, 1)
        self.assertEqual(state.fallback_count, 0)
        self.assertEqual(state.joint_clamp_count, 0)
        self.assertEqual(state.candidate_limit_rejection_count, 1)
        self.assertEqual(state.first_candidate_limit_rejection_row, 0)
        self.assertEqual(state.max_candidate_limit_rejections_per_step, 1)
        self.assertAlmostEqual(state.qpos[7], 0.25)

    def test_single_gpu_retry_forwards_only_explicit_compatible_rejections(self):
        rows = 80
        artifacts = ArtifactSet.empty(rows, 31)
        artifacts.range_starts = np.asarray((0,), dtype=np.int32)
        artifacts.range_stops = np.asarray((rows,), dtype=np.int32)
        artifacts.positions[:, 0, 1] = 0.9
        artifacts.rotations[..., 0] = 1.0
        artifacts.contacts[2:] = (True, False)
        values = np.full((rows, 31), 10.0, dtype=np.float32)
        values[0] = 0.0
        values[1] = 0.0
        values[1, 0] = 0.1
        corpus = SimpleNamespace(
            artifacts=artifacts,
            features=FeatureSet(
                values, np.zeros(31, np.float32), np.ones(31, np.float32)
            ),
            family_ids=np.asarray((0,), dtype=np.int32),
            family_names=("flat",),
            candidate_retry_budget=32,
            fps=60.0,
            horizons=(20, 40, 60),
        )

        class SelectiveGenerator(_Generator):
            def decode(self, features: np.ndarray, latent: np.ndarray) -> np.ndarray:
                row = int(np.asarray(latent).reshape(-1)[0])
                output = np.zeros(458, dtype=np.float32)
                output[0] = 0.75 if row == 0 else 0.25
                return output

        fake = _FakeSingleGpuSearch()
        fake.responses.extend(
            (
                _gpu_candidates((0, 1), candidate_count=2, close_candidate_count=1),
                _gpu_candidates((1,), candidate_count=1, close_candidate_count=1),
            )
        )
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch",
            return_value=fake,
        ):
            matcher = HybridMatcher(
                corpus,
                SelectiveGenerator(rows),
                TerrainAuthority.flat(),
                native_model=_limited_native_model(),
                pose_converter=_pose_converter,
                search_device="cuda:5",
            )

        with (
            mock.patch.object(
                matcher,
                "_match_exclusions",
                side_effect=AssertionError(
                    "GPU retry must not expand contact incompatibilities"
                ),
            ),
            mock.patch(
                "numpy.union1d",
                side_effect=AssertionError("GPU retry must not sort exclusions"),
            ),
        ):
            state = matcher.select_query(np.zeros(31, dtype=np.float64))

        self.assertEqual(state.row, 1)
        self.assertEqual(state.candidate_limit_rejection_count, 1)
        self.assertGreater(
            np.count_nonzero(
                np.any(
                    artifacts.contacts[matcher.searchable_rows]
                    != artifacts.contacts[0],
                    axis=1,
                )
            ),
            matcher.candidate_retry_budget,
        )
        self.assertEqual(len(fake.calls), 2)
        np.testing.assert_array_equal(fake.calls[0]["excluded_rows"], ())
        np.testing.assert_array_equal(fake.calls[1]["excluded_rows"], (0,))

    def test_single_gpu_invalid_contact_transition_successor_retries_without_exclusion(
        self,
    ):
        values = np.full((8, 31), 10.0, dtype=np.float32)
        values[0:2] = 0.0
        values[0:2, 21:27] = np.tile((0.0, 1.0), 3)
        corpus = _corpus(values)
        corpus.artifacts.contacts[1] = (True, False)
        corpus.fps = 60.0
        corpus.horizons = (20, 40, 60)

        class SuccessorGenerator(_Generator):
            def decode(self, features: np.ndarray, latent: np.ndarray) -> np.ndarray:
                row = int(np.asarray(latent).reshape(-1)[0])
                output = np.zeros(458, dtype=np.float32)
                output[0] = 0.75 if row == 1 else 0.25
                return output

        fake = _FakeSingleGpuSearch()
        fake.responses.append(
            _gpu_candidates((0,), candidate_count=5, close_candidate_count=1)
        )
        with mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search.SingleGpuExactSearch",
            return_value=fake,
        ):
            matcher = HybridMatcher(
                corpus,
                SuccessorGenerator(),
                TerrainAuthority.flat(),
                native_model=_limited_native_model(),
                pose_converter=_pose_converter,
                search_device="cuda:5",
            )
        matcher._elapsed_since_search = 0.0
        matcher._last_command = CommandState()
        matcher._last_terrain_class = "flat"
        np.testing.assert_array_equal(corpus.artifacts.contacts[0], (False, False))
        np.testing.assert_array_equal(corpus.artifacts.contacts[1], (True, False))

        with (
            mock.patch.object(
                matcher,
                "_match_exclusions",
                side_effect=AssertionError(
                    "GPU retry must not expand contact incompatibilities"
                ),
            ),
            mock.patch(
                "numpy.union1d",
                side_effect=AssertionError("GPU retry must not sort exclusions"),
            ),
        ):
            state = matcher.step(CommandState(), dt=0.04)

        self.assertEqual(state.row, 0)
        self.assertEqual(state.pose_source, "learned")
        self.assertEqual(state.candidate_limit_rejection_count, 1)
        self.assertEqual(state.first_candidate_limit_rejection_row, 1)
        self.assertEqual(state.max_candidate_limit_rejections_per_step, 1)
        self.assertEqual(len(fake.calls), 1)
        np.testing.assert_array_equal(fake.calls[0]["excluded_rows"], ())

    def test_native_invalid_successor_retries_the_same_query_exactly(self):
        values = np.full((8, 31), 10.0, dtype=np.float32)
        values[0:2] = 0.0
        values[0:2, 21:27] = np.tile((0.0, 1.0), 3)

        class SuccessorGenerator(_Generator):
            def decode(self, features: np.ndarray, latent: np.ndarray) -> np.ndarray:
                row = int(np.asarray(latent).reshape(-1)[0])
                output = np.zeros(458, dtype=np.float32)
                output[0] = 0.75 if row == 1 else 0.25
                return output

        matcher = HybridMatcher(
            _corpus(values),
            SuccessorGenerator(),
            TerrainAuthority.flat(),
            native_model=_limited_native_model(),
            pose_converter=_pose_converter,
        )
        matcher._elapsed_since_search = 0.0
        matcher._last_command = CommandState()
        matcher._last_terrain_class = "flat"

        state = matcher.step(CommandState(), dt=0.04)

        self.assertEqual(state.row, 0)
        self.assertEqual(state.pose_source, "learned")
        self.assertEqual(state.decode_count, 1)
        self.assertEqual(state.fallback_count, 0)
        self.assertEqual(state.joint_clamp_count, 0)
        self.assertEqual(state.candidate_limit_rejection_count, 1)
        self.assertEqual(state.first_candidate_limit_rejection_row, 1)
        self.assertEqual(state.max_candidate_limit_rejections_per_step, 1)

    def test_native_limit_retry_exhaustion_holds_complete_previous_state(self):
        rows = 40
        artifacts = ArtifactSet.empty(rows, 31)
        artifacts.range_starts = np.asarray((0,), dtype=np.int32)
        artifacts.range_stops = np.asarray((rows,), dtype=np.int32)
        artifacts.positions[:, 0, 1] = 0.9
        artifacts.rotations[..., 0] = 1.0
        values = np.zeros((rows, 31), dtype=np.float32)
        values[:, 0] = np.arange(rows, dtype=np.float32) * 0.01
        corpus = SimpleNamespace(
            artifacts=artifacts,
            features=FeatureSet(
                values, np.zeros(31, np.float32), np.ones(31, np.float32)
            ),
            family_ids=np.asarray((0,), dtype=np.int32),
            family_names=("flat",),
            candidate_retry_budget=4,
            fps=60.0,
            horizons=(20, 40, 60),
        )

        class AlwaysInvalidGenerator(_Generator):
            def decode(self, features: np.ndarray, latent: np.ndarray) -> np.ndarray:
                output = np.zeros(458, dtype=np.float32)
                output[0] = 0.75
                return output

        matcher = HybridMatcher(
            corpus,
            AlwaysInvalidGenerator(rows),
            TerrainAuthority.flat(),
            native_model=_limited_native_model(),
            pose_converter=_pose_converter,
        )
        before = matcher.state
        before_root = matcher._root_xy.copy()
        before_heading = matcher._heading
        before_elapsed = matcher._elapsed_since_search
        before_command = matcher._last_command
        before_terrain_class = matcher._last_terrain_class

        state = matcher.step(
            CommandState(speed=1.0, steering=0.5),
            dt=matcher.dt,
            force_search=True,
        )

        self.assertEqual(state.row, before.row)
        np.testing.assert_array_equal(state.qpos, before.qpos)
        np.testing.assert_array_equal(state.query, before.query)
        np.testing.assert_array_equal(
            state.root_position_world, before.root_position_world
        )
        np.testing.assert_array_equal(matcher._root_xy, before_root)
        self.assertEqual(matcher._heading, before_heading)
        self.assertEqual(matcher._elapsed_since_search, before_elapsed)
        self.assertEqual(matcher._last_command, before_command)
        self.assertEqual(matcher._last_terrain_class, before_terrain_class)
        self.assertEqual(state.pose_source, before.pose_source)
        self.assertEqual(state.decode_count, before.decode_count)
        self.assertEqual(state.fallback_count, before.fallback_count)
        self.assertEqual(state.joint_clamp_count, before.joint_clamp_count)
        self.assertTrue(state.candidate_exhausted)
        self.assertEqual(state.candidate_exhaustion_count, 1)
        self.assertEqual(state.candidate_retry_budget, 4)
        self.assertEqual(state.candidate_limit_rejection_count, 4)
        self.assertEqual(state.first_candidate_limit_rejection_row, 0)
        self.assertEqual(state.max_candidate_limit_rejections_per_step, 4)

    def test_explicit_25hz_corpus_retains_legacy_retry_fallback(self):
        corpus = _corpus()
        corpus.fps = 25.0
        corpus.horizons = (8, 17, 25)
        matcher = HybridMatcher(
            corpus,
            _Generator(),
            TerrainAuthority.flat(),
            native_model=_limited_native_model(),
            pose_converter=_pose_converter,
        )

        state = matcher.select_query(np.asarray(matcher.features[1], np.float64))

        self.assertFalse(state.candidate_exhausted)
        self.assertEqual(state.candidate_exhaustion_count, 0)
        self.assertEqual(state.pose_source, "canonical-fallback-joint-clamped")
        self.assertEqual(state.fallback_count, 1)
        self.assertEqual(state.joint_clamp_count, 1)

    def test_retry_exhaustion_never_uses_canonical_clamp(self):
        corpus = _corpus()
        corpus.fps = 60.0
        corpus.horizons = (20, 40, 60)
        matcher = HybridMatcher(
            corpus,
            _Generator(),
            TerrainAuthority.flat(),
            native_model=_limited_native_model(),
            pose_converter=_pose_converter,
        )

        state = matcher.select_query(np.asarray(matcher.features[1], np.float64))

        self.assertEqual(state.row, 0)
        self.assertTrue(state.candidate_exhausted)
        self.assertEqual(state.candidate_exhaustion_count, 1)
        self.assertEqual(state.fallback_count, 0)
        self.assertEqual(state.joint_clamp_count, 0)
        self.assertAlmostEqual(state.max_joint_clamp_magnitude, 0.0)
        self.assertEqual(state.candidate_limit_rejection_count, 6)
        self.assertEqual(state.first_candidate_limit_rejection_row, 1)
        self.assertEqual(state.max_candidate_limit_rejections_per_step, 6)
        self.assertAlmostEqual(state.qpos[7], 0.0)

    def test_canonical_joint_clamp_does_not_hide_invalid_root_step(self):
        fail_root = [False]

        def converter(*arguments):
            qpos = _pose_converter(*arguments)
            decoded = np.asarray(arguments[0])
            if fail_root[0] and float(decoded[0]) < 100.0:
                qpos[0] += 2.0
            return qpos

        generator = _Generator()
        matcher = HybridMatcher(
            _corpus(),
            generator,
            TerrainAuthority.flat(),
            native_model=_limited_native_model(),
            pose_converter=converter,
        )
        before = matcher.state
        fail_root[0] = True
        generator.invalid = True

        with self.assertRaisesRegex(ValueError, "root exceeded"):
            matcher.select_query(np.asarray(matcher.features[1], np.float64))

        self.assertIs(matcher.state, before)

    def test_canonical_joint_clamp_does_not_hide_nonfinite_root(self):
        fail_root = [False]

        def converter(*arguments):
            qpos = _pose_converter(*arguments)
            decoded = np.asarray(arguments[0])
            if fail_root[0] and float(decoded[0]) < 100.0:
                qpos[0] = np.nan
            return qpos

        matcher = HybridMatcher(
            _corpus(),
            _Generator(),
            TerrainAuthority.flat(),
            native_model=_limited_native_model(),
            pose_converter=converter,
        )
        before = matcher.state
        fail_root[0] = True

        with self.assertRaisesRegex(
            ValueError, "canonical source conversion is invalid"
        ):
            matcher.select_query(np.asarray(matcher.features[1], np.float64))

        self.assertIs(matcher.state, before)

    def test_keyboard_and_gamepad_commands_are_normalized(self):
        self.assertEqual(
            CommandState.from_keyboard({"w", "a"}),
            CommandState(speed=1.0, steering=1.0),
        )
        self.assertEqual(
            CommandState.from_keyboard({"w", "s", "a", "d"}), CommandState()
        )
        self.assertEqual(
            CommandState.from_gamepad(speed_axis=-2.0, steering_axis=0.5),
            CommandState(speed=1.0, steering=0.5),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            CommandState(speed=np.nan)

    def test_failed_reset_does_not_mutate_runtime_or_rebuild_generator(self):
        generator = _Generator()
        fail = [False]

        def converter(*arguments):
            if fail[0]:
                raise ValueError("conversion unavailable")
            return _pose_converter(*arguments)

        matcher = HybridMatcher(
            _corpus(), generator, TerrainAuthority.flat(), pose_converter=converter
        )
        matcher.step(CommandState(1.0, 0.5), dt=0.04, force_search=True)
        before = matcher.state
        before_root = matcher._root_xy.copy()
        before_heading = matcher._heading
        before_command = matcher._last_command
        before_elapsed = matcher._elapsed_since_search
        fail[0] = True

        with self.assertRaisesRegex(ValueError, "conversion unavailable"):
            matcher.reset()

        self.assertIs(matcher.state, before)
        np.testing.assert_array_equal(matcher._root_xy, before_root)
        self.assertEqual(matcher._heading, before_heading)
        self.assertEqual(matcher._last_command, before_command)
        self.assertEqual(matcher._elapsed_since_search, before_elapsed)
        self.assertIs(matcher.generator, generator)

    def test_failed_learned_and_canonical_transaction_holds_all_runtime_state(self):
        generator = _Generator()
        fail = [False]

        def converter(*arguments):
            if fail[0]:
                raise ValueError("no valid pose")
            return _pose_converter(*arguments)

        matcher = HybridMatcher(
            _corpus(), generator, TerrainAuthority.flat(), pose_converter=converter
        )
        before = matcher.state
        before_root = matcher._root_xy.copy()
        before_heading = matcher._heading
        fail[0] = True

        with self.assertRaisesRegex(ValueError, "no valid pose"):
            matcher.step(CommandState(1.0, 0.5), dt=0.04, force_search=True)

        self.assertIs(matcher.state, before)
        np.testing.assert_array_equal(matcher._root_xy, before_root)
        self.assertEqual(matcher._heading, before_heading)

    def test_scripted_1000_frame_smoke_is_finite_and_exercises_hybrid_path(self):
        values = np.zeros((8, 31), dtype=np.float32)
        values[:, 21:27] = np.tile((0.0, 1.0), 3)
        values[4:8, 15:21] = np.asarray((0.0, 0.45 * 0.32, 0.0, 0.45 * 0.68, 0.0, 0.45))
        values[[0, 4], 27:31] = -1.0
        values[[1, 5], 27:31] = 1.0
        matcher = HybridMatcher(
            _corpus(values),
            _Generator(),
            TerrainAuthority.multi_hill(),
            pose_converter=_pose_converter,
        )

        receipt = run_headless_smoke(matcher, frames=1_000)

        self.assertFalse(receipt["accepted"])
        self.assertEqual(
            receipt["evidence_status"],
            "runtime-only-diagnostic-not-acceptance",
        )
        self.assertFalse(receipt["acceptance_eligible"])
        self.assertTrue(receipt["diagnostic_checks_passed"])
        self.assertGreaterEqual(len(receipt["selected_ranges"]), 2)
        self.assertGreater(receipt["learned_decode_count"], 0)
        self.assertGreater(receipt["terrain_variance"], 0.0)
        self.assertGreater(receipt["tree_brute_parity_samples"], 0)
        self.assertGreater(receipt["retrieval_audit_samples"], 0)
        self.assertEqual(receipt["retrieval_audit_failures"], 0)
        self.assertGreaterEqual(receipt["range_diversity_count"], 2)
        self.assertEqual(receipt["search_scope"], "full-range-safe-corpus")
        self.assertEqual(receipt["total_safe_row_count"], len(matcher.searchable_rows))
        self.assertEqual(receipt["searched_row_count"], len(matcher.searchable_rows))
        self.assertEqual(receipt["searchable_row_count"], len(matcher.searchable_rows))
        self.assertEqual(len(receipt["search_view_sha256"]), 64)
        self.assertEqual(receipt["transition_penalty"], 0.1)
        self.assertEqual(receipt["root_motion_source"], "canonical-simulation-se2")
        self.assertEqual(receipt["fps"], matcher.fps)
        self.assertEqual(receipt["dt"], matcher.dt)
        self.assertEqual(receipt["candidate_exhaustion_count"], 0)
        self.assertEqual(receipt["candidate_limit_rejection_count"], 0)
        self.assertIsNone(receipt["first_candidate_limit_rejection_row"])
        self.assertEqual(receipt["max_candidate_limit_rejections_per_step"], 0)

    def test_headless_acceptance_rejects_short_or_capped_diagnostics(self):
        values = np.zeros((8, 31), dtype=np.float32)
        values[:, 21:27] = np.tile((0.0, 1.0), 3)
        values[4:8, 15:21] = np.asarray((0.0, 0.45 * 0.32, 0.0, 0.45 * 0.68, 0.0, 0.45))
        values[[0, 4], 27:31] = -1.0
        values[[1, 5], 27:31] = 1.0
        short = HybridMatcher(
            _corpus(values),
            _Generator(),
            TerrainAuthority.multi_hill(),
            pose_converter=_pose_converter,
        )
        capped = HybridMatcher(
            _corpus(values),
            _Generator(),
            TerrainAuthority.multi_hill(),
            pose_converter=_pose_converter,
            max_search_rows=4,
        )

        short_receipt = run_headless_smoke(short, frames=999)
        capped_receipt = run_headless_smoke(capped, frames=1_000)

        self.assertFalse(short_receipt["accepted"])
        self.assertIn("minimum-1000-frames", short_receipt["acceptance_failures"])
        self.assertFalse(capped_receipt["accepted"])
        self.assertIn("full-search-required", capped_receipt["acceptance_failures"])
        self.assertEqual(capped_receipt["search_scope"], "diagnostic-stratified-cap")


if __name__ == "__main__":
    unittest.main()
