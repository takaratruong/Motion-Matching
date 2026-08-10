from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mm_sonic.full_walking_terrain_lmm_evaluation import (
    EvaluationSeries,
    FormalRoute,
    SlipAccumulator,
    compare_baseline_candidate,
)


def _route(scene: str = "flat-standard") -> FormalRoute:
    return FormalRoute(
        scene_id=scene,
        duration_seconds=2.0,
        command_times=np.asarray((0.0, 0.5, 1.5)),
        command_speed=np.asarray((0.0, 1.0, 0.5)),
        command_steering=np.asarray((0.0, 0.25, -0.25)),
    )


def test_formal_route_samples_one_continuous_command_authority_at_each_rate() -> None:
    route = _route()

    baseline_times = route.sample_times(25.0)
    candidate_times = route.sample_times(60.0)

    assert len(baseline_times) == 51
    assert len(candidate_times) == 121
    np.testing.assert_allclose(baseline_times[[0, -1]], (0.0, 2.0))
    np.testing.assert_allclose(candidate_times[[0, -1]], (0.0, 2.0))
    speed, steering = route.commands_at(np.asarray((0.49, 0.5, 1.49, 1.5)))
    np.testing.assert_array_equal(speed, (0.0, 1.0, 1.0, 0.5))
    np.testing.assert_array_equal(steering, (0.0, 0.25, 0.25, -0.25))


def test_slip_accumulator_counts_only_continuing_contact_planar_motion() -> None:
    accumulator = SlipAccumulator()
    previous = np.zeros((2, 3), dtype=np.float64)
    current = np.asarray(((0.03, 0.04, 10.0), (2.0, 0.0, 0.0)))

    accumulator.update(
        previous,
        current,
        np.asarray((True, False)),
        np.asarray((True, True)),
        dt=0.5,
    )

    assert accumulator.planted_speeds_mps == [0.1]
    assert accumulator.summary() == {
        "sample_count": 1,
        "median_mps": 0.1,
        "p95_mps": 0.1,
    }


class _Evaluator:
    def __init__(self, fps: float, *, scale: float) -> None:
        self.fps = fps
        self.scale = scale
        self.calls: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, Path]] = []

    def evaluate_formal_route(
        self,
        route: FormalRoute,
        sample_times: np.ndarray,
        command_speed: np.ndarray,
        command_steering: np.ndarray,
        *,
        g1_xml: Path,
    ) -> EvaluationSeries:
        self.calls.append(
            (
                route.scene_id,
                sample_times.copy(),
                command_speed.copy(),
                command_steering.copy(),
                g1_xml,
            )
        )
        count = len(sample_times)
        probes = np.zeros((count, 2, 3), dtype=np.float64)
        probes[:, 0, 0] = sample_times * 0.02 * self.scale
        contacts = np.ones((count, 2), dtype=np.bool_)
        root_xy = np.column_stack((sample_times * 0.4, np.zeros(count)))
        return EvaluationSeries(
            probes=probes,
            contacts=contacts,
            root_xy=root_xy,
            desired_speed_mps=np.full(count, 0.4),
            query_distances=np.full(count, self.scale),
            canonical_source_ids=tuple(
                "source-a" if index % 2 == 0 else "source-b" for index in range(count)
            ),
            forward_count=count,
            safety_counts={},
        )


def test_comparison_reuses_routes_and_reduces_coverage_slip_and_speed() -> None:
    baseline = _Evaluator(25.0, scale=1.0)
    candidate = _Evaluator(60.0, scale=0.4)
    routes = tuple(_route(scene) for scene in ("flat-standard", "stairs-standard"))

    receipt = compare_baseline_candidate(
        baseline=baseline,
        candidate=candidate,
        routes=routes,
        g1_xml=Path("g1.xml"),
    )

    assert len(baseline.calls) == len(candidate.calls) == 2
    for route, baseline_call, candidate_call in zip(
        routes, baseline.calls, candidate.calls, strict=True
    ):
        assert baseline_call[0] == candidate_call[0]
        baseline_expected = route.commands_at(baseline_call[1])
        candidate_expected = route.commands_at(candidate_call[1])
        for index in (2, 3):
            np.testing.assert_array_equal(
                baseline_call[index], baseline_expected[index - 2]
            )
            np.testing.assert_array_equal(
                candidate_call[index], candidate_expected[index - 2]
            )
    assert receipt["baseline"]["forward_count"] == 102
    assert receipt["candidate"]["forward_count"] == 242
    assert receipt["comparison"]["query_distance_p95_reduction_fraction"] == 0.6
    assert receipt["comparison"]["slip_p95_reduction_fraction"] == pytest.approx(0.6)
    assert receipt["candidate"]["speed_mae_mps"] < 1.0e-12
    assert receipt["candidate"]["by_terrain"]["stair"]["canonical_identity_count"] == 2
    assert (
        receipt["route_command_authority_sha256"]
        == receipt["baseline"]["route_command_authority_sha256"]
    )
    assert (
        receipt["route_command_authority_sha256"]
        == receipt["candidate"]["route_command_authority_sha256"]
    )


def test_2500_candidate_frames_imply_exact_1042_baseline_frames() -> None:
    route = FormalRoute(
        scene_id="ramp-10-up-down",
        duration_seconds=2499.0 / 60.0,
        command_times=np.asarray((0.0,)),
        command_speed=np.asarray((1.0,)),
        command_steering=np.asarray((0.0,)),
    )

    assert len(route.sample_times(60.0)) == 2500
    assert len(route.sample_times(25.0)) == 1042
