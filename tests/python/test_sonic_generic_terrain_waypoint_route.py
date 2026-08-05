from __future__ import annotations

from pathlib import Path

import pytest

from mm_sonic import generate_generic_terrain_waypoint_route as waypoint_route


def test_waypoint_route_accumulates_events_without_turning_flat_legs_into_portals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_generate(*, start_xy, end_xy, output_dir, **_unused):
        index = int(Path(output_dir).name.rsplit("_", 1)[1])
        length = float(((end_xy[0] - start_xy[0]) ** 2 + (end_xy[1] - start_xy[1]) ** 2) ** 0.5)
        if index == 1:
            result = {
                "status": "accepted",
                "backend": "live_motionbricks_flat",
                "motion": None,
            }
            kind = "flat"
        else:
            result = {
                "status": "accepted",
                "backend": "continuous_slope_warp" if index == 0 else "curb_step_course",
                "motion": str(tmp_path / f"motion_{index}.npz"),
                "transition_count": 1 if index == 2 else 0,
            }
            kind = "slope" if index == 0 else "curb"
        return {
            "status": "accepted",
            "start_xy": list(start_xy),
            "end_xy": list(end_xy),
            "profile": {"kind": kind, "route_length_m": length},
            "result": result,
        }

    monkeypatch.setattr(waypoint_route, "generate_segment", fake_generate)
    result = waypoint_route.generate(
        terrain_usd=tmp_path / "terrain.usda",
        terrain_position=(0.0, 0.0, 0.0),
        terrain_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        waypoints_xy=((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)),
        output_dir=tmp_path / "route",
    )

    assert result["status"] == "accepted"
    assert result["profile"]["segment_count"] == 3
    assert result["profile"]["turn_count"] == 2
    events = result["result"]["events"]
    assert [event["segment_index"] for event in events] == [0, 2]
    assert events[0]["start_distance_m"] == 0.0
    assert events[1]["start_distance_m"] == pytest.approx(1.0 + 2.0**0.5)
    assert events[1]["result"]["transition_count"] == 1
    assert (tmp_path / "route" / "summary.json").is_file()


def test_waypoint_route_rejects_degenerate_leg(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 0.20 m"):
        waypoint_route.generate(
            terrain_usd=tmp_path / "terrain.usda",
            terrain_position=(0.0, 0.0, 0.0),
            terrain_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            waypoints_xy=((0.0, 0.0), (0.1, 0.0)),
            output_dir=tmp_path / "route",
        )
