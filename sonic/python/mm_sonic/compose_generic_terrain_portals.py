"""Compile a generated terrain route into MotionBricks runtime portal courses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_motionbricks_terrain_course import compose
from .generate_generic_stair_route import load_target_mesh


DEFAULT_APPROACH = Path(
    "/move/u/bodow/Projects/Motion-Matching-takara-corpus/artifacts/"
    "global_scene_terrain/motionbricks_course_target26_angle_neg45_v1/"
    "generation/approach_raw.npz"
)
DEFAULT_EXIT = Path(
    "/move/u/bodow/Projects/Motion-Matching-takara-corpus/artifacts/"
    "global_scene_terrain/motionbricks_course_target26_pos45_v1/"
    "reused_generation/exit_raw.npz"
)
DEFAULT_DESCENDING_EXITS = (
    Path(
        "/move/u/bodow/Projects/Motion-Matching-takara-corpus/artifacts/"
        "global_scene_terrain/motionbricks_course_target97_down_angle0_v1/"
        "generation/exit_raw.npz"
    ),
    Path(
        "/move/u/bodow/Projects/Motion-Matching-takara-corpus/artifacts/"
        "global_scene_terrain/motionbricks_course_target26_pos45_v1/"
        "reused_generation/exit_raw.npz"
    ),
    Path(
        "/move/u/bodow/Projects/Motion-Matching-takara-corpus/artifacts/"
        "global_scene_terrain/motionbricks_course_target26_exact_pos45_v1/"
        "generation/exit_raw.npz"
    ),
    Path(
        "/move/u/bodow/Projects/Motion-Matching-takara-corpus/artifacts/"
        "global_scene_terrain/motionbricks_course_cpu_smoke_v1/"
        "generation/exit_raw.npz"
    ),
)


def _generated_events(summary: dict[str, object]) -> list[dict[str, object]]:
    result = summary["result"]
    if not isinstance(result, dict):
        raise ValueError("route summary has no result object")
    if "events" in result:
        events = result["events"]
        if not isinstance(events, list):
            raise ValueError("route events must be a list")
        return [dict(event) for event in events]
    return [
        {
            "event_index": 0,
            "kind": summary["profile"]["kind"],
            "backend": result.get("backend"),
            "start_distance_m": 0.0,
            "end_distance_m": summary["profile"]["route_length_m"],
            "start_xy": summary["start_xy"],
            "end_xy": summary["end_xy"],
            "result": result,
        }
    ]


def _primitive_candidates(event: dict[str, object]) -> tuple[Path, ...]:
    result = event.get("result")
    if not isinstance(result, dict) or result.get("status") != "accepted":
        return ()
    motion = result.get("motion")
    if motion is None:
        return ()
    primary = Path(str(motion)).expanduser().resolve()
    alternatives = (
        sorted(primary.parent.glob("transition_*candidate_*.npz"))
        if int(result.get("transition_count", 0)) == 1
        else []
    )
    return tuple(dict.fromkeys((primary, *alternatives)))


def _primitive_descends(path: Path, *, minimum_drop_m: float = 0.03) -> bool:
    """Return whether the authored terrain traversal ends below its start."""

    with np.load(path, allow_pickle=False) as arrays:
        root = np.asarray(arrays["root_position_world"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1:] != (3,) or len(root) < 2:
        raise ValueError(
            f"terrain primitive has invalid root trajectory: {path}"
        )
    return bool(float(root[-1, 2] - root[0, 2]) < -float(minimum_drop_m))


def _exit_candidates(
    primitive: Path,
    *,
    exit_raw: Path,
    descending_exit_raws: tuple[Path, ...],
) -> tuple[Path, ...]:
    if not _primitive_descends(primitive):
        return (exit_raw,)
    # A flat recording is a reusable gait-phase bank: its original global
    # placement is discarded by the course composer.  Descending landings are
    # sufficiently different from elevated ascent exits that searching only
    # one recording can miss a compatible support phase by centimetres.
    return tuple(dict.fromkeys((*descending_exit_raws, exit_raw)))


def compile_portals(
    *,
    route_summary: Path,
    output_dir: Path,
    approach_raw: Path = DEFAULT_APPROACH,
    exit_raw: Path = DEFAULT_EXIT,
    descending_exit_raws: tuple[Path, ...] = DEFAULT_DESCENDING_EXITS,
    motion_archive: Path = DEFAULT_ARCHIVE,
    model_path: Path = DEFAULT_G1_MJCF,
    maximum_primitive_candidates: int = 3,
) -> dict[str, object]:
    source = route_summary.expanduser().resolve()
    route = json.loads(source.read_text())
    if route.get("status") != "accepted":
        raise ValueError("cannot compile portals from a rejected terrain route")
    terrain_usd = Path(str(route["terrain_usd"])).expanduser().resolve()
    terrain_position = tuple(float(v) for v in route["terrain_position_world"])
    terrain_quaternion = tuple(
        float(v) for v in route["terrain_quaternion_world_from_usd_wxyz"]
    )
    mesh = load_target_mesh(
        terrain_usd,
        position_world=terrain_position,
        quaternion_world_from_usd_wxyz=terrain_quaternion,
    )
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    compiled: list[dict[str, object]] = []
    events = _generated_events(route)
    for event_position, event in enumerate(events):
        event_index = int(event["event_index"])
        terminal_event = event_position == len(events) - 1
        candidates = _primitive_candidates(event)[: int(maximum_primitive_candidates)]
        attempts: list[dict[str, object]] = []
        selected: dict[str, object] | None = None
        # End each portal on its exact-safe authored landing and let the
        # runtime phase-match a live MotionBricks continuation from the actual
        # final four frames.  A prerecorded exit both delays steering and can
        # carry the character off a finite landing before the next waypoint
        # command takes effect.  Keep the old two-seam course only as a compile
        # fallback when an entry-only seam has no physically valid phase.
        if candidates:
            for phase_limit, combination_limit in ((4, 8), (16, 128)):
                for candidate_index, primitive in enumerate(candidates):
                    candidate_output = (
                        destination
                        / f"event_{event_index:02d}"
                        / (
                            f"primitive_{candidate_index:02d}_entry_only_"
                            f"phase_{phase_limit:02d}"
                        )
                    )
                    try:
                        cached_summary = candidate_output / "summary.json"
                        if cached_summary.is_file():
                            result = json.loads(cached_summary.read_text())
                        else:
                            result = compose(
                                approach_raw=approach_raw.expanduser().resolve(),
                                exit_raw=exit_raw.expanduser().resolve(),
                                stair_motion=primitive,
                                stairs_archive=motion_archive.expanduser().resolve(),
                                target_clip_index=None,
                                target_mesh=mesh,
                                model_path=model_path.expanduser().resolve(),
                                output_dir=candidate_output,
                                phase_candidate_limit=phase_limit,
                                maximum_candidate_combinations=(
                                    combination_limit
                                ),
                                render=False,
                                include_exit=False,
                            )
                        row = {
                            "primitive_motion": str(primitive),
                            "exit_raw": None,
                            "runtime_live_exit": True,
                            "terminal_event": terminal_event,
                            "phase_candidate_limit": phase_limit,
                            "maximum_candidate_combinations": combination_limit,
                            "maximum_seam_foot_error_m": 0.018,
                            "status": result["status"],
                            "summary": str(candidate_output / "summary.json"),
                        }
                        attempts.append(row)
                        if result["status"] == "accepted":
                            selected = {
                                **row,
                                "course_motion": str(
                                    (
                                        candidate_output
                                        / str(result["artifacts"]["motion"])
                                    ).resolve()
                                ),
                            }
                            break
                    except (RuntimeError, ValueError) as error:
                        attempts.append(
                            {
                                "primitive_motion": str(primitive),
                                "exit_raw": None,
                                "runtime_live_exit": True,
                                "terminal_event": terminal_event,
                                "phase_candidate_limit": phase_limit,
                                "maximum_candidate_combinations": (
                                    combination_limit
                                ),
                                "maximum_seam_foot_error_m": 0.018,
                                "status": "rejected",
                                "reason": str(error),
                            }
                        )
                if selected is not None:
                    break
        # Prefer source diversity over an exhaustive phase search on one
        # primitive.  Only if every retained primitive misses do we broaden
        # the flat-phase search.
        search_tiers = () if selected is not None else (
            (4, 8, 0.018, None),
            (16, 128, 0.018, None),
            # The seam error is an IK-convergence guard, not the physical
            # acceptance gate.  A final 1 mm band is allowed only for the
            # ordinary exit recording; mechanics and the unchanged 5 mm foot
            # / zero body exact-mesh audit still decide acceptance below.
            (16, 128, 0.019, exit_raw.expanduser().resolve()),
        )
        for (
            phase_limit,
            combination_limit,
            seam_foot_error_m,
            only_exit_raw,
        ) in search_tiers:
            for candidate_index, primitive in enumerate(candidates):
                exit_candidates = _exit_candidates(
                    primitive,
                    exit_raw=exit_raw,
                    descending_exit_raws=descending_exit_raws,
                )
                for exit_index, selected_exit_raw in enumerate(exit_candidates):
                    resolved_exit_raw = selected_exit_raw.expanduser().resolve()
                    if (
                        only_exit_raw is not None
                        and resolved_exit_raw != only_exit_raw
                    ):
                        continue
                    phase_label = f"phase_{phase_limit:02d}"
                    if len(exit_candidates) > 1:
                        phase_label = f"exit_{exit_index:02d}_{phase_label}"
                    if seam_foot_error_m > 0.018:
                        phase_label = f"{phase_label}_ik19"
                    candidate_output = (
                        destination
                        / f"event_{event_index:02d}"
                        / f"primitive_{candidate_index:02d}_{phase_label}"
                    )
                    try:
                        cached_summary = candidate_output / "summary.json"
                        if cached_summary.is_file():
                            result = json.loads(cached_summary.read_text())
                        else:
                            result = compose(
                                approach_raw=approach_raw.expanduser().resolve(),
                                exit_raw=resolved_exit_raw,
                                stair_motion=primitive,
                                stairs_archive=(
                                    motion_archive.expanduser().resolve()
                                ),
                                target_clip_index=None,
                                target_mesh=mesh,
                                model_path=model_path.expanduser().resolve(),
                                output_dir=candidate_output,
                                phase_candidate_limit=phase_limit,
                                maximum_candidate_combinations=combination_limit,
                                maximum_seam_foot_error_m=seam_foot_error_m,
                                render=False,
                            )
                        row = {
                            "primitive_motion": str(primitive),
                            "exit_raw": str(
                                resolved_exit_raw
                            ),
                            "phase_candidate_limit": phase_limit,
                            "maximum_candidate_combinations": combination_limit,
                            "maximum_seam_foot_error_m": seam_foot_error_m,
                            "status": result["status"],
                            "summary": str(candidate_output / "summary.json"),
                        }
                        attempts.append(row)
                        if result["status"] == "accepted":
                            course = candidate_output / str(
                                result["artifacts"]["motion"]
                            )
                            selected = {
                                **row,
                                "course_motion": str(course.resolve()),
                            }
                            break
                    except (RuntimeError, ValueError) as error:
                        attempts.append(
                            {
                                "primitive_motion": str(primitive),
                                "exit_raw": str(
                                    resolved_exit_raw
                                ),
                                "phase_candidate_limit": phase_limit,
                                "maximum_candidate_combinations": (
                                    combination_limit
                                ),
                                "maximum_seam_foot_error_m": (
                                    seam_foot_error_m
                                ),
                                "status": "rejected",
                                "reason": str(error),
                            }
                        )
                if selected is not None:
                    break
            if selected is not None:
                break
        compiled.append(
            {
                "event_index": event_index,
                "kind": event["kind"],
                "backend": event["backend"],
                "start_distance_m": event["start_distance_m"],
                "end_distance_m": event["end_distance_m"],
                "start_xy": event["start_xy"],
                "end_xy": event["end_xy"],
                "status": "accepted" if selected is not None else "rejected",
                "selected": selected,
                "attempts": attempts,
            }
        )
    accepted = bool(compiled) and all(
        event["status"] == "accepted" for event in compiled
    )
    manifest = {
        "schema": "generic-terrain-portal-manifest/v1",
        "status": "accepted" if accepted else "rejected",
        "route_summary": str(source),
        "terrain_usd": str(terrain_usd),
        "terrain_position_world": list(terrain_position),
        "terrain_quaternion_world_from_usd_wxyz": list(terrain_quaternion),
        "events": compiled,
        "course_motions": [
            event["selected"]["course_motion"]
            for event in compiled
            if event["selected"] is not None
        ],
    }
    (destination / "portal_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approach-raw", type=Path, default=DEFAULT_APPROACH)
    parser.add_argument("--exit-raw", type=Path, default=DEFAULT_EXIT)
    parser.add_argument(
        "--descending-exit-raw",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--motion-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--maximum-primitive-candidates", type=int, default=3)
    arguments = parser.parse_args(argv)
    result = compile_portals(
        route_summary=arguments.route_summary,
        output_dir=arguments.output_dir,
        approach_raw=arguments.approach_raw,
        exit_raw=arguments.exit_raw,
        descending_exit_raws=(
            DEFAULT_DESCENDING_EXITS
            if arguments.descending_exit_raw is None
            else tuple(arguments.descending_exit_raw)
        ),
        motion_archive=arguments.motion_archive,
        model_path=arguments.model_path,
        maximum_primitive_candidates=arguments.maximum_primitive_candidates,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
