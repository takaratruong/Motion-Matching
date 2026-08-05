"""Synthesize a leave-source-out G1 stair route for an arbitrary USD mesh.

This is the globally privileged bootstrap used before the interactive runtime:
the caller supplies a terrain mesh and a commanded world-space route.  The
matcher reads only the terrain geometry along that route, retrieves the
longest compatible source-contiguous stair blocks, warps them to the mesh, and
accepts a composition only after the exact G1 collision and continuity audit.
No target scene identifier or target robot motion is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_coherent_block_plan import compose
from .evaluate_coherent_block_coverage import (
    DEFAULT_FRAGMENT_BANK,
    evaluate,
)
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE
from .reconstruct_stair_fragment_route import (
    _parser as _fragment_parser,
    run as run_fragment_reconstruction,
    write_reconstruction_bundle,
    write_rejection_bundle,
)
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.source_grail import _load_usd_mesh
from .terrain_oracle.stair_geometry_warp import (
    DEFAULT_MOTION_ROUTE_PROFILE_CATALOG,
)
from .terrain_oracle.stair_fragment_reconstruction import (
    PlannedFragmentReconstructionRejected,
)
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stair_support_route import sample_stair_support_route
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


def load_target_mesh(
    usd_path: str | Path,
    *,
    position_world: object = (0.0, 0.0, 0.0),
    quaternion_world_from_usd_wxyz: object = (1.0, 0.0, 0.0, 0.0),
) -> TerrainMeshIndex:
    """Load one USD terrain asset at an arbitrary world transform."""

    path = Path(usd_path).expanduser().resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mesh = _load_usd_mesh(path, source_asset_sha256=digest)
    transform = RigidTransform(
        np.asarray(position_world, dtype=np.float32),
        np.asarray(quaternion_world_from_usd_wxyz, dtype=np.float32),
    )
    return TerrainMeshIndex(mesh, transform)


def _request_payload(
    *,
    terrain_usd: Path,
    terrain_position: tuple[float, float, float],
    terrain_quaternion_wxyz: tuple[float, float, float, float],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    excluded_source_clip_indices: tuple[int, ...],
    maximum_block_transitions: int,
    coverage_candidates_per_span: int,
    composition_candidates_per_span: int,
    accepted_candidates_per_span: int,
    maximum_candidate_chains: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "generic-stair-route-request/v1",
        "terrain_usd": str(terrain_usd),
        "terrain_sha256": hashlib.sha256(terrain_usd.read_bytes()).hexdigest(),
        "terrain_position_world": list(terrain_position),
        "terrain_quaternion_world_from_usd_wxyz": list(
            terrain_quaternion_wxyz
        ),
        "start_xy": list(start_xy),
        "end_xy": list(end_xy),
        "excluded_source_clip_indices": sorted(
            int(value) for value in excluded_source_clip_indices
        ),
        "maximum_block_transitions": int(maximum_block_transitions),
        "coverage_candidates_per_span": int(coverage_candidates_per_span),
        "composition_candidates_per_span": int(
            composition_candidates_per_span
        ),
        "accepted_candidates_per_span": int(accepted_candidates_per_span),
        "maximum_candidate_chains": int(maximum_candidate_chains),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["request_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _run_fragment_fallback(
    *,
    target_mesh: TerrainMeshIndex,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    output_dir: Path,
    archive_path: Path,
    model_path: Path,
    fragment_bank: Path,
    excluded_source_clip_indices: tuple[int, ...],
) -> tuple[dict[str, object], object | None]:
    """Run the validated compact-riser fallback on the same arbitrary mesh."""

    parsed = _fragment_parser().parse_args(
        [
            "--output",
            str(output_dir),
            # The parser retains the archive-evaluation option, but the value
            # is cleared below so no target clip supplies runtime geometry.
            "--target-clip-index",
            "0",
            "--start-xy",
            str(start_xy[0]),
            str(start_xy[1]),
            "--end-xy",
            str(end_xy[0]),
            str(end_xy[1]),
            "--maximum-plans",
            "48",
            "--planner-pool-plans",
            "192",
            "--preserve-ranked-plans",
            "8",
            "--diversify-boundaries",
            "--beam-width",
            "256",
            "--decay-frames",
            "16.0",
            "--maximum-foot-target-error-m",
            "0.004",
            "--maximum-swing-foot-target-error-m",
            "0.11",
            "--maximum-root-acceleration-m-s2",
            "40.0",
            "--maximum-full-foot-penetration-m",
            "0.005",
            "--maximum-forbidden-body-penetration-m",
            "0.0",
        ]
    )
    arguments = argparse.Namespace(
        **{
            **vars(parsed),
            "archive": archive_path,
            "fragment_bank": fragment_bank,
            "model": model_path,
            "target_clip_index": None,
        }
    )
    accepted_audits: dict[int, object] = {}

    def validate(reconstruction: object) -> None:
        audit = audit_stair_motion_collisions(
            reconstruction.motion,
            archive_path=archive_path,
            target_mesh=target_mesh,
            model_path=model_path,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=0.0,
        )
        accepted_audits[id(reconstruction)] = audit
        if not audit.accepted:
            raise PlannedFragmentReconstructionRejected(
                "full_body_collision",
                "complete-foot penetration "
                f"{audit.maximum_foot_penetration_m:.6f} m; "
                "forbidden-body penetration "
                f"{audit.maximum_forbidden_body_penetration_m:.6f} m",
            )

    summary, reconstruction = run_fragment_reconstruction(
        arguments,
        target_mesh=target_mesh,
        excluded_source_clip_indices=excluded_source_clip_indices,
        validate_reconstruction=validate,
    )
    if reconstruction is None:
        write_rejection_bundle(output_dir, summary)
        return summary, None
    audit = accepted_audits[id(reconstruction)]
    summary["full_body_collision_audit"] = audit.to_dict()
    write_reconstruction_bundle(output_dir, reconstruction, summary)
    return summary, reconstruction


def generate(
    *,
    terrain_usd: Path,
    terrain_position: tuple[float, float, float],
    terrain_quaternion_wxyz: tuple[float, float, float, float],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    output_dir: Path,
    archive_path: Path = DEFAULT_STAIRS_ARCHIVE,
    model_path: Path = DEFAULT_G1_MJCF,
    fragment_bank: Path = DEFAULT_FRAGMENT_BANK,
    route_catalog: Path = DEFAULT_MOTION_ROUTE_PROFILE_CATALOG,
    excluded_source_clip_indices: tuple[int, ...] = (),
    maximum_block_transitions: int = 7,
    coverage_candidates_per_span: int = 24,
    composition_candidates_per_span: int = 32,
    accepted_candidates_per_span: int = 4,
    maximum_candidate_chains: int = 32,
    ground_fallback_height_m: float = 0.0,
    sole_half_length_m: float = 0.10,
    sole_half_width_m: float = 0.055,
    route_sample_spacing_m: float = 0.01,
) -> dict[str, object]:
    """Generate or reuse one exact-audited arbitrary-mesh stair traversal."""

    terrain_path = terrain_usd.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    request = _request_payload(
        terrain_usd=terrain_path,
        terrain_position=terrain_position,
        terrain_quaternion_wxyz=terrain_quaternion_wxyz,
        start_xy=start_xy,
        end_xy=end_xy,
        excluded_source_clip_indices=excluded_source_clip_indices,
        maximum_block_transitions=maximum_block_transitions,
        coverage_candidates_per_span=coverage_candidates_per_span,
        composition_candidates_per_span=composition_candidates_per_span,
        accepted_candidates_per_span=accepted_candidates_per_span,
        maximum_candidate_chains=maximum_candidate_chains,
    )
    request_path = destination / "request.json"
    composition_summary_path = destination / "composition" / "summary.json"
    top_level_summary_path = destination / "summary.json"
    if request_path.is_file() and top_level_summary_path.is_file():
        previous_request = json.loads(request_path.read_text())
        previous_summary = json.loads(top_level_summary_path.read_text())
        if (
            previous_request.get("request_sha256") == request["request_sha256"]
            and previous_summary.get("status") == "accepted"
        ):
            return previous_summary

    target_mesh = load_target_mesh(
        terrain_path,
        position_world=terrain_position,
        quaternion_world_from_usd_wxyz=terrain_quaternion_wxyz,
    )
    route = sample_stair_support_route(
        target_mesh,
        start_xy,
        end_xy,
        ground_fallback_height_m,
        sole_half_length_m,
        sole_half_width_m,
        sample_spacing_m=route_sample_spacing_m,
    )
    destination.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    coverage = evaluate(
        archive_path=archive_path.expanduser().resolve(),
        model_path=model_path.expanduser().resolve(),
        fragment_bank=fragment_bank.expanduser().resolve(),
        route_catalog=route_catalog.expanduser().resolve(),
        target_clip_index=None,
        target_mesh=target_mesh,
        target_route=route,
        target_name=terrain_path.stem,
        excluded_source_clip_indices=excluded_source_clip_indices,
        maximum_block_transitions=maximum_block_transitions,
        candidate_limit_per_span=coverage_candidates_per_span,
        output_dir=destination / "coverage",
        exclude_target_source=False,
    )
    result = None
    if bool(coverage["supported"]):
        result = compose(
            coverage_summary=destination / "coverage" / "summary.json",
            archive_path=archive_path.expanduser().resolve(),
            model_path=model_path.expanduser().resolve(),
            fragment_bank=fragment_bank.expanduser().resolve(),
            route_catalog=route_catalog.expanduser().resolve(),
            candidate_limit_per_span=composition_candidates_per_span,
            accepted_candidates_per_span=accepted_candidates_per_span,
            maximum_candidate_chains=maximum_candidate_chains,
            output_dir=destination / "composition",
            render=False,
            target_mesh=target_mesh,
            target_name=terrain_path.stem,
        )
    if result is not None and result["status"] == "accepted":
        top_level = {
            "schema": "generic-stair-route/v1",
            "status": "accepted",
            "method": "source_contiguous_coherent_blocks",
            "request": request,
            "coverage_summary": str(
                (destination / "coverage" / "summary.json").resolve()
            ),
            "composition_summary": str(composition_summary_path.resolve()),
            "motion": str(
                (destination / "composition" / "motion.npz").resolve()
            ),
        }
        top_level_summary_path.write_text(
            json.dumps(top_level, indent=2, sort_keys=True) + "\n"
        )
        return top_level

    fallback_dir = destination / "fragment_fallback"
    fallback_summary, fallback_motion = _run_fragment_fallback(
        target_mesh=target_mesh,
        start_xy=start_xy,
        end_xy=end_xy,
        output_dir=fallback_dir,
        archive_path=archive_path.expanduser().resolve(),
        model_path=model_path.expanduser().resolve(),
        fragment_bank=fragment_bank.expanduser().resolve(),
        excluded_source_clip_indices=excluded_source_clip_indices,
    )
    top_level = {
        "schema": "generic-stair-route/v1",
        "status": "accepted" if fallback_motion is not None else "rejected",
        "method": (
            "compact_riser_fragments" if fallback_motion is not None else None
        ),
        "request": request,
        "coverage_summary": str(
            (destination / "coverage" / "summary.json").resolve()
        ),
        "coherent_composition_summary": (
            str(composition_summary_path.resolve())
            if result is not None
            else None
        ),
        "coherent_status": (
            result["status"] if result is not None else "no_coherent_coverage"
        ),
        "fragment_fallback_summary": str(
            (fallback_dir / "summary.json").resolve()
        ),
        "motion": (
            str((fallback_dir / "motion.npz").resolve())
            if fallback_motion is not None
            else None
        ),
    }
    top_level_summary_path.write_text(
        json.dumps(top_level, indent=2, sort_keys=True) + "\n"
    )
    return top_level


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-usd", type=Path, required=True)
    parser.add_argument("--terrain-position", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument(
        "--terrain-quaternion-wxyz",
        type=float,
        nargs=4,
        default=(1, 0, 0, 0),
    )
    parser.add_argument("--start-xy", type=float, nargs=2, required=True)
    parser.add_argument("--end-xy", type=float, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--fragment-bank", type=Path, default=DEFAULT_FRAGMENT_BANK)
    parser.add_argument(
        "--route-catalog",
        type=Path,
        default=DEFAULT_MOTION_ROUTE_PROFILE_CATALOG,
    )
    parser.add_argument(
        "--exclude-source-clip-index",
        action="append",
        type=int,
        default=[],
    )
    parser.add_argument("--maximum-block-transitions", type=int, default=7)
    parser.add_argument("--coverage-candidates-per-span", type=int, default=24)
    parser.add_argument("--composition-candidates-per-span", type=int, default=32)
    parser.add_argument("--accepted-candidates-per-span", type=int, default=4)
    parser.add_argument("--maximum-candidate-chains", type=int, default=32)
    parser.add_argument("--ground-fallback-height-m", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = generate(
        terrain_usd=arguments.terrain_usd,
        terrain_position=tuple(arguments.terrain_position),
        terrain_quaternion_wxyz=tuple(arguments.terrain_quaternion_wxyz),
        start_xy=tuple(arguments.start_xy),
        end_xy=tuple(arguments.end_xy),
        output_dir=arguments.output_dir,
        archive_path=arguments.archive,
        model_path=arguments.model,
        fragment_bank=arguments.fragment_bank,
        route_catalog=arguments.route_catalog,
        excluded_source_clip_indices=tuple(arguments.exclude_source_clip_index),
        maximum_block_transitions=arguments.maximum_block_transitions,
        coverage_candidates_per_span=arguments.coverage_candidates_per_span,
        composition_candidates_per_span=(
            arguments.composition_candidates_per_span
        ),
        accepted_candidates_per_span=arguments.accepted_candidates_per_span,
        maximum_candidate_chains=arguments.maximum_candidate_chains,
        ground_fallback_height_m=arguments.ground_fallback_height_m,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
