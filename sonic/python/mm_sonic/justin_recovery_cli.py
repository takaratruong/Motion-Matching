"""Build a provenance-rich Justin S13 tracker-recovery proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .justin_recovery import (
    DEFAULT_REWINDS_S,
    DEFAULT_TOP_K,
    DEFAULT_S13_SCENE,
    RecoverySceneContract,
    build_recovery_proposal,
    load_recovery_trace,
    load_reference_bank,
    write_recovery_proposal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    failure = parser.add_mutually_exclusive_group(required=True)
    failure.add_argument("--first-fall-physics-step", type=int)
    failure.add_argument(
        "--failure-physics-step",
        type=int,
        help="terminal failure/degradation step when no fall was formally confirmed",
    )
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument(
        "--scene-contract",
        type=Path,
        help="optional JSON planar/stair contract for a non-Justin scene",
    )
    parser.add_argument("--allow-single-reference", action="store_true")
    parser.add_argument("--rewind-s", action="append", type=float)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = load_recovery_trace(args.trace)
    bank = load_reference_bank(
        args.reference, minimum_clips=1 if args.allow_single_reference else 4
    )
    scene = DEFAULT_S13_SCENE
    if args.scene_contract is not None:
        payload = json.loads(args.scene_contract.resolve().read_text())
        scene = RecoverySceneContract(
            scene_id=str(payload["scene_id"]),
            learner_front_xy=tuple(payload["learner_front_xy"]),
            reference_front_xy=tuple(payload["reference_front_xy"]),
            learner_heading_yaw_rad=float(payload["learner_heading_yaw_rad"]),
            reference_heading_yaw_rad=float(payload["reference_heading_yaw_rad"]),
            tread_m=float(payload["tread_m"]),
            num_steps=int(payload["num_steps"]),
        )
    proposal = build_recovery_proposal(
        trace,
        bank,
        first_fall_physics_step=(
            args.first_fall_physics_step
            if args.first_fall_physics_step is not None
            else args.failure_physics_step
        ),
        failure_source=(
            "reported_first_fall"
            if args.first_fall_physics_step is not None
            else "terminal_degradation"
        ),
        rewinds_s=tuple(args.rewind_s or DEFAULT_REWINDS_S),
        top_k=args.top_k,
        scene=scene,
    )
    if args.scene_contract is not None and "evaluation_geometry" in payload:
        proposal["evaluation_geometry"] = payload["evaluation_geometry"]
    destination = write_recovery_proposal(proposal, args.out)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
