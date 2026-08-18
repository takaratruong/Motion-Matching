"""Build a provenance-rich Justin S13 tracker-recovery proposal."""

from __future__ import annotations

import argparse
from pathlib import Path

from .justin_recovery import (
    DEFAULT_REWINDS_S,
    DEFAULT_TOP_K,
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
    parser.add_argument("--rewind-s", action="append", type=float)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = load_recovery_trace(args.trace)
    bank = load_reference_bank(args.reference)
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
    )
    destination = write_recovery_proposal(proposal, args.out)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
