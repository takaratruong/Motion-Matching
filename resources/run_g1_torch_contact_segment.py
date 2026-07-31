#!/usr/bin/env python3
"""Run and save the deterministic G1 contact-segment terrain experiment."""

from __future__ import annotations

import argparse
import json

from mm_sonic.torch_contact_segment_rollout import (
    load_contact_segment_config,
    resolve_contact_segment_config,
    run_contact_segment_rollout,
    save_contact_segment_rollout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved = resolve_contact_segment_config(
        args.dataset,
        load_contact_segment_config(args.config),
        device=args.device,
    )
    rollout = run_contact_segment_rollout(resolved, args.g1_xml)
    save_contact_segment_rollout(rollout, args.output)
    print(
        json.dumps(
            {
                "accepted": rollout.accepted,
                "metrics": dict(rollout.metrics),
                "output": args.output,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if rollout.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
