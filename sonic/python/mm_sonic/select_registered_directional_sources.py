"""Select diverse exact-grounded sources for continuous-terrain direction warps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .select_terrain_compound_review import _farthest_order, _source_features


def select(bank_root: Path, *, count: int) -> dict[str, object]:
    root = bank_root.expanduser().resolve()
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("clip_*/clip_summary.json")):
        summary = json.loads(summary_path.read_text())
        source_motion = summary_path.parent / "source" / "motion.npz"
        if (
            summary.get("source_report", {}).get("status") != "accepted"
            or not source_motion.is_file()
        ):
            continue
        rows.append(
            {
                "clip_index": int(summary["clip_index"]),
                "clip_name": str(summary["clip_name"]),
                "clip_traversal": str(summary["clip_traversal"]),
                "source_motion": str(source_motion.resolve()),
                "features": _source_features(source_motion),
            }
        )
    if len(rows) < int(count):
        raise ValueError(
            f"requested {count} sources but only {len(rows)} are exact-grounded"
        )
    order = _farthest_order(
        np.asarray([row["features"] for row in rows], dtype=np.float64)
    )
    selected = []
    for rank, index in enumerate(order[: int(count)]):
        row = dict(rows[index])
        row.pop("features")
        row["selection_rank"] = rank
        selected.append(row)
    return {
        "schema": "registered-continuous-terrain-source-selection/v1",
        "bank_root": str(root),
        "source_count": len(selected),
        "sources": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.count < 1:
        parser.error("--count must be positive")
    payload = select(arguments.bank_root, count=arguments.count)
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "source_count": len(payload["sources"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
