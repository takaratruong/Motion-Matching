"""Select a balanced, diverse bank from admitted stairs500 maneuvers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE
from .select_stairs500_maneuver_sources import _farthest_point_order


DEFAULT_GROUPS = (
    ("up", "stop_restart"),
    ("up", "reverse"),
    ("down", "stop_restart"),
    ("down", "reverse"),
)


def select_admitted(
    bank_root: str | Path,
    archive_path: str | Path,
    *,
    per_group: int = 50,
) -> dict[str, object]:
    """Choose equal up/down and stop/reverse coverage from a generated bank."""

    import zarr

    bank_root = Path(bank_root).expanduser().resolve()
    archive_path = Path(archive_path).expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    family = np.asarray(archive["clip_family"][:]).astype(str)
    family_names = tuple(sorted(set(family.tolist())))
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for manifest_path in sorted(bank_root.glob("clip_*/pilots/manifest.json")):
        summary_path = manifest_path.parent.parent / "clip_summary.json"
        summary = json.loads(summary_path.read_text())
        index = int(summary["clip_index"])
        manifest = json.loads(manifest_path.read_text())
        features = (
            float(family_names.index(str(family[index]))),
            float(archive["stair_n_steps"][index]),
            float(archive["stair_rise_m"][index]),
            float(archive["stair_tread_m"][index]),
            float(archive["clip_approach_heading_delta_deg"][index]),
            abs(float(archive["root_delta_z_m"][index])),
        )
        for pilot in manifest.get("pilots", ()):
            if not bool(pilot.get("automatic_gate_accepted", False)):
                continue
            mode = str(pilot["mode"])
            traversal = str(summary["clip_traversal"])
            groups[(traversal, mode)].append(
                {
                    "pilot": f"{manifest_path}#{pilot['label']}",
                    "clip_index": index,
                    "clip_name": str(summary["clip_name"]),
                    "clip_family": str(summary["clip_family"]),
                    "clip_traversal": traversal,
                    "mode": mode,
                    "features": features,
                    "maximum_foot_penetration_m": float(
                        pilot["collision_audit"]["maximum_foot_penetration_m"]
                    ),
                    "maximum_stance_foot_step_m": float(
                        pilot["quality"]["maximum_stance_foot_step_m"]
                    ),
                    "maximum_stance_run_drift_m": float(
                        pilot["quality"]["maximum_stance_run_drift_m"]
                    ),
                }
            )

    selections: list[dict[str, object]] = []
    available: dict[str, int] = {}
    for traversal, mode in DEFAULT_GROUPS:
        candidates = groups[(traversal, mode)]
        key = f"{traversal}|{mode}"
        available[key] = len(candidates)
        if len(candidates) < per_group:
            raise ValueError(
                f"{key} has {len(candidates)} admitted pilots, fewer than "
                f"requested {per_group}"
            )
        order = _farthest_point_order(
            np.asarray([row["features"] for row in candidates], dtype=np.float64)
        )
        for group_rank, candidate_index in enumerate(order[:per_group]):
            row = dict(candidates[candidate_index])
            row.pop("features")
            row["group_rank"] = group_rank
            row["selection_rank"] = len(selections)
            selections.append(row)

    return {
        "schema": "stairs500-admitted-maneuver-selection/v1",
        "bank_root": str(bank_root),
        "archive": str(archive_path),
        "per_group": int(per_group),
        "available": available,
        "selection_count": len(selections),
        "selections": selections,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--per-group", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.per_group < 1:
        parser.error("--per-group must be positive")
    result = select_admitted(
        arguments.bank_root,
        arguments.archive,
        per_group=arguments.per_group,
    )
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "selection_count": result["selection_count"],
                "available": result["available"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
