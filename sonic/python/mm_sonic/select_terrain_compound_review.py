"""Select a strict, terrain-diverse visual gate for compound maneuvers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _source_features(source_motion: Path) -> np.ndarray:
    with np.load(source_motion, allow_pickle=False) as payload:
        root = np.asarray(payload["root_position_world"], dtype=np.float64)
    step = np.diff(root, axis=0)
    planar_distance = float(np.sum(np.linalg.norm(step[:, :2], axis=1)))
    vertical = root[:, 2] - root[0, 2]
    vertical_step = np.diff(vertical)
    return np.asarray(
        (
            planar_distance,
            float(np.ptp(vertical)),
            float(vertical[-1]),
            float(np.sum(np.abs(vertical_step))),
            float(np.percentile(np.abs(vertical_step), 95.0)),
        ),
        dtype=np.float64,
    )


def _farthest_order(features: np.ndarray) -> list[int]:
    values = np.asarray(features, dtype=np.float64)
    if len(values) <= 1:
        return list(range(len(values)))
    scale = np.ptp(values, axis=0)
    scale[scale < 1.0e-9] = 1.0
    normalized = (values - np.min(values, axis=0)) / scale
    selected = [int(np.argmax(np.linalg.norm(normalized - 0.5, axis=1)))]
    remaining = set(range(len(values))) - set(selected)
    while remaining:
        index = max(
            remaining,
            key=lambda candidate: min(
                float(np.linalg.norm(normalized[candidate] - normalized[kept]))
                for kept in selected
            ),
        )
        selected.append(index)
        remaining.remove(index)
    return selected


def _strict(row: dict[str, object]) -> bool:
    collision = row["collision_audit"]
    quality = row["quality"]
    mechanics = row["mechanics"]
    return bool(
        row.get("automatic_gate_accepted")
        and float(collision["maximum_foot_penetration_m"]) <= 0.005
        and float(collision["maximum_forbidden_body_penetration_m"]) <= 0.0
        and float(quality["maximum_stance_run_drift_m"]) <= 0.010
        and float(quality["maximum_stance_foot_step_m"]) <= 0.003
        and float(mechanics["maximum_joint_step_rad"]) <= 0.20
        and float(mechanics["maximum_root_acceleration_m_s2"]) <= 30.0
    )


def select(bank_root: Path, *, count: int) -> dict[str, object]:
    root = bank_root.expanduser().resolve()
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    attempted = 0
    strict_count = 0
    for manifest_path in sorted(root.glob("clip_*/pilots/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(
            (manifest_path.parent.parent / "clip_summary.json").read_text()
        )
        source_motion = manifest_path.parent.parent / "source" / "motion.npz"
        features = _source_features(source_motion)
        by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in manifest["pilots"]:
            attempted += 1
            if _strict(row):
                strict_count += 1
                by_mode[str(row["mode"])].append(row)
        for mode, rows in by_mode.items():
            # One source per mode is enough for the visual set.  Alternate
            # early/late pivots by source identity so both are represented.
            rows.sort(key=lambda value: float(value["pivot"]["terrain_progress"]))
            chosen = rows[-1] if int(summary["clip_index"]) % 2 else rows[0]
            if mode == "bounce":
                chosen = rows[0]
            candidates[mode].append(
                {
                    "manifest": manifest_path,
                    "row": chosen,
                    "summary": summary,
                    "features": features,
                }
            )

    requested = int(count)
    quotas = {
        "bounce": (requested + 1) // 2,
        "reverse": requested // 4,
        "stop_restart": requested - (requested + 1) // 2 - requested // 4,
    }
    selections: list[dict[str, object]] = []
    used_sources: dict[int, int] = defaultdict(int)
    for mode in ("bounce", "reverse", "stop_restart"):
        rows = candidates.get(mode, [])
        if not rows:
            continue
        order = _farthest_order(
            np.asarray([value["features"] for value in rows], dtype=np.float64)
        )
        for index in order:
            value = rows[index]
            source = int(value["summary"]["clip_index"])
            if used_sources[source] >= 2:
                continue
            row = value["row"]
            selections.append(
                {
                    "pilot": f"{value['manifest']}#{row['label']}",
                    "clip_index": source,
                    "clip_name": str(value["summary"]["clip_name"]),
                    "traversal": str(value["summary"]["clip_traversal"]),
                    "mode": mode,
                    "terrain_progress": float(row["pivot"]["terrain_progress"]),
                    "maximum_foot_penetration_m": float(
                        row["collision_audit"]["maximum_foot_penetration_m"]
                    ),
                    "maximum_stance_run_drift_m": float(
                        row["quality"]["maximum_stance_run_drift_m"]
                    ),
                }
            )
            used_sources[source] += 1
            if sum(value["mode"] == mode for value in selections) >= quotas[mode]:
                break
    if len(selections) < requested:
        raise ValueError(
            f"strict review requested {requested} clips but selected "
            f"{len(selections)}"
        )
    return {
        "schema": "terrain-compound-strict-review/v1",
        "bank_root": str(root),
        "attempted_pilot_count": attempted,
        "strict_pilot_count": strict_count,
        "selection_count": len(selections),
        "distinct_source_count": len(
            {int(row["clip_index"]) for row in selections}
        ),
        "quotas": quotas,
        "selections": selections,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.count < 1:
        parser.error("--count must be positive")
    payload = select(arguments.bank_root, count=arguments.count)
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(destination),
                "selection_count": payload["selection_count"],
                "distinct_source_count": payload["distinct_source_count"],
                "strict_pilot_count": payload["strict_pilot_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
