"""Select a behavior-diverse strict video review for compound stair motions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


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


def _pivot_bin(row: dict[str, object]) -> str:
    progress = float(row["pivot"]["terrain_progress"])
    if progress < 1.0 / 3.0:
        return "lower"
    if progress < 2.0 / 3.0:
        return "middle"
    return "upper"


def select(
    source_selection: Path,
    bank_root: Path,
    *,
    count: int,
    maximum_per_physical_source: int = 1,
) -> dict[str, object]:
    """Greedily cover behavior, maneuver, traversal, and progress categories."""

    source_rows = json.loads(source_selection.expanduser().resolve().read_text())
    root = bank_root.expanduser().resolve()
    candidates: list[dict[str, object]] = []
    automatic_count = 0
    for rank, source_row in enumerate(source_rows):
        source_report_path = Path(str(source_row["report"])).expanduser().resolve()
        source_report = json.loads(source_report_path.read_text())
        warp = source_report.get("warp", {})
        path_angle_deg = float(warp.get("maximum_path_angle_deg", 0.0))
        facing_offset_deg = float(warp.get("maximum_facing_offset_deg", 0.0))
        lateral_range_m = float(warp.get("lateral_offset_range_m", 0.0))
        # One normalized intensity score keeps the visual review from filling
        # every behavior family with its easiest micro/gentle member.  The
        # categorical terms below still guarantee broad behavior coverage.
        source_intensity = sum(
            (
                path_angle_deg / 18.0,
                facing_offset_deg / 12.0,
                lateral_range_m / 0.40,
            )
        )
        manifest_path = (
            root
            / f"clip_{rank:03d}_{source_report['label']}"
            / "pilots"
            / "manifest.json"
        )
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for pilot in manifest["pilots"]:
            automatic_count += int(bool(pilot.get("automatic_gate_accepted")))
            if not _strict(pilot):
                continue
            candidates.append(
                {
                    "manifest": manifest_path,
                    "pilot": pilot,
                    "source": source_row,
                    "pivot_bin": _pivot_bin(pilot),
                    "source_intensity": source_intensity,
                    "source_path_angle_deg": path_angle_deg,
                    "source_facing_offset_deg": facing_offset_deg,
                    "source_lateral_range_m": lateral_range_m,
                }
            )

    requested = int(count)
    maximum = int(maximum_per_physical_source)
    if requested < 1 or maximum < 1:
        raise ValueError("review counts must be positive")
    if not candidates:
        raise ValueError("no strict compound stair candidates found")

    selected: list[dict[str, object]] = []
    used_candidate: set[tuple[str, str]] = set()
    per_source: dict[int, int] = defaultdict(int)
    seen_family: set[str] = set()
    seen_family_mode: set[tuple[str, str]] = set()
    seen_family_traversal: set[tuple[str, str]] = set()
    seen_mode: set[str] = set()
    seen_traversal: set[str] = set()
    seen_source_kind: set[str] = set()
    seen_mirror: set[bool] = set()
    seen_pivot_bin: set[str] = set()

    while len(selected) < requested:
        best: tuple[float, dict[str, object]] | None = None
        for candidate in candidates:
            manifest_path = candidate["manifest"]
            pilot = candidate["pilot"]
            source = candidate["source"]
            identity = (str(manifest_path), str(pilot["label"]))
            clip_index = int(source["clip_index"])
            if identity in used_candidate or per_source[clip_index] >= maximum:
                continue
            family = str(source["behavior_family"])
            mode = str(pilot["mode"])
            traversal = str(source["traversal"])
            source_kind = str(source["source_kind"])
            mirrored = bool(source["mirrored"])
            pivot_bin = str(candidate["pivot_bin"])
            quality = pilot["quality"]
            collision = pilot["collision_audit"]
            score = 0.0
            score += 100.0 * (family not in seen_family)
            score += 35.0 * ((family, mode) not in seen_family_mode)
            score += 20.0 * ((family, traversal) not in seen_family_traversal)
            score += 12.0 * (mode not in seen_mode)
            score += 8.0 * (traversal not in seen_traversal)
            score += 5.0 * (source_kind not in seen_source_kind)
            score += 4.0 * (mirrored not in seen_mirror)
            score += 3.0 * (pivot_bin not in seen_pivot_bin)
            score += 24.0 * min(float(candidate["source_intensity"]), 4.0)
            score += {"bounce": 3.0, "reverse": 2.0, "stop_restart": 1.0}.get(
                mode, 0.0
            )
            score -= 100.0 * float(quality["maximum_stance_run_drift_m"])
            score -= 25.0 * float(collision["maximum_foot_penetration_m"])
            tie_break = (
                family,
                mode,
                traversal,
                clip_index,
                str(pilot["label"]),
            )
            ranked = (score, tuple(-ord(char) for char in repr(tie_break)))
            if best is None or ranked > best[0]:
                best = (ranked, candidate)
        if best is None:
            break
        candidate = best[1]
        pilot = candidate["pilot"]
        source = candidate["source"]
        manifest_path = candidate["manifest"]
        family = str(source["behavior_family"])
        mode = str(pilot["mode"])
        traversal = str(source["traversal"])
        source_kind = str(source["source_kind"])
        mirrored = bool(source["mirrored"])
        pivot_bin = str(candidate["pivot_bin"])
        clip_index = int(source["clip_index"])
        identity = (str(manifest_path), str(pilot["label"]))
        selected.append(
            {
                "pilot": f"{manifest_path}#{pilot['label']}",
                "clip_index": clip_index,
                "traversal": traversal,
                "source_kind": source_kind,
                "behavior_family": family,
                "source_mode": str(source["mode"]),
                "compound_mode": mode,
                "mirrored": mirrored,
                "pivot_bin": pivot_bin,
                "source_intensity": float(candidate["source_intensity"]),
                "source_path_angle_deg": float(
                    candidate["source_path_angle_deg"]
                ),
                "source_facing_offset_deg": float(
                    candidate["source_facing_offset_deg"]
                ),
                "source_lateral_range_m": float(
                    candidate["source_lateral_range_m"]
                ),
                "terrain_progress": float(pilot["pivot"]["terrain_progress"]),
                "maximum_foot_penetration_m": float(
                    pilot["collision_audit"]["maximum_foot_penetration_m"]
                ),
                "maximum_stance_run_drift_m": float(
                    pilot["quality"]["maximum_stance_run_drift_m"]
                ),
            }
        )
        used_candidate.add(identity)
        per_source[clip_index] += 1
        seen_family.add(family)
        seen_family_mode.add((family, mode))
        seen_family_traversal.add((family, traversal))
        seen_mode.add(mode)
        seen_traversal.add(traversal)
        seen_source_kind.add(source_kind)
        seen_mirror.add(mirrored)
        seen_pivot_bin.add(pivot_bin)

    if len(selected) < requested:
        raise ValueError(
            f"strict review requested {requested} clips but selected "
            f"{len(selected)} under maximum_per_physical_source={maximum}"
        )
    return {
        "schema": "stairs500-compound-diverse-review/v1",
        "source_selection": str(source_selection.expanduser().resolve()),
        "bank_root": str(root),
        "strict_candidate_count": len(candidates),
        "automatic_gate_accepted_count": automatic_count,
        "selection_count": len(selected),
        "distinct_physical_source_count": len(
            {int(row["clip_index"]) for row in selected}
        ),
        "behavior_families": sorted(
            {str(row["behavior_family"]) for row in selected}
        ),
        "selections": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--maximum-per-physical-source", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = select(
        arguments.source_selection,
        arguments.bank_root,
        count=arguments.count,
        maximum_per_physical_source=arguments.maximum_per_physical_source,
    )
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(destination),
                "selection_count": payload["selection_count"],
                "distinct_physical_source_count": payload[
                    "distinct_physical_source_count"
                ],
                "behavior_families": payload["behavior_families"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
