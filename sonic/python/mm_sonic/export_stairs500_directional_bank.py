"""Export exact-audited directional stair reports as a SONIC selection.

The directional generator writes one self-contained report per motion, while
the existing SONIC bundle builder consumes legacy ``MANIFEST#LABEL`` entries.
This adapter keeps that interface small: it admits only motions that actually
traverse a staircase and realize a path/facing change (or an exact full-stair
temporal reverse), then writes one terrain manifest per archive clip.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Sequence

import zarr

from .mirror_stairs500_omnidirectional_pilots import (
    DEFAULT_ARCHIVE,
    _realized_directional_motion,
)


def _quality_tier(report: dict[str, object]) -> str:
    """Separate clean training candidates from mechanically marginal ones."""

    warp = report.get("warp")
    if not isinstance(warp, dict):
        return "silver"
    if bool(warp.get("identity_temporal_reverse")):
        return "gold"
    collision = report.get("collision_audit")
    if not isinstance(collision, dict):
        return "silver"
    gold = bool(
        float(warp.get("maximum_stance_sole_target_error_m", 1.0)) <= 0.006
        and float(warp.get("maximum_swing_sole_target_error_m", 1.0)) <= 0.030
        and float(warp.get("maximum_stance_run_drift_m", 1.0)) <= 0.006
        and float(report.get("clearance_repair_maximum_m", 1.0)) <= 0.013
        and float(warp.get("maximum_joint_correction_rad", 1.0)) <= 0.42
        and float(collision.get("maximum_foot_penetration_m", 1.0)) <= 0.0045
        and float(collision.get("maximum_forbidden_body_penetration_m", 1.0))
        == 0.0
    )
    return "gold" if gold else "silver"


def export(
    roots: Sequence[Path],
    *,
    archive_path: Path,
    output: Path,
    source_kinds: Sequence[str],
    quality_tiers: Sequence[str],
) -> dict[str, object]:
    archive_path = archive_path.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    allowed_kinds = frozenset(source_kinds)
    allowed_quality = frozenset(quality_tiers)
    report_paths = sorted(
        {
            report.resolve()
            for root in roots
            for report in root.expanduser().resolve().rglob("report.json")
        }
    )
    admitted: list[tuple[Path, dict[str, object]]] = []
    rejected_status = 0
    rejected_kind = 0
    rejected_unrealized = 0
    rejected_quality = 0
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        if report.get("status") != "accepted":
            rejected_status += 1
            continue
        if str(report.get("source_kind")) not in allowed_kinds:
            rejected_kind += 1
            continue
        if not _realized_directional_motion(report_path, report, archive):
            rejected_unrealized += 1
            continue
        if allowed_quality and _quality_tier(report) not in allowed_quality:
            rejected_quality += 1
            continue
        admitted.append((report_path, report))
    if not admitted:
        raise ValueError("no realized directional reports were admitted")

    destination = output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[tuple[Path, dict[str, object]]]] = defaultdict(list)
    for report_path, report in admitted:
        grouped[int(report["clip_index"])].append((report_path, report))

    selections: list[dict[str, object]] = []
    manifest_paths: list[str] = []
    for clip_index, rows in sorted(grouped.items()):
        manifest_dir = destination / "manifests" / f"clip_{clip_index:03d}"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        used_labels: set[str] = set()
        pilots: list[dict[str, object]] = []
        for report_path, report in sorted(rows, key=lambda item: str(item[0])):
            label = str(report["label"])
            if label in used_labels:
                label = f"{report_path.parent.parent.name}_{label}"
            used_labels.add(label)
            motion = Path(str(report.get("motion", report_path.parent / "motion.npz")))
            pilot = {
                "label": label,
                "kind": "stairs500_directional",
                "motion": str(motion.expanduser().resolve()),
                "automatic_gate_accepted": True,
                "directional_report": str(report_path),
                "source_kind": str(report["source_kind"]),
                "mode": str(report["mode"]),
                "traversal": str(report["traversal"]),
                "symmetry_mirrored": bool(report.get("mirror_of")),
                "quality_tier": _quality_tier(report),
            }
            if "warp" in report:
                pilot["warp"] = report["warp"]
            pilots.append(pilot)
            selections.append(
                {
                    "pilot": f"{manifest_path}#{label}",
                    "clip_index": clip_index,
                    "source_kind": pilot["source_kind"],
                    "mode": pilot["mode"],
                    "traversal": pilot["traversal"],
                    "symmetry_mirrored": pilot["symmetry_mirrored"],
                    "quality_tier": pilot["quality_tier"],
                }
            )
        manifest = {
            "schema": "stairs500-directional-compatibility-manifest/v1",
            "archive": str(archive_path),
            "clip_index": clip_index,
            "clip_name": str(archive["clip_names"][clip_index]),
            "terrain_usd": str(archive["terrain_usd_path"][clip_index]),
            "terrain_position_world": [
                float(value) for value in archive["terrain_position_env"][clip_index]
            ],
            "terrain_quaternion_world_from_usd_wxyz": [
                float(value)
                for value in archive["terrain_rotation_env_wxyz"][clip_index]
            ],
            "pilots": pilots,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifest_paths.append(str(manifest_path))

    payload = {
        "schema": "stairs500-directional-selection/v1",
        "archive": str(archive_path),
        "roots": [str(path.expanduser().resolve()) for path in roots],
        "source_kinds": sorted(allowed_kinds),
        "quality_tiers": sorted(allowed_quality),
        "report_count": len(report_paths),
        "admitted_count": len(admitted),
        "rejected_status_count": rejected_status,
        "rejected_kind_count": rejected_kind,
        "rejected_unrealized_count": rejected_unrealized,
        "rejected_quality_count": rejected_quality,
        "manifest_count": len(manifest_paths),
        "manifests": manifest_paths,
        "visual_review_required_before_sonic_training": True,
        "selections": selections,
    }
    selection_path = destination / "selection.json"
    selection_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "selection": str(selection_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-kind",
        action="append",
        default=[],
        help="Admitted source kind; defaults to native and temporal_reverse.",
    )
    parser.add_argument(
        "--quality-tier",
        choices=("gold", "silver"),
        action="append",
        default=[],
        help="Optional admitted quality tier; by default both are exported.",
    )
    arguments = parser.parse_args()
    source_kinds = tuple(arguments.source_kind) or ("native", "temporal_reverse")
    summary = export(
        arguments.root,
        archive_path=arguments.archive,
        output=arguments.output,
        source_kinds=source_kinds,
        quality_tiers=tuple(arguments.quality_tier),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
