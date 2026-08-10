"""Add exact-audited timing maneuvers to paired motion/terrain co-warps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .build_terrain_maneuver_pilots import build_pilots


def _selection_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.expanduser().resolve().read_text())
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("co-warp selection must be a list or contain 'rows'")
    return [dict(value) for value in rows]


def _safe_label(value: object) -> str:
    text = str(value)
    return "".join(
        character if character.isalnum() else "_" for character in text
    )


def build_row(
    *,
    selection_path: Path,
    row_index: int,
    output_root: Path,
    motion_archive: Path,
    model_path: Path,
    modes: tuple[str, ...] = ("stop_restart", "reverse", "bounce"),
    pivot_count: int = 3,
    ramp_frames: int = 16,
    hold_frames: int = 4,
    reverse_ramp_frames: int = 16,
    reverse_hold_frames: int = 4,
) -> dict[str, object]:
    """Retimes one accepted co-warp while retaining its exact paired mesh."""

    selection_path = selection_path.expanduser().resolve()
    rows = _selection_rows(selection_path)
    index = int(row_index)
    if not 0 <= index < len(rows):
        raise ValueError(f"row index {index} is outside [0, {len(rows)})")
    source_row = rows[index]
    report_path = Path(str(source_row["report"])).expanduser().resolve()
    report = json.loads(report_path.read_text())
    if report.get("status") != "accepted":
        raise ValueError(f"co-warp source is not accepted: {report_path}")
    motion_path = Path(
        str(source_row.get("motion", report_path.parent / "motion.npz"))
    ).expanduser().resolve()
    terrain_path = Path(
        str(source_row.get("terrain_usd", report["terrain_usd"]))
    ).expanduser().resolve()
    if not motion_path.is_file() or not terrain_path.is_file():
        raise FileNotFoundError(
            f"missing paired co-warp assets: {motion_path}, {terrain_path}"
        )

    clip_index = int(source_row.get("clip_index", report["clip_index"]))
    source_mode = str(source_row.get("mode", report["mode"]))
    label = f"source_{index:03d}_clip_{clip_index:03d}_{_safe_label(source_mode)}"
    root = output_root.expanduser().resolve() / label
    arguments = SimpleNamespace(
        source_motion=[motion_path],
        terrain_usd=terrain_path,
        # Generated co-warp USD vertices are already in the motion's world
        # frame; archive lookup would incorrectly select the original mesh.
        terrain_position=[0.0, 0.0, 0.0],
        terrain_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        portal_manifest=Path("unused.json"),
        output=root / "pilots",
        motion_archive=motion_archive.expanduser().resolve(),
        model_path=model_path.expanduser().resolve(),
        event_index=[],
        mode=list(modes),
        pivot_count=int(pivot_count),
        ramp_frames=int(ramp_frames),
        hold_frames=int(hold_frames),
        reverse_ramp_frames=int(reverse_ramp_frames),
        reverse_hold_frames=int(reverse_hold_frames),
        pivot_clearance_m=0.008,
        pivot_speed_mps=0.20,
        pivot_central_fraction=(0.05, 0.95),
        support_clearance_m=0.005,
        support_speed_mps=0.10,
        maximum_any_support_clearance_m=0.060,
        maximum_foot_penetration_m=0.005,
        maximum_stance_step_m=0.003,
        # A two-centimetre run can satisfy collision/contact while looking
        # visibly skatey during a long stop.  Keep compound plants within one
        # centimetre; this does not alter source authoring or IK authority.
        maximum_stance_run_drift_m=0.010,
    )
    pilots = build_pilots(arguments)
    result: dict[str, object] = {
        "schema": "cowarped-compound-source/v1",
        "selection": str(selection_path),
        "row_index": index,
        "clip_index": clip_index,
        "clip_name": str(source_row.get("clip_name", report["clip_name"])),
        "clip_family": str(source_row.get("clip_family", "unknown")),
        "clip_traversal": str(source_row.get("clip_traversal", "unknown")),
        "source_mode": source_mode,
        "source_report": str(report_path),
        "source_motion": str(motion_path),
        "terrain_usd": str(terrain_path),
        "pilot_manifest": str((root / "pilots" / "manifest.json").resolve()),
        "pilot_count": int(pilots["pilot_count"]),
        "automatic_gate_accepted_count": int(
            pilots["automatic_gate_accepted_count"]
        ),
        "status": (
            "pending_dense_visual_review"
            if pilots["automatic_gate_accepted_count"]
            else "no_accepted_compound_motion"
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.json").write_text(
        json.dumps(source_row, indent=2, sort_keys=True) + "\n"
    )
    (root / "clip_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--motion-archive", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument(
        "--mode",
        choices=("stop_restart", "reverse", "bounce"),
        action="append",
        default=[],
    )
    parser.add_argument("--pivot-count", type=int, default=3)
    parser.add_argument("--ramp-frames", type=int, default=16)
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument("--reverse-ramp-frames", type=int, default=16)
    parser.add_argument("--reverse-hold-frames", type=int, default=4)
    arguments = parser.parse_args(argv)
    if arguments.pivot_count < 1:
        parser.error("--pivot-count must be positive")
    result = build_row(
        selection_path=arguments.selection,
        row_index=arguments.row_index,
        output_root=arguments.output_root,
        motion_archive=arguments.motion_archive,
        model_path=arguments.model_path,
        modes=tuple(arguments.mode or ("stop_restart", "reverse", "bounce")),
        pivot_count=arguments.pivot_count,
        ramp_frames=arguments.ramp_frames,
        hold_frames=arguments.hold_frames,
        reverse_ramp_frames=arguments.reverse_ramp_frames,
        reverse_hold_frames=arguments.reverse_hold_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
