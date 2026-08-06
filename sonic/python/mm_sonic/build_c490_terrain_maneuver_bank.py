"""Build exact-grounded stop/reversal pilots from C490 terrain clips.

Each task first exports and exact-mesh-audits one clean archive clip on its own
terrain.  A bounded vertical clearance repair is allowed when the source is a
few millimetres below the collision envelope.  Only an accepted source is then
passed to the terrain maneuver composer.  This keeps the scale-up path honest:
every output retains the source terrain registration instead of assuming that
the USD mesh is already aligned with the motion world frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import zarr

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .build_terrain_maneuver_pilots import build_pilots
from .evaluate_archive_identity_traversal import evaluate


def build_clip(
    *,
    archive_path: Path,
    model_path: Path,
    clip_index: int,
    output_root: Path,
    ramp_frames: int = 24,
    hold_frames: int = 30,
    pivot_speed_mps: float = 0.20,
) -> dict[str, object]:
    """Materialize, qualify, and compose one registered archive clip."""

    archive_path = archive_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    index = int(clip_index)
    count = len(archive["clip_names"])
    if not 0 <= index < count:
        raise ValueError(f"clip index {index} is outside [0, {count})")
    if str(archive["clip_family"][index]) != "c490_slope":
        raise ValueError(f"clip {index} is not a C490 slope")

    clip_name = str(archive["clip_names"][index])
    clip_root = output_root / f"clip_{index:03d}_{clip_name}"
    source_dir = clip_root / "source"
    source_report = evaluate(
        archive_path=archive_path,
        model_path=model_path,
        target_clip_index=index,
        output_dir=source_dir,
        render=False,
        repair_clearance=True,
    )
    result: dict[str, object] = {
        "schema": "c490-terrain-maneuver-bank-clip/v1",
        "clip_index": index,
        "clip_name": clip_name,
        "clip_family": str(archive["clip_family"][index]),
        "clip_traversal": str(archive["clip_traversal"][index]),
        "root_delta_z_m": float(archive["root_delta_z_m"][index]),
        "terrain_usd": str(Path(str(archive["terrain_usd_path"][index])).resolve()),
        "source_report": source_report,
        "status": "source_rejected",
    }
    if source_report["status"] != "accepted":
        clip_root.mkdir(parents=True, exist_ok=True)
        (clip_root / "clip_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result

    arguments = SimpleNamespace(
        source_motion=[source_dir / "motion.npz"],
        terrain_usd=Path(str(archive["terrain_usd_path"][index])),
        terrain_position=None,
        terrain_quaternion_wxyz=None,
        portal_manifest=Path("unused.json"),
        output=clip_root / "pilots",
        motion_archive=archive_path,
        model_path=model_path,
        event_index=[],
        mode=["stop_restart", "reverse"],
        ramp_frames=int(ramp_frames),
        hold_frames=int(hold_frames),
        pivot_clearance_m=0.008,
        pivot_speed_mps=float(pivot_speed_mps),
        pivot_central_fraction=(0.20, 0.80),
        support_clearance_m=0.005,
        support_speed_mps=0.10,
        maximum_any_support_clearance_m=0.060,
        maximum_foot_penetration_m=0.005,
        maximum_stance_step_m=0.003,
        maximum_stance_run_drift_m=0.012,
    )
    try:
        pilots = build_pilots(arguments)
    except ValueError as error:
        result["status"] = "no_supported_pivot"
        result["reason"] = str(error)
    else:
        result["pilot_manifest"] = str(
            (clip_root / "pilots" / "manifest.json").resolve()
        )
        result["pilot_count"] = int(pilots["pilot_count"])
        result["automatic_gate_accepted_count"] = int(
            pilots["automatic_gate_accepted_count"]
        )
        result["status"] = (
            "pending_dense_visual_review"
            if pilots["automatic_gate_accepted_count"]
            else "pilot_gate_rejected"
        )
    (clip_root / "clip_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--clip-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ramp-frames", type=int, default=24)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--pivot-speed-mps", type=float, default=0.20)
    arguments = parser.parse_args(argv)
    result = build_clip(
        archive_path=arguments.archive,
        model_path=arguments.model,
        clip_index=arguments.clip_index,
        output_root=arguments.output_root,
        ramp_frames=arguments.ramp_frames,
        hold_frames=arguments.hold_frames,
        pivot_speed_mps=arguments.pivot_speed_mps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pending_dense_visual_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
