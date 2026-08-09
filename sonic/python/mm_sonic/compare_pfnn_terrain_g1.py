"""Compare a native G1 retarget against one exact fitted PFNN terrain."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from mm_sonic.pfnn_terrain_fit import PFNNTerrainFit, load_terrain_fit, terrain_height_g1


HEEL = np.array([-0.066, 0.0, -0.034], dtype=np.float64)
TOE = np.array([0.12, 0.0, -0.034], dtype=np.float64)


def classify_probe_gaps(
    gaps_m: np.ndarray, *, stance: bool
) -> tuple[str, ...]:
    """Classify signed sole height minus terrain height using gate thresholds."""

    gaps = np.asarray(gaps_m, dtype=np.float64)
    if gaps.ndim != 1 or not np.isfinite(gaps).all():
        raise ValueError("probe gaps must be a finite vector")
    if type(stance) is not bool:
        raise ValueError("stance must be a bool")
    if not stance:
        return tuple("swing" for _ in gaps)
    return tuple(
        "penetrating" if gap < -0.01 else "floating" if gap > 0.02 else "contact"
        for gap in gaps
    )


def first_support_failure(
    gaps_m: np.ndarray,
    contacts: np.ndarray,
    *,
    source_start_frame: int,
) -> dict[str, object] | None:
    """Return the first failure using only PFNN-authored stance probes."""

    gaps = np.asarray(gaps_m, dtype=np.float64)
    stance = np.asarray(contacts)
    if gaps.ndim != 2 or gaps.shape[1] != 4 or not np.isfinite(gaps).all():
        raise ValueError("gaps_m must have finite shape [T, 4]")
    if stance.dtype != np.bool_ or stance.shape != gaps.shape:
        raise ValueError("contacts must have bool shape matching gaps_m")
    if type(source_start_frame) is not int or source_start_frame < 0:
        raise ValueError("source_start_frame must be a nonnegative integer")
    names = (
        ("left", "heel"),
        ("left", "toe"),
        ("right", "heel"),
        ("right", "toe"),
    )
    for frame in range(len(gaps)):
        for columns in ((0, 1), (2, 3)):
            stance_columns = [column for column in columns if stance[frame, column]]
            if not stance_columns:
                continue
            stance_gaps = gaps[frame, stance_columns]
            penetrating = np.flatnonzero(stance_gaps < -0.01)
            contact = (stance_gaps >= -0.01) & (stance_gaps <= 0.02)
            if len(penetrating):
                local = int(penetrating[np.argmin(stance_gaps[penetrating])])
            elif not np.any(contact):
                local = int(np.argmax(stance_gaps))
            else:
                continue
            column = stance_columns[local]
            return {
                "frame": frame,
                "source_frame": source_start_frame + frame,
                "foot": names[column][0],
                "probe": names[column][1],
                "gap_m": float(gaps[frame, column]),
                "labels": list(classify_probe_gaps(stance_gaps, stance=True)),
            }
    return None


def sole_probes(motion: dict[str, object], model: object) -> np.ndarray:
    """Evaluate G1 heel and toe probes as [T, foot, probe, xyz]."""

    import mujoco

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_quat = np.asarray(motion["root_quat"], dtype=np.float64)
    dof = np.asarray(motion["dof"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError("motion root_pos is invalid")
    if root_quat.shape != (len(root_pos), 4) or dof.shape[0] != len(root_pos):
        raise ValueError("motion timelines do not match")
    body_ids = tuple(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"
        )
        for side in ("left", "right")
    )
    if any(body_id < 0 for body_id in body_ids):
        raise ValueError("G1 model is missing ankle roll links")
    data = mujoco.MjData(model)
    result = np.empty((len(root_pos), 2, 2, 3), dtype=np.float64)
    for frame in range(len(root_pos)):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame, [3, 0, 1, 2]]
        data.qpos[7:] = dof[frame]
        mujoco.mj_forward(model, data)
        for foot, body_id in enumerate(body_ids):
            rotation = data.xmat[body_id].reshape(3, 3)
            position = data.xpos[body_id]
            result[frame, foot, 0] = rotation @ HEEL + position
            result[frame, foot, 1] = rotation @ TOE + position
    if not np.isfinite(result).all():
        raise ValueError("G1 sole probes are nonfinite")
    return result


@dataclass(frozen=True)
class SoleGapReport:
    accepted: bool
    gaps_m: np.ndarray
    first_failure: dict[str, object] | None
    stance_probe_count: int
    contact_probe_count: int
    penetrating_probe_count: int
    floating_probe_count: int
    minimum_stance_gap_m: float
    maximum_stance_gap_m: float


def compare_motion_to_terrain(
    motion: dict[str, object],
    fit: PFNNTerrainFit,
    model: object,
    *,
    terrain_scale: float = 1.0,
) -> SoleGapReport:
    """Measure exact terrain gaps and apply the PFNN transfer contact gate."""

    probes = sole_probes(motion, model)
    if len(probes) != fit.source_frame_count:
        raise ValueError("motion and terrain source intervals differ")
    flat = probes.reshape(-1, 3)
    terrain = terrain_height_g1(
        fit, flat[:, :2], scale=terrain_scale
    ).reshape(len(probes), 4)
    gaps = probes.reshape(len(probes), 4, 3)[:, :, 2] - terrain
    contacts = fit.source_contacts
    stance_values = gaps[contacts]
    if len(stance_values) == 0:
        raise ValueError("PFNN terrain fit contains no stance probes")
    labels = np.full(gaps.shape, "swing", dtype="<U11")
    labels[contacts & (gaps < -0.01)] = "penetrating"
    labels[contacts & (gaps >= -0.01) & (gaps <= 0.02)] = "contact"
    labels[contacts & (gaps > 0.02)] = "floating"
    first_failure = first_support_failure(
        gaps,
        contacts,
        source_start_frame=fit.source_start_frame,
    )
    return SoleGapReport(
        accepted=first_failure is None,
        gaps_m=gaps,
        first_failure=first_failure,
        stance_probe_count=int(np.count_nonzero(contacts)),
        contact_probe_count=int(np.count_nonzero(labels == "contact")),
        penetrating_probe_count=int(np.count_nonzero(labels == "penetrating")),
        floating_probe_count=int(np.count_nonzero(labels == "floating")),
        minimum_stance_gap_m=float(np.min(stance_values)),
        maximum_stance_gap_m=float(np.max(stance_values)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--terrain-scale", type=float, choices=(1.0, 0.875), default=1.0
    )
    return parser


def main() -> None:
    import mujoco
    from mm_sonic.view_g1_retarget import load_motion

    arguments = _parser().parse_args()
    motion = load_motion(arguments.motion)
    fit = load_terrain_fit(arguments.terrain)
    model_path = (
        arguments.gmr_root.resolve(strict=True)
        / "assets"
        / "unitree_g1"
        / "g1_mocap_29dof.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve(strict=True)))
    report = compare_motion_to_terrain(
        motion, fit, model, terrain_scale=arguments.terrain_scale
    )
    document = {
        "schema": "native-g1-pfnn-terrain-comparison/v1",
        "status": "accepted" if report.accepted else "rejected",
        "accepted": report.accepted,
        "terrain_scale": arguments.terrain_scale,
        "first_failure": report.first_failure,
        "stance_probe_count": report.stance_probe_count,
        "contact_probe_count": report.contact_probe_count,
        "penetrating_probe_count": report.penetrating_probe_count,
        "floating_probe_count": report.floating_probe_count,
        "minimum_stance_gap_m": report.minimum_stance_gap_m,
        "maximum_stance_gap_m": report.maximum_stance_gap_m,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
