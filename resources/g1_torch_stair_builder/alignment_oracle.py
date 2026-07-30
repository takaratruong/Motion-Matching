"""Protected real-data oracle for the native GRAIL stair alignment.

This module is deliberately outside the runtime.  It checks the transformed
USD surface against recorded G1 foot contacts at the original source frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from resources.g1_terrain_builder.kinematics import G1Kinematics

from .corpus import STAIR_BASES, load_pinned_sources
from .surface import load_source_surface


DEFAULT_GRAIL_ROOT = Path("/home/ubuntu/datasets/GRAIL/data/stair_p1")
DEFAULT_G1_XML = Path(
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)
def compare_real_contact_alignment(
    *,
    grail_root: str | Path = DEFAULT_GRAIL_ROOT,
    g1_xml: str | Path = DEFAULT_G1_XML,
) -> float:
    """Return the worst foot-contact height error in metres.

    The G1 ankle-roll body origins are 3.5 cm above the sole.  A correctly
    transformed stair therefore places the minimum recorded left and right foot
    clearance at 0.035 m.  Requiring many near-contact frames prevents a single
    accidental triangle intersection from satisfying this oracle.
    """
    grail_root = Path(grail_root).resolve()
    g1_xml = Path(g1_xml).resolve()
    if not g1_xml.is_file():
        raise FileNotFoundError(f"G1 XML is missing: {g1_xml}")
    base = STAIR_BASES[0]
    pinned = load_pinned_sources(grail_root)[0]
    if pinned.base != base:
        raise ValueError("pinned stair source order changed")
    surface = load_source_surface(pinned)
    kinematics = G1Kinematics(str(g1_xml))
    body_position, _body_quaternion = kinematics.world_from_qpos(
        pinned.robot_qpos_mujoco
    )
    clearance: list[np.ndarray] = []
    for name in ("LeftToe", "RightToe"):
        body_index = kinematics.names.index(name)
        foot = body_position[:, body_index]
        values = foot[:, 2] - surface.height_xy(foot[:, :2])
        if int(np.count_nonzero(values <= 0.04)) < 100:
            raise ValueError(
                f"real stair alignment has too few {name} contact frames"
            )
        clearance.append(values)
    minimum = np.array([float(values.min()) for values in clearance])
    return float(np.max(np.abs(minimum - 0.035)))
