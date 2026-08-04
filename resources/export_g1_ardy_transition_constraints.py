#!/usr/bin/env python3
"""Export two G1 traversal frames as ARDY full-body endpoint constraints."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ardy.constraints import FullBodyConstraintSet, save_constraints_lst
from ardy.skeleton import G1Skeleton34
from mm_sonic.joints import ContractError
from mm_sonic.torch_g1_fk import target_state_qpos


_MUJOCO_TO_ARDY = Rotation.from_euler(
    "x", -90, degrees=True
) * Rotation.from_euler("z", -90, degrees=True)


def _frame_qpos(artifact: Path, frame: int) -> np.ndarray:
    with np.load(artifact, allow_pickle=False) as archive:
        joints = np.asarray(archive["joint_position"], dtype=np.float64)
        roots = np.asarray(
            archive["root_position_world"], dtype=np.float64
        )
        quaternions = np.asarray(
            archive["root_orientation_world_wxyz"], dtype=np.float64
        )
    if not -len(joints) <= frame < len(joints):
        raise ContractError("ARDY endpoint frame is out of range")
    frame %= len(joints)
    return target_state_qpos(
        joints[frame], roots[frame], quaternions[frame]
    )


def _qpos_to_ardy(
    *,
    qpos: np.ndarray,
    skeleton: G1Skeleton34,
    g1_xml: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.asarray(qpos, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1:] != (36,)
        or not np.isfinite(values).all()
    ):
        raise ContractError("G1 qpos endpoints are invalid")
    frames = len(values)
    local = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (frames, skeleton.nbjoints, 3, 3),
    ).copy()
    root_rotation = Rotation.from_quat(values[:, 3:7], scalar_first=True)
    local[:, skeleton.root_idx] = (
        _MUJOCO_TO_ARDY
        * root_rotation
        * _MUJOCO_TO_ARDY.inv()
    ).as_matrix()
    tree = ET.parse(g1_xml)
    xml_root = tree.getroot()
    classes = {
        node.get("class"): node.findall("joint")[0].get("axis")
        for node in tree.findall(".//default")
        if node.get("class") and node.findall("joint")
    }
    parent_map = {
        child: parent for parent in xml_root.iter() for child in parent
    }
    hinges = [
        joint
        for joint in xml_root.find("worldbody").findall(".//joint")
        if joint.get("type") != "free"
    ]
    if len(hinges) != 29:
        raise ContractError("G1 XML does not contain 29 hinge joints")
    for joint_index, joint in enumerate(hinges):
        skeleton_name = joint.get("name").replace("_joint", "_skel")
        skeleton_index = skeleton.bone_index[skeleton_name]
        axis = np.asarray(
            [
                float(item)
                for item in (
                    joint.get("axis") or classes[joint.get("class")]
                ).split()
            ],
            dtype=np.float64,
        )
        rotation = Rotation.from_rotvec(
            values[:, 7 + joint_index, None] * axis[None, :]
        )
        body = parent_map[joint]
        if "quat" in body.attrib:
            offset = Rotation.from_quat(
                [float(item) for item in body.get("quat").split()],
                scalar_first=True,
            )
            rotation = offset * rotation
        local[:, skeleton_index] = (
            _MUJOCO_TO_ARDY
            * rotation
            * _MUJOCO_TO_ARDY.inv()
        ).as_matrix()
    roots = _MUJOCO_TO_ARDY.apply(values[:, :3])
    return (
        torch.tensor(local, dtype=torch.float32),
        torch.tensor(roots, dtype=torch.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--outgoing", type=Path, required=True)
    parser.add_argument("--incoming-frame", type=int, required=True)
    parser.add_argument("--outgoing-frame", type=int, required=True)
    parser.add_argument("--transition-frames", type=int, default=50)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.transition_frames < 4:
        raise ContractError("ARDY transition duration is too short")
    qpos = np.stack(
        (
            _frame_qpos(args.incoming, args.incoming_frame),
            _frame_qpos(args.outgoing, args.outgoing_frame),
        )
    )
    skeleton = G1Skeleton34()
    local, roots = _qpos_to_ardy(
        qpos=qpos, skeleton=skeleton, g1_xml=args.g1_xml
    )
    # ARDY canonicalizes the first generated root to the horizontal origin.
    # Express both endpoint constraints in that same frame; the generated
    # qpos is translated back to the incoming traversal frame at import time.
    roots[:, (0, 2)] -= roots[0, (0, 2)].clone()
    global_rotations, global_positions, _ = skeleton.fk(local, roots)
    constraints = FullBodyConstraintSet(
        skeleton,
        frame_indices=torch.tensor(
            (0, args.transition_frames - 1), dtype=torch.int64
        ),
        global_joints_positions=global_positions,
        global_joints_rots=global_rotations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_constraints_lst(str(args.output), [constraints])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
