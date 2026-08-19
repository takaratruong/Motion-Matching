from pathlib import Path

import numpy as np

from mm_sonic.render_justin_recovery_comparison import (
    _camera_offset_for_absolute_azimuth,
    _recovered_motion,
    _scene_contract,
)
from mm_sonic.terrain_oracle.stitch import FrameProvenance, StitchedMotion


def _learner_motion() -> StitchedMotion:
    roots = np.asarray(
        ((0.0, 0.0, 0.75), (1.0, 0.0, 0.75), (2.0, 0.0, 0.75), (3.0, 0.0, 0.2)),
        dtype=np.float32,
    )
    return StitchedMotion(
        fps=50.0,
        root_position_world=roots,
        root_quaternion_world_wxyz=np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (4, 1)
        ),
        joint_position=np.zeros((4, 29), dtype=np.float32),
        provenance=tuple(
            FrameProvenance(0, frame, "dense") for frame in range(4)
        ),
        seam_indices=(),
    )


def test_recovered_motion_keeps_prefix_and_appends_tracker(tmp_path: Path) -> None:
    learner = _learner_motion()
    tracker_roots = np.asarray(
        ((2.01, 0.0, 0.75), (2.5, 0.0, 1.1)), dtype=np.float32
    )
    np.savez(
        tmp_path / "tracker_rollout.npz",
        root_pos=tracker_roots,
        root_quat_wxyz=np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (2, 1)
        ),
        joint_pos=np.ones((2, 29), dtype=np.float32),
        fps=np.asarray((50,), dtype=np.int64),
    )

    recovered, seam = _recovered_motion(
        learner, {"trace_index": 2}, tmp_path
    )

    np.testing.assert_array_equal(
        recovered.root_position_world[:3], learner.root_position_world[:3]
    )
    np.testing.assert_array_equal(recovered.root_position_world[3:], tracker_roots)
    assert recovered.seam_indices == (3,)
    assert seam["handoff_frame"] == 2
    assert seam["tracker_first_frame"] == 3
    assert seam["handoff_time_s"] == 0.04
    assert seam["handoff_position_step_m"] < 0.011


def test_camera_offset_produces_requested_absolute_azimuth() -> None:
    learner = _learner_motion()
    offset = _camera_offset_for_absolute_azimuth(learner, 270.0)
    assert offset == 270.0


def test_scene_contract_uses_generic_proposal_transform() -> None:
    scene = _scene_contract(
        {
            "scene_transform": {
                "scene_id": "karen10_d150",
                "learner_front_xy": [1.5, 0.0],
                "reference_front_xy": [0.0, 0.0],
                "learner_heading_yaw_rad": 0.0,
                "reference_heading_yaw_rad": 0.0,
                "tread_m": 0.33074625,
                "num_steps": 10,
            }
        }
    )

    assert scene.scene_id == "karen10_d150"
    assert scene.learner_front_xy == (1.5, 0.0)
    assert scene.num_steps == 10
