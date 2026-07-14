from pathlib import Path

import joblib
import numpy as np

from resources.g1_interaction_builder.schema import SourcePaths


def write_source_fixture(
    root: Path,
    sequence_id: str = "pickup_table__cup_2__001",
    object_id: str = "cup_2",
    frames: int = 25,
) -> SourcePaths:
    robot_dir = root / "robot"
    objects_dir = root / "objects"
    meta_dir = root / "meta"
    object_usd_dir = root / "object_usd"
    for directory in (robot_dir, objects_dir, meta_dir, object_usd_dir):
        directory.mkdir(parents=True, exist_ok=True)

    robot_record = {
        "dof": np.zeros((frames, 29), np.float32),
        "root_trans_offset": np.zeros((frames, 3), np.float32),
        "root_rot": np.tile(
            np.array([0, 0, 0, 1], np.float32), (frames, 1)
        ),
        "fps": 25.0,
        "hand_dof_pos": np.zeros((frames, 14), np.float32),
    }
    object_record = {
        "root_pos": np.zeros((frames, 1, 3), np.float32),
        "root_quat": np.tile(
            np.array([0, 0, 0, 1], np.float32), (frames, 1, 1)
        ),
        "fps": 25.0,
        "contact_points_left_hand": {
            i: (
                np.array([[0.0, 0.0, 0.0]], np.float32)
                if i >= 10
                else np.empty((0, 3), np.float32)
            )
            for i in range(frames)
        },
        "contact_points_right_hand": {
            i: np.empty((0, 3), np.float32) for i in range(frames)
        },
    }
    meta_record = {
        "object_name": object_id,
        "table_pos": np.array([0.0, 0.0, 0.75], np.float32),
        "table_quat": np.array([0, 0, 0, 1], np.float32),
        "table_size": np.array([1.2, 0.8, 0.05], np.float32),
    }

    robot = robot_dir / f"{sequence_id}.pkl"
    objects = objects_dir / f"{sequence_id}.pkl"
    meta = meta_dir / f"{sequence_id}.pkl"
    object_usd = object_usd_dir / f"{sequence_id}.usd"
    joblib.dump({sequence_id: robot_record}, robot)
    joblib.dump({sequence_id: object_record}, objects)
    joblib.dump({sequence_id: meta_record}, meta)
    object_usd.touch()
    return SourcePaths(
        sequence_id=sequence_id,
        object_id=object_id,
        robot=robot,
        objects=objects,
        meta=meta,
        object_usd=object_usd,
    )
