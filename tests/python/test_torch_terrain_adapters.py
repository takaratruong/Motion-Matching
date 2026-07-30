from pathlib import Path
from types import MappingProxyType
import json
import tempfile
import unittest

import numpy as np

from resources.g1_torch_stair_builder.conversion import NativeMotionArrays
from resources.g1_torch_terrain_builder.registry import (
    ResolvedSource,
    SourceSpec,
)
from resources.g1_torch_terrain_builder.terrain import (
    build_source_terrain,
)


def _motion() -> NativeMotionArrays:
    frames = 60
    joint = np.zeros((frames, 29), np.float32)
    body = np.zeros((frames, 30, 3), np.float32)
    body[:, 0, 0] = np.linspace(-1.0, 1.0, frames)
    body[:, 0, 2] = 0.8
    quaternion = np.zeros((frames, 30, 4), np.float32)
    quaternion[..., 0] = 1.0
    return NativeMotionArrays(
        fps=50,
        joint_position=joint,
        joint_velocity=np.zeros_like(joint),
        body_position_world=body,
        body_quaternion_world_wxyz=quaternion,
        body_linear_velocity_world=np.zeros_like(body),
        body_angular_velocity_world=np.zeros_like(body),
    )


def _source(
    root: Path,
    *,
    family: str,
    terrain_adapter: str,
    geometry: tuple[Path, ...] = (),
) -> ResolvedSource:
    motion = root / "motion.npz"
    motion.write_bytes(b"fixture")
    return ResolvedSource(
        spec=SourceSpec(
            logical_name=f"{family}-fixture",
            family=family,
            source_adapter="native-npz",
            motion_relative_path="motion.npz",
            motion_sha256="0" * 64,
            terrain_adapter=terrain_adapter,
            geometry_relative_paths=tuple(path.name for path in geometry),
            geometry_sha256=tuple("1" * 64 for _ in geometry),
        ),
        motion_path=motion,
        geometry_paths=geometry,
        source_sha256=MappingProxyType({}),
    )


class ExpandedTerrainAdapterTests(unittest.TestCase):
    def test_fixed_staircase_has_documented_axis_heights_and_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = build_source_terrain(
                _source(
                    root,
                    family="stair-local",
                    terrain_adapter="fixed-staircase",
                ),
                _motion(),
            )
        sample = evidence.grid.sample_xy(
            np.array(
                [
                    [0.10, 0.00],
                    [-0.20, 0.00],
                    [-0.55, 0.00],
                    [0.50, 0.00],
                    [0.10, 0.60],
                ],
                np.float32,
            )
        )
        np.testing.assert_allclose(
            sample, [0.1778, 0.3556, 0.5334, 0.0, 0.0], atol=0.011
        )
        self.assertEqual(evidence.adapter, "fixed-staircase")
        self.assertEqual(evidence.motion_to_terrain_xy_yaw, (0.0, 0.0, 0.0))

    def test_scaled_staircase_uses_pinned_urdf_scale_and_scene_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            urdf = root / "stairs.urdf"
            objects = tuple(root / f"box{index}.obj" for index in range(1, 4))
            y_bounds = ((-0.345, 0.0), (-0.01, 0.335), (0.334, 0.999))
            for index, (path, (minimum_y, maximum_y)) in enumerate(
                zip(objects, y_bounds), start=1
            ):
                top = 0.17 * index
                path.write_text(
                    "\n".join(
                        (
                            f"v -0.542 {minimum_y} 0",
                            f"v 0.620 {maximum_y} {top}",
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
            urdf.write_text(
                """<?xml version="1.0"?>
<robot name="multi_boxes">
  <link name="world"/>
  <link name="multi_boxes_box1_link"><collision><geometry>
    <mesh filename="box1.obj" scale="0.8380952380952381 0.8380952380952381 0.8380952380952381"/>
  </geometry></collision></link>
  <link name="multi_boxes_box2_link"><collision><geometry>
    <mesh filename="box2.obj" scale="0.8380952380952381 0.8380952380952381 0.8380952380952381"/>
  </geometry></collision></link>
  <link name="multi_boxes_box3_link"><collision><geometry>
    <mesh filename="box3.obj" scale="0.8380952380952381 0.8380952380952381 0.8380952380952381"/>
  </geometry></collision></link>
</robot>
""",
                encoding="utf-8",
            )
            evidence = build_source_terrain(
                _source(
                    root,
                    family="stair-local",
                    terrain_adapter="scaled-staircase-084",
                    geometry=(urdf, *objects),
                ),
                _motion(),
            )
        np.testing.assert_allclose(
            evidence.grid.sample_xy(
                np.array([[0.0, -0.10], [0.0, 0.50]], np.float32)
            ),
            [0.14247619, 0.42742857],
            atol=0.011,
        )
        self.assertEqual(evidence.adapter, "scaled-staircase-084")

    def test_karen_metadata_builds_exact_declared_boxes(self):
        metadata = {
            "axis_conversion": "fbx_y_up_to_z_up_with_negated_y",
            "stairs": [
                {
                    "name": "step_01",
                    "bounds_min_m": [-0.4, -0.5, 0.0],
                    "bounds_max_m": [0.0, 0.5, 0.2],
                },
                {
                    "name": "step_02",
                    "bounds_min_m": [-0.8, -0.5, 0.0],
                    "bounds_max_m": [-0.4, 0.5, 0.4],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "staircase_metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            evidence = build_source_terrain(
                _source(
                    root,
                    family="stair-karen",
                    terrain_adapter="karen-metadata",
                    geometry=(path,),
                ),
                _motion(),
            )
        np.testing.assert_allclose(
            evidence.grid.sample_xy(
                np.array([[-0.2, 0.0], [-0.6, 0.0], [0.3, 0.0]])
            ),
            [0.2, 0.4, 0.0],
            atol=0.011,
        )

    def test_chair_box_requires_checked_config_and_uses_box_top(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source(
                root,
                family="curb-chair",
                terrain_adapter="chair-object",
            )
            with self.assertRaisesRegex(ValueError, "chair configuration"):
                build_source_terrain(
                    source,
                    _motion(),
                    chair_config_path=root / "missing.py",
                    chair_config_sha256=None,
                )
            config = root / "chair.py"
            config.write_text(
                "BOX_POSITION = [-0.1, 0.50, 0.15]\n"
                "BOX_SIZE = [0.4572, 0.4064, 0.3]\n",
                encoding="utf-8",
            )
            evidence = build_source_terrain(
                source,
                _motion(),
                chair_config_path=config,
                chair_config_sha256=None,
            )
        np.testing.assert_allclose(
            evidence.grid.sample_xy(
                np.array([[-0.1, 0.5], [0.5, 0.5]], np.float32)
            ),
            [0.3, 0.0],
            atol=0.011,
        )

    def test_invalid_karen_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "staircase_metadata.json"
            path.write_text('{"stairs":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Karen"):
                build_source_terrain(
                    _source(
                        root,
                        family="stair-karen",
                        terrain_adapter="karen-metadata",
                        geometry=(path,),
                    ),
                    _motion(),
                )


if __name__ == "__main__":
    unittest.main()
