# G1 Terrain Artifact Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build validated, versioned Holden animation and terrain artifacts directly from native Takara and GRAIL G1 motion.

**Architecture:** A focused Python package loads source-specific G1 qpos, converts MuJoCo Z-up transforms into Holden Y-up local bones, converts each clip independently to the canonical 25 Hz runtime rate, derives contacts and terrain features, and atomically publishes a Holden database plus sidecars. Source adapters, kinematics, terrain sampling, serialization, and validation remain separate modules.

**Tech Stack:** Python 3.11 from /home/ubuntu/miniconda3/envs/diffsim/bin/python; NumPy 2.4; SciPy 1.17; MuJoCo 3.9; joblib 1.5; USD pxr; standard-library unittest; Holden resources/quat.py.

## Global Constraints

- Run Python commands with /home/ubuntu/miniconda3/envs/diffsim/bin/python.
- Add no Python dependency; use standard-library unittest rather than pytest.
- Input G1 motion is Z-up; output is right-handed Y-up with canonical simulation forward +Z.
- Output rate is exactly 25 Hz; GRAIL stays native and Takara is downsampled independently from 50 Hz.
- Process all 1,769 locally available GRAIL curb clips by default; a limit is diagnostic-only.
- Preserve explicit clip ranges and source-frame mappings.
- Emit four terrain values at geometric centerline distances 0.25, 0.50, 0.75, and 1.00 m.
- Never overwrite resources/database.bin or resources/features.bin.
- Publish resources/g1_terrain only after every validation passes.
- Use shortest-arc quaternion interpolation and keep norms within 1e-4 of one.
- Require sampled source-to-export FK positional error no greater than 1 mm.

---

## File Map

- resources/g1_terrain_builder/schema.py: canonical source, skeleton, converted clip, and artifact dataclasses.
- resources/g1_terrain_builder/sources.py: Takara and GRAIL qpos adapters.
- resources/g1_terrain_builder/resample.py: scalar/vector interpolation and wxyz quaternion slerp.
- resources/g1_terrain_builder/kinematics.py: G1 MuJoCo FK, change of basis, local transforms, and simulation bone.
- resources/g1_terrain_builder/terrain.py: GRAIL mesh alignment, height queries, centerlines, and heightfield export.
- resources/g1_terrain_builder/database.py: velocities, contacts, ranges, Holden database serialization.
- resources/g1_terrain_builder/artifacts.py: terrain sidecar, manifest, validation summary, and atomic publish.
- resources/build_g1_terrain_database.py: command-line orchestration.
- resources/validate_g1_terrain_database.py: independent artifact validation entry point.
- tests/python/: standard-library unit and integration tests.

### Task 1: Canonical schemas and validation

**Files:**
- Create: resources/g1_terrain_builder/__init__.py
- Create: resources/g1_terrain_builder/schema.py
- Create: tests/python/__init__.py
- Create: tests/python/test_schema.py

**Interfaces:**
- Produces: SourceClip.validate() -> None.
- Produces: SkeletonSpec.signature() -> str.
- Produces: HoldenClip.validate() -> None.
- Produces: ArtifactSet.validate() -> None.

- [ ] **Step 1: Write the failing schema tests**

~~~python
# tests/python/test_schema.py
import unittest
import numpy as np

from resources.g1_terrain_builder.schema import (
    ArtifactSet, HoldenClip, SkeletonSpec, SourceClip,
)


class SchemaTests(unittest.TestCase):
    def test_source_clip_rejects_bad_qpos_width(self):
        clip = SourceClip(
            name="bad", fps=25.0, qpos=np.zeros((2, 35), np.float32),
            source_frames=np.arange(2), terrain_id="flat",
        )
        with self.assertRaisesRegex(ValueError, "qpos shape"):
            clip.validate()

    def test_skeleton_signature_changes_with_parent_order(self):
        a = SkeletonSpec(("Simulation", "Hips"), np.array([-1, 0], np.int32))
        b = SkeletonSpec(("Simulation", "Hips"), np.array([-1, -1], np.int32))
        self.assertNotEqual(a.signature(), b.signature())

    def test_holden_clip_requires_four_terrain_columns(self):
        clip = HoldenClip.empty(frames=3, bones=2)
        clip.terrain_features = np.zeros((3, 3), np.float32)
        with self.assertRaisesRegex(ValueError, "terrain feature shape"):
            clip.validate()

    def test_artifact_set_rejects_range_overlap(self):
        artifacts = ArtifactSet.empty(frames=4, bones=2)
        artifacts.range_starts = np.array([0, 2], np.int32)
        artifacts.range_stops = np.array([3, 4], np.int32)
        with self.assertRaisesRegex(ValueError, "ranges overlap"):
            artifacts.validate()


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the test and verify the missing module failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_schema -v
~~~

Expected: ERROR with ModuleNotFoundError for resources.g1_terrain_builder.

- [ ] **Step 3: Implement the canonical dataclasses**

~~~python
# resources/g1_terrain_builder/schema.py
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass
class SourceClip:
    name: str
    fps: float
    qpos: np.ndarray
    source_frames: np.ndarray
    terrain_id: str

    def validate(self) -> None:
        if self.qpos.ndim != 2 or self.qpos.shape[1] != 36:
            raise ValueError(f"qpos shape must be (T, 36), got {self.qpos.shape}")
        if self.fps <= 0 or not np.isfinite(self.fps):
            raise ValueError(f"invalid fps {self.fps}")
        if self.source_frames.shape != (len(self.qpos),):
            raise ValueError("source frame shape does not match qpos")
        if not np.isfinite(self.qpos).all():
            raise ValueError("qpos contains non-finite values")


@dataclass(frozen=True)
class SkeletonSpec:
    names: tuple[str, ...]
    parents: np.ndarray

    def signature(self) -> str:
        payload = json.dumps(
            {"names": self.names, "parents": self.parents.tolist()},
            separators=(",", ":"), sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class HoldenClip:
    name: str
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    contacts: np.ndarray
    terrain_features: np.ndarray
    source_frames: np.ndarray
    terrain_id: str

    @classmethod
    def empty(cls, frames: int, bones: int) -> "HoldenClip":
        return cls(
            "empty",
            np.zeros((frames, bones, 3), np.float32),
            np.zeros((frames, bones, 3), np.float32),
            np.tile(np.array([1, 0, 0, 0], np.float32), (frames, bones, 1)),
            np.zeros((frames, bones, 3), np.float32),
            np.zeros((frames, 2), np.uint8),
            np.zeros((frames, 4), np.float32),
            np.arange(frames),
            "flat",
        )

    def validate(self) -> None:
        frames, bones, xyz = self.positions.shape
        if xyz != 3 or self.velocities.shape != (frames, bones, 3):
            raise ValueError("position/velocity shape mismatch")
        if self.rotations.shape != (frames, bones, 4):
            raise ValueError("rotation shape mismatch")
        if self.angular_velocities.shape != (frames, bones, 3):
            raise ValueError("angular velocity shape mismatch")
        if self.contacts.shape != (frames, 2):
            raise ValueError("contact shape mismatch")
        if self.terrain_features.shape != (frames, 4):
            raise ValueError("terrain feature shape must be (T, 4)")
        arrays = (
            self.positions, self.velocities, self.rotations,
            self.angular_velocities, self.terrain_features,
        )
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("converted clip contains non-finite values")


@dataclass
class ArtifactSet:
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    parents: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    contacts: np.ndarray
    terrain_features: np.ndarray

    @classmethod
    def empty(cls, frames: int, bones: int) -> "ArtifactSet":
        clip = HoldenClip.empty(frames, bones)
        return cls(
            clip.positions, clip.velocities, clip.rotations,
            clip.angular_velocities, np.arange(-1, bones - 1, dtype=np.int32),
            np.array([0], np.int32), np.array([frames], np.int32),
            clip.contacts, clip.terrain_features,
        )

    def validate(self) -> None:
        frames = len(self.positions)
        if self.range_starts.shape != self.range_stops.shape:
            raise ValueError("range arrays differ")
        if len(self.range_starts) == 0:
            raise ValueError("no animation ranges")
        if self.range_starts[0] != 0 or self.range_stops[-1] != frames:
            raise ValueError("ranges do not cover all frames")
        if np.any(self.range_starts[1:] < self.range_stops[:-1]):
            raise ValueError("ranges overlap")
        if np.any(self.range_starts >= self.range_stops):
            raise ValueError("empty animation range")
        if self.terrain_features.shape != (frames, 4):
            raise ValueError("terrain feature shape must be (N, 4)")
~~~

Create an empty resources/g1_terrain_builder/__init__.py and tests/python/__init__.py.

- [ ] **Step 4: Run the schema tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_schema -v
~~~

Expected: all focused terrain tests pass.

- [ ] **Step 5: Commit the schema boundary**

~~~bash
git add resources/g1_terrain_builder tests/python
git commit -m "test: define G1 terrain artifact schemas"
~~~

### Task 2: Native Takara and GRAIL source adapters

**Files:**
- Create: resources/g1_terrain_builder/sources.py
- Create: tests/python/test_sources.py

**Interfaces:**
- Consumes: SourceClip from schema.py.
- Produces: load_takara(path: str, remap_path: str) -> SourceClip.
- Produces: load_grail(path: str) -> SourceClip.

- [ ] **Step 1: Write integration tests against the local source data**

~~~python
# tests/python/test_sources.py
import glob
import unittest
import numpy as np

from resources.g1_terrain_builder.sources import load_grail, load_takara

TAKARA = "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz"
REMAP = "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy"
GRAIL_GLOB = "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl"


class SourceTests(unittest.TestCase):
    def test_takara_is_native_g1_qpos(self):
        clip = load_takara(TAKARA, REMAP)
        self.assertEqual(clip.qpos.shape[1], 36)
        self.assertEqual(clip.fps, 50.0)
        np.testing.assert_allclose(
            np.linalg.norm(clip.qpos[:, 3:7], axis=1), 1.0, atol=1e-4,
        )

    def test_grail_is_native_g1_qpos(self):
        path = sorted(glob.glob(GRAIL_GLOB))[0]
        clip = load_grail(path)
        self.assertEqual(clip.qpos.shape, (250, 36))
        self.assertEqual(clip.fps, 25.0)
        self.assertEqual(clip.source_frames[-1], 249)
        np.testing.assert_allclose(
            np.linalg.norm(clip.qpos[:, 3:7], axis=1), 1.0, atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the source tests and verify the import failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sources -v
~~~

Expected: ERROR because sources.py does not exist.

- [ ] **Step 3: Implement both adapters with explicit quaternion ordering**

~~~python
# resources/g1_terrain_builder/sources.py
import os
import joblib
import numpy as np

from .schema import SourceClip


def _normalized_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("zero-length root quaternion")
    q = q / n
    if not np.isfinite(q).all():
        raise ValueError("non-finite root quaternion")
    return q.astype(np.float32)


def load_takara(path: str, remap_path: str) -> SourceClip:
    data = np.load(path)
    remap = np.load(remap_path)
    joints = np.asarray(data["joint_pos"], np.float32)
    root = np.asarray(data["body_pos_w"][:, 0], np.float32)
    quat = _normalized_wxyz(data["body_quat_w"][:, 0])
    qpos = np.zeros((len(joints), 36), np.float32)
    qpos[:, :3] = root
    qpos[:, 3:7] = quat
    qpos[:, 7:] = joints[:, remap]
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    clip = SourceClip(
        "takara_walk_50hz", fps, qpos, np.arange(len(qpos)), "flat",
    )
    clip.validate()
    return clip


def load_grail(path: str) -> SourceClip:
    records = joblib.load(path)
    if len(records) != 1:
        raise ValueError(f"{path}: expected one robot record, got {len(records)}")
    record = next(iter(records.values()))
    dof = np.asarray(record["dof"], np.float32)
    root = np.asarray(record["root_trans_offset"], np.float32)
    xyzw = np.asarray(record["root_rot"], np.float32)
    quat = _normalized_wxyz(xyzw[:, [3, 0, 1, 2]])
    qpos = np.zeros((len(dof), 36), np.float32)
    qpos[:, :3] = root
    qpos[:, 3:7] = quat
    qpos[:, 7:] = dof
    fps = float(record["fps"])
    name = os.path.splitext(os.path.basename(path))[0]
    clip = SourceClip(name, fps, qpos, np.arange(len(qpos)), name)
    clip.validate()
    return clip
~~~

- [ ] **Step 4: Run the source tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_sources -v
~~~

Expected: 2 tests, OK.

- [ ] **Step 5: Commit the source adapters**

~~~bash
git add resources/g1_terrain_builder/sources.py tests/python/test_sources.py
git commit -m "feat: load native Takara and GRAIL G1 motion"
~~~

### Task 3: Quaternion resampling and G1 kinematic conversion

**Files:**
- Create: resources/g1_terrain_builder/resample.py
- Create: resources/g1_terrain_builder/kinematics.py
- Create: tests/python/test_resample.py
- Create: tests/python/test_kinematics.py

**Interfaces:**
- Consumes: SourceClip and SkeletonSpec.
- Produces: resample_vectors(values, source_fps, target_fps) -> ndarray.
- Produces: resample_quaternions_wxyz(values, source_fps, target_fps) -> ndarray.
- Produces: G1Kinematics(xml_path: str).
- Produces: G1Kinematics.world_from_qpos(qpos) -> tuple[ndarray, ndarray].
- Produces: change_basis_zup_to_yup(positions, rotations) -> tuple[ndarray, ndarray].
- Produces: world_to_local(positions, rotations, parents) -> tuple[ndarray, ndarray].
- Produces: forward_local_hierarchy(positions, rotations, parents) -> tuple[ndarray, ndarray].
- Produces: heading_quaternions(forward) -> ndarray, including the antiparallel case.
- Produces: convert_source_clip(source, kinematics) -> tuple[HoldenClip, SkeletonSpec, dict].

- [ ] **Step 1: Write the resampling and FK round-trip tests**

~~~python
# tests/python/test_resample.py
import unittest
import numpy as np
from resources.g1_terrain_builder.resample import (
    output_frame_count, resample_quaternions_wxyz, resample_vectors,
)


class ResampleTests(unittest.TestCase):
    def test_duration_is_preserved(self):
        self.assertEqual(output_frame_count(250, 25.0, 25.0), 250)

    def test_linear_translation(self):
        src = np.array([[0, 0, 0], [1, 0, 0]], np.float64)
        out = resample_vectors(src, 1.0, 2.0)
        np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0], atol=1e-7)

    def test_slerp_takes_short_arc_and_normalizes(self):
        src = np.array([[1, 0, 0, 0], [-0.70710678, 0, -0.70710678, 0]])
        out = resample_quaternions_wxyz(src, 1.0, 2.0)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-7)
        self.assertGreater(out[1, 0], 0.9)

    def test_native_25hz_samples_are_not_time_warped(self):
        src = np.arange(250, dtype=np.float64)[:, None]
        out = resample_vectors(src, 25.0, 25.0)
        np.testing.assert_array_equal(out, src)

    def test_slerp_unrolls_each_bone_independently(self):
        src = np.array([
            [[1,0,0,0], [1,0,0,0]],
            [[-1,0,0,0], [-.70710678,0,-.70710678,0]],
        ], np.float64)
        out = resample_quaternions_wxyz(src, 1.0, 2.0)
        self.assertEqual(out.shape, (3,2,4))
        np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-7)
        self.assertGreater(out[1,0,0], 0.999)
        self.assertGreater(out[1,1,0], 0.9)

    def test_invalid_resampling_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            resample_vectors(np.empty((0, 3)), 25.0, 25.0)
        with self.assertRaises(ValueError):
            resample_vectors(np.zeros((2, 3)), 0.0, 25.0)
        with self.assertRaises(ValueError):
            resample_quaternions_wxyz(np.zeros((2, 4)), 25.0, 25.0)


# tests/python/test_kinematics.py
import unittest
import numpy as np
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics, change_basis_zup_to_yup, convert_source_clip,
    forward_local_hierarchy, heading_quaternions, world_to_local,
)
from resources.g1_terrain_builder.schema import SourceClip

G1_XML = "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"


class KinematicsTests(unittest.TestCase):
    def test_change_of_basis_maps_z_to_y(self):
        p = np.array([[[0.0, 0.0, 1.0]]])
        q = np.array([[[1.0, 0.0, 0.0, 0.0]]])
        py, qy = change_basis_zup_to_yup(p, q)
        np.testing.assert_allclose(py[0, 0], [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(qy, axis=-1), 1, atol=1e-7)

    def test_change_of_basis_conjugates_nontrivial_rotation(self):
        c = 2**-0.5
        p = np.zeros((1,1,3))
        qz = np.array([[[c,0,0,c]]], np.float64)
        _, qy = change_basis_zup_to_yup(p, qz)
        expected = np.array([c,0,c,0])
        self.assertLess(
            min(np.linalg.norm(qy[0,0]-expected),
                np.linalg.norm(qy[0,0]+expected)), 1e-7)

    def test_heading_handles_antiparallel_forward(self):
        forward = np.array([[0,0,1], [0,0,-1], [1,0,0]], np.float64)
        q = heading_quaternions(forward)
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-7)
        c = 2**-0.5
        np.testing.assert_allclose(
            q, [[1,0,0,0], [0,0,1,0], [c,0,c,0]], atol=1e-7)

    def test_world_local_round_trip_is_submillimeter(self):
        kin = G1Kinematics(G1_XML)
        qpos = kin.keyframe_or_zero_qpos()
        gp, gq = kin.world_from_qpos(qpos[None])
        lp, lq = world_to_local(gp, gq, kin.parents)
        rp, rq = kin.forward_local(lp, lq)
        self.assertLess(np.max(np.linalg.norm(rp - gp, axis=-1)), 1e-6)

    def test_convert_source_clip_prepends_simulation_bone(self):
        kin = G1Kinematics(G1_XML)
        qpos = np.tile(kin.keyframe_or_zero_qpos(), (5, 1))
        qpos[:, 0] = np.arange(5) * 0.01
        qpos[:, 7] = np.linspace(0.0, 0.1, 5)
        source = SourceClip(
            "synthetic", 25.0, qpos, np.arange(5), "flat")
        clip, skeleton, report = convert_source_clip(source, kin)
        self.assertEqual(clip.positions.shape, (5, 31, 3))
        self.assertEqual(skeleton.names[0], "Simulation")
        self.assertEqual(skeleton.parents[0], -1)
        self.assertLessEqual(report["fk_max_error_m"], 0.001)
        self.assertLessEqual(report["duration_error_s"], 1.0/25.0)
        np.testing.assert_array_equal(clip.source_frames, np.arange(5))
        np.testing.assert_allclose(
            np.linalg.norm(clip.rotations, axis=-1), 1.0, atol=1e-4)
        self.assertTrue(np.all(
            np.sum(clip.rotations[1:] * clip.rotations[:-1], axis=-1)
            >= -1e-7))
        expected_gp, expected_gq = kin.world_from_qpos(qpos)
        expected_gp, _ = change_basis_zup_to_yup(expected_gp, expected_gq)
        exported_gp, _ = forward_local_hierarchy(
            clip.positions.astype(np.float64),
            clip.rotations.astype(np.float64), skeleton.parents)
        self.assertLess(np.max(np.linalg.norm(
            exported_gp[:, 1:] - expected_gp, axis=-1)), 0.001)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run both tests and verify missing-module failures**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_resample tests.python.test_kinematics -v
~~~

Expected: ERROR because resample.py and kinematics.py do not exist.

- [ ] **Step 3: Implement time-based interpolation and shortest-arc slerp**

~~~python
# resources/g1_terrain_builder/resample.py
import numpy as np


def output_frame_count(frames: int, source_fps: float, target_fps: float) -> int:
    if frames < 1 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("frames and sample rates must be positive")
    duration = (frames - 1) / source_fps
    return int(np.floor(duration * target_fps + 1e-9)) + 1


def _times(frames: int, fps: float) -> np.ndarray:
    return np.arange(frames, dtype=np.float64) / fps


def resample_vectors(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    values = np.asarray(values, np.float64)
    output_frame_count(len(values), source_fps, target_fps)
    if not np.all(np.isfinite(values)):
        raise ValueError("vector samples must be finite")
    src_t = _times(len(values), source_fps)
    count = output_frame_count(len(values), source_fps, target_fps)
    dst_t = _times(count, target_fps)
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(dst_t, src_t, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((count,) + values.shape[1:])


def resample_quaternions_wxyz(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    q = np.asarray(values, np.float64).copy()
    output_frame_count(len(q), source_fps, target_fps)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q)) or np.any(norms < 1e-12):
        raise ValueError("quaternion samples must be finite and nonzero")
    q /= norms
    flat = q.reshape(len(q), -1, 4)
    for t in range(1, len(flat)):
        signs = np.sum(flat[t - 1] * flat[t], axis=-1) < 0
        flat[t, signs] *= -1
    src_t = _times(len(q), source_fps)
    count = output_frame_count(len(q), source_fps, target_fps)
    dst_t = _times(count, target_fps)
    out = np.empty((count, flat.shape[1], 4), np.float64)
    for j in range(flat.shape[1]):
        for k, t in enumerate(dst_t):
            hi = min(np.searchsorted(src_t, t, side="right"), len(src_t) - 1)
            lo = max(0, hi - 1)
            u = 0.0 if hi == lo else (t - src_t[lo]) / (src_t[hi] - src_t[lo])
            a, b = flat[lo, j], flat[hi, j]
            dot = np.clip(np.dot(a, b), -1.0, 1.0)
            if dot > 0.9995:
                x = a + u * (b - a)
            else:
                theta = np.arccos(dot)
                x = (np.sin((1-u)*theta) * a + np.sin(u*theta) * b) / np.sin(theta)
            out[k, j] = x / np.linalg.norm(x)
    return out.reshape((count,) + q.shape[1:])
~~~

- [ ] **Step 4: Implement the G1 FK and coordinate conversion**

Implement resources/g1_terrain_builder/kinematics.py with these exact public members:

~~~python
import sys
import mujoco
import numpy as np
from scipy import signal

sys.path.insert(0, "/home/ubuntu/projects/motion-matching/resources")
import quat as holden_quat

from .resample import resample_quaternions_wxyz, resample_vectors
from .schema import HoldenClip, SkeletonSpec, SourceClip

Q_ZUP_TO_YUP = np.array([2**-0.5, -2**-0.5, 0, 0], np.float64)


class G1Kinematics:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.body_ids = np.arange(1, self.model.nbody)
        id_to_index = {int(body): i for i, body in enumerate(self.body_ids)}
        rename = {
            "pelvis":"Hips",
            "left_hip_pitch_link":"LeftHipPitch",
            "left_hip_roll_link":"LeftHipRoll",
            "left_hip_yaw_link":"LeftHipYaw",
            "left_knee_link":"LeftKnee",
            "left_ankle_pitch_link":"LeftAnkle",
            "left_ankle_roll_link":"LeftToe",
            "right_hip_pitch_link":"RightHipPitch",
            "right_hip_roll_link":"RightHipRoll",
            "right_hip_yaw_link":"RightHipYaw",
            "right_knee_link":"RightKnee",
            "right_ankle_pitch_link":"RightAnkle",
            "right_ankle_roll_link":"RightToe",
            "waist_yaw_link":"Spine",
            "waist_roll_link":"Spine1",
            "torso_link":"Spine2",
            "left_shoulder_pitch_link":"LeftShoulderPitch",
            "left_shoulder_roll_link":"LeftShoulderRoll",
            "left_shoulder_yaw_link":"LeftShoulderYaw",
            "left_elbow_link":"LeftElbow",
            "left_wrist_roll_link":"LeftWristRoll",
            "left_wrist_pitch_link":"LeftWristPitch",
            "left_wrist_yaw_link":"LeftWrist",
            "right_shoulder_pitch_link":"RightShoulderPitch",
            "right_shoulder_roll_link":"RightShoulderRoll",
            "right_shoulder_yaw_link":"RightShoulderYaw",
            "right_elbow_link":"RightElbow",
            "right_wrist_roll_link":"RightWristRoll",
            "right_wrist_pitch_link":"RightWristPitch",
            "right_wrist_yaw_link":"RightWrist",
        }
        raw_names = tuple(self.model.body(int(i)).name for i in self.body_ids)
        missing = [name for name in raw_names if name not in rename]
        if missing:
            raise ValueError(f"unmapped G1 bodies: {missing}")
        self.names = tuple(rename[name] for name in raw_names)
        self.parents = np.array([
            -1 if int(self.model.body(int(i)).parentid[0]) == 0
            else id_to_index[int(self.model.body(int(i)).parentid[0])]
            for i in self.body_ids
        ], np.int32)

    def keyframe_or_zero_qpos(self) -> np.ndarray:
        if self.model.nkey:
            return self.model.key_qpos[0].copy()
        q = np.zeros(self.model.nq)
        q[3] = 1.0
        return q

    def world_from_qpos(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos)
        gp = np.empty((len(qpos), len(self.body_ids), 3))
        gq = np.empty((len(qpos), len(self.body_ids), 4))
        for t, q in enumerate(qpos):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            gp[t] = self.data.xpos[self.body_ids]
            gq[t] = self.data.xquat[self.body_ids]
        return gp, gq

    def forward_local(self, lp: np.ndarray, lq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return forward_local_hierarchy(lp, lq, self.parents)


def forward_local_hierarchy(
    lp: np.ndarray, lq: np.ndarray, parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gp = np.empty_like(lp)
    gq = np.empty_like(lq)
    for i, parent in enumerate(parents):
        if parent < 0:
            gp[:, i], gq[:, i] = lp[:, i], lq[:, i]
        else:
            gq[:, i] = holden_quat.mul(gq[:, parent], lq[:, i])
            gp[:, i] = gp[:, parent] + holden_quat.mul_vec(
                gq[:, parent], lp[:, i])
    return gp, gq


def change_basis_zup_to_yup(gp: np.ndarray, gq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.broadcast_to(Q_ZUP_TO_YUP, gq.shape)
    qi = np.broadcast_to(holden_quat.inv(Q_ZUP_TO_YUP), gq.shape)
    return holden_quat.mul_vec(q, gp), holden_quat.mul(holden_quat.mul(q, gq), qi)


def world_to_local(gp: np.ndarray, gq: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lp = np.empty_like(gp)
    lq = np.empty_like(gq)
    for i, parent in enumerate(parents):
        if parent < 0:
            lp[:, i], lq[:, i] = gp[:, i], gq[:, i]
        else:
            inv = holden_quat.inv(gq[:, parent])
            lp[:, i] = holden_quat.mul_vec(inv, gp[:, i] - gp[:, parent])
            lq[:, i] = holden_quat.mul(inv, gq[:, i])
    return lp, lq


def heading_quaternions(forward: np.ndarray) -> np.ndarray:
    forward = np.asarray(forward, np.float64)
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    q = np.stack([
        np.cos(0.5*yaw), np.zeros_like(yaw),
        np.sin(0.5*yaw), np.zeros_like(yaw),
    ], axis=-1)
    return holden_quat.unroll(q)
~~~

Add these functions after the primitives:

~~~python
def _prepend_simulation(
    gp: np.ndarray, gq: np.ndarray, names: tuple[str, ...],
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SkeletonSpec]:
    hips = names.index("Hips")
    torso = names.index("Spine2")
    sim_position = gp[:, torso].copy()
    sim_position[:, 1] = 0.0
    pos_window = min(13, len(sim_position) if len(sim_position)%2 else len(sim_position)-1)
    if pos_window >= 5:
        sim_position = signal.savgol_filter(
            sim_position, pos_window, min(3, pos_window-2),
            axis=0, mode="interp")
    forward = holden_quat.mul_vec(
        gq[:, hips], np.array([1.0, 0.0, 0.0], np.float64))
    forward[:, 1] = 0.0
    lengths = np.linalg.norm(forward, axis=1, keepdims=True)
    if np.any(lengths < 1e-6):
        raise ValueError("G1 pelvis forward projects to zero")
    forward /= lengths
    dir_window = min(25, len(forward) if len(forward)%2 else len(forward)-1)
    if dir_window >= 5:
        forward = signal.savgol_filter(
            forward, dir_window, min(3, dir_window-2),
            axis=0, mode="interp")
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    sim_rotation = heading_quaternions(forward)
    lp, lq = world_to_local(gp, gq, parents)
    lp[:, hips] = holden_quat.mul_vec(
        holden_quat.inv(sim_rotation), gp[:, hips] - sim_position)
    lq[:, hips] = holden_quat.mul(
        holden_quat.inv(sim_rotation), gq[:, hips])
    positions = np.concatenate([sim_position[:, None], lp], axis=1)
    rotations = np.concatenate([sim_rotation[:, None], lq], axis=1)
    rotations = holden_quat.unroll(holden_quat.normalize(rotations))
    skeleton = SkeletonSpec(
        ("Simulation",) + names,
        np.concatenate([np.array([-1], np.int32), parents + 1]),
    )
    return positions, rotations, skeleton


def convert_source_clip(
    source: SourceClip, kinematics: G1Kinematics, target_fps: float = 25.0,
) -> tuple[HoldenClip, SkeletonSpec, dict]:
    source.validate()
    gp_z, gq_z = kinematics.world_from_qpos(source.qpos)
    gp_y, gq_y = change_basis_zup_to_yup(gp_z, gq_z)
    gp = resample_vectors(gp_y, source.fps, target_fps)
    gq = resample_quaternions_wxyz(gq_y, source.fps, target_fps)
    positions, rotations, skeleton = _prepend_simulation(
        gp, gq, kinematics.names, kinematics.parents)
    source_t = np.arange(len(source.qpos)) / source.fps
    output_t = np.arange(len(positions), dtype=np.float64) / target_fps
    output_frames = np.rint(output_t * source.fps).astype(np.int64)
    output_frames = np.clip(output_frames, 0, len(source.qpos) - 1)
    positions = positions.astype(np.float32)
    rotations = rotations.astype(np.float32)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
        raise ValueError(f"{source.name}: non-finite exported transform")
    quaternion_error = float(np.max(np.abs(
        np.linalg.norm(rotations, axis=-1) - 1.0)))
    if quaternion_error > 1e-4:
        raise ValueError(
            f"{source.name}: exported quaternion norm error {quaternion_error}")
    exported_gp, _ = forward_local_hierarchy(
        positions.astype(np.float64), rotations.astype(np.float64),
        skeleton.parents)
    fk_error = float(np.max(np.linalg.norm(
        exported_gp[:, 1:] - gp, axis=-1)))
    if fk_error > 0.001:
        raise ValueError(f"{source.name}: exported FK error {fk_error} m")
    clip = HoldenClip(
        source.name,
        positions,
        np.zeros_like(positions, np.float32),
        rotations,
        np.zeros_like(positions, np.float32),
        np.zeros((len(positions), 2), np.uint8),
        np.zeros((len(positions), 4), np.float32),
        source.source_frames[output_frames],
        source.terrain_id,
    )
    source_duration = (len(source.qpos)-1) / source.fps
    output_duration = (len(positions)-1) / target_fps
    duration_error = abs(output_duration-source_duration)
    if duration_error > 1.0 / target_fps + 1e-12:
        raise ValueError(f"{source.name}: duration error {duration_error} s")
    return clip, skeleton, {
        "fk_max_error_m": fk_error,
        "duration_error_s": duration_error,
        "quaternion_norm_max_error": quaternion_error,
    }
~~~

- [ ] **Step 5: Run the resampling and kinematics tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_resample tests.python.test_kinematics -v
~~~

Expected: all focused resampling and kinematics tests pass.

- [ ] **Step 6: Commit the kinematic conversion**

~~~bash
git add resources/g1_terrain_builder/resample.py \
  resources/g1_terrain_builder/kinematics.py \
  tests/python/test_resample.py tests/python/test_kinematics.py
git commit -m "feat: convert native G1 motion to Holden bones"
~~~

### Task 4: GRAIL terrain alignment and terrain feature sampling

**Files:**
- Create: resources/g1_terrain_builder/terrain.py
- Create: tests/python/test_terrain.py

**Interfaces:**
- Produces: FlatTerrain.height(x: float, z: float) -> float.
- Produces: GrailTerrain.from_base(base: str) -> GrailTerrain.
- Produces: GrailTerrain.export_obj(path: str) -> None from the same transformed source mesh.
- Produces: build_facing_centerline(root_xz, headings_xz, path_xz) -> ndarray.
- Produces: sample_terrain_features(terrain, centerline) -> ndarray shape (4,).
- Produces: export_heightfield(terrain, bounds, cell_size, path) -> dict.

- [ ] **Step 1: Write synthetic centerline and height tests**

~~~python
# tests/python/test_terrain.py
import tempfile
import unittest
import os
import struct
import numpy as np
from resources.g1_terrain_builder.terrain import (
    StepTerrain, build_facing_centerline, export_heightfield,
    sample_terrain_features,
)


class TerrainTests(unittest.TestCase):
    def test_stationary_centerline_extends_current_heading(self):
        root = np.array([0.0, 0.0])
        headings = np.array([[1.0, 0.0], [1.0, 0.0]])
        path = np.array([[0.0, 0.0], [0.0, 0.0]])
        line = build_facing_centerline(root, headings, path)
        np.testing.assert_allclose(line[-1], [1.0, 0.0], atol=1e-6)

    def test_step_features_are_ground_relative(self):
        terrain = StepTerrain(edge_x=0.5, height=0.29)
        line = np.array([[0, 0], [.25, 0], [.5, 0], [.75, 0], [1, 0]])
        f = sample_terrain_features(terrain, line)
        np.testing.assert_allclose(f, [0.0, 0.29, 0.29, 0.29], atol=1e-6)

    def test_heightfield_binary_contract(self):
        terrain = StepTerrain(edge_x=0.5, height=0.29)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "terrain.bin")
            meta = export_heightfield(
                terrain, (0.0, 1.0, 0.0, 1.0), 0.5, path)
            with open(path, "rb") as stream:
                header = stream.read(struct.calcsize("<4sIII4f"))
                magic, version, nx, nz, ox, oz, cell, exterior = \
                    struct.unpack("<4sIII4f", header)
                values = np.frombuffer(stream.read(), dtype="<f4")
        self.assertEqual((magic, version, nx, nz), (b"G1HF", 1, 3, 3))
        self.assertEqual(meta["nx"], 3)
        np.testing.assert_allclose([ox, oz, cell, exterior], [0,0,.5,0])
        np.testing.assert_allclose(
            values.reshape(nz, nx)[0], [0.0, 0.29, 0.29], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the terrain tests and verify the import failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_terrain -v
~~~

Expected: ERROR because terrain.py does not exist.

- [ ] **Step 3: Implement the height interface and centerline contract**

Start resources/g1_terrain_builder/terrain.py with:

~~~python
from dataclasses import dataclass
import os
import pickle
import struct
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import cKDTree

LOOKAHEAD = np.array([0.25, 0.50, 0.75, 1.00], np.float64)


class FlatTerrain:
    def height(self, x: float, z: float) -> float:
        return 0.0


@dataclass(frozen=True)
class StepTerrain:
    edge_x: float
    height_value: float = 0.29

    def __init__(self, edge_x: float, height: float):
        object.__setattr__(self, "edge_x", edge_x)
        object.__setattr__(self, "height_value", height)

    def height(self, x: float, z: float) -> float:
        return self.height_value if x >= self.edge_x else 0.0


def _normalized(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return fallback.copy() if n < 1e-8 else v / n


def build_facing_centerline(
    root_xz: np.ndarray, headings_xz: np.ndarray, path_xz: np.ndarray,
) -> np.ndarray:
    points = [np.asarray(root_xz, np.float64)]
    last_dir = _normalized(np.asarray(headings_xz[0]), np.array([0.0, 1.0]))
    for p, heading in zip(path_xz[1:], headings_xz[1:]):
        segment = np.asarray(p) - points[-1]
        if np.linalg.norm(segment) > 1e-6:
            points.append(np.asarray(p, np.float64))
        last_dir = _normalized(np.asarray(heading), last_dir)
    travelled = sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:]))
    if travelled < LOOKAHEAD[-1]:
        points.append(points[-1] + last_dir * (LOOKAHEAD[-1] - travelled))
    return np.asarray(points)


def _point_at_arc_distance(line: np.ndarray, distance: float) -> np.ndarray:
    remaining = distance
    for a, b in zip(line, line[1:]):
        length = np.linalg.norm(b - a)
        if remaining <= length:
            return a + (remaining / max(length, 1e-8)) * (b - a)
        remaining -= length
    return line[-1]


def sample_terrain_features(terrain, centerline: np.ndarray) -> np.ndarray:
    h0 = terrain.height(*centerline[0])
    return np.array([
        terrain.height(*_point_at_arc_distance(centerline, d)) - h0
        for d in LOOKAHEAD
    ], np.float32)
~~~

Move the validated USD mesh loading, quad densification, reconstruction transform,
and KD-tree height lookup from /home/ubuntu/projects/g1_mm/terrain.py into the
same file as GrailTerrain. Replace its random triangle-fan samples with a fixed
barycentric lattice so identical inputs produce byte-identical artifacts.
Convert MuJoCo (x, y, z) points to Holden terrain coordinates (x, z, -y) before
constructing the XZ KD-tree, and return Holden Y as height.

Add heightfield export with the runtime plan's exact binary contract:

~~~python
def export_heightfield(terrain, bounds, cell_size: float, path: str) -> dict:
    xmin, xmax, zmin, zmax = bounds
    nx = int(np.ceil((xmax-xmin)/cell_size)) + 1
    nz = int(np.ceil((zmax-zmin)/cell_size)) + 1
    values = np.empty((nz, nx), np.float32)
    for iz in range(nz):
        for ix in range(nx):
            values[iz, ix] = terrain.height(
                xmin + ix*cell_size, zmin + iz*cell_size)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<4sIII4f", b"G1HF", 1, nx, nz,
                                 xmin, zmin, cell_size, 0.0))
        stream.write(values.tobytes())
    return {"nx":nx, "nz":nz, "origin_x":xmin, "origin_z":zmin,
            "cell_size":cell_size, "exterior_height":0.0}
~~~

Export terrain.obj from the same transformed vertices and face indices. Write
v x y z lines in Holden coordinates and one-based f indices. Do not use the
densified query points as render vertices. Implement this as
`GrailTerrain.export_obj(path)` and preserve the USD face winding under the
right-handed coordinate change.

- [ ] **Step 4: Add a real GRAIL alignment assertion**

Extend tests/python/test_terrain.py:

~~~python
from resources.g1_terrain_builder.terrain import GrailTerrain

    def test_real_grail_curb_has_expected_height(self):
        terrain = GrailTerrain.from_base("terrain_curbs__curb_000__000")
        footprint = terrain.footprint()
        self.assertGreater(footprint["height"], 0.1)
        self.assertLess(footprint["height"], 0.5)
        cx = 0.5 * (footprint["x"][0] + footprint["x"][1])
        cz = 0.5 * (footprint["z"][0] + footprint["z"][1])
        self.assertAlmostEqual(terrain.height(cx, cz), footprint["height"], places=2)
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.obj")
            second = os.path.join(tmp, "second.obj")
            terrain.export_obj(first)
            terrain.export_obj(second)
            with open(first, "rb") as a, open(second, "rb") as b:
                self.assertEqual(a.read(), b.read())
            text = open(first, encoding="utf-8").read()
        self.assertIn("\nv ", "\n" + text)
        self.assertIn("\nf ", "\n" + text)
~~~

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_terrain -v
~~~

Expected: 4 tests, OK.

- [ ] **Step 5: Commit terrain sampling**

~~~bash
git add resources/g1_terrain_builder/terrain.py tests/python/test_terrain.py
git commit -m "feat: sample aligned GRAIL terrain"
~~~

### Task 5: Derived motion arrays, contacts, and database ranges

**Files:**
- Create: resources/g1_terrain_builder/database.py
- Create: tests/python/test_database_builder.py

**Interfaces:**
- Consumes: converted 25 Hz HoldenClip skeleton motion and a terrain provider.
- Produces: ContactConfig with explicit speed, height, and median-filter thresholds.
- Produces: derive_velocities(positions, rotations, fps) -> tuple[ndarray, ndarray].
- Produces: derive_contacts(global_positions, terrain, left, right, fps, config) -> ndarray.
- Produces: forward_kinematics_arrays(positions, rotations, parents) -> tuple[ndarray, ndarray].
- Produces: combine_clips(clips, skeleton) -> ArtifactSet.
- Produces: write_holden_database(path, artifacts) -> None.
- Produces: read_holden_database(path) -> ArtifactSet for independent round-trip validation.

- [ ] **Step 1: Write tests for range isolation and binary round-trip**

~~~python
# tests/python/test_database_builder.py
import os
import struct
import tempfile
import unittest
import numpy as np
from resources.g1_terrain_builder.database import (
    ContactConfig, combine_clips, derive_contacts, derive_velocities,
    forward_kinematics_arrays, read_holden_database, write_holden_database,
)
from resources.g1_terrain_builder.schema import HoldenClip, SkeletonSpec
from resources.g1_terrain_builder.terrain import FlatTerrain, StepTerrain


class DatabaseBuilderTests(unittest.TestCase):
    def test_clips_become_nonoverlapping_ranges(self):
        skeleton = SkeletonSpec(("Simulation", "Hips"), np.array([-1, 0], np.int32))
        artifacts = combine_clips(
            [HoldenClip.empty(3, 2), HoldenClip.empty(5, 2)], skeleton,
        )
        np.testing.assert_array_equal(artifacts.range_starts, [0, 3])
        np.testing.assert_array_equal(artifacts.range_stops, [3, 8])

    def test_derivatives_are_physical_and_isolated_per_clip(self):
        fps = 25.0
        positions = np.zeros((5, 1, 3), np.float64)
        positions[:, 0, 0] = np.arange(5) / fps
        angle = np.arange(5) * 0.1
        rotations = np.zeros((5, 1, 4), np.float64)
        rotations[:, 0, 0] = np.cos(angle / 2)
        rotations[:, 0, 2] = np.sin(angle / 2)
        velocity, angular = derive_velocities(positions, rotations, fps)
        np.testing.assert_allclose(velocity[:, 0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(angular[:, 0, 1], 2.5, atol=1e-5)

        shifted = positions.copy()
        shifted[:, 0, 0] += 100.0
        shifted_velocity, _ = derive_velocities(shifted, rotations, fps)
        clip_a, clip_b = HoldenClip.empty(5, 1), HoldenClip.empty(5, 1)
        clip_a.positions, clip_a.velocities = positions, velocity
        clip_b.positions, clip_b.velocities = shifted, shifted_velocity
        artifacts = combine_clips(
            [clip_a, clip_b],
            SkeletonSpec(("Simulation",), np.array([-1], np.int32)))
        self.assertAlmostEqual(artifacts.velocities[4, 0, 0], 1.0)
        self.assertAlmostEqual(artifacts.velocities[5, 0, 0], 1.0)

    def test_contacts_use_terrain_relative_height_and_speed(self):
        feet = np.zeros((5, 2, 3), np.float64)
        feet[:, 0] = [0.75, 0.31, 0.0]
        feet[:, 1, 0] = np.arange(5) * 0.02
        feet[:, 1, 1] = 0.02
        contacts = derive_contacts(
            feet, StepTerrain(0.5, 0.29), 0, 1, 25.0, ContactConfig())
        np.testing.assert_array_equal(contacts[:, 0], np.ones(5, np.uint8))
        np.testing.assert_array_equal(contacts[:, 1], np.zeros(5, np.uint8))
        feet[:, 0, 1] = 0.0
        penetrated = derive_contacts(
            feet, StepTerrain(0.5, 0.29), 0, 1, 25.0, ContactConfig())
        np.testing.assert_array_equal(
            penetrated[:, 0], np.zeros(5, np.uint8))

    def test_invalid_derivative_and_contact_inputs_are_rejected(self):
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1, 1))
        with self.assertRaisesRegex(ValueError, "three"):
            derive_velocities(np.zeros((2, 1, 3)), rotations, 25.0)
        feet = np.zeros((5, 2, 3))
        with self.assertRaisesRegex(ValueError, "foot indices"):
            derive_contacts(feet, FlatTerrain(), 0, 2, 25.0)
        with self.assertRaisesRegex(ValueError, "filter"):
            derive_contacts(
                feet, FlatTerrain(), 0, 1, 25.0,
                ContactConfig(median_filter_frames=2))

    def test_forward_kinematics_uses_parent_rotation(self):
        positions = np.zeros((1, 2, 3), np.float64)
        positions[:, 1, 0] = 1.0
        rotations = np.zeros((1, 2, 4), np.float64)
        rotations[:, :, 0] = 1.0
        c = 2**-0.5
        rotations[:, 0] = [c, 0, c, 0]
        gp, gq = forward_kinematics_arrays(
            positions, rotations, np.array([-1, 0], np.int32))
        np.testing.assert_allclose(gp[0, 1], [0, 0, -1], atol=1e-7)
        np.testing.assert_allclose(gq[0, 1], rotations[0, 0], atol=1e-7)

    def test_holden_binary_round_trip(self):
        skeleton = SkeletonSpec(("Simulation", "Hips"), np.array([-1, 0], np.int32))
        artifacts = combine_clips([HoldenClip.empty(4, 2)], skeleton)
        artifacts.positions[:] = np.arange(
            artifacts.positions.size, dtype=np.float32).reshape(
                artifacts.positions.shape) / 10.0
        artifacts.velocities[:] = -artifacts.positions
        artifacts.contacts[:, 0] = [0, 1, 0, 1]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "database.bin")
            write_holden_database(path, artifacts)
            loaded = read_holden_database(path)
            with open(path, "rb") as stream:
                self.assertEqual(struct.unpack("<II", stream.read(8)), (4, 2))
        for name in (
            "positions", "velocities", "rotations", "angular_velocities",
            "parents", "range_starts", "range_stops", "contacts",
        ):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(artifacts, name))
        self.assertEqual(loaded.positions.dtype, np.dtype("<f4"))
        self.assertEqual(loaded.parents.dtype, np.dtype("<i4"))

    def test_holden_reader_rejects_truncation_and_trailing_bytes(self):
        skeleton = SkeletonSpec(("Simulation",), np.array([-1], np.int32))
        artifacts = combine_clips([HoldenClip.empty(4, 1)], skeleton)
        with tempfile.TemporaryDirectory() as td:
            valid = os.path.join(td, "valid.bin")
            write_holden_database(valid, artifacts)
            payload = open(valid, "rb").read()
            for name, corrupt in (
                ("truncated.bin", payload[:-1]),
                ("trailing.bin", payload + b"x"),
            ):
                path = os.path.join(td, name)
                with open(path, "wb") as stream:
                    stream.write(corrupt)
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "truncated|trailing"):
                        read_holden_database(path)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the database tests and verify the import failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_database_builder -v
~~~

Expected: ERROR because database.py does not exist.

- [ ] **Step 3: Implement per-clip derivatives and exact Holden serialization**

The binary writer must match database_load in database.h:

~~~python
# resources/g1_terrain_builder/database.py
from dataclasses import dataclass
import struct
import sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, "/home/ubuntu/projects/motion-matching/resources")
import quat as holden_quat

from .schema import ArtifactSet, HoldenClip, SkeletonSpec


@dataclass(frozen=True)
class ContactConfig:
    speed_threshold: float = 0.15
    height_threshold: float = 0.06
    median_filter_frames: int = 3


def _write_array2(f, a: np.ndarray, dtype) -> None:
    a = np.ascontiguousarray(a, dtype=np.dtype(dtype))
    f.write(struct.pack("<II", a.shape[0], a.shape[1]))
    f.write(a.tobytes())


def _write_array1(f, a: np.ndarray, dtype) -> None:
    a = np.ascontiguousarray(a, dtype=np.dtype(dtype))
    f.write(struct.pack("<I", len(a)))
    f.write(a.tobytes())


def combine_clips(clips: list[HoldenClip], skeleton: SkeletonSpec) -> ArtifactSet:
    if not clips:
        raise ValueError("at least one clip is required")
    for clip in clips:
        clip.validate()
        if clip.positions.shape[1] != len(skeleton.parents):
            raise ValueError(f"{clip.name}: bone count does not match skeleton")
    lengths = np.array([len(c.positions) for c in clips], np.int32)
    stops = np.cumsum(lengths, dtype=np.int32)
    starts = np.concatenate([np.array([0], np.int32), stops[:-1]])
    out = ArtifactSet(
        np.concatenate([c.positions for c in clips]).astype(np.float32),
        np.concatenate([c.velocities for c in clips]).astype(np.float32),
        np.concatenate([c.rotations for c in clips]).astype(np.float32),
        np.concatenate([c.angular_velocities for c in clips]).astype(np.float32),
        skeleton.parents.astype(np.int32),
        starts, stops,
        np.concatenate([c.contacts for c in clips]).astype(np.uint8),
        np.concatenate([c.terrain_features for c in clips]).astype(np.float32),
    )
    out.validate()
    return out


def write_holden_database(path: str, a: ArtifactSet) -> None:
    a.validate()
    with open(path, "wb") as f:
        _write_array2(f, a.positions, "<f4")
        _write_array2(f, a.velocities, "<f4")
        _write_array2(f, a.rotations, "<f4")
        _write_array2(f, a.angular_velocities, "<f4")
        _write_array1(f, a.parents, "<i4")
        _write_array1(f, a.range_starts, "<i4")
        _write_array1(f, a.range_stops, "<i4")
        _write_array2(f, a.contacts, "u1")
~~~

Add these exact readers and derivations:

~~~python
def _read_array1(stream, dtype):
    header = stream.read(4)
    if len(header) != 4:
        raise ValueError("truncated array1 header")
    rows, = struct.unpack("<I", header)
    shape = (rows,)
    count = int(np.prod(shape))
    raw = stream.read(count * np.dtype(dtype).itemsize)
    if len(raw) != count * np.dtype(dtype).itemsize:
        raise ValueError("truncated array payload")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _read_array2(stream, dtype, components=()):
    header = stream.read(8)
    if len(header) != 8:
        raise ValueError("truncated array2 header")
    rows, cols = struct.unpack("<II", header)
    shape = (rows, cols) + tuple(components)
    count = int(np.prod(shape))
    raw = stream.read(count * np.dtype(dtype).itemsize)
    if len(raw) != count * np.dtype(dtype).itemsize:
        raise ValueError("truncated array payload")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def read_holden_database(path: str) -> ArtifactSet:
    with open(path, "rb") as stream:
        positions = _read_array2(stream, "<f4", (3,))
        velocities = _read_array2(stream, "<f4", (3,))
        rotations = _read_array2(stream, "<f4", (4,))
        angular = _read_array2(stream, "<f4", (3,))
        parents = _read_array1(stream, "<i4")
        starts = _read_array1(stream, "<i4")
        stops = _read_array1(stream, "<i4")
        contacts = _read_array2(stream, "u1")
        if stream.read(1):
            raise ValueError("trailing database bytes")
    out = ArtifactSet(
        positions, velocities, rotations, angular, parents,
        starts, stops, contacts,
        np.zeros((len(positions), 4), np.float32),
    )
    out.validate()
    return out


def derive_velocities(positions, rotations, fps):
    positions = np.asarray(positions, np.float64)
    rotations = np.asarray(rotations, np.float64)
    if len(positions) < 3 or positions.shape[:2] != rotations.shape[:2] or \
            rotations.shape[-1] != 4 or positions.shape[-1] != 3:
        raise ValueError("derivatives require at least three aligned bone frames")
    if fps <= 0 or not np.isfinite(fps) or not np.all(np.isfinite(positions)) \
            or not np.all(np.isfinite(rotations)):
        raise ValueError("derivative inputs and fps must be finite and valid")
    velocity = np.gradient(positions, axis=0, edge_order=2) * fps
    angular = np.zeros(positions.shape, np.float64)
    forward = holden_quat.to_scaled_angle_axis(holden_quat.abs(
        holden_quat.mul_inv(rotations[1:], rotations[:-1]))) * fps
    angular[0] = forward[0]
    angular[-1] = forward[-1]
    angular[1:-1] = 0.5 * (forward[:-1] + forward[1:])
    return velocity.astype(np.float32), angular.astype(np.float32)


def derive_contacts(
    global_positions, terrain, left, right, fps,
    config: ContactConfig = ContactConfig(),
):
    global_positions = np.asarray(global_positions, np.float64)
    if len(global_positions) < 3 or global_positions.ndim != 3 or \
            global_positions.shape[-1] != 3:
        raise ValueError("contacts require at least three global-position frames")
    if fps <= 0 or not np.isfinite(fps) or not np.all(np.isfinite(global_positions)):
        raise ValueError("contact positions and fps must be finite and valid")
    if left < 0 or right < 0 or left >= global_positions.shape[1] or \
            right >= global_positions.shape[1]:
        raise ValueError("contact foot indices are out of range")
    if not np.all(np.isfinite([
            config.speed_threshold, config.height_threshold])) or \
            config.speed_threshold <= 0 or config.height_threshold <= 0 or \
            config.median_filter_frames < 1 or config.median_filter_frames % 2 == 0:
        raise ValueError("contact thresholds must be positive and filter size odd")
    feet = global_positions[:, [left, right]]
    velocity = np.gradient(feet, axis=0, edge_order=2) * fps
    speed = np.linalg.norm(velocity, axis=-1)
    ground = np.empty((len(feet), 2), np.float64)
    for t in range(len(feet)):
        for side in range(2):
            ground[t, side] = terrain.height(
                feet[t, side, 0], feet[t, side, 2])
    if not np.all(np.isfinite(ground)):
        raise ValueError("terrain returned non-finite contact height")
    relative_height = feet[:, :, 1] - ground
    contacts = (speed < config.speed_threshold) & \
               (np.abs(relative_height) < config.height_threshold)
    for side in range(2):
        contacts[:, side] = ndimage.median_filter(
            contacts[:, side], size=config.median_filter_frames, mode="nearest")
    return contacts.astype(np.uint8)


def forward_kinematics_arrays(positions, rotations, parents):
    gp = np.empty_like(positions)
    gq = np.empty_like(rotations)
    for bone, parent in enumerate(parents):
        if parent < 0:
            gp[:, bone] = positions[:, bone]
            gq[:, bone] = rotations[:, bone]
        else:
            gq[:, bone] = holden_quat.mul(
                gq[:, parent], rotations[:, bone])
            gp[:, bone] = gp[:, parent] + holden_quat.mul_vec(
                gq[:, parent], positions[:, bone])
    return gp, gq
~~~

Call derive_velocities and derive_contacts separately for every clip before
combine_clips, so np.gradient never crosses a source boundary.

- [ ] **Step 4: Run the database tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_database_builder -v
~~~

Expected: all focused database-builder tests pass.

- [ ] **Step 5: Commit database derivation**

~~~bash
git add resources/g1_terrain_builder/database.py \
  tests/python/test_database_builder.py
git commit -m "feat: serialize validated Holden G1 clips"
~~~

### Task 6: Versioned sidecars and atomic publication

**Files:**
- Create: resources/g1_terrain_builder/artifacts.py
- Create: tests/python/test_artifacts.py

**Interfaces:**
- Produces: write_terrain_sidecar(path, features) -> None.
- Produces: read_terrain_sidecar(path) -> ndarray.
- Produces: publish_artifacts(output_dir, artifacts, skeleton, manifest) -> None.
- Sidecar header: magic bytes G1TF, uint32 version 1, uint32 frames, uint32 dims 4.

- [ ] **Step 1: Write corruption and atomic-publish tests**

~~~python
# tests/python/test_artifacts.py
import json
import os
import tempfile
import unittest
import numpy as np
from resources.g1_terrain_builder.artifacts import (
    read_terrain_sidecar, write_terrain_sidecar,
)


class ArtifactTests(unittest.TestCase):
    def test_terrain_sidecar_round_trip(self):
        f = np.arange(20, dtype=np.float32).reshape(5, 4)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "terrain_features.bin")
            write_terrain_sidecar(path, f)
            out = read_terrain_sidecar(path)
        np.testing.assert_array_equal(out, f)

    def test_terrain_sidecar_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.bin")
            with open(path, "wb") as stream:
                stream.write(b"G1TF" + (1).to_bytes(4, "little"))
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_terrain_sidecar(path)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the artifact tests and verify the import failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts -v
~~~

Expected: ERROR because artifacts.py does not exist.

- [ ] **Step 3: Implement the sidecar and atomic directory replace**

~~~python
# resources/g1_terrain_builder/artifacts.py
import json
import os
import shutil
import struct
import tempfile
import numpy as np

MAGIC = b"G1TF"
VERSION = 1
DIMS = 4


def write_terrain_sidecar(path: str, features: np.ndarray) -> None:
    f = np.ascontiguousarray(features, np.float32)
    if f.ndim != 2 or f.shape[1] != DIMS or not np.isfinite(f).all():
        raise ValueError(f"terrain features must be finite (N, 4), got {f.shape}")
    with open(path, "wb") as stream:
        stream.write(struct.pack("<4sIII", MAGIC, VERSION, len(f), DIMS))
        stream.write(f.tobytes())


def read_terrain_sidecar(path: str) -> np.ndarray:
    data = open(path, "rb").read()
    if len(data) < 16:
        raise ValueError(f"{path}: truncated terrain sidecar header")
    magic, version, frames, dims = struct.unpack("<4sIII", data[:16])
    if magic != MAGIC or version != VERSION or dims != DIMS:
        raise ValueError(f"{path}: unsupported terrain sidecar schema")
    expected = 16 + frames * dims * 4
    if len(data) != expected:
        raise ValueError(f"{path}: truncated or trailing terrain data")
    return np.frombuffer(data, np.float32, offset=16).reshape(frames, dims).copy()
~~~

Add:

~~~python
def publish_artifacts(output_dir, artifacts, manifest, terrain_writer):
    parent = os.path.dirname(os.path.abspath(output_dir))
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".g1_terrain-", dir=parent)
    backup = output_dir + ".previous"
    try:
        write_holden_database(os.path.join(staging, "database.bin"), artifacts)
        write_terrain_sidecar(
            os.path.join(staging, "terrain_features.bin"),
            artifacts.terrain_features)
        with open(os.path.join(staging, "manifest.json"), "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        with open(os.path.join(staging, "validation.json"), "w") as stream:
            json.dump(manifest["validation"], stream, indent=2, sort_keys=True)
        terrain_writer(staging)
        loaded = read_holden_database(os.path.join(staging, "database.bin"))
        loaded.terrain_features = read_terrain_sidecar(
            os.path.join(staging, "terrain_features.bin"))
        loaded.validate()
        if os.path.exists(backup):
            shutil.rmtree(backup)
        if os.path.exists(output_dir):
            os.replace(output_dir, backup)
        os.replace(staging, output_dir)
        staging = ""
        if os.path.exists(backup):
            shutil.rmtree(backup)
    except Exception:
        if os.path.exists(backup) and not os.path.exists(output_dir):
            os.replace(backup, output_dir)
        raise
    finally:
        if staging and os.path.exists(staging):
            shutil.rmtree(staging)
~~~

Import write_holden_database and read_holden_database from database.py.

- [ ] **Step 4: Run the artifact tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts -v
~~~

Expected: 2 tests, OK.

- [ ] **Step 5: Commit artifact publication**

~~~bash
git add resources/g1_terrain_builder/artifacts.py tests/python/test_artifacts.py
git commit -m "feat: publish versioned G1 terrain artifacts"
~~~

### Task 7: Builder and independent validator CLIs

**Files:**
- Create: resources/build_g1_terrain_database.py
- Create: resources/validate_g1_terrain_database.py
- Create: tests/python/test_build_cli.py

**Interfaces:**
- build CLI: --output, --grail-glob, --grail-limit, --g1-xml, --takara, --remap.
- validate CLI: one positional artifact directory; exit 0 only on full validation.
- Produces manifest schema g1-terrain-artifacts/v1.

- [ ] **Step 1: Write the one-clip CLI integration test**

~~~python
# tests/python/test_build_cli.py
import json
import os
import subprocess
import tempfile
import unittest

PYTHON = "/home/ubuntu/miniconda3/envs/diffsim/bin/python"


class BuildCliTests(unittest.TestCase):
    def test_one_grail_clip_builds_and_validates(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([
                PYTHON, "resources/build_g1_terrain_database.py",
                "--output", os.path.join(td, "g1_terrain"),
                "--grail-limit", "1",
            ], check=True)
            subprocess.run([
                PYTHON, "resources/validate_g1_terrain_database.py",
                os.path.join(td, "g1_terrain"),
            ], check=True)
            manifest = json.load(open(os.path.join(td, "g1_terrain", "manifest.json")))
        self.assertEqual(manifest["schema"], "g1-terrain-artifacts/v1")
        self.assertEqual(manifest["feature_dimensions"], 31)
        self.assertEqual(manifest["terrain_dimensions"], 4)
        self.assertEqual(manifest["grail_clips"], 1)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the CLI test and verify the missing-script failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_build_cli -v
~~~

Expected: FAIL because build_g1_terrain_database.py is missing.

- [ ] **Step 3: Implement the builder orchestration**

The builder main function must use these defaults:

~~~python
DEFAULTS = {
    "output": "resources/g1_terrain",
    "grail_glob": "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl",
    "g1_xml": "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml",
    "takara": "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz",
    "remap": "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy",
}
~~~

Use this finalization function:

~~~python
def finalize_clip(source, terrain, kin):
    clip, skeleton, report = convert_source_clip(source, kin, 25.0)
    gp, gq = forward_kinematics_arrays(
        clip.positions, clip.rotations, skeleton.parents)
    clip.velocities, clip.angular_velocities = derive_velocities(
        clip.positions, clip.rotations, 25.0)
    clip.contacts = derive_contacts(
        gp, terrain,
        skeleton.names.index("LeftToe"),
        skeleton.names.index("RightToe"), 25.0)
    for frame in range(len(clip.positions)):
        stop = min(frame + 51, len(clip.positions))
        path = gp[frame:stop, 0][:, [0, 2]]
        headings3 = holden_quat.mul_vec(
            gq[frame:stop, 0],
            np.array([0.0, 0.0, 1.0], np.float64))
        headings = headings3[:, [0, 2]]
        centerline = build_facing_centerline(path[0], headings, path)
        clip.terrain_features[frame] = sample_terrain_features(
            terrain, centerline)
    clip.validate()
    return clip, skeleton, report
~~~

forward_kinematics_arrays is the same parents-first loop as
G1Kinematics.forward_local but accepts the prepended Simulation skeleton.

The CLI build loop is:

~~~python
kin = G1Kinematics(args.g1_xml)
sources = [load_takara(args.takara, args.remap)]
grail_paths = sorted(glob.glob(args.grail_glob))
if args.grail_limit is not None:
    grail_paths = grail_paths[:args.grail_limit]
sources.extend(load_grail(path) for path in grail_paths)
clips, reports, source_manifest = [], [], []
expected_skeleton = None
for source in sources:
    terrain = FlatTerrain() if source.terrain_id == "flat" \
        else GrailTerrain.from_base(source.terrain_id)
    clip, skeleton, report = finalize_clip(source, terrain, kin)
    if expected_skeleton is None:
        expected_skeleton = skeleton
    elif skeleton.signature() != expected_skeleton.signature():
        raise ValueError(f"{source.name}: skeleton signature changed")
    clips.append(clip)
    reports.append(report)
    source_manifest.append({
        "name":source.name, "terrain_id":source.terrain_id,
        "source_frames":len(source.qpos), "output_frames":len(clip.positions),
        "source_frame_map":clip.source_frames.tolist(),
    })
artifacts = combine_clips(clips, expected_skeleton)
~~~

Build the manifest from these values and call publish_artifacts. The terrain_writer
exports the base selected by --runtime-terrain, default
terrain_curbs__curb_000__000, to terrain.bin and terrain.obj with a 0.02 m cell
size and a 2 m flat border around its transformed footprint.

Normal mode must abort on the first skipped clip. The --grail-limit option slices
the sorted input list and writes diagnostic_mode=true to the manifest. The
manifest must contain source and output frame counts, range/source mapping,
skeleton names/parents/signature, 25 Hz output rate, contact thresholds, terrain
distances, coordinate mapping, and per-clip FK error.

- [ ] **Step 4: Implement the independent validator**

The validator must reload every artifact rather than trusting the in-memory build.
It checks:

~~~python
assert manifest["schema"] == "g1-terrain-artifacts/v1"
assert manifest["output_fps"] == 25.0
assert manifest["feature_dimensions"] == 31
assert manifest["terrain_dimensions"] == 4
assert len(database.positions) == len(terrain_features)
assert database.positions.shape[1] == len(manifest["skeleton"]["names"])
assert max(manifest["validation"]["fk_max_error_m"]) <= 0.001
assert np.max(np.abs(np.linalg.norm(database.rotations, axis=-1) - 1.0)) <= 1e-4
~~~

It also calls ArtifactSet.validate(), verifies every source range and source-frame
mapping length, checks terrain.bin and terrain.obj exist, and prints a single
summary line:

    VALID g1-terrain-artifacts/v1 frames=<N> clips=<C> bones=31 terrain_dims=4

- [ ] **Step 5: Run the complete Python suite**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v
~~~

Expected: all tests OK, including the one-GRAIL-clip build.

- [ ] **Step 6: Build and validate a diagnostic artifact set**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/build_g1_terrain_database.py \
  --output /tmp/g1_terrain_10 --grail-limit 10
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py /tmp/g1_terrain_10
~~~

Expected: VALID line with 11 clips: one Takara plus ten GRAIL.

- [ ] **Step 7: Commit the CLIs**

~~~bash
git add resources/build_g1_terrain_database.py \
  resources/validate_g1_terrain_database.py tests/python/test_build_cli.py
git commit -m "feat: build validated G1 terrain artifacts"
~~~

### Task 8: Full-corpus build gate

**Files:**
- Generated, not committed: resources/g1_terrain/
- Modify: .gitignore

**Interfaces:**
- Consumes the completed builder and validator.
- Produces a local full-corpus artifact set accepted by the C++ runtime plan.

- [ ] **Step 1: Ignore generated artifacts without ignoring builder source**

Add exactly this entry to .gitignore:

~~~gitignore
/resources/g1_terrain/
~~~

- [ ] **Step 2: Run all unit and diagnostic integration tests**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v
~~~

Expected: all tests OK.

- [ ] **Step 3: Build the complete local corpus**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/build_g1_terrain_database.py
~~~

Expected: exit 0, grail_clips=1769, skipped_clips=0, and artifacts published only
to resources/g1_terrain.

- [ ] **Step 4: Independently validate the published corpus**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py resources/g1_terrain
~~~

Expected: VALID line with clips=1770 and bones=31.

- [ ] **Step 5: Inspect repository scope and commit the ignore rule**

Run:

~~~bash
git status --short
git diff --check
~~~

Expected: resources/g1_terrain is absent from status; unrelated pre-existing user
changes remain unstaged.

~~~bash
git add .gitignore
git commit -m "chore: ignore generated G1 terrain artifacts"
~~~
