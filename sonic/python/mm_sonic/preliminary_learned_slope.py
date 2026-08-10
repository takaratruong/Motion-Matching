"""Preliminary projector-free learned replay of one authenticated G1 slope.

This module is intentionally outside the production manifest lifecycle.  It
exists only to make the complete Orange Duck compressor/decompressor/stepper
path visible on the exact 595-row authored route.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Protocol

import numpy as np

from resources import quat
from resources.build_g1_terrain_database import _assemble_authored_slope_source
from resources.g1_lmm.dataset import G1LmmDimensions, TrainingBundle
from resources.g1_terrain_builder.database import combine_clips
from resources.g1_terrain_builder.features import build_matching_features
from resources.g1_terrain_builder.kinematics import Q_ZUP_TO_YUP, G1Kinematics
from resources.g1_terrain_builder.resample import (
    resample_quaternions_wxyz,
    resample_vectors,
)
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    HoldenClip,
    SkeletonSpec,
)
from resources.g1_terrain_builder.sources import load_authenticated_grail_slope
from resources.g1_terrain_builder.terrain import (
    build_facing_centerline,
    sample_terrain_features,
)

PRELIMINARY_LABEL = (
    "PRELIMINARY LEARNED EXACT-ROUTE OVERFIT (NOT ACCEPTED/NO GENERALIZATION)"
)
PRELIMINARY_VISUAL_V2_LABEL = (
    "PRELIMINARY V2 LEARNED EXACT-ROUTE OVERFIT "
    "(POST-HOC VISUAL GATE; NOT ACCEPTED/NO GENERALIZATION)"
)
MODEL_ARTIFACT_NAMES = ("latent.bin", "decompressor.bin", "stepper.bin")
CLIP_ID = "terrain_slopes__slope_000__000"
FPS = 60.0
DT = 1.0 / FPS
REFERENCE_QPOS_SHA256 = (
    "b2abecf5e0423708b6b9330347aab051502158a965e9fb0fa2d7873b9d8ee4be"
)
GPU3_UUID = "GPU-87fb0777-4169-d4e9-12cc-382bdf7c0730"

DEFAULT_SLOPE_ROOT = Path("/home/ubuntu/datasets/GRAIL/data/slope")
DEFAULT_SLOPE_ROBOT = DEFAULT_SLOPE_ROOT / "robot" / f"{CLIP_ID}.pkl"
DEFAULT_SLOPE_USD = DEFAULT_SLOPE_ROOT / "object_usd" / f"{CLIP_ID}.usd"
DEFAULT_SLOPE_RECON = DEFAULT_SLOPE_ROOT / "recon" / f"{CLIP_ID}.pkl"
DEFAULT_SLOPE_METADATA = DEFAULT_SLOPE_ROOT / "meta" / f"{CLIP_ID}.pkl"
DEFAULT_G1_XML = Path(
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_OUTPUT = (
    REPOSITORY_ROOT / "sonic/runs/g1-lmm-authored-slope-preliminary/model-v1"
).resolve()
SOLE_FIT_CLAIM = (
    DEFAULT_MODEL_OUTPUT.parent / ".preliminary-learned-slope-sole-fit.claim"
)
DEFAULT_VISUAL_V2_MODEL_OUTPUT = (
    REPOSITORY_ROOT
    / "sonic/runs/g1-lmm-authored-slope-preliminary/model-v2-visual"
).resolve()
VISUAL_V2_SOLE_FIT_CLAIM = (
    DEFAULT_VISUAL_V2_MODEL_OUTPUT.parent
    / ".preliminary-learned-slope-visual-v2-sole-fit.claim"
)


class _Network(Protocol):
    def evaluate(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PreliminaryTrainingConfig:
    seed: int = 1234
    device: str = "cuda:0"
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    overfit_steps: int = 1_000
    decompressor_steps: int = 100_000
    stepper_steps: int = 100_000
    stepper_window: int = 20
    dt: float = DT

    def __post_init__(self) -> None:
        if self.seed != 1234 or self.device != "cuda:0":
            raise ValueError("preliminary fit seed/device are frozen")
        if (
            self.batch_size != 32
            or self.learning_rate != 1.0e-3
            or self.overfit_steps != 1_000
            or self.decompressor_steps != 100_000
            or self.stepper_steps != 100_000
            or self.stepper_window != 20
            or self.dt != DT
        ):
            raise ValueError("preliminary fit budget is frozen")


@dataclass(frozen=True)
class _ProjectorFreeOrangeDuckConfig:
    seed: int
    device: str
    batch_size: int
    learning_rate: float
    overfit_steps: int
    decompressor_steps: int
    stepper_steps: int
    stepper_window: int
    withheld_frames: int
    withheld_halo: int
    single_clip_overfit_canary: bool
    dt: float


def orange_duck_training_config(config: PreliminaryTrainingConfig):
    if not isinstance(config, PreliminaryTrainingConfig):
        raise TypeError("expected the frozen preliminary training config")
    return _ProjectorFreeOrangeDuckConfig(
        seed=config.seed,
        device=config.device,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        overfit_steps=config.overfit_steps,
        decompressor_steps=config.decompressor_steps,
        stepper_steps=config.stepper_steps,
        stepper_window=config.stepper_window,
        withheld_frames=64,
        withheld_halo=60,
        single_clip_overfit_canary=True,
        dt=config.dt,
    )


def build_preliminary_model_manifest(
    *,
    data_manifest_sha256: str,
    artifacts: Mapping[str, Mapping[str, object]],
    training_receipt: Mapping[str, object],
    numerical_gates_passed: bool,
    visual_v2: bool = False,
) -> dict[str, object]:
    if (
        type(data_manifest_sha256) is not str
        or len(data_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in data_manifest_sha256
        )
    ):
        raise ValueError("data manifest SHA-256 must be lowercase hexadecimal")
    if type(numerical_gates_passed) is not bool or not numerical_gates_passed:
        raise ValueError("a preliminary manifest requires green numerical gates")
    if type(artifacts) is not dict or set(artifacts) != set(MODEL_ARTIFACT_NAMES):
        raise ValueError("preliminary manifest binds exactly three model artifacts")
    if (
        type(training_receipt) is not dict
        or set(training_receipt) != {"path", "size_bytes", "sha256"}
        or training_receipt.get("path") != "training.json"
        or type(training_receipt.get("size_bytes")) is not int
        or training_receipt["size_bytes"] <= 0
        or type(training_receipt.get("sha256")) is not str
        or len(training_receipt["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in training_receipt["sha256"]
        )
    ):
        raise ValueError("preliminary manifest requires one bound training receipt")
    ordered = {name: dict(artifacts[name]) for name in MODEL_ARTIFACT_NAMES}
    return {
        "schema": (
            "g1-lmm-preliminary-slope-visual-model/v2"
            if visual_v2
            else "g1-lmm-preliminary-slope-model/v1"
        ),
        "label": PRELIMINARY_VISUAL_V2_LABEL if visual_v2 else PRELIMINARY_LABEL,
        "status": "preliminary-not-accepted",
        "accepted": False,
        "generalization_claim": "none",
        "model_scope": "slope-only-all-row-overfit",
        "evaluation_scope": "exact-route-all-row-overfit",
        "projector": "absent",
        "rows": 595,
        "output_fps": FPS,
        "data_manifest_sha256": data_manifest_sha256,
        "numerical_gates_passed": True,
        "training_receipt": dict(training_receipt),
        "artifacts": ordered,
    }


def visual_v2_decompressor_gate(gate: Mapping[str, object]) -> dict[str, object]:
    """Copy the V1 decoder evidence under the isolated 2 mm visual policy."""

    if type(gate) is not dict:
        raise TypeError("visual V2 decompressor gate must be a dictionary")
    result = copy.deepcopy(gate)
    fitted = result.get("fitted")
    withheld = result.get("withheld")
    if type(fitted) is not dict or type(withheld) is not list:
        raise ValueError("visual V2 decompressor gate structure changed")

    def bounded(name: str, limit: float) -> bool:
        value = fitted.get(name)
        return (
            type(value) in (int, float)
            and np.isfinite(value)
            and float(value) <= limit
        )

    contacts = fitted.get("contact_f1")
    contacts_green = (
        type(contacts) is list
        and len(contacts) == 2
        and all(
            type(value) in (int, float)
            and np.isfinite(value)
            and float(value) >= 0.95
            for value in contacts
        )
    )
    fitted["accepted"] = bool(
        fitted.get("finite") is True
        and fitted.get("root_velocity_finite") is True
        and bounded("joint_mae_rad", 0.010)
        and bounded("joint_p95_frame_max_rad", 0.050)
        and bounded("joint_max_rad", 0.100)
        and bounded("fk_max_body_position_error_m", 0.010)
        and bounded("sole_max_position_error_m", 0.010)
        and bounded("local_translation_max_error_m", 0.002)
        and contacts_green
    )
    result["local_translation_max_error_limit_m"] = 0.002
    result["accepted"] = bool(fitted["accepted"] and not withheld)
    return result


@dataclass(frozen=True)
class PreliminarySlopeBundle:
    clip_id: str
    fps: float
    training: TrainingBundle
    terrain: object
    reference_qpos: np.ndarray
    terrain_vertices_native: np.ndarray
    terrain_faces: np.ndarray
    native_model: object
    g1_xml_path: Path
    hashes: Mapping[str, str]
    source_receipt: Mapping[str, object]


@dataclass(frozen=True)
class PreliminaryModel:
    root: Path
    latent: np.ndarray
    decompressor: _Network
    stepper: _Network
    manifest: Mapping[str, object]
    training_receipt: Mapping[str, object]


class PreliminaryTrainingError(RuntimeError):
    pass


class LearnedKeys:
    """Thread-safe W/release levels for the learned MuJoCo viewer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._w_down = False
        self._stopped = False

    def press(self, character: str | None = None, *, escape: bool = False) -> None:
        value = character.lower() if isinstance(character, str) else ""
        with self._lock:
            if value == "w":
                self._w_down = True
            if value == "x" or escape:
                self._stopped = True

    def release(self, character: str | None = None) -> None:
        value = character.lower() if isinstance(character, str) else ""
        with self._lock:
            if value == "w":
                self._w_down = False

    def snapshot(self) -> tuple[bool, bool]:
        with self._lock:
            return self._w_down, self._stopped


def _frozen(values: object, dtype: np.dtype | str) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _admitted_clip(candidate: object) -> HoldenClip:
    source = candidate.source
    start = 3
    result = HoldenClip(
        source.name,
        np.array(source.positions[start:], np.float32, copy=True),
        np.array(source.velocities[start:], np.float32, copy=True),
        np.array(source.rotations[start:], np.float32, copy=True),
        np.array(source.angular_velocities[start:], np.float32, copy=True),
        np.array(source.contacts[start:], np.uint8, copy=True),
        np.array(source.terrain_features[start:], np.float32, copy=True),
        np.array(source.terrain_support[start:], np.float32, copy=True),
        np.array(source.source_frames[start:], copy=True),
        source.terrain_id,
        np.array(source.source_left_indices[start:], np.int32, copy=True),
        np.array(source.source_right_indices[start:], np.int32, copy=True),
        np.array(source.source_alpha[start:], np.float32, copy=True),
    )
    result.validate()
    if len(result.positions) != 595:
        raise ValueError("Task1 authored slope admission changed from 595 rows")
    return result


def _reference_qpos(robot_path: Path) -> np.ndarray:
    source = load_authenticated_grail_slope(str(robot_path))
    if source.fps != 25.0 or source.qpos.shape != (250, 36):
        raise ValueError("authenticated authored slope source dimensions changed")
    provisional = np.empty((598, 36), dtype=np.float32)
    provisional[:, :3] = resample_vectors(source.qpos[:, :3], 25.0, FPS)
    provisional[:, 3:7] = resample_quaternions_wxyz(source.qpos[:, 3:7], 25.0, FPS)
    provisional[:, 7:] = resample_vectors(source.qpos[:, 7:], 25.0, FPS)
    return _frozen(provisional[3:], np.float32)


def _holden_vertices_to_native(vertices: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def _manifest_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_preliminary_slope_bundle(
    *,
    slope_robot: Path = DEFAULT_SLOPE_ROBOT,
    slope_usd: Path = DEFAULT_SLOPE_USD,
    slope_recon: Path = DEFAULT_SLOPE_RECON,
    slope_metadata: Path = DEFAULT_SLOPE_METADATA,
    g1_xml: Path = DEFAULT_G1_XML,
) -> PreliminarySlopeBundle:
    """Build the slope-only all-row training table from the exact Task1 source."""

    robot, usd, recon, metadata, model_path = tuple(
        Path(value).expanduser().resolve()
        for value in (slope_robot, slope_usd, slope_recon, slope_metadata, g1_xml)
    )
    candidate = _assemble_authored_slope_source(
        SimpleNamespace(
            output_fps=FPS,
            slope_robot=str(robot),
            slope_usd=str(usd),
            slope_recon=str(recon),
            slope_metadata=str(metadata),
            g1_xml=str(model_path),
        )
    )
    receipt = candidate.provenance
    if (
        receipt.get("clip_id") != CLIP_ID
        or receipt.get("admitted_output_frames") != 595
        or receipt.get("rejected_output_ranges") != [[0, 3]]
        or receipt.get("target_fps") != FPS
    ):
        raise ValueError("Task1 authored slope receipt changed")

    clip = _admitted_clip(candidate)
    skeleton = SkeletonSpec(
        G1_SKELETON_NAMES,
        np.asarray(G1_SKELETON_PARENTS, dtype=np.int32),
    )
    artifacts = combine_clips([clip], skeleton)
    features = build_matching_features(artifacts, FPS, (20, 40, 60))
    if features.values.shape != (595, 31):
        raise ValueError("preliminary learned feature table changed")
    terrain_scale = features.scale[27:31]
    if (
        not np.isfinite(terrain_scale).all()
        or np.any(terrain_scale <= 0.0)
        or np.any(terrain_scale >= np.finfo(np.float32).max)
    ):
        raise ValueError("all four preliminary terrain dimensions must stay active")
    raw_terrain = features.values[:, 27:31] * terrain_scale + features.offset[27:31]
    if not np.all(np.any(raw_terrain[165:539] != 0.0, axis=1)):
        raise ValueError("Task1 intended nonflat exposure changed")

    manifest = {
        "schema": "g1-lmm-preliminary-slope-data/v1",
        "label": PRELIMINARY_LABEL,
        "status": "preliminary-not-accepted",
        "generalization_claim": "none",
        "clip_id": CLIP_ID,
        "output_fps": FPS,
        "rows": 595,
        "ranges": [[0, 595]],
        "source_hashes": dict(receipt["hashes"]),
        "terrain_feature_indices": [27, 28, 29, 30],
        "projector": "absent",
    }
    admitted_mask = np.ones(595, dtype=bool)
    admitted_mask.setflags(write=False)
    training = TrainingBundle(
        root=model_path.parent,
        manifest=manifest,
        manifest_sha256=_manifest_digest(manifest),
        dimensions=G1LmmDimensions(),
        positions=artifacts.positions,
        rotations=artifacts.rotations,
        velocities=artifacts.velocities,
        angular_velocities=artifacts.angular_velocities,
        parents=artifacts.parents,
        range_starts=artifacts.range_starts,
        range_stops=artifacts.range_stops,
        contacts=artifacts.contacts,
        features=features.values,
        feature_offset=features.offset,
        feature_scale=features.scale,
        admitted_mask=admitted_mask,
    )
    reference_qpos = _reference_qpos(robot)
    reference_digest = hashlib.sha256(
        reference_qpos.astype("<f4").tobytes()
    ).hexdigest()
    if reference_digest != REFERENCE_QPOS_SHA256:
        raise ValueError(
            "authored slope resampling bytes changed; use the pinned NumPy 2.x "
            f"source environment ({reference_digest})"
        )
    native_model = G1Kinematics(str(model_path)).model
    return PreliminarySlopeBundle(
        clip_id=CLIP_ID,
        fps=FPS,
        training=training,
        terrain=candidate.terrain,
        reference_qpos=reference_qpos,
        terrain_vertices_native=_frozen(
            _holden_vertices_to_native(candidate.terrain.vertices), np.float64
        ),
        terrain_faces=_frozen(candidate.terrain.triangles, np.int32),
        native_model=native_model,
        g1_xml_path=model_path,
        hashes=MappingProxyType(dict(receipt["hashes"])),
        source_receipt=MappingProxyType(dict(receipt)),
    )


def _bound_artifact(root: Path, manifest: dict, name: str) -> Path:
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != set(MODEL_ARTIFACT_NAMES):
        raise ValueError(
            "preliminary manifest must bind only the three model artifacts"
        )
    descriptor = artifacts.get(name)
    if type(descriptor) is not dict or descriptor.get("path") != name:
        raise ValueError(f"preliminary manifest path for {name} is invalid")
    path = root / name
    payload = path.read_bytes()
    if descriptor.get("size_bytes") != len(payload):
        raise ValueError(f"preliminary {name} size changed")
    if descriptor.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"preliminary {name} SHA-256 changed")
    return path


def _load_latent(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ValueError("preliminary latent.bin is truncated")
    rows, columns = struct.unpack_from("<II", payload)
    if (rows, columns) != (595, 32) or len(payload) != 8 + rows * columns * 4:
        raise ValueError("preliminary latent.bin dimensions changed")
    latent = np.frombuffer(payload, dtype="<f4", offset=8).reshape(rows, columns).copy()
    if not np.isfinite(latent).all():
        raise ValueError("preliminary latent.bin is non-finite")
    latent.setflags(write=False)
    return latent


def _load_bound_training_receipt(
    root: Path,
    manifest: dict[str, object],
    bundle: PreliminarySlopeBundle,
    *,
    visual_v2: bool = False,
) -> dict[str, object]:
    descriptor = manifest.get("training_receipt")
    if (
        type(descriptor) is not dict
        or set(descriptor) != {"path", "size_bytes", "sha256"}
        or descriptor.get("path") != "training.json"
    ):
        raise ValueError("preliminary training receipt descriptor is invalid")
    try:
        payload = (root / "training.json").read_bytes()
    except OSError as error:
        raise ValueError("preliminary training receipt is missing") from error
    if descriptor.get("size_bytes") != len(payload):
        raise ValueError("preliminary training receipt size changed")
    if descriptor.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("preliminary training receipt SHA-256 changed")
    try:
        receipt = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("preliminary training receipt is invalid JSON") from error

    required = {
        "schema": (
            "g1-lmm-preliminary-slope-visual-training/v2"
            if visual_v2
            else "g1-lmm-preliminary-slope-training/v1"
        ),
        "label": PRELIMINARY_VISUAL_V2_LABEL if visual_v2 else PRELIMINARY_LABEL,
        "status": "preliminary-not-accepted",
        "accepted": False,
        "generalization_claim": "none",
        "evaluation_scope": "exact-route-all-row-overfit",
        "data_manifest_sha256": bundle.training.manifest_sha256,
        "numerical_gates_passed": True,
        "stopped_after": "full-route-rollout",
    }
    if type(receipt) is not dict or any(
        type(receipt.get(name)) is not type(expected) or receipt.get(name) != expected
        for name, expected in required.items()
    ):
        raise ValueError("preliminary training receipt scope or gate changed")
    if receipt.get("source_hashes") != dict(bundle.hashes):
        raise ValueError("preliminary training receipt source hashes changed")

    expected_config = {
        "seed": 1234,
        "stage_seeds": {
            "overfit": 1234,
            "decompressor": 1235,
            "stepper": 1236,
        },
        "device": "cuda:0",
        "batch_size": 32,
        "learning_rate": 1.0e-3,
        "overfit_steps": 1_000,
        "decompressor_steps": 100_000,
        "stepper_steps": 100_000,
        "stepper_window": 20,
        "dt": DT,
        "projector": "absent",
    }
    expected_architecture = {
        "compressor": [908, 512, 512, 512, 32],
        "decompressor": [63, 512, 458],
        "stepper": [63, 512, 512, 63],
        "projector": None,
    }
    if receipt.get("config") != expected_config:
        raise ValueError("preliminary training receipt config changed")
    if receipt.get("architecture") != expected_architecture:
        raise ValueError("preliminary training receipt architecture changed")

    hardware = receipt.get("hardware")
    expected_hardware = {
        "physical_index": 3,
        "logical_device": "cuda:0",
        "uuid": GPU3_UUID,
        "name": "NVIDIA L40S",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }
    if type(hardware) is not dict or any(
        type(hardware.get(name)) is not type(expected) or hardware.get(name) != expected
        for name, expected in expected_hardware.items()
    ):
        raise ValueError("preliminary training receipt GPU3 identity changed")

    claim = receipt.get("sole_fit_claim")
    expected_claim = VISUAL_V2_SOLE_FIT_CLAIM if visual_v2 else SOLE_FIT_CLAIM
    if (
        type(claim) is not dict
        or claim.get("path") != str(expected_claim)
        or type(claim.get("size_bytes")) is not int
        or claim["size_bytes"] <= 0
        or type(claim.get("sha256")) is not str
        or len(claim["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in claim["sha256"])
    ):
        raise ValueError("preliminary training receipt sole-fit claim changed")
    if receipt.get("artifacts") != manifest.get("artifacts"):
        raise ValueError("preliminary training receipt artifact binding changed")

    for stage in ("overfit_gate", "decompressor_gate", "stepper_gate"):
        gate = receipt.get(stage)
        if type(gate) is not dict or gate.get("accepted") is not True:
            raise ValueError(f"preliminary training receipt {stage} is not green")
    if visual_v2 and receipt["decompressor_gate"].get(
        "local_translation_max_error_limit_m"
    ) != 0.002:
        raise ValueError("preliminary training receipt visual V2 gate changed")
    rollout = receipt.get("full_route_rollout_gate")
    if type(rollout) is not dict or rollout.get("accepted") is not True:
        raise ValueError("preliminary training receipt full-route gate is not green")
    numeric = (
        rollout.get("joint_mae_rad"),
        rollout.get("maximum_planar_root_error_m"),
    )
    contacts = rollout.get("contact_f1")
    if (
        rollout.get("evaluated_pose_rows") != 595
        or rollout.get("projector_evaluations") != 0
        or any(
            type(value) not in (int, float) or not np.isfinite(value)
            for value in numeric
        )
        or numeric[0] > 0.05
        or numeric[1] > 0.10
        or type(contacts) is not list
        or len(contacts) != 2
        or any(
            type(value) not in (int, float) or not np.isfinite(value) or value < 0.90
            for value in contacts
        )
    ):
        raise ValueError("preliminary training receipt full-route metrics changed")
    return receipt


def load_preliminary_model(
    model_directory: str | Path,
    bundle: PreliminarySlopeBundle,
    *,
    visual_v2: bool = False,
) -> PreliminaryModel:
    """Authenticate the isolated three-artifact model; reject any projector."""

    if not isinstance(bundle, PreliminarySlopeBundle):
        raise TypeError("model load requires the exact preliminary slope bundle")
    root = Path(model_directory).expanduser().resolve()
    if (root / "projector.bin").exists():
        raise ValueError("preliminary learned slope forbids every projector artifact")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid preliminary model manifest") from error
    required = {
        "schema": (
            "g1-lmm-preliminary-slope-visual-model/v2"
            if visual_v2
            else "g1-lmm-preliminary-slope-model/v1"
        ),
        "label": PRELIMINARY_VISUAL_V2_LABEL if visual_v2 else PRELIMINARY_LABEL,
        "status": "preliminary-not-accepted",
        "accepted": False,
        "generalization_claim": "none",
        "projector": "absent",
        "rows": 595,
        "output_fps": FPS,
        "data_manifest_sha256": bundle.training.manifest_sha256,
        "numerical_gates_passed": True,
    }
    if type(manifest) is not dict or any(
        type(manifest.get(name)) is not type(expected) or manifest.get(name) != expected
        for name, expected in required.items()
    ):
        raise ValueError("preliminary model manifest scope or gate changed")
    training_receipt = _load_bound_training_receipt(
        root, manifest, bundle, visual_v2=visual_v2
    )

    paths = {
        name: _bound_artifact(root, manifest, name) for name in MODEL_ARTIFACT_NAMES
    }
    from resources.g1_lmm.training import load_exported_network

    decompressor = load_exported_network(
        paths["decompressor.bin"], expected_layers=[(63, 512), (512, 458)]
    )
    stepper = load_exported_network(
        paths["stepper.bin"],
        expected_layers=[(63, 512), (512, 512), (512, 63)],
    )
    manifest["artifacts"] = {
        name: manifest["artifacts"][name] for name in MODEL_ARTIFACT_NAMES
    }
    return PreliminaryModel(
        root=root,
        latent=_load_latent(paths["latent.bin"]),
        decompressor=decompressor,
        stepper=stepper,
        manifest=MappingProxyType(manifest),
        training_receipt=MappingProxyType(training_receipt),
    )


def _holden_to_native_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("Holden vector must have shape (3,)")
    return np.asarray((vector[0], -vector[2], vector[1]), dtype=np.float64)


def _holden_to_native_quaternions(values: np.ndarray) -> np.ndarray:
    rotations = np.asarray(values, dtype=np.float64)
    if rotations.shape[-1:] != (4,):
        raise ValueError("Holden quaternions must end in four components")
    basis = np.broadcast_to(Q_ZUP_TO_YUP, rotations.shape)
    inverse = np.broadcast_to(quat.inv(Q_ZUP_TO_YUP), rotations.shape)
    return quat.normalize(quat.mul(quat.mul(inverse, rotations), basis))


def decoded_pose_to_native_qpos(
    decoded: np.ndarray,
    simulation_position: np.ndarray,
    simulation_rotation: np.ndarray,
    native_model: object,
) -> np.ndarray:
    """Convert one decoded 458-D local pose to the canonical native G1 ABI."""

    output = np.asarray(decoded, dtype=np.float32)
    sim_position = np.asarray(simulation_position, dtype=np.float32)
    sim_rotation = np.asarray(simulation_rotation, dtype=np.float32)
    if output.shape != (458,):
        raise ValueError("decoded G1 pose must have 458 components")
    if sim_position.shape != (3,) or sim_rotation.shape != (4,):
        raise ValueError("Simulation transform has invalid dimensions")
    if (
        not np.isfinite(output).all()
        or not np.isfinite(sim_position).all()
        or not np.isfinite(sim_rotation).all()
    ):
        raise ValueError("decoded G1 pose must be finite")
    if (
        int(native_model.nq) != 36
        or int(native_model.nbody) != 31
        or int(native_model.njnt) != 30
    ):
        raise ValueError("native model is not the canonical 36-qpos G1")

    non_root = 30
    position_end = 3 * non_root
    rotation_end = position_end + 6 * non_root
    local_position = output[:position_end].reshape(non_root, 3)
    local_xy = output[position_end:rotation_end].reshape(non_root, 3, 2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        local_rotation = quat.from_xform_xy(local_xy).astype(np.float64)
    if not np.isfinite(local_rotation).all():
        raise ValueError("decoded G1 rotations are degenerate")

    hips_position = sim_position + quat.mul_vec(sim_rotation, local_position[0])
    hips_rotation = quat.normalize(quat.mul(sim_rotation, local_rotation[0]))
    native_local = _holden_to_native_quaternions(local_rotation)
    qpos = np.zeros(36, dtype=np.float64)
    qpos[:3] = _holden_to_native_vector(hips_position)
    qpos[3:7] = _holden_to_native_quaternions(hips_rotation)

    for joint_id in range(1, int(native_model.njnt)):
        body_id = int(native_model.jnt_bodyid[joint_id])
        if not 2 <= body_id < int(native_model.nbody):
            raise ValueError("canonical hinge is attached to an unexpected body")
        fixed = np.asarray(native_model.body_quat[body_id], dtype=np.float64)
        delta = quat.mul(quat.inv(fixed), native_local[body_id - 1])
        angle_axis = quat.to_scaled_angle_axis(quat.abs(delta))
        angle = float(np.dot(angle_axis, native_model.jnt_axis[joint_id]))
        qpos[int(native_model.jnt_qposadr[joint_id])] = angle
    if not np.isfinite(qpos).all():
        raise ValueError("decoded native qpos is non-finite")
    return qpos


def normalized_terrain_features(
    bundle: PreliminarySlopeBundle,
    row: int,
) -> np.ndarray:
    """Recompute one fitted terrain row from its exact authored route ahead."""

    if not isinstance(bundle, PreliminarySlopeBundle):
        raise TypeError("terrain resample requires the exact preliminary bundle")
    if type(row) is not int or not 0 <= row < bundle.training.frames:
        raise ValueError("terrain resample row is outside the exact route")
    offset = np.asarray(bundle.training.feature_offset, dtype=np.float32)
    scale = np.asarray(bundle.training.feature_scale, dtype=np.float32)
    if offset.shape != (31,) or scale.shape != (31,):
        raise ValueError("terrain resample requires the exact 31-D normalization")
    terrain_scale = scale[27:31]
    if not np.isfinite(terrain_scale).all() or np.any(terrain_scale <= 0.0):
        raise ValueError("terrain normalization is inactive")

    stop = min(row + round(2.0 * FPS) + 1, bundle.training.frames)
    path = np.asarray(bundle.training.positions[row:stop, 0], np.float64)[:, [0, 2]]
    headings = quat.mul_vec(
        bundle.training.rotations[row:stop, 0],
        np.asarray((0.0, 0.0, 1.0), np.float64),
    )[:, [0, 2]]
    centerline = build_facing_centerline(path[0], headings, path)
    raw = sample_terrain_features(bundle.terrain, centerline)
    normalized = (raw - offset[27:31]) / terrain_scale
    if not np.isfinite(normalized).all():
        raise ValueError("terrain resample produced non-finite normalized values")
    return normalized.astype(np.float32)


def build_viewer_model(bundle: PreliminarySlopeBundle):
    """Compose the canonical native G1 with the authenticated exact ramp."""

    if not isinstance(bundle, PreliminarySlopeBundle):
        raise TypeError("viewer model requires the exact preliminary bundle")
    import mujoco

    spec = mujoco.MjSpec.from_file(str(bundle.g1_xml_path))
    spec.add_mesh(
        name="preliminary_learned_slope_exact_mesh",
        uservert=bundle.terrain_vertices_native.ravel(),
        userface=bundle.terrain_faces.ravel(),
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    spec.worldbody.add_geom(
        name="preliminary_learned_slope_exterior_flat",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, float(bundle.terrain.exterior_height)),
        size=(10.0, 10.0, 0.05),
        contype=0,
        conaffinity=0,
        rgba=(0.16, 0.24, 0.18, 1.0),
    )
    spec.worldbody.add_geom(
        name="preliminary_learned_slope_exact_mesh",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="preliminary_learned_slope_exact_mesh",
        contype=0,
        conaffinity=0,
        rgba=(0.42, 0.62, 0.20, 1.0),
    )
    model = spec.compile()
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("canonical G1 model lost its original floor geom")
    model.geom_rgba[floor_id, 3] = 0.0
    model.geom_contype[floor_id] = 0
    model.geom_conaffinity[floor_id] = 0
    return model


def overlay_text(
    *,
    row: int,
    frame_count: int,
    w_down: bool,
    label: str = PRELIMINARY_LABEL,
) -> tuple[str, str]:
    if type(row) is not int or type(frame_count) is not int:
        raise TypeError("overlay row and frame count must be integers")
    if frame_count < 1 or not 0 <= row < frame_count:
        raise ValueError("overlay row is outside the learned route")
    if type(w_down) is not bool:
        raise TypeError("overlay W state must be an exact boolean")
    if label not in (PRELIMINARY_LABEL, PRELIMINARY_VISUAL_V2_LABEL):
        raise ValueError("overlay label is outside the preliminary scope")
    mode = (
        "ROUTE COMPLETE"
        if row == frame_count - 1
        else "ADVANCING LEARNED RECURRENCE (W held)"
        if w_down
        else "BITWISE PAUSED (W released)"
    )
    return (
        label,
        (
            f"row {row + 1}/{frame_count} @ 60 Hz | {mode}\n"
            "W: one LEARNED recurrent step/tick | release: exact pause | X: exit\n"
            "authoritative exact-mesh terrain overwrite | NO PROJECTOR"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="run the one frozen GPU3 fit")
    train.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser(
        "smoke", help="run the complete learned route without a window"
    )
    smoke.add_argument("--model", type=Path, required=True)
    view = commands.add_parser("view", help="open the interactive MuJoCo viewer")
    view.add_argument("--model", type=Path, required=True)
    train_v2 = commands.add_parser(
        "train-v2", help="run the one frozen GPU3 post-hoc visual fit"
    )
    train_v2.add_argument("--output", type=Path, required=True)
    smoke_v2 = commands.add_parser(
        "smoke-v2", help="run the post-hoc learned route without a window"
    )
    smoke_v2.add_argument("--model", type=Path, required=True)
    view_v2 = commands.add_parser(
        "view-v2", help="open the post-hoc interactive MuJoCo viewer"
    )
    view_v2.add_argument("--model", type=Path, required=True)
    return parser


def _integrate_simulation(
    position: np.ndarray, rotation: np.ndarray, decoded: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(decoded, dtype=np.float32)
    root_offset = 15 * 30
    local_velocity = output[root_offset : root_offset + 3]
    local_angular = output[root_offset + 3 : root_offset + 6]
    next_position = np.asarray(position, np.float32) + quat.mul_vec(
        rotation, local_velocity
    ) * np.float32(DT)
    world_angular = quat.mul_vec(rotation, local_angular) * np.float32(DT)
    next_rotation = quat.normalize(
        quat.mul(rotation, quat.from_scaled_angle_axis(world_angular))
    ).astype(np.float32)
    if not np.isfinite(next_position).all() or not np.isfinite(next_rotation).all():
        raise ValueError("learned root integration became non-finite")
    return next_position.astype(np.float32), next_rotation


class LearnedSlopeState:
    """Transactional projector-free recurrence with exact W/release semantics."""

    def __init__(
        self,
        bundle: PreliminarySlopeBundle,
        *,
        latent: np.ndarray,
        decompressor: _Network,
        stepper: _Network,
    ) -> None:
        if not isinstance(bundle, PreliminarySlopeBundle):
            raise TypeError("learned slope state requires its exact source bundle")
        latent_rows = np.asarray(latent, dtype=np.float32)
        if latent_rows.shape != (595, 32) or not np.isfinite(latent_rows).all():
            raise ValueError("learned slope latent table must be finite (595, 32)")
        self.bundle = bundle
        self.decompressor = decompressor
        self.stepper = stepper
        self.row = 0
        self.terrain_overwrite_count = 0
        self.recurrent_state = np.concatenate(
            (bundle.training.features[0], latent_rows[0])
        ).astype(np.float32)
        self.simulation_position = np.array(
            bundle.training.positions[0, 0], np.float32, copy=True
        )
        self.simulation_rotation = np.array(
            bundle.training.rotations[0, 0], np.float32, copy=True
        )
        self.decoded = self._evaluate_decompressor(self.recurrent_state)
        self.qpos = decoded_pose_to_native_qpos(
            self.decoded,
            self.simulation_position,
            self.simulation_rotation,
            bundle.native_model,
        )
        self._require_joint_limits(self.qpos)

    def _evaluate_decompressor(self, state: np.ndarray) -> np.ndarray:
        result = np.asarray(
            self.decompressor.evaluate(state[None])[0], dtype=np.float32
        )
        if result.shape != (458,) or not np.isfinite(result).all():
            raise ValueError("learned decompressor produced an invalid pose")
        return result

    def _require_joint_limits(self, qpos: np.ndarray) -> None:
        model = self.bundle.native_model
        for joint_id in range(1, int(model.njnt)):
            address = int(model.jnt_qposadr[joint_id])
            value = float(qpos[address])
            lower, upper = np.asarray(model.jnt_range[joint_id], np.float64)
            if value < lower - 1e-5 or value > upper + 1e-5:
                raise ValueError(
                    f"learned decoded joint {joint_id} left its native limit"
                )

    def fingerprint(self) -> str:
        payload = b"".join(
            (
                np.asarray((self.row, self.terrain_overwrite_count), "<i8").tobytes(),
                np.asarray(self.recurrent_state, "<f4").tobytes(),
                np.asarray(self.simulation_position, "<f4").tobytes(),
                np.asarray(self.simulation_rotation, "<f4").tobytes(),
                np.asarray(self.decoded, "<f4").tobytes(),
                np.asarray(self.qpos, "<f8").tobytes(),
            )
        )
        return hashlib.sha256(payload).hexdigest()

    def tick(self, *, w_down: bool) -> int:
        if type(w_down) is not bool:
            raise TypeError("w_down must be an exact boolean key level")
        if not w_down or self.row == self.bundle.training.frames - 1:
            return self.row

        derivative = np.asarray(
            self.stepper.evaluate(self.recurrent_state[None])[0],
            dtype=np.float32,
        )
        if derivative.shape != (63,) or not np.isfinite(derivative).all():
            raise ValueError("learned stepper produced an invalid derivative")
        proposed_state = (self.recurrent_state + np.float32(DT) * derivative).astype(
            np.float32
        )
        proposed_position, proposed_rotation = _integrate_simulation(
            self.simulation_position, self.simulation_rotation, self.decoded
        )
        proposed_state[27:31] = normalized_terrain_features(self.bundle, self.row + 1)
        proposed_decoded = self._evaluate_decompressor(proposed_state)
        proposed_qpos = decoded_pose_to_native_qpos(
            proposed_decoded,
            proposed_position,
            proposed_rotation,
            self.bundle.native_model,
        )
        self._require_joint_limits(proposed_qpos)

        self.recurrent_state = proposed_state
        self.simulation_position = proposed_position
        self.simulation_rotation = proposed_rotation
        self.decoded = proposed_decoded
        self.qpos = proposed_qpos
        self.row += 1
        self.terrain_overwrite_count += 1
        return self.row


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _json_safe_diagnostic(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe_diagnostic(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe_diagnostic(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_diagnostic(item) for item in value]
    return str(value)


def _acquire_atomic_fit_claim(
    claim: Path, output: Path, *, visual_v2: bool = False
) -> dict[str, object]:
    claim_path = Path(claim).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    payload = _json_bytes(
        {
            "schema": (
                "g1-lmm-preliminary-slope-visual-sole-fit-claim/v2"
                if visual_v2
                else "g1-lmm-preliminary-slope-sole-fit-claim/v1"
            ),
            "label": PRELIMINARY_VISUAL_V2_LABEL if visual_v2 else PRELIMINARY_LABEL,
            "output": str(target),
        }
    )
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"sole preliminary fit was already claimed: {claim_path}"
        ) from error
    # A partial claim remains intentionally fail-closed if writing or syncing
    # fails: no later fit may reinterpret that failure as another permission.
    with os.fdopen(descriptor, "wb") as output_stream:
        output_stream.write(payload)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    return {
        "path": str(claim_path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _artifact_descriptor(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _gpu3_receipt(config: PreliminaryTrainingConfig) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise ValueError("the sole preliminary fit requires CUDA_VISIBLE_DEVICES=3")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError(
            "the sole preliminary fit requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    import torch

    from resources.g1_lmm.training import _configure_determinism

    _configure_determinism(config.seed, torch.device(config.device))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("GPU3 must be the only CUDA device exposed to the fit")
    command = (
        "nvidia-smi",
        "--id=3",
        "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        fields = tuple(part.strip() for part in completed.stdout.strip().split(","))
        uuid, name, memory_used, memory_total, utilization = fields
        memory_used_mib = int(memory_used)
        memory_total_mib = int(memory_total)
        utilization_percent = int(utilization)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as error:
        raise ValueError("cannot authenticate physical GPU3 with nvidia-smi") from error
    if uuid != GPU3_UUID or name != "NVIDIA L40S":
        raise ValueError("physical GPU3 identity changed")
    if memory_used_mib > 512 or utilization_percent > 5:
        raise ValueError(
            "physical GPU3 is not idle enough for the sole immutable fit: "
            f"memory={memory_used_mib} MiB utilization={utilization_percent}%"
        )
    return {
        "physical_index": 3,
        "logical_device": config.device,
        "uuid": uuid,
        "name": name,
        "memory_used_mib_before_fit": memory_used_mib,
        "memory_total_mib": memory_total_mib,
        "utilization_percent_before_fit": utilization_percent,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _quaternion_step(left: np.ndarray, right: np.ndarray) -> float:
    dot = float(abs(np.dot(left, right)))
    return 2.0 * float(np.arccos(np.clip(dot, 0.0, 1.0)))


def _full_route_rollout(
    bundle: PreliminarySlopeBundle,
    model: PreliminaryModel,
) -> dict[str, object]:
    state = LearnedSlopeState(
        bundle,
        latent=model.latent,
        decompressor=model.decompressor,
        stepper=model.stepper,
    )
    held_before = state.fingerprint()
    state.tick(w_down=False)
    release_pause_bitwise = state.fingerprint() == held_before
    maximum_joint_step = 0.0
    maximum_root_step = 0.0
    maximum_root_angle_step = 0.0
    predicted_qpos = [state.qpos.copy()]
    predicted_decoded = [state.decoded.copy()]
    previous = state.qpos.copy()
    for expected_row in range(1, bundle.training.frames):
        actual_row = state.tick(w_down=True)
        if actual_row != expected_row:
            raise ValueError("learned route cursor skipped an authored tick")
        current = state.qpos
        maximum_joint_step = max(
            maximum_joint_step,
            float(np.max(np.abs(current[7:] - previous[7:]))),
        )
        maximum_root_step = max(
            maximum_root_step,
            float(np.linalg.norm(current[:3] - previous[:3])),
        )
        maximum_root_angle_step = max(
            maximum_root_angle_step,
            _quaternion_step(current[3:7], previous[3:7]),
        )
        predicted_qpos.append(current.copy())
        predicted_decoded.append(state.decoded.copy())
        previous = current.copy()
    terminal = state.fingerprint()
    for _ in range(4):
        state.tick(w_down=True)
    terminal_hold_bitwise = state.fingerprint() == terminal
    all_qpos = np.asarray(predicted_qpos, dtype=np.float64)
    all_decoded = np.asarray(predicted_decoded, dtype=np.float32)
    if all_qpos.shape != bundle.reference_qpos.shape or all_decoded.shape != (595, 458):
        raise ValueError("full learned route did not produce all 595 decoded poses")
    all_joint_errors = (all_qpos[:, 7:] - bundle.reference_qpos[:, 7:] + np.pi) % (
        2.0 * np.pi
    ) - np.pi
    planar_root_errors = np.linalg.norm(
        all_qpos[:, :2] - bundle.reference_qpos[:, :2], axis=1
    )
    non_root = bundle.training.dimensions.bones - 1
    contact_offset = 15 * non_root + 6
    predicted_contacts = all_decoded[:, contact_offset:] > np.float32(0.5)
    expected_contacts = np.asarray(bundle.training.contacts, dtype=bool)
    if predicted_contacts.shape != expected_contacts.shape:
        raise ValueError("full learned route produced invalid decoded contacts")
    contact_f1 = []
    for column in range(expected_contacts.shape[1]):
        truth = expected_contacts[:, column]
        estimate = predicted_contacts[:, column]
        true_positive = int(np.count_nonzero(truth & estimate))
        false_positive = int(np.count_nonzero(~truth & estimate))
        false_negative = int(np.count_nonzero(truth & ~estimate))
        denominator = 2 * true_positive + false_positive + false_negative
        contact_f1.append(
            1.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
    maximum_planar_root_error = float(np.max(planar_root_errors))
    joint_mae = float(np.mean(np.abs(all_joint_errors)))
    finite = bool(
        np.isfinite(state.recurrent_state).all()
        and np.isfinite(state.qpos).all()
        and np.isfinite(all_qpos).all()
        and np.isfinite(all_decoded).all()
        and np.isfinite(all_joint_errors).all()
        and np.isfinite(planar_root_errors).all()
    )
    accepted = bool(
        finite
        and release_pause_bitwise
        and terminal_hold_bitwise
        and state.row == 594
        and state.terrain_overwrite_count == 594
        and maximum_joint_step <= 0.25
        and maximum_root_step <= 0.05
        and maximum_root_angle_step <= 0.20
        and joint_mae <= 0.05
        and maximum_planar_root_error <= 0.10
        and min(contact_f1) >= 0.90
    )
    return {
        "accepted": accepted,
        "finite": finite,
        "release_pause_bitwise": release_pause_bitwise,
        "terminal_hold_bitwise": terminal_hold_bitwise,
        "terminal_row": state.row,
        "learned_stepper_evaluations": 594,
        "learned_decompressor_evaluations": 595,
        "terrain_overwrites": state.terrain_overwrite_count,
        "projector_evaluations": 0,
        "evaluated_pose_rows": len(all_qpos),
        "maximum_joint_step_rad": maximum_joint_step,
        "maximum_root_step_m": maximum_root_step,
        "maximum_root_angle_step_rad": maximum_root_angle_step,
        "maximum_planar_root_error_m": maximum_planar_root_error,
        "joint_mae_rad": joint_mae,
        "joint_max_rad_observed": float(np.max(np.abs(all_joint_errors))),
        "contact_f1": contact_f1,
        "accuracy_gate_limits": {
            "maximum_planar_root_error_m": 0.10,
            "joint_mae_rad": 0.05,
            "minimum_contact_f1": 0.90,
        },
    }


def _publish_rejected_fit(
    staging: Path,
    output: Path,
    receipt: dict[str, object],
    *,
    stopped_after: str,
    error: BaseException,
) -> None:
    for name in (*MODEL_ARTIFACT_NAMES, "manifest.json"):
        path = staging / name
        if path.exists():
            path.unlink()
    receipt.update(
        {
            "status": "rejected",
            "stopped_after": stopped_after,
            "numerical_gates_passed": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    )
    (staging / "training.json").write_bytes(_json_bytes(_json_safe_diagnostic(receipt)))
    os.replace(staging, output)


def train_preliminary_model(
    output_directory: str | Path,
    *,
    bundle: PreliminarySlopeBundle | None = None,
    config: PreliminaryTrainingConfig | None = None,
    visual_v2: bool = False,
) -> dict[str, object]:
    """Run the one immutable all-row slope-only fit; never train a projector."""

    output = Path(output_directory).expanduser().resolve()
    config = PreliminaryTrainingConfig() if config is None else config
    if not isinstance(config, PreliminaryTrainingConfig):
        raise TypeError("training requires the frozen preliminary config")
    expected_output = (
        DEFAULT_VISUAL_V2_MODEL_OUTPUT if visual_v2 else DEFAULT_MODEL_OUTPUT
    )
    expected_claim = VISUAL_V2_SOLE_FIT_CLAIM if visual_v2 else SOLE_FIT_CLAIM
    label = PRELIMINARY_VISUAL_V2_LABEL if visual_v2 else PRELIMINARY_LABEL
    if output != expected_output:
        raise ValueError(
            f"sole preliminary fit output is fixed at {expected_output}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable fit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = load_preliminary_slope_bundle() if bundle is None else bundle
    if not isinstance(source, PreliminarySlopeBundle):
        raise TypeError("training requires the exact preliminary source bundle")
    if np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion("2.0.0"):
        raise ValueError("the exact Task1 resampling contract requires NumPy 2.x")

    from resources.g1_lmm.dataset import build_training_arrays
    from resources.g1_lmm.training import (
        _train_64_frame_overfit,
        _train_decompressor_stage,
        _train_stepper_stage,
        load_exported_network,
    )

    hardware = _gpu3_receipt(config)
    arrays = build_training_arrays(source.training)
    fit_config = orange_duck_training_config(config)
    sole_fit_claim = _acquire_atomic_fit_claim(
        expected_claim, output, visual_v2=visual_v2
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    receipt: dict[str, object] = {
        "schema": (
            "g1-lmm-preliminary-slope-visual-training/v2"
            if visual_v2
            else "g1-lmm-preliminary-slope-training/v1"
        ),
        "label": label,
        "status": "fitting",
        "accepted": False,
        "generalization_claim": "none",
        "evaluation_scope": "exact-route-all-row-overfit",
        "data_manifest_sha256": source.training.manifest_sha256,
        "source_hashes": dict(source.hashes),
        "sole_fit_claim": sole_fit_claim,
        "hardware": hardware,
        "config": {
            "seed": config.seed,
            "stage_seeds": {
                "overfit": config.seed,
                "decompressor": config.seed + 1,
                "stepper": config.seed + 2,
            },
            "device": config.device,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "overfit_steps": config.overfit_steps,
            "decompressor_steps": config.decompressor_steps,
            "stepper_steps": config.stepper_steps,
            "stepper_window": config.stepper_window,
            "dt": config.dt,
            "projector": "absent",
        },
        "architecture": {
            "compressor": [908, 512, 512, 512, 32],
            "decompressor": [63, 512, 458],
            "stepper": [63, 512, 512, 63],
            "projector": None,
        },
    }
    stopped_after = "preflight"
    try:
        print(f"{label}: 64-row capacity preflight", flush=True)
        overfit = _train_64_frame_overfit(source.training, arrays, fit_config)
        receipt["overfit_gate"] = overfit
        if not overfit["accepted"]:
            raise PreliminaryTrainingError("64-row capacity gate failed")

        stopped_after = "decompressor"
        print(
            f"{label}: fitting compressor/latent + decompressor "
            f"for {config.decompressor_steps} steps",
            flush=True,
        )
        empty = np.empty(0, dtype=np.int64)
        autoencoder = _train_decompressor_stage(
            staging,
            source.training,
            arrays,
            fit_config,
            empty,
            empty,
        )
        receipt["decompressor_training"] = autoencoder.training_metrics
        decompressor_gate = (
            visual_v2_decompressor_gate(autoencoder.gate)
            if visual_v2
            else autoencoder.gate
        )
        receipt["decompressor_gate"] = decompressor_gate
        if not decompressor_gate["accepted"]:
            raise PreliminaryTrainingError("decompressor fit gate failed")

        stopped_after = "stepper"
        print(
            f"{label}: fitting recurrent stepper for "
            f"{config.stepper_steps} steps (NO PROJECTOR)",
            flush=True,
        )
        stepper = _train_stepper_stage(
            staging,
            source.training,
            arrays,
            autoencoder,
            fit_config,
            empty,
            empty,
        )
        receipt["stepper_training"] = stepper.training_metrics
        receipt["stepper_gate"] = stepper.gate
        if not stepper.gate["accepted"]:
            raise PreliminaryTrainingError("recurrent stepper gate failed")

        stopped_after = "full-route-rollout"
        reloaded = PreliminaryModel(
            root=staging,
            latent=_load_latent(staging / "latent.bin"),
            decompressor=load_exported_network(
                staging / "decompressor.bin",
                expected_layers=[(63, 512), (512, 458)],
            ),
            stepper=load_exported_network(
                staging / "stepper.bin",
                expected_layers=[(63, 512), (512, 512), (512, 63)],
            ),
            manifest=MappingProxyType({}),
            training_receipt=MappingProxyType({}),
        )
        rollout = _full_route_rollout(source, reloaded)
        receipt["full_route_rollout_gate"] = rollout
        if not rollout["accepted"]:
            raise PreliminaryTrainingError(
                "full learned exact-route rollout gate failed"
            )

        artifacts = {
            name: _artifact_descriptor(staging / name) for name in MODEL_ARTIFACT_NAMES
        }
        receipt.update(
            {
                "status": "preliminary-not-accepted",
                "stopped_after": "full-route-rollout",
                "numerical_gates_passed": True,
                "artifacts": artifacts,
            }
        )
        (staging / "training.json").write_bytes(_json_bytes(receipt))
        manifest = build_preliminary_model_manifest(
            data_manifest_sha256=source.training.manifest_sha256,
            artifacts=artifacts,
            training_receipt=_artifact_descriptor(staging / "training.json"),
            numerical_gates_passed=True,
            visual_v2=visual_v2,
        )
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, output)
        print(
            f"{label}: numerical fit/rollout gates passed; "
            f"published {output}",
            flush=True,
        )
        return receipt
    except Exception as error:
        if staging.exists() and not output.exists():
            _publish_rejected_fit(
                staging,
                output,
                receipt,
                stopped_after=stopped_after,
                error=error,
            )
        raise PreliminaryTrainingError(
            f"preliminary fit stopped after {stopped_after}: {error}"
        ) from error


def train_preliminary_visual_v2_model(
    output_directory: str | Path,
    *,
    bundle: PreliminarySlopeBundle | None = None,
    config: PreliminaryTrainingConfig | None = None,
) -> dict[str, object]:
    """Run the one post-hoc visual-only V2 fit with the 2 mm decoder gate."""

    return train_preliminary_model(
        output_directory, bundle=bundle, config=config, visual_v2=True
    )


def smoke_preliminary_model(
    bundle: PreliminarySlopeBundle,
    model: PreliminaryModel,
    *,
    visual_v2: bool = False,
) -> dict[str, object]:
    receipt = _full_route_rollout(bundle, model)
    return {
        "schema": (
            "g1-lmm-preliminary-slope-visual-smoke/v2"
            if visual_v2
            else "g1-lmm-preliminary-slope-smoke/v1"
        ),
        "label": PRELIMINARY_VISUAL_V2_LABEL if visual_v2 else PRELIMINARY_LABEL,
        "status": "passed" if receipt["accepted"] else "rejected",
        "accepted": False,
        "generalization_claim": "none",
        "numerical_rollout_gate_passed": receipt["accepted"],
        "rollout": receipt,
    }


def _load_pynput_keyboard():
    try:
        return importlib.import_module("pynput.keyboard")
    except ModuleNotFoundError as error:
        if error.name not in ("pynput", "pynput.keyboard"):
            raise
    sonic_root = Path(__file__).resolve().parents[2]
    candidates = tuple(
        sorted((sonic_root / ".torch-mm-venv" / "lib").glob("python*/site-packages"))
    )
    if len(candidates) != 1:
        raise ModuleNotFoundError(
            "pinned viewer environment does not provide one pynput backend"
        )
    path = str(candidates[0])
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("pynput.keyboard")


def _enforce_solid_rendering(flags: object) -> None:
    import mujoco

    index = int(mujoco.mjtRndFlag.mjRND_WIREFRAME)
    try:
        flags[index] = 0  # type: ignore[index]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(
            "MuJoCo flags cannot disable the W wireframe toggle"
        ) from error


def run_interactive(
    bundle: PreliminarySlopeBundle,
    learned_model: PreliminaryModel,
    *,
    visual_v2: bool = False,
) -> int:
    rollout = _full_route_rollout(bundle, learned_model)
    if not rollout["accepted"]:
        raise PreliminaryTrainingError(
            "interactive viewer refused a failed fresh full-route rollout gate"
        )
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("interactive learned slope viewer requires DISPLAY")
    import mujoco
    import mujoco.viewer

    keyboard = _load_pynput_keyboard()
    model = build_viewer_model(bundle)
    data = mujoco.MjData(model)
    state = LearnedSlopeState(
        bundle,
        latent=learned_model.latent,
        decompressor=learned_model.decompressor,
        stepper=learned_model.stepper,
    )
    data.qpos[:] = state.qpos
    mujoco.mj_forward(model, data)
    label = PRELIMINARY_VISUAL_V2_LABEL if visual_v2 else PRELIMINARY_LABEL
    print(label, flush=True)
    print(
        "Hold W to advance the learned recurrent state and decoded G1 pose; "
        "release W for an exact pause; X exits. NO PROJECTOR.",
        flush=True,
    )
    keys = LearnedKeys()

    def on_press(key: object) -> bool | None:
        escape = key == keyboard.Key.esc
        keys.press(getattr(key, "char", None), escape=escape)
        return False if keys.snapshot()[1] else None

    def on_release(key: object) -> None:
        keys.release(getattr(key, "char", None))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    period = DT
    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                started = time.monotonic()
                w_down, stopped = keys.snapshot()
                if stopped:
                    break
                state.tick(w_down=w_down)
                data.qpos[:] = state.qpos
                data.time = state.row / FPS
                mujoco.mj_forward(model, data)
                title, body = overlay_text(
                    row=state.row,
                    frame_count=bundle.training.frames,
                    w_down=w_down,
                    label=label,
                )
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                    _enforce_solid_rendering(viewer.user_scn.flags)
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        title,
                        body,
                    )
                )
                viewer.sync()
                remaining = period - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        listener.stop()
        listener.join(timeout=1.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "train":
        receipt = train_preliminary_model(arguments.output)
        print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        return 0
    if arguments.command == "train-v2":
        receipt = train_preliminary_visual_v2_model(arguments.output)
        print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        return 0
    visual_v2 = arguments.command.endswith("-v2")
    bundle = load_preliminary_slope_bundle()
    model = load_preliminary_model(arguments.model, bundle, visual_v2=visual_v2)
    if arguments.command in ("smoke", "smoke-v2"):
        receipt = smoke_preliminary_model(bundle, model, visual_v2=visual_v2)
        print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        return 0 if receipt["numerical_rollout_gate_passed"] else 2
    if arguments.command in ("view", "view-v2"):
        return run_interactive(bundle, model, visual_v2=visual_v2)
    raise AssertionError(f"unreachable command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
