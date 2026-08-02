"""Trusted, deterministic media rendering for audited terrain-oracle inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin, UnidentifiedImageError

from mm_sonic.joints import ContractError

from .audit import structural_model_sha256, terrain_query_sha256
from .contact import CanonicalMeshQuery
from .math3d import RigidTransform
from .storage import read_clip, read_mesh


FIXED_RENDER_CONFIG = {
    "render_backend": "osmesa-cpu-headless",
    "width": 320,
    "height": 240,
    "fps": 50,
    "max_frames": 100000,
    "camera": {
        "azimuth": 135.0,
        "elevation": -20.0,
        "distance": 3.0,
        "lookat_offset_z": 0.0,
    },
    "background_rgb": [18, 22, 30],
    "left_contact_rgb": [40, 220, 100],
    "right_contact_rgb": [255, 155, 35],
}
_REQUEST_FIELDS = {
    "schema",
    "kind",
    "interval_keys",
    "inputs",
    "model",
    "render_config",
}
_INPUT_FIELDS = {
    "interval_key",
    "interval",
    "clip",
    "terrain_mesh",
    "terrain_query",
}
_CLIP_FIELDS = {
    "path",
    "size_bytes",
    "sha256",
    "clip_id",
    "source_sha256",
    "frame_count",
}
_MESH_FIELDS = {
    "path",
    "size_bytes",
    "sha256",
    "source_asset_sha256",
}
_QUERY_FIELDS = {
    "mesh_sha256",
    "query_sha256",
    "world_from_terrain",
}
_TRANSFORM_FIELDS = {
    "translation_world",
    "quaternion_world_from_local_wxyz",
}
_MODEL_FIELDS = {"path", "size_bytes", "sha256", "structural_sha256"}
_SHA256_DIGITS = frozenset("0123456789abcdef")
_TRUSTED_MODULE_NAMES = (
    "render_media.py",
    "audit.py",
    "canonical.py",
    "contact.py",
    "math3d.py",
    "storage.py",
)
_SUBPROCESS_TIMEOUT_SECONDS = 3600.0


def _canonical_json_bytes(value: object, label: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be finite canonical JSON") from error


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError(f"{label} fields do not match v1")
    return value


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_DIGITS for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _absolute_regular_path(value: object, label: str) -> Path:
    if type(value) is not str or not value:
        raise ContractError(f"{label} must be an absolute canonical path")
    path = Path(value)
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContractError(f"{label} does not resolve to a regular file") from error
    if path != resolved or path.is_symlink():
        raise ContractError(f"{label} must be canonical and contain no symlink")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ContractError(f"{label} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} must be a regular file")
    return path


def _read_regular_file(value: object, label: str) -> tuple[Path, bytes]:
    path = _absolute_regular_path(value, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(f"{label} must be a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ContractError(f"{label} cannot be read safely") from error
    return path, b"".join(chunks)


def _match_file_record(
    value: dict[str, Any],
    fields: set[str],
    label: str,
) -> tuple[Path, bytes]:
    record = _exact_dict(value, fields, label)
    path, payload = _read_regular_file(record["path"], f"{label}.path")
    if (
        _positive_int(record["size_bytes"], f"{label}.size_bytes") != len(payload)
        or _digest(record["sha256"], f"{label}.sha256")
        != hashlib.sha256(payload).hexdigest()
    ):
        raise ContractError(f"{label} file authority is stale")
    return path, payload


def _validated_transform(value: object) -> RigidTransform:
    transform = _exact_dict(value, _TRANSFORM_FIELDS, "world_from_terrain")
    try:
        return RigidTransform(
            translation_world=transform["translation_world"],
            quaternion_world_from_local_wxyz=transform[
                "quaternion_world_from_local_wxyz"
            ],
        )
    except (TypeError, ValueError) as error:
        raise ContractError("world_from_terrain is invalid") from error


def validate_render_request(value: object) -> dict[str, Any]:
    """Validate and reload every authority bound by a v1 render request."""

    request = _exact_dict(value, _REQUEST_FIELDS, "render request")
    if request["schema"] != "terrain-oracle-render-request/v1":
        raise ContractError("unsupported render request schema")
    if request["kind"] not in {
        "accepted_interval",
        "contact_sheet",
        "full_video",
    }:
        raise ContractError("unsupported render request kind")
    if request["render_config"] != FIXED_RENDER_CONFIG:
        raise ContractError("render request configuration is not the fixed profile")

    model_record = _exact_dict(request["model"], _MODEL_FIELDS, "model")
    model_path, model_payload = _match_file_record(
        model_record, _MODEL_FIELDS, "model"
    )
    _digest(model_record["structural_sha256"], "model.structural_sha256")
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(model_path))
    except (ImportError, ValueError) as error:
        raise ContractError("model must be a loadable MuJoCo XML model") from error
    if structural_model_sha256(model) != model_record["structural_sha256"]:
        raise ContractError("model structural authority is stale")
    if len(model_payload) != model_record["size_bytes"]:
        raise ContractError("model byte authority is stale")

    keys = request["interval_keys"]
    inputs = request["inputs"]
    if (
        not isinstance(keys, list)
        or not keys
        or any(type(key) is not str or not key for key in keys)
        or len(set(keys)) != len(keys)
        or keys != sorted(keys)
        or not isinstance(inputs, list)
        or not inputs
        or len(inputs) != len(keys)
    ):
        raise ContractError("render request must bind an exact ordered interval set")
    if request["kind"] == "accepted_interval" and len(inputs) != 1:
        raise ContractError("accepted_interval requests bind exactly one input")

    loaded_inputs: list[dict[str, object]] = []
    for expected_key, raw_input in zip(keys, inputs, strict=True):
        item = _exact_dict(raw_input, _INPUT_FIELDS, "render input")
        if item["interval_key"] != expected_key:
            raise ContractError("render input order does not match interval_keys")
        interval = item["interval"]
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(frame) is not int for frame in interval)
        ):
            raise ContractError("render interval must be a two-integer list")
        start, end = interval
        if start < 0 or end <= start:
            raise ContractError("render interval must be nonempty and half-open")

        clip_record = _exact_dict(item["clip"], _CLIP_FIELDS, "clip")
        clip_path, _ = _match_file_record(clip_record, _CLIP_FIELDS, "clip")
        if clip_path.stem != clip_record["sha256"]:
            raise ContractError("clip path is not content addressed")
        clip = read_clip(clip_path)
        if (
            clip.clip_id != clip_record["clip_id"]
            or clip.source.source_sha256 != clip_record["source_sha256"]
            or clip.frame_count
            != _positive_int(clip_record["frame_count"], "clip.frame_count")
            or end > clip.frame_count
        ):
            raise ContractError("clip identity or interval is stale")
        canonical_key = f"{clip_record['sha256']}:{start}:{end}"
        if expected_key != canonical_key:
            raise ContractError("interval_key does not bind clip and frames")
        if clip.terrain is None:
            raise ContractError("render input clip has no terrain binding")

        mesh_record = _exact_dict(
            item["terrain_mesh"], _MESH_FIELDS, "terrain_mesh"
        )
        mesh_path, _ = _match_file_record(
            mesh_record, _MESH_FIELDS, "terrain_mesh"
        )
        if mesh_path.stem != mesh_record["sha256"]:
            raise ContractError("terrain mesh path is not content addressed")
        mesh = read_mesh(mesh_path)
        if (
            mesh.source_asset_sha256 != mesh_record["source_asset_sha256"]
            or clip.terrain.mesh_sha256 != mesh_record["sha256"]
            or clip.terrain.asset_sha256 != mesh.source_asset_sha256
        ):
            raise ContractError("clip and terrain mesh identities do not match")

        query_record = _exact_dict(
            item["terrain_query"], _QUERY_FIELDS, "terrain_query"
        )
        if query_record["mesh_sha256"] != mesh_record["sha256"]:
            raise ContractError("terrain query does not bind the canonical mesh")
        transform = _validated_transform(query_record["world_from_terrain"])
        if (
            not np.array_equal(
                transform.translation_world,
                clip.terrain.world_from_terrain.translation_world,
            )
            or not np.array_equal(
                transform.quaternion_world_from_local_wxyz,
                clip.terrain.world_from_terrain.quaternion_world_from_local_wxyz,
            )
        ):
            raise ContractError("terrain transform does not match the clip binding")
        _digest(query_record["query_sha256"], "terrain_query.query_sha256")
        query = CanonicalMeshQuery(mesh, transform)
        if terrain_query_sha256(query) != query_record["query_sha256"]:
            raise ContractError("terrain query authority is stale")
        loaded_inputs.append(
            {
                "clip": clip,
                "mesh": mesh,
                "transform": transform,
                "start": start,
                "end": end,
                "interval_key": expected_key,
                "clip_path": clip_path,
                "mesh_path": mesh_path,
            }
        )
    if sum(int(item["end"]) - int(item["start"]) for item in loaded_inputs) > int(
        FIXED_RENDER_CONFIG["max_frames"]
    ):
        raise ContractError("render request exceeds the bounded frame limit")
    if len(loaded_inputs) > int(FIXED_RENDER_CONFIG["width"]) * int(
        FIXED_RENDER_CONFIG["height"]
    ):
        raise ContractError("render request has too many inputs for exact tile coverage")
    return {**request, "_loaded_inputs": loaded_inputs, "_model": model}


def load_render_request(path: Path) -> dict[str, object]:
    request_path, payload = _read_regular_file(str(Path(path)), "request")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("render request is not ASCII JSON") from error
    if payload != _canonical_json_bytes(value, "render request"):
        raise ContractError("render request is not canonical JSON")
    validated = validate_render_request(value)
    validated["_request_path"] = request_path
    validated["_request_sha256"] = hashlib.sha256(payload).hexdigest()
    return validated


def _tool_path(name: str) -> Path:
    discovered = shutil.which(name)
    if discovered is None:
        raise ContractError(f"required media tool is unavailable: {name}")
    path = Path(discovered).resolve()
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"required media tool is not a regular file: {name}")
    return path


def _run_bounded(
    argv: list[str], *, input_bytes: bytes | None = None, timeout: float = 30.0
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            argv,
            input=input_bytes,
            capture_output=True,
            shell=False,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError(f"bounded subprocess failed: {Path(argv[0]).name}") from error


def _file_identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _tool_identity(name: str) -> dict[str, object]:
    path = _tool_path(name)
    version = _run_bounded([str(path), "-version"], timeout=10.0)
    if version.returncode != 0 or not version.stdout:
        raise ContractError(f"{name} version query failed")
    output = version.stdout.decode("utf-8", errors="strict")
    return {
        **_file_identity(path),
        "version_output": output,
        "version_output_sha256": hashlib.sha256(version.stdout).hexdigest(),
    }


def runtime_identity() -> dict[str, object]:
    """Return a canonical identity for all trusted code, tools, and dependencies."""

    package = Path(__file__).resolve().parent
    package_files = []
    for name in _TRUSTED_MODULE_NAMES:
        path = package / name
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"trusted package module is unavailable: {name}")
        package_files.append({"name": name, **_file_identity(path)})
    module = next(
        value for value in package_files if value["name"] == "render_media.py"
    )
    config_bytes = _canonical_json_bytes(FIXED_RENDER_CONFIG, "render config")
    python_path = Path(sys.executable).resolve()
    if not python_path.is_file():
        raise ContractError("Python executable is unavailable")
    without_hash: dict[str, object] = {
        "schema": "terrain-oracle-render-runtime/v1",
        "module": {
            **module,
            "trusted_package_files": package_files,
        },
        "config": {
            "size_bytes": len(config_bytes),
            "bytes_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "python": _file_identity(python_path),
        "ffmpeg": _tool_identity("ffmpeg"),
        "ffprobe": _tool_identity("ffprobe"),
        "dependencies": {
            name: importlib.metadata.version(distribution)
            for name, distribution in (
                ("mujoco", "mujoco"),
                ("numpy", "numpy"),
                ("Pillow", "Pillow"),
            )
        },
    }
    return {
        **without_hash,
        "content_sha256": hashlib.sha256(
            _canonical_json_bytes(without_hash, "runtime identity")
        ).hexdigest(),
    }


def _validated_new_output_path(value: Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ContractError(f"{label} must be an absolute canonical path")
    parent = path.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{label} parent does not exist") from error
    if parent != resolved_parent or not parent.is_dir() or parent.is_symlink():
        raise ContractError(f"{label} parent must be a canonical regular directory")
    if path.exists() or path.is_symlink():
        raise ContractError(f"{label} must not already exist")
    return path


def _build_scene_model(
    model_path: Path,
    mesh: object,
    transform: RigidTransform,
) -> tuple[object, int]:
    try:
        import mujoco

        spec = mujoco.MjSpec.from_file(str(model_path))
        faces = np.asarray(mesh.faces[mesh.valid_faces], dtype=np.int32)
        spec.add_mesh(
            name="terrain_oracle_surface",
            uservert=np.asarray(mesh.vertices_local, dtype=np.float64).ravel(),
            userface=faces.ravel(),
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
        )
        spec.worldbody.add_geom(
            name="terrain_oracle_surface",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="terrain_oracle_surface",
            pos=np.asarray(transform.translation_world, dtype=np.float64),
            quat=np.asarray(
                transform.quaternion_world_from_local_wxyz, dtype=np.float64
            ),
            contype=0,
            conaffinity=0,
            rgba=[0.20, 0.34, 0.24, 1.0],
        )
        model = spec.compile()
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractError("failed to compile G1 model with canonical terrain") from error
    visual_mesh_count = sum(
        int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        for index in range(model.ngeom)
    ) - 1
    if visual_mesh_count < 1:
        raise ContractError("G1 model has no actual visual mesh geometry")
    return model, visual_mesh_count


def _set_canonical_qpos(model: object, data: object, clip: object, frame: int) -> None:
    import mujoco

    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    if base_id < 0 or int(model.jnt_type[base_id]) != int(
        mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ContractError("G1 model is missing the canonical floating base")
    base_address = int(model.jnt_qposadr[base_id])
    data.qpos[base_address : base_address + 3] = clip.root_position_world[frame]
    data.qpos[base_address + 3 : base_address + 7] = (
        clip.root_quaternion_world_wxyz[frame]
    )
    if tuple(clip.joint_names) != tuple(dict.fromkeys(clip.joint_names)):
        raise ContractError("canonical clip joint names are not unique")
    for clip_index, name in enumerate(clip.joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ContractError(f"G1 model is missing canonical joint {name}")
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address] = clip.joint_position[frame, clip_index]


def _annotate_contact(frame: np.ndarray, contact: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    height = image.height
    colors = (
        tuple(FIXED_RENDER_CONFIG["left_contact_rgb"]),
        tuple(FIXED_RENDER_CONFIG["right_contact_rgb"]),
    )
    positions = ((8, height - 24, 66, height - 7), (73, height - 24, 131, height - 7))
    for index, (box, color) in enumerate(zip(positions, colors, strict=True)):
        active = float(contact[index]) >= 0.5
        draw.rectangle(
            box,
            fill=color if active else (36, 40, 48),
            outline=color,
            width=2,
        )
        if active:
            inner = (box[0] + 5, box[1] + 5, box[2] - 5, box[3] - 5)
            draw.rectangle(inner, outline=(255, 255, 255), width=1)
    return np.asarray(image, dtype=np.uint8)


def _render_input_frames(
    item: dict[str, object],
    model_path: Path,
    consume: Any,
) -> tuple[np.ndarray, dict[str, int], int]:
    import mujoco

    clip = item["clip"]
    model, visual_mesh_count = _build_scene_model(
        model_path, item["mesh"], item["transform"]
    )
    data = mujoco.MjData(model)
    width = int(FIXED_RENDER_CONFIG["width"])
    height = int(FIXED_RENDER_CONFIG["height"])
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = float(FIXED_RENDER_CONFIG["camera"]["azimuth"])
    camera.elevation = float(FIXED_RENDER_CONFIG["camera"]["elevation"])
    camera.distance = float(FIXED_RENDER_CONFIG["camera"]["distance"])
    start = int(item["start"])
    end = int(item["end"])
    representative_index = start + (end - start - 1) // 2
    representative: np.ndarray | None = None
    try:
        for frame_index in range(start, end):
            _set_canonical_qpos(model, data, clip, frame_index)
            root = np.asarray(
                clip.root_position_world[frame_index], dtype=np.float64
            )
            camera.lookat[:] = (
                float(root[0]),
                float(root[1]),
                float(root[2])
                + float(FIXED_RENDER_CONFIG["camera"]["lookat_offset_z"]),
            )
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            annotated = _annotate_contact(pixels, clip.contact[frame_index])
            consume(annotated)
            if frame_index == representative_index:
                representative = annotated.copy()
    finally:
        renderer.close()
    if representative is None:
        raise ContractError("render interval produced no representative frame")
    return representative, {
        "visual_mesh_geom_count": visual_mesh_count,
        "terrain_face_count": int(np.count_nonzero(item["mesh"].valid_faces)),
    }, end - start


def _encode_video(
    path: Path,
    inputs: Sequence[dict[str, object]],
    model_path: Path,
) -> tuple[list[np.ndarray], dict[str, int], int]:
    ffmpeg = _tool_path("ffmpeg")
    width = int(FIXED_RENDER_CONFIG["width"])
    height = int(FIXED_RENDER_CONFIG["height"])
    fps = int(FIXED_RENDER_CONFIG["fps"])
    argv = [
        str(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
        "-sc_threshold",
        "0",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        "-y",
        str(path),
    ]
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        shell=False,
    )
    representative_frames: list[np.ndarray] = []
    evidence = {"visual_mesh_geom_count": 0, "terrain_face_count": 0}
    frame_count = 0
    try:
        assert process.stdin is not None
        def consume(frame: np.ndarray) -> None:
            assert process.stdin is not None
            process.stdin.write(
                np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
            )

        for item in inputs:
            representative, item_evidence, item_frames = _render_input_frames(
                item, model_path, consume
            )
            representative_frames.append(representative)
            evidence["visual_mesh_geom_count"] = max(
                evidence["visual_mesh_geom_count"],
                item_evidence["visual_mesh_geom_count"],
            )
            evidence["terrain_face_count"] += item_evidence["terrain_face_count"]
            frame_count += item_frames
        process.stdin.close()
        returncode = process.wait(timeout=_SUBPROCESS_TIMEOUT_SECONDS)
        assert process.stderr is not None
        errors = process.stderr.read()
        process.stderr.close()
    except BaseException as error:
        process.kill()
        process.wait()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.stderr is not None and not process.stderr.closed:
            process.stderr.close()
        if isinstance(error, ContractError):
            raise
        raise ContractError("ffmpeg video encoding failed") from error
    if returncode != 0:
        raise ContractError(
            "ffmpeg video encoding failed: "
            + errors.decode("utf-8", errors="replace")[:400]
        )
    return representative_frames, evidence, frame_count


def _save_overlay(
    path: Path,
    representative_frames: Sequence[np.ndarray],
    interval_keys: Sequence[str],
) -> None:
    width = int(FIXED_RENDER_CONFIG["width"])
    height = int(FIXED_RENDER_CONFIG["height"])
    columns = max(1, math.ceil(math.sqrt(len(representative_frames))))
    rows = math.ceil(len(representative_frames) / columns)
    tile_width = width // columns
    tile_height = height // rows
    sheet = Image.new(
        "RGB", (width, height), tuple(FIXED_RENDER_CONFIG["background_rgb"])
    )
    for index, frame in enumerate(representative_frames):
        tile = Image.fromarray(frame)
        tile.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width + (tile_width - tile.width) // 2
        y = (index // columns) * tile_height + (tile_height - tile.height) // 2
        sheet.paste(tile, (x, y))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text(
        "terrain_oracle_interval_keys",
        json.dumps(
            list(interval_keys),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ),
    )
    metadata.add_text("terrain_oracle_tile_count", str(len(representative_frames)))
    sheet.save(path, format="PNG", compress_level=9, optimize=False, pnginfo=metadata)


def validate_video(
    path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frame_count: int,
) -> dict[str, object]:
    """Require exact H.264 metadata and a successful bounded full decode."""

    media_path = _absolute_regular_path(str(Path(path)), "video")
    if any(
        type(value) is not int or value < 1
        for value in (width, height, fps, frame_count)
    ):
        raise ContractError("expected video metadata must be positive integers")
    probe = _run_bounded(
        [
            str(_tool_path("ffprobe")),
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(media_path),
        ],
        timeout=30.0,
    )
    try:
        document = json.loads(probe.stdout.decode("utf-8"))
        streams = document["streams"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ContractError("ffprobe did not return valid stream metadata") from error
    if probe.returncode != 0 or not isinstance(streams, list) or len(streams) != 1:
        raise ContractError("video must contain exactly one valid stream")
    stream = streams[0]
    try:
        actual = {
            "codec_name": stream["codec_name"],
            "codec_type": stream["codec_type"],
            "pix_fmt": stream["pix_fmt"],
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "r_frame_rate": stream["r_frame_rate"],
            "avg_frame_rate": stream["avg_frame_rate"],
            "nb_frames": int(stream["nb_frames"]),
            "nb_read_frames": int(stream["nb_read_frames"]),
        }
        exact = (
            actual["codec_name"] == "h264"
            and actual["codec_type"] == "video"
            and actual["pix_fmt"] == "yuv420p"
            and actual["width"] == width
            and actual["height"] == height
            and Fraction(str(actual["r_frame_rate"])) == fps
            and Fraction(str(actual["avg_frame_rate"])) == fps
            and actual["nb_frames"] == frame_count
            and actual["nb_read_frames"] == frame_count
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ContractError("video metadata is incomplete") from error
    if not exact:
        raise ContractError("video metadata does not match the declared profile")
    decode = _run_bounded(
        [
            str(_tool_path("ffmpeg")),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(media_path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )
    if decode.returncode != 0 or decode.stderr:
        raise ContractError("video failed a complete ffmpeg decode")
    return actual


def validate_png(
    path: Path,
    *,
    width: int,
    height: int,
    interval_keys: Sequence[str],
) -> dict[str, object]:
    """Verify PNG structure, pixels, and the exact complete tile/input manifest."""

    png_path = _absolute_regular_path(str(Path(path)), "contact overlay")
    try:
        with Image.open(png_path) as image:
            image.verify()
        with Image.open(png_path) as image:
            image.load()
            mode = image.mode
            size = image.size
            info = dict(image.info)
            pixels = np.asarray(image)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ContractError("contact overlay is not a valid complete PNG") from error
    if mode != "RGB" or size != (width, height):
        raise ContractError("contact overlay dimensions or mode are wrong")
    if pixels.size == 0 or np.all(pixels == pixels.reshape(-1, 3)[0]):
        raise ContractError("contact overlay pixels are blank")
    try:
        declared_keys = json.loads(info["terrain_oracle_interval_keys"])
        tile_count = int(info["terrain_oracle_tile_count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError("contact overlay coverage metadata is invalid") from error
    expected_keys = list(interval_keys)
    if (
        declared_keys != expected_keys
        or tile_count != len(expected_keys)
        or tile_count < 1
    ):
        raise ContractError("contact overlay does not cover every declared input")
    return {
        "mode": mode,
        "width": width,
        "height": height,
        "tile_count": tile_count,
        "interval_keys": declared_keys,
    }


validate_contact_overlay = validate_png


def _artifact_result(path: Path, metadata: dict[str, object]) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "validated_metadata": metadata,
    }


def _publish_no_replace(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise ContractError(f"output appeared concurrently: {destination}") from error
    source.unlink()


def render_media(
    request_path: Path,
    video_path: Path,
    overlay_path: Path,
) -> dict[str, object]:
    """Render, encode, validate, and receipt one authenticated request."""

    video = _validated_new_output_path(Path(video_path), "video output")
    overlay = _validated_new_output_path(Path(overlay_path), "overlay output")
    if video == overlay:
        raise ContractError("video and overlay outputs must be distinct")
    request = load_render_request(Path(request_path))
    model_path = Path(request["model"]["path"])
    temporary_paths: list[Path] = []
    try:
        video_descriptor, video_name = tempfile.mkstemp(
            prefix=f".{video.name}.", suffix=".mp4", dir=video.parent
        )
        os.close(video_descriptor)
        temporary_video = Path(video_name)
        temporary_paths.append(temporary_video)
        overlay_descriptor, overlay_name = tempfile.mkstemp(
            prefix=f".{overlay.name}.", suffix=".png", dir=overlay.parent
        )
        os.close(overlay_descriptor)
        temporary_overlay = Path(overlay_name)
        temporary_paths.append(temporary_overlay)
        representative_frames, evidence, frame_count = _encode_video(
            temporary_video, request["_loaded_inputs"], model_path
        )
        _save_overlay(
            temporary_overlay, representative_frames, request["interval_keys"]
        )
        video_metadata = validate_video(
            temporary_video,
            width=int(FIXED_RENDER_CONFIG["width"]),
            height=int(FIXED_RENDER_CONFIG["height"]),
            fps=int(FIXED_RENDER_CONFIG["fps"]),
            frame_count=frame_count,
        )
        png_metadata = validate_png(
            temporary_overlay,
            width=int(FIXED_RENDER_CONFIG["width"]),
            height=int(FIXED_RENDER_CONFIG["height"]),
            interval_keys=request["interval_keys"],
        )
        _publish_no_replace(temporary_video, video)
        temporary_paths.remove(temporary_video)
        _publish_no_replace(temporary_overlay, overlay)
        temporary_paths.remove(temporary_overlay)
    except BaseException:
        video.unlink(missing_ok=True)
        overlay.unlink(missing_ok=True)
        raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
    result = {
        "schema": "terrain-oracle-render-result/v1",
        "request_sha256": request["_request_sha256"],
        "kind": request["kind"],
        "interval_keys": request["interval_keys"],
        "video": _artifact_result(video, video_metadata),
        "contact_overlay": _artifact_result(overlay, png_metadata),
        "runtime_identity": runtime_identity(),
        "render_evidence": evidence,
        "completed": True,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render authenticated terrain-oracle media."
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--overlay", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = render_media(
            Path(arguments.request),
            Path(arguments.video),
            Path(arguments.overlay),
        )
        sys.stdout.buffer.write(
            _canonical_json_bytes(result, "render result")
        )
        sys.stdout.buffer.flush()
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(f"render_media: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
