from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np

from .artifacts import verify_run_inventory
from .joints import ContractError

FPS = 50
FRAME_COUNT = 600
WIDTH = 1280
HEIGHT = 720
DISCLAIMER = "POST-RUN SEALED STATE REPLAY"
_SOURCE_FILES = (
    "manifest.json",
    "inventory.json",
    "stage-b-evidence.json",
    "scene/gear_scene.xml",
    "scene/gear_robot.xml",
    "dynamic/stream/scored-sim-logs/state.jsonl",
)


@dataclass(frozen=True)
class ReplayState:
    step: int
    sim_time_s: float
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass(frozen=True)
class ReplayMetrics:
    run_id: str
    dynamic_pass: bool
    duration_s: float
    forbidden_contact_count: int
    minimum_pelvis_height_m: float
    minimum_pelvis_up_dot: float
    joint_rmse_rad: float
    orientation_rms_rad: float
    horizontal_path_drift_m: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"replay JSON is unreadable: {path}") from error
    if type(value) is not dict:
        raise ContractError(f"replay JSON must be an object: {path}")
    return value


def load_state_stream(path: Path, *, expected_count: int, nq: int, nv: int) -> tuple[ReplayState, ...]:
    states: list[ReplayState] = []
    try:
        lines = path.read_text("utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ContractError("state stream is unreadable") from error
    if len(lines) != expected_count:
        raise ContractError(
            f"state stream must contain exactly {expected_count} rows"
        )
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"state row {index + 1} is invalid JSON") from error
        expected_step = 4 * (index + 1)
        expected_time = 0.02 * (index + 1)
        if type(row) is not dict or row.get("step") != expected_step:
            raise ContractError("state step sequence is not contiguous")
        sim_time = row.get("sim_time_s")
        if not isinstance(sim_time, (int, float)) or isinstance(sim_time, bool):
            raise ContractError("state time sequence is invalid")
        if not math.isfinite(float(sim_time)) or not math.isclose(
            float(sim_time), expected_time, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ContractError("state time sequence is not contiguous")
        nested = row.get("state")
        if type(nested) is not dict:
            raise ContractError("state payload must be an object")
        try:
            qpos = np.asarray(nested.get("qpos"), dtype=np.float64)
            qvel = np.asarray(nested.get("qvel"), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ContractError("state vectors must be numeric") from error
        if qpos.shape != (nq,) or qvel.shape != (nv,):
            raise ContractError("state vector width does not match the model")
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
            raise ContractError("state vectors must be finite")
        states.append(ReplayState(expected_step, float(sim_time), qpos, qvel))
    return tuple(states)


def _metrics(manifest: Mapping[str, Any], evidence: Mapping[str, Any]) -> ReplayMetrics:
    try:
        dynamic = evidence["metrics"]["stage_b_dynamic"]
        safety = dynamic["safety"]
        tracking = dynamic["tracking"]
        secondary = dynamic["secondary"]
        value = ReplayMetrics(
            run_id=str(manifest["run_id"]),
            dynamic_pass=evidence["dynamic_pass"] is True,
            duration_s=float(evidence["expected_sim_time_s"]),
            forbidden_contact_count=len(safety["forbidden_contact_groups"]),
            minimum_pelvis_height_m=float(safety["minimum_pelvis_local_height_m"]),
            minimum_pelvis_up_dot=float(safety["minimum_pelvis_up_dot"]),
            joint_rmse_rad=float(tracking["joint_position_rmse_rad"]),
            orientation_rms_rad=float(tracking["pelvis_orientation_rms_rad"]),
            horizontal_path_drift_m=float(secondary["horizontal_path_drift_m"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("Stage B replay metrics are incomplete") from error
    numeric = tuple(
        item for item in asdict(value).values()
        if isinstance(item, float)
    )
    if not value.dynamic_pass or value.duration_s != 12.0 or not all(
        math.isfinite(item) for item in numeric
    ):
        raise ContractError("Stage B replay metrics violate the sealed contract")
    return value


def _mujoco() -> Any:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
    except ImportError as error:
        raise ContractError("MuJoCo Python bindings are unavailable") from error
    return mujoco


def _ensure_offscreen_framebuffer(model: object, *, width: int, height: int) -> None:
    global_visual = model.vis.global_
    global_visual.offwidth = max(int(global_visual.offwidth), width)
    global_visual.offheight = max(int(global_visual.offheight), height)


def load_replay_inputs(run_root: Path) -> tuple[object, tuple[ReplayState, ...], ReplayMetrics, dict[str, str]]:
    root = run_root.expanduser().resolve(strict=True)
    if not root.is_dir() or not verify_run_inventory(root):
        raise ContractError("sealed run inventory does not verify")
    sources = {relative: _sha256(root / relative) for relative in _SOURCE_FILES}
    manifest = _json(root / "manifest.json")
    evidence = _json(root / "stage-b-evidence.json")
    mujoco = _mujoco()
    try:
        model = mujoco.MjModel.from_xml_path(str(root / "scene/gear_scene.xml"))
    except Exception as error:
        raise ContractError("run-local MuJoCo scene cannot be loaded") from error
    if model.nq != 50 or model.nv != 49:
        raise ContractError("run-local model must have nq=50 and nv=49")
    states = load_state_stream(
        root / "dynamic/stream/scored-sim-logs/state.jsonl",
        expected_count=FRAME_COUNT,
        nq=model.nq,
        nv=model.nv,
    )
    return model, states, _metrics(manifest, evidence), sources


def validate_probe(probe: Mapping[str, object]) -> None:
    streams = probe.get("streams")
    media_format = probe.get("format")
    if type(streams) is not list or len(streams) != 1 or type(media_format) is not dict:
        raise ContractError("ffprobe result must contain one video stream")
    stream = streams[0]
    if type(stream) is not dict or stream.get("codec_name") != "h264":
        raise ContractError("video codec must be H.264")
    if stream.get("width") != WIDTH or stream.get("height") != HEIGHT:
        raise ContractError("video dimensions must be 1280x720")
    if stream.get("pix_fmt") != "yuv420p":
        raise ContractError("video pixel format must be yuv420p")
    if stream.get("avg_frame_rate") != "50/1":
        raise ContractError("video frame rate must be 50/1")
    if stream.get("nb_frames") != str(FRAME_COUNT):
        raise ContractError("video must contain exactly 600 frames")
    try:
        duration = float(media_format["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("video duration is unavailable") from error
    if not math.isclose(duration, 12.0, rel_tol=0.0, abs_tol=1e-6):
        raise ContractError("video duration must be 12.0 seconds")


def _run_checked(argv: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        raise ContractError(
            f"media command failed ({argv[0]}): {completed.stderr[-2000:]}"
        )
    return completed


def render_replay(run_root: Path, output_dir: Path, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict[str, object]:
    root = run_root.expanduser().resolve(strict=True)
    out = output_dir.expanduser().resolve(strict=False)
    if out == root or root in out.parents:
        raise ContractError("video output must be outside the sealed run")
    model, states, metrics, before_hashes = load_replay_inputs(root)
    out.mkdir(parents=True, exist_ok=True)
    stem = "g1-sonic-stage-b-r13"
    final_video = out / f"{stem}-replay.mp4"
    final_montage = out / f"{stem}-montage.png"
    final_metadata = out / f"{stem}-video-metadata.json"
    suffix = f".tmp-{os.getpid()}"
    tmp_video = out / f"{stem}{suffix}.mp4"
    tmp_montage = out / f"{stem}{suffix}.png"
    tmp_metadata = out / f"{stem}{suffix}.json"
    temporary = (tmp_video, tmp_montage, tmp_metadata)
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    if not font.is_file() or re.fullmatch(r"[A-Za-z0-9._-]+", metrics.run_id) is None:
        raise ContractError("safe replay label resources are unavailable")
    label = (
        "drawbox=x=0:y=0:w=iw:h=96:color=black@0.72:t=fill,"
        f"drawtext=fontfile={font}:text='{DISCLAIMER}':fontcolor=yellow:fontsize=24:x=20:y=10,"
        f"drawtext=fontfile={font}:text='run {metrics.run_id} | frame %{{eif\\:n+1\\:d}}/600 | t=%{{pts\\:hms}}':"
        "fontcolor=white:fontsize=17:x=20:y=42,"
        f"drawtext=fontfile={font}:text='Stage B PASS | path drift {metrics.horizontal_path_drift_m:.3f} m | "
        f"forbidden contacts {metrics.forbidden_contact_count} | pelvis min {metrics.minimum_pelvis_height_m:.3f} m':"
        "fontcolor=white:fontsize=16:x=20:y=68,"
        "drawbox=x=639:y=96:w=2:h=624:color=white@0.5:t=fill"
    )
    encode_argv = [
        ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-vf", label,
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp_video),
    ]
    mujoco = _mujoco()
    _ensure_offscreen_framebuffer(model, width=WIDTH // 2, height=HEIGHT)
    data = mujoco.MjData(model)
    fixed = mujoco.MjvCamera()
    tracking = mujoco.MjvCamera()
    xy = np.asarray([state.qpos[:2] for state in states])
    midpoint = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    fixed.lookat[:] = (midpoint[0], midpoint[1], 0.8)
    fixed.distance = max(4.0, float(np.ptp(xy, axis=0).max()) * 1.8 + 2.0)
    fixed.azimuth = 135.0
    fixed.elevation = -25.0
    tracking.distance = 3.0
    tracking.azimuth = 135.0
    tracking.elevation = -15.0
    left = right = encoder = None
    try:
        left = mujoco.Renderer(model, height=HEIGHT, width=WIDTH // 2)
        right = mujoco.Renderer(model, height=HEIGHT, width=WIDTH // 2)
        encoder = subprocess.Popen(
            encode_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert encoder.stdin is not None
        for state in states:
            data.qpos[:] = state.qpos
            data.qvel[:] = state.qvel
            data.time = state.sim_time_s
            mujoco.mj_forward(model, data)
            tracking.lookat[:] = state.qpos[:3]
            left.update_scene(data, camera=fixed)
            right.update_scene(data, camera=tracking)
            frame = np.concatenate((left.render(), right.render()), axis=1)
            encoder.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        encoder.stdin.close()
        assert encoder.stderr is not None
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")
        returncode = encoder.wait()
        if returncode != 0:
            raise ContractError(f"ffmpeg encoding failed: {stderr[-2000:]}")
        encoder = None
        select = "select='" + "+".join(
            f"eq(n,{frame})" for frame in (0, 119, 239, 359, 479, 599)
        ) + "',scale=640:360,tile=3x2"
        _run_checked([
            ffmpeg, "-v", "error", "-y", "-i", str(tmp_video), "-vf", select,
            "-frames:v", "1", str(tmp_montage),
        ])
        probe_text = _run_checked([
            ffprobe, "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(tmp_video),
        ]).stdout
        probe = json.loads(probe_text)
        validate_probe(probe)
        after_hashes = {relative: _sha256(root / relative) for relative in _SOURCE_FILES}
        if after_hashes != before_hashes or not verify_run_inventory(root):
            raise ContractError("sealed source changed during replay rendering")
        output_hashes = {
            final_video.name: _sha256(tmp_video),
            final_montage.name: _sha256(tmp_montage),
        }
        metadata: dict[str, object] = {
            "schema": "mm-sonic-video-evidence/v1",
            "disclaimer": DISCLAIMER,
            "claim_boundaries": {"post_run_replay": True, "live_control": False,
                                 "physical_hardware": False},
            "source_run": str(root),
            "source_hashes": before_hashes,
            "output_hashes": output_hashes,
            "renderer_sha256": _sha256(Path(__file__)),
            "metrics": asdict(metrics),
            "media": probe,
            "camera": {"left": "fixed-world", "right": "pelvis-tracking"},
        }
        tmp_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", "utf-8"
        )
        os.replace(tmp_video, final_video)
        os.replace(tmp_montage, final_montage)
        os.replace(tmp_metadata, final_metadata)
        return metadata
    finally:
        if left is not None:
            left.close()
        if right is not None:
            right.close()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        for path in temporary:
            path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a sealed Stage B state replay")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        model, states, metrics, sources = load_replay_inputs(args.run_root)
        result: dict[str, object] = {
            "inventory_verified": True,
            "state_count": len(states),
            "nq": model.nq,
            "nv": model.nv,
            "metrics": asdict(metrics),
            "source_hashes": sources,
        }
    else:
        result = render_replay(args.run_root, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
