# G1 Sonic Sealed Replay Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible 12-second MP4 and six-frame montage from the sealed G1 Sonic Stage B state log without rerunning or modifying the source run.

**Architecture:** Add one focused `mm_sonic.replay_video` module that validates the sealed inputs, replays recorded generalized state through run-local MuJoCo forward kinematics, streams a dual-camera RGB composition through `ffmpeg`, and verifies the encoded media through `ffprobe`. Unit tests cover the fail-closed input and media contracts; the final controller-owned step invokes the module against the immutable r13 run and independently verifies hashes, inventory, frames, and process cleanup.

**Tech Stack:** Python 3.10+, NumPy, MuJoCo Python bindings, standard library, system `ffmpeg`/`ffprobe`, `unittest`.

## Global Constraints

- The source is `sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/stage-b/stage-b-20260717T190656038623Z-b746af3f`.
- Never write inside the source run or regenerate its `inventory.json`.
- Output lives in the sibling `video-evidence/` directory outside the source run.
- Render exactly 600 frames at 50 fps, 1280×720, H.264/yuv420p, with a 12.0-second duration.
- Every frame visibly says `POST-RUN SEALED STATE REPLAY`; metadata repeats that disclaimer and denies live-control or hardware claims.
- The overlay includes the 3.813 m horizontal path drift rather than hiding it.
- Rendering fails closed on input, inventory, model, encoder, media-property, or source-hash errors.
- Do not add a Python dependency; use NumPy for composition and system `ffmpeg` for labels, encoding, and montage generation.
- Do not start or modify any Reliable Claude job using a runtime other than release SHA `11f44df060fa011db504199917beb3e8bb5200dd`.

---

### Task 1: Fail-closed sealed-state replay renderer

**Files:**
- Create: `sonic/python/mm_sonic/replay_video.py`
- Create: `tests/python/test_sonic_replay_video.py`

**Interfaces:**
- Consumes: a sealed run root containing `manifest.json`, `inventory.json`, `stage-b-evidence.json`, `scene/gear_scene.xml`, and `dynamic/stream/scored-sim-logs/state.jsonl`.
- Produces: `ReplayState(step: int, sim_time_s: float, qpos: np.ndarray, qvel: np.ndarray)`, `ReplayMetrics`, `load_replay_inputs(run_root: Path)`, `validate_probe(probe: Mapping[str, object])`, `render_replay(run_root: Path, output_dir: Path, ffmpeg: str, ffprobe: str)`, and `main(argv: Sequence[str] | None = None) -> int`.
- CLI: `python -m mm_sonic.replay_video --run-root RUN --output-dir OUT [--validate-only]`.

- [ ] **Step 1: Write failing contract tests**

Create `tests/python/test_sonic_replay_video.py` with synthetic fixtures and these concrete cases:

```python
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.replay_video import (
    ReplayState,
    load_state_stream,
    validate_probe,
)


class ReplayStateContractTests(unittest.TestCase):
    def test_loads_exact_nested_state_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.jsonl"
            rows = [
                {"step": 4, "sim_time_s": 0.02,
                 "state": {"qpos": [0.0] * 50, "qvel": [0.0] * 49}},
                {"step": 8, "sim_time_s": 0.04,
                 "state": {"qpos": [1.0] * 50, "qvel": [2.0] * 49}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
            states = load_state_stream(path, expected_count=2, nq=50, nv=49)
        self.assertEqual([state.step for state in states], [4, 8])
        self.assertEqual([state.sim_time_s for state in states], [0.02, 0.04])
        np.testing.assert_array_equal(states[1].qpos, np.ones(50))

    def test_rejects_noncontiguous_step_or_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.jsonl"
            path.write_text(json.dumps({
                "step": 8,
                "sim_time_s": 0.02,
                "state": {"qpos": [0.0] * 50, "qvel": [0.0] * 49},
            }) + "\n", "utf-8")
            with self.assertRaisesRegex(ContractError, "step sequence"):
                load_state_stream(path, expected_count=1, nq=50, nv=49)

    def test_rejects_nonfinite_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.jsonl"
            path.write_text(json.dumps({
                "step": 4,
                "sim_time_s": 0.02,
                "state": {"qpos": [float("nan")] + [0.0] * 49,
                          "qvel": [0.0] * 49},
            }) + "\n", "utf-8")
            with self.assertRaisesRegex(ContractError, "finite"):
                load_state_stream(path, expected_count=1, nq=50, nv=49)

    def test_probe_requires_exact_media_contract(self) -> None:
        valid = {"streams": [{"codec_name": "h264", "width": 1280,
                 "height": 720, "pix_fmt": "yuv420p", "avg_frame_rate": "50/1",
                 "nb_frames": "600"}], "format": {"duration": "12.000000"}}
        validate_probe(valid)
        invalid = json.loads(json.dumps(valid))
        invalid["streams"][0]["nb_frames"] = "599"
        with self.assertRaisesRegex(ContractError, "600 frames"):
            validate_probe(invalid)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 sonic/.venv/bin/python -m unittest \
  tests.python.test_sonic_replay_video -v
```

Expected: failure because `mm_sonic.replay_video` does not exist.

- [ ] **Step 3: Implement validation, rendering, encoding, and metadata**

Create `sonic/python/mm_sonic/replay_video.py` with the following implementation. The worker may factor private helpers differently only when the focused tests and all listed invariants remain exact:

```python
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
```

- [ ] **Step 4: Run focused tests and input-only validation**

Run:

```bash
PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 sonic/.venv/bin/python -m unittest \
  tests.python.test_sonic_replay_video -v
MUJOCO_GL=egl PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 sonic/.venv/bin/python \
  -m mm_sonic.replay_video \
  --run-root sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/stage-b/stage-b-20260717T190656038623Z-b746af3f \
  --output-dir sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/video-evidence \
  --validate-only
```

Expected: all focused tests pass; validation prints 600 states, `nq=50`, `nv=49`, and inventory verification true without creating output files.

- [ ] **Step 5: Commit the renderer slice**

```bash
git add sonic/python/mm_sonic/replay_video.py tests/python/test_sonic_replay_video.py
git commit -m "feat(sonic): render sealed stage b replay evidence"
```

### Task 2: Render and independently qualify the companion evidence

**Files:**
- Generate (ignored): `sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/video-evidence/g1-sonic-stage-b-r13-replay.mp4`
- Generate (ignored): `sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/video-evidence/g1-sonic-stage-b-r13-montage.png`
- Generate (ignored): `sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/video-evidence/g1-sonic-stage-b-r13-video-metadata.json`
- Copy (ignored): `sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/video-evidence/render_stage_b_replay.py`

**Interfaces:**
- Consumes: committed `mm_sonic.replay_video` and the frozen r13 source run.
- Produces: user-viewable video, montage, exact renderer copy, and machine-checkable provenance metadata.

- [ ] **Step 1: Record pre-render inventory and source hashes**

Run a protected Python command that calls `verify_run_inventory(run_root)`, SHA-256 hashes `inventory.json`, `manifest.json`, `stage-b-evidence.json`, `scene/gear_scene.xml`, `scene/gear_robot.xml`, and `dynamic/stream/scored-sim-logs/state.jsonl`, and saves the values outside the run under `/tmp/g1-sonic-r13-pre-render-hashes.json`.

Expected: verifier returns true and the inventory hash is `79d623a86993385f5ec88d90b7fa9605b6ec43b8d02906ddc98a7e8cbc6c27b7`.

- [ ] **Step 2: Produce the replay artifacts**

Run:

```bash
MUJOCO_GL=egl PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 sonic/.venv/bin/python \
  -m mm_sonic.replay_video \
  --run-root sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/stage-b/stage-b-20260717T190656038623Z-b746af3f \
  --output-dir sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/video-evidence
cp sonic/python/mm_sonic/replay_video.py \
  sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/video-evidence/render_stage_b_replay.py
```

Expected: the four specified companion files exist outside the sealed run root.

- [ ] **Step 3: Run independent media and provenance gates**

Run `ffprobe -v error -show_streams -show_format -of json` on the MP4 and independently assert codec `h264`, pixel format `yuv420p`, 1280×720, `50/1`, 600 frames, and duration 12.0 seconds. Recompute every source and output SHA-256 and compare to metadata. Compare the source hashes to `/tmp/g1-sonic-r13-pre-render-hashes.json` and call `verify_run_inventory` again.

Expected: every assertion passes and the sealed source hashes are byte-identical before and after.

- [ ] **Step 4: Inspect representative visuals**

Open the montage, then extract frames 0, 299, and 599 to `/tmp/g1-sonic-r13-inspection/` with this command and inspect each image:

```bash
mkdir -p /tmp/g1-sonic-r13-inspection
ffmpeg -v error -y \
  -i sonic/runs/stage-b-official-sonic-4c97fc2-clean-final-r13-20260717/video-evidence/g1-sonic-stage-b-r13-replay.mp4 \
  -vf "select='eq(n,0)+eq(n,299)+eq(n,599)'" -vsync 0 \
  /tmp/g1-sonic-r13-inspection/frame-%02d.png
```

Confirm the robot and terrain are visible in both views, pose changes are plausible, labels are legible, the replay disclaimer is present, and the final frame is not blank or corrupted.

Expected: visual inspection passes with no clipping that hides the robot or statistics.

- [ ] **Step 5: Run regression and cleanup gates**

Run:

```bash
PYTHONPATH=sonic/python PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -B -W error -m unittest \
  tests.python.test_sonic_artifacts tests.python.test_sonic_cli \
  tests.python.test_sonic_commands tests.python.test_sonic_coordinator \
  tests.python.test_sonic_external tests.python.test_sonic_gated_sim \
  tests.python.test_sonic_joints tests.python.test_sonic_metrics \
  tests.python.test_sonic_process tests.python.test_sonic_reference \
  tests.python.test_sonic_replay_video tests.python.test_sonic_resample \
  tests.python.test_sonic_runtime_parity tests.python.test_sonic_scene \
  tests.python.test_sonic_schema tests.python.test_sonic_timeline \
  tests.python.test_sonic_timing tests.python.test_sonic_transform \
  tests.python.test_sonic_zmq_v1
pgrep -af 'mm_sonic.replay_video|ffmpeg.*g1-sonic-stage-b-r13' || true
git status --short --branch
```

Expected: the repository suite passes, no render process remains, and only intentional tracked renderer/test commits plus ignored evidence artifacts exist.
