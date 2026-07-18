"""Render a fast visual checkpoint from a rolling manual SONIC run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

import numpy as np

from .joints import ContractError
from .replay_video import (
    FPS,
    FRAME_COUNT,
    HEIGHT,
    WIDTH,
    _ensure_offscreen_framebuffer,
    _finish_encoder,
    _mujoco,
    _run_checked,
    load_state_stream,
)


def render_manual_replay(run_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = run_root.expanduser().resolve(strict=True)
    out = output_dir.expanduser().resolve(strict=False)
    scene = root / "scene/gear_scene.xml"
    state_log = root / "scored-sim-logs/state.jsonl"
    if not scene.is_file() or not state_log.is_file():
        raise ContractError("manual run is missing its scene or scored state log")

    mujoco = _mujoco()
    model = mujoco.MjModel.from_xml_path(str(scene))
    states = load_state_stream(
        state_log,
        expected_count=FRAME_COUNT,
        nq=model.nq,
        nv=model.nv,
    )
    out.mkdir(parents=True, exist_ok=True)
    video = out / "g1-sonic-rolling-replay.mp4"
    montage = out / "g1-sonic-rolling-montage.png"
    temporary = out / f"g1-sonic-rolling-replay.tmp-{os.getpid()}.mp4"
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    label = (
        "drawbox=x=0:y=0:w=iw:h=72:color=black@0.72:t=fill,"
        f"drawtext=fontfile={font}:text='ROLLING MM -> UNMODIFIED SONIC | POST-RUN STATE REPLAY':"
        "fontcolor=yellow:fontsize=21:x=20:y=10,"
        "drawtext=fontfile=" + str(font) + ":"
        "text='frame %{eif\\:n+1\\:d}/600 | t=%{pts\\:hms} | visual checkpoint (not hardware)':"
        "fontcolor=white:fontsize=16:x=20:y=42,"
        "drawbox=x=639:y=72:w=2:h=648:color=white@0.5:t=fill"
    )
    encode = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s:v", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "-", "-vf", label, "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]

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
            encode,
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
        _finish_encoder(encoder)
        encoder = None
        os.replace(temporary, video)
        select = "select='" + "+".join(
            f"eq(n,{frame})" for frame in (0, 119, 239, 359, 479, 599)
        ) + "',scale=640:360,tile=3x2"
        _run_checked([
            "ffmpeg", "-v", "error", "-y", "-i", str(video),
            "-vf", select, "-frames:v", "1", str(montage),
        ])
        return video, montage
    finally:
        if left is not None:
            left.close()
        if right is not None:
            right.close()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    video, montage = render_manual_replay(args.run_root, args.output_dir)
    print(video)
    print(montage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
