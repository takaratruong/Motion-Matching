"""Authenticated authored GRAIL slope playback (not learned inference)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import threading
from types import MappingProxyType, SimpleNamespace
from typing import Mapping
import time

import numpy as np

from resources.build_g1_terrain_database import _assemble_authored_slope_source
from resources.g1_terrain_builder.resample import (
    resample_quaternions_wxyz,
    resample_vectors,
)
from resources.g1_terrain_builder.sources import load_authenticated_grail_slope


AUTHORED_PLAYBACK_LABEL = "AUTHORED SOURCE PLAYBACK (NOT LEARNED)"
AUTHORED_SLOPE_SUPPORT_CALIBRATION_M = 0.012000000104308128
AUTHORED_SLOPE_CLIP_ID = "terrain_slopes__slope_000__000"
DEFAULT_SLOPE_ROOT = Path("/home/ubuntu/datasets/GRAIL/data/slope")
DEFAULT_SLOPE_ROBOT = DEFAULT_SLOPE_ROOT / "robot" / f"{AUTHORED_SLOPE_CLIP_ID}.pkl"
DEFAULT_SLOPE_USD = (
    DEFAULT_SLOPE_ROOT / "object_usd" / f"{AUTHORED_SLOPE_CLIP_ID}.usd"
)
DEFAULT_SLOPE_RECON = DEFAULT_SLOPE_ROOT / "recon" / f"{AUTHORED_SLOPE_CLIP_ID}.pkl"
DEFAULT_SLOPE_METADATA = DEFAULT_SLOPE_ROOT / "meta" / f"{AUTHORED_SLOPE_CLIP_ID}.pkl"
DEFAULT_G1_XML = Path(
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)


@dataclass(frozen=True)
class AuthoredSlopeBundle:
    clip_id: str
    fps: float
    qpos: np.ndarray
    terrain_vertices_native: np.ndarray
    terrain_faces: np.ndarray
    exterior_height_native_m: float
    support_calibration_m: float
    hashes: Mapping[str, str]
    g1_xml_path: Path


def _frozen_array(values: object, dtype: np.dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


class AuthoredPlaybackState:
    def __init__(self, qpos: np.ndarray) -> None:
        values = np.array(qpos, dtype=np.float32, copy=True)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 36:
            raise ValueError("authored playback qpos must have shape [rows, 36]")
        if not np.isfinite(values).all():
            raise ValueError("authored playback qpos must be finite")
        values.setflags(write=False)
        self._qpos = values
        self.row = 0

    def tick(self, *, w_down: bool) -> int:
        if type(w_down) is not bool:
            raise TypeError("w_down must be a boolean key level")
        if w_down and self.row < len(self._qpos) - 1:
            self.row += 1
        return self.row

    def current_qpos(self) -> np.ndarray:
        return self._qpos[self.row]


class AuthoredPlaybackKeys:
    """Thread-safe key levels shared by pynput and the 60 Hz render loop."""

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


def holden_vertices_to_native(vertices: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,) or not len(values):
        raise ValueError("Holden terrain vertices must have shape [N, 3]")
    if not np.isfinite(values).all():
        raise ValueError("Holden terrain vertices must be finite")
    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def _admitted_native_qpos(robot_path: Path) -> np.ndarray:
    source = load_authenticated_grail_slope(str(robot_path))
    if source.fps != 25.0 or source.qpos.shape != (250, 36):
        raise ValueError("authenticated authored slope source dimensions changed")
    provisional = np.empty((598, 36), dtype=np.float32)
    provisional[:, :3] = resample_vectors(source.qpos[:, :3], 25.0, 60.0)
    provisional[:, 3:7] = resample_quaternions_wxyz(
        source.qpos[:, 3:7], 25.0, 60.0
    )
    provisional[:, 7:] = resample_vectors(source.qpos[:, 7:], 25.0, 60.0)
    admitted = provisional[3:]
    if admitted.shape != (595, 36) or not np.isfinite(admitted).all():
        raise ValueError("authored slope admitted qpos rows changed")
    return _frozen_array(admitted, np.dtype(np.float32))


def load_authored_slope_bundle(
    *,
    slope_robot: Path = DEFAULT_SLOPE_ROBOT,
    slope_usd: Path = DEFAULT_SLOPE_USD,
    slope_recon: Path = DEFAULT_SLOPE_RECON,
    slope_metadata: Path = DEFAULT_SLOPE_METADATA,
    g1_xml: Path = DEFAULT_G1_XML,
) -> AuthoredSlopeBundle:
    """Authenticate and load the one approved slope pair for direct replay."""

    paths = tuple(
        Path(value).expanduser().resolve()
        for value in (slope_robot, slope_usd, slope_recon, slope_metadata, g1_xml)
    )
    robot, usd, recon, metadata, model = paths
    candidate = _assemble_authored_slope_source(
        SimpleNamespace(
            output_fps=60.0,
            slope_robot=str(robot),
            slope_usd=str(usd),
            slope_recon=str(recon),
            slope_metadata=str(metadata),
            g1_xml=str(model),
        )
    )
    receipt = candidate.provenance
    if (
        receipt.get("clip_id") != AUTHORED_SLOPE_CLIP_ID
        or receipt.get("provisional_output_frames") != 598
        or receipt.get("admitted_output_frames") != 595
        or receipt.get("rejected_output_ranges") != [[0, 3]]
        or receipt.get("support_calibration_applied_to") != "terrain-only"
        or receipt.get("support_calibration_m")
        != AUTHORED_SLOPE_SUPPORT_CALIBRATION_M
    ):
        raise ValueError("authored slope provenance receipt changed")
    terrain = candidate.terrain
    if (
        terrain.support_calibration_m != AUTHORED_SLOPE_SUPPORT_CALIBRATION_M
        or terrain.exterior_height != -AUTHORED_SLOPE_SUPPORT_CALIBRATION_M
    ):
        raise ValueError("authored slope terrain calibration changed")
    native_vertices = holden_vertices_to_native(terrain.vertices)
    qpos = _admitted_native_qpos(robot)
    return AuthoredSlopeBundle(
        clip_id=AUTHORED_SLOPE_CLIP_ID,
        fps=60.0,
        qpos=qpos,
        terrain_vertices_native=_frozen_array(
            native_vertices, np.dtype(np.float64)
        ),
        terrain_faces=_frozen_array(terrain.triangles, np.dtype(np.int32)),
        exterior_height_native_m=float(terrain.exterior_height),
        support_calibration_m=AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
        hashes=MappingProxyType(dict(receipt["hashes"])),
        g1_xml_path=model,
    )


def build_playback_model(bundle: AuthoredSlopeBundle):
    """Compose the canonical native G1 with the authenticated ramp and flat."""

    import mujoco

    if not isinstance(bundle, AuthoredSlopeBundle):
        raise TypeError("playback model requires an authored slope bundle")
    spec = mujoco.MjSpec.from_file(str(bundle.g1_xml_path))
    spec.add_mesh(
        name="authored_slope_exact_mesh",
        uservert=bundle.terrain_vertices_native.ravel(),
        userface=bundle.terrain_faces.ravel(),
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    spec.worldbody.add_geom(
        name="authored_slope_exterior_flat",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, bundle.exterior_height_native_m),
        size=(10.0, 10.0, 0.05),
        contype=0,
        conaffinity=0,
        rgba=(0.18, 0.28, 0.18, 1.0),
    )
    spec.worldbody.add_geom(
        name="authored_slope_exact_mesh",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="authored_slope_exact_mesh",
        contype=0,
        conaffinity=0,
        rgba=(0.34, 0.52, 0.22, 1.0),
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
    *, row: int, frame_count: int, focused: bool, w_down: bool
) -> tuple[str, str]:
    if type(row) is not int or type(frame_count) is not int:
        raise TypeError("overlay row and frame count must be integers")
    if frame_count < 1 or not 0 <= row < frame_count:
        raise ValueError("overlay row is outside the authored playback")
    if type(focused) is not bool or type(w_down) is not bool:
        raise TypeError("overlay focus and W state must be booleans")
    mode = (
        "TERMINAL HOLD"
        if row == frame_count - 1
        else "ADVANCING (W held)"
        if focused and w_down
        else "PAUSED (W released)"
        if focused
        else "PAUSED (viewer not focused)"
    )
    return (
        AUTHORED_PLAYBACK_LABEL,
        f"row {row + 1}/{frame_count} @ 60 Hz | {mode}\n"
        "Hold W: advance one authored row/tick | release: exact pause | X: exit\n"
        f"{AUTHORED_SLOPE_SUPPORT_CALIBRATION_M * 1000.0:.3f} mm "
        "terrain-only calibration | exact reconstructed GRAIL ramp",
    )


def authored_playback_smoke(bundle: AuthoredSlopeBundle) -> dict[str, object]:
    """Prove key-level pause and end-of-route behavior without opening a window."""

    if not isinstance(bundle, AuthoredSlopeBundle):
        raise TypeError("authored playback smoke requires its authenticated bundle")
    state = AuthoredPlaybackState(bundle.qpos)
    state.tick(w_down=True)
    held = state.current_qpos().tobytes()
    for _ in range(5):
        state.tick(w_down=False)
    release_pause_bitwise = state.current_qpos().tobytes() == held
    advance_ticks = 1
    while state.row < len(bundle.qpos) - 1:
        state.tick(w_down=True)
        advance_ticks += 1
    terminal = state.current_qpos().tobytes()
    for _ in range(5):
        state.tick(w_down=True)
    terminal_hold_bitwise = state.current_qpos().tobytes() == terminal
    return {
        "label": AUTHORED_PLAYBACK_LABEL,
        "status": (
            "accepted"
            if release_pause_bitwise and terminal_hold_bitwise
            else "rejected"
        ),
        "clip_id": bundle.clip_id,
        "frame_count": len(bundle.qpos),
        "advance_ticks": advance_ticks,
        "release_pause_bitwise": release_pause_bitwise,
        "terminal_hold_bitwise": terminal_hold_bitwise,
        "terminal_row": state.row,
        "support_calibration_m": bundle.support_calibration_m,
        "hashes": dict(bundle.hashes),
    }


def load_pynput_keyboard():
    """Load the viewer's existing pynput backend across the split repo envs."""

    try:
        return importlib.import_module("pynput.keyboard")
    except ModuleNotFoundError as error:
        if error.name not in ("pynput", "pynput.keyboard"):
            raise
    sonic_root = Path(__file__).resolve().parents[2]
    candidates = tuple(
        sorted(
            (sonic_root / ".torch-mm-venv" / "lib").glob(
                "python*/site-packages"
            )
        )
    )
    if len(candidates) != 1:
        raise ModuleNotFoundError(
            "pynput is unavailable and the pinned terrain-viewer environment "
            "does not contain exactly one site-packages directory"
        )
    path = str(candidates[0])
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("pynput.keyboard")


def enforce_solid_rendering(flags: object) -> None:
    """Undo MuJoCo's built-in W wireframe hotkey for this W-driven viewer."""

    import mujoco

    index = int(mujoco.mjtRndFlag.mjRND_WIREFRAME)
    try:
        flags[index] = 0  # type: ignore[index]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("MuJoCo render flags cannot disable wireframe") from error


def run_interactive(bundle: AuthoredSlopeBundle) -> int:
    """Open the exact-surface MuJoCo viewer and gate authored rows on held W."""

    if not os.environ.get("DISPLAY"):
        raise RuntimeError("interactive authored playback requires DISPLAY")
    import mujoco
    import mujoco.viewer

    keyboard = load_pynput_keyboard()

    model = build_playback_model(bundle)
    data = mujoco.MjData(model)
    state = AuthoredPlaybackState(bundle.qpos)
    data.qpos[:] = state.current_qpos()
    mujoco.mj_forward(model, data)
    print(AUTHORED_PLAYBACK_LABEL, flush=True)
    print(
        "Hold W to advance the authenticated 60 Hz route; release W to pause "
        "the exact row; X exits.",
        flush=True,
    )
    print(
        f"clip={bundle.clip_id} rows={len(bundle.qpos)} "
        f"support_calibration_m={bundle.support_calibration_m:.18g}",
        flush=True,
    )
    keys = AuthoredPlaybackKeys()

    def on_press(key: object) -> bool | None:
        escape = key == keyboard.Key.esc
        keys.press(getattr(key, "char", None), escape=escape)
        if keys.snapshot()[1]:
            return False
        return None

    def on_release(key: object) -> None:
        keys.release(getattr(key, "char", None))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    period = 1.0 / bundle.fps
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
                data.qpos[:] = state.current_qpos()
                data.time = state.row / bundle.fps
                mujoco.mj_forward(model, data)
                title, body = overlay_text(
                    row=state.row,
                    frame_count=len(bundle.qpos),
                    focused=True,
                    w_down=w_down,
                )
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                    enforce_solid_rendering(viewer.user_scn.flags)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slope-robot", type=Path, default=DEFAULT_SLOPE_ROBOT)
    parser.add_argument("--slope-usd", type=Path, default=DEFAULT_SLOPE_USD)
    parser.add_argument("--slope-recon", type=Path, default=DEFAULT_SLOPE_RECON)
    parser.add_argument(
        "--slope-metadata", type=Path, default=DEFAULT_SLOPE_METADATA
    )
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    parser.add_argument(
        "--headless-smoke",
        action="store_true",
        help="authenticate and prove W/release/terminal semantics without a window",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = load_authored_slope_bundle(
        slope_robot=arguments.slope_robot,
        slope_usd=arguments.slope_usd,
        slope_recon=arguments.slope_recon,
        slope_metadata=arguments.slope_metadata,
        g1_xml=arguments.g1_xml,
    )
    if arguments.headless_smoke:
        print(json.dumps(authored_playback_smoke(bundle), sort_keys=True))
        return 0
    return run_interactive(bundle)


if __name__ == "__main__":
    raise SystemExit(main())
