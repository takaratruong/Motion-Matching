"""Strict loading and passive MuJoCo playback for contact-oracle routes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
import time
from typing import Mapping

import numpy as np

from .joints import ContractError
from .operator_x11 import X11KeyStateProvider
from .torch_contact_oracle_rollout import (
    ORACLE_ARRAY_SHAPES,
    oracle_arrays_sha256,
)


_INTEGER_ARRAYS = frozenset(
    {
        "command_segment_index",
        "selected_source_frame",
        "selected_action_index",
        "planned_horizon",
        "plan_time_ns",
    }
)


@dataclass(frozen=True)
class SavedOracleRoute:
    route_name: str
    arrays: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, object]
    arrays_sha256: str

    @property
    def frame_count(self) -> int:
        return int(self.arrays["qpos"].shape[0])


def load_oracle_route(path: str | Path) -> SavedOracleRoute:
    """Load and authenticate one pickle-free oracle route directory."""

    route_dir = Path(path).resolve()
    if not route_dir.is_dir():
        raise ContractError(f"contact oracle route directory is missing: {path}")
    try:
        diagnostics = json.loads(
            (route_dir / "diagnostics.json").read_text(encoding="utf-8")
        )
    except Exception as error:
        raise ContractError("contact oracle diagnostics are invalid") from error
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("schema") != "g1-contact-space-oracle-route/v1"
        or diagnostics.get("route") != route_dir.name
        or not isinstance(diagnostics.get("arrays_sha256"), str)
        or len(diagnostics["arrays_sha256"]) != 64
    ):
        raise ContractError("contact oracle route identity is invalid")
    try:
        with np.load(route_dir / "rollout.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(ORACLE_ARRAY_SHAPES):
                raise ContractError("contact oracle saved array inventory is invalid")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("contact oracle rollout archive is invalid") from error
    frame_count = arrays["qpos"].shape[0]
    if frame_count < 1:
        raise ContractError("contact oracle saved route has no frames")
    for name, tail in ORACLE_ARRAY_SHAPES.items():
        array = arrays[name]
        if array.shape != (frame_count, *tail):
            raise ContractError(f"contact oracle saved {name} shape is invalid")
        if name == "selected_clip_path":
            if array.dtype.kind not in "US":
                raise ContractError("contact oracle saved clip paths are invalid")
        elif name in _INTEGER_ARRAYS:
            if array.dtype.kind not in "iu":
                raise ContractError(f"contact oracle saved {name} dtype is invalid")
        elif array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ContractError(f"contact oracle saved {name} values are invalid")
        array.setflags(write=False)
    observed = oracle_arrays_sha256(arrays)
    if observed != diagnostics["arrays_sha256"]:
        raise ContractError("contact oracle saved array hash mismatch")
    return SavedOracleRoute(
        route_name=route_dir.name,
        arrays=MappingProxyType(arrays),
        diagnostics=MappingProxyType(dict(diagnostics)),
        arrays_sha256=observed,
    )


def apply_oracle_frame(
    mujoco_module,
    model,
    data,
    saved: SavedOracleRoute,
    frame: int,
) -> str:
    """Apply one exact saved qpos and return the diagnostic overlay text."""

    if (
        not isinstance(saved, SavedOracleRoute)
        or type(frame) is not int
        or not 0 <= frame < saved.frame_count
        or getattr(model, "nq", None) != 36
        or np.asarray(data.qpos).shape != (36,)
    ):
        raise ContractError("contact oracle viewer frame contract is invalid")
    qpos = saved.arrays["qpos"][frame]
    if not np.isfinite(qpos).all():
        raise ContractError("contact oracle viewer qpos is non-finite")
    data.qpos[:] = qpos
    if hasattr(data, "qvel"):
        data.qvel[:] = 0.0
    mujoco_module.mj_forward(model, data)
    arrays = saved.arrays
    return (
        f"route={saved.route_name}  frame={frame + 1}/{saved.frame_count}\n"
        f"source={arrays['selected_clip_path'][frame]}:"
        f"{int(arrays['selected_source_frame'][frame])}  "
        f"action={int(arrays['selected_action_index'][frame])}  "
        f"horizon={int(arrays['planned_horizon'][frame])}\n"
        f"cost={float(arrays['plan_total_cost'][frame]):.3f}  "
        f"stance_error={float(arrays['stance_error_m'][frame]):.4f}  "
        f"landing_error={float(arrays['landing_error_m'][frame]):.4f}  "
        f"swing_clearance={float(arrays['minimum_swing_clearance_m'][frame]):.4f}"
    )


def run_viewer(
    *,
    route_dir: str | Path,
    dataset: str | Path,
    config: str | Path,
    g1_xml: str | Path,
    device: str,
) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise ContractError("contact oracle viewer requires mujoco viewer") from error
    from .torch_contact_oracle_search import load_contact_oracle_config
    from .torch_contact_segment_rollout import (
        load_contact_segment_config,
        resolve_contact_segment_config,
    )
    from .torch_terrain_live_viewer import build_kinematic_scene

    saved = load_oracle_route(route_dir)
    oracle = load_contact_oracle_config(config)
    terrain = load_contact_segment_config(oracle.terrain_config)
    resolved = resolve_contact_segment_config(dataset, terrain, device=device)
    model, data = build_kinematic_scene(g1_xml, resolved)
    frame = 0
    paused = False
    previous = frozenset()
    provider = None
    try:
        apply_oracle_frame(mujoco, model, data, saved, frame)
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.distance = 3.2
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            provider = X11KeyStateProvider()
            next_tick = time.perf_counter()
            print(
                "CONTACT ORACLE PLAYBACK: Space pause/resume, arrows step "
                "while paused, Backspace rewind, X exit.",
                flush=True,
            )
            while viewer.is_running():
                levels = provider.sample()
                pressed = levels.pressed

                def edge(key: str) -> bool:
                    return key in pressed and key not in previous

                if edge("X"):
                    break
                if edge("SPACE"):
                    paused = not paused
                if edge("BACKSPACE"):
                    frame = 0
                    paused = True
                if paused and edge("LEFT"):
                    frame = max(0, frame - 1)
                if paused and edge("RIGHT"):
                    frame = min(saved.frame_count - 1, frame + 1)
                overlay = apply_oracle_frame(
                    mujoco, model, data, saved, frame
                )
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                viewer.set_texts(
                    (
                        None,
                        None,
                        "KINEMATIC ORACLE / NO PHYSICS\n"
                        "Space pause  arrows step  Backspace rewind  X exit",
                        overlay,
                    )
                )
                viewer.sync()
                previous = pressed
                if not paused:
                    if frame + 1 < saved.frame_count:
                        frame += 1
                    else:
                        paused = True
                next_tick += 0.02
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -0.02:
                    next_tick = time.perf_counter()
    finally:
        if provider is not None:
            provider.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_viewer(
        route_dir=args.route_dir,
        dataset=args.dataset,
        config=args.config,
        g1_xml=args.g1_xml,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
