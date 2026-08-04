#!/usr/bin/env python3
"""Passive MuJoCo playback for an offline stair-pivot graph walk."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from mm_sonic.joints import ContractError
from mm_sonic.operator_x11 import X11KeyStateProvider
from mm_sonic.torch_g1_fk import target_state_qpos
from mm_sonic.torch_terrain_live_viewer import build_kinematic_scene
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _load_connector(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name], dtype=np.float64).copy()
                for name in (
                    "joint_position",
                    "root_position_world",
                    "root_orientation_world_wxyz",
                )
            }
    except Exception as error:
        raise ContractError("stair pivot connector archive is invalid") from error
    frame_count = len(arrays["joint_position"])
    if (
        frame_count < 1
        or arrays["joint_position"].shape != (frame_count, 29)
        or arrays["root_position_world"].shape != (frame_count, 3)
        or arrays["root_orientation_world_wxyz"].shape != (frame_count, 4)
        or not all(np.isfinite(value).all() for value in arrays.values())
        or np.any(
            np.abs(
                np.linalg.norm(
                    arrays["root_orientation_world_wxyz"],
                    axis=1,
                )
                - 1.0
            )
            > 1e-4
        )
    ):
        raise ContractError("stair pivot connector arrays are invalid")
    return arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connector", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--frames-per-second", type=float, default=50.0)
    parser.add_argument("--render-contact-sheet", type=Path)
    return parser


def _render_contact_sheet(
    *,
    model: object,
    data: object,
    arrays: dict[str, np.ndarray],
    output: Path,
) -> None:
    import mujoco

    frame_count = len(arrays["joint_position"])
    indices = sorted(
        {
            int(round(fraction * (frame_count - 1)))
            for fraction in np.linspace(0.0, 1.0, 6)
        }
    )
    width = 480
    height = 360
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    camera = mujoco.MjvCamera()
    roots = arrays["root_position_world"]
    camera.lookat[:] = (
        float(np.mean(roots[:, 0])),
        float(np.mean(roots[:, 1])),
        float(max(0.7, np.mean(roots[:, 2]) - 0.1)),
    )
    span = float(np.ptp(roots[:, :2], axis=0).max())
    camera.distance = max(2.8, 2.2 + span * 2.0)
    camera.azimuth = 135.0
    camera.elevation = -22.0
    canvas = Image.new("RGB", (width * 3, height * 2), "black")
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for panel, frame in enumerate(indices):
            data.qpos[:] = target_state_qpos(
                arrays["joint_position"][frame],
                arrays["root_position_world"][frame],
                arrays["root_orientation_world_wxyz"][frame],
            )
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 118, 25), fill=(0, 0, 0))
            draw.text(
                (8, 6),
                f"frame {frame + 1}/{frame_count}",
                fill=(255, 255, 255),
            )
            canvas.paste(
                image,
                ((panel % 3) * width, (panel // 3) * height),
            )
    finally:
        renderer.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    args = _parser().parse_args()
    if (
        not math.isfinite(args.frames_per_second)
        or args.frames_per_second <= 0.0
    ):
        raise ContractError("viewer frame rate must be positive")
    frame_delay = 1.0 / args.frames_per_second
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise ContractError("stair pivot viewer requires MuJoCo") from error
    arrays = _load_connector(args.connector)
    resolved = resolve_stair_config(
        args.dataset,
        load_experiment_config(args.config),
        device="cpu",
    )
    model, data = build_kinematic_scene(args.g1_xml, resolved)
    if args.render_contact_sheet is not None:
        _render_contact_sheet(
            model=model,
            data=data,
            arrays=arrays,
            output=args.render_contact_sheet,
        )
        print(f"wrote {args.render_contact_sheet}", flush=True)
        return 0
    frame_count = len(arrays["joint_position"])
    frame = 0
    paused = False
    previous = frozenset()
    provider = None
    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.distance = 3.2
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            provider = X11KeyStateProvider()
            next_tick = time.perf_counter()
            print(
                "STAIR PIVOT PLAYBACK: Space pause, arrows step, "
                "Backspace rewind, X exit.",
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
                    frame = min(frame_count - 1, frame + 1)
                data.qpos[:] = target_state_qpos(
                    arrays["joint_position"][frame],
                    arrays["root_position_world"][frame],
                    arrays["root_orientation_world_wxyz"][frame],
                )
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                viewer.set_texts(
                    (
                        None,
                        None,
                        "KINEMATIC STAIR PIVOT / NO PHYSICS\n"
                        "Space pause  arrows step  Backspace rewind  X exit",
                        f"frame={frame + 1}/{frame_count}",
                    )
                )
                viewer.sync()
                previous = pressed
                if not paused:
                    if frame + 1 < frame_count:
                        frame += 1
                    elif args.loop:
                        frame = 0
                    else:
                        paused = True
                next_tick += frame_delay
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -frame_delay:
                    next_tick = time.perf_counter()
    finally:
        if provider is not None:
            provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
