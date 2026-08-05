#!/usr/bin/env python3
"""Passive MuJoCo playback for an offline stair-pivot graph walk."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--grid-summary", type=Path)
    parser.add_argument("--playlist-metadata", type=Path)
    return parser


def _load_grid_overlay(
    summary_path: Path,
    playlist_path: Path,
    *,
    frame_count: int,
) -> dict[str, object]:
    try:
        summary = json.loads(summary_path.read_text("utf-8"))
        playlist = json.loads(playlist_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("grid viewer metadata is unreadable") from error
    lanes = summary.get("lanes")
    segments = playlist.get("segments")
    if (
        summary.get("schema") != "g1-horizontal-grid-coverage/v1"
        or summary.get("grid", {}).get("lane_count") != 11
        or not isinstance(lanes, list)
        or len(lanes) != 11
        or playlist.get("schema") != "g1-horizontal-grid-playlist/v1"
        or playlist.get("frame_count") != frame_count
        or not isinstance(segments, list)
    ):
        raise ContractError("grid viewer metadata contract is invalid")
    validated_lanes = []
    lane_ids = set()
    for expected_index, lane in enumerate(lanes):
        try:
            lane_id = str(lane["lane_id"])
            lane_index = int(lane["lane_index"])
            center_y = float(lane["center_y_m"])
            classification = str(lane["classification"])
            polyline = np.asarray(
                lane["path_polyline_matcher_xyz"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("grid viewer lane is invalid") from error
        if (
            not lane_id
            or lane_id in lane_ids
            or lane_index != expected_index
            or not math.isfinite(center_y)
            or classification not in ("full", "partial", "infeasible")
            or polyline.ndim != 2
            or polyline.shape[0] < 2
            or polyline.shape[1] != 3
            or not np.isfinite(polyline).all()
        ):
            raise ContractError("grid viewer lane is invalid")
        lane_ids.add(lane_id)
        validated_lanes.append(
            {
                **lane,
                "lane_id": lane_id,
                "lane_index": lane_index,
                "center_y_m": center_y,
                "classification": classification,
                "path_polyline_matcher_xyz": polyline,
            }
        )
    validated_segments = []
    previous_stop = 0
    for segment in segments:
        try:
            lane_id = str(segment["lane_id"])
            start, stop = (
                int(value) for value in segment["segment_frames"]
            )
            motion_start, motion_stop = (
                int(value) for value in segment["motion_frames"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("grid viewer segment is invalid") from error
        if (
            lane_id not in lane_ids
            or not 0 <= start <= motion_start < motion_stop <= stop
            or stop > frame_count
            or start != previous_stop
        ):
            raise ContractError("grid viewer segment is invalid")
        validated_segments.append(
            {
                **segment,
                "lane_id": lane_id,
                "segment_frames": [start, stop],
                "motion_frames": [motion_start, motion_stop],
            }
        )
        previous_stop = stop
    if validated_segments and previous_stop != frame_count:
        raise ContractError("grid viewer playlist does not cover all frames")
    return {
        "lanes": validated_lanes,
        "segments": validated_segments,
    }


def _grid_line_colors(
    overlay: dict[str, object],
    *,
    current_lane_id: str | None,
) -> tuple[tuple[float, float, float, float], ...]:
    palette = {
        "full": (0.1, 0.9, 0.2, 0.9),
        "partial": (1.0, 0.65, 0.0, 0.9),
        "infeasible": (0.9, 0.1, 0.1, 0.9),
    }
    return tuple(
        (
            (1.0, 1.0, 1.0, 1.0)
            if lane["lane_id"] == current_lane_id
            else palette[str(lane["classification"])]
        )
        for lane in overlay["lanes"]
    )


def _active_grid_segment(
    overlay: dict[str, object], frame: int
) -> dict[str, object] | None:
    for segment in overlay["segments"]:
        start, stop = segment["segment_frames"]
        if start <= frame < stop:
            return segment
    return None


def _set_grid_markers(
    mujoco_module,
    viewer,
    overlay: dict[str, object],
    *,
    current_lane_id: str | None,
) -> None:
    scene = viewer.user_scn
    segment_count = sum(
        len(lane["path_polyline_matcher_xyz"]) - 1
        for lane in overlay["lanes"]
    )
    if scene is None or scene.maxgeom < segment_count:
        raise ContractError("MuJoCo user scene cannot hold path grid")
    colors = _grid_line_colors(
        overlay, current_lane_id=current_lane_id
    )
    scene.ngeom = 0
    for lane, color in zip(overlay["lanes"], colors):
        polyline = lane["path_polyline_matcher_xyz"]
        for start, stop in zip(polyline[:-1], polyline[1:]):
            mujoco_module.mjv_connector(
                scene.geoms[scene.ngeom],
                mujoco_module.mjtGeom.mjGEOM_CAPSULE,
                0.008,
                start,
                stop,
            )
            scene.geoms[scene.ngeom].rgba[:] = np.asarray(
                color, dtype=np.float32
            )
            scene.ngeom += 1


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
    if (args.grid_summary is None) != (args.playlist_metadata is None):
        raise ContractError(
            "grid summary and playlist metadata must be provided together"
        )
    grid_overlay = (
        None
        if args.grid_summary is None
        else _load_grid_overlay(
            args.grid_summary,
            args.playlist_metadata,
            frame_count=len(arrays["joint_position"]),
        )
    )
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
                active = (
                    None
                    if grid_overlay is None
                    else _active_grid_segment(grid_overlay, frame)
                )
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                    if grid_overlay is not None:
                        _set_grid_markers(
                            mujoco,
                            viewer,
                            grid_overlay,
                            current_lane_id=(
                                None
                                if active is None
                                else str(active["lane_id"])
                            ),
                        )
                lane_text = ""
                if active is not None:
                    lane_text = (
                        f"\nlane={active['lane_id']} "
                        f"y={float(active['center_y_m']):+.1f}m "
                        f"{active['classification']}"
                    )
                viewer.set_texts(
                    (
                        None,
                        None,
                        "KINEMATIC STAIR PIVOT / NO PHYSICS\n"
                        "Space pause  arrows step  Backspace rewind  X exit",
                        f"frame={frame + 1}/{frame_count}{lane_text}",
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
