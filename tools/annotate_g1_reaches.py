#!/usr/bin/env python3
import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resources.g1_interaction_builder.schema import G1_SKELETON
from resources.g1_reach_builder.annotations import (
    create_annotation_document,
    load_annotations,
    update_annotation,
    write_annotations_atomic,
)
from resources.g1_reach_builder.review import read_review_corpus
from resources.g1_terrain_builder.kinematics import forward_local_hierarchy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Confirm outbound grab frames in a flat G1 skeleton viewer"
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def _world_sources(corpus):
    result = {}
    for source, sequence_id in enumerate(corpus.sequence_ids):
        start = int(corpus.range_starts[source])
        stop = int(corpus.range_stops[source])
        world, _ = forward_local_hierarchy(
            corpus.positions[start:stop].astype(np.float64),
            corpus.rotations[start:stop].astype(np.float64),
            G1_SKELETON.parents,
        )
        result[sequence_id] = world
    return result


def _segments(joints):
    return tuple(
        (joints[int(parent)], joints[bone])
        for bone, parent in enumerate(G1_SKELETON.parents)
        if parent >= 0
    )


def run(args) -> int:
    corpus = read_review_corpus(args.review)
    document = (
        load_annotations(args.annotations, args.review)
        if args.annotations.exists()
        else create_annotation_document(args.review)
    )
    if not args.annotations.exists():
        write_annotations_atomic(args.annotations, document)
    if not document.annotations:
        raise ValueError("reach review has no proposals to annotate")

    worlds = _world_sources(corpus)
    first = document.annotations[0]
    if args.smoke_test:
        segments = _segments(worlds[first.sequence_id][first.grab_frame])
        if len(segments) != len(G1_SKELETON.names) - 1:
            raise RuntimeError("flat annotation skeleton is incomplete")
        return 0

    import tkinter as tk

    root = tk.Tk()
    root.title("G1 outbound reach annotation")
    canvas = tk.Canvas(root, width=1280, height=800, bg="#101318")
    canvas.pack(fill="both", expand=True)
    wrist = G1_SKELETON.names.index("LeftWrist")
    speeds = {
        name: np.linalg.norm(
            np.gradient(world[:, wrist], 1.0 / corpus.fps, axis=0), axis=1
        )
        for name, world in worlds.items()
    }
    state = {
        "index": 0,
        "frame": first.departure_frame,
        "departure": first.departure_frame,
        "grab": first.grab_frame,
        "playing": False,
        "message": "",
    }

    def current():
        return document.annotations[state["index"]]

    def load_current() -> None:
        annotation = current()
        state["frame"] = annotation.departure_frame
        state["departure"] = annotation.departure_frame
        state["grab"] = annotation.grab_frame
        state["message"] = ""

    def project(point, *, side=False):
        horizontal = point[2] if side else point[0]
        center = 930 if side else 330
        return center + 230 * horizontal, 610 - 230 * point[1]

    def draw_view(joints, *, side=False) -> None:
        for start, stop in _segments(joints):
            x0, y0 = project(start, side=side)
            x1, y1 = project(stop, side=side)
            canvas.create_line(x0, y0, x1, y1, fill="#2997ff", width=4)
        x, y = project(joints[wrist], side=side)
        canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#ff453a", outline="")

    def draw_timeline(annotation) -> None:
        values = speeds[annotation.sequence_id]
        maximum = max(float(np.max(values)), 1e-6)
        left, right, top, bottom = 60, 1220, 665, 750
        points = []
        for frame in range(0, len(values), max(1, len(values) // 600)):
            x = left + (right - left) * frame / max(1, len(values) - 1)
            y = bottom - (bottom - top) * float(values[frame]) / maximum
            points.extend((x, y))
        if len(points) >= 4:
            canvas.create_line(*points, fill="#8e8e93", width=2)
        for frame, color in (
            (state["departure"], "#30d158"),
            (state["grab"], "#ff453a"),
            (state["frame"], "#64d2ff"),
        ):
            x = left + (right - left) * frame / max(1, len(values) - 1)
            canvas.create_line(x, top, x, bottom, fill=color, width=3)

    def redraw() -> None:
        canvas.delete("all")
        annotation = current()
        world = worlds[annotation.sequence_id]
        frame = int(np.clip(state["frame"], 0, len(world) - 1))
        canvas.create_text(
            30,
            25,
            anchor="nw",
            fill="white",
            font=("Sans", 18, "bold"),
            text=(
                f"{state['index'] + 1}/{len(document.annotations)}  "
                f"{annotation.sequence_id}  {annotation.status.upper()}\n"
                f"frame {frame}   departure {state['departure']}   "
                f"grab {state['grab']}   {state['message']}"
            ),
        )
        canvas.create_text(330, 95, text="FRONT", fill="#aeaeb2", font=("Sans", 14))
        canvas.create_text(930, 95, text="SIDE", fill="#aeaeb2", font=("Sans", 14))
        draw_view(world[frame], side=False)
        draw_view(world[frame], side=True)
        draw_timeline(annotation)
        canvas.create_text(
            640,
            780,
            fill="#d1d1d6",
            font=("Sans", 12),
            text=(
                "Space play | J/L ±1 | Shift+J/L ±10 | [/] proposal | "
                "1 departure | 2 grab | A accept | R reject | S save | Esc close"
            ),
        )

    def toggle_play():
        state["playing"] = not state["playing"]

    def scrub(delta: int):
        stop = len(worlds[current().sequence_id])
        state["frame"] = int(np.clip(state["frame"] + delta, 0, stop - 1))
        redraw()

    def previous_proposal():
        state["index"] = (state["index"] - 1) % len(document.annotations)
        load_current()
        redraw()

    def next_proposal():
        state["index"] = (state["index"] + 1) % len(document.annotations)
        load_current()
        redraw()

    def set_departure():
        state["departure"] = min(state["frame"], state["grab"])
        redraw()

    def set_grab():
        state["grab"] = max(state["frame"], state["departure"])
        redraw()

    def replace_current(status: str) -> bool:
        nonlocal document
        try:
            updated = update_annotation(
                corpus,
                current(),
                departure_frame=state["departure"],
                grab_frame=state["grab"],
                status=status,
                note="",
            )
        except ValueError as error:
            state["message"] = str(error)
            redraw()
            return False
        values = list(document.annotations)
        values[state["index"]] = updated
        document = replace(document, annotations=tuple(values))
        write_annotations_atomic(args.annotations, document)
        return True

    def accept_current():
        if replace_current("accepted"):
            next_proposal()

    def reject_current():
        if replace_current("rejected"):
            next_proposal()

    def save():
        write_annotations_atomic(args.annotations, document)
        state["message"] = "saved"
        redraw()

    def escape():
        write_annotations_atomic(args.annotations, document)
        root.destroy()

    def on_key(event) -> None:
        actions = {
            "space": toggle_play,
            "j": lambda: scrub(-1),
            "l": lambda: scrub(1),
            "J": lambda: scrub(-10),
            "L": lambda: scrub(10),
            "bracketleft": previous_proposal,
            "bracketright": next_proposal,
            "1": set_departure,
            "2": set_grab,
            "a": accept_current,
            "r": reject_current,
            "s": save,
            "Escape": escape,
        }
        action = actions.get(event.keysym)
        if action is not None:
            action()

    def tick() -> None:
        if state["playing"]:
            loop_stop = min(
                len(worlds[current().sequence_id]) - 1,
                state["grab"] + 15,
            )
            state["frame"] += 1
            if state["frame"] > loop_stop:
                state["frame"] = state["departure"]
            redraw()
        root.after(40, tick)

    root.bind("<KeyPress>", on_key)
    root.protocol("WM_DELETE_WINDOW", escape)
    redraw()
    root.after(40, tick)
    root.mainloop()
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
