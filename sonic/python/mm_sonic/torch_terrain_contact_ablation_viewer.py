"""MuJoCo playback for saved terrain contact-composition previews."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .joints import ContractError
from .operator_x11 import X11KeyStateProvider


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"contact preview {name} is invalid")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SavedContactPreview:
    state_id: str
    route_name: str
    route_frame: int
    reason: str
    action_index: int
    accepted: bool
    contact_qpos: np.ndarray
    projected_qpos: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state_id, str)
            or len(self.state_id) != 64
            or not isinstance(self.route_name, str)
            or not self.route_name
            or type(self.route_frame) is not int
            or self.route_frame < 0
            or not isinstance(self.reason, str)
            or not self.reason
            or type(self.action_index) is not int
            or self.action_index < 0
            or type(self.accepted) is not bool
        ):
            raise ContractError("contact preview identity is invalid")
        contact = np.asarray(self.contact_qpos)
        projected = np.asarray(self.projected_qpos)
        if contact.ndim != 2 or contact.shape[1:] != (36,) or contact.shape[0] < 2:
            raise ContractError("contact preview qpos is invalid")
        object.__setattr__(
            self,
            "contact_qpos",
            _readonly(contact, contact.shape, "contact qpos"),
        )
        object.__setattr__(
            self,
            "projected_qpos",
            _readonly(projected, contact.shape, "projected qpos"),
        )

    @property
    def frame_count(self) -> int:
        return int(self.contact_qpos.shape[0])


def load_contact_preview_catalog(
    artifact_root: str | Path,
) -> tuple[SavedContactPreview, ...]:
    """Load the preferred successful action for every saved preview state."""

    root = Path(artifact_root).resolve()
    try:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    except Exception as error:
        raise ContractError("contact preview summary is invalid") from error
    if (
        not isinstance(summary, dict)
        or summary.get("schema") != "g1-terrain-contact-composition-ablation/v1"
        or not isinstance(summary.get("states"), list)
        or not isinstance(summary.get("deterministic_sha256"), str)
    ):
        raise ContractError("contact preview summary identity is invalid")
    identity = {
        key: value for key, value in summary.items() if key != "deterministic_sha256"
    }
    if hashlib.sha256(_canonical(identity)).hexdigest() != summary["deterministic_sha256"]:
        raise ContractError("contact preview summary authentication failed")

    output = []
    for record in summary["states"]:
        if not isinstance(record, dict):
            raise ContractError("contact preview state record is invalid")
        accepted = record.get("accepted_action_indices")
        best = record.get("best_action_index")
        if not isinstance(accepted, list):
            raise ContractError("contact preview accepted actions are invalid")
        action_index = accepted[0] if accepted else best
        if action_index is None:
            continue
        if type(action_index) is not int or action_index < 0:
            raise ContractError("contact preview action index is invalid")
        prefix = f"action_{action_index}"
        path = root / "states" / str(record.get("state_id")) / "contact-previews.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                contact = np.array(archive[f"{prefix}_contact_qpos"], copy=True)
                projected = np.array(archive[f"{prefix}_projected_qpos"], copy=True)
        except Exception as error:
            raise ContractError("contact preview archive is invalid") from error
        output.append(
            SavedContactPreview(
                state_id=str(record.get("state_id")),
                route_name=str(record.get("route_name")),
                route_frame=record.get("route_frame"),
                reason=str(record.get("reason")),
                action_index=action_index,
                accepted=action_index in accepted,
                contact_qpos=contact,
                projected_qpos=projected,
            )
        )
    if not output:
        raise ContractError("contact preview artifact has no successful actions")
    return tuple(output)


def apply_contact_preview_frame(
    mujoco_module,
    model,
    data,
    preview: SavedContactPreview,
    frame: int,
    *,
    projected: bool,
) -> str:
    """Apply one kinematic frame and return its exact diagnostic label."""

    if (
        not isinstance(preview, SavedContactPreview)
        or type(frame) is not int
        or not 0 <= frame < preview.frame_count
        or type(projected) is not bool
        or np.asarray(data.qpos).shape != (36,)
    ):
        raise ContractError("contact preview frame contract is invalid")
    qpos = preview.projected_qpos[frame] if projected else preview.contact_qpos[frame]
    data.qpos[:] = qpos
    if hasattr(data, "qvel"):
        data.qvel[:] = 0.0
    mujoco_module.mj_forward(model, data)
    stage = "PROJECTED" if projected else "UNPROJECTED"
    verdict = "accepted" if preview.accepted else "diagnostic-only"
    return (
        f"{stage} / {verdict}\n"
        f"route={preview.route_name} source_frame={preview.route_frame} "
        f"reason={preview.reason}\n"
        f"action={preview.action_index} frame={frame + 1}/{preview.frame_count}"
    )


def run_viewer(
    *,
    artifact_root: str | Path,
    dataset: str | Path,
    config: str | Path,
    g1_xml: str | Path,
    device: str,
) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise ContractError("contact preview viewer requires mujoco viewer") from error
    from .torch_terrain_live_viewer import build_kinematic_scene
    from .torch_terrain_rollout import load_experiment_config, resolve_stair_config

    catalog = load_contact_preview_catalog(artifact_root)
    resolved = resolve_stair_config(
        dataset, load_experiment_config(config), device=device
    )
    model, data = build_kinematic_scene(g1_xml, resolved)
    preview_index = 0
    frame = 0
    paused = False
    projected = True
    previous = frozenset()
    provider = None
    try:
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.distance = 3.2
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            provider = X11KeyStateProvider()
            next_tick = time.perf_counter()
            print(
                "CONTACT PREVIEW: Space pause, arrows choose/step, P toggles "
                "projection, Backspace rewinds, X exits.",
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
                if edge("P"):
                    projected = not projected
                if edge("BACKSPACE"):
                    frame = 0
                    paused = True
                if edge("UP"):
                    preview_index = (preview_index - 1) % len(catalog)
                    frame = 0
                if edge("DOWN"):
                    preview_index = (preview_index + 1) % len(catalog)
                    frame = 0
                preview = catalog[preview_index]
                if paused and edge("LEFT"):
                    frame = max(0, frame - 1)
                if paused and edge("RIGHT"):
                    frame = min(preview.frame_count - 1, frame + 1)
                overlay = apply_contact_preview_frame(
                    mujoco, model, data, preview, frame, projected=projected
                )
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                viewer.set_texts(
                    (
                        None,
                        None,
                        "KINEMATIC CONTACT PREVIEW / NO PHYSICS\n"
                        "P projected toggle  Up/Down state  Space pause",
                        overlay,
                    )
                )
                viewer.sync()
                previous = pressed
                if not paused:
                    frame = (frame + 1) % preview.frame_count
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
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_viewer(
        artifact_root=args.artifact_root,
        dataset=args.dataset,
        config=args.config,
        g1_xml=args.g1_xml,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
