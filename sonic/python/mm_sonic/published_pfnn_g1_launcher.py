"""Prepare and run the published terrain-PFNN to G1 prototype."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence

from .published_pfnn_g1_prepare import (
    prepare_exporter,
    prepare_terrain,
    terrain_cache_path,
)
from .published_pfnn_g1_scenes import SceneSpec, load_scenes


DEFAULT_SCENE = 6
DEFAULT_SOURCE_DEMO = Path("/home/ubuntu/datasets/pfnn/pfnn/demo")
DEFAULT_GMR_ROOT = Path("/home/ubuntu/.cache/native-g1-pfnn/GMR")
DEFAULT_PYTHON = Path("/home/ubuntu/.cache/native-g1-pfnn/venv/bin/python")
DEFAULT_SCENE_XML = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/assets/skeletons/g1/scene_29dof.xml"
)
DEFAULT_CACHE_ROOT = Path.home() / ".cache/native-g1-pfnn/published-g1"
DEFAULT_RUNTIME_ROOT = Path.home() / ".cache/native-g1-pfnn/published-g1-live"
DEFAULT_DISPLAY = ":1"
DEFAULT_RESOURCES = (
    Path(__file__).resolve().parents[2] / "resources/published_pfnn_g1"
)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    command_sha256: str


@dataclass(frozen=True)
class LaunchSpec:
    scene: SceneSpec
    exporter_command: tuple[str, ...]
    bridge_command: tuple[str, ...]
    viewer_command: tuple[str, ...]
    terrain_path: Path


def _read_process(pid: int) -> tuple[int, bytes]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as error:
        raise ProcessLookupError(pid) from error
    close = stat.rfind(")")
    fields = stat[close + 2 :].split()
    if close < 0 or len(fields) < 20 or fields[0] == "Z" or not command:
        raise ProcessLookupError(pid)
    return int(fields[19]), command


def process_identity(pid: int) -> ProcessIdentity:
    start_ticks, command = _read_process(pid)
    return ProcessIdentity(
        pid=pid,
        start_ticks=start_ticks,
        command_sha256=hashlib.sha256(command).hexdigest(),
    )


def _identity_matches(identity: ProcessIdentity) -> bool:
    try:
        return process_identity(identity.pid) == identity
    except ProcessLookupError:
        return False


def stop_recorded_process(
    identity: ProcessIdentity, *, timeout_seconds: float = 5.0
) -> bool:
    try:
        actual = process_identity(identity.pid)
    except ProcessLookupError:
        return True
    if actual != identity:
        raise ValueError(f"process identity mismatch for PID {identity.pid}")
    os.kill(identity.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _identity_matches(identity):
            return True
        time.sleep(0.02)
    return not _identity_matches(identity)


def build_launch_spec(
    *,
    scene: SceneSpec,
    python: Path,
    exporter: Path,
    fifo: Path,
    terrain: Path,
    scene_xml: Path,
    gmr_root: Path,
) -> LaunchSpec:
    python_text = str(python)
    world = str(scene.world_id)
    fifo_text = str(fifo)
    return LaunchSpec(
        scene=scene,
        exporter_command=(
            str(exporter),
            "--world",
            world,
            "--export",
            fifo_text,
        ),
        bridge_command=(
            python_text,
            "-m",
            "mm_sonic.published_pfnn_g1_bridge",
            "--input",
            fifo_text,
            "--gmr-root",
            str(gmr_root),
        ),
        viewer_command=(
            python_text,
            "-m",
            "mm_sonic.published_pfnn_g1_viewer",
            "--input",
            "-",
            "--format",
            "jsonl",
            "--gmr-root",
            str(gmr_root),
            "--scene-xml",
            str(scene_xml),
            "--terrain-npz",
            str(terrain),
            "--hide-floor",
            "--fps",
            "60",
            "--trace-every",
            "60",
            "--expected-world",
            world,
        ),
        terrain_path=terrain,
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _read_receipt(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"invalid launcher receipt {path}")
    return value


def _receipt_identities(receipt: Mapping[str, object]) -> dict[str, ProcessIdentity]:
    if receipt.get("schema") != "published-pfnn-g1-live/v1":
        raise ValueError("launcher receipt schema is invalid")
    raw = receipt.get("processes")
    if not isinstance(raw, dict):
        raise ValueError("launcher receipt has no process identities")
    if set(raw) != {"exporter", "bridge", "viewer"}:
        raise ValueError("launcher receipt must contain exact process identities")
    result: dict[str, ProcessIdentity] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError("invalid launcher process identity")
        result[name] = ProcessIdentity(**value)
    return result


def _prepare(arguments: argparse.Namespace) -> tuple[SceneSpec, Path, Path]:
    scenes = load_scenes()
    scene = scenes[arguments.scene]
    cache_root = arguments.cache_root.expanduser()
    exporter = prepare_exporter(
        arguments.source_demo,
        cache_root / "export-demo",
        arguments.resources,
    )
    terrain = prepare_terrain(scene, arguments.source_demo, cache_root)
    return scene, exporter, terrain


def _environment(gmr_root: Path, display: str) -> dict[str, str]:
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        package_root if not existing else os.pathsep.join((package_root, existing))
    )
    environment["DISPLAY"] = display
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _planned_spec(arguments: argparse.Namespace) -> LaunchSpec:
    scene = load_scenes()[arguments.scene]
    cache_root = arguments.cache_root.expanduser()
    run_dir = (
        arguments.runtime_root.expanduser()
        / "runs"
        / f"scene-{scene.scene}-dry-run"
    )
    return build_launch_spec(
        scene=scene,
        python=arguments.python,
        exporter=cache_root / "export-demo" / "pfnn_export",
        fifo=run_dir / "frames.fifo",
        terrain=terrain_cache_path(scene, arguments.source_demo, cache_root),
        scene_xml=arguments.scene_xml,
        gmr_root=arguments.gmr_root,
    )


def _spec_json(spec: LaunchSpec) -> dict[str, object]:
    return {
        "scene": spec.scene.scene,
        "world_id": spec.scene.world_id,
        "heightmap": spec.scene.heightmap,
        "terrain": str(spec.terrain_path),
        "exporter_command": list(spec.exporter_command),
        "bridge_command": list(spec.bridge_command),
        "viewer_command": list(spec.viewer_command),
    }


def _stop_receipt(receipt_path: Path) -> dict[str, object]:
    receipt = _read_receipt(receipt_path)
    if receipt is None:
        return {"running": False, "stopped": []}
    identities = _receipt_identities(receipt)
    for identity in identities.values():
        if _identity_matches(identity):
            continue
        try:
            process_identity(identity.pid)
        except ProcessLookupError:
            continue
        raise ValueError(f"process identity mismatch for PID {identity.pid}")
    stopped: list[str] = []
    survivors: list[str] = []
    for name in ("exporter", "bridge", "viewer"):
        identity = identities.get(name)
        if identity is None:
            continue
        if stop_recorded_process(identity):
            stopped.append(name)
        else:
            survivors.append(name)
    if not survivors:
        receipt_path.unlink(missing_ok=True)
    return {"running": bool(survivors), "stopped": stopped, "survivors": survivors}


def _start(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.dry_run:
        return {"dry_run": True, **_spec_json(_planned_spec(arguments))}
    runtime_root = arguments.runtime_root.expanduser()
    receipt_path = runtime_root / "active.json"
    existing = _read_receipt(receipt_path)
    if existing is not None:
        identities = _receipt_identities(existing)
        if any(_identity_matches(identity) for identity in identities.values()):
            raise RuntimeError("published PFNN G1 launcher is already running")
        receipt_path.unlink()

    scene, exporter, terrain = _prepare(arguments)
    run_id = f"scene-{scene.scene}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    run_dir = runtime_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    fifo = run_dir / "frames.fifo"
    os.mkfifo(fifo)
    spec = build_launch_spec(
        scene=scene,
        python=arguments.python,
        exporter=exporter,
        fifo=fifo,
        terrain=terrain,
        scene_xml=arguments.scene_xml,
        gmr_root=arguments.gmr_root,
    )
    environment = _environment(arguments.gmr_root, arguments.display)
    started: list[subprocess.Popen[bytes]] = []
    try:
        with (run_dir / "bridge.log").open("ab", buffering=0) as bridge_log:
            bridge = subprocess.Popen(
                spec.bridge_command,
                stdout=subprocess.PIPE,
                stderr=bridge_log,
                env=environment,
                start_new_session=True,
            )
            started.append(bridge)
        assert bridge.stdout is not None
        with (run_dir / "viewer.log").open("ab", buffering=0) as viewer_log:
            viewer = subprocess.Popen(
                spec.viewer_command,
                stdin=bridge.stdout,
                stdout=viewer_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            started.append(viewer)
        bridge.stdout.close()
        with (run_dir / "exporter.log").open("ab", buffering=0) as exporter_log:
            exporter_process = subprocess.Popen(
                spec.exporter_command,
                cwd=exporter.parent,
                stdout=exporter_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            started.append(exporter_process)
        time.sleep(0.5)
        failed = [process.pid for process in started if process.poll() is not None]
        if failed:
            raise RuntimeError(f"launcher child exited during startup: {failed}")
        identities = {
            "bridge": process_identity(bridge.pid),
            "viewer": process_identity(viewer.pid),
            "exporter": process_identity(exporter_process.pid),
        }
        receipt = {
            "schema": "published-pfnn-g1-live/v1",
            "scene": scene.scene,
            "world_id": scene.world_id,
            "heightmap": scene.heightmap,
            "terrain": str(terrain),
            "run_dir": str(run_dir),
            "processes": {name: asdict(value) for name, value in identities.items()},
        }
        _write_json_atomic(receipt_path, receipt)
        return {"running": True, **receipt}
    except BaseException:
        for process in reversed(started):
            if process.poll() is None:
                process.terminate()
        raise


def _status(receipt_path: Path) -> dict[str, object]:
    receipt = _read_receipt(receipt_path)
    if receipt is None:
        return {"running": False}
    identities = _receipt_identities(receipt)
    live = {name: _identity_matches(identity) for name, identity in identities.items()}
    return {**receipt, "running": all(live.values()), "live": live}


def _add_common(parser: argparse.ArgumentParser, *, dry_run: bool) -> None:
    parser.add_argument("--scene", type=int, choices=range(1, 7), default=DEFAULT_SCENE)
    parser.add_argument("--source-demo", type=Path, default=DEFAULT_SOURCE_DEMO)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--resources", type=Path, default=DEFAULT_RESOURCES)
    parser.add_argument("--display", default=DEFAULT_DISPLAY)
    if dry_run:
        parser.add_argument("--dry-run", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_common(commands.add_parser("prepare"), dry_run=True)
    _add_common(commands.add_parser("start"), dry_run=True)
    _add_common(commands.add_parser("switch"), dry_run=True)
    for name in ("status", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt_path = arguments.runtime_root.expanduser() / "active.json"
    if arguments.command == "prepare":
        if arguments.dry_run:
            result: dict[str, object] = {
                "dry_run": True,
                **_spec_json(_planned_spec(arguments)),
            }
        else:
            scene, exporter, terrain = _prepare(arguments)
            result = {
                "prepared": True,
                "scene": scene.scene,
                "world_id": scene.world_id,
                "exporter": str(exporter),
                "terrain": str(terrain),
            }
    elif arguments.command == "start":
        result = _start(arguments)
    elif arguments.command == "switch":
        if arguments.dry_run:
            result = _start(arguments)
        else:
            stopped = _stop_receipt(receipt_path)
            result = {"previous": stopped, "current": _start(arguments)}
    elif arguments.command == "status":
        result = _status(receipt_path)
    else:
        result = _stop_receipt(receipt_path)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LaunchSpec",
    "ProcessIdentity",
    "build_launch_spec",
    "main",
    "process_identity",
    "stop_recorded_process",
]
