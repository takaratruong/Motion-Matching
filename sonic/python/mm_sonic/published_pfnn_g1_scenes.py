"""Load the exact published PFNN world-to-heightmap scene mapping."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


_MANIFEST = Path(__file__).with_suffix(".json")
_FIELDS = {"scene", "world_id", "key", "heightmap", "display_stride"}


@dataclass(frozen=True)
class SceneSpec:
    scene: int
    world_id: int
    key: int
    heightmap: str
    display_stride: int


def load_scenes(path: Path | None = None) -> dict[int, SceneSpec]:
    manifest = _MANIFEST if path is None else Path(path)
    try:
        entries = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid PFNN scene manifest {manifest}: {error}") from error
    if not isinstance(entries, list):
        raise ValueError("PFNN scene manifest must be a list")

    scenes: dict[int, SceneSpec] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _FIELDS:
            raise ValueError("PFNN scene manifest entries must have exact SceneSpec fields")
        try:
            spec = SceneSpec(**entry)
        except TypeError as error:
            raise ValueError("invalid PFNN scene manifest entry") from error
        if (any(type(value) is not int for value in (spec.scene, spec.world_id, spec.key, spec.display_stride))
                or not isinstance(spec.heightmap, str) or not spec.heightmap
                or spec.display_stride < 1):
            raise ValueError("invalid PFNN scene manifest value")
        if spec.scene in scenes:
            raise ValueError(f"duplicate PFNN scene {spec.scene}")
        scenes[spec.scene] = spec
    if set(scenes) != set(range(1, 7)):
        raise ValueError("PFNN scene manifest must contain scenes 1 through 6")
    if {spec.world_id for spec in scenes.values()} != set(range(6)):
        raise ValueError("PFNN scene manifest must contain worlds 0 through 5")
    return scenes
