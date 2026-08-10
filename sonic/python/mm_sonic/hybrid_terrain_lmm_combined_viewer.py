"""MuJoCo smoke/view wrapper for an authenticated combined terrain corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_sonic import hybrid_terrain_lmm_viewer as viewer
from mm_sonic.hybrid_terrain_lmm_combined import load_combined_cache
from mm_sonic.hybrid_terrain_lmm_runtime import HybridMatcher


def _load_native_model(path: Path) -> object:
    import mujoco

    return mujoco.MjModel.from_xml_path(str(path))


def _load_combined_matcher(
    arguments: argparse.Namespace,
) -> tuple[HybridMatcher, viewer.SceneTerrainAdapter]:
    corpus = load_combined_cache(arguments.cache)
    generator = viewer._load_generator(arguments.model, corpus)
    adapter = viewer.load_scene_terrain(arguments.scene, corpus=corpus)
    matcher = HybridMatcher(
        corpus,
        generator,
        adapter.authority,
        native_model=_load_native_model(arguments.g1_xml),
        max_search_rows=arguments.search_rows,
        transition_penalty=arguments.transition_penalty,
        initial_root_xy=adapter.spawn_native_xy,
        initial_heading=adapter.spawn_heading,
        cache_manifest_path=Path(arguments.cache) / "manifest.json",
    )
    return matcher, adapter


def main(argv: list[str] | None = None) -> int:
    arguments = viewer.build_parser().parse_args(argv)
    matcher, terrain = _load_combined_matcher(arguments)
    if arguments.command == "smoke":
        receipt = viewer.run_mujoco_headless_smoke(
            matcher,
            terrain,
            g1_xml=arguments.g1_xml,
            frames=arguments.frames,
        )
    else:
        receipt = viewer.run_interactive(
            matcher,
            terrain,
            g1_xml=arguments.g1_xml,
            gamepad=arguments.gamepad,
            max_render_frames=arguments.max_render_frames,
        )
    payload = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    print(payload, flush=True)
    if arguments.receipt is not None:
        viewer.write_receipt_exclusive(arguments.receipt, receipt)
    return 0 if bool(receipt.get("accepted", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
