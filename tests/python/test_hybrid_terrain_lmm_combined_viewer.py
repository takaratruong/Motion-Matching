from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mm_sonic.hybrid_terrain_lmm_combined_viewer import (
    _load_combined_matcher,
    main,
)
from mm_sonic.hybrid_terrain_lmm_viewer import DEFAULT_G1_XML


class HybridTerrainLmmCombinedViewerTests(unittest.TestCase):
    def test_matcher_loader_uses_only_the_combined_cache_authority(self) -> None:
        arguments = SimpleNamespace(
            cache=Path("combined"),
            model=Path("model"),
            scene="hills",
            g1_xml=DEFAULT_G1_XML,
            search_rows=None,
            transition_penalty=0.1,
        )
        corpus = object()
        generator = object()
        terrain = SimpleNamespace(
            authority=object(),
            spawn_native_xy=(1.0, 2.0),
            spawn_heading=0.25,
        )
        native_model = object()
        matcher = object()
        with (
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer.load_combined_cache",
                return_value=corpus,
            ) as combined_loader,
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer.viewer._load_generator",
                return_value=generator,
            ) as generator_loader,
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer.viewer.load_scene_terrain",
                return_value=terrain,
            ) as scene_loader,
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer._load_native_model",
                return_value=native_model,
            ),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer.HybridMatcher",
                return_value=matcher,
            ) as matcher_type,
        ):
            loaded_matcher, loaded_terrain = _load_combined_matcher(arguments)

        self.assertIs(loaded_matcher, matcher)
        self.assertIs(loaded_terrain, terrain)
        combined_loader.assert_called_once_with(Path("combined"))
        generator_loader.assert_called_once_with(Path("model"), corpus)
        scene_loader.assert_called_once_with("hills", corpus=corpus)
        matcher_type.assert_called_once_with(
            corpus,
            generator,
            terrain.authority,
            native_model=native_model,
            max_search_rows=None,
            transition_penalty=0.1,
            initial_root_xy=(1.0, 2.0),
            initial_heading=0.25,
            cache_manifest_path=Path("combined") / "manifest.json",
        )

    def test_smoke_cli_reuses_the_existing_mujoco_gate(self) -> None:
        matcher, terrain = object(), object()
        receipt = {"accepted": True, "mujoco_forward_count": 1_000}
        stdout = io.StringIO()
        with (
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer._load_combined_matcher",
                return_value=(matcher, terrain),
            ),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer.viewer.run_mujoco_headless_smoke",
                return_value=receipt,
            ) as smoke,
            contextlib.redirect_stdout(stdout),
        ):
            result = main(
                ["smoke", "--cache", "cache", "--model", "model", "--frames", "1000"]
            )

        self.assertEqual(result, 0)
        smoke.assert_called_once_with(
            matcher, terrain, g1_xml=DEFAULT_G1_XML, frames=1_000
        )
        self.assertEqual(json.loads(stdout.getvalue()), receipt)

    def test_combined_receipt_reuses_exclusive_atomic_publisher(self) -> None:
        matcher, terrain = object(), object()
        receipt = {"accepted": True, "mujoco_forward_count": 1_000}
        target = Path("combined-smoke.json")
        with (
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer._load_combined_matcher",
                return_value=(matcher, terrain),
            ),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer.viewer.run_mujoco_headless_smoke",
                return_value=receipt,
            ),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_combined_viewer.viewer.write_receipt_exclusive"
            ) as publish,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = main(
                [
                    "smoke", "--cache", "cache", "--model", "model",
                    "--receipt", str(target),
                ]
            )

        self.assertEqual(result, 0)
        publish.assert_called_once_with(target, receipt)


if __name__ == "__main__":
    unittest.main()
