#!/usr/bin/env python3

import unittest


class PublishedPFNNG1ScenesTests(unittest.TestCase):
    def test_manifest_binds_all_six_published_worlds_exactly(self) -> None:
        from mm_sonic.published_pfnn_g1_scenes import load_scenes

        expected = {
            1: (0, 1, "hmap_000_smooth.txt"),
            2: (1, 2, "hmap_000_smooth.txt"),
            3: (2, 3, "hmap_004_smooth.txt"),
            4: (3, 4, "hmap_007_smooth.txt"),
            5: (4, 5, "hmap_013_smooth.txt"),
            6: (5, 6, "hmap_urban_001_smooth.txt"),
        }

        scenes = load_scenes()

        self.assertEqual(set(scenes), set(expected))
        self.assertEqual(
            {
                scene: (spec.world_id, spec.key, spec.heightmap)
                for scene, spec in scenes.items()
            },
            expected,
        )


if __name__ == "__main__":
    unittest.main()
