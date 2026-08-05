from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mm_sonic.render_stitched_motion import render_stitched_motion
from mm_sonic.terrain_oracle.stitch import FragmentSelection, stitch_archive


_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)
_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/g1_29dof_rev_1_0.xml"
)


@unittest.skipUnless(
    _ARCHIVE.is_dir() and _MODEL.is_file(),
    "real stairs500 archive or G1 model unavailable",
)
class StitchedMotionRenderTests(unittest.TestCase):
    def test_renders_actual_g1_and_target_mesh_from_stitched_arrays(self):
        motion = stitch_archive(
            _ARCHIVE,
            (FragmentSelection(37, 3, 8),),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.mp4"

            result = render_stitched_motion(
                motion,
                archive_path=_ARCHIVE,
                target_clip_index=37,
                model_path=_MODEL,
                output_path=output,
                width=320,
                height=240,
            )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1_000)
            self.assertEqual(result["frame_count"], 5)
            self.assertGreater(result["terrain_face_count"], 0)
            self.assertGreater(result["visual_mesh_geom_count"], 0)


if __name__ == "__main__":
    unittest.main()
