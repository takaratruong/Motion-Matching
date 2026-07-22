import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from resources.g1_reach_builder.review import (
    convert_review_sources,
    read_review_corpus,
    write_review_corpus,
)
from resources.g1_reach_builder.schema import GMRSource
from resources.g1_terrain_builder.kinematics import G1Kinematics


G1_XML = os.environ.get(
    "G1_XML",
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml",
)


class ReviewCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = G1Kinematics(G1_XML)

    def source(self) -> GMRSource:
        qpos = np.tile(self.kinematics.keyframe_or_zero_qpos(), (7, 1))
        qpos[:, 0] = np.arange(7) * 0.01
        return GMRSource(
            sequence_id="pickup_north_0",
            archive_member="g1_retargeted_motions/gmr_pkl/pickup_north_0.pkl",
            archive_path=Path("/trusted/motions.zip"),
            archive_sha256="a" * 64,
            fps=30.0,
            fps_overridden=False,
            qpos=qpos,
            source_frames=np.arange(7, dtype=np.int32),
            disposition="included",
        )

    def test_converts_to_canonical_25_hz_review_corpus(self):
        corpus = convert_review_sources([self.source()], self.kinematics)

        self.assertEqual(corpus.fps, 25.0)
        self.assertEqual(corpus.sequence_ids, ("pickup_north_0",))
        self.assertEqual(corpus.range_starts.tolist(), [0])
        self.assertEqual(corpus.range_stops.tolist(), [len(corpus.positions)])
        self.assertEqual(corpus.positions.shape[1:], (31, 3))
        self.assertEqual(corpus.rotations.shape[1:], (31, 4))
        self.assertEqual(corpus.source_fps.tolist(), [30.0])
        self.assertTrue(np.all(np.diff(corpus.source_frames) >= 0))
        np.testing.assert_allclose(
            np.linalg.norm(corpus.rotations, axis=-1), 1.0, atol=1e-4
        )

    def test_review_directory_round_trips_without_pickle_arrays(self):
        corpus = convert_review_sources([self.source()], self.kinematics)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"

            write_review_corpus(output, corpus)
            loaded = read_review_corpus(output)

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"manifest.json", "review_motions.npz", "proposals.json"},
            )
            np.testing.assert_array_equal(loaded.positions, corpus.positions)
            np.testing.assert_array_equal(loaded.rotations, corpus.rotations)
            np.testing.assert_array_equal(
                loaded.source_frames, corpus.source_frames
            )
            self.assertEqual(loaded.sequence_ids, corpus.sequence_ids)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["target_fps"], 25.0)
            self.assertEqual(manifest["source_fps"], [30.0])
            with np.load(output / "review_motions.npz", allow_pickle=False) as data:
                self.assertEqual(set(data.files), {
                    "positions", "rotations", "range_starts", "range_stops",
                    "source_frames",
                })

    def test_rejects_excluded_sources_from_conversion(self):
        source = self.source()
        source.disposition = "excluded"
        with self.assertRaisesRegex(ValueError, "excluded source"):
            convert_review_sources([source], self.kinematics)


if __name__ == "__main__":
    unittest.main()
