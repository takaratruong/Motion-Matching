import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from resources.g1_interaction_builder.artifacts import write_artifact_set
from resources.g1_interaction_builder.schema import InteractionPhase
from tests.python.test_interaction_artifacts import artifact_fixture


class InteractionQueryParityTests(unittest.TestCase):
    def test_probe_reconstructs_first_reach_query_for_every_clip(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "pack"
            write_artifact_set(
                pack, artifact, features, split, manifest, report
            )

            for clip, (start, stop) in enumerate(
                zip(artifact.range_starts, artifact.range_stops)
            ):
                relative_reach = np.flatnonzero(
                    artifact.phases[start:stop]
                    == int(InteractionPhase.REACH)
                )
                self.assertGreater(len(relative_reach), 0)
                frame = int(start + relative_reach[0])

                with self.subTest(clip=clip, frame=frame):
                    completed = subprocess.run(
                        [
                            "./interaction_query_probe",
                            str(pack),
                            str(clip),
                            str(frame),
                            "--json",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    payload = json.loads(completed.stdout)
                    self.assertEqual(payload["dimension"], 71)
                    self.assertLessEqual(payload["max_abs_error"], 2e-4)
                    self.assertEqual(
                        payload["groups"],
                        [[0, 33], [33, 45], [45, 57], [57, 65], [65, 71]],
                    )
                    self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
