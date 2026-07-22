import hashlib
import pickle
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from resources.g1_reach_builder.sources import load_gmr_archive


def gmr_record(frames: int = 4, *, fps=30) -> dict:
    root_pos = np.zeros((frames, 3), np.float64)
    root_pos[:, 0] = np.arange(frames) * 0.01
    root_rot = np.zeros((frames, 4), np.float64)
    root_rot[:, 3] = 1.0
    dof_pos = np.zeros((frames, 29), np.float64)
    dof_pos[:, 0] = np.linspace(0.0, 0.1, frames)
    result = {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    if fps is not None:
        result["fps"] = fps
    return result


def write_archive(path: Path, members: dict[str, dict]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for sequence_id, record in members.items():
            archive.writestr(
                f"g1_retargeted_motions/gmr_pkl/{sequence_id}.pkl",
                pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
            )
        archive.writestr(
            "g1_retargeted_motions/soma_csv/pickup_north_0.csv",
            "ignored,csv\n",
        )


class GMRArchiveTests(unittest.TestCase):
    def test_loads_in_scope_sources_and_reorders_root_quaternion(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "motions.zip"
            pickup = gmr_record()
            write_archive(
                archive,
                {
                    "pickup_north_0": pickup,
                    "drawers": gmr_record(),
                    "left_to_right_2": gmr_record(),
                    "right_to_left_2": gmr_record(),
                    "walking": gmr_record(),
                    "carry_walking": gmr_record(),
                },
            )

            sources = load_gmr_archive(archive)

            self.assertEqual(
                [source.sequence_id for source in sources],
                [
                    "drawers",
                    "left_to_right_2",
                    "pickup_north_0",
                    "right_to_left_2",
                ],
            )
            source = sources[2]
            self.assertEqual(source.fps, 30.0)
            self.assertEqual(source.qpos.shape, (4, 36))
            np.testing.assert_array_equal(
                source.qpos[:, :3], pickup["root_pos"]
            )
            np.testing.assert_array_equal(
                source.qpos[:, 3:7], pickup["root_rot"][:, [3, 0, 1, 2]]
            )
            np.testing.assert_array_equal(
                source.qpos[:, 7:], pickup["dof_pos"]
            )
            np.testing.assert_array_equal(source.source_frames, np.arange(4))
            self.assertEqual(
                source.archive_sha256,
                hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
            self.assertEqual(source.disposition, "included")

    def test_can_report_but_not_publish_excluded_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "motions.zip"
            write_archive(
                archive,
                {"walking": gmr_record(), "carry_walking": gmr_record()},
            )

            sources = load_gmr_archive(archive, include_excluded=True)

            self.assertEqual(
                {source.sequence_id: source.disposition for source in sources},
                {"carry_walking": "excluded", "walking": "excluded"},
            )
            self.assertEqual(load_gmr_archive(archive), [])

    def test_requires_explicit_override_when_fps_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "motions.zip"
            write_archive(archive, {"pickup_north_0": gmr_record(fps=None)})

            with self.assertRaisesRegex(ValueError, "missing fps"):
                load_gmr_archive(archive)
            source = load_gmr_archive(archive, fps_override=30.0)[0]
            self.assertEqual(source.fps, 30.0)
            self.assertTrue(source.fps_overridden)

    def test_rejects_nonfinite_and_nonunit_source_arrays(self):
        for label, mutation, message in (
            (
                "nonfinite",
                lambda record: record["root_pos"].__setitem__((0, 0), np.nan),
                "non-finite",
            ),
            (
                "quaternion",
                lambda record: record["root_rot"].__setitem__((0, 3), 2.0),
                "quaternion",
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "motions.zip"
                record = gmr_record()
                mutation(record)
                write_archive(archive, {"pickup_north_0": record})
                with self.assertRaisesRegex(ValueError, message):
                    load_gmr_archive(archive)


if __name__ == "__main__":
    unittest.main()
