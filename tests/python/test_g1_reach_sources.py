import csv
import hashlib
import pickle
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from resources.g1_reach_builder.sources import (
    SOMA_COLUMNS,
    load_gmr_archive,
    load_soma_csv_directory,
)


def write_soma_csv(path: Path, frames: int = 4) -> np.ndarray:
    values = np.zeros((frames, len(SOMA_COLUMNS)), np.float64)
    values[:, 0] = np.arange(frames)
    values[:, 1:4] = np.array([100.0, -200.0, 75.0])
    values[:, 4:7] = np.array([10.0, 20.0, 30.0])
    values[:, 7:] = np.arange(29) + np.arange(frames)[:, None]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(SOMA_COLUMNS)
        writer.writerows(values)
    return values


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


    def test_loads_soma_csv_directory_with_exact_units_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "height0_top.csv"
            values = write_soma_csv(csv_path)

            source = load_soma_csv_directory(root)[0]

            self.assertEqual(source.sequence_id, "tabletop_soma/height0_top")
            self.assertEqual(source.fps, 100.0)
            self.assertTrue(source.fps_overridden)
            self.assertEqual(source.disposition, "included")
            self.assertEqual(source.qpos.shape, (4, 36))
            np.testing.assert_allclose(
                source.qpos[:, :3], values[:, 1:4] / 100.0
            )
            expected_xyzw = ScipyRotation.from_euler(
                "ZYX", values[:, 4:7][:, [2, 1, 0]], degrees=True
            ).as_quat()
            np.testing.assert_allclose(
                source.qpos[:, 3:7], expected_xyzw[:, [3, 0, 1, 2]]
            )
            np.testing.assert_allclose(
                source.qpos[:, 7:], np.radians(values[:, 7:])
            )
            np.testing.assert_array_equal(source.source_frames, np.arange(4))
            self.assertEqual(source.archive_member, "height0_top.csv")
            self.assertEqual(source.archive_path, csv_path.resolve())
            self.assertEqual(
                source.archive_sha256,
                hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            )

    def test_orders_and_namespaces_multiple_soma_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_soma_csv(root / "b_second.csv")
            write_soma_csv(root / "a_first.csv")

            sources = load_soma_csv_directory(root)

            self.assertEqual(
                [source.sequence_id for source in sources],
                ["tabletop_soma/a_first", "tabletop_soma/b_second"],
            )

    def test_rejects_malformed_soma_csv_sources(self):
        def mutate_header(rows: list[list]) -> None:
            rows[0][0] = "NotFrame"

        def mutate_frame_to_two(rows: list[list]) -> None:
            rows[1][0] = "2"

        def mutate_joint_to_nan(rows: list[list]) -> None:
            rows[1][7] = "nan"

        for label, mutation, message in (
            ("header", mutate_header, "SOMA header"),
            ("frame gap", mutate_frame_to_two, "contiguous from zero"),
            ("nonfinite", mutate_joint_to_nan, "non-finite"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                csv_path = root / "height0_top.csv"
                values = write_soma_csv(csv_path)
                rows = [list(SOMA_COLUMNS)] + [
                    [repr(float(cell)) for cell in row] for row in values
                ]
                mutation(rows)
                with csv_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, message):
                    load_soma_csv_directory(root)

    def test_rejects_invalid_fps_and_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_soma_csv(root / "height0_top.csv")
            with self.assertRaisesRegex(ValueError, "invalid fps"):
                load_soma_csv_directory(root, fps=0.0)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no SOMA CSV"):
                load_soma_csv_directory(Path(directory))


if __name__ == "__main__":
    unittest.main()
