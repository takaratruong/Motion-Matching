import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_parallel_path_grid.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_parallel_path_grid", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ParallelPathGridRunnerTests(unittest.TestCase):
    def test_parser_uses_automatic_staircase_band_by_default(self):
        args = _MODULE.parser().parse_args(
            [
                "--source-dataset", "source",
                "--target-scene", "scene",
                "--g1-xml", "g1.xml",
                "--center-start", "0.1", "0.2",
                "--heading-degrees", "-45",
                "--path-length-m", "2.4",
                "--spacing-m", "0.2",
                "--output", "output",
            ]
        )

        self.assertIsNone(args.minimum_lateral_offset_m)
        self.assertIsNone(args.maximum_lateral_offset_m)
        self.assertEqual(
            args.validation_config,
            Path(
                "sonic/configs/experiments/"
                "torch_grail_raw_horizontal_preview.json"
            ),
        )

    def test_flat_contract_never_dispatches_root_search(self):
        dispatch = _MODULE.searchable_contracts(
            (
                SimpleNamespace(classification="flat_only_excluded"),
                SimpleNamespace(classification="staircase_intersecting"),
            )
        )

        self.assertEqual(len(dispatch), 1)
        self.assertEqual(
            dispatch[0].classification, "staircase_intersecting"
        )

    def test_search_failure_is_reported_as_missing_required_phase(self):
        self.assertEqual(
            _MODULE.classify_search_failure(
                log_text=(
                    "ContractError: no complete ordered-level chain"
                ),
                diagnostic={"first_missing_phase": "dismount"},
            ),
            "no_dismount_motion",
        )

    def test_route_contract_record_carries_staircase_profile(self):
        path = _MODULE.parallel_path_grid(
            center_start_scene_xy=(0.0, 0.0),
            heading_scene_xy=(1.0, 0.0),
            path_length_m=1.0,
            minimum_lateral_offset_m=0.0,
            maximum_lateral_offset_m=0.0,
            spacing_m=0.2,
        )[0]
        contract = _MODULE.StaircasePathContract(
            path=path,
            classification="staircase_intersecting",
            ordered_surface_heights_m=(0.0, 0.2, 0.0),
            elevated_intervals_m=((0.3, 0.7),),
        )

        record = _MODULE.route_contract_record(contract)

        self.assertEqual(record["classification"], "staircase_intersecting")
        self.assertEqual(record["start_scene_xy"], [0.0, 0.0])
        self.assertEqual(record["ordered_surface_heights_m"], [0.0, 0.2, 0.0])

    def test_playlist_entries_include_only_independently_validated_results(self):
        paths = _MODULE.parallel_path_grid(
            center_start_scene_xy=(0.0, 0.0),
            heading_scene_xy=(1.0, 0.0),
            path_length_m=1.0,
            minimum_lateral_offset_m=0.0,
            maximum_lateral_offset_m=0.2,
            spacing_m=0.2,
        )
        with tempfile.TemporaryDirectory() as directory:
            traversal = Path(directory) / "traversal.npz"
            np.savez_compressed(
                traversal,
                joint_position=np.zeros((3, 29)),
                root_position_world=np.zeros((3, 3)),
                root_orientation_world_wxyz=np.tile(
                    (1.0, 0.0, 0.0, 0.0), (3, 1)
                ),
            )
            entries = _MODULE.playlist_entries(
                paths=paths,
                results=(
                    {
                        "path_id": paths[0].path_id,
                        "status": "validated",
                        "independently_validated": True,
                        "traversal": str(traversal),
                        "ordered_surface_heights_m": [0.0, 0.2, 0.0],
                        "elevated_intervals_m": [[0.3, 0.7]],
                        "quality": {"validated": True},
                    },
                    {
                        "path_id": paths[1].path_id,
                        "status": "failed",
                        "failure_code": "validation_failed",
                    },
                ),
            )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][0].path_id, paths[0].path_id)

    def test_elevated_cell_centers_exclude_ground_cells(self):
        points = _MODULE.elevated_grid_cell_centers(
            origin=np.array((-1.0, 2.0)),
            cell=0.1,
            height=np.array(((0.0, 0.2), (0.0, 0.4))),
        )

        np.testing.assert_allclose(
            points,
            ((-0.85, 2.05), (-0.85, 2.15)),
            atol=1.0e-12,
        )

    def test_child_environment_adds_repo_root_and_preserves_pythonpath(self):
        environment = _MODULE.child_environment(
            existing={"PYTHONPATH": "sonic/python", "KEEP": "yes"},
            repo_root=Path("/workspace/repo"),
        )

        self.assertEqual(environment["KEEP"], "yes")
        self.assertEqual(
            environment["PYTHONPATH"],
            "/workspace/repo:sonic/python",
        )

    def test_parser_exposes_only_generic_path_grid_geometry(self):
        args = _MODULE.parser().parse_args(
            [
                "--source-dataset", "source",
                "--target-scene", "scene",
                "--g1-xml", "g1.xml",
                "--center-start", "0.1", "0.2",
                "--heading-degrees", "37",
                "--path-length-m", "2.4",
                "--minimum-lateral-offset-m", "-0.3",
                "--maximum-lateral-offset-m", "0.3",
                "--spacing-m", "0.15",
                "--output", "output",
            ]
        )

        self.assertEqual(args.heading_degrees, 37.0)
        self.assertEqual(args.spacing_m, 0.15)
        self.assertFalse(hasattr(args, "path_y"))
        self.assertFalse(hasattr(args, "lane_scene_y"))

    def test_root_search_command_forwards_computed_endpoints(self):
        command = _MODULE.root_search_command(
            python=Path("python"),
            runner=Path("root-search.py"),
            source_dataset=Path("source"),
            target_scene="scene",
            g1_xml=Path("g1.xml"),
            start_scene_xy=(0.25, -0.5),
            stop_scene_xy=(1.75, 0.5),
            output=Path("result"),
            segment_length_m=0.9,
            search_stride_m=0.2,
            step_width_m=0.2,
            workers=7,
            coarse_results=123,
            placement_shortlist=45,
            candidates_per_anchor=16,
            blend_frames=4,
            ordered_contact_levels=True,
        )

        self.assertEqual(command[command.index("--path-start") + 1:
                                 command.index("--path-stop")],
                         ["0.25", "-0.5"])
        self.assertEqual(command[command.index("--path-stop") + 1:
                                 command.index("--segment-length-m")],
                         ["1.75", "0.5"])
        self.assertIn("--ordered-contact-levels", command)

    def test_independent_validator_receives_route_contract(self):
        command = _MODULE.independent_validation_command(
            python=Path("python"),
            runner=Path("validator.py"),
            traversal=Path("path/traversal.npz"),
            target_dataset=Path("path/dataset"),
            config=Path("config.json"),
            g1_xml=Path("g1.xml"),
            route_contract=Path("path/route-contract.json"),
            expected_heading_degrees=-45.0,
            output=Path("path/independent-validation.json"),
            maximum_unsupported_frames=50,
        )

        self.assertIn("--route-contract", command)
        self.assertEqual(
            command[command.index("--route-contract") + 1],
            "path/route-contract.json",
        )
        self.assertEqual(
            command[command.index("--expected-heading-degrees") + 1],
            "-45.0",
        )

    def test_summary_keeps_excluded_failed_and_validated_paths(self):
        summary = _MODULE.grid_summary(
            geometry={"heading_degrees": 37.0, "spacing_m": 0.15},
            results=(
                {
                    "path_id": "path-000",
                    "status": "excluded",
                    "failure_code": "flat_only_excluded",
                },
                {
                    "path_id": "path-001",
                    "status": "failed",
                    "failure_code": "no_mount_motion",
                },
                {"path_id": "path-002", "status": "validated"},
            ),
        )

        self.assertEqual(summary["requested_path_count"], 3)
        self.assertEqual(summary["staircase_path_count"], 2)
        self.assertEqual(summary["excluded_path_count"], 1)
        self.assertEqual(summary["validated_path_count"], 1)
        self.assertEqual(summary["failed_path_count"], 1)
        self.assertEqual(len(summary["results"]), 3)

    def test_split_results_merge_into_canonical_grid_without_missing_paths(self):
        paths = _MODULE.parallel_path_grid(
            center_start_scene_xy=(0.0, 0.0),
            heading_scene_xy=(1.0, 0.0),
            path_length_m=1.0,
            minimum_lateral_offset_m=-0.2,
            maximum_lateral_offset_m=0.2,
            spacing_m=0.2,
        )
        merged = _MODULE.merge_grid_results(
            paths=paths,
            result_groups=(
                (
                    {
                        "path_id": "split-negative",
                        "lateral_offset_m": -0.2,
                        "status": "failed",
                    },
                ),
                (
                    {
                        "path_id": "split-center",
                        "lateral_offset_m": 0.0,
                        "status": "validated",
                    },
                    {
                        "path_id": "split-positive",
                        "lateral_offset_m": 0.2,
                        "status": "validated",
                    },
                ),
            ),
        )

        self.assertEqual(len(merged), 3)
        self.assertEqual(
            [item["path_id"] for item in merged],
            [path.path_id for path in paths],
        )
        self.assertEqual(
            [item["status"] for item in merged],
            ["failed", "validated", "validated"],
        )


if __name__ == "__main__":
    unittest.main()
