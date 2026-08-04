import unittest

from mm_sonic.joints import ContractError
from mm_sonic.torch_horizontal_traversal_library import (
    HorizontalGridLine,
    HorizontalTraversalLibrary,
    TraversalArtifact,
    TraversalFailure,
    horizontal_grid_lines,
    horizontal_library_from_dict,
)
from mm_sonic.torch_terrain_omni_routes import StairFrame


class HorizontalTraversalLibraryTest(unittest.TestCase):
    def setUp(self):
        self.frame = StairFrame(
            origin_world_xy=(0.0, 0.0),
            ascent_world_yaw=0.0,
            width_m=0.6223,
            tread_depth_m=0.3302,
            riser_height_m=0.1778,
            tread_count=3,
        )

    def test_grid_has_ten_centimeter_centers_in_both_directions(self):
        lines = horizontal_grid_lines(self.frame, spacing_m=0.10)

        self.assertEqual({line.direction for line in lines}, {-1, 1})
        self.assertEqual(
            [round(line.stair_u_m, 2) for line in lines if line.direction == 1],
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        )

    def test_library_rejects_success_and_failure_for_same_line(self):
        line = HorizontalGridLine(0, 0.05, 1)
        artifact = TraversalArtifact(
            line_id=line.line_id,
            route_sha256="a" * 64,
            route_path="routes/a.npz",
            source_family="curb",
            source_clip="grail-curb-a",
            phase_boundaries=(0, 10, 20, 30, 40, 50),
            minimum_sole_clearance_m=-0.01,
            maximum_planted_error_m=0.01,
            maximum_joint_speed_rad_s=2.0,
            maximum_joint_acceleration_rad_s2=20.0,
        )
        failure = TraversalFailure(
            line_id=line.line_id,
            failed_phase="mount",
            rejection_histogram=(("height_envelope", 1),),
        )

        with self.assertRaisesRegex(ContractError, "duplicate outcome"):
            HorizontalTraversalLibrary(
                dataset_manifest_sha256="b" * 64,
                stair_frame=self.frame,
                spacing_m=0.10,
                lines=(line,),
                artifacts=(artifact,),
                failures=(failure,),
            )

    def test_round_trip_authenticates_line_identity(self):
        lines = horizontal_grid_lines(self.frame, spacing_m=0.10)
        library = HorizontalTraversalLibrary(
            dataset_manifest_sha256="b" * 64,
            stair_frame=self.frame,
            spacing_m=0.10,
            lines=lines,
            artifacts=(),
            failures=tuple(
                TraversalFailure(
                    line_id=line.line_id,
                    failed_phase="inventory",
                    rejection_histogram=(("no_seed", 1),),
                )
                for line in lines
            ),
        )

        restored = horizontal_library_from_dict(library.to_dict())

        self.assertEqual(restored.library_id, library.library_id)
        payload = library.to_dict()
        payload["lines"][0]["line_id"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "identity"):
            horizontal_library_from_dict(payload)


if __name__ == "__main__":
    unittest.main()
