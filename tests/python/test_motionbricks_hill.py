from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import math
from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.motionbricks_hill import (
    DEFAULT_HILL_HEIGHT_M,
    DEFAULT_HILL_SLOPE_DEGREES,
    GentleHillProfile,
)


class GentleHillProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hill = GentleHillProfile()

    def test_flat_aprons_and_cosine_crest(self) -> None:
        self.assertEqual(self.hill.height((-2.0, 0.0)), 0.0)
        self.assertEqual(self.hill.height((11.0, 0.0)), 0.0)
        self.assertAlmostEqual(
            self.hill.height(
                (self.hill.hill_start_x + 0.5 * self.hill.hill_length, 0.0)
            ),
            self.hill.height_m,
        )

    def test_default_mound_has_exact_eighteen_degree_grade(self) -> None:
        self.assertEqual(DEFAULT_HILL_SLOPE_DEGREES, 18.0)
        self.assertAlmostEqual(DEFAULT_HILL_HEIGHT_M, 0.7239760607, places=9)
        self.assertAlmostEqual(self.hill.height_m, DEFAULT_HILL_HEIGHT_M)
        self.assertAlmostEqual(self.hill.max_slope_degrees, 18.0, places=10)

    def test_default_map_has_room_to_steer_around_the_hill(self) -> None:
        approach_length = self.hill.hill_start_x - self.hill.domain_x[0]
        exit_length = (
            self.hill.domain_x[1]
            - self.hill.hill_start_x
            - self.hill.hill_length
        )
        self.assertGreaterEqual(self.hill.half_width, 8.0)
        self.assertGreaterEqual(approach_length, 6.0)
        self.assertGreaterEqual(exit_length, 6.0)

    def test_ground_is_flat_outside_the_local_hill_map(self) -> None:
        self.assertEqual(
            self.hill.height((self.hill.domain_x[1] + 100.0, 0.0)), 0.0
        )
        self.assertEqual(
            self.hill.height((0.0, self.hill.half_width + 100.0)), 0.0
        )

    def test_hill_falls_off_smoothly_in_the_lateral_direction(self) -> None:
        center_x = self.hill.hill_start_x + 0.5 * self.hill.hill_length
        radius = 0.5 * self.hill.hill_length
        self.assertAlmostEqual(self.hill.height((center_x, 0.0)), self.hill.height_m)
        self.assertAlmostEqual(
            self.hill.height((center_x, 0.5 * radius)),
            0.5 * self.hill.height_m,
        )
        self.assertEqual(self.hill.height((center_x, radius)), 0.0)

    def test_profile_is_continuous_at_hill_boundaries(self) -> None:
        epsilon = 1.0e-7
        start = self.hill.hill_start_x
        end = start + self.hill.hill_length
        self.assertLess(abs(self.hill.height((start + epsilon, 0.0))), 1.0e-12)
        self.assertLess(abs(self.hill.height((end - epsilon, 0.0))), 1.0e-12)

    def test_mesh_uses_the_same_height_contract(self) -> None:
        vertices, faces = self.hill.mesh(sample_count=41)
        self.assertGreater(vertices.shape[0], 1000)
        self.assertGreater(faces.shape[0], 2000)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.isfinite(faces).all())
        for vertex in vertices:
            self.assertAlmostEqual(
                float(vertex[2]),
                self.hill.height(vertex[:2]),
                places=12,
            )
        self.assertAlmostEqual(float(vertices[:, 1].min()), -self.hill.half_width)
        self.assertAlmostEqual(float(vertices[:, 1].max()), self.hill.half_width)

    def test_mesh_rejects_too_few_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count"):
            self.hill.mesh(sample_count=1)


class HillViewerCliTest(unittest.TestCase):
    def test_default_interactive_budget_runs_for_hours(self) -> None:
        from mm_sonic.motionbricks_hill_viewer import _parser

        arguments = _parser().parse_args([])

        self.assertGreaterEqual(arguments.max_steps, 1_000_000)

    def test_help_is_available_without_importing_motionbricks(self) -> None:
        from mm_sonic.motionbricks_hill_viewer import main

        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("--motionbricks-root", help_text)
        self.assertIn("--no-viewer", help_text)
        self.assertIn("--no-ik", help_text)
        self.assertIn("--smoke-steps", help_text)
        self.assertIn("--three-hills", help_text)

    def test_three_hill_mode_uses_the_shared_render_and_query_map(self) -> None:
        from mm_sonic.motionbricks_hill_viewer import _parser, _selected_profile

        profile = _selected_profile(_parser().parse_args(["--three-hills"]))
        vertices, faces = profile.mesh(17)

        self.assertEqual(profile.max_slope_degrees, 18.9)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        self.assertTrue(any(profile.height((hill.start_x + 1.0, 0.0)) > 0.0 for hill in profile.map.hills))
        for vertex in vertices[:: max(1, len(vertices) // 100)]:
            self.assertAlmostEqual(
                float(vertex[2]), profile.height(vertex[:2]), places=12
            )

    def test_trace_reports_ik_fallback_reason(self) -> None:
        from mm_sonic.motionbricks_authored_contacts import (
            AuthoredFootContacts,
        )
        from mm_sonic.motionbricks_hill_ik import HillFootIKDiagnostics
        from mm_sonic.motionbricks_hill_viewer import _trace_line

        diagnostics = HillFootIKDiagnostics(
            authored_stance=(True, False),
            locked=(True, False),
            root_height_correction_m=0.025,
            raw_penetration_m=0.02,
            corrected_penetration_m=0.02,
            maximum_target_residual_m=0.0,
            maximum_joint_correction_rad=0.0,
            iterations=0,
            accepted=False,
            reason="terrain sample invalid",
        )
        contacts = AuthoredFootContacts(
            channels=(True, False, False, False),
            stance=(True, False),
            valid=True,
            reason="ok",
        )

        line = _trace_line(
            3,
            np.asarray((0.0, 0.0, 0.8)),
            GentleHillProfile(),
            SimpleNamespace(latest_trace=None),
            diagnostics,
            contacts,
        )

        self.assertIn("contacts=1000", line)
        self.assertIn("stance=10", line)
        self.assertIn("locked=10", line)
        self.assertIn("root_dz=0.025", line)
        self.assertIn("accepted=0", line)
        self.assertIn("reason=terrain_sample_invalid", line)


if __name__ == "__main__":
    unittest.main()
