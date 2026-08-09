import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.compare_pfnn_terrain_g1 import (
    classify_probe_gaps,
    first_support_failure,
)
from mm_sonic.pfnn_terrain_fit import (
    PFNNTerrainFit,
    _fit_patch_bank,
    _install_numpy_umath_tests_compat,
    display_contacts,
    load_terrain_fit,
    patch_height,
    save_terrain_fit,
    terrain_height_g1,
    terrain_height_pfnn,
    validate_footstep_cycle,
)
from mm_sonic.view_g1_retarget import terrain_mesh


def _fit() -> PFNNTerrainFit:
    return PFNNTerrainFit(
        patch=np.arange(16, dtype=np.float64).reshape(4, 4),
        patch_coord=np.array([2.0, 3.0, 4.0, 5.0]),
        contact_center_xz=np.array([0.0, 0.0]),
        patch_height_mean=22.5,
        stance_height_mean=100.0,
        rbf_centers_xz=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        rbf_epsilon=np.array([0.5, 0.5]),
        rbf_weights=np.array([[0.25, -0.25]]),
        source_contacts=np.array(
            [[True, True, False, False], [False, False, True, True]],
            dtype=np.bool_,
        ),
        source_start_frame=8160,
        source_frame_count=2,
        cycle_start_frame=8135,
        cycle_stop_frame=8229,
        selected_patch_index=7,
        fitting_error=0.125,
        source_sha256="a" * 64,
        patches_sha256="b" * 64,
    )


class PFNNTerrainTransferTests(unittest.TestCase):
    def test_uniform_terrain_scale_preserves_shape_and_grade(self) -> None:
        fit = _fit()
        native_xy = np.array([[-0.01, 0.0], [0.0, 0.01], [0.01, -0.01]])
        native_height = terrain_height_g1(fit, native_xy)
        scale = 0.875
        np.testing.assert_allclose(
            terrain_height_g1(fit, native_xy * scale, scale=scale),
            native_height * scale,
            atol=1.0e-12,
            rtol=0.0,
        )
        offset = 0.05224985936713168
        np.testing.assert_allclose(
            terrain_height_g1(
                fit,
                native_xy * scale,
                scale=scale,
                z_offset=offset,
            ),
            native_height * scale + offset,
            atol=1.0e-12,
            rtol=0.0,
        )
        with self.assertRaises(ValueError):
            terrain_height_g1(fit, native_xy, scale=0.0)

    def test_support_failure_ignores_the_nonstance_probe_on_a_stance_foot(self) -> None:
        gaps = np.array([[0.05, 0.0, 0.0, 0.0]])
        contacts = np.array([[False, True, False, False]], dtype=np.bool_)
        self.assertIsNone(first_support_failure(gaps, contacts, source_start_frame=10))
        gaps[0, 1] = 0.021
        self.assertEqual(
            first_support_failure(gaps, contacts, source_start_frame=10),
            {
                "frame": 0,
                "source_frame": 10,
                "foot": "left",
                "probe": "toe",
                "gap_m": 0.021,
                "labels": ["floating"],
            },
        )

    def test_installs_only_the_removed_numpy_matrix_multiply_compatibility(self) -> None:
        module = _install_numpy_umath_tests_compat()
        left = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        right = np.eye(2, dtype=np.float64)[None].repeat(2, axis=0)
        np.testing.assert_array_equal(module.matrix_multiply(left, right), left)

    def test_mesh_vertices_use_the_exact_continuous_terrain_query(self) -> None:
        fit = _fit()
        x = np.array([-0.01, 0.0, 0.01])
        y = np.array([-0.01, 0.01])
        vertices, faces = terrain_mesh(fit, x_samples=x, y_samples=y)
        self.assertEqual(vertices.shape, (6, 3))
        self.assertEqual(faces.shape, (4, 3))
        np.testing.assert_allclose(
            vertices[:, 2],
            terrain_height_g1(fit, vertices[:, :2]),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_classifies_stance_and_swing_probe_gaps(self) -> None:
        gaps = np.array([-0.011, -0.01, 0.02, 0.021])
        self.assertEqual(
            classify_probe_gaps(gaps, stance=True),
            ("penetrating", "contact", "contact", "floating"),
        )
        self.assertEqual(
            classify_probe_gaps(gaps, stance=False),
            ("swing", "swing", "swing", "swing"),
        )

    def test_patch_height_matches_pfnn_bilinear_sampling(self) -> None:
        patch = np.arange(16, dtype=np.float64).reshape(4, 4)
        # With hscale=1 and vscale=1, (0,0) addresses patch center [2,2].
        queries = np.array([[0.0, 0.0], [-0.5, -0.5], [0.5, 0.5]])
        np.testing.assert_allclose(
            patch_height(patch, queries, hscale=1.0, vscale=1.0),
            [10.0, 7.5, 12.5],
            atol=0.0,
            rtol=0.0,
        )

    def test_g1_query_uses_released_axis_and_unit_transform(self) -> None:
        fit = _fit()
        g1_xy = np.array([[0.0, 0.0], [0.01, 0.0], [-0.01, 0.0]])
        pfnn_xz = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        np.testing.assert_allclose(
            terrain_height_g1(fit, g1_xy),
            terrain_height_pfnn(fit, pfnn_xz) / 100.0,
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_round_trips_safe_numeric_terrain_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.npz"
            receipt = save_terrain_fit(path, _fit())
            loaded = load_terrain_fit(path)
            self.assertTrue(receipt.is_file())
            self.assertEqual(loaded.selected_patch_index, 7)
            self.assertEqual(loaded.source_start_frame, 8160)
            np.testing.assert_array_equal(loaded.patch, _fit().patch)
            np.testing.assert_array_equal(
                loaded.source_contacts, _fit().source_contacts
            )

    def test_rejects_nonfinite_rbf_weights(self) -> None:
        fit = _fit()
        values = dict(fit.__dict__)
        values["rbf_weights"] = np.array([[np.nan, 0.0]])
        with self.assertRaisesRegex(ValueError, "rbf_weights"):
            PFNNTerrainFit(**values)

    def test_selects_exact_enclosing_footstep_cycle(self) -> None:
        footsteps = ("100 125\n", "200 225\n", "300 325\n")
        validate_footstep_cycle(footsteps, cycle_start=100, cycle_stop=200)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            validate_footstep_cycle(footsteps, cycle_start=100, cycle_stop=300)

    def test_maps_original_60_hz_contacts_to_native_120_hz_frames(self) -> None:
        contacts = np.array(
            [
                [True, False, False, True],
                [False, True, True, False],
                [True, True, False, False],
            ],
            dtype=np.bool_,
        )
        np.testing.assert_array_equal(
            display_contacts(
                contacts,
                display_start_frame=1,
                display_frame_count=4,
            ),
            contacts[[0, 1, 1, 2]],
        )

    def test_selects_lowest_error_patch_and_fits_contact_residual(self) -> None:
        patches = np.zeros((2, 4, 4), dtype=np.float64)
        patches[1] = np.arange(4, dtype=np.float64)[:, None]
        down_xz = np.array([[-0.5, 0.0], [0.5, 0.0]])
        # Patch 1 evaluates to 1.5 and 2.5 at these points.
        down_y = np.array([[11.5], [12.5]])
        up_xz = np.array([[0.0, 0.0]])
        up_y = np.array([[20.0]])
        selected = _fit_patch_bank(
            patches,
            down_xz=down_xz,
            down_y=down_y,
            up_xz=up_xz,
            up_y=up_y,
            hscale=1.0,
            vscale=1.0,
        )
        self.assertEqual(selected.patch_index, 1)
        np.testing.assert_allclose(selected.residual, 0.0, atol=1.0e-12)
        self.assertLess(selected.fitting_error, 1.0e-12)


if __name__ == "__main__":
    unittest.main()
