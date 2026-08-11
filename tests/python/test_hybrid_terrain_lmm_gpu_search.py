from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from mm_sonic.hybrid_terrain_lmm_gpu_search import (
    SingleGpuExactSearch,
    configure_single_gpu_visibility,
)


@dataclass(frozen=True)
class _FakeDevice:
    id: int = 0
    platform: str = "gpu"


class _FakeArray:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)
        self.synchronized = False

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.asarray(self.value, dtype=dtype)

    def block_until_ready(self):
        self.synchronized = True
        return self

    def astype(self, dtype):
        return _FakeArray(self.value.astype(dtype))


class _FakeConfig:
    x64_enabled = True


class _FakeJax:
    def __init__(self, devices: tuple[_FakeDevice, ...] = (_FakeDevice(),)) -> None:
        self._devices = devices
        self.config = _FakeConfig()
        self.device_put_calls: list[tuple[np.ndarray, _FakeDevice]] = []
        self.jit_devices: list[_FakeDevice] = []
        self.compiled_calls: list[tuple[object, ...]] = []
        self.outputs = (
            _FakeArray((9, 3, -1)),
            _FakeArray(1.25),
            _FakeArray(2),
            _FakeArray(7),
        )

    def devices(self, backend: str | None = None):
        if backend != "gpu":
            raise AssertionError("only the isolated GPU backend may be enumerated")
        return list(self._devices)

    def device_put(self, value: object, device: _FakeDevice | None = None):
        if device is None:
            raise AssertionError("every transfer must name the isolated device")
        self.device_put_calls.append((np.asarray(value), device))
        return _FakeArray(value)

    def jit(self, function, *, device: _FakeDevice | None = None):
        del function
        if device is None:
            raise AssertionError("the compiled call must name the isolated device")
        self.jit_devices.append(device)

        def compiled(*arguments):
            self.compiled_calls.append(arguments)
            return self.outputs

        return compiled

    def pmap(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("pmap must not be used")


def _tables(rows: int = 10) -> tuple[np.ndarray, ...]:
    features = np.arange(rows * 31, dtype=np.float32).reshape(rows, 31) / 100.0
    searchable_rows = np.asarray((1, 3, 5, 7, 9), dtype=np.int64)
    row_ranges = np.arange(rows, dtype=np.int32) % 3
    contacts = np.zeros((rows, 2), dtype=np.bool_)
    contacts[:, 0] = np.arange(rows) % 2 == 0
    contacts[:, 1] = np.arange(rows) % 3 == 0
    return features, searchable_rows, row_ranges, contacts


def _fake_search(
    fake_jax: _FakeJax | None = None,
) -> tuple[SingleGpuExactSearch, _FakeJax]:
    jax = fake_jax or _FakeJax()
    with (
        mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}, clear=False),
        mock.patch(
            "mm_sonic.hybrid_terrain_lmm_gpu_search._load_jax",
            return_value=(jax, np),
        ),
    ):
        search = SingleGpuExactSearch(
            *_tables(),
            physical_device_index=5,
            transition_penalty=0.1,
            exclusion_budget=2,
        )
    return search, jax


class SingleGpuVisibilityTests(unittest.TestCase):
    def test_configures_exact_physical_gpu_before_jax_import(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(sys.modules),
        ):
            sys.modules.pop("jax", None)

            configured = configure_single_gpu_visibility("cuda:5")

            self.assertEqual(configured, ("cuda:5", 5))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "5")
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
            self.assertEqual(os.environ["JAX_ENABLE_X64"], "true")
            self.assertNotIn("jax", sys.modules)

    def test_rejects_malformed_search_devices(self):
        malformed = ("cuda", "cuda:", "cuda:-1", "cuda: 5", "cuda:5,6", "gpu:5", 5)
        for spec in malformed:
            with (
                self.subTest(spec=spec),
                self.assertRaisesRegex(ValueError, "cuda:<physical-index>"),
            ):
                configure_single_gpu_visibility(spec)

    def test_rejects_conflicting_existing_visibility(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "visibility conflicts"):
                configure_single_gpu_visibility("cuda:5")

    def test_rejects_jax_imported_before_isolation(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(sys.modules, {"jax": object()}),
            self.assertRaisesRegex(RuntimeError, "JAX was imported before"),
        ):
            configure_single_gpu_visibility("cuda:5")


class SingleGpuExactSearchContractTests(unittest.TestCase):
    def test_requires_visibility_to_name_the_requested_physical_gpu(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA visibility"):
                SingleGpuExactSearch(
                    *_tables(),
                    physical_device_index=5,
                    transition_penalty=0.1,
                )

    def test_reports_missing_jax_explicitly(self):
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}, clear=True),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_gpu_search._load_jax",
                side_effect=RuntimeError("JAX is required for single-GPU search"),
            ),
            self.assertRaisesRegex(RuntimeError, "JAX is required"),
        ):
            SingleGpuExactSearch(
                *_tables(),
                physical_device_index=5,
                transition_penalty=0.1,
            )

    def test_rejects_missing_or_multiple_visible_gpus(self):
        for devices in ((), (_FakeDevice(), _FakeDevice(id=1))):
            with self.subTest(device_count=len(devices)):
                fake = _FakeJax(devices)
                with (
                    mock.patch.dict(
                        os.environ, {"CUDA_VISIBLE_DEVICES": "5"}, clear=True
                    ),
                    mock.patch(
                        "mm_sonic.hybrid_terrain_lmm_gpu_search._load_jax",
                        return_value=(fake, np),
                    ),
                    self.assertRaisesRegex(RuntimeError, "exactly one visible GPU"),
                ):
                    SingleGpuExactSearch(
                        *_tables(),
                        physical_device_index=5,
                        transition_penalty=0.1,
                    )

    def test_transfers_and_executes_only_on_the_selected_device(self):
        search, fake = _fake_search()
        selected = fake._devices[0]

        result = search.match_candidates(
            np.zeros(31, dtype=np.float64),
            current_range=1,
            active_contact_code=2,
            excluded_rows=(5,),
        )

        self.assertTrue(fake.device_put_calls)
        self.assertTrue(all(device is selected for _, device in fake.device_put_calls))
        self.assertEqual(fake.jit_devices, [selected])
        self.assertGreaterEqual(len(fake.compiled_calls), 2)
        self.assertTrue(all(len(arguments) == 8 for arguments in fake.compiled_calls))
        self.assertIs(fake.compiled_calls[-1][0], search._device_features)
        self.assertIs(fake.compiled_calls[-1][1], search._device_rows)
        self.assertIs(fake.compiled_calls[-1][2], search._device_range_ids)
        self.assertIs(fake.compiled_calls[-1][3], search._device_contact_codes)
        np.testing.assert_array_equal(result.rows, (3, 9))
        self.assertEqual(result.candidate_count, 7)
        self.assertEqual(result.close_candidate_count, 2)
        self.assertEqual(result.device_minimum_score, 1.25)
        self.assertGreaterEqual(result.elapsed_ms, 0.0)
        self.assertFalse(result.rows.flags.writeable)

    def test_rejects_non_finite_or_wrong_shape_queries(self):
        search, _ = _fake_search()
        for query in (
            np.zeros(30, dtype=np.float64),
            np.full(31, np.nan, dtype=np.float64),
            np.full(31, np.inf, dtype=np.float64),
        ):
            with (
                self.subTest(shape=query.shape),
                self.assertRaisesRegex(ValueError, "finite 31-D"),
            ):
                search.match_candidates(
                    query,
                    current_range=0,
                    active_contact_code=0,
                    excluded_rows=(),
                )

    def test_rejects_exclusions_beyond_fixed_budget(self):
        search, _ = _fake_search()
        with self.assertRaisesRegex(ValueError, "exclusion budget"):
            search.match_candidates(
                np.zeros(31, dtype=np.float64),
                current_range=0,
                active_contact_code=0,
                excluded_rows=(1, 3, 5),
            )

    def test_rejects_empty_compatible_candidate_set(self):
        fake = _FakeJax()
        fake.outputs = (
            _FakeArray((-1, -1)),
            _FakeArray(np.inf),
            _FakeArray(0),
            _FakeArray(0),
        )
        search, _ = _fake_search(fake)

        with self.assertRaisesRegex(ValueError, "no compatible candidate"):
            search.match_candidates(
                np.zeros(31, dtype=np.float64),
                current_range=0,
                active_contact_code=0,
                excluded_rows=(),
            )


class SingleGpuExactSearchGpuTests(unittest.TestCase):
    def _assert_gpu5_script(self, script: str, marker: str) -> None:
        root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "5"
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(root), str(root / "resources"), str(root / "sonic" / "python"))
        )
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn(marker, completed.stdout)

    def test_rejects_float64_feature_tables_instead_of_silently_narrowing(self):
        features, searchable_rows, row_ranges, contacts = _tables()
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}, clear=True),
            mock.patch(
                "mm_sonic.hybrid_terrain_lmm_gpu_search._load_jax",
                side_effect=AssertionError("dtype rejection must precede JAX import"),
            ),
            self.assertRaisesRegex(ValueError, "native float32"),
        ):
            SingleGpuExactSearch(
                features.astype(np.float64),
                searchable_rows,
                row_ranges,
                contacts,
                physical_device_index=5,
                transition_penalty=0.1,
            )

    def test_gpu5_handles_near_maximum_and_overflowed_scores(self):
        self._assert_gpu5_script(
            """
            import numpy as np

            from mm_sonic.hybrid_terrain_lmm_gpu_search import (
                SingleGpuExactSearch,
                configure_single_gpu_visibility,
            )

            _, physical_index = configure_single_gpu_visibility("cuda:5")
            maximum = np.finfo(np.float32).max
            near_maximum_delta = np.sqrt(
                maximum * np.float32(1.0 - 1.0e-5)
            )
            features = np.zeros((4, 31), dtype=np.float32)
            features[:, 0] = near_maximum_delta
            contacts = np.zeros((4, 2), dtype=np.bool_)
            contacts[2:, 0] = True
            search = SingleGpuExactSearch(
                features,
                np.arange(4, dtype=np.int64),
                np.zeros(4, dtype=np.int32),
                contacts,
                physical_device_index=physical_index,
                transition_penalty=0.1,
            )
            result = search.match_candidates(
                np.zeros(31, dtype=np.float64),
                current_range=0,
                active_contact_code=0,
                excluded_rows=(),
            )
            assert np.isfinite(result.device_minimum_score)
            assert result.candidate_count == 2
            assert result.close_candidate_count == 2
            np.testing.assert_array_equal(result.rows, (0, 1))

            overflow_features = np.zeros((2, 31), dtype=np.float32)
            overflow_features[:, 0] = maximum
            overflow_search = SingleGpuExactSearch(
                overflow_features,
                np.arange(2, dtype=np.int64),
                np.zeros(2, dtype=np.int32),
                np.zeros((2, 2), dtype=np.bool_),
                physical_device_index=physical_index,
                transition_penalty=0.1,
            )
            try:
                overflow_search.match_candidates(
                    np.zeros(31, dtype=np.float64),
                    current_range=0,
                    active_contact_code=0,
                    excluded_rows=(),
                )
            except FloatingPointError as error:
                assert "scores overflowed" in str(error)
            else:
                raise AssertionError("all-score overflow was not reported")
            print("GPU_SCORE_OVERFLOW_OK")
            """,
            "GPU_SCORE_OVERFLOW_OK",
        )

    def test_gpu5_constructed_controls_flip_winners_and_signal_tie_overflow(self):
        self._assert_gpu5_script(
            """
            import numpy as np

            from mm_sonic.hybrid_terrain_lmm_gpu_search import (
                SingleGpuExactSearch,
                configure_single_gpu_visibility,
            )

            _, physical_index = configure_single_gpu_visibility("cuda:5")
            features = np.zeros((2, 31), dtype=np.float32)
            features[0, 0] = np.sqrt(0.08)
            row_ranges = np.asarray((0, 1), dtype=np.int32)
            search = SingleGpuExactSearch(
                features,
                np.arange(2, dtype=np.int64),
                row_ranges,
                np.zeros((2, 2), dtype=np.bool_),
                physical_device_index=physical_index,
                transition_penalty=0.1,
            )
            query = np.zeros(31, dtype=np.float64)
            raw_scores = np.sum(np.square(features - query), axis=1)
            assert int(np.argmin(raw_scores)) == 1
            penalized = raw_scores + 0.1 * (row_ranges != 0)
            assert int(np.argmin(penalized)) == 0

            penalty_result = search.match_candidates(
                query,
                current_range=0,
                active_contact_code=0,
                excluded_rows=(),
            )
            np.testing.assert_array_equal(penalty_result.rows, (0,))
            exclusion_result = search.match_candidates(
                query,
                current_range=0,
                active_contact_code=0,
                excluded_rows=(0,),
            )
            np.testing.assert_array_equal(exclusion_result.rows, (1,))
            assert exclusion_result.candidate_count == 1

            tie_count = 140
            tie_search = SingleGpuExactSearch(
                np.zeros((tie_count, 31), dtype=np.float32),
                np.arange(tie_count, dtype=np.int64),
                np.zeros(tie_count, dtype=np.int32),
                np.zeros((tie_count, 2), dtype=np.bool_),
                physical_device_index=physical_index,
                transition_penalty=0.1,
            )
            ties = tie_search.match_candidates(
                query,
                current_range=0,
                active_contact_code=0,
                excluded_rows=(),
            )
            assert ties.candidate_count == tie_count
            assert ties.close_candidate_count == tie_count
            np.testing.assert_array_equal(ties.rows, np.arange(128))
            print("GPU_CONSTRUCTED_CONTROLS_OK")
            """,
            "GPU_CONSTRUCTED_CONTROLS_OK",
        )

    def test_gpu5_matches_randomized_and_adversarial_cpu_oracle(self):
        script = textwrap.dedent(
            """
            import numpy as np

            from mm_sonic.hybrid_terrain_lmm_gpu_search import (
                SingleGpuExactSearch,
                configure_single_gpu_visibility,
            )

            configured, physical_index = configure_single_gpu_visibility("cuda:5")
            assert configured == "cuda:5"

            rng = np.random.default_rng(20260810)
            corpus_rows = 521
            features = rng.normal(size=(corpus_rows, 31)).astype(np.float32)
            searchable_rows = np.arange(3, corpus_rows - 3, dtype=np.int64)
            row_ranges = rng.integers(0, 4, size=corpus_rows, dtype=np.int32)
            contacts = np.zeros((corpus_rows, 2), dtype=np.bool_)
            contact_codes = np.arange(corpus_rows, dtype=np.uint8) % 4
            contacts[:, 0] = (contact_codes & 1) != 0
            contacts[:, 1] = (contact_codes & 2) != 0
            transition_penalty = 0.1

            duplicate_rows = searchable_rows[12:14]
            features[duplicate_rows] = features[duplicate_rows[0]]
            row_ranges[duplicate_rows] = 2
            contacts[duplicate_rows] = (True, True)

            search = SingleGpuExactSearch(
                features,
                searchable_rows,
                row_ranges,
                contacts,
                physical_device_index=physical_index,
                transition_penalty=transition_penalty,
                exclusion_budget=32,
            )

            import jax

            devices = jax.devices("gpu")
            assert len(devices) == 1, devices
            assert devices[0].platform == "gpu"

            cases = []
            for contact_code in range(4):
                for current_range in range(4):
                    query = rng.normal(size=31).astype(np.float64)
                    exclusions = tuple(
                        int(row)
                        for row in searchable_rows[
                            (contact_codes[searchable_rows] == contact_code)
                        ][:2]
                    )
                    cases.append((query, current_range, contact_code, exclusions))
            cases.append(
                (
                    features[duplicate_rows[0]].astype(np.float64),
                    2,
                    3,
                    (),
                )
            )
            for _ in range(24):
                cases.append(
                    (
                        rng.normal(size=31).astype(np.float64),
                        int(rng.integers(0, 4)),
                        int(rng.integers(0, 4)),
                        (int(rng.choice(searchable_rows)),),
                    )
                )

            searchable_codes = (
                contacts[searchable_rows, 0].astype(np.uint8)
                | (contacts[searchable_rows, 1].astype(np.uint8) << 1)
            )
            for query, current_range, contact_code, exclusions in cases:
                eligible = searchable_codes == contact_code
                if exclusions:
                    eligible &= ~np.isin(searchable_rows, exclusions)
                rows = searchable_rows[eligible]
                assert len(rows)
                delta = features[rows].astype(np.float64) - query[None, :]
                scores = np.sum(np.square(delta), axis=1, dtype=np.float64)
                scores += transition_penalty * (row_ranges[rows] != current_range)
                order = np.lexsort((rows, scores))
                cpu_row = int(rows[order[0]])

                result = search.match_candidates(
                    query,
                    current_range=current_range,
                    active_contact_code=contact_code,
                    excluded_rows=exclusions,
                )
                assert result.candidate_count == len(rows)
                assert result.close_candidate_count >= len(result.rows)
                assert cpu_row in result.rows, (cpu_row, result)

                candidate_delta = (
                    features[result.rows].astype(np.float64) - query[None, :]
                )
                candidate_scores = np.sum(
                    np.square(candidate_delta), axis=1, dtype=np.float64
                )
                candidate_scores += transition_penalty * (
                    row_ranges[result.rows] != current_range
                )
                candidate_order = np.lexsort((result.rows, candidate_scores))
                assert int(result.rows[candidate_order[0]]) == cpu_row

            duplicate_result = search.match_candidates(
                features[duplicate_rows[0]].astype(np.float64),
                current_range=2,
                active_contact_code=3,
                excluded_rows=(),
            )
            assert set(map(int, duplicate_rows)).issubset(duplicate_result.rows)
            print(f"GPU_PARITY_OK cases={len(cases)} device={devices[0]}")
            """
        )
        self._assert_gpu5_script(script, "GPU_PARITY_OK cases=41")


if __name__ == "__main__":
    unittest.main()
