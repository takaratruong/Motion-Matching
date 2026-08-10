"""Lazy, exhaustive float64 motion search on one isolated GPU."""

from __future__ import annotations

import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

_FEATURE_COUNT = 31
_MAX_RETURNED_CANDIDATES = 128


@dataclass(frozen=True)
class GpuSearchCandidates:
    """Bounded near-minimum rows returned by the exhaustive device scorer."""

    rows: np.ndarray
    candidate_count: int
    device_minimum_score: float
    close_candidate_count: int
    elapsed_ms: float

    def __post_init__(self) -> None:
        rows = np.array(self.rows, dtype=np.int64, copy=True)
        rows.setflags(write=False)
        object.__setattr__(self, "rows", rows)


def configure_single_gpu_visibility(spec: str) -> tuple[str, int]:
    """Pin JAX visibility to one physical CUDA index before JAX is imported."""

    match = re.fullmatch(r"cuda:([0-9]+)", str(spec))
    if match is None:
        raise ValueError("search device must be cuda:<physical-index>")
    index = int(match.group(1))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, str(index)):
        raise RuntimeError("CUDA visibility conflicts with the requested search GPU")
    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before single-GPU isolation")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    return f"cuda:{index}", index


def _load_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("JAX is required for single-GPU search") from error
    return jax, jnp


def _integer_vector(
    values: object, *, name: str, length: int | None = None
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain one value per corpus row")
    return result


class SingleGpuExactSearch:
    """Exhaustively score every searchable row on exactly one visible GPU."""

    def __init__(
        self,
        features: object,
        searchable_rows: object,
        row_ranges: object,
        contacts: object,
        *,
        physical_device_index: int,
        transition_penalty: float,
        exclusion_budget: int = 32,
    ) -> None:
        if (
            not isinstance(physical_device_index, (int, np.integer))
            or isinstance(physical_device_index, (bool, np.bool_))
            or int(physical_device_index) < 0
        ):
            raise ValueError("physical GPU index must be a nonnegative integer")
        physical_device_index = int(physical_device_index)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_device_index):
            raise RuntimeError(
                "CUDA visibility must name only the requested physical search GPU"
            )
        if type(exclusion_budget) is not int or exclusion_budget < 1:
            raise ValueError("exclusion budget must be a positive integer")
        try:
            penalty = float(transition_penalty)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "transition penalty must be finite and nonnegative"
            ) from error
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError("transition penalty must be finite and nonnegative")

        feature_values = np.asarray(features)
        if feature_values.ndim != 2 or feature_values.shape[1] != _FEATURE_COUNT:
            raise ValueError("features must have shape (rows, 31)")
        corpus_rows = len(feature_values)
        rows = _integer_vector(searchable_rows, name="searchable rows").astype(
            np.int64, copy=False
        )
        if not len(rows):
            raise ValueError("searchable rows must not be empty")
        if rows[0] < 0 or rows[-1] >= corpus_rows:
            raise IndexError("searchable row is outside the feature corpus")
        if np.any(rows[1:] <= rows[:-1]):
            raise ValueError("searchable rows must be sorted and unique")
        ranges = _integer_vector(
            row_ranges, name="row ranges", length=corpus_rows
        ).astype(np.int32, copy=False)
        if np.any(ranges < 0):
            raise ValueError("row ranges must be nonnegative")
        contact_values = np.asarray(contacts)
        if contact_values.shape != (corpus_rows, 2):
            raise ValueError("contacts must have shape (rows, 2)")
        if not np.all((contact_values == 0) | (contact_values == 1)):
            raise ValueError("contacts must contain only booleans")

        searchable_features = np.asarray(feature_values[rows], dtype=np.float32)
        if not np.isfinite(searchable_features).all():
            raise ValueError("searchable features must be finite")
        searchable_ranges = np.asarray(ranges[rows], dtype=np.int32)
        searchable_contacts = np.asarray(contact_values[rows], dtype=np.uint8)
        contact_codes = searchable_contacts[:, 0] | (searchable_contacts[:, 1] << 1)

        jax, jnp = _load_jax()
        try:
            devices = tuple(jax.devices("gpu"))
        except Exception as error:
            raise RuntimeError(
                "requested single-GPU search device is unavailable"
            ) from error
        if len(devices) != 1 or getattr(devices[0], "platform", None) != "gpu":
            raise RuntimeError("single-GPU search requires exactly one visible GPU")
        if not bool(getattr(jax.config, "x64_enabled", False)):
            raise RuntimeError("single-GPU exact search requires JAX float64 support")

        self.physical_device_index = physical_device_index
        self.transition_penalty = penalty
        self.exclusion_budget = exclusion_budget
        self._jax = jax
        self._device = devices[0]
        self._rows = np.array(rows, dtype=np.int64, copy=True)
        self._rows.setflags(write=False)
        self._corpus_row_count = corpus_rows

        warm_query = np.asarray(searchable_features[0], dtype=np.float64)
        device_features_float32 = jax.device_put(searchable_features, self._device)
        self._device_features = device_features_float32.astype(jnp.float64)
        self._device_rows = jax.device_put(self._rows, self._device)
        self._device_range_ids = jax.device_put(searchable_ranges, self._device)
        self._device_contact_codes = jax.device_put(contact_codes, self._device)

        top_count = min(_MAX_RETURNED_CANDIDATES, len(self._rows))

        def kernel(
            resident_features,
            global_rows,
            range_ids,
            resident_contact_codes,
            query,
            current_range,
            active_contact_code,
            excluded_positions,
        ):
            scores = jnp.sum(jnp.square(resident_features - query[None, :]), axis=1)
            scores += penalty * (range_ids != current_range)
            scores = jnp.where(
                resident_contact_codes == active_contact_code, scores, jnp.inf
            )
            padded = jnp.concatenate(
                (scores, jnp.asarray((jnp.inf,), dtype=jnp.float64))
            )
            safe_exclusions = jnp.where(
                excluded_positions >= 0, excluded_positions, len(scores)
            )
            padded = padded.at[safe_exclusions].set(jnp.inf)
            scores = padded[:-1]
            values, positions = jax.lax.top_k(-scores, top_count)
            candidate_scores = -values
            minimum = candidate_scores[0]
            tolerance = (
                jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, jnp.abs(minimum)) * 512
            )
            close = jnp.isfinite(candidate_scores) & (
                candidate_scores <= minimum + tolerance
            )
            candidate_rows = jnp.where(
                close,
                global_rows[positions],
                jnp.asarray(-1, dtype=global_rows.dtype),
            )
            close_count = jnp.count_nonzero(scores <= minimum + tolerance)
            candidate_count = jnp.count_nonzero(jnp.isfinite(scores))
            return candidate_rows, minimum, close_count, candidate_count

        self._kernel = jax.jit(kernel, device=self._device)
        del device_features_float32, searchable_features
        warm_exclusions = np.full(self.exclusion_budget, -1, dtype=np.int64)
        warm_outputs = self._execute(
            warm_query,
            current_range=int(searchable_ranges[0]),
            active_contact_code=int(contact_codes[0]),
            excluded_positions=warm_exclusions,
        )
        self._synchronize(warm_outputs)

    @staticmethod
    def _synchronize(outputs: tuple[object, ...]) -> None:
        for output in outputs:
            output.block_until_ready()

    def _execute(
        self,
        query: np.ndarray,
        *,
        current_range: int,
        active_contact_code: int,
        excluded_positions: np.ndarray,
    ) -> tuple[object, ...]:
        device_query = self._jax.device_put(query, self._device)
        device_range = self._jax.device_put(
            np.asarray(current_range, dtype=np.int32), self._device
        )
        device_contact = self._jax.device_put(
            np.asarray(active_contact_code, dtype=np.uint8), self._device
        )
        device_exclusions = self._jax.device_put(excluded_positions, self._device)
        return self._kernel(
            self._device_features,
            self._device_rows,
            self._device_range_ids,
            self._device_contact_codes,
            device_query,
            device_range,
            device_contact,
            device_exclusions,
        )

    def _excluded_positions(self, excluded_rows: object) -> np.ndarray:
        try:
            values = tuple(excluded_rows)
        except TypeError as error:
            raise TypeError(
                "excluded rows must be an iterable of row integers"
            ) from error
        if len(values) > self.exclusion_budget:
            raise ValueError("excluded rows exceed the fixed exclusion budget")
        normalized: list[int] = []
        for value in values:
            if not isinstance(value, (int, np.integer)) or isinstance(
                value, (bool, np.bool_)
            ):
                raise TypeError("excluded rows must contain only row integers")
            row = int(value)
            if not 0 <= row < self._corpus_row_count:
                raise IndexError("excluded row is outside the corpus")
            normalized.append(row)
        positions = np.full(self.exclusion_budget, -1, dtype=np.int64)
        if not normalized:
            return positions
        unique = np.unique(np.asarray(normalized, dtype=np.int64))
        indices = np.searchsorted(self._rows, unique)
        present = indices < len(self._rows)
        present[present] &= self._rows[indices[present]] == unique[present]
        matched = indices[present]
        positions[: len(matched)] = matched
        return positions

    def match_candidates(
        self,
        query: object,
        *,
        current_range: int,
        active_contact_code: int,
        excluded_rows: object,
    ) -> GpuSearchCandidates:
        """Return all device candidates inside the conservative roundoff window."""

        value = np.asarray(query, dtype=np.float64)
        if value.shape != (_FEATURE_COUNT,) or not np.isfinite(value).all():
            raise ValueError("search query must be one finite 31-D row")
        if not isinstance(current_range, (int, np.integer)) or isinstance(
            current_range, (bool, np.bool_)
        ):
            raise TypeError("current range must be an integer")
        if (
            not isinstance(active_contact_code, (int, np.integer))
            or isinstance(active_contact_code, (bool, np.bool_))
            or not 0 <= int(active_contact_code) <= 3
        ):
            raise ValueError("active contact code must be an integer from 0 through 3")
        excluded_positions = self._excluded_positions(excluded_rows)

        started = time.perf_counter()
        outputs = self._execute(
            value,
            current_range=int(current_range),
            active_contact_code=int(active_contact_code),
            excluded_positions=excluded_positions,
        )
        self._synchronize(outputs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        candidate_count = int(np.asarray(outputs[3]))
        minimum = float(np.asarray(outputs[1]))
        if candidate_count < 1 or not math.isfinite(minimum):
            raise ValueError("single-GPU search found no compatible candidate")
        close_count = int(np.asarray(outputs[2]))
        rows = np.asarray(outputs[0], dtype=np.int64)
        rows = np.sort(rows[rows >= 0])
        return GpuSearchCandidates(
            rows=rows,
            candidate_count=candidate_count,
            device_minimum_score=minimum,
            close_candidate_count=close_count,
            elapsed_ms=elapsed_ms,
        )
