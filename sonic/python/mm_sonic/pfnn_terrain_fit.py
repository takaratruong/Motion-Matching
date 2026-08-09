"""Reproduce and serialize one fitted terrain from the released PFNN data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
from scipy import linalg
from scipy.spatial import distance


SCHEMA = "native-g1-pfnn-terrain-fit/v1"
PFNN_POSITION_SCALE = 5.6444
PFNN_PATCH_HSCALE = 3.937007874
PFNN_PATCH_VSCALE = 3.0
PFNN_WINDOW = 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_string(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a lowercase SHA-256 string") from error
    if value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 string")
    return value


def _finite_array(
    value: object, label: str, *, shape: tuple[int | None, ...]
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite real values")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f"{label} has invalid shape {array.shape}")
    result = np.asarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PFNNTerrainFit:
    """Safe numeric representation of PFNN's fitted continuous height function."""

    patch: np.ndarray
    patch_coord: np.ndarray
    contact_center_xz: np.ndarray
    patch_height_mean: float
    stance_height_mean: float
    rbf_centers_xz: np.ndarray
    rbf_epsilon: np.ndarray
    rbf_weights: np.ndarray
    source_contacts: np.ndarray
    source_start_frame: int
    source_frame_count: int
    cycle_start_frame: int
    cycle_stop_frame: int
    selected_patch_index: int
    fitting_error: float
    source_sha256: str
    patches_sha256: str

    def __post_init__(self) -> None:
        patch = _finite_array(self.patch, "patch", shape=(None, None))
        if min(patch.shape) < 2:
            raise ValueError("patch dimensions must be at least 2")
        coord = _finite_array(self.patch_coord, "patch_coord", shape=(4,))
        center = _finite_array(
            self.contact_center_xz, "contact_center_xz", shape=(2,)
        )
        centers = _finite_array(
            self.rbf_centers_xz, "rbf_centers_xz", shape=(None, 2)
        )
        if len(centers) < 1:
            raise ValueError("rbf_centers_xz must be nonempty")
        epsilon = _finite_array(
            self.rbf_epsilon, "rbf_epsilon", shape=(len(centers),)
        )
        weights = _finite_array(
            self.rbf_weights, "rbf_weights", shape=(1, len(centers))
        )
        contacts = np.asarray(self.source_contacts)
        if contacts.dtype != np.bool_ or contacts.shape != (
            self.source_frame_count,
            4,
        ):
            raise ValueError("source_contacts must have bool shape [source_frame_count, 4]")
        contacts = contacts.copy()
        contacts.setflags(write=False)
        for label, value in (
            ("source_start_frame", self.source_start_frame),
            ("source_frame_count", self.source_frame_count),
            ("cycle_start_frame", self.cycle_start_frame),
            ("cycle_stop_frame", self.cycle_stop_frame),
            ("selected_patch_index", self.selected_patch_index),
        ):
            if type(value) is not int:
                raise ValueError(f"{label} must be an integer")
        if self.source_start_frame < 0 or self.source_frame_count < 1:
            raise ValueError("source frame interval is invalid")
        if not 0 <= self.cycle_start_frame < self.cycle_stop_frame:
            raise ValueError("cycle frame interval is invalid")
        if self.selected_patch_index < 0:
            raise ValueError("selected_patch_index must be nonnegative")
        for label, value in (
            ("patch_height_mean", self.patch_height_mean),
            ("stance_height_mean", self.stance_height_mean),
            ("fitting_error", self.fitting_error),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.fitting_error < 0.0:
            raise ValueError("fitting_error must be nonnegative")
        object.__setattr__(self, "patch", patch)
        object.__setattr__(self, "patch_coord", coord)
        object.__setattr__(self, "contact_center_xz", center)
        object.__setattr__(self, "rbf_centers_xz", centers)
        object.__setattr__(self, "rbf_epsilon", epsilon)
        object.__setattr__(self, "rbf_weights", weights)
        object.__setattr__(self, "source_contacts", contacts)
        object.__setattr__(
            self, "source_sha256", _hash_string(self.source_sha256, "source_sha256")
        )
        object.__setattr__(
            self,
            "patches_sha256",
            _hash_string(self.patches_sha256, "patches_sha256"),
        )


@dataclass(frozen=True)
class _SelectedPatchFit:
    patch_index: int
    fitting_error: float
    contact_center_xz: np.ndarray
    patch_height_mean: float
    stance_height_mean: float
    residual: np.ndarray


def patch_height(
    patch: np.ndarray,
    query_xz: np.ndarray,
    *,
    hscale: float = PFNN_PATCH_HSCALE,
    vscale: float = PFNN_PATCH_VSCALE,
) -> np.ndarray:
    """Apply the released PFNN bilinear ``patchfunc`` to one height patch."""

    values = _finite_array(patch, "patch", shape=(None, None))
    query = _finite_array(query_xz, "query_xz", shape=(None, 2))
    if min(values.shape) < 2:
        raise ValueError("patch dimensions must be at least 2")
    if not np.isfinite(hscale) or hscale <= 0.0:
        raise ValueError("hscale must be positive and finite")
    if not np.isfinite(vscale):
        raise ValueError("vscale must be finite")
    coordinates = query / float(hscale) + np.array(
        [values.shape[0] // 2, values.shape[1] // 2], dtype=np.float64
    )
    fraction = np.fmod(coordinates, 1.0)
    lower = np.clip(
        np.floor(coordinates).astype(np.int64),
        0,
        np.array(values.shape, dtype=np.int64) - 1,
    )
    upper = np.clip(
        np.ceil(coordinates).astype(np.int64),
        0,
        np.array(values.shape, dtype=np.int64) - 1,
    )
    h00 = values[lower[:, 0], lower[:, 1]]
    h01 = values[lower[:, 0], upper[:, 1]]
    h10 = values[upper[:, 0], lower[:, 1]]
    h11 = values[upper[:, 0], upper[:, 1]]
    left = (1.0 - fraction[:, 0]) * h00 + fraction[:, 0] * h10
    right = (1.0 - fraction[:, 0]) * h01 + fraction[:, 0] * h11
    return float(vscale) * (
        (1.0 - fraction[:, 1]) * left + fraction[:, 1] * right
    )


def _patch_bank_height(
    patches: np.ndarray,
    query_xz: np.ndarray,
    *,
    hscale: float,
    vscale: float,
) -> np.ndarray:
    """Vectorized released ``patchfunc`` for fitting a complete patch bank."""

    bank = np.asarray(patches)
    query = _finite_array(query_xz, "query_xz", shape=(None, 2))
    if bank.ndim != 3 or min(bank.shape[1:]) < 2:
        raise ValueError("patches must have shape [P, H, W] with H,W >= 2")
    if bank.dtype.kind not in "iuf" or not np.isfinite(bank).all():
        raise ValueError("patches must contain finite real values")
    bank = np.asarray(bank, dtype=np.float64)
    coordinates = query / float(hscale) + np.array(
        [bank.shape[1] // 2, bank.shape[2] // 2], dtype=np.float64
    )
    fraction = np.fmod(coordinates, 1.0)
    lower = np.clip(
        np.floor(coordinates).astype(np.int64),
        0,
        np.array(bank.shape[1:], dtype=np.int64) - 1,
    )
    upper = np.clip(
        np.ceil(coordinates).astype(np.int64),
        0,
        np.array(bank.shape[1:], dtype=np.int64) - 1,
    )
    h00 = bank[:, lower[:, 0], lower[:, 1]]
    h01 = bank[:, lower[:, 0], upper[:, 1]]
    h10 = bank[:, upper[:, 0], lower[:, 1]]
    h11 = bank[:, upper[:, 0], upper[:, 1]]
    left = (1.0 - fraction[:, 0]) * h00 + fraction[:, 0] * h10
    right = (1.0 - fraction[:, 0]) * h01 + fraction[:, 0] * h11
    return float(vscale) * (
        (1.0 - fraction[:, 1]) * left + fraction[:, 1] * right
    )


def _fit_patch_bank(
    patches: np.ndarray,
    *,
    down_xz: np.ndarray,
    down_y: np.ndarray,
    up_xz: np.ndarray,
    up_y: np.ndarray,
    hscale: float = PFNN_PATCH_HSCALE,
    vscale: float = PFNN_PATCH_VSCALE,
) -> _SelectedPatchFit:
    """Select PFNN's lowest-error rocky terrain patch before RBF fitting."""

    down_position = _finite_array(down_xz, "down_xz", shape=(None, 2))
    down_height = _finite_array(down_y, "down_y", shape=(len(down_position), 1))
    up_position = _finite_array(up_xz, "up_xz", shape=(None, 2))
    up_height = _finite_array(up_y, "up_y", shape=(len(up_position), 1))
    if len(down_position) < 2:
        raise ValueError("terrain fitting requires at least two stance probes")
    center = down_position.mean(axis=0)
    stance_mean = float(down_height.mean(axis=0)[0])
    terrain_down = _patch_bank_height(
        patches,
        down_position - center,
        hscale=hscale,
        vscale=vscale,
    )
    terrain_mean = terrain_down.mean(axis=1)
    down_error = 0.1 * np.mean(
        (
            (terrain_down - terrain_mean[:, None])
            - (down_height[:, 0] - stance_mean)[None, :]
        )
        ** 2,
        axis=1,
    )
    if len(up_position):
        terrain_up = _patch_bank_height(
            patches,
            up_position - center,
            hscale=hscale,
            vscale=vscale,
        )
        up_error = np.mean(
            np.maximum(
                (terrain_up - terrain_mean[:, None])
                - (up_height[:, 0] - stance_mean)[None, :],
                0.0,
            )
            ** 2,
            axis=1,
        )
    else:
        up_error = np.zeros(len(terrain_down), dtype=np.float64)
    total = down_error + up_error
    if not np.isfinite(total).all():
        raise ValueError("terrain fitting errors must be finite")
    selected = int(np.argsort(total)[0])
    base_at_contacts = (
        terrain_down[selected] - terrain_mean[selected] + stance_mean
    )
    residual = down_height[:, 0] - base_at_contacts
    return _SelectedPatchFit(
        patch_index=selected,
        fitting_error=float(total[selected]),
        contact_center_xz=center,
        patch_height_mean=float(terrain_mean[selected]),
        stance_height_mean=stance_mean,
        residual=residual,
    )


def validate_footstep_cycle(
    footsteps: tuple[str, ...] | list[str], *, cycle_start: int, cycle_stop: int
) -> None:
    """Require exact consecutive first-column markers from PFNN's footstep file."""

    if type(cycle_start) is not int or type(cycle_stop) is not int:
        raise ValueError("cycle markers must be integers")
    starts: list[int] = []
    for line in footsteps:
        fields = line.split()
        if not fields:
            continue
        try:
            starts.append(int(fields[0]))
        except ValueError as error:
            raise ValueError("footstep marker is not an integer") from error
    for index in range(len(starts) - 1):
        if starts[index] == cycle_start and starts[index + 1] == cycle_stop:
            return
    raise ValueError("cycle markers must be consecutive PFNN footsteps")


def display_contacts(
    contacts_60hz: np.ndarray,
    *,
    display_start_frame: int,
    display_frame_count: int,
) -> np.ndarray:
    """Map PFNN's even-frame 60 Hz contacts back to native 120 Hz frames."""

    contacts = np.asarray(contacts_60hz)
    if contacts.dtype != np.bool_ or contacts.ndim != 2 or contacts.shape[1] != 4:
        raise ValueError("contacts_60hz must have bool shape [T, 4]")
    if type(display_start_frame) is not int or display_start_frame < 0:
        raise ValueError("display_start_frame must be a nonnegative integer")
    if type(display_frame_count) is not int or display_frame_count < 1:
        raise ValueError("display_frame_count must be a positive integer")
    indices = np.arange(
        display_start_frame,
        display_start_frame + display_frame_count,
        dtype=np.int64,
    ) // 2
    if len(contacts) == 0 or int(indices[-1]) >= len(contacts):
        raise ValueError("display interval exceeds the 60 Hz contact timeline")
    return contacts[indices].copy()


def _pfnn_modules(pfnn_root: Path):
    _install_numpy_umath_tests_compat()
    motion_root = pfnn_root / "motion"
    if not motion_root.is_dir():
        raise FileNotFoundError("PFNN motion utilities directory is required")
    sys.path.insert(0, str(motion_root))
    try:
        bvh = importlib.import_module("BVH")
        animation = importlib.import_module("Animation")
    finally:
        if sys.path and sys.path[0] == str(motion_root):
            sys.path.pop(0)
    return bvh, animation


def _install_numpy_umath_tests_compat() -> object:
    """Provide only the batched matmul symbol removed from modern NumPy."""

    name = "numpy.core.umath_tests"
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
    except RuntimeError as error:
        if "cannot load _umath_tests module" not in str(error):
            raise
    module = types.ModuleType(name)
    module.matrix_multiply = np.matmul
    sys.modules[name] = module
    setattr(np.core, "umath_tests", module)
    return module


def _global_positions(animation_module: object, animation: object) -> np.ndarray:
    transforms = animation_module.transforms_global(animation)
    positions = transforms[:, :, :3, 3] / transforms[:, :, 3:, 3]
    result = np.asarray(positions, dtype=np.float64)
    if result.ndim != 3 or result.shape[2] != 3 or not np.isfinite(result).all():
        raise ValueError("PFNN global positions are invalid")
    return result


def _pfnn_contacts(global_positions: np.ndarray) -> np.ndarray:
    if len(global_positions) < 2 or global_positions.shape[1] <= 10:
        raise ValueError("PFNN motion lacks the required foot timeline")
    foot_ids = np.array([4, 5, 9, 10], dtype=np.int64)
    delta = global_positions[1:, foot_ids] - global_positions[:-1, foot_ids]
    contacts = np.sum(delta**2, axis=2) < 0.02
    return np.concatenate((contacts, contacts[-1:]), axis=0)


def released_pfnn_contacts_for_interval(
    *,
    pfnn_root: Path,
    source: Path,
    display_start_frame: int,
    display_frame_count: int,
) -> np.ndarray:
    """Reconstruct released heel/toe contacts for one native 120 Hz interval."""

    root = Path(pfnn_root).resolve(strict=True)
    motion_source = Path(source)
    if not motion_source.is_absolute():
        motion_source = root / motion_source
    motion_source = motion_source.resolve(strict=True)
    bvh, animation_module = _pfnn_modules(root)
    animation, _names, _frametime = bvh.load(str(motion_source))
    animation.offsets *= PFNN_POSITION_SCALE
    animation.positions *= PFNN_POSITION_SCALE
    positions_60hz = _global_positions(animation_module, animation[::2])
    contacts_60hz = _pfnn_contacts(positions_60hz)
    return display_contacts(
        contacts_60hz,
        display_start_frame=display_start_frame,
        display_frame_count=display_frame_count,
    )


def _foot_samples(
    global_positions: np.ndarray, contacts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    foot_ids = (4, 5, 9, 10)
    offsets = (5.0, 4.0, 5.0, 4.0)
    down: list[np.ndarray] = []
    up: list[np.ndarray] = []
    for column, (joint, height) in enumerate(zip(foot_ids, offsets)):
        offset = np.array([0.0, height, 0.0], dtype=np.float64)
        down.append(global_positions[contacts[:, column], joint] - offset)
        up.append(global_positions[~contacts[:, column], joint] - offset)
    down_points = np.concatenate(down, axis=0)
    up_points = np.concatenate(up, axis=0)
    return (
        down_points[:, [0, 2]],
        down_points[:, 1:2],
        up_points[:, [0, 2]],
        up_points[:, 1:2],
    )


def _fit_linear_rbf(
    centers_xz: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distances = distance.cdist(centers_xz, centers_xz)
    mean_distance = float(distances.mean())
    if not np.isfinite(mean_distance) or mean_distance <= 0.0:
        raise ValueError("PFNN stance probes cannot fit a linear RBF")
    epsilon = np.ones(len(distances), dtype=np.float64) / mean_distance
    kernel = epsilon * distances
    weights = linalg.lu_solve(
        linalg.lu_factor(kernel.T - np.eye(len(kernel)) * 0.1),
        residual[:, None],
    ).T
    if not np.isfinite(weights).all():
        raise ValueError("PFNN linear RBF weights are nonfinite")
    return epsilon, weights


def fit_terrain_cycle(
    *,
    pfnn_root: Path,
    patches_path: Path,
    source: Path,
    cycle_start: int,
    cycle_stop: int,
    display_start: int,
    display_count: int,
) -> PFNNTerrainFit:
    """Reproduce the rank-zero rocky fit for one released PFNN footstep cycle."""

    root = Path(pfnn_root).resolve(strict=True)
    patch_source = Path(patches_path).resolve(strict=True)
    motion_source = Path(source)
    if not motion_source.is_absolute():
        motion_source = root / motion_source
    motion_source = motion_source.resolve(strict=True)
    footstep_source = motion_source.with_name(
        f"{motion_source.stem}_footsteps.txt"
    ).resolve(strict=True)
    footsteps = tuple(footstep_source.read_text().splitlines())
    validate_footstep_cycle(
        footsteps, cycle_start=cycle_start, cycle_stop=cycle_stop
    )
    if type(display_count) is not int or display_count < 1:
        raise ValueError("display_count must be a positive integer")
    if not cycle_start <= display_start < cycle_stop:
        raise ValueError("display interval must begin inside the selected cycle")

    bvh, animation_module = _pfnn_modules(root)
    animation, _names, _frametime = bvh.load(str(motion_source))
    source_frames = len(animation)
    if display_start + display_count > source_frames:
        raise ValueError("display interval exceeds the PFNN source timeline")
    animation.offsets *= PFNN_POSITION_SCALE
    animation.positions *= PFNN_POSITION_SCALE
    animation_60hz = animation[::2]
    expanded_start = cycle_start // 2 - PFNN_WINDOW
    expanded_stop = cycle_stop // 2 + PFNN_WINDOW + 1
    if expanded_start < 0 or expanded_stop > len(animation_60hz):
        raise ValueError("footstep cycle lacks PFNN's complete fitting window")
    fit_animation = animation_60hz[expanded_start:expanded_stop]
    fit_positions = _global_positions(animation_module, fit_animation)
    fit_contacts = _pfnn_contacts(fit_positions)
    down_xz, down_y, up_xz, up_y = _foot_samples(
        fit_positions, fit_contacts
    )

    with np.load(patch_source, allow_pickle=False) as archive:
        if set(archive.files) != {"X", "C"}:
            raise ValueError("PFNN patches archive must contain exactly X and C")
        patches = np.asarray(archive["X"], dtype=np.float64)
        patch_coords = np.asarray(archive["C"], dtype=np.float64)
    if patches.ndim != 3 or patch_coords.shape != (len(patches), 4):
        raise ValueError("PFNN patch arrays have invalid shapes")
    if len(patches) < 1 or not np.isfinite(patches).all() or not np.isfinite(
        patch_coords
    ).all():
        raise ValueError("PFNN patch arrays must be nonempty and finite")
    selected = _fit_patch_bank(
        patches,
        down_xz=down_xz,
        down_y=down_y,
        up_xz=up_xz,
        up_y=up_y,
    )
    epsilon, weights = _fit_linear_rbf(down_xz, selected.residual)

    full_positions = _global_positions(animation_module, animation_60hz)
    full_contacts = _pfnn_contacts(full_positions)
    contacts = display_contacts(
        full_contacts,
        display_start_frame=display_start,
        display_frame_count=display_count,
    )
    return PFNNTerrainFit(
        patch=patches[selected.patch_index],
        patch_coord=patch_coords[selected.patch_index],
        contact_center_xz=selected.contact_center_xz,
        patch_height_mean=selected.patch_height_mean,
        stance_height_mean=selected.stance_height_mean,
        rbf_centers_xz=down_xz,
        rbf_epsilon=epsilon,
        rbf_weights=weights,
        source_contacts=contacts,
        source_start_frame=display_start,
        source_frame_count=display_count,
        cycle_start_frame=cycle_start,
        cycle_stop_frame=cycle_stop,
        selected_patch_index=selected.patch_index,
        fitting_error=selected.fitting_error,
        source_sha256=_sha256(motion_source),
        patches_sha256=_sha256(patch_source),
    )


def terrain_height_pfnn(fit: PFNNTerrainFit, query_xz: np.ndarray) -> np.ndarray:
    """Evaluate the fitted continuous terrain in PFNN's processed units."""

    query = _finite_array(query_xz, "query_xz", shape=(None, 2))
    base = patch_height(fit.patch, query - fit.contact_center_xz)
    base = base - fit.patch_height_mean + fit.stance_height_mean
    distances = distance.cdist(query, fit.rbf_centers_xz)
    residual = (fit.rbf_weights @ (fit.rbf_epsilon * distances).T).T[:, 0]
    result = base + residual
    if not np.isfinite(result).all():
        raise ValueError("terrain query produced nonfinite heights")
    return result


def terrain_height_g1(
    fit: PFNNTerrainFit,
    query_xy_m: np.ndarray,
    *,
    scale: float = 1.0,
    z_offset: float = 0.0,
) -> np.ndarray:
    """Evaluate the same physical surface in GMR's Z-up G1 world."""

    query = _finite_array(query_xy_m, "query_xy_m", shape=(None, 2))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("terrain scale must be positive and finite")
    if not np.isfinite(z_offset):
        raise ValueError("terrain Z offset must be finite")
    source_xz = np.column_stack(
        (100.0 * query[:, 0] / scale, -100.0 * query[:, 1] / scale)
    )
    return scale * terrain_height_pfnn(fit, source_xz) / 100.0 + z_offset


_ARTIFACT_FIELDS = {
    "schema",
    "patch",
    "patch_coord",
    "contact_center_xz",
    "patch_height_mean",
    "stance_height_mean",
    "rbf_centers_xz",
    "rbf_epsilon",
    "rbf_weights",
    "source_contacts",
    "source_start_frame",
    "source_frame_count",
    "cycle_start_frame",
    "cycle_stop_frame",
    "selected_patch_index",
    "fitting_error",
    "source_sha256",
    "patches_sha256",
}


def save_terrain_fit(path: Path, fit: PFNNTerrainFit) -> Path:
    """Atomically save a validated, pickle-free terrain fit and receipt."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(SCHEMA),
                patch=fit.patch,
                patch_coord=fit.patch_coord,
                contact_center_xz=fit.contact_center_xz,
                patch_height_mean=np.asarray(fit.patch_height_mean),
                stance_height_mean=np.asarray(fit.stance_height_mean),
                rbf_centers_xz=fit.rbf_centers_xz,
                rbf_epsilon=fit.rbf_epsilon,
                rbf_weights=fit.rbf_weights,
                source_contacts=fit.source_contacts,
                source_start_frame=np.asarray(fit.source_start_frame),
                source_frame_count=np.asarray(fit.source_frame_count),
                cycle_start_frame=np.asarray(fit.cycle_start_frame),
                cycle_stop_frame=np.asarray(fit.cycle_stop_frame),
                selected_patch_index=np.asarray(fit.selected_patch_index),
                fitting_error=np.asarray(fit.fitting_error),
                source_sha256=np.asarray(fit.source_sha256),
                patches_sha256=np.asarray(fit.patches_sha256),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = path.with_suffix(".receipt.json")
    document = {
        "schema": SCHEMA,
        "status": "accepted",
        "artifact_sha256": _sha256(path),
        "source_sha256": fit.source_sha256,
        "patches_sha256": fit.patches_sha256,
        "selected_patch_index": fit.selected_patch_index,
        "patch_coord": fit.patch_coord.tolist(),
        "fitting_error": fit.fitting_error,
        "source_start_frame": fit.source_start_frame,
        "source_frame_count": fit.source_frame_count,
        "cycle_start_frame": fit.cycle_start_frame,
        "cycle_stop_frame": fit.cycle_stop_frame,
    }
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    receipt.write_bytes(payload)
    return receipt


def load_terrain_fit(path: Path) -> PFNNTerrainFit:
    """Load and strictly validate a pickle-free terrain artifact."""

    source = Path(path).resolve(strict=True)
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != _ARTIFACT_FIELDS:
            raise ValueError("terrain artifact fields do not match schema")
        if str(np.asarray(archive["schema"]).item()) != SCHEMA:
            raise ValueError("terrain artifact schema is invalid")
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}
    return PFNNTerrainFit(
        patch=values["patch"],
        patch_coord=values["patch_coord"],
        contact_center_xz=values["contact_center_xz"],
        patch_height_mean=float(values["patch_height_mean"].item()),
        stance_height_mean=float(values["stance_height_mean"].item()),
        rbf_centers_xz=values["rbf_centers_xz"],
        rbf_epsilon=values["rbf_epsilon"],
        rbf_weights=values["rbf_weights"],
        source_contacts=values["source_contacts"],
        source_start_frame=int(values["source_start_frame"].item()),
        source_frame_count=int(values["source_frame_count"].item()),
        cycle_start_frame=int(values["cycle_start_frame"].item()),
        cycle_stop_frame=int(values["cycle_stop_frame"].item()),
        selected_patch_index=int(values["selected_patch_index"].item()),
        fitting_error=float(values["fitting_error"].item()),
        source_sha256=str(values["source_sha256"].item()),
        patches_sha256=str(values["patches_sha256"].item()),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfnn-root", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cycle-start", type=int, required=True)
    parser.add_argument("--cycle-stop", type=int, required=True)
    parser.add_argument("--display-start", type=int, required=True)
    parser.add_argument("--display-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    fit = fit_terrain_cycle(
        pfnn_root=arguments.pfnn_root,
        patches_path=arguments.patches,
        source=arguments.source,
        cycle_start=arguments.cycle_start,
        cycle_stop=arguments.cycle_stop,
        display_start=arguments.display_start,
        display_count=arguments.display_count,
    )
    receipt = save_terrain_fit(arguments.output, fit)
    print(
        json.dumps(
            {
                "status": "accepted",
                "output": str(arguments.output.resolve()),
                "receipt": str(receipt.resolve()),
                "selected_patch_index": fit.selected_patch_index,
                "patch_coord": fit.patch_coord.tolist(),
                "fitting_error": fit.fitting_error,
                "stance_probe_count": len(fit.rbf_centers_xz),
                "source_start_frame": fit.source_start_frame,
                "source_frame_count": fit.source_frame_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
