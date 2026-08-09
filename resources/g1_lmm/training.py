"""Deterministic training helpers and Orange Duck network binary I/O."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

import numpy as np

_DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _DETERMINISTIC_CUBLAS_WORKSPACE

import torch
from torch import nn

_CUDA_INITIALIZED_AT_IMPORT = torch.cuda.is_initialized()
_CUDA_DETERMINISM_CONFIGURED = False

from resources import quat, txform

from .dataset import (
    G1LmmDimensions,
    TrainingArrays,
    TrainingBundle,
    build_training_arrays,
    load_training_bundle,
    range_safe_windows,
)
from .models import Compressor, Decompressor, Projector, Stepper


class ProjectorTargets(NamedTuple):
    indices: np.ndarray
    features: np.ndarray
    latent: np.ndarray
    distance: np.ndarray


class TrainingGateError(RuntimeError):
    """A staged training gate failed after publishing its immutable receipt."""

    def __init__(self, stage: str, output: Path) -> None:
        super().__init__(f"{stage} gate failed; later stages were not started (receipt: {output})")
        self.stage = stage
        self.output = output


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1234
    device: str = "cuda:1"
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    overfit_steps: int = 1000
    decompressor_steps: int = 50_000
    stepper_steps: int = 100_000
    projector_steps: int = 50_000
    stepper_window: int = 20
    withheld_frames: int = 64
    withheld_halo: int = 60
    dt: float = 1.0 / 60.0

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "batch_size",
            "overfit_steps",
            "decompressor_steps",
            "stepper_steps",
            "projector_steps",
            "stepper_window",
            "withheld_frames",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.withheld_halo < 0:
            raise ValueError("withheld_halo must be non-negative")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.dt) or self.dt != 1.0 / 60.0:
            raise ValueError("dt must be exactly 1/60 for the 60 Hz model ABI")


def deterministic_withheld_ranges(
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    *,
    frames: int,
    seed: int,
    margin: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one reproducible contiguous withheld block per source range."""

    starts = np.asarray(range_starts, dtype=np.int64)
    stops = np.asarray(range_stops, dtype=np.int64)
    if starts.ndim != 1 or starts.shape != stops.shape or len(starts) == 0:
        raise ValueError("range starts/stops must be aligned vectors")
    if frames <= 0 or margin < 0 or seed < 0:
        raise ValueError("frames must be positive and margin/seed non-negative")
    if np.any(starts < 0) or np.any(stops <= starts) or np.any(starts[1:] < stops[:-1]):
        raise ValueError("ranges must be ordered, non-overlapping, and non-empty")
    generator = np.random.default_rng(seed)
    withheld_starts = []
    for start, stop in zip(starts, stops):
        first = int(start) + margin
        last = int(stop) - margin - frames
        if last < first:
            raise ValueError("range is too short for the withheld block and halo")
        withheld_starts.append(int(generator.integers(first, last + 1)))
    result_starts = np.asarray(withheld_starts, dtype=np.int64)
    return result_starts, result_starts + int(frames)


@dataclass(frozen=True)
class ExportedLayer:
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class ExportedNetwork:
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    layers: tuple[ExportedLayer, ...]

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        output = (np.asarray(values, dtype=np.float32) - self.input_mean) / self.input_std
        for index, layer in enumerate(self.layers):
            output = output @ layer.weight.T + layer.bias
            if index + 1 != len(self.layers):
                output = np.maximum(output, np.float32(0.0))
        return (output * self.output_std + self.output_mean).astype(np.float32, copy=False)


def _float_vector(name: str, values: np.ndarray | torch.Tensor, *, positive: bool = False) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    vector = np.asarray(values, dtype="<f4")
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    if positive and np.any(vector <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return np.ascontiguousarray(vector)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def export_network(
    path: str | Path,
    layers: Iterable[nn.Linear],
    *,
    input_mean: np.ndarray | torch.Tensor,
    input_std: np.ndarray | torch.Tensor,
    output_mean: np.ndarray | torch.Tensor,
    output_std: np.ndarray | torch.Tensor,
) -> None:
    """Write the exact Orange Duck matrix order using explicit little endian."""

    layer_list = tuple(layers)
    if not layer_list or any(not isinstance(layer, nn.Linear) for layer in layer_list):
        raise ValueError("layers must contain at least one torch.nn.Linear")
    means_in = _float_vector("input_mean", input_mean)
    stds_in = _float_vector("input_std", input_std, positive=True)
    means_out = _float_vector("output_mean", output_mean)
    stds_out = _float_vector("output_std", output_std, positive=True)
    if len(means_in) != len(stds_in) or len(means_out) != len(stds_out):
        raise ValueError("normalization vector dimensions do not match")
    if layer_list[0].in_features != len(means_in) or layer_list[-1].out_features != len(means_out):
        raise ValueError("normalization vectors do not match network dimensions")

    payload = bytearray()
    for vector in (means_in, stds_in, means_out, stds_out):
        payload.extend(struct.pack("<I", len(vector)))
        payload.extend(vector.tobytes(order="C"))
    payload.extend(struct.pack("<I", len(layer_list)))
    previous = len(means_in)
    with torch.no_grad():
        for layer in layer_list:
            if layer.in_features != previous:
                raise ValueError("network layers are not contiguous")
            weight = np.ascontiguousarray(layer.weight.detach().cpu().numpy().T, dtype="<f4")
            bias = np.ascontiguousarray(layer.bias.detach().cpu().numpy(), dtype="<f4")
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError("network contains non-finite parameters")
            payload.extend(struct.pack("<II", *weight.shape))
            payload.extend(weight.tobytes(order="C"))
            payload.extend(struct.pack("<I", len(bias)))
            payload.extend(bias.tobytes(order="C"))
            previous = layer.out_features
    _atomic_write(Path(path), bytes(payload))


class _BinaryCursor:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def unpack(self, fmt: str) -> tuple[int, ...]:
        size = struct.calcsize(fmt)
        if self.offset + size > len(self.payload):
            raise ValueError("truncated network binary")
        result = struct.unpack_from(fmt, self.payload, self.offset)
        self.offset += size
        return result

    def floats(self, count: int) -> np.ndarray:
        if count <= 0 or self.offset + count * 4 > len(self.payload):
            raise ValueError("truncated or invalid network vector")
        result = np.frombuffer(self.payload, dtype="<f4", count=count, offset=self.offset).copy()
        self.offset += count * 4
        if not np.isfinite(result).all():
            raise ValueError("network binary contains non-finite values")
        return result


def load_exported_network(
    path: str | Path,
    *,
    expected_layers: Sequence[tuple[int, int]] | None = None,
) -> ExportedNetwork:
    cursor = _BinaryCursor(Path(path).read_bytes())

    vectors = []
    for _ in range(4):
        (size,) = cursor.unpack("<I")
        vectors.append(cursor.floats(size))
    input_mean, input_std, output_mean, output_std = vectors
    if len(input_mean) != len(input_std) or len(output_mean) != len(output_std):
        raise ValueError("network normalization dimensions disagree")
    if np.any(input_std <= 0.0) or np.any(output_std <= 0.0):
        raise ValueError("network normalization scales must be positive")

    (layer_count,) = cursor.unpack("<I")
    if layer_count <= 0:
        raise ValueError("network must contain at least one layer")
    if expected_layers is not None and layer_count != len(expected_layers):
        raise ValueError("network layer count does not match expectation")
    layers = []
    previous = len(input_mean)
    for index in range(layer_count):
        rows, columns = cursor.unpack("<II")
        if rows != previous or rows <= 0 or columns <= 0:
            raise ValueError("network matrix dimensions are invalid")
        stored = cursor.floats(rows * columns).reshape(rows, columns)
        (bias_size,) = cursor.unpack("<I")
        if bias_size != columns:
            raise ValueError("network bias dimension is invalid")
        bias = cursor.floats(bias_size)
        if expected_layers is not None and (rows, columns) != tuple(expected_layers[index]):
            raise ValueError("network layer dimensions do not match expectation")
        layers.append(ExportedLayer(weight=stored.T.copy(), bias=bias))
        previous = columns
    if previous != len(output_mean):
        raise ValueError("network output dimension does not match normalization")
    if cursor.offset != len(cursor.payload):
        raise ValueError("network binary has trailing bytes")
    return ExportedNetwork(input_mean, input_std, output_mean, output_std, tuple(layers))


def normalized_projector_targets(
    query_normalized: np.ndarray,
    features_normalized: np.ndarray,
    latent: np.ndarray,
) -> ProjectorTargets:
    query = np.asarray(query_normalized, dtype=np.float32)
    features = np.asarray(features_normalized, dtype=np.float32)
    latent_values = np.asarray(latent, dtype=np.float32)
    if query.ndim != 2 or features.ndim != 2 or latent_values.ndim != 2:
        raise ValueError("projector arrays must be two-dimensional")
    if query.shape[1] != features.shape[1] or len(features) != len(latent_values):
        raise ValueError("projector array dimensions do not agree")
    if len(features) == 0 or not all(np.isfinite(values).all() for values in (query, features, latent_values)):
        raise ValueError("projector arrays must be non-empty and finite")
    delta = query[:, None, :] - features[None, :, :]
    all_distances = np.sqrt(np.sum(delta * delta, axis=2))
    indices = np.argmin(all_distances, axis=1).astype(np.int64)
    return ProjectorTargets(
        indices=indices,
        features=features[indices].copy(),
        latent=latent_values[indices].copy(),
        distance=all_distances[np.arange(len(query)), indices].astype(np.float32, copy=False),
    )


def _configure_determinism(seed: int, device: torch.device) -> None:
    global _CUDA_DETERMINISM_CONFIGURED
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported deterministic training device: {device}")
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _DETERMINISTIC_CUBLAS_WORKSPACE:
            raise RuntimeError(
                "CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization"
            )
        if not _CUDA_DETERMINISM_CONFIGURED and (
            _CUDA_INITIALIZED_AT_IMPORT or torch.cuda.is_initialized()
        ):
            raise RuntimeError(
                "CUDA was initialized before fail-closed deterministic training configuration"
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA training device is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA training device is unavailable: {device}")
        _CUDA_DETERMINISM_CONFIGURED = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cpu":
        torch.set_num_threads(1)


def _safe_std(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).std(axis=axis, dtype=np.float64).astype(np.float32)
    return np.maximum(result, np.float32(1.0e-5))


@dataclass(frozen=True)
class AutoencoderNormalization:
    compressor_mean: np.ndarray
    compressor_std: np.ndarray
    decompressor_mean: np.ndarray
    decompressor_std: np.ndarray


def orange_duck_decompressor_losses(
    prediction: torch.Tensor,
    latent: torch.Tensor,
    *,
    local_positions: torch.Tensor,
    local_rotation_xy: torch.Tensor,
    local_velocities: torch.Tensor,
    local_angular_velocities: torch.Tensor,
    character_positions: torch.Tensor,
    character_transforms: torch.Tensor,
    character_velocities: torch.Tensor,
    character_angular_velocities: torch.Tensor,
    root_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    contacts: torch.Tensor,
    parents: np.ndarray,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Compute the exact denormalized Orange Duck decompressor objective."""

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("decompressor loss dt must be positive and finite")
    if prediction.ndim != 3 or latent.ndim != 3 or prediction.shape[:2] != latent.shape[:2]:
        raise ValueError("decompressor prediction and latent must be [batch, window, channels]")
    if prediction.shape[1] < 2:
        raise ValueError("decompressor temporal loss requires a two-frame window")
    bones = local_positions.shape[-2]
    non_root = bones - 1
    position_end = 3 * non_root
    rotation_end = 9 * non_root
    velocity_end = 12 * non_root
    angular_end = 15 * non_root
    if prediction.shape[-1] != angular_end + 6 + contacts.shape[-1]:
        raise ValueError("decompressor prediction has the wrong output dimension")

    predicted_position = prediction[..., :position_end].reshape(
        *prediction.shape[:2], non_root, 3
    )
    predicted_xy = prediction[..., position_end:rotation_end].reshape(
        *prediction.shape[:2], non_root, 3, 2
    )
    predicted_velocity = prediction[..., rotation_end:velocity_end].reshape(
        *prediction.shape[:2], non_root, 3
    )
    predicted_angular = prediction[..., velocity_end:angular_end].reshape(
        *prediction.shape[:2], non_root, 3
    )
    predicted_root_velocity = prediction[..., angular_end : angular_end + 3]
    predicted_root_angular = prediction[..., angular_end + 3 : angular_end + 6]
    predicted_contacts = prediction[..., angular_end + 6 :]

    predicted_position = torch.cat(
        (local_positions[..., :1, :], predicted_position), dim=-2
    )
    predicted_xy = torch.cat(
        (local_rotation_xy[..., :1, :, :], predicted_xy), dim=-3
    )
    predicted_velocity = torch.cat(
        (local_velocities[..., :1, :], predicted_velocity), dim=-2
    )
    predicted_angular = torch.cat(
        (local_angular_velocities[..., :1, :], predicted_angular), dim=-2
    )
    predicted_transform = txform.from_xy(predicted_xy)
    global_transform, global_position, global_velocity, global_angular = txform.fk_vel(
        predicted_transform,
        predicted_position,
        predicted_velocity,
        predicted_angular,
        parents,
    )
    predicted_character_transform = txform.inv_mul(
        global_transform[..., 0:1, :, :], global_transform
    )
    predicted_character_position = txform.inv_mul_vec(
        global_transform[..., 0:1, :, :],
        global_position - global_position[..., 0:1, :],
    )
    predicted_character_velocity = txform.inv_mul_vec(
        global_transform[..., 0:1, :, :], global_velocity
    )
    predicted_character_angular = txform.inv_mul_vec(
        global_transform[..., 0:1, :, :], global_angular
    )
    inverse_dt = np.float32(1.0 / dt)

    terms = {
        "local_position": torch.mean(75.0 * torch.abs(local_positions - predicted_position)),
        "local_rotation_xy": torch.mean(
            10.0 * torch.abs(local_rotation_xy - predicted_xy)
        ),
        "local_velocity": torch.mean(
            10.0 * torch.abs(local_velocities - predicted_velocity)
        ),
        "local_angular_velocity": torch.mean(
            1.25 * torch.abs(local_angular_velocities - predicted_angular)
        ),
        "root_velocity": torch.mean(
            2.0 * torch.abs(root_velocity - predicted_root_velocity)
        ),
        "root_angular_velocity": torch.mean(
            2.0 * torch.abs(root_angular_velocity - predicted_root_angular)
        ),
        "contacts": torch.mean(2.0 * torch.abs(contacts - predicted_contacts)),
        "character_position": torch.mean(
            15.0 * torch.abs(character_positions - predicted_character_position)
        ),
        "character_transform": torch.mean(
            5.0 * torch.abs(character_transforms - predicted_character_transform)
        ),
        "character_velocity": torch.mean(
            2.0 * torch.abs(character_velocities - predicted_character_velocity)
        ),
        "character_angular_velocity": torch.mean(
            0.75
            * torch.abs(
                character_angular_velocities - predicted_character_angular
            )
        ),
        "local_position_delta": torch.mean(
            10.0
            * torch.abs(
                torch.diff(local_positions, dim=1) * inverse_dt
                - torch.diff(predicted_position, dim=1) * inverse_dt
            )
        ),
        "local_rotation_xy_delta": torch.mean(
            1.75
            * torch.abs(
                torch.diff(local_rotation_xy, dim=1) * inverse_dt
                - torch.diff(predicted_xy, dim=1) * inverse_dt
            )
        ),
        "character_position_delta": torch.mean(
            2.0
            * torch.abs(
                torch.diff(character_positions, dim=1) * inverse_dt
                - torch.diff(predicted_character_position, dim=1) * inverse_dt
            )
        ),
        "character_transform_delta": torch.mean(
            0.75
            * torch.abs(
                torch.diff(character_transforms, dim=1) * inverse_dt
                - torch.diff(predicted_character_transform, dim=1) * inverse_dt
            )
        ),
        "latent_l1": torch.mean(0.1 * torch.abs(latent)),
        "latent_l2": torch.mean(0.1 * torch.square(latent)),
        "latent_velocity": torch.mean(
            0.01 * torch.abs(torch.diff(latent, dim=1) * inverse_dt)
        ),
    }
    terms["total"] = sum(terms.values())
    return terms


@dataclass(frozen=True)
class _DecompressorTrainingKernel:
    objective: Callable[[torch.Tensor], torch.Tensor]
    compiled: bool
    name: str

    def begin_step(self) -> None:
        if self.compiled:
            torch.compiler.cudagraph_mark_step_begin()

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.objective(batch)


def _build_decompressor_training_kernel(
    *,
    compressor: Compressor,
    decompressor: Decompressor,
    compressor_rows: torch.Tensor,
    feature_rows: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
    ground_rows: dict[str, torch.Tensor],
    parents: tuple[int, ...],
    dt: float,
    device: torch.device,
) -> _DecompressorTrainingKernel:
    """Build the fixed-shape training-only objective after eager validation."""

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported decompressor training device: {device}")
    if (
        type(parents) is not tuple
        or not parents
        or any(type(parent) is not int for parent in parents)
    ):
        raise TypeError("compiled decompressor parents must be a non-empty tuple[int, ...]")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("compiled decompressor dt must be positive and finite")
    if compressor_rows.ndim != 2 or feature_rows.ndim != 2:
        raise ValueError("compiled decompressor row tables must be rank two")
    frames = compressor_rows.shape[0]
    if feature_rows.shape[0] != frames:
        raise ValueError("compiled decompressor row tables must have equal frame counts")
    if output_mean.ndim != 1 or output_std.shape != output_mean.shape:
        raise ValueError("compiled decompressor output normalization is malformed")
    required_ground = {
        "local_positions",
        "local_rotation_xy",
        "local_velocities",
        "local_angular_velocities",
        "character_positions",
        "character_transforms",
        "character_velocities",
        "character_angular_velocities",
        "root_velocity",
        "root_angular_velocity",
        "contacts",
    }
    if set(ground_rows) != required_ground:
        raise ValueError("compiled decompressor ground-row fields are incomplete")
    tensors = {
        "compressor_rows": compressor_rows,
        "feature_rows": feature_rows,
        "output_mean": output_mean,
        "output_std": output_std,
        **ground_rows,
    }
    tensors.update(
        {
            f"compressor.{name}": parameter
            for name, parameter in compressor.named_parameters()
        }
    )
    tensors.update(
        {
            f"decompressor.{name}": parameter
            for name, parameter in decompressor.named_parameters()
        }
    )
    actual_devices = {values.device for values in tensors.values()}
    if len(actual_devices) != 1:
        raise ValueError("compiled decompressor tensors must share one actual device")
    actual_device = next(iter(actual_devices))
    if actual_device.type != device.type:
        raise ValueError("compiled decompressor requested device type does not match tensors")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index != actual_device.index
    ):
        raise ValueError("compiled decompressor requested CUDA index does not match tensors")
    for name, values in ground_rows.items():
        if values.shape[0] != frames:
            raise ValueError(f"compiled decompressor {name} has the wrong frame count")
    if ground_rows["local_positions"].shape[-2] != len(parents):
        raise ValueError("compiled decompressor parents do not match the skeleton")

    def objective(batch: torch.Tensor) -> torch.Tensor:
        encoded = compressor(compressor_rows[batch])
        decoded = (
            decompressor(torch.cat((feature_rows[batch], encoded), dim=-1))
            * output_std
            + output_mean
        )
        return orange_duck_decompressor_losses(
            decoded,
            encoded,
            parents=parents,
            dt=dt,
            **{name: values[batch] for name, values in ground_rows.items()},
        )["total"]

    if device.type == "cuda":
        return _DecompressorTrainingKernel(
            objective=torch.compile(
                objective,
                mode="reduce-overhead",
                fullgraph=True,
            ),
            compiled=True,
            name="torch-compile-reduce-overhead/fullgraph/v1",
        )
    return _DecompressorTrainingKernel(
        objective=objective,
        compiled=False,
        name="eager/v1",
    )


def _decompressor_training_step(
    kernel: _DecompressorTrainingKernel,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one fail-closed optimizer step against the training-only kernel."""

    kernel.begin_step()
    optimizer.zero_grad(set_to_none=True)
    loss = kernel(batch)
    if not torch.isfinite(loss):
        raise FloatingPointError("decompressor training loss became non-finite")
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _legacy_autoencoder_normalization(
    bundle: TrainingBundle,
    arrays: TrainingArrays,
    fitted_rows: np.ndarray,
) -> AutoencoderNormalization:
    fitted = np.asarray(fitted_rows)
    if fitted.dtype != np.bool_ or fitted.shape != (bundle.frames,):
        raise ValueError("autoencoder fitted rows must be an explicit frame mask")
    if not np.any(fitted) or np.any(fitted & ~bundle.admitted_mask):
        raise ValueError("autoencoder fitted rows must be non-empty admitted rows")
    dimensions = bundle.dimensions
    non_root = dimensions.bones - 1
    compressor_rows = arrays.compressor_input[fitted]
    target_rows = arrays.decompressor_target[fitted]
    compressor_mean = compressor_rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    widths = (
        non_root * 3,
        non_root * 6,
        non_root * 3,
        non_root * 3,
        non_root * 3,
        non_root * 6,
        non_root * 3,
        non_root * 3,
        3,
        3,
        dimensions.contacts,
    )
    compressor_std_parts = []
    first = 0
    for width in widths:
        group = compressor_rows[:, first : first + width]
        compressor_std_parts.append(
            np.full(
                width,
                max(float(group.std(dtype=np.float64)), 1.0e-5),
                dtype=np.float32,
            )
        )
        first += width
    compressor_std = np.concatenate(compressor_std_parts)
    decompressor_mean = target_rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    decompressor_std = _safe_std(target_rows, axis=0)
    if compressor_mean.shape != (dimensions.compressor_input,) or compressor_std.shape != (
        dimensions.compressor_input,
    ):
        raise AssertionError("legacy compressor normalization dimensions changed")
    return AutoencoderNormalization(
        compressor_mean=compressor_mean,
        compressor_std=compressor_std,
        decompressor_mean=decompressor_mean,
        decompressor_std=decompressor_std,
    )


def _first_adjacent_block(bundle: TrainingBundle, frames: int) -> tuple[int, int]:
    for start, stop in zip(bundle.range_starts, bundle.range_stops):
        if int(stop) - int(start) >= frames:
            return int(start), int(start) + frames
    raise ValueError(f"no source range contains {frames} adjacent frames")


def _train_64_frame_overfit(
    bundle: TrainingBundle,
    arrays: TrainingArrays,
    config: TrainingConfig,
) -> dict[str, object]:
    start, stop = _first_adjacent_block(bundle, 64)
    device = torch.device(config.device)
    _configure_determinism(config.seed, device)
    fitted = np.zeros(bundle.frames, dtype=bool)
    fitted[start:stop] = True
    fitted &= bundle.admitted_mask
    normalization = _legacy_autoencoder_normalization(bundle, arrays, fitted)
    compressor_input = torch.as_tensor(
        (arrays.compressor_input[start:stop] - normalization.compressor_mean)
        / normalization.compressor_std,
        device=device,
    )
    features = torch.as_tensor(bundle.features[start:stop], device=device)
    target = torch.as_tensor(
        (arrays.decompressor_target[start:stop] - normalization.decompressor_mean)
        / normalization.decompressor_std,
        device=device,
    )
    compressor = Compressor(bundle.dimensions).to(device)
    decompressor = Decompressor(bundle.dimensions).to(device)
    optimizer = torch.optim.AdamW(
        [*compressor.parameters(), *decompressor.parameters()],
        lr=config.learning_rate,
        amsgrad=True,
        weight_decay=1.0e-3,
    )

    def objective() -> torch.Tensor:
        latent = compressor(compressor_input)
        prediction = decompressor(torch.cat((features, latent), dim=-1))
        return torch.mean(torch.square(prediction - target))

    with torch.no_grad():
        initial_loss = float(objective().item())
    for _ in range(config.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError("64-frame overfit loss became non-finite")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(objective().item())
    finite = bool(np.isfinite([initial_loss, final_loss]).all())
    return {
        "accepted": bool(finite and final_loss <= 0.1 * initial_loss),
        "final_loss": final_loss,
        "frames": 64,
        "initial_loss": initial_loss,
        "range": [start, stop],
        "steps": config.overfit_steps,
    }


@dataclass(frozen=True)
class AutoencoderStage:
    compressor: Compressor
    decompressor: Decompressor
    latent: np.ndarray
    normalization: AutoencoderNormalization
    training_metrics: dict[str, object]
    gate: dict[str, object]


@dataclass(frozen=True)
class StepperStage:
    model: Stepper
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    training_metrics: dict[str, object]
    gate: dict[str, object]


@dataclass(frozen=True)
class ProjectorStage:
    model: Projector
    training_metrics: dict[str, object]
    gate: dict[str, object]


def _f1_scores(expected: np.ndarray, predicted: np.ndarray) -> list[float]:
    scores = []
    for column in range(expected.shape[1]):
        truth = expected[:, column].astype(bool)
        estimate = predicted[:, column].astype(bool)
        true_positive = int(np.count_nonzero(truth & estimate))
        false_positive = int(np.count_nonzero(~truth & estimate))
        false_negative = int(np.count_nonzero(truth & ~estimate))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(1.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return scores


def _decode_reconstruction_metrics(
    bundle: TrainingBundle,
    arrays: TrainingArrays,
    prediction: np.ndarray,
    rows: np.ndarray,
    *,
    contact_threshold: float,
) -> dict[str, object]:
    dimensions = bundle.dimensions
    non_root = dimensions.bones - 1
    selected = np.asarray(rows, dtype=np.int64)
    output = np.asarray(prediction[selected], dtype=np.float32)
    position_end = 3 * non_root
    rotation_end = 9 * non_root
    velocity_end = 12 * non_root
    angular_end = 15 * non_root
    predicted_position = output[:, :position_end].reshape(-1, non_root, 3)
    predicted_xy = output[:, position_end:rotation_end].reshape(-1, non_root, 3, 2)
    predicted_velocity = output[:, rotation_end:velocity_end].reshape(-1, non_root, 3)
    predicted_angular = output[:, velocity_end:angular_end].reshape(-1, non_root, 3)
    predicted_root_velocity = output[:, angular_end : angular_end + 3]
    predicted_root_angular = output[:, angular_end + 3 : angular_end + 6]
    predicted_contacts = output[:, angular_end + 6 :]

    finite = bool(np.isfinite(output).all())
    if finite:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            predicted_rotation = quat.from_xform_xy(predicted_xy).astype(np.float32)
        finite = bool(np.isfinite(predicted_rotation).all())
    if not finite:
        return {
            "accepted": False,
            "contact_f1": [0.0] * dimensions.contacts,
            "finite": False,
            "fk_max_body_position_error_m": None,
            "joint_mae_rad": None,
            "joint_max_rad": None,
            "joint_p95_frame_max_rad": None,
            "local_translation_max_error_m": None,
            "root_velocity_finite": False,
            "sole_max_position_error_m": None,
        }

    actual_rotation = bundle.rotations[selected, 1:]
    dot = np.clip(np.abs(np.sum(predicted_rotation * actual_rotation, axis=2)), 0.0, 1.0)
    joint_error = 2.0 * np.arccos(dot)
    frame_max = np.max(joint_error, axis=1)
    local_translation_error = np.linalg.norm(
        predicted_position - bundle.positions[selected, 1:], axis=2
    )
    local_rotation = np.concatenate((bundle.rotations[selected, :1], predicted_rotation), axis=1)
    local_position = np.concatenate((bundle.positions[selected, :1], predicted_position), axis=1)
    _, predicted_global_position = quat.fk(local_rotation, local_position, bundle.parents)
    _, actual_global_position = quat.fk(
        bundle.rotations[selected], bundle.positions[selected], bundle.parents
    )
    body_error = np.linalg.norm(predicted_global_position - actual_global_position, axis=2)
    skeleton_names = bundle.manifest.get("skeleton", {}).get("names", [])
    sole_indices = [
        index
        for index, name in enumerate(skeleton_names)
        if isinstance(name, str)
        and ("ankle_roll" in name.lower() or "foot" in name.lower() or "toe" in name.lower())
    ]
    if not sole_indices:
        sole_indices = [min(6, dimensions.bones - 1), min(12, dimensions.bones - 1)]
    contact_f1 = _f1_scores(
        bundle.contacts[selected], predicted_contacts > np.float32(0.5)
    )
    maximum_contact_requirement = min(contact_f1) >= contact_threshold
    metrics = {
        "contact_f1": contact_f1,
        "finite": True,
        "fk_max_body_position_error_m": float(np.max(body_error)),
        "joint_mae_rad": float(np.mean(joint_error)),
        "joint_max_rad": float(np.max(joint_error)),
        "joint_p95_frame_max_rad": float(np.percentile(frame_max, 95.0)),
        "local_translation_max_error_m": float(np.max(local_translation_error)),
        "root_velocity_finite": bool(
            np.isfinite(predicted_root_velocity).all()
            and np.isfinite(predicted_root_angular).all()
            and np.isfinite(predicted_velocity).all()
            and np.isfinite(predicted_angular).all()
        ),
        "sole_max_position_error_m": float(np.max(body_error[:, sole_indices])),
    }
    metrics["accepted"] = bool(
        metrics["joint_mae_rad"] <= 0.010
        and metrics["joint_p95_frame_max_rad"] <= 0.050
        and metrics["joint_max_rad"] <= 0.100
        and metrics["fk_max_body_position_error_m"] <= 0.010
        and metrics["sole_max_position_error_m"] <= 0.010
        and metrics["local_translation_max_error_m"] <= 0.001
        and metrics["root_velocity_finite"]
        and maximum_contact_requirement
    )
    return metrics


def _train_decompressor_stage(
    staging: Path,
    bundle: TrainingBundle,
    arrays: TrainingArrays,
    config: TrainingConfig,
    withheld_starts: np.ndarray,
    withheld_stops: np.ndarray,
) -> AutoencoderStage:
    device = torch.device(config.device)
    _configure_determinism(config.seed + 1, device)
    windows = range_safe_windows(
        bundle.range_starts,
        bundle.range_stops,
        2,
        excluded_starts=withheld_starts,
        excluded_stops=withheld_stops,
        exclusion_halo=config.withheld_halo,
    )
    if len(windows) == 0:
        raise ValueError("no fitted decompressor windows remain after withheld halo")
    fitted_mask = np.zeros(bundle.frames, dtype=bool)
    fitted_mask[np.unique(windows.reshape(-1))] = True
    fitted_mask &= bundle.admitted_mask
    normalization = _legacy_autoencoder_normalization(
        bundle, arrays, fitted_mask
    )
    compressor_rows = torch.as_tensor(
        (arrays.compressor_input - normalization.compressor_mean) / normalization.compressor_std,
        device=device,
    )
    feature_rows = torch.as_tensor(bundle.features, device=device)
    output_mean = torch.as_tensor(normalization.decompressor_mean, device=device)
    output_std = torch.as_tensor(normalization.decompressor_std, device=device)
    ground_rows = {
        "local_positions": torch.as_tensor(arrays.local_positions, device=device),
        "local_rotation_xy": torch.as_tensor(arrays.local_rotation_xy, device=device),
        "local_velocities": torch.as_tensor(arrays.local_velocities, device=device),
        "local_angular_velocities": torch.as_tensor(
            arrays.local_angular_velocities, device=device
        ),
        "character_positions": torch.as_tensor(
            arrays.character_positions, device=device
        ),
        "character_transforms": torch.as_tensor(
            arrays.character_transforms, device=device
        ),
        "character_velocities": torch.as_tensor(
            arrays.character_velocities, device=device
        ),
        "character_angular_velocities": torch.as_tensor(
            arrays.character_angular_velocities, device=device
        ),
        "root_velocity": torch.as_tensor(arrays.root_velocity, device=device),
        "root_angular_velocity": torch.as_tensor(
            arrays.root_angular_velocity, device=device
        ),
        "contacts": torch.as_tensor(
            bundle.contacts.astype(np.float32), device=device
        ),
    }
    compressor = Compressor(bundle.dimensions).to(device)
    decompressor = Decompressor(bundle.dimensions).to(device)
    optimizer = torch.optim.AdamW(
        [*compressor.parameters(), *decompressor.parameters()],
        lr=config.learning_rate,
        amsgrad=True,
        weight_decay=1.0e-3,
    )
    generator = np.random.default_rng(config.seed + 1)

    def objective_terms(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = compressor(compressor_rows[batch])
        decoded = (
            decompressor(torch.cat((feature_rows[batch], encoded), dim=-1))
            * output_std
            + output_mean
        )
        return orange_duck_decompressor_losses(
            decoded,
            encoded,
            parents=bundle.parents,
            dt=config.dt,
            **{name: values[batch] for name, values in ground_rows.items()},
        )

    audit_windows = torch.as_tensor(windows[: min(len(windows), 512)], device=device)
    with torch.no_grad():
        initial_terms = objective_terms(audit_windows)
        initial_loss = float(initial_terms["total"].item())
    training_kernel = _build_decompressor_training_kernel(
        compressor=compressor,
        decompressor=decompressor,
        compressor_rows=compressor_rows,
        feature_rows=feature_rows,
        output_mean=output_mean,
        output_std=output_std,
        ground_rows=ground_rows,
        parents=tuple(int(parent) for parent in bundle.parents),
        dt=config.dt,
        device=device,
    )
    last_loss = initial_loss
    for _ in range(config.decompressor_steps):
        selected = windows[
            generator.integers(0, len(windows), size=config.batch_size, endpoint=False)
        ]
        batch = torch.as_tensor(selected, device=device)
        last_loss = _decompressor_training_step(training_kernel, batch, optimizer)

    latent_parts = []
    prediction_parts = []
    with torch.no_grad():
        final_terms = objective_terms(audit_windows)
        final_loss = float(final_terms["total"].item())
        for first in range(0, bundle.frames, 512):
            stop = min(bundle.frames, first + 512)
            encoded = compressor(compressor_rows[first:stop])
            decoded = decompressor(torch.cat((feature_rows[first:stop], encoded), dim=-1))
            latent_parts.append(encoded.cpu().numpy().astype(np.float32))
            prediction_parts.append(
                (
                    decoded.cpu().numpy() * normalization.decompressor_std
                    + normalization.decompressor_mean
                ).astype(np.float32)
            )
    latent = np.concatenate(latent_parts)
    prediction = np.concatenate(prediction_parts)
    _atomic_write(staging / "latent.bin", _latent_bytes(latent))
    export_network(
        staging / "decompressor.bin",
        decompressor.layers,
        input_mean=np.zeros(bundle.dimensions.state, dtype=np.float32),
        input_std=np.ones(bundle.dimensions.state, dtype=np.float32),
        output_mean=normalization.decompressor_mean,
        output_std=normalization.decompressor_std,
    )

    withheld_mask = np.zeros(bundle.frames, dtype=bool)
    for start, stop in zip(withheld_starts, withheld_stops):
        withheld_mask[int(start) : int(stop)] = True
    fitted_metrics = _decode_reconstruction_metrics(
        bundle,
        arrays,
        prediction,
        np.flatnonzero(fitted_mask),
        contact_threshold=0.95,
    )
    withheld_metrics = []
    for start, stop in zip(withheld_starts, withheld_stops):
        metrics = _decode_reconstruction_metrics(
            bundle,
            arrays,
            prediction,
            np.arange(int(start), int(stop)),
            contact_threshold=0.90,
        )
        metrics["range"] = [int(start), int(stop)]
        withheld_metrics.append(metrics)
    gate = {
        "accepted": bool(
            fitted_metrics["accepted"]
            and withheld_metrics
            and all(metrics["accepted"] for metrics in withheld_metrics)
        ),
        "fitted": fitted_metrics,
        "withheld": withheld_metrics,
    }
    return AutoencoderStage(
        compressor=compressor,
        decompressor=decompressor,
        latent=latent,
        normalization=normalization,
        training_metrics={
            "final_audit_loss": final_loss,
            "final_audit_loss_terms": {
                name: float(value.item()) for name, value in final_terms.items()
            },
            "final_batch_loss": last_loss,
            "initial_audit_loss": initial_loss,
            "initial_audit_loss_terms": {
                name: float(value.item()) for name, value in initial_terms.items()
            },
            "objective": "orange-duck-denormalized-weighted/v1",
            "steps": config.decompressor_steps,
            "training_kernel": training_kernel.name,
            "windows": len(windows),
        },
        gate=gate,
    )


def _decode_state_rows(
    autoencoder: AutoencoderStage,
    state: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for first in range(0, len(state), 512):
            rows = torch.as_tensor(state[first : first + 512], device=device)
            decoded = autoencoder.decompressor(rows).cpu().numpy()
            outputs.append(
                decoded * autoencoder.normalization.decompressor_std
                + autoencoder.normalization.decompressor_mean
            )
    return np.concatenate(outputs).astype(np.float32)


def _rollout_root(
    initial_position: np.ndarray,
    initial_rotation: np.ndarray,
    decoded: np.ndarray,
    dimensions: G1LmmDimensions,
    dt: float,
) -> np.ndarray:
    non_root = dimensions.bones - 1
    root_offset = 15 * non_root
    position = np.asarray(initial_position, dtype=np.float32).copy()
    rotation = np.asarray(initial_rotation, dtype=np.float32).copy()
    positions = [position.copy()]
    for row in decoded[:-1]:
        local_velocity = row[root_offset : root_offset + 3]
        local_angular = row[root_offset + 3 : root_offset + 6]
        position = position + quat.mul_vec(rotation, local_velocity) * np.float32(dt)
        world_angular = quat.mul_vec(rotation, local_angular) * np.float32(dt)
        rotation = quat.mul(rotation, quat.from_scaled_angle_axis(world_angular))
        rotation = quat.normalize(rotation)
        positions.append(position.copy())
    return np.asarray(positions, dtype=np.float32)


def _rollout_pose_metrics(
    bundle: TrainingBundle,
    decoded: np.ndarray,
    start: int,
    horizon: int,
) -> dict[str, object]:
    dimensions = bundle.dimensions
    non_root = dimensions.bones - 1
    position_end = 3 * non_root
    rotation_end = 9 * non_root
    rows = decoded[: horizon + 1]
    predicted_xy = rows[:, position_end:rotation_end].reshape(-1, non_root, 3, 2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        predicted_rotation = quat.from_xform_xy(predicted_xy).astype(np.float32)
    finite = bool(np.isfinite(rows).all() and np.isfinite(predicted_rotation).all())
    if not finite:
        return {"contact_f1": [0.0] * dimensions.contacts, "finite": False, "joint_mae_rad": None}
    actual_rotation = bundle.rotations[start : start + horizon + 1, 1:]
    dot = np.clip(np.abs(np.sum(predicted_rotation * actual_rotation, axis=2)), 0.0, 1.0)
    joint_error = 2.0 * np.arccos(dot)
    contact_offset = 15 * non_root + 6
    predicted_contacts = rows[:, contact_offset:] > np.float32(0.5)
    return {
        "contact_f1": _f1_scores(
            bundle.contacts[start : start + horizon + 1], predicted_contacts
        ),
        "finite": True,
        "joint_mae_rad": float(np.mean(joint_error)),
    }


def _stepper_rollout_gate(
    bundle: TrainingBundle,
    arrays: TrainingArrays,
    autoencoder: AutoencoderStage,
    model: Stepper,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
    config: TrainingConfig,
) -> dict[str, object]:
    device = torch.device(config.device)
    state = np.concatenate((bundle.features, autoencoder.latent), axis=1).astype(np.float32)
    horizon_frames = (60, 120, 240)
    rollouts = []
    skipped_ranges = []
    for range_index, (range_start, range_stop) in enumerate(
        zip(bundle.range_starts, bundle.range_stops)
    ):
        start = int(range_start)
        stop = int(range_stop)
        if stop - start <= horizon_frames[-1]:
            skipped_ranges.append(
                {
                    "range_index": range_index,
                    "reason": "range has 240 or fewer valid successors",
                }
            )
            continue
        predicted = [state[start].copy()]
        current = torch.as_tensor(state[start : start + 1], device=device)
        with torch.no_grad():
            for _ in range(horizon_frames[-1]):
                derivative = (
                    model(
                        (current - torch.as_tensor(input_mean, device=device))
                        / torch.as_tensor(input_std, device=device)
                    )
                    * torch.as_tensor(output_std, device=device)
                    + torch.as_tensor(output_mean, device=device)
                )
                current = current + np.float32(config.dt) * derivative
                predicted.append(current[0].cpu().numpy().astype(np.float32))
        predicted_state = np.asarray(predicted, dtype=np.float32)
        finite = bool(np.isfinite(predicted_state).all())
        decoded = _decode_state_rows(autoencoder, predicted_state, device) if finite else None
        finite = bool(finite and decoded is not None and np.isfinite(decoded).all())
        if not finite:
            rollouts.append(
                {
                    "accepted": False,
                    "finite": False,
                    "range_index": range_index,
                    "seed": start,
                }
            )
            continue
        root_positions = _rollout_root(
            bundle.positions[start, 0],
            bundle.rotations[start, 0],
            decoded,
            bundle.dimensions,
            config.dt,
        )
        horizon_receipts = []
        for horizon in horizon_frames:
            row_metrics = _rollout_pose_metrics(bundle, decoded, start, horizon)
            state_rmse = float(
                np.sqrt(np.mean(np.square(predicted_state[horizon] - state[start + horizon])))
            )
            planar_root_error = float(
                np.linalg.norm(
                    root_positions[horizon, (0, 2)]
                    - bundle.positions[start + horizon, 0, (0, 2)]
                )
            )
            horizon_receipts.append(
                {
                    "contact_f1": row_metrics["contact_f1"],
                    "finite": row_metrics["finite"],
                    "frames": horizon,
                    "joint_mae_rad": row_metrics["joint_mae_rad"],
                    "planar_root_error_m": planar_root_error,
                    "seconds": horizon / 60.0,
                    "state_rmse": state_rmse,
                }
            )
        final = horizon_receipts[-1]
        accepted = bool(
            all(receipt["finite"] for receipt in horizon_receipts)
            and final["joint_mae_rad"] <= 0.050
            and final["planar_root_error_m"] <= 0.10
            and min(final["contact_f1"]) >= 0.90
        )
        rollouts.append(
            {
                "accepted": accepted,
                "horizons": horizon_receipts,
                "range_index": range_index,
                "seed": start,
            }
        )
    return {
        "accepted": bool(rollouts and all(receipt["accepted"] for receipt in rollouts)),
        "rollouts": rollouts,
        "skipped_ranges": skipped_ranges,
    }


def _train_stepper_stage(
    staging: Path,
    bundle: TrainingBundle,
    arrays: TrainingArrays,
    autoencoder: AutoencoderStage,
    config: TrainingConfig,
    withheld_starts: np.ndarray,
    withheld_stops: np.ndarray,
) -> StepperStage:
    device = torch.device(config.device)
    _configure_determinism(config.seed + 2, device)
    dimensions = bundle.dimensions
    state = np.concatenate((bundle.features, autoencoder.latent), axis=1).astype(np.float32)
    pair_windows = range_safe_windows(
        bundle.range_starts,
        bundle.range_stops,
        2,
        excluded_starts=withheld_starts,
        excluded_stops=withheld_stops,
        exclusion_halo=config.withheld_halo,
    )
    windows = range_safe_windows(
        bundle.range_starts,
        bundle.range_stops,
        config.stepper_window,
        excluded_starts=withheld_starts,
        excluded_stops=withheld_stops,
        exclusion_halo=config.withheld_halo,
    )
    if len(pair_windows) == 0 or len(windows) == 0:
        raise ValueError("no fitted stepper windows remain after withheld halo")
    fitted_rows = np.unique(windows.reshape(-1))
    if not np.all(bundle.admitted_mask[fitted_rows]):
        raise AssertionError("stepper windows include a non-admitted row")
    features_fitted = bundle.features[fitted_rows]
    latent_fitted = autoencoder.latent[fitted_rows]
    input_mean = np.concatenate(
        (
            features_fitted.mean(axis=0, dtype=np.float64),
            latent_fitted.mean(axis=0, dtype=np.float64),
        )
    ).astype(np.float32)
    input_std = np.concatenate(
        (
            np.full(dimensions.features, max(float(features_fitted.std()), 1.0e-5)),
            np.full(dimensions.latent, max(float(latent_fitted.std()), 1.0e-5)),
        )
    ).astype(np.float32)
    derivatives = (
        state[pair_windows[:, 1]] - state[pair_windows[:, 0]]
    ) * np.float32(1.0 / config.dt)
    output_mean = derivatives.mean(axis=0, dtype=np.float64).astype(np.float32)
    output_std = _safe_std(derivatives, axis=0)

    state_rows = torch.as_tensor(state, device=device)
    input_mean_tensor = torch.as_tensor(input_mean, device=device)
    input_std_tensor = torch.as_tensor(input_std, device=device)
    output_mean_tensor = torch.as_tensor(output_mean, device=device)
    output_std_tensor = torch.as_tensor(output_std, device=device)
    model = Stepper(dimensions).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        amsgrad=True,
        weight_decay=1.0e-3,
    )
    generator = np.random.default_rng(config.seed + 2)

    def objective(batch: torch.Tensor) -> torch.Tensor:
        ground = state_rows[batch]
        predicted = [ground[:, 0]]
        for _ in range(1, config.stepper_window):
            derivative = (
                model((predicted[-1] - input_mean_tensor) / input_std_tensor)
                * output_std_tensor
                + output_mean_tensor
            )
            predicted.append(predicted[-1] + np.float32(config.dt) * derivative)
        estimate = torch.stack(predicted, dim=1)
        ground_velocity = (ground[:, 1:] - ground[:, :-1]) / np.float32(config.dt)
        estimate_velocity = (estimate[:, 1:] - estimate[:, :-1]) / np.float32(config.dt)
        return (
            torch.mean(2.0 * torch.abs(ground[..., : dimensions.features] - estimate[..., : dimensions.features]))
            + torch.mean(7.5 * torch.abs(ground[..., dimensions.features :] - estimate[..., dimensions.features :]))
            + torch.mean(
                0.2
                * torch.abs(
                    ground_velocity[..., : dimensions.features]
                    - estimate_velocity[..., : dimensions.features]
                )
            )
            + torch.mean(
                0.5
                * torch.abs(
                    ground_velocity[..., dimensions.features :]
                    - estimate_velocity[..., dimensions.features :]
                )
            )
        )

    audit = torch.as_tensor(windows[: min(64, len(windows))], device=device)
    with torch.no_grad():
        initial_loss = float(objective(audit).item())
    last_loss = initial_loss
    for _ in range(config.stepper_steps):
        batch_rows = windows[
            generator.integers(0, len(windows), size=config.batch_size, endpoint=False)
        ]
        batch = torch.as_tensor(batch_rows, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("stepper training loss became non-finite")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    with torch.no_grad():
        final_loss = float(objective(audit).item())
    export_network(
        staging / "stepper.bin",
        model.layers,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )
    gate = _stepper_rollout_gate(
        bundle,
        arrays,
        autoencoder,
        model,
        input_mean,
        input_std,
        output_mean,
        output_std,
        config,
    )
    return StepperStage(
        model=model,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        training_metrics={
            "derivative_pairs": len(pair_windows),
            "final_audit_loss": final_loss,
            "final_batch_loss": last_loss,
            "initial_audit_loss": initial_loss,
            "steps": config.stepper_steps,
            "windows": len(windows),
        },
        gate=gate,
    )


def _exact_projector_pairs(
    queries: np.ndarray,
    features: np.ndarray,
    latent: np.ndarray,
) -> ProjectorTargets:
    indices = []
    distances = []
    for first in range(0, len(queries), 128):
        result = normalized_projector_targets(
            queries[first : first + 128], features, latent
        )
        indices.append(result.indices)
        distances.append(result.distance)
    nearest = np.concatenate(indices)
    return ProjectorTargets(
        indices=nearest,
        features=features[nearest].copy(),
        latent=latent[nearest].copy(),
        distance=np.concatenate(distances),
    )


def _train_projector_stage(
    staging: Path,
    bundle: TrainingBundle,
    autoencoder: AutoencoderStage,
    config: TrainingConfig,
    withheld_starts: np.ndarray,
    withheld_stops: np.ndarray,
) -> ProjectorStage:
    device = torch.device(config.device)
    _configure_determinism(config.seed + 3, device)
    fitted = bundle.admitted_mask.copy()
    for start, stop in zip(withheld_starts, withheld_stops):
        fitted[
            max(0, int(start) - config.withheld_halo) : min(
                bundle.frames, int(stop) + config.withheld_halo
            )
        ] = False
    features = bundle.features[fitted]
    latent = autoencoder.latent[fitted]
    if len(features) == 0:
        raise ValueError("projector has no fitted oracle rows")
    state = np.concatenate((features, latent), axis=1).astype(np.float32)
    input_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.full(
        bundle.dimensions.features, max(float(features.std()), 1.0e-5), np.float32
    )
    output_mean = state.mean(axis=0, dtype=np.float64).astype(np.float32)
    output_std = _safe_std(state, axis=0)
    noise_std = features.std(axis=0, dtype=np.float64).astype(np.float32) + np.float32(1.0)
    generator = np.random.default_rng(config.seed + 3)
    sample_count = 4096
    samples = generator.integers(0, len(features), size=sample_count)
    sigma = generator.uniform(size=(sample_count, 1)).astype(np.float32)
    noise = generator.normal(size=(sample_count, bundle.dimensions.features)).astype(np.float32)
    queries = features[samples] + noise_std * sigma * noise
    targets = _exact_projector_pairs(queries, features, latent)

    query_tensor = torch.as_tensor(queries, device=device)
    target_feature = torch.as_tensor(targets.features, device=device)
    target_latent = torch.as_tensor(targets.latent, device=device)
    target_distance = torch.as_tensor(targets.distance, device=device)
    input_mean_tensor = torch.as_tensor(input_mean, device=device)
    input_std_tensor = torch.as_tensor(input_std, device=device)
    output_mean_tensor = torch.as_tensor(output_mean, device=device)
    output_std_tensor = torch.as_tensor(output_std, device=device)
    model = Projector(bundle.dimensions).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        amsgrad=True,
        weight_decay=1.0e-3,
    )

    def objective(indices: torch.Tensor) -> torch.Tensor:
        query = query_tensor[indices]
        output = (
            model((query - input_mean_tensor) / input_std_tensor) * output_std_tensor
            + output_mean_tensor
        )
        projected_feature = output[:, : bundle.dimensions.features]
        projected_latent = output[:, bundle.dimensions.features :]
        projected_distance = torch.sqrt(
            torch.sum(torch.square(query - projected_feature), dim=1)
        )
        return (
            torch.mean(torch.abs(target_feature[indices] - projected_feature))
            + 5.0 * torch.mean(torch.abs(target_latent[indices] - projected_latent))
            + 0.3 * torch.mean(torch.abs(target_distance[indices] - projected_distance))
        )

    audit = torch.arange(min(512, sample_count), device=device)
    with torch.no_grad():
        initial_loss = float(objective(audit).item())
    last_loss = initial_loss
    for _ in range(config.projector_steps):
        selected = generator.integers(0, sample_count, size=config.batch_size)
        indices = torch.as_tensor(selected, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(indices)
        if not torch.isfinite(loss):
            raise FloatingPointError("projector training loss became non-finite")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    with torch.no_grad():
        final_loss = float(objective(audit).item())

    gate_generator = np.random.default_rng(config.seed + 4)
    gate_count = 512
    gate_samples = gate_generator.integers(0, len(features), size=gate_count)
    gate_sigma = gate_generator.uniform(size=(gate_count, 1)).astype(np.float32)
    gate_noise = gate_generator.normal(
        size=(gate_count, bundle.dimensions.features)
    ).astype(np.float32)
    gate_queries = features[gate_samples] + noise_std * gate_sigma * gate_noise
    gate_targets = _exact_projector_pairs(gate_queries, features, latent)
    with torch.no_grad():
        query = torch.as_tensor(gate_queries, device=device)
        output = (
            model((query - input_mean_tensor) / input_std_tensor) * output_std_tensor
            + output_mean_tensor
        ).cpu().numpy()
    feature_error = output[:, : bundle.dimensions.features] - gate_targets.features
    latent_error = output[:, bundle.dimensions.features :] - gate_targets.latent
    finite = bool(np.isfinite(output).all())
    feature_rmse = float(np.sqrt(np.mean(np.square(feature_error))))
    latent_rmse = float(np.sqrt(np.mean(np.square(latent_error))))
    maximum_error = float(
        max(np.max(np.abs(feature_error)), np.max(np.abs(latent_error)))
    )
    gate = {
        "accepted": bool(
            finite
            and feature_rmse <= 0.05
            and latent_rmse <= 0.10
            and maximum_error <= 0.50
        ),
        "finite": finite,
        "maximum_normalized_component_error": maximum_error,
        "normalized_feature_rmse": feature_rmse,
        "normalized_latent_rmse": latent_rmse,
        "queries": gate_count,
    }
    export_network(
        staging / "projector.bin",
        model.layers,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )
    return ProjectorStage(
        model=model,
        training_metrics={
            "final_audit_loss": final_loss,
            "final_batch_loss": last_loss,
            "initial_audit_loss": initial_loss,
            "oracle_rows": len(features),
            "samples": sample_count,
            "steps": config.projector_steps,
        },
        gate=gate,
    )


def _config_receipt(config: TrainingConfig) -> dict[str, object]:
    return {
        "batch_size": config.batch_size,
        "decompressor_steps": config.decompressor_steps,
        "device": config.device,
        "dt": config.dt,
        "learning_rate": config.learning_rate,
        "overfit_steps": config.overfit_steps,
        "projector_steps": config.projector_steps,
        "seed": config.seed,
        "stepper_steps": config.stepper_steps,
        "stepper_window": config.stepper_window,
        "withheld_frames": config.withheld_frames,
        "withheld_halo": config.withheld_halo,
    }


def _publish_directory(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable output directory: {output}")
    os.replace(staging, output)


def _sanitize_for_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return _sanitize_for_json(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _publish_rejected_receipts(
    staging: Path,
    output: Path,
    receipt: dict[str, object],
    evaluation: dict[str, object],
    *,
    stopped_after: str,
    diagnostics: dict[str, object] | None = None,
) -> None:
    for name in (
        "latent.bin",
        "decompressor.bin",
        "stepper.bin",
        "projector.bin",
        "manifest.json",
    ):
        path = staging / name
        if path.exists():
            path.unlink()
    receipt.update(
        {
            "accepted": False,
            "artifacts": {},
            "status": "rejected",
            "stopped_after": stopped_after,
        }
    )
    evaluation.update(
        {
            "accepted": False,
            "artifacts": {},
            "status": "rejected",
            "stopped_after": stopped_after,
        }
    )
    if diagnostics is not None:
        receipt["diagnostics"] = diagnostics
        evaluation["diagnostics"] = diagnostics
    _atomic_write(staging / "training.json", _json_bytes(receipt))
    _atomic_write(staging / "evaluation.json", _json_bytes(evaluation))
    _publish_directory(staging, output)


def train_flat_bundle(
    data_directory: str | Path,
    output_directory: str | Path,
    *,
    config: TrainingConfig,
    stage: str = "all",
) -> dict[str, object]:
    """Run a gated training stage against an accepted immutable Task 1 bundle."""

    if stage not in {"all", "overfit", "decompressor", "stepper", "projector"}:
        raise ValueError(f"unsupported training stage: {stage}")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = load_training_bundle(data_directory)
    arrays = build_training_arrays(bundle)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    receipt = {
        "accepted": False,
        "config": _config_receipt(config),
        "data_manifest_sha256": bundle.manifest_sha256,
        "schema": "g1-lmm-training-v1",
        "stage": stage,
    }
    evaluation = {
        "accepted": False,
        "schema": "g1-lmm-evaluation-v1",
    }
    current_stage = "overfit"
    try:
        overfit_gate = _train_64_frame_overfit(bundle, arrays, config)
        receipt["overfit_gate"] = overfit_gate
        evaluation["overfit_gate"] = overfit_gate
        if not overfit_gate["accepted"]:
            _publish_rejected_receipts(
                staging, output, receipt, evaluation, stopped_after="overfit"
            )
            raise TrainingGateError("overfit", output)
        if stage == "overfit":
            _publish_rejected_receipts(
                staging,
                output,
                receipt,
                evaluation,
                stopped_after="overfit",
                diagnostics={"reason": "requested partial stage completed"},
            )
            return receipt

        current_stage = "decompressor"
        range_lengths = (
            bundle.range_stops.astype(np.int64) - bundle.range_starts.astype(np.int64)
        )
        eligible = np.flatnonzero(
            range_lengths >= config.withheld_frames + 2 * config.withheld_halo
        )
        if len(eligible) == 0:
            raise ValueError("no continuity-safe range can hold the withheld block and halo")
        longest = int(eligible[np.argmax(range_lengths[eligible])])
        withheld_starts, withheld_stops = deterministic_withheld_ranges(
            bundle.range_starts[longest : longest + 1],
            bundle.range_stops[longest : longest + 1],
            frames=config.withheld_frames,
            seed=config.seed,
            margin=config.withheld_halo,
        )
        receipt["withheld"] = {
            "halo": config.withheld_halo,
            "ranges": [
                [int(start), int(stop)]
                for start, stop in zip(withheld_starts, withheld_stops)
            ],
        }
        autoencoder = _train_decompressor_stage(
            staging,
            bundle,
            arrays,
            config,
            withheld_starts,
            withheld_stops,
        )
        receipt["decompressor_training"] = autoencoder.training_metrics
        receipt["decompressor_gate"] = autoencoder.gate
        evaluation["decompressor_gate"] = autoencoder.gate
        if not autoencoder.gate["accepted"]:
            _publish_rejected_receipts(
                staging, output, receipt, evaluation, stopped_after="decompressor"
            )
            raise TrainingGateError("decompressor", output)
        if stage == "decompressor":
            _publish_rejected_receipts(
                staging,
                output,
                receipt,
                evaluation,
                stopped_after="decompressor",
                diagnostics={"reason": "requested partial stage completed"},
            )
            return receipt

        current_stage = "stepper"
        stepper = _train_stepper_stage(
            staging,
            bundle,
            arrays,
            autoencoder,
            config,
            withheld_starts,
            withheld_stops,
        )
        receipt["stepper_training"] = stepper.training_metrics
        receipt["stepper_gate"] = stepper.gate
        evaluation["stepper_gate"] = stepper.gate
        if not stepper.gate["accepted"]:
            _publish_rejected_receipts(
                staging, output, receipt, evaluation, stopped_after="stepper"
            )
            raise TrainingGateError("stepper", output)
        if stage == "stepper":
            _publish_rejected_receipts(
                staging,
                output,
                receipt,
                evaluation,
                stopped_after="stepper",
                diagnostics={"reason": "requested partial stage completed"},
            )
            return receipt

        current_stage = "projector"
        projector = _train_projector_stage(
            staging,
            bundle,
            autoencoder,
            config,
            withheld_starts,
            withheld_stops,
        )
        receipt["projector_training"] = projector.training_metrics
        receipt["projector_gate"] = projector.gate
        evaluation["projector_gate"] = projector.gate
        if not projector.gate["accepted"]:
            _publish_rejected_receipts(
                staging, output, receipt, evaluation, stopped_after="projector"
            )
            raise TrainingGateError("projector", output)

        current_stage = "reload"
        reload_metrics = evaluate_exported_networks(staging)
        receipt.update(
            {
                "accepted": True,
                "reload_metrics": reload_metrics,
                "status": "accepted",
                "stopped_after": "projector",
            }
        )
        evaluation.update(
            {
                "accepted": True,
                "reload_metrics": reload_metrics,
                "status": "accepted",
                "stopped_after": "projector",
            }
        )
        _atomic_write(staging / "training.json", _json_bytes(receipt))
        _atomic_write(staging / "evaluation.json", _json_bytes(evaluation))
        artifact_names = (
            "latent.bin",
            "decompressor.bin",
            "stepper.bin",
            "projector.bin",
            "training.json",
            "evaluation.json",
        )
        artifacts = {
            name: {
                "path": name,
                "size_bytes": (staging / name).stat().st_size,
                "sha256": _sha256(staging / name),
            }
            for name in artifact_names
        }
        manifest = {
            "artifacts": artifacts,
            "data_artifacts": {
                name: descriptor["sha256"]
                for name, descriptor in bundle.manifest["artifacts"].items()
            },
            "data_manifest_schema": bundle.manifest["schema"],
            "data_manifest_sha256": bundle.manifest_sha256,
            "dimensions": {
                "bones": bundle.dimensions.bones,
                "contacts": bundle.dimensions.contacts,
                "features": bundle.dimensions.features,
                "latent": bundle.dimensions.latent,
            },
            "output_fps": 60.0,
            "schema": "g1-lmm-model/v1",
            "status": "accepted",
        }
        _atomic_write(staging / "manifest.json", _json_bytes(manifest))
        _publish_directory(staging, output)
        return receipt
    except TrainingGateError:
        raise
    except Exception as error:
        if staging.exists() and not output.exists():
            _publish_rejected_receipts(
                staging,
                output,
                receipt,
                evaluation,
                stopped_after=current_stage,
                diagnostics={
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        raise TrainingGateError(current_stage, output) from error


def _synthetic_fixture(seed: int, dimensions: G1LmmDimensions) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)
    frames = 64
    phase = np.arange(frames, dtype=np.float32)[:, None] / np.float32(frames)
    frequencies = np.arange(1, dimensions.latent + 1, dtype=np.float32)[None, :]
    code = np.sin(np.float32(2.0 * np.pi) * phase * frequencies).astype(np.float32)
    features = np.zeros((frames, dimensions.features), dtype=np.float32)
    features[:, : dimensions.features] = np.cos(
        np.float32(2.0 * np.pi) * phase * np.arange(1, dimensions.features + 1, dtype=np.float32)[None]
    )
    compressor_input = generator.normal(0.0, 0.01, (frames, dimensions.compressor_input)).astype(np.float32)
    compressor_input[:, : dimensions.latent] = code
    target = np.zeros((frames, dimensions.decompressor_output), dtype=np.float32)
    repeats = (dimensions.decompressor_output + dimensions.latent - 1) // dimensions.latent
    target[:] = np.tile(code, (1, repeats))[:, : dimensions.decompressor_output] * np.float32(0.1)
    return features, compressor_input, target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    sanitized = _sanitize_for_json(value)
    return (json.dumps(sanitized, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _latent_bytes(latent: np.ndarray) -> bytes:
    latent = np.ascontiguousarray(latent, dtype="<f4")
    return struct.pack("<II", *latent.shape) + latent.tobytes(order="C")


def _evaluation_inputs(width: int) -> np.ndarray:
    return np.linspace(-0.75, 0.75, num=3 * width, dtype=np.float32).reshape(3, width)


def evaluate_exported_networks(output_directory: str | Path) -> dict[str, object]:
    root = Path(output_directory)
    dimensions = G1LmmDimensions()
    specs = {
        "decompressor.bin": [(dimensions.state, 512), (512, dimensions.decompressor_output)],
        "stepper.bin": [(dimensions.state, 512), (512, 512), (512, dimensions.state)],
        "projector.bin": [
            (dimensions.features, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, dimensions.state),
        ],
    }
    metrics: dict[str, object] = {}
    for name, layers in specs.items():
        network = load_exported_network(root / name, expected_layers=layers)
        output = network.evaluate(_evaluation_inputs(layers[0][0]))
        if not np.isfinite(output).all():
            raise ValueError(f"{name} reload produced non-finite output")
        metrics[name] = {
            "maximum_absolute_output": float(np.max(np.abs(output))),
            "output_sha256": hashlib.sha256(np.ascontiguousarray(output, dtype="<f4").tobytes()).hexdigest(),
        }
    metrics["latent.bin"] = {"sha256": _sha256(root / "latent.bin")}
    return metrics


def train_tiny_fixture(
    output_directory: str | Path,
    *,
    seed: int = 1234,
    device: str = "cpu",
) -> dict[str, object]:
    """Train and export a deterministic 64-frame CPU or CUDA contract fixture."""

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=False)
    torch_device = torch.device(device)
    _configure_determinism(seed, torch_device)
    dimensions = G1LmmDimensions()
    features, compressor_rows, target = _synthetic_fixture(seed, dimensions)

    compressor_mean = compressor_rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    compressor_std = _safe_std(compressor_rows, axis=0)
    target_mean = target.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_std = _safe_std(target, axis=0)
    compressor_input = torch.as_tensor((compressor_rows - compressor_mean) / compressor_std, device=torch_device)
    target_normalized = torch.as_tensor((target - target_mean) / target_std, device=torch_device)
    feature_tensor = torch.as_tensor(features, device=torch_device)

    compressor = Compressor(dimensions).to(torch_device)
    decompressor = Decompressor(dimensions).to(torch_device)
    optimizer = torch.optim.AdamW(
        [*compressor.parameters(), *decompressor.parameters()],
        lr=2.0e-3,
        amsgrad=True,
        weight_decay=1.0e-3,
    )

    def loss_value() -> torch.Tensor:
        latent_value = compressor(compressor_input)
        prediction = decompressor(torch.cat([feature_tensor, latent_value], dim=-1))
        return torch.mean(torch.square(prediction - target_normalized))

    with torch.no_grad():
        initial_loss = float(loss_value().item())
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        if not torch.isfinite(loss):
            raise FloatingPointError("64-frame overfit loss became non-finite")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(loss_value().item())
        latent = compressor(compressor_input).cpu().numpy().astype(np.float32)

    stepper = Stepper(dimensions).to(torch_device)
    projector = Projector(dimensions).to(torch_device)
    state = np.concatenate([features, latent], axis=1).astype(np.float32)
    state_std = np.full(dimensions.state, max(float(state.std()), 1.0e-5), dtype=np.float32)
    state_mean = state.mean(axis=0, dtype=np.float64).astype(np.float32)
    derivative_mean = np.zeros(dimensions.state, dtype=np.float32)
    derivative_std = np.ones(dimensions.state, dtype=np.float32)
    feature_std = np.full(dimensions.features, max(float(features.std()), 1.0e-5), dtype=np.float32)
    projector_output_std = _safe_std(state, axis=0)

    _atomic_write(root / "latent.bin", _latent_bytes(latent))
    export_network(
        root / "decompressor.bin",
        decompressor.layers,
        input_mean=np.zeros(dimensions.state, dtype=np.float32),
        input_std=np.ones(dimensions.state, dtype=np.float32),
        output_mean=target_mean,
        output_std=target_std,
    )
    export_network(
        root / "stepper.bin",
        stepper.layers,
        input_mean=state_mean,
        input_std=state_std,
        output_mean=derivative_mean,
        output_std=derivative_std,
    )
    export_network(
        root / "projector.bin",
        projector.layers,
        input_mean=features.mean(axis=0, dtype=np.float64).astype(np.float32),
        input_std=feature_std,
        output_mean=state_mean,
        output_std=projector_output_std,
    )

    finite = bool(np.isfinite(latent).all() and np.isfinite([initial_loss, final_loss]).all())
    gate = {
        "accepted": bool(finite and final_loss <= 0.1 * initial_loss),
        "final_loss": final_loss,
        "frames": 64,
        "initial_loss": initial_loss,
        "steps": 24,
    }
    training_receipt = {
        "determinism": {
            "cublas_workspace_config": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if torch_device.type == "cuda"
                else None
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "device": str(torch_device),
        "dimensions": {
            "bones": dimensions.bones,
            "contacts": dimensions.contacts,
            "features": dimensions.features,
            "latent": dimensions.latent,
        },
        "finite": finite,
        "overfit_gate": gate,
        "schema": "g1-lmm-training-v1",
        "seed": seed,
        "synthetic": True,
    }
    _atomic_write(root / "training.json", _json_bytes(training_receipt))
    reload_metrics = evaluate_exported_networks(root)
    _atomic_write(root / "evaluation.json", _json_bytes({"reload_metrics": reload_metrics}))

    artifact_names = ("latent.bin", "decompressor.bin", "stepper.bin", "projector.bin")
    artifacts = {
        name: {"bytes": (root / name).stat().st_size, "sha256": _sha256(root / name)}
        for name in artifact_names
    }
    manifest = {
        "artifacts": artifacts,
        "data_manifest_sha256": "synthetic",
        "dimensions": training_receipt["dimensions"],
        "schema": "g1-lmm-model-v1",
        "training_accepted": gate["accepted"],
    }
    _atomic_write(root / "manifest.json", _json_bytes(manifest))
    return {
        **training_receipt,
        "artifact_sha256": {name: artifacts[name]["sha256"] for name in artifact_names},
        "reload_metrics": reload_metrics,
    }
