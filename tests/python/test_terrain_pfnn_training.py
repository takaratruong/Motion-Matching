from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

import mm_sonic.train_terrain_pfnn as train_module
import mm_sonic.terrain_pfnn.training as training_module
from mm_sonic.evaluate_terrain_pfnn import (
    claim_sealed_test_receipt,
    validate_checkpoint_kinematics,
)
from mm_sonic.train_terrain_pfnn import (
    consecutive_overfit_subset,
    fitted_subset_metadata,
    materialize_subset,
    overfit_gate_accepted,
    build_pipeline_promotion_receipt,
    promote_pipeline_best,
    provisional_pipeline_selection,
    stratified_subset,
)
from mm_sonic.terrain_pfnn.dataset import normalize_pfnn_input, pfnn_input_sha256
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT, TRAJECTORY_TIMES_S
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.terrain_pfnn.recurrence import (
    PlannedTrajectory,
    advance_recurrent_state,
    derive_training_desired_velocity,
    initialize_recurrent_state,
    pack_recurrent_input,
    plan_recurrent_trajectory,
)
from mm_sonic.terrain_pfnn.training import (
    CHECKPOINT_SCHEMA,
    DEFAULT_LOSS_WEIGHTS,
    LOSS_WEIGHT_KEYS,
    autoregressive_unroll,
    choose_runtime_seed,
    fitted_row_sha256,
    finite_runtime_seed,
    load_checkpoint,
    pfnn_losses,
    restore_training_state,
    save_checkpoint,
    selection_metadata,
    training_phase_advance_q99,
)
from mm_sonic.terrain_oracle.canonical import ISAACLAB_JOINT_NAMES


def _normalization() -> dict[str, np.ndarray]:
    return {
        "x_mean": np.zeros(INPUT_LAYOUT.size, np.float32),
        "x_std": np.ones(INPUT_LAYOUT.size, np.float32),
        "y_mean": np.zeros(OUTPUT_LAYOUT.size, np.float32),
        "y_std": np.ones(OUTPUT_LAYOUT.size, np.float32),
    }


class _FittedEnvelopeRows:
    split = "train"
    x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
    x_std = np.ones(INPUT_LAYOUT.size, np.float32)
    y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
    y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

    def __init__(self, prediction: np.ndarray) -> None:
        predecessor = self._physical_output()
        current = self._physical_output()
        self.rows = [
            self._row(10, predecessor, phase=0.25),
            self._row(11, current, phase=0.75),
        ]

    @staticmethod
    def _physical_output() -> np.ndarray:
        value = np.zeros(OUTPUT_LAYOUT.size, np.float64)
        value[OUTPUT_LAYOUT["trajectory_direction"]] = np.tile((1.0, 0.0), 12)
        value[OUTPUT_LAYOUT["root_height"]] = 0.8
        value[OUTPUT_LAYOUT["phase_advance"]] = 0.1
        return value

    @staticmethod
    def _row(
        center: int, target: np.ndarray, *, phase: float
    ) -> dict[str, object]:
        x = np.zeros(INPUT_LAYOUT.size, np.float32)
        x[0] = np.float32(center)
        return {
            "x": x,
            "y": np.asarray(target, dtype=np.float32),
            "phase": np.float32(phase),
            "clip_id": "terrain_slopes__slope_000__000",
            "center_frame": center,
            "split_identity": "slope_000",
            "split": "train",
            "sequence_lane": "motion",
            "terrain_class": "flat",
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


class _FittedEnvelopeModel(torch.nn.Module):
    def __init__(self, output: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("output", torch.as_tensor(output, dtype=torch.float64))
        self.phases: list[float] = []

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        self.phases.extend(float(value) for value in phase.detach().cpu())
        return self.output.to(device=x.device).expand(len(x), -1)


class _TransitionPairRows:
    split = "train"
    x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
    x_std = np.ones(INPUT_LAYOUT.size, np.float32)
    y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
    y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

    _LOGICAL_ROWS = (
        ("clip_b", "motion", 11, "descent"),
        ("clip_a", "motion", 3, "flat"),
        ("clip_a", "motion", 1, "ascent"),
        ("clip_a", "idle_phase_0", 2, "transition"),
        ("clip_b", "motion", 10, "ascent"),
        ("clip_a", "motion", 2, "ascent"),
        ("clip_a", "idle_phase_0", 1, "flat"),
    )

    def __init__(
        self,
        logical_rows: object | None = None,
    ) -> None:
        values = self._LOGICAL_ROWS if logical_rows is None else logical_rows
        self.rows = [
            self._row(clip, lane, center, terrain)
            for clip, lane, center, terrain in values
        ]
        self.calls: list[int] = []

    @staticmethod
    def _row(
        clip: str, lane: str, center: int, terrain: str
    ) -> dict[str, object]:
        x = np.zeros(INPUT_LAYOUT.size, np.float64)
        y = np.zeros(OUTPUT_LAYOUT.size, np.float64)
        x[0] = float(center)
        y[0] = float(center) / 100.0
        return {
            "x": x,
            "y": y,
            "phase": np.float32(0.25),
            "clip_id": clip,
            "center_frame": center,
            "split_identity": f"{clip}_train",
            "split": "train",
            "sequence_lane": lane,
            "terrain_class": terrain,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        self.calls.append(index)
        return self.rows[index]


class _TargetEnvelopePairRows:
    def __init__(self) -> None:
        predecessor = np.zeros(OUTPUT_LAYOUT.size, np.float64)
        predecessor[OUTPUT_LAYOUT["joint_position"].start + 1] = 0.98
        boundary = predecessor.copy()
        boundary[OUTPUT_LAYOUT["joint_position"].start] = 0.225
        boundary[OUTPUT_LAYOUT["joint_position"].start + 1] = 0.98
        boundary[OUTPUT_LAYOUT["phase_advance"]] = 0.48
        tolerance = 9.5367431640625e-7
        negative_tolerance = boundary.copy()
        negative_tolerance[OUTPUT_LAYOUT["phase_advance"]] = -tolerance
        self.rows = [
            {"current": {"y": boundary}, "predecessor_y": predecessor.copy()},
            {
                "current": {"y": negative_tolerance},
                "predecessor_y": predecessor.copy(),
            },
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


def _global_envelope_worker(
    rank: int,
    world_size: int,
    init_method: str,
    case: str,
    queue: object,
) -> None:
    """Run one bounded global-envelope collective case in a spawned process."""

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=5),
    )
    original_all_gather = dist.all_gather
    counters = {"metadata": 0, "risk": 0, "ordinal": 0}
    call_count = 0

    def tracked_all_gather(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        name = ("metadata", "risk", "ordinal")[min(call_count - 1, 2)]
        counters[name] += 1
        return original_all_gather(*args, **kwargs)

    configurations: dict[str, tuple[tuple[float, ...], tuple[int, ...], float]] = {
        "cross_rank_max_tie": (
            (2.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.25,
        ),
        "tail_rank_1_only": (
            (0.0, 0.0) if rank == 0 else (4.0, 3.0),
            (0, 2) if rank == 0 else (1, 3),
            0.50,
        ),
        "unequal_counts": (
            (1.0, 0.0) if rank == 0 else (1.0,),
            (0, 2) if rank == 0 else (1,),
            0.10,
        ),
        "rank_mismatch": (
            (1.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10,
        ),
        "rank_integer_overflow": (
            (1.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10,
        ),
        "world_size_integer_overflow": (
            (1.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10,
        ),
        "ordinal_gap": (
            (1.0, 0.0),
            (0, 3) if rank == 0 else (1, 4),
            0.10,
        ),
        "one_rank_nonfinite": (
            (1.0, 0.0) if rank == 0 else (float("inf"), 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10,
        ),
        "dtype_mismatch": (
            (1.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10,
        ),
        "device_mismatch": (
            (1.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10,
        ),
        "fraction_mismatch": (
            (1.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10 if rank == 0 else 0.20,
        ),
        "invalid_fraction": (
            (1.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10 if rank == 0 else float("nan"),
        ),
        "safe_rank_appended": (
            (9.0, 0.0) if rank == 0 else (0.0, 0.0),
            (0, 2) if rank == 0 else (1, 3),
            0.10,
        ),
    }
    coefficients, ordinals, tail_fraction = configurations[case]
    supplied_rank = (
        2**100
        if case == "rank_integer_overflow" and rank == 0
        else (0 if case == "rank_mismatch" else rank)
    )
    supplied_world_size = (
        2**100
        if case == "world_size_integer_overflow" and rank == 0
        else world_size
    )
    device = torch.device(
        "cuda", torch.cuda.current_device()
    ) if case == "device_mismatch" and rank == 1 else torch.device("cpu")
    dtype = (
        torch.float32
        if case == "dtype_mismatch" and rank == 0
        else torch.float64
    )
    parameter = torch.tensor(1.0, dtype=dtype, device=device, requires_grad=True)
    local_risk = torch.tensor(coefficients, dtype=dtype, device=device) * parameter
    global_ordinals = torch.tensor(ordinals, dtype=torch.int64, device=device)
    payload: dict[str, object]
    try:
        with patch.object(dist, "all_gather", side_effect=tracked_all_gather):
            reduction = training_module.global_max_plus_tail(
                local_risk,
                global_ordinals=global_ordinals,
                tail_fraction=tail_fraction,
                rank=supplied_rank,
                world_size=supplied_world_size,
            )
        maximum_gradient = torch.autograd.grad(
            reduction.maximum, local_risk, retain_graph=True
        )[0]
        parameter_gradients = {
            "maximum": float(
                torch.autograd.grad(
                    reduction.maximum, parameter, retain_graph=True
                )[0].detach().cpu()
            ),
            "cvar": float(
                torch.autograd.grad(
                    reduction.cvar, parameter, retain_graph=True
                )[0].detach().cpu()
            ),
            "positive_tail_mean": float(
                torch.autograd.grad(
                    reduction.positive_tail_mean,
                    parameter,
                    retain_graph=True,
                )[0].detach().cpu()
            ),
            "loss": float(
                torch.autograd.grad(reduction.loss, parameter)[0].detach().cpu()
            ),
        }
        payload = {
            "rank": rank,
            "error": None,
            "ordinals": list(ordinals),
            "maximum_gradient": maximum_gradient.detach().cpu().tolist(),
            "parameter_gradients": parameter_gradients,
            "maximum": float(reduction.maximum.detach().cpu()),
            "cvar": float(reduction.cvar.detach().cpu()),
            "tail_count": reduction.tail_count,
            "active_count": reduction.active_count,
            "positive_tail_mean": float(
                reduction.positive_tail_mean.detach().cpu()
            ),
            "loss": float(reduction.loss.detach().cpu()),
            "counters": counters,
        }
    except Exception as error:  # sent to the parent for collective assertions
        payload = {
            "rank": rank,
            "error": (type(error).__name__, str(error)),
            "counters": counters,
        }
    finally:
        dist.destroy_process_group()
    queue.put(payload)


def _spawn_global_envelope_workers(case: str) -> list[dict[str, object]]:
    context = torch.multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as temporary:
        init_method = (Path(temporary) / "gloo-init").as_uri()
        torch.multiprocessing.spawn(
            _global_envelope_worker,
            args=(2, init_method, case, queue),
            nprocs=2,
            join=True,
        )
        results = [queue.get() for _ in range(2)]
    return sorted(results, key=lambda item: int(item["rank"]))


def _single_rank_global_envelope_reference(
    case: str,
) -> tuple[training_module.GlobalEnvelopeReduction, dict[str, float]]:
    coefficients, tail_fraction = {
        "cross_rank_max_tie": ((2.0, 2.0, 0.0, 0.0), 0.25),
        "tail_rank_1_only": ((0.0, 4.0, 0.0, 3.0), 0.50),
    }[case]
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    reduction = training_module.global_max_plus_tail(
        torch.tensor(coefficients, dtype=torch.float64) * parameter,
        global_ordinals=torch.arange(4, dtype=torch.int64),
        tail_fraction=tail_fraction,
        rank=0,
        world_size=1,
    )
    gradients = {
        "maximum": float(
            torch.autograd.grad(
                reduction.maximum, parameter, retain_graph=True
            )[0]
        ),
        "cvar": float(
            torch.autograd.grad(reduction.cvar, parameter, retain_graph=True)[0]
        ),
        "positive_tail_mean": float(
            torch.autograd.grad(
                reduction.positive_tail_mean, parameter, retain_graph=True
            )[0]
        ),
        "loss": float(torch.autograd.grad(reduction.loss, parameter)[0]),
    }
    return reduction, gradients


class TerrainPFNNTrainingTests(unittest.TestCase):
    def _fitted_envelope(
        self,
        output: np.ndarray,
        *,
        joint_limits: np.ndarray | None = None,
        phase_advance_q99: float = 0.2,
    ) -> tuple[object, _FittedEnvelopeRows, _FittedEnvelopeModel]:
        rows = _FittedEnvelopeRows(output)
        model = _FittedEnvelopeModel(output)
        report = training_module.evaluate_fitted_transition_envelope(
            model,
            rows,
            ((0, 1),),
            normalization=rows,
            joint_limits=(
                np.asarray([[-2.0, 2.0]] * 29, np.float64)
                if joint_limits is None
                else joint_limits
            ),
            phase_advance_q99=phase_advance_q99,
        )
        return report, rows, model

    def _assert_metadata_rejection(self, case: str) -> None:
        results = _spawn_global_envelope_workers(case)
        errors = [result["error"] for result in results]
        self.assertEqual(errors[0], errors[1])
        self.assertEqual(
            errors[0],
            ("ValueError", "global envelope metadata is invalid"),
        )
        for result in results:
            self.assertEqual(
                result["counters"],
                {"metadata": 1, "risk": 0, "ordinal": 0},
            )

    def test_physical_envelope_risks_use_exact_margins_and_phase_gradients(
        self,
    ) -> None:
        contract = training_module.PhysicalEnvelopeObjective()
        prediction = torch.zeros(
            (5, OUTPUT_LAYOUT.size), dtype=torch.float64, requires_grad=True
        )
        reached = torch.zeros_like(prediction)
        prediction.data[:, OUTPUT_LAYOUT["joint_position"]] = 0.0
        reached.data[:, OUTPUT_LAYOUT["joint_position"]] = 0.0
        prediction.data[0, OUTPUT_LAYOUT["joint_position"].start] = 0.225
        prediction.data[1, OUTPUT_LAYOUT["joint_position"].start] = 0.250
        prediction.data[2, OUTPUT_LAYOUT["phase_advance"]] = 0.0
        prediction.data[3, OUTPUT_LAYOUT["phase_advance"]] = 0.48
        prediction.data[4, OUTPUT_LAYOUT["phase_advance"]] = -1.0e-12
        risks = training_module.physical_envelope_risks(
            prediction,
            reached,
            normalization=_normalization(),
            joint_limits=torch.tensor([[-1.0, 1.0]] * 29),
            phase_advance_cap=0.50,
            contract=contract,
        )
        self.assertEqual(float(risks["joint_step"][0].detach()), 0.0)
        self.assertAlmostEqual(
            float(risks["joint_step"][1].detach()), 1.0, places=6
        )
        self.assertEqual(float(risks["phase"][2].detach()), 0.0)
        self.assertEqual(float(risks["phase"][3].detach()), 0.0)
        self.assertAlmostEqual(
            float(risks["phase"][4].detach()), 5.0e-11, places=20
        )
        sum(value.sum() for value in risks.values()).backward()
        self.assertEqual(
            float(
                prediction.grad[0, OUTPUT_LAYOUT["joint_position"].start]
            ),
            0.0,
        )
        self.assertEqual(
            float(prediction.grad[2, OUTPUT_LAYOUT["phase_advance"].start]),
            0.0,
        )
        self.assertEqual(
            float(prediction.grad[4, OUTPUT_LAYOUT["phase_advance"].start]),
            -50.0,
        )

    def test_physical_envelope_risks_denormalize_once_and_keep_worst_joint(
        self,
    ) -> None:
        prediction = torch.zeros(
            (6, OUTPUT_LAYOUT.size), dtype=torch.float64, requires_grad=True
        )
        reached = torch.zeros_like(prediction)
        joint = OUTPUT_LAYOUT["joint_position"]
        phase = OUTPUT_LAYOUT["phase_advance"]
        normal = _normalization()
        normal["y_std"][joint.start + 28] = 2.0
        prediction.data[0, joint.start + 28] = 0.125
        prediction.data[1, joint.start] = -1.0
        prediction.data[2, joint.start + 1] = 1.0
        prediction.data[3, phase] = 0.50
        prediction.data[4, joint.start] = -0.98
        prediction.data[5, joint.start] = 0.98
        risks = training_module.physical_envelope_risks(
            prediction,
            reached,
            normalization=normal,
            joint_limits=torch.tensor([[-1.0, 1.0]] * 29),
            phase_advance_cap=0.50,
            contract=training_module.PhysicalEnvelopeObjective(),
        )
        torch.testing.assert_close(
            risks["joint_step"],
            torch.tensor(
                (1.0, 961.0, 961.0, 0.0, 912.04, 912.04),
                dtype=torch.float64,
            ),
        )
        self.assertAlmostEqual(
            float(risks["joint_limit"][1].detach()), 1.0, places=12
        )
        self.assertAlmostEqual(
            float(risks["joint_limit"][2].detach()), 1.0, places=12
        )
        self.assertEqual(float(risks["joint_limit"][4].detach()), 0.0)
        self.assertEqual(float(risks["joint_limit"][5].detach()), 0.0)
        self.assertAlmostEqual(
            float(risks["phase"][3].detach()), 1.0, places=12
        )

    def test_physical_envelope_satisfied_boundaries_have_zero_gradient(
        self,
    ) -> None:
        prediction = torch.zeros(
            (5, OUTPUT_LAYOUT.size), dtype=torch.float64, requires_grad=True
        )
        reached = torch.zeros_like(prediction)
        joint = OUTPUT_LAYOUT["joint_position"]
        phase = OUTPUT_LAYOUT["phase_advance"]
        prediction.data[0, joint.start] = 0.225
        prediction.data[1, joint.start] = -0.98
        reached.data[1, joint.start] = -0.98
        prediction.data[2, joint.start] = 0.98
        reached.data[2, joint.start] = 0.98
        prediction.data[3, phase] = 0.0
        prediction.data[4, phase] = 0.48
        risks = training_module.physical_envelope_risks(
            prediction,
            reached,
            normalization=_normalization(),
            joint_limits=torch.tensor([[-1.0, 1.0]] * 29),
            phase_advance_cap=0.50,
            contract=training_module.PhysicalEnvelopeObjective(),
        )
        for risk in risks.values():
            torch.testing.assert_close(risk, torch.zeros_like(risk))
        sum(risk.sum() for risk in risks.values()).backward()
        self.assertEqual(float(prediction.grad.abs().sum()), 0.0)

    def test_physical_envelope_risks_reject_invalid_shapes_dtypes_and_contracts(
        self,
    ) -> None:
        prediction = torch.zeros(2, OUTPUT_LAYOUT.size)
        reached = torch.zeros_like(prediction)
        arguments = {
            "normalization": _normalization(),
            "joint_limits": torch.tensor([[-1.0, 1.0]] * 29),
            "phase_advance_cap": 0.50,
            "contract": training_module.PhysicalEnvelopeObjective(),
        }
        invalid_tensors = (
            (prediction.reshape(1, 2, -1), reached.reshape(1, 2, -1)),
            (prediction[:, :-1], reached[:, :-1]),
            (prediction.to(torch.int64), reached.to(torch.int64)),
            (prediction, reached.to(torch.float64)),
        )
        for invalid_prediction, invalid_reached in invalid_tensors:
            with self.subTest(shape=tuple(invalid_prediction.shape)):
                with self.assertRaisesRegex(ValueError, "tensors"):
                    training_module.physical_envelope_risks(
                        invalid_prediction, invalid_reached, **arguments
                    )
        nonfinite = prediction.clone()
        nonfinite[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            training_module.physical_envelope_risks(
                nonfinite, reached, **arguments
            )
        for phase_cap in (0.020, 0.0, float("nan")):
            with self.subTest(phase_cap=phase_cap):
                with self.assertRaisesRegex(ValueError, "phase cap"):
                    training_module.physical_envelope_risks(
                        prediction,
                        reached,
                        **{**arguments, "phase_advance_cap": phase_cap},
                    )
        with self.assertRaisesRegex(ValueError, "joint limits"):
            training_module.physical_envelope_risks(
                prediction,
                reached,
                **{
                    **arguments,
                    "joint_limits": torch.tensor([[-0.01, 0.01]] * 29),
                },
            )
        with self.assertRaisesRegex(ValueError, "contract"):
            training_module.physical_envelope_risks(
                prediction,
                reached,
                **{
                    **arguments,
                    "contract": training_module.PhysicalEnvelopeObjective(
                        tail_fraction=0.20
                    ),
                },
            )

    def test_global_max_plus_tail_is_invariant_to_appended_safe_elements(
        self,
    ) -> None:
        for count in (1, 9, 10, 11, 32, 100, 1000):
            offender = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
            risk = torch.cat((offender.reshape(1), torch.zeros(count - 1)))
            reduction = training_module.global_max_plus_tail(
                risk,
                global_ordinals=torch.arange(count, dtype=torch.int64),
                tail_fraction=0.10,
                rank=0,
                world_size=1,
            )
            self.assertEqual(float(reduction.maximum.detach()), 9.0)
            self.assertEqual(reduction.active_count, 1)
            self.assertEqual(float(reduction.positive_tail_mean.detach()), 9.0)
            self.assertEqual(float(reduction.loss.detach()), 18.0)
            reduction.loss.backward()
            self.assertEqual(float(offender.grad), 2.0)

    def test_global_max_plus_tail_ties_use_ascending_global_ordinal(self) -> None:
        risk = torch.ones(20, requires_grad=True)
        ordinals = torch.arange(20, dtype=torch.int64).flip(0)
        reduction = training_module.global_max_plus_tail(
            risk,
            global_ordinals=ordinals,
            tail_fraction=0.10,
            rank=0,
            world_size=1,
        )
        reduction.positive_tail_mean.backward()
        selected = set(
            ordinals[torch.nonzero(risk.grad, as_tuple=False).flatten()].tolist()
        )
        self.assertEqual(selected, {0, 1})

    def test_global_max_plus_tail_rejects_invalid_local_contract(self) -> None:
        valid_risk = torch.ones(2)
        valid_ordinals = torch.arange(2, dtype=torch.int64)
        cases = (
            (valid_risk.reshape(1, 2), valid_ordinals, 0.10, 0, 1),
            (valid_risk.to(torch.float16), valid_ordinals, 0.10, 0, 1),
            (torch.empty(0), torch.empty(0, dtype=torch.int64), 0.10, 0, 1),
            (valid_risk, valid_ordinals.to(torch.int32), 0.10, 0, 1),
            (valid_risk, valid_ordinals[:1], 0.10, 0, 1),
            (valid_risk, valid_ordinals, 0.0, 0, 1),
            (valid_risk, valid_ordinals, 1.01, 0, 1),
            (valid_risk, valid_ordinals, float("nan"), 0, 1),
            (valid_risk, valid_ordinals, 0.10, 1, 1),
            (valid_risk, valid_ordinals, 0.10, 0, 2),
        )
        for risk, ordinals, fraction, rank, world_size in cases:
            with self.subTest(
                shape=tuple(risk.shape),
                dtype=risk.dtype,
                fraction=fraction,
                rank=rank,
                world_size=world_size,
            ):
                with self.assertRaisesRegex(ValueError, "metadata"):
                    training_module.global_max_plus_tail(
                        risk,
                        global_ordinals=ordinals,
                        tail_fraction=fraction,
                        rank=rank,
                        world_size=world_size,
                    )
        for ordinals in (
            torch.tensor((0, 2)),
            torch.tensor((0, 0)),
            torch.tensor((-1, 0)),
            torch.tensor((0, 3)),
        ):
            with self.subTest(ordinals=ordinals.tolist()):
                with self.assertRaisesRegex(ValueError, "ordinals"):
                    training_module.global_max_plus_tail(
                        valid_risk,
                        global_ordinals=ordinals,
                        tail_fraction=0.10,
                        rank=0,
                        world_size=1,
                    )

    def test_physical_envelope_loss_safe_append_is_invariant_for_batch_and_rollout(
        self,
    ) -> None:
        references: list[tuple[str, float, float]] = []
        for label, sizes in (
            ("batch", ((1,), (11,), (100,))),
            ("rollout", ((1, 1), (5, 2), (11, 3))),
        ):
            for shape in sizes:
                count = math.prod(shape)
                prediction = torch.zeros(
                    (*shape, OUTPUT_LAYOUT.size), dtype=torch.float64
                ).reshape(count, OUTPUT_LAYOUT.size)
                prediction.requires_grad_()
                prediction.data[0, OUTPUT_LAYOUT["joint_position"].start] = 0.30
                reached = torch.zeros_like(prediction)
                losses = training_module.physical_envelope_loss(
                    prediction,
                    reached,
                    normalization=_normalization(),
                    joint_limits=torch.tensor([[-1.0, 1.0]] * 29),
                    phase_advance_cap=0.50,
                    contract=training_module.PhysicalEnvelopeObjective(),
                    global_ordinals=torch.arange(count, dtype=torch.int64),
                    rank=0,
                    world_size=1,
                )
                losses["joint_step_envelope"].backward()
                observed = (
                    float(losses["joint_step_envelope"].detach()),
                    float(
                        prediction.grad[
                            0, OUTPUT_LAYOUT["joint_position"].start
                        ]
                    ),
                )
                if not references or references[-1][0] != label:
                    references.append((label, *observed))
                self.assertEqual(observed, references[-1][1:])

    def test_physical_envelope_loss_reports_independent_family_reductions(
        self,
    ) -> None:
        prediction = torch.zeros(3, OUTPUT_LAYOUT.size, dtype=torch.float64)
        prediction[0, OUTPUT_LAYOUT["joint_position"].start] = 0.25
        prediction[1, OUTPUT_LAYOUT["joint_position"].start] = 1.0
        prediction[2, OUTPUT_LAYOUT["phase_advance"]] = -0.02
        losses = training_module.physical_envelope_loss(
            prediction,
            torch.zeros_like(prediction),
            normalization=_normalization(),
            joint_limits=torch.tensor([[-1.0, 1.0]] * 29),
            phase_advance_cap=0.50,
            contract=training_module.PhysicalEnvelopeObjective(),
            global_ordinals=torch.arange(3, dtype=torch.int64),
            rank=0,
            world_size=1,
        )
        expected = {
            f"{family}_{suffix}"
            for family in ("joint_step", "joint_limit", "phase")
            for suffix in ("maximum", "cvar", "positive_tail_mean", "envelope")
        }
        self.assertEqual(set(losses), {"physical_envelope_total", *expected})
        torch.testing.assert_close(
            losses["physical_envelope_total"],
            sum(losses[f"{family}_envelope"] for family in (
                "joint_step", "joint_limit", "phase"
            )),
        )

    def test_two_rank_global_max_plus_tail_table_and_gradients(self) -> None:
        cases = {
            "cross_rank_max_tie": (2.0, 2.0, 1, 1, 2.0, 4.0),
            "tail_rank_1_only": (4.0, 3.5, 2, 2, 3.5, 7.5),
        }
        for case, expected in cases.items():
            with self.subTest(case=case):
                reference, reference_gradients = (
                    _single_rank_global_envelope_reference(case)
                )
                results = _spawn_global_envelope_workers(case)
                self.assertTrue(all(result["error"] is None for result in results))
                for result in results:
                    observed = (
                        result["maximum"], result["cvar"],
                        result["tail_count"], result["active_count"],
                        result["positive_tail_mean"], result["loss"],
                    )
                    self.assertEqual(observed, expected)
                    self.assertEqual(
                        observed,
                        (
                            float(reference.maximum.detach()),
                            float(reference.cvar.detach()),
                            reference.tail_count,
                            reference.active_count,
                            float(reference.positive_tail_mean.detach()),
                            float(reference.loss.detach()),
                        ),
                    )
                    self.assertEqual(
                        result["counters"],
                        {"metadata": 1, "risk": 1, "ordinal": 1},
                    )
                if case == "cross_rank_max_tie":
                    nonzero_ordinals = {
                        ordinal
                        for result in results
                        for ordinal, gradient in zip(
                            result["ordinals"], result["maximum_gradient"]
                        )
                        if gradient != 0.0
                    }
                    self.assertEqual(nonzero_ordinals, {0})
                averaged_gradients = {
                    name: sum(
                        float(result["parameter_gradients"][name])
                        for result in results
                    ) / len(results)
                    for name in reference_gradients
                }
                self.assertEqual(averaged_gradients, reference_gradients)

    def test_two_rank_global_max_plus_tail_rejects_collectively(self) -> None:
        for case in (
            "unequal_counts",
            "rank_mismatch",
            "rank_integer_overflow",
            "world_size_integer_overflow",
            "one_rank_nonfinite",
        ):
            with self.subTest(case=case):
                self._assert_metadata_rejection(case)
        results = _spawn_global_envelope_workers("ordinal_gap")
        self.assertEqual(results[0]["error"], results[1]["error"])
        self.assertEqual(
            results[0]["error"],
            (
                "ValueError",
                "global envelope ordinals must be exactly 0..N-1",
            ),
        )
        for result in results:
            self.assertEqual(
                result["counters"],
                {"metadata": 1, "risk": 1, "ordinal": 1},
            )

    def test_two_rank_metadata_rejects_risk_dtype_mismatch_before_value_gather(
        self,
    ) -> None:
        self._assert_metadata_rejection("dtype_mismatch")

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is unavailable; gloo CPU/CUDA mismatch requires CUDA",
    )
    def test_two_rank_metadata_rejects_device_type_mismatch_before_value_gather(
        self,
    ) -> None:
        self._assert_metadata_rejection("device_mismatch")

    def test_two_rank_metadata_rejects_fraction_bit_mismatch_before_value_gather(
        self,
    ) -> None:
        self._assert_metadata_rejection("fraction_mismatch")

    def test_two_rank_metadata_rejects_one_invalid_fraction_before_value_gather(
        self,
    ) -> None:
        self._assert_metadata_rejection("invalid_fraction")

    def test_two_rank_safe_rank_append_preserves_loss_and_ddp_gradient(self) -> None:
        parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        reference = training_module.global_max_plus_tail(
            parameter.reshape(1) * 9.0,
            global_ordinals=torch.tensor((0,), dtype=torch.int64),
            tail_fraction=0.10,
            rank=0,
            world_size=1,
        )
        reference_gradient = torch.autograd.grad(reference.loss, parameter)[0]
        results = _spawn_global_envelope_workers("safe_rank_appended")
        self.assertTrue(all(result["error"] is None for result in results))
        self.assertTrue(
            all(
                result["loss"] == float(reference.loss.detach())
                for result in results
            )
        )
        averaged_gradient = sum(
            float(result["parameter_gradients"]["loss"])
            for result in results
        ) / len(results)
        self.assertEqual(averaged_gradient, float(reference_gradient))

    def test_fitted_transition_report_is_canonical_and_uses_stored_phase(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        report, rows, model = self._fitted_envelope(output)
        self.assertIsInstance(report, training_module.FittedTransitionReport)
        self.assertTrue(report.accepted)
        self.assertEqual(report.sample_count, 1)
        self.assertIsNone(report.first_failure)
        self.assertEqual(model.phases, [float(rows[1]["phase"])])
        payload = report.to_dict()
        self.assertEqual(payload["schema"], "mm-sonic-fitted-transition-report/v1")
        self.assertEqual(set(payload), {
            "schema", "accepted", "sample_count", "maxima", "first_failure",
            "rows_sha256", "report_sha256",
        })
        base = {key: value for key, value in payload.items() if key != "report_sha256"}
        self.assertEqual(
            payload["report_sha256"],
            hashlib.sha256(json.dumps(
                base, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest(),
        )

    def test_fitted_transition_rejects_diluted_joint_step_with_exact_row(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        output[OUTPUT_LAYOUT["joint_position"]][0] = 0.251
        rows = _FittedEnvelopeRows(output)
        normalized_mse = np.square(
            output.astype(np.float32) - np.asarray(rows[1]["y"])
        ).mean()
        self.assertLess(normalized_mse, 0.001)
        report, _, _ = self._fitted_envelope(output)
        self.assertFalse(report.accepted)
        self.assertEqual(report.first_failure["clip_id"], rows[1]["clip_id"])
        self.assertEqual(report.first_failure["sequence_lane"], "motion")
        self.assertEqual(report.first_failure["center_frame"], 11)
        self.assertEqual(report.first_failure["field"], "joint_position")
        self.assertEqual(report.first_failure["joint"], ISAACLAB_JOINT_NAMES[0])
        self.assertAlmostEqual(report.first_failure["value"], 0.251)
        self.assertEqual(report.first_failure["limit"], 0.25)

    def test_fitted_transition_rejects_root_translation_step(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        output[OUTPUT_LAYOUT["root_planar_velocity"]] = (0.061 * 30.0, 0.0)
        report, _, _ = self._fitted_envelope(output)
        self.assertEqual(report.first_failure["field"], "root_translation_step_m")
        self.assertAlmostEqual(report.first_failure["value"], 0.061)
        self.assertEqual(report.first_failure["limit"], 0.060)

    def test_fitted_transition_rejects_root_rotation_step(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        output[OUTPUT_LAYOUT["root_yaw_velocity"]] = 0.351 * 30.0
        report, _, _ = self._fitted_envelope(output)
        self.assertEqual(report.first_failure["field"], "root_rotation_step_rad")
        self.assertAlmostEqual(report.first_failure["value"], 0.351)
        self.assertEqual(report.first_failure["limit"], 0.35)

    def test_fitted_transition_rejects_native_joint_limit(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        output[OUTPUT_LAYOUT["joint_position"]][3] = 0.11
        limits = np.asarray([[-2.0, 2.0]] * 29, np.float64)
        limits[3] = (-0.1, 0.1)
        report, _, _ = self._fitted_envelope(output, joint_limits=limits)
        self.assertEqual(report.first_failure["field"], "joint_limit")
        self.assertEqual(report.first_failure["joint"], ISAACLAB_JOINT_NAMES[3])
        self.assertAlmostEqual(report.first_failure["value"], 0.11)
        self.assertEqual(report.first_failure["limit"], 0.1)

    def test_fitted_transition_rejects_raw_direction_norm(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        output[OUTPUT_LAYOUT["trajectory_direction"]][:2] = (0.49, 0.0)
        report, _, _ = self._fitted_envelope(output)
        self.assertEqual(report.first_failure["field"], "trajectory_direction_norm")
        self.assertAlmostEqual(report.first_failure["value"], 0.49)
        self.assertEqual(report.first_failure["limit"], 0.5)

    def test_fitted_transition_rejects_phase_advance_bounds(self) -> None:
        for value, limit in ((-0.001, 0.0), (0.301, 0.3)):
            with self.subTest(value=value):
                output = _FittedEnvelopeRows._physical_output()
                output[OUTPUT_LAYOUT["phase_advance"]] = value
                report, _, _ = self._fitted_envelope(output)
                self.assertEqual(report.first_failure["field"], "phase_advance")
                self.assertAlmostEqual(report.first_failure["value"], value)
                self.assertAlmostEqual(report.first_failure["limit"], limit)

    def test_fitted_transition_rejects_nonfinite_phase_and_output(self) -> None:
        for field, offset in (
            ("phase_advance", OUTPUT_LAYOUT["phase_advance"].start),
            ("output", OUTPUT_LAYOUT["body_position"].start),
        ):
            with self.subTest(field=field):
                output = _FittedEnvelopeRows._physical_output()
                output[offset] = np.nan
                report, _, _ = self._fitted_envelope(output)
                self.assertEqual(report.first_failure["field"], field)
                self.assertEqual(report.first_failure["value"], "nan")
                self.assertEqual(report.first_failure["limit"], "finite")

    def test_fitted_transition_rejects_nonpositive_root_height(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        output[OUTPUT_LAYOUT["root_height"]] = 0.0
        report, _, _ = self._fitted_envelope(output)
        self.assertEqual(report.first_failure["field"], "root_height")
        self.assertEqual(report.first_failure["value"], 0.0)
        self.assertEqual(report.first_failure["limit"], 0.0)

    def test_fitted_transition_rejects_nonfinite_reconstructed_quaternion(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        output[OUTPUT_LAYOUT["root_tilt"]] = (1.5e308, 1.5e308)
        report, _, _ = self._fitted_envelope(output)
        self.assertEqual(report.first_failure["field"], "root_quaternion_wxyz")
        self.assertEqual(report.first_failure["value"], "nonfinite")
        self.assertEqual(report.first_failure["limit"], "finite_normalized")

    def test_fitted_adjacent_indices_are_complete_lane_safe_and_hash_bound(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        rows = _FittedEnvelopeRows(output)
        idle = dict(rows.rows[1])
        idle["sequence_lane"] = "idle_phase_0"
        rows.rows.append(idle)
        self.assertEqual(
            training_module.fitted_adjacent_indices(rows, range(len(rows))),
            ((0, 1),),
        )
        report, _, _ = self._fitted_envelope(output)
        changed = dict(rows.rows[1])
        changed["phase"] = np.float32(0.76)
        rows.rows[1] = changed
        changed_model = _FittedEnvelopeModel(output)
        changed_report = training_module.evaluate_fitted_transition_envelope(
            changed_model,
            rows,
            ((0, 1),),
            normalization=rows,
            joint_limits=np.asarray([[-2.0, 2.0]] * 29, np.float64),
            phase_advance_q99=0.2,
        )
        self.assertNotEqual(report.rows_sha256, changed_report.rows_sha256)

    def test_predecessor_pair_receipt_is_complete_lane_safe_and_hash_bound(
        self,
    ) -> None:
        rows = _TransitionPairRows()
        pairs = train_module.canonical_transition_pairs(rows)
        self.assertEqual(
            [
                (
                    pair.clip_id,
                    pair.sequence_lane,
                    pair.predecessor_center_frame,
                    pair.center_frame,
                )
                for pair in pairs
            ],
            [
                ("clip_a", "idle_phase_0", 1, 2),
                ("clip_a", "motion", 1, 2),
                ("clip_a", "motion", 2, 3),
                ("clip_b", "motion", 10, 11),
            ],
        )
        receipt = train_module.fitted_transition_pair_receipt(rows)
        self.assertEqual(
            receipt["schema"],
            "mm-sonic-fitted-transition-pair-receipt/v1",
        )
        self.assertEqual(receipt["active_row_count"], 7)
        self.assertEqual(receipt["sample_count"], 4)
        self.assertEqual(
            receipt["class_counts"],
            {"flat": 1, "ascent": 1, "descent": 1, "transition": 1},
        )
        self.assertNotIn("pairs", receipt)
        self.assertEqual(
            train_module.validate_fitted_transition_pair_receipt(receipt, rows),
            receipt,
        )

        reversed_rows = _TransitionPairRows(reversed(_TransitionPairRows._LOGICAL_ROWS))
        reversed_receipt = train_module.fitted_transition_pair_receipt(reversed_rows)
        self.assertEqual(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            json.dumps(reversed_receipt, sort_keys=True, separators=(",", ":")),
        )

        tamper_cases = {
            "pairs digest": lambda value: value.__setitem__(
                "pairs_sha256", "0" * 64
            ),
            "sample count": lambda value: value.__setitem__(
                "sample_count", value["sample_count"] + 1
            ),
            "active row count": lambda value: value.__setitem__(
                "active_row_count", value["active_row_count"] + 1
            ),
            "active row digest": lambda value: value.__setitem__(
                "active_rows_sha256", "1" * 64
            ),
            "class count": lambda value: value["class_counts"].__setitem__(
                "flat", value["class_counts"]["flat"] + 1
            ),
            "receipt digest": lambda value: value.__setitem__(
                "receipt_sha256", "2" * 64
            ),
        }
        for name, mutate in tamper_cases.items():
            with self.subTest(tamper=name):
                tampered = json.loads(json.dumps(receipt))
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, "pair receipt"):
                    train_module.validate_fitted_transition_pair_receipt(
                        tampered, rows
                    )

        candidate_cases = {
            "omitted": pairs[1:],
            "extra nonadjacent": (
                *pairs,
                replace(pairs[-1], predecessor_center_frame=8),
            ),
            "duplicated": (*pairs, pairs[1]),
            "reordered": (pairs[0], pairs[2], pairs[1], pairs[3]),
        }
        for name, candidate in candidate_cases.items():
            with self.subTest(candidate=name):
                with self.assertRaisesRegex(ValueError, "recomputation"):
                    train_module.validate_canonical_transition_pairs(
                        rows, candidate
                    )

        split_tampered = _TransitionPairRows()
        split_tampered.rows[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "transition-pair"):
            train_module.validate_canonical_transition_pairs(
                split_tampered, pairs
            )

        real_adjacent = training_module.fitted_adjacent_indices
        with patch.object(
            train_module,
            "canonical_transition_pairs",
            wraps=train_module.canonical_transition_pairs,
        ) as primary, patch.object(
            train_module,
            "fitted_adjacent_indices",
            side_effect=lambda dataset, indices: tuple(
                reversed(real_adjacent(dataset, indices))
            ),
        ) as independent:
            self.assertEqual(
                train_module.fitted_transition_pair_receipt(rows), receipt
            )
        self.assertGreaterEqual(primary.call_count, 1)
        independent.assert_called_once()

        with patch.object(
            train_module,
            "fitted_adjacent_indices",
            side_effect=lambda dataset, indices: real_adjacent(dataset, indices)[1:],
        ):
            with self.assertRaisesRegex(ValueError, "recomputation"):
                train_module.fitted_transition_pair_receipt(rows)

    def test_synthetic_target_envelope_audit_accepts_exact_boundaries_and_rejects_each_excess(
        self,
    ) -> None:
        rows = _TargetEnvelopePairRows()
        arguments = {
            "normalization": _normalization(),
            "joint_limits": np.asarray([[-1.0, 1.0]] * 29, np.float64),
            "phase_advance_cap": 0.50,
            "contract": training_module.DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
        }
        audit = training_module.fitted_target_envelope_audit(rows, **arguments)
        tolerance = 9.5367431640625e-7
        self.assertEqual(
            training_module.PHASE_AUDIT_NEGATIVE_TOLERANCE_RAD,
            tolerance,
        )
        self.assertEqual(
            training_module.PHASE_AUDIT_NEGATIVE_TOLERANCE_RAD,
            8 * np.finfo(np.float32).eps,
        )
        self.assertEqual(audit["sample_count"], 2)
        self.assertEqual(audit["maximum_joint_step_rad"], 0.225)
        self.assertAlmostEqual(
            audit["minimum_joint_limit_clearance_rad"], 0.02, places=15
        )
        self.assertEqual(audit["minimum_phase_advance_unclamped_rad"], -tolerance)
        self.assertEqual(audit["minimum_phase_advance_rad"], 0.0)
        self.assertEqual(audit["maximum_phase_advance_rad"], 0.48)
        self.assertEqual(audit["joint_step_excess_count"], 0)
        self.assertEqual(audit["joint_limit_margin_excess_count"], 0)
        self.assertEqual(audit["phase_negative_excess_count"], 0)
        self.assertEqual(audit["phase_upper_excess_count"], 0)
        base = {
            name: value
            for name, value in audit.items()
            if name != "audit_sha256"
        }
        self.assertEqual(
            audit["audit_sha256"],
            hashlib.sha256(
                json.dumps(
                    base,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        )

        cases = (
            (
                "joint step",
                0,
                OUTPUT_LAYOUT["joint_position"].start,
                0.225001,
            ),
            (
                "joint limit",
                0,
                OUTPUT_LAYOUT["joint_position"].start + 1,
                0.980001,
            ),
            (
                "phase minimum",
                1,
                OUTPUT_LAYOUT["phase_advance"].start,
                -1.0e-6,
            ),
            (
                "phase upper",
                0,
                OUTPUT_LAYOUT["phase_advance"].start,
                0.480001,
            ),
        )
        for field, row_index, offset, value in cases:
            with self.subTest(field=field):
                changed = _TargetEnvelopePairRows()
                current = dict(changed.rows[row_index]["current"])
                target = np.asarray(current["y"]).copy()
                target[offset] = value
                current["y"] = target
                changed.rows[row_index] = {
                    **changed.rows[row_index],
                    "current": current,
                }
                maximum = np.maximum
                with patch.object(
                    training_module.np, "maximum", wraps=maximum
                ) as audit_clamp:
                    with self.assertRaisesRegex(ValueError, field):
                        training_module.fitted_target_envelope_audit(
                            changed, **arguments
                        )
                if field == "phase minimum":
                    self.assertFalse(
                        any(
                            len(call.args) == 2 and call.args[1] == 0.0
                            for call in audit_clamp.call_args_list
                        )
                    )

    def test_transition_pairs_keep_lanes_distinct_and_gaps_terminate_runs(
        self,
    ) -> None:
        rows = _TransitionPairRows(
            (
                ("clip", "motion", 1, "flat"),
                ("clip", "idle_phase_0", 1, "ascent"),
                ("clip", "motion", 2, "descent"),
                ("clip", "idle_phase_0", 2, "transition"),
                ("clip", "motion", 4, "flat"),
                ("clip", "motion", 6, "ascent"),
            )
        )
        pairs = train_module.canonical_transition_pairs(rows)
        self.assertEqual(
            [
                (pair.sequence_lane, pair.predecessor_center_frame, pair.center_frame)
                for pair in pairs
            ],
            [("idle_phase_0", 1, 2), ("motion", 1, 2)],
        )

    def test_transition_pairs_reject_duplicate_logical_rows(self) -> None:
        rows = _TransitionPairRows()
        rows.rows.append(dict(rows.rows[2]))
        with self.assertRaisesRegex(ValueError, "duplicate transition-pair"):
            train_module.canonical_transition_pairs(rows)

    def test_transition_pair_order_and_receipt_ignore_storage_order(self) -> None:
        rows = _TransitionPairRows()
        reverse = _TransitionPairRows(reversed(_TransitionPairRows._LOGICAL_ROWS))
        ordered_pairs = train_module.canonical_transition_pairs(rows)
        reversed_pairs = train_module.canonical_transition_pairs(reverse)
        self.assertEqual(
            [
                (
                    pair.clip_id,
                    pair.sequence_lane,
                    pair.predecessor_center_frame,
                    pair.center_frame,
                    pair.predecessor_row_sha256,
                    pair.current_row_sha256,
                )
                for pair in ordered_pairs
            ],
            [
                (
                    pair.clip_id,
                    pair.sequence_lane,
                    pair.predecessor_center_frame,
                    pair.center_frame,
                    pair.predecessor_row_sha256,
                    pair.current_row_sha256,
                )
                for pair in reversed_pairs
            ],
        )

    def test_transition_pair_materialization_is_immutable_reads_once_and_excludes_run_starts(
        self,
    ) -> None:
        rows = _TransitionPairRows()
        pairs = train_module.materialize_transition_pairs(rows)
        self.assertEqual(rows.calls, list(range(len(rows))))
        self.assertEqual(len(pairs), 4)
        pair_classes = [
            str(pairs[index]["current"]["terrain_class"])
            for index in range(len(pairs))
        ]
        self.assertEqual(
            pair_classes,
            ["transition", "ascent", "flat", "descent"],
        )
        self.assertEqual(
            pair_classes,
            [pairs[index]["pair"].terrain_class for index in range(len(pairs))],
        )
        sampled = train_module._balanced_epoch_indices(
            list(range(len(pairs))), pair_classes, seed=7, epoch=0
        )
        self.assertEqual(set(sampled), set(range(len(pairs))))
        current_keys = {
            (
                pairs[index]["pair"].clip_id,
                pairs[index]["pair"].sequence_lane,
                pairs[index]["pair"].center_frame,
            )
            for index in sampled
        }
        self.assertTrue(
            {
                ("clip_a", "idle_phase_0", 1),
                ("clip_a", "motion", 1),
                ("clip_b", "motion", 10),
            }.isdisjoint(current_keys)
        )
        predecessor_before = np.asarray(pairs[0]["predecessor_y"]).copy()
        rows.rows[6]["y"][0] = 999.0
        np.testing.assert_array_equal(
            pairs[0]["predecessor_y"], predecessor_before
        )
        with self.assertRaises(TypeError):
            pairs[0]["current"] = {}
        with self.assertRaises(ValueError):
            pairs[0]["predecessor_y"][0] = 1.0
        batch = train_module._predecessor_batch(
            pairs, tuple(range(len(pairs))), torch.device("cpu")
        )
        self.assertEqual(
            [tuple(value.shape) for value in batch],
            [(4, INPUT_LAYOUT.size), (4,), (4, OUTPUT_LAYOUT.size), (4, OUTPUT_LAYOUT.size)],
        )

    def test_fitted_transition_report_canonicalizes_adjacent_pair_order(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        output[OUTPUT_LAYOUT["joint_position"]][0] = 0.251
        rows = _FittedEnvelopeRows(output)
        rows.rows.append(rows._row(
            12, rows._physical_output(), phase=1.25
        ))
        arguments = {
            "normalization": rows,
            "joint_limits": np.asarray([[-2.0, 2.0]] * 29, np.float64),
            "phase_advance_q99": 0.2,
        }
        ordered = training_module.evaluate_fitted_transition_envelope(
            _FittedEnvelopeModel(output), rows, ((0, 1), (1, 2)), **arguments
        )
        reversed_pairs = training_module.evaluate_fitted_transition_envelope(
            _FittedEnvelopeModel(output), rows, ((1, 2), (0, 1)), **arguments
        )
        self.assertEqual(ordered.to_dict(), reversed_pairs.to_dict())
        self.assertEqual(ordered.first_failure["center_frame"], 11)

    def test_fitted_transition_rejects_duplicate_logical_pairs(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        rows = _FittedEnvelopeRows(output)
        rows.rows.extend((dict(rows.rows[0]), dict(rows.rows[1])))
        arguments = {
            "normalization": rows,
            "joint_limits": np.asarray([[-2.0, 2.0]] * 29, np.float64),
            "phase_advance_q99": 0.2,
        }
        with self.assertRaisesRegex(ValueError, "duplicate logical"):
            training_module.evaluate_fitted_transition_envelope(
                _FittedEnvelopeModel(output),
                rows,
                ((2, 3), (0, 1)),
                **arguments,
            )

        rows.rows.append(rows._row(12, rows._physical_output(), phase=1.25))
        ordered = training_module.evaluate_fitted_transition_envelope(
            _FittedEnvelopeModel(output), rows, ((0, 1), (1, 4)), **arguments
        )
        permuted = training_module.evaluate_fitted_transition_envelope(
            _FittedEnvelopeModel(output), rows, ((1, 4), (0, 1)), **arguments
        )
        self.assertEqual(ordered.rows_sha256, permuted.rows_sha256)
        self.assertEqual(ordered.report_sha256, permuted.report_sha256)

    def test_fitted_transition_overflow_is_canonical_finite_rejection(self) -> None:
        maximum = float(np.finfo(np.float64).max)
        cases: list[tuple[str, np.ndarray, str]] = []
        planar = _FittedEnvelopeRows._physical_output()
        planar[OUTPUT_LAYOUT["root_planar_velocity"]] = (maximum, maximum)
        cases.append(("planar velocity", planar, "root_translation_step_m"))
        direction = _FittedEnvelopeRows._physical_output()
        direction[OUTPUT_LAYOUT["trajectory_direction"]][:2] = (maximum, maximum)
        cases.append(("trajectory direction", direction, "trajectory_direction_norm"))

        for label, output, field in cases:
            with self.subTest(field=label):
                report, _, _ = self._fitted_envelope(output)
                self.assertFalse(report.accepted)
                self.assertEqual(report.first_failure["field"], field)
                self.assertTrue(math.isfinite(float(report.first_failure["value"])))
                self.assertTrue(all(math.isfinite(value) for value in report.maxima.values()))
                self.assertEqual(
                    training_module.validate_fitted_transition_report(report)[
                        "report_sha256"
                    ],
                    report.report_sha256,
                )

        joint = _FittedEnvelopeRows._physical_output()
        joint[OUTPUT_LAYOUT["joint_position"]][0] = 3.0e38
        rows = _FittedEnvelopeRows(joint)
        rows.rows[0]["y"][OUTPUT_LAYOUT["joint_position"]][0] = np.float32(-3.0e38)
        rows.y_std = np.ones(OUTPUT_LAYOUT.size, np.float64)
        rows.y_std[OUTPUT_LAYOUT["joint_position"]] = 3.0e269
        limits = np.asarray([[-maximum, maximum]] * 29, np.float64)
        report = training_module.evaluate_fitted_transition_envelope(
            _FittedEnvelopeModel(joint),
            rows,
            ((0, 1),),
            normalization=rows,
            joint_limits=limits,
            phase_advance_q99=0.2,
        )
        self.assertFalse(report.accepted)
        self.assertEqual(report.first_failure["field"], "joint_position")
        self.assertEqual(report.first_failure["value"], maximum)
        self.assertTrue(all(math.isfinite(value) for value in report.maxima.values()))
        training_module.validate_fitted_transition_report(report)

    def test_reloaded_transition_gate_precedes_and_can_block_pipeline_verifier(self) -> None:
        output = _FittedEnvelopeRows._physical_output()
        rejected_output = output.copy()
        rejected_output[OUTPUT_LAYOUT["joint_position"]][0] = 0.251
        rejected_report = self._fitted_envelope(rejected_output)[0]
        accepted_report = self._fitted_envelope(output)[0]
        rows = _FittedEnvelopeRows(output)
        fitted_subset = fitted_subset_metadata(rows, (0, 1))
        events: list[str] = []
        ordinary_model = torch.nn.Linear(1, 1, device="meta")

        class ReloadedTransitionModel(_FittedEnvelopeModel):
            def to(self, *args: object, **kwargs: object) -> torch.nn.Module:
                device = torch.device(args[0] if args else kwargs["device"])
                events.append(f"place_reloaded_model_{device.type}")
                return super().to(*args, **kwargs)

        class Loaded:
            normalization = {
                "x_mean": torch.zeros(INPUT_LAYOUT.size),
                "x_std": torch.ones(INPUT_LAYOUT.size),
                "y_mean": torch.zeros(OUTPUT_LAYOUT.size),
                "y_std": torch.ones(OUTPUT_LAYOUT.size),
            }
            joint_limits = torch.tensor([[-2.0, 2.0]] * 29, dtype=torch.float64)
            phase_advance_q99 = 0.2

            def build_model(self) -> torch.nn.Module:
                events.append("build_reloaded_model")
                return ReloadedTransitionModel(output)

        def loaded(*args: object, **kwargs: object) -> Loaded:
            events.append("reload_checkpoint")
            return Loaded()

        def validate(*args: object, **kwargs: object) -> None:
            events.append("validate_exact_subset")

        def evaluated(*args: object, **kwargs: object) -> object:
            events.append("evaluate_transitions")
            transition_model = args[0]
            self.assertEqual(ordinary_model.weight.device.type, "meta")
            self.assertEqual(next(transition_model.buffers()).device.type, "cpu")
            self.assertTrue(torch.equal(
                next(transition_model.buffers()), torch.as_tensor(output)
            ))
            return rejected_report

        def verifier(*args: object, **kwargs: object) -> dict[str, object]:
            events.append("open_terrain_verifier")
            return {"accepted": True}

        with (
            patch.object(train_module, "load_checkpoint", side_effect=loaded),
            patch.object(
                train_module, "validate_resume_fitted_subset", side_effect=validate
            ),
            patch.object(
                train_module,
                "evaluate_fitted_transition_envelope",
                side_effect=evaluated,
            ),
        ):
            report, receipt = train_module._verify_reloaded_pipeline_candidate(
                candidate_path=Path("candidate.pt"),
                dataset=rows,
                fitted_indices=(0, 1),
                fitted_subset=fitted_subset,
                dataset_digest_sha256="a" * 64,
                kinematic_signature_sha256="b" * 64,
                joint_limits=np.asarray([[-2.0, 2.0]] * 29, np.float64),
                phase_advance_q99=0.2,
                verification_request={"binding": "exact"},
                pipeline_verifier=verifier,
            )
        self.assertEqual(report, rejected_report)
        self.assertIsNone(receipt)
        self.assertEqual(events, [
            "reload_checkpoint", "validate_exact_subset", "build_reloaded_model",
            "place_reloaded_model_cpu", "evaluate_transitions",
        ])

        requests: list[dict[str, object]] = []
        with (
            patch.object(train_module, "load_checkpoint", return_value=Loaded()),
            patch.object(train_module, "validate_resume_fitted_subset"),
            patch.object(
                train_module,
                "evaluate_fitted_transition_envelope",
                return_value=accepted_report,
            ),
        ):
            report, receipt = train_module._verify_reloaded_pipeline_candidate(
                candidate_path=Path("candidate.pt"),
                dataset=rows,
                fitted_indices=(0, 1),
                fitted_subset=fitted_subset,
                dataset_digest_sha256="a" * 64,
                kinematic_signature_sha256="b" * 64,
                joint_limits=np.asarray([[-2.0, 2.0]] * 29, np.float64),
                phase_advance_q99=0.2,
                verification_request={"binding": "exact"},
                pipeline_verifier=lambda _path, request: (
                    requests.append(request) or {"accepted": True}
                ),
            )
        self.assertEqual(report, accepted_report)
        self.assertEqual(receipt, {"accepted": True})
        self.assertEqual(requests[0]["binding"], "exact")
        self.assertEqual(
            requests[0]["fitted_transition_report"], accepted_report.to_dict()
        )
        self.assertEqual(
            requests[0]["fitted_transition_report_sha256"],
            accepted_report.report_sha256,
        )

    def test_three_step_rollout_matches_shared_runtime_recurrence_and_gradient(self) -> None:
        dtype = torch.float64
        normal = {
            "x_mean": np.linspace(-0.4, 0.6, INPUT_LAYOUT.size, dtype=np.float64),
            "x_std": np.linspace(0.7, 1.3, INPUT_LAYOUT.size, dtype=np.float64),
            "y_mean": np.linspace(-0.3, 0.5, OUTPUT_LAYOUT.size, dtype=np.float64),
            "y_std": np.linspace(0.8, 1.4, OUTPUT_LAYOUT.size, dtype=np.float64),
        }
        normal["y_mean"][OUTPUT_LAYOUT["contact_logit"]] = 0.0
        normal["y_std"][OUTPUT_LAYOUT["contact_logit"]] = 1.0
        x_mean = torch.as_tensor(normal["x_mean"], dtype=dtype)
        x_std = torch.as_tensor(normal["x_std"], dtype=dtype)
        y_mean = torch.as_tensor(normal["y_mean"], dtype=dtype)
        y_std = torch.as_tensor(normal["y_std"], dtype=dtype)

        raw_inputs = torch.zeros((3, 1, INPUT_LAYOUT.size), dtype=dtype)
        times = torch.as_tensor(TRAJECTORY_TIMES_S, dtype=dtype)
        for step in range(3):
            trajectory = raw_inputs[
                step, :, INPUT_LAYOUT["trajectory_position"]
            ].reshape(1, 12, 2)
            trajectory[..., 0] = (0.18 + 0.04 * step) * times
            trajectory[..., 1] = 0.01 * step * times.square()
            direction = raw_inputs[
                step, :, INPUT_LAYOUT["trajectory_direction"]
            ].reshape(1, 12, 2)
            direction[..., 0] = 1.0
            terrain = raw_inputs[
                step, :, INPUT_LAYOUT["terrain_height"]
            ].reshape(1, 12, 3)
            terrain[..., 0] = 0.02 * step
            terrain[..., 1] = torch.linspace(-0.03, 0.04, 12, dtype=dtype)
            semantic = raw_inputs[
                step, :, INPUT_LAYOUT["semantic_intent"]
            ].reshape(1, 12, 2)
            semantic[..., 1] = 1.0
            raw_inputs[
                step, :, INPUT_LAYOUT["previous_body_position"]
            ] = 0.03 * (step + 1)
            raw_inputs[
                step, :, INPUT_LAYOUT["previous_body_velocity"]
            ] = -0.02 * (step + 1)
        normalized_inputs = (raw_inputs - x_mean) / x_std
        for field in ("previous_body_position", "previous_body_velocity"):
            normalized_inputs[..., INPUT_LAYOUT[field]] *= 0.1

        class DeterministicPFNN(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for name, value in zip(
                    ("W0", "b0", "W1", "b1", "W2", "b2"),
                    (0.07, 0.01, -0.02, 0.03, -0.01, 0.02),
                ):
                    setattr(
                        self,
                        name,
                        torch.nn.Parameter(torch.tensor((value,), dtype=dtype)),
                    )
                base = torch.zeros(OUTPUT_LAYOUT.size, dtype=dtype)
                base[OUTPUT_LAYOUT["trajectory_direction"]] = torch.tensor(
                    (1.0, 0.0), dtype=dtype
                ).repeat(12)
                base[OUTPUT_LAYOUT["body_position"]] = 0.12
                base[OUTPUT_LAYOUT["body_velocity"]] = -0.04
                base[OUTPUT_LAYOUT["root_height"]] = 0.72
                base[OUTPUT_LAYOUT["root_planar_velocity"]] = torch.tensor(
                    (0.24, 0.03), dtype=dtype
                )
                base[OUTPUT_LAYOUT["root_yaw_velocity"]] = 0.08
                base[OUTPUT_LAYOUT["phase_advance"]] = 0.11
                self.register_buffer("physical_base", base)

            def forward(
                self, normalized: torch.Tensor, phase: torch.Tensor
            ) -> torch.Tensor:
                body_start = INPUT_LAYOUT["previous_body_position"].start
                signal = (
                    self.W0 * normalized[:, body_start : body_start + 1]
                    + self.b0 * phase[:, None]
                )
                physical = self.physical_base.expand(len(normalized), -1).clone()
                physical[:, OUTPUT_LAYOUT["trajectory_position"]] += signal
                physical[:, OUTPUT_LAYOUT["body_position"]] += signal
                physical[:, OUTPUT_LAYOUT["body_velocity"]] -= 0.5 * signal
                physical[:, OUTPUT_LAYOUT["root_planar_velocity"]] += signal
                physical[:, OUTPUT_LAYOUT["root_yaw_velocity"]] += 0.25 * signal
                physical[:, OUTPUT_LAYOUT["phase_advance"]] += 0.1 * signal
                return (physical - y_mean) / y_std

        model = DeterministicPFNN()
        targets = torch.zeros((3, 1, OUTPUT_LAYOUT.size), dtype=dtype)
        targets[..., OUTPUT_LAYOUT["contact_logit"]] = 1.0
        initial_phase = torch.tensor((0.35,), dtype=dtype)
        phase_cap = min(np.pi, 1.5 * 0.24)
        result = autoregressive_unroll(
            model,
            normalized_inputs,
            initial_phase,
            targets,
            normalization=normal,
            phase_advance_cap=phase_cap,
        )

        first_raw = normalized_inputs[0].clone()
        for field in ("previous_body_position", "previous_body_velocity"):
            first_raw[:, INPUT_LAYOUT[field]] /= 0.1
        first_raw = first_raw * x_std + x_mean
        first_semantic = first_raw[:, INPUT_LAYOUT["semantic_intent"]].reshape(
            1, 12, 2
        )
        first_raw[:, INPUT_LAYOUT["semantic_intent"]] = torch.nn.functional.one_hot(
            torch.argmax(first_semantic, dim=-1), num_classes=2
        ).to(dtype).reshape(1, -1)
        state = initialize_recurrent_state(
            trajectory_position_local=first_raw[
                :, INPUT_LAYOUT["trajectory_position"]
            ].reshape(1, 12, 2),
            trajectory_direction_local=first_raw[
                :, INPUT_LAYOUT["trajectory_direction"]
            ].reshape(1, 12, 2),
            semantic_intent=first_raw[
                :, INPUT_LAYOUT["semantic_intent"]
            ].reshape(1, 12, 2),
            previous_body_position_local=first_raw[
                :, INPUT_LAYOUT["previous_body_position"]
            ].reshape(1, 30, 3),
            previous_body_velocity_local=first_raw[
                :, INPUT_LAYOUT["previous_body_velocity"]
            ].reshape(1, 30, 3),
            phase=initial_phase,
            root_world_xy=torch.zeros((1, 2), dtype=dtype),
            root_yaw_world=torch.zeros(1, dtype=dtype),
        )
        planned = PlannedTrajectory(
            position_world_xy=state.predicted_position_world_xy,
            direction_world_xy=state.predicted_direction_world_xy,
            semantic_intent=first_raw[
                :, INPUT_LAYOUT["semantic_intent"]
            ].reshape(1, 12, 2),
        )
        direct_inputs = [normalized_inputs[0]]
        direct_phases = [initial_phase]
        cap = torch.full((1,), phase_cap, dtype=dtype)
        for step, prediction in enumerate(result.predictions[:-1]):
            physical_output = prediction * y_std + y_mean
            state = advance_recurrent_state(
                state,
                planned,
                physical_output,
                phase_advance_cap=cap,
            )
            next_raw = normalized_inputs[step + 1].clone()
            for field in ("previous_body_position", "previous_body_velocity"):
                next_raw[:, INPUT_LAYOUT[field]] /= 0.1
            next_raw = next_raw * x_std + x_mean
            next_semantic = next_raw[:, INPUT_LAYOUT["semantic_intent"]].reshape(
                1, 12, 2
            )
            next_raw[:, INPUT_LAYOUT["semantic_intent"]] = (
                torch.nn.functional.one_hot(
                    torch.argmax(next_semantic, dim=-1), num_classes=2
                ).to(dtype).reshape(1, -1)
            )
            desired_velocity = derive_training_desired_velocity(
                next_raw[:, INPUT_LAYOUT["trajectory_position"]].reshape(1, 12, 2),
                next_raw[:, INPUT_LAYOUT["semantic_intent"]].reshape(1, 12, 2),
                state.root_yaw_world,
            )
            planned = plan_recurrent_trajectory(state, desired_velocity)
            direct_inputs.append(
                pack_recurrent_input(
                    state=state,
                    planned=planned,
                    terrain_height=next_raw[
                        :, INPUT_LAYOUT["terrain_height"]
                    ].reshape(1, 12, 3),
                    x_mean=x_mean,
                    x_std=x_std,
                    body_scale=0.1,
                )
            )
            direct_phases.append(state.phase)

        for step, (trained, direct) in enumerate(zip(result.inputs, direct_inputs)):
            for field in (
                "trajectory_position",
                "trajectory_direction",
                "terrain_height",
                "semantic_intent",
                "previous_body_position",
                "previous_body_velocity",
            ):
                with self.subTest(step=step, field=field):
                    torch.testing.assert_close(
                        trained[:, INPUT_LAYOUT[field]],
                        direct[:, INPUT_LAYOUT[field]],
                        atol=3.0e-6,
                        rtol=0.0,
                    )
            torch.testing.assert_close(
                result.phases[step], direct_phases[step], atol=3.0e-6, rtol=0.0
            )
        first_prediction = result.predictions[0]
        first_prediction.retain_grad()
        tick_three_loss = pfnn_losses(
            result.predictions[2],
            targets[2],
            model=model,
            normalization=normal,
        )["total"]
        tick_three_loss.backward()
        self.assertIsNotNone(first_prediction.grad)
        self.assertTrue(torch.isfinite(first_prediction.grad).all())
        self.assertGreater(float(first_prediction.grad.abs().sum()), 0.0)

    def test_overfit_subset_is_consecutive_source_sealed_and_mixed_terrain(self) -> None:
        class Rows:
            split = "train"

            def __init__(self) -> None:
                self.rows: list[dict[str, object]] = []
                for clip, centers in (
                    ("terrain_slopes__slope_000__001", range(20, 24)),
                    ("terrain_slopes__slope_000__000", range(10, 15)),
                    ("terrain_slopes__slope_001__000", range(30, 36)),
                ):
                    for center in centers:
                        self.rows.append({
                            "x": np.full(INPUT_LAYOUT.size, center, np.float32),
                            "y": np.full(OUTPUT_LAYOUT.size, center, np.float32),
                            "phase": np.float32(center * 0.01),
                            "clip_id": clip,
                            "center_frame": center,
                            "split_identity": clip.split("__")[1],
                            "split": "train",
                            "sequence_lane": "motion",
                            "terrain_class": (
                                "flat" if center in (10, 11) else "ascent"
                            ),
                        })
                # Duplicate augmentation at center 12 must remove that center and
                # split the otherwise consecutive source run into two runs.
                duplicate = next(
                    row for row in self.rows
                    if row["clip_id"].endswith("__000")
                    and row["center_frame"] == 12
                )
                self.rows.append(dict(duplicate))

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        rows = Rows()
        selected = consecutive_overfit_subset(rows, 6)
        keys = [
            (rows[index]["clip_id"], rows[index]["center_frame"])
            for index in selected
        ]
        self.assertEqual(keys, [
            ("terrain_slopes__slope_000__000", 10),
            ("terrain_slopes__slope_000__000", 11),
            ("terrain_slopes__slope_000__000", 13),
            ("terrain_slopes__slope_000__000", 14),
            ("terrain_slopes__slope_000__001", 20),
            ("terrain_slopes__slope_000__001", 21),
        ])
        metadata = fitted_subset_metadata(rows, selected)
        self.assertEqual(metadata["split"], "train")
        self.assertEqual(metadata["class_counts"], {
            "flat": 2, "ascent": 4, "descent": 0, "transition": 0,
        })
        self.assertEqual(len(metadata["rows"]), 6)
        self.assertEqual(len(metadata["rows_sha256"]), 64)
        self.assertTrue(all(len(row["row_sha256"]) == 64 for row in metadata["rows"]))

    def test_pipeline_subset_contains_complete_lowest_speed_seed_run(self) -> None:
        class Rows:
            split = "train"

            def __init__(self) -> None:
                self.rows: list[dict[str, object]] = []
                for clip, identity, centers, terrain_class in (
                    ("walk1_subject2", "walk1_subject2", range(100, 108), "flat"),
                    ("terrain_slopes__slope_001__000", "slope_001", range(20, 30), "ascent"),
                    ("terrain_slopes__slope_001__001", "slope_001", range(40, 50), "descent"),
                ):
                    for center in centers:
                        lanes = (
                            ("motion", "idle_phase_0")
                            if clip == "walk1_subject2"
                            else ("motion",)
                        )
                        for lane in lanes:
                            self.rows.append({
                                "x": np.full(INPUT_LAYOUT.size, center, np.float32),
                                "y": np.full(OUTPUT_LAYOUT.size, center, np.float32),
                                "phase": np.float32(center * 0.01),
                                "clip_id": clip,
                                "center_frame": center,
                                "split_identity": identity,
                                "split": "train",
                                "sequence_lane": lane,
                                "terrain_class": terrain_class,
                            })

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        rows = Rows()
        selected = consecutive_overfit_subset(
            rows,
            18,
            required_seed_key=("walk1_subject2", "idle_phase_0", 103),
            known_terrain_identity="slope_001",
        )
        keys = [
            (
                rows[index]["clip_id"],
                rows[index]["sequence_lane"],
                rows[index]["center_frame"],
            )
            for index in selected
        ]
        self.assertEqual(
            keys[:8],
            [
                ("walk1_subject2", "idle_phase_0", center)
                for center in range(100, 108)
            ],
        )
        self.assertEqual(
            keys[8:],
            [
                ("terrain_slopes__slope_001__000", "motion", center)
                for center in range(20, 30)
            ],
        )

    def test_runtime_seed_predecessor_and_first_input_are_inside_fitted_run(self) -> None:
        class Rows:
            split = "train"
            x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
            x_std = np.ones(INPUT_LAYOUT.size, np.float32)
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            def __init__(self, start: int = 10) -> None:
                self.rows = [self._row(center) for center in range(start, 31)]

            @staticmethod
            def _state(center: int) -> tuple[np.ndarray, np.ndarray]:
                trajectory = np.zeros((12, 2), np.float32)
                trajectory[:, 0] = np.linspace(-0.2, 0.8, 12) + center * 0.001
                direction = np.zeros((12, 2), np.float32)
                direction[:, 0] = 1.0
                body = np.full((30, 3), center * 0.001, np.float32)
                return np.concatenate((trajectory.ravel(), direction.ravel())), body

            def _row(self, center: int) -> dict[str, object]:
                current_trajectory, current_body = self._state(center)
                target_trajectory, target_body = self._state(center + 1)
                x = np.zeros(INPUT_LAYOUT.size, np.float32)
                x[INPUT_LAYOUT["trajectory_position"]] = current_trajectory[:24]
                x[INPUT_LAYOUT["trajectory_direction"]] = current_trajectory[24:]
                x[INPUT_LAYOUT["previous_body_position"]] = current_body.ravel() * 0.1
                x[INPUT_LAYOUT["previous_body_velocity"]] = current_body.ravel() * 0.1
                x[INPUT_LAYOUT["semantic_intent"]] = np.tile(
                    (1.0, 0.0), (12, 1)
                ).ravel()
                y = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                y[OUTPUT_LAYOUT["trajectory_position"]] = target_trajectory[:24]
                y[OUTPUT_LAYOUT["trajectory_direction"]] = target_trajectory[24:]
                y[OUTPUT_LAYOUT["body_position"]] = target_body.ravel()
                y[OUTPUT_LAYOUT["body_velocity"]] = target_body.ravel()
                y[OUTPUT_LAYOUT["root_height"]] = 0.8
                y[OUTPUT_LAYOUT["root_planar_velocity"]] = (center * 0.001, 0.0)
                y[OUTPUT_LAYOUT["phase_advance"]] = 0.1
                y[OUTPUT_LAYOUT["contact_logit"]] = (1.0, 0.0, 1.0, 0.0)
                return {
                    "x": x,
                    "y": y,
                    "phase": np.float32(center * 0.1),
                    "clip_id": "terrain_slopes__slope_000__000",
                    "center_frame": center,
                    "split_identity": "slope_000",
                    "split": "train",
                    "sequence_lane": "motion",
                    "terrain_class": "flat",
                }

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        rows = Rows()
        seed = choose_runtime_seed(rows, torch.tensor([[-2.0, 2.0]] * 29))
        self.assertEqual(
            {
                name: seed["provenance"][name]
                for name in (
                    "predecessor_clip_id", "predecessor_center_frame",
                    "predecessor_sequence_lane", "first_fitted_clip_id",
                    "first_fitted_center_frame", "first_fitted_sequence_lane",
                )
            },
            {
            "predecessor_clip_id": "terrain_slopes__slope_000__000",
            "predecessor_center_frame": 10,
            "predecessor_sequence_lane": "motion",
            "first_fitted_clip_id": "terrain_slopes__slope_000__000",
            "first_fitted_center_frame": 11,
            "first_fitted_sequence_lane": "motion",
            },
        )
        self.assertAlmostEqual(seed["provenance"]["speed"], 0.011)
        torch.testing.assert_close(seed["phase"], torch.tensor(1.1))
        torch.testing.assert_close(
            seed["body_position"], torch.full((30, 3), 0.011)
        )
        torch.testing.assert_close(
            seed["semantic_intent"], torch.tensor([[1.0, 0.0]] * 12)
        )
        torch.testing.assert_close(seed["terrain_height"], torch.zeros(12, 3))
        self.assertEqual(
            seed["normalized_input_sha256"],
            pfnn_input_sha256(rows[1]["x"]),
        )
        isolated = Rows()
        isolated.rows = isolated.rows[:1]
        with self.assertRaisesRegex(ValueError, "fitted consecutive"):
            choose_runtime_seed(isolated, torch.tensor([[-2.0, 2.0]] * 29))

    def test_runtime_seed_keys_stationary_rows_by_sequence_lane(self) -> None:
        class Rows:
            split = "train"
            x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
            x_std = np.ones(INPUT_LAYOUT.size, np.float32)
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            @staticmethod
            def _state(center: int) -> tuple[np.ndarray, np.ndarray]:
                trajectory = np.zeros((12, 2), np.float32)
                trajectory[:, 0] = center * 0.001
                direction = np.zeros((12, 2), np.float32)
                direction[:, 0] = 1.0
                body = np.full((30, 3), center * 0.001, np.float32)
                return np.concatenate((trajectory.ravel(), direction.ravel())), body

            @classmethod
            def _row(
                cls, center: int, lane: str, speed: float
            ) -> dict[str, object]:
                current_trajectory, current_body = cls._state(center)
                target_trajectory, target_body = cls._state(center + 1)
                x = np.zeros(INPUT_LAYOUT.size, np.float32)
                x[INPUT_LAYOUT["trajectory_position"]] = current_trajectory[:24]
                x[INPUT_LAYOUT["trajectory_direction"]] = current_trajectory[24:]
                x[INPUT_LAYOUT["previous_body_position"]] = current_body.ravel() * 0.1
                x[INPUT_LAYOUT["previous_body_velocity"]] = current_body.ravel() * 0.1
                x[INPUT_LAYOUT["semantic_intent"]] = np.tile(
                    (1.0, 0.0), (12, 1)
                ).ravel()
                y = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                y[OUTPUT_LAYOUT["trajectory_position"]] = target_trajectory[:24]
                y[OUTPUT_LAYOUT["trajectory_direction"]] = target_trajectory[24:]
                y[OUTPUT_LAYOUT["body_position"]] = target_body.ravel()
                y[OUTPUT_LAYOUT["body_velocity"]] = target_body.ravel()
                y[OUTPUT_LAYOUT["root_height"]] = 0.8
                y[OUTPUT_LAYOUT["root_planar_velocity"]] = (speed, 0.0)
                y[OUTPUT_LAYOUT["phase_advance"]] = 0.1
                y[OUTPUT_LAYOUT["contact_logit"]] = (1.0, 0.0, 1.0, 0.0)
                return {
                    "x": x,
                    "y": y,
                    "phase": np.float32(center * 0.1),
                    "clip_id": "terrain_slopes__slope_000__000",
                    "center_frame": center,
                    "split_identity": "slope_000",
                    "split": "train",
                    "sequence_lane": lane,
                    "terrain_class": "flat",
                }

            def __init__(self, duplicate_lane: str | None = None) -> None:
                self.rows: list[dict[str, object]] = []
                for center in range(10, 27):
                    for lane_index in reversed(range(8)):
                        lane = f"idle_phase_{lane_index}"
                        speed = 0.01 * max(lane_index, 1) if center == 11 else 0.5
                        self.rows.append(self._row(center, lane, speed))
                # This row has the fastest target, but only an idle-lane row at
                # center 11. It must never borrow that wrong-lane predecessor.
                self.rows.append(self._row(12, "motion", 0.0001))
                if duplicate_lane is not None:
                    duplicate = next(
                        row
                        for row in self.rows
                        if row["center_frame"] == 11
                        and row["sequence_lane"] == duplicate_lane
                    )
                    self.rows.append(dict(duplicate))

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        seed = choose_runtime_seed(Rows(), torch.tensor([[-2.0, 2.0]] * 29))
        self.assertEqual(seed["provenance"]["predecessor_center_frame"], 10)
        self.assertEqual(seed["provenance"]["first_fitted_center_frame"], 11)
        self.assertEqual(
            seed["provenance"]["predecessor_sequence_lane"], "idle_phase_0"
        )
        self.assertEqual(
            seed["provenance"]["first_fitted_sequence_lane"], "idle_phase_0"
        )
        self.assertAlmostEqual(seed["provenance"]["speed"], 0.01)

        duplicate = choose_runtime_seed(
            Rows(duplicate_lane="idle_phase_0"),
            torch.tensor([[-2.0, 2.0]] * 29),
        )
        self.assertEqual(
            duplicate["provenance"]["predecessor_sequence_lane"], "idle_phase_1"
        )
        self.assertEqual(
            duplicate["provenance"]["first_fitted_sequence_lane"], "idle_phase_1"
        )
        self.assertAlmostEqual(duplicate["provenance"]["speed"], 0.01)

    def test_runtime_seed_filters_sixteen_frame_horizon_before_speed_and_is_sampled(self) -> None:
        class Rows:
            split = "train"
            x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
            x_std = np.ones(INPUT_LAYOUT.size, np.float32)
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            @staticmethod
            def state(center: int) -> tuple[np.ndarray, np.ndarray]:
                trajectory = np.zeros((12, 2), np.float32)
                trajectory[:, 0] = center * 0.001
                direction = np.zeros((12, 2), np.float32)
                direction[:, 0] = 1.0
                body = np.full((30, 3), center * 0.001, np.float32)
                return np.concatenate((trajectory.ravel(), direction.ravel())), body

            @classmethod
            def row(cls, clip: str, center: int, speed: float) -> dict[str, object]:
                current_trajectory, current_body = cls.state(center)
                target_trajectory, target_body = cls.state(center + 1)
                x = np.zeros(INPUT_LAYOUT.size, np.float32)
                x[INPUT_LAYOUT["trajectory_position"]] = current_trajectory[:24]
                x[INPUT_LAYOUT["trajectory_direction"]] = current_trajectory[24:]
                x[INPUT_LAYOUT["previous_body_position"]] = current_body.ravel() * 0.1
                x[INPUT_LAYOUT["previous_body_velocity"]] = current_body.ravel() * 0.1
                x[INPUT_LAYOUT["semantic_intent"]] = np.tile((1.0, 0.0), (12, 1)).ravel()
                y = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                y[OUTPUT_LAYOUT["trajectory_position"]] = target_trajectory[:24]
                y[OUTPUT_LAYOUT["trajectory_direction"]] = target_trajectory[24:]
                y[OUTPUT_LAYOUT["body_position"]] = target_body.ravel()
                y[OUTPUT_LAYOUT["body_velocity"]] = target_body.ravel()
                y[OUTPUT_LAYOUT["root_height"]] = 0.8
                y[OUTPUT_LAYOUT["root_planar_velocity"]] = (speed, 0.0)
                y[OUTPUT_LAYOUT["phase_advance"]] = 0.1
                y[OUTPUT_LAYOUT["contact_logit"]] = (1.0, 0.0, 1.0, 0.0)
                return {
                    "x": x, "y": y, "phase": np.float32(center * 0.1),
                    "clip_id": clip, "center_frame": center,
                    "split_identity": clip, "split": "train",
                    "sequence_lane": "motion", "terrain_class": "flat",
                }

            def __init__(self) -> None:
                self.rows = [
                    self.row("walk1_subject1", center, 0.4)
                    for center in range(10, 30)
                ]
                self.rows.extend(
                    (
                        self.row("walk4_subject1", 49, 0.4),
                        self.row("walk4_subject1", 50, 0.001),
                    )
                )

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        seed = choose_runtime_seed(Rows(), torch.tensor([[-2.0, 2.0]] * 29))
        self.assertEqual(
            seed["provenance"]["first_fitted_clip_id"], "walk1_subject1"
        )
        self.assertEqual(seed["provenance"]["first_fitted_center_frame"], 11)
        self.assertAlmostEqual(seed["provenance"]["speed"], 0.4)

        rows = Rows()
        sequences, _ = train_module._consecutive_starts(rows, 16)
        chosen_key = (
            seed["provenance"]["first_fitted_clip_id"],
            seed["provenance"]["first_fitted_sequence_lane"],
            seed["provenance"]["first_fitted_center_frame"],
        )
        chosen_sequence = next(
            sequence_index
            for sequence_index, sequence in enumerate(sequences)
            if (
                rows[sequence[0]]["clip_id"],
                rows[sequence[0]]["sequence_lane"],
                rows[sequence[0]]["center_frame"],
            ) == chosen_key
        )
        sampler = train_module.DeterministicSequenceSampler(
            len(sequences), batch_size=3, seed=109
        )
        first_pass = [
            index
            for _ in range((len(sequences) + 2) // 3)
            for index in sampler.next_batch()
        ][: len(sequences)]
        self.assertIn(chosen_sequence, first_pass)

    def test_seed_receipt_membership_adjacency_and_resume_subset_fail_closed(self) -> None:
        rows = [
            {
                "x": np.full(INPUT_LAYOUT.size, center, np.float32),
                "y": np.full(OUTPUT_LAYOUT.size, center + 1, np.float32),
                "phase": np.float32(center * 0.1),
                "clip_id": "clip", "center_frame": center,
                "split_identity": "clip", "split": "train",
                "sequence_lane": "motion", "terrain_class": "flat",
            }
            for center in (10, 11, 12)
        ]

        class Dataset:
            split = "train"
            def __len__(self) -> int: return len(rows)
            def __getitem__(self, index: int) -> dict[str, object]: return rows[index]

        receipt = fitted_subset_metadata(Dataset(), range(3))
        seed = finite_runtime_seed()
        seed["provenance"].update({
            "predecessor_clip_id": "clip",
            "predecessor_center_frame": 10,
            "first_fitted_clip_id": "clip",
            "first_fitted_center_frame": 11,
            "predecessor_row_sha256": receipt["rows"][0]["row_sha256"],
            "first_fitted_row_sha256": receipt["rows"][1]["row_sha256"],
            "fitted_subset_rows_sha256": receipt["rows_sha256"],
        })
        training_module.validate_runtime_seed_fitted_subset(seed, receipt)
        self.assertEqual(seed["provenance"]["predecessor_sequence_lane"], "motion")
        self.assertEqual(seed["provenance"]["first_fitted_sequence_lane"], "motion")
        lane_seed = dict(seed)
        lane_seed["provenance"] = dict(seed["provenance"])
        lane_seed["provenance"]["predecessor_sequence_lane"] = "idle_phase_0"
        with self.assertRaisesRegex(ValueError, "membership"):
            training_module.validate_runtime_seed_fitted_subset(lane_seed, receipt)
        missing = fitted_subset_metadata(Dataset(), (0, 2))
        with self.assertRaisesRegex(ValueError, "membership"):
            training_module.validate_runtime_seed_fitted_subset(seed, missing)
        nonadjacent = dict(seed)
        nonadjacent["provenance"] = dict(seed["provenance"])
        nonadjacent["provenance"]["first_fitted_center_frame"] = 12
        nonadjacent["provenance"]["first_fitted_row_sha256"] = receipt["rows"][2]["row_sha256"]
        with self.assertRaisesRegex(ValueError, "adjacent"):
            training_module.validate_runtime_seed_fitted_subset(nonadjacent, receipt)

        checkpoint = type("Checkpoint", (), {"fitted_subset": receipt})()
        training_module.validate_resume_fitted_subset(checkpoint, receipt)
        with self.assertRaisesRegex(ValueError, "fitted subset"):
            training_module.validate_resume_fitted_subset(checkpoint, missing)
        rows[0]["sequence_lane"] = "idle_phase_0"
        changed_lane = fitted_subset_metadata(Dataset(), range(3))
        with self.assertRaisesRegex(ValueError, "fitted subset"):
            training_module.validate_resume_fitted_subset(checkpoint, changed_lane)

    def test_fitted_row_and_subset_receipts_bind_sequence_lane(self) -> None:
        base = {
            "x": np.zeros(INPUT_LAYOUT.size, np.float32),
            "y": np.zeros(OUTPUT_LAYOUT.size, np.float32),
            "phase": np.float32(0.0),
            "clip_id": "walk1_subject1",
            "center_frame": 30,
            "split_identity": "walk1_subject1",
            "split": "train",
            "sequence_lane": "motion",
            "terrain_class": "flat",
        }
        idle = dict(base)
        idle["sequence_lane"] = "idle_phase_0"
        self.assertNotEqual(fitted_row_sha256(base), fitted_row_sha256(idle))

        class Dataset:
            split = "train"
            rows = (base, idle)
            def __len__(self) -> int: return len(self.rows)
            def __getitem__(self, index: int) -> dict[str, object]: return self.rows[index]

        both = fitted_subset_metadata(Dataset(), (0, 1))
        self.assertEqual(
            [row["sequence_lane"] for row in both["rows"]],
            ["motion", "idle_phase_0"],
        )
        motion = fitted_subset_metadata(Dataset(), (0,))
        idle_only = fitted_subset_metadata(Dataset(), (1,))
        self.assertNotEqual(motion["rows_sha256"], idle_only["rows_sha256"])
        missing = dict(base)
        del missing["sequence_lane"]
        with self.assertRaisesRegex(ValueError, "fitted row receipt source"):
            fitted_row_sha256(missing)

    def test_rollout_default_depends_on_pipeline_mode_and_explicit_value_wins(self) -> None:
        resolve = train_module.resolve_rollout_finetune_frames
        self.assertEqual(resolve(None, pipeline_overfit=True), 0)
        self.assertEqual(resolve(None, pipeline_overfit=False), 16)
        self.assertEqual(resolve(3, pipeline_overfit=True), 3)
        self.assertEqual(resolve(0, pipeline_overfit=False), 0)
        for invalid in (-1, 1, 17):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "rollout fine-tuning"):
                    resolve(invalid, pipeline_overfit=False)

    def test_overfit_gate_uses_the_final_post_rollout_score(self) -> None:
        self.assertTrue(overfit_gate_accepted(10.0, 4.9, 0.5))
        self.assertFalse(overfit_gate_accepted(10.0, 5.1, 0.5))
        selection = provisional_pipeline_selection(
            one_step_score=0.25, accepted=True
        )
        self.assertEqual(selection["one_step_score"], 0.25)
        self.assertTrue(selection["provisional"])
        self.assertIsNone(
            provisional_pipeline_selection(one_step_score=0.25, accepted=False)
        )

    def test_pipeline_best_requires_bound_fixed_sample_and_traversal_receipt(self) -> None:
        metrics = {
            "gates": {
                "finite_20_seconds": True,
                "no_invalid_hold": True,
                "no_phase_reversal": True,
                "no_phase_freeze": True,
                "root_translation_step_within_limit": True,
                "root_rotation_step_within_limit": True,
                "joint_step_within_limit": True,
                "no_joint_limit_violation": True,
                "sole_penetration_within_limit": True,
                "zero_forbidden_body_penetration": True,
                "stance_sole_speed_within_limit": True,
            },
            "global": {"predicted_stance_sample_count": 12},
            "known_train_gate": {
                "crossed_supported_known_terrain": True,
                "returned_to_supported_continuation": True,
                "realized_nonstationary_traversal": True,
                "responsive_flat_motion_after_return": True,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "checkpoint-step-00000001.pt"
            candidate.write_bytes(b"candidate")
            best = root / "best.pt"
            bindings = {
                "dataset_digest_sha256": "a" * 64,
                "kinematic_signature_sha256": "b" * 64,
                "scenario_provenance_sha256": "c" * 64,
                "fitted_subset_rows_sha256": "d" * 64,
            }
            transition_report = self._fitted_envelope(
                _FittedEnvelopeRows._physical_output()
            )[0].to_dict()
            bindings.update({
                "fitted_transition_report": transition_report,
                "fitted_transition_report_sha256": transition_report[
                    "report_sha256"
                ],
            })
            promotion_expectations = {
                "expected_fixed_sample_score": 1.0,
                "expected_fixed_sample_count": 2048,
                "observed_fixed_sample_score": 1.0,
                "observed_fixed_sample_count": 2048,
            }
            self.assertFalse(
                promote_pipeline_best(
                    candidate_path=candidate,
                    best_path=best,
                    receipt=None,
                    **bindings,
                    **promotion_expectations,
                )
            )
            self.assertFalse(best.exists())

            mismatch = build_pipeline_promotion_receipt(
                checkpoint_path=candidate,
                expected_fixed_sample_score=1.0,
                observed_fixed_sample_score=1.001,
                expected_fixed_sample_count=2048,
                observed_fixed_sample_count=2048,
                closed_loop_metrics=metrics,
                **bindings,
            )
            self.assertFalse(mismatch["accepted"])
            self.assertFalse(
                promote_pipeline_best(
                    candidate_path=candidate,
                    best_path=best,
                    receipt=mismatch,
                    **bindings,
                    **promotion_expectations,
                )
            )
            self.assertFalse(best.exists())

            stationary_metrics = json.loads(json.dumps(metrics))
            stationary_metrics["known_train_gate"][
                "realized_nonstationary_traversal"
            ] = False
            stationary = build_pipeline_promotion_receipt(
                checkpoint_path=candidate,
                expected_fixed_sample_score=1.0,
                observed_fixed_sample_score=1.0,
                expected_fixed_sample_count=2048,
                observed_fixed_sample_count=2048,
                closed_loop_metrics=stationary_metrics,
                **bindings,
            )
            self.assertFalse(stationary["accepted"])

            unresponsive_metrics = json.loads(json.dumps(metrics))
            unresponsive_metrics["known_train_gate"][
                "responsive_flat_motion_after_return"
            ] = False
            unresponsive = build_pipeline_promotion_receipt(
                checkpoint_path=candidate,
                expected_fixed_sample_score=1.0,
                observed_fixed_sample_score=1.0,
                expected_fixed_sample_count=2048,
                observed_fixed_sample_count=2048,
                closed_loop_metrics=unresponsive_metrics,
                **bindings,
            )
            self.assertFalse(unresponsive["accepted"])

            passing = build_pipeline_promotion_receipt(
                checkpoint_path=candidate,
                expected_fixed_sample_score=1.0,
                observed_fixed_sample_score=1.0,
                expected_fixed_sample_count=2048,
                observed_fixed_sample_count=2048,
                closed_loop_metrics=metrics,
                **bindings,
            )
            self.assertTrue(passing["accepted"])
            self.assertEqual(
                passing["schema"], "mm-sonic-pipeline-promotion-receipt/v2"
            )
            self.assertEqual(passing["fitted_transition_report"], transition_report)
            for field, forged_value in (
                ("expected_fixed_sample_score", 2.0),
                ("observed_fixed_sample_score", 2.0),
                ("expected_fixed_sample_count", 4096),
                ("observed_fixed_sample_count", 4096),
            ):
                tampered = json.loads(json.dumps(passing))
                tampered[field] = forged_value
                receipt_base = {
                    key: value for key, value in tampered.items()
                    if key != "receipt_sha256"
                }
                tampered["receipt_sha256"] = hashlib.sha256(json.dumps(
                    receipt_base, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ).encode()).hexdigest()
                tampered_best = root / f"tampered-{field}.pt"
                with self.subTest(tamper=field):
                    self.assertFalse(promote_pipeline_best(
                        candidate_path=candidate,
                        best_path=tampered_best,
                        receipt=tampered,
                        **bindings,
                        **promotion_expectations,
                    ))
                    self.assertFalse(tampered_best.exists())
            self.assertTrue(
                promote_pipeline_best(
                    candidate_path=candidate,
                    best_path=best,
                    receipt=passing,
                    **bindings,
                    **promotion_expectations,
                )
            )
            self.assertEqual(best.read_bytes(), b"candidate")
            self.assertEqual(best.stat().st_mode & 0o222, 0)
            forged = dict(passing)
            forged["dataset_digest_sha256"] = "e" * 64
            forged_best = root / "forged-best.pt"
            self.assertFalse(
                promote_pipeline_best(
                    candidate_path=candidate,
                    best_path=forged_best,
                    receipt=forged,
                    **bindings,
                    **promotion_expectations,
                )
            )
            self.assertFalse(forged_best.exists())

            for label, mutate in (
                (
                    "maxima",
                    lambda value: value["maxima"].__setitem__(
                        "joint_step_rad", 0.125
                    ),
                ),
                (
                    "first failure",
                    lambda value: value.__setitem__("first_failure", {
                        "clip_id": "forged", "sequence_lane": "motion",
                        "center_frame": 1, "field": "joint_position",
                        "joint": ISAACLAB_JOINT_NAMES[0], "value": 0.1,
                        "limit": 0.25,
                    }),
                ),
                (
                    "sample count",
                    lambda value: value.__setitem__(
                        "sample_count", value["sample_count"] + 1
                    ),
                ),
                (
                    "row digest",
                    lambda value: value.__setitem__("rows_sha256", "e" * 64),
                ),
            ):
                tampered = json.loads(json.dumps(passing))
                nested = tampered["fitted_transition_report"]
                mutate(nested)
                nested_base = {
                    key: value for key, value in nested.items()
                    if key != "report_sha256"
                }
                nested["report_sha256"] = hashlib.sha256(json.dumps(
                    nested_base, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ).encode()).hexdigest()
                tampered["fitted_transition_report_sha256"] = nested[
                    "report_sha256"
                ]
                receipt_base = {
                    key: value for key, value in tampered.items()
                    if key != "receipt_sha256"
                }
                tampered["receipt_sha256"] = hashlib.sha256(json.dumps(
                    receipt_base, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ).encode()).hexdigest()
                tampered_best = root / f"tampered-{label.replace(' ', '-')}.pt"
                with self.subTest(tamper=label):
                    self.assertFalse(promote_pipeline_best(
                        candidate_path=candidate,
                        best_path=tampered_best,
                        receipt=tampered,
                        **bindings,
                        **promotion_expectations,
                    ))
                    self.assertFalse(tampered_best.exists())

            digest_tampered = json.loads(json.dumps(passing))
            digest_tampered["fitted_transition_report_sha256"] = "f" * 64
            receipt_base = {
                key: value for key, value in digest_tampered.items()
                if key != "receipt_sha256"
            }
            digest_tampered["receipt_sha256"] = hashlib.sha256(json.dumps(
                receipt_base, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode()).hexdigest()
            self.assertFalse(promote_pipeline_best(
                candidate_path=candidate,
                best_path=root / "tampered-report-digest.pt",
                receipt=digest_tampered,
                **bindings,
                **promotion_expectations,
            ))
            old_v1 = dict(passing)
            old_v1["schema"] = "mm-sonic-pipeline-promotion-receipt/v1"
            old_base = {
                key: value for key, value in old_v1.items()
                if key != "receipt_sha256"
            }
            old_v1["receipt_sha256"] = hashlib.sha256(json.dumps(
                old_base, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode()).hexdigest()
            self.assertFalse(promote_pipeline_best(
                candidate_path=candidate,
                best_path=root / "old-v1.pt",
                receipt=old_v1,
                **bindings,
                **promotion_expectations,
            ))
            nonjson = json.loads(json.dumps(passing))
            nonjson["fitted_transition_report"]["accepted"] = False
            nonjson["fitted_transition_report"]["first_failure"] = {
                "clip_id": "forged", "sequence_lane": "motion",
                "center_frame": 1, "field": "output", "joint": None,
                "value": {"not-json"}, "limit": "finite",
            }
            self.assertFalse(promote_pipeline_best(
                candidate_path=candidate,
                best_path=root / "non-json-report.pt",
                receipt=nonjson,
                **bindings,
                **promotion_expectations,
            ))

    def test_materialized_overfit_subset_reads_source_once_in_sorted_order(self) -> None:
        class Rows:
            split = "train"
            x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
            x_std = np.ones(INPUT_LAYOUT.size, np.float32)
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            def __init__(self) -> None:
                self.calls: list[int] = []

            def __getitem__(self, index: int) -> dict[str, object]:
                self.calls.append(index)
                return {
                    "x": np.full(INPUT_LAYOUT.size, index, np.float32),
                    "y": np.full(OUTPUT_LAYOUT.size, index, np.float32),
                    "phase": np.float32(index),
                    "clip_id": f"clip-{index}",
                    "center_frame": index,
                    "sequence_lane": "motion",
                    "terrain_class": "flat",
                }

        source = Rows()
        cached = materialize_subset(source, [2, 0])
        self.assertEqual(source.calls, [0, 2])
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached[0]["clip_id"], "clip-2")
        self.assertEqual(cached[1]["clip_id"], "clip-0")
        self.assertEqual(source.calls, [0, 2])

    def test_phase_q99_clamps_only_normalized_round_trip_zero(self) -> None:
        class TrainingRows:
            split = "train"
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            def __init__(self) -> None:
                self.y_mean[OUTPUT_LAYOUT["phase_advance"]] = np.float32(0.11376687)
                self.y_std[OUTPUT_LAYOUT["phase_advance"]] = np.float32(0.10085002)
                self.row = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                # This is the exact normalized float32 emitted for a raw zero
                # in the remediated canary normalization contract.
                self.row[OUTPUT_LAYOUT["phase_advance"]] = np.float32(-1.1280799)

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> dict[str, object]:
                return {"y": self.row}

        self.assertEqual(training_phase_advance_q99(TrainingRows()), 0.0)

    def test_stratified_subset_is_balanced_and_deterministic(self) -> None:
        classes = ["flat"] * 9 + ["ascent"] * 5 + ["descent"] * 4 + ["transition"] * 2
        first = stratified_subset(classes, 8, seed=7)
        second = stratified_subset(classes, 8, seed=7)
        self.assertEqual(first, second)
        counts = {name: sum(classes[index] == name for index in first) for name in set(classes)}
        self.assertEqual(set(counts.values()), {2})

    def test_sealed_test_receipt_is_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            receipt = claim_sealed_test_receipt(
                run, checkpoint_sha256="a" * 64, dataset_digest="b" * 64
            )
            self.assertTrue(receipt.is_file())
            with self.assertRaisesRegex(RuntimeError, "already"):
                claim_sealed_test_receipt(
                    run, checkpoint_sha256="a" * 64, dataset_digest="b" * 64
                )

    def test_losses_are_finite_and_report_every_group(self) -> None:
        torch.manual_seed(3)
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        prediction = model(
            torch.zeros(4, INPUT_LAYOUT.size), torch.linspace(0.0, 1.0, 4)
        )
        target = torch.zeros_like(prediction)
        target[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor(
            (0.0, 1.0, 0.0, 1.0)
        )
        losses = pfnn_losses(
            prediction, target, model=model, normalization=_normalization()
        )
        self.assertEqual(
            set(losses),
            {
                "trajectory_mse", "body_mse", "root_pose_mse", "joint_mse",
                "root_motion_mse", "phase_mse", "contact_bce",
                "trajectory_direction", "phase_nonnegative", "fk_consistency",
                "regularization", "total",
            },
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_loss_multipliers_are_frozen_to_exactly_one_at_every_boundary(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=4, dropout_probability=0.0)
        prediction = torch.zeros(2, OUTPUT_LAYOUT.size)
        target = torch.zeros_like(prediction)
        for name in LOSS_WEIGHT_KEYS:
            with self.subTest(training_weight=name):
                weights = dict(DEFAULT_LOSS_WEIGHTS)
                weights[name] = 0.5
                with self.assertRaisesRegex(ValueError, "loss weights"):
                    pfnn_losses(
                        prediction,
                        target,
                        model=model,
                        normalization=_normalization(),
                        loss_weights=weights,
                    )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0e-3, weight_decay=0.0
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            for weights in (
                {name: 1.0 for name in LOSS_WEIGHT_KEYS[:-1]},
                {**DEFAULT_LOSS_WEIGHTS, "extra": 1.0},
            ):
                with self.subTest(save_weights=tuple(weights)):
                    with self.assertRaisesRegex(ValueError, "loss weights"):
                        save_checkpoint(
                            path,
                            model,
                            optimizer,
                            _normalization(),
                            dataset_digest="abc",
                            kinematic_signature_sha256="def",
                            runtime_seed=finite_runtime_seed(),
                            step=1,
                            loss_weights=weights,
                        )
            save_checkpoint(
                path,
                model,
                optimizer,
                _normalization(),
                dataset_digest="abc",
                kinematic_signature_sha256="def",
                runtime_seed=finite_runtime_seed(),
                step=1,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["loss_weights"]["regularization"] = 2.0
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "loss weights"):
                load_checkpoint(
                    path,
                    expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )

    def test_contact_loss_uses_raw_logits_and_structural_terms_are_physical(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=4, dropout_probability=0.0)
        prediction = torch.zeros(2, OUTPUT_LAYOUT.size)
        target = torch.zeros_like(prediction)
        prediction[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor(
            ((-2.0, 2.0, -1.0, 1.0), (-3.0, 3.0, -4.0, 4.0))
        )
        target[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor(
            ((0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0))
        )
        normal = _normalization()
        normal["y_mean"][OUTPUT_LAYOUT["trajectory_direction"]] = 10.0
        normal["y_std"][OUTPUT_LAYOUT["trajectory_direction"]] = 2.0
        losses = pfnn_losses(
            prediction, target, model=model, normalization=normal
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[:, OUTPUT_LAYOUT["contact_logit"]],
            target[:, OUTPUT_LAYOUT["contact_logit"]],
        )
        torch.testing.assert_close(losses["contact_bce"], expected)
        # A normalized zero direction denormalizes to (10,10), so this proves
        # the unit-direction constraint is evaluated in physical space.
        self.assertGreater(float(losses["trajectory_direction"]), 100.0)

    def test_checkpoint_round_trip_and_contract_mismatch(self) -> None:
        torch.manual_seed(5)
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0e-3, weight_decay=0.0
        )
        seed = finite_runtime_seed()
        fitted_rows = [
            {
                "clip_id": "synthetic", "center_frame": center,
                "split_identity": "synthetic", "sequence_lane": "motion",
                "terrain_class": "flat",
                "row_sha256": digest,
            }
            for center, digest in ((0, "1" * 64), (1, "2" * 64))
        ]
        fitted_subset = {
            "split": "train",
            "rows": fitted_rows,
            "rows_sha256": hashlib.sha256(json.dumps(
                fitted_rows, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "class_counts": {
                "flat": 2, "ascent": 0, "descent": 0, "transition": 0,
            },
        }
        seed["provenance"].update({
            "predecessor_row_sha256": "1" * 64,
            "first_fitted_row_sha256": "2" * 64,
            "fitted_subset_rows_sha256": fitted_subset["rows_sha256"],
        })
        sequence_sampler = train_module.DeterministicSequenceSampler(
            7, batch_size=3, seed=41
        )
        sequence_sampler.next_batch()
        sequence_sampler_state = sequence_sampler.state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path, model, optimizer, _normalization(),
                dataset_digest="abc", kinematic_signature_sha256="def",
                runtime_seed=seed, step=17, epoch=3,
                sampler_epoch=2, sampler_global_offset=8,
                sequence_sampler_state=sequence_sampler_state,
                fitted_subset=fitted_subset,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["schema"], "mm-sonic-terrain-pfnn-checkpoint/v5")
            self.assertEqual(payload["input_layout"], [list(field) for field in INPUT_LAYOUT.fields])
            self.assertEqual(payload["output_layout"], [list(field) for field in OUTPUT_LAYOUT.fields])
            self.assertEqual(
                payload["normalization_contract"]["root_tilt_encoding"],
                "angle_axis_xy",
            )
            self.assertEqual(
                payload["sampler_state"],
                {
                    "epoch": 2,
                    "global_offset": 8,
                    "sequence": sequence_sampler_state,
                },
            )
            loaded = load_checkpoint(
                path, expected_dataset_digest="abc",
                expected_kinematic_signature_sha256="def",
            )
            self.assertEqual(loaded.schema, CHECKPOINT_SCHEMA)
            self.assertEqual(loaded.step, 17)
            self.assertEqual(loaded.sampler_epoch, 2)
            self.assertEqual(loaded.sampler_global_offset, 8)
            self.assertEqual(
                loaded.sequence_sampler_state, sequence_sampler_state
            )
            self.assertEqual(loaded.runtime_seed.keys(), seed.keys())
            self.assertEqual(loaded.fitted_subset, fitted_subset)
            old_schema = payload["schema"]
            payload["schema"] = "mm-sonic-terrain-pfnn-checkpoint/v4"
            torch.save(payload, path)
            with patch(
                "mm_sonic.terrain_pfnn.training.PhaseFunctionedNetwork",
                side_effect=AssertionError("old schema opened model state"),
            ) as model_constructor:
                with self.assertRaisesRegex(ValueError, "schema"):
                    load_checkpoint(
                        path,
                        expected_dataset_digest="abc",
                        expected_kinematic_signature_sha256="def",
                    )
                model_constructor.assert_not_called()
            payload["schema"] = old_schema
            torch.save(payload, path)
            original_rows_sha256 = payload["fitted_subset"]["rows_sha256"]
            payload["fitted_subset"]["rows"][0]["sequence_lane"] = "idle_phase_0"
            payload["fitted_subset"]["rows_sha256"] = hashlib.sha256(
                json.dumps(
                    payload["fitted_subset"]["rows"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "membership"):
                load_checkpoint(
                    path,
                    expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )
            payload["fitted_subset"]["rows"][0]["sequence_lane"] = "motion"
            payload["fitted_subset"]["rows_sha256"] = original_rows_sha256
            torch.save(payload, path)
            class MatchingKinematics:
                kinematic_signature_sha256 = "def"
                joint_limits = torch.tensor(
                    [[-np.pi, np.pi]] * 29, dtype=torch.float64
                )

            validate_checkpoint_kinematics(loaded, MatchingKinematics())
            MatchingKinematics.joint_limits[0, 0] = -1.0
            with self.assertRaisesRegex(ValueError, "joint limits"):
                validate_checkpoint_kinematics(loaded, MatchingKinematics())
            resume_model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
            resume_optimizer = torch.optim.Adam(
                resume_model.parameters(), lr=1.0e-3, weight_decay=0.0
            )
            self.assertEqual(
                restore_training_state(
                    loaded, resume_model, resume_optimizer,
                    normalization=_normalization(),
                    train_identities=(), validation_identities=(),
                ),
                (17, 3),
            )
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(model.parameters(), resume_model.parameters())
                )
            )
            with self.assertRaisesRegex(ValueError, "dataset digest"):
                load_checkpoint(
                    path, expected_dataset_digest="different",
                    expected_kinematic_signature_sha256="def",
                )
            with self.assertRaisesRegex(ValueError, "kinematic signature"):
                load_checkpoint(
                    path, expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="different",
                )
            original_w0 = payload["model_state"]["W0"]
            payload["model_state"]["W0"] = torch.zeros_like(
                original_w0, dtype=torch.int64
            )
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "model state"):
                load_checkpoint(
                    path, expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )
            payload["model_state"]["W0"] = original_w0
            original_optimizer_state = payload["optimizer_state"]
            payload["optimizer_state"] = {"unexpected": []}
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "optimizer state"):
                load_checkpoint(
                    path, expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )
            payload["optimizer_state"] = original_optimizer_state
            payload["selection"] = {"provisional": False}
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "selection"):
                load_checkpoint(
                    path, expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )

    def test_train_rejects_active_sequence_sampler_mismatch_before_restore(self) -> None:
        class Rows:
            split = "train"

            def __init__(self) -> None:
                self.x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
                self.x_std = np.ones(INPUT_LAYOUT.size, np.float32)
                self.y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                self.y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)
                self.rows = [self._row(center) for center in range(10, 27)]

            @staticmethod
            def _state(center: int) -> tuple[np.ndarray, np.ndarray]:
                trajectory = np.zeros((12, 2), np.float32)
                trajectory[:, 0] = np.linspace(-0.2, 0.8, 12) + center * 0.001
                direction = np.zeros((12, 2), np.float32)
                direction[:, 0] = 1.0
                body = np.full((30, 3), center * 0.001, np.float32)
                return np.concatenate((trajectory.ravel(), direction.ravel())), body

            @classmethod
            def _row(cls, center: int) -> dict[str, object]:
                current_trajectory, current_body = cls._state(center)
                target_trajectory, target_body = cls._state(center + 1)
                x = np.zeros(INPUT_LAYOUT.size, np.float32)
                x[INPUT_LAYOUT["trajectory_position"]] = current_trajectory[:24]
                x[INPUT_LAYOUT["trajectory_direction"]] = current_trajectory[24:]
                x[INPUT_LAYOUT["semantic_intent"]] = np.tile(
                    (1.0, 0.0), (12, 1)
                ).ravel()
                x[INPUT_LAYOUT["previous_body_position"]] = (
                    current_body.ravel() * 0.1
                )
                x[INPUT_LAYOUT["previous_body_velocity"]] = (
                    current_body.ravel() * 0.1
                )
                y = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                y[OUTPUT_LAYOUT["trajectory_position"]] = target_trajectory[:24]
                y[OUTPUT_LAYOUT["trajectory_direction"]] = target_trajectory[24:]
                y[OUTPUT_LAYOUT["body_position"]] = target_body.ravel()
                y[OUTPUT_LAYOUT["body_velocity"]] = target_body.ravel()
                y[OUTPUT_LAYOUT["root_height"]] = 0.8
                y[OUTPUT_LAYOUT["root_planar_velocity"]] = (center * 0.001, 0.0)
                y[OUTPUT_LAYOUT["phase_advance"]] = 0.1
                y[OUTPUT_LAYOUT["contact_logit"]] = (1.0, 0.0, 1.0, 0.0)
                return {
                    "x": x,
                    "y": y,
                    "phase": np.float32(center * 0.1),
                    "clip_id": "walk1_subject1",
                    "center_frame": center,
                    "split_identity": "walk1_subject1",
                    "split": "train",
                    "sequence_lane": "motion",
                    "terrain_class": "flat",
                }

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        class Kinematics:
            kinematic_signature_sha256 = "def"
            joint_limits = torch.tensor(
                [[-2.0, 2.0]] * 29, dtype=torch.float64
            )

            def to(self, device: torch.device) -> Kinematics:
                return self

        rows = Rows()
        active_seed = 7 + 97
        active_sequence_count = 2
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.30)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0e-3, weight_decay=0.0
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps({
                "dataset_digest_sha256": "abc",
                "split_identities": {"train": [], "validation": []},
            }))
            arguments = train_module.argparse.Namespace(
                dataset=str(root),
                model_path="unused.xml",
                output=str(root / "output"),
                overfit_samples=None,
                steps=2,
                seed=7,
                hidden_size=8,
                batch_size=1,
                learning_rate=1.0e-3,
                evaluation_batch_size=1,
                overfit_acceptance_ratio=0.5,
                rollout_finetune_frames=16,
                rollout_finetune_steps=1,
                resume=str(root / "checkpoint.pt"),
            )
            mismatches = {
                "count": train_module.DeterministicSequenceSampler(
                    active_sequence_count + 1,
                    batch_size=1,
                    seed=active_seed,
                ).state_dict(),
                "seed": train_module.DeterministicSequenceSampler(
                    active_sequence_count,
                    batch_size=1,
                    seed=active_seed + 1,
                ).state_dict(),
            }
            for label, sampler_state in mismatches.items():
                checkpoint_path = root / f"checkpoint-{label}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    _normalization(),
                    dataset_digest="abc",
                    kinematic_signature_sha256="def",
                    runtime_seed=finite_runtime_seed(),
                    step=1,
                    epoch=1,
                    seed=7,
                    sampler_epoch=0,
                    sampler_global_offset=0,
                    sequence_sampler_state=sampler_state,
                )
                loaded = load_checkpoint(
                    checkpoint_path,
                    expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )
                arguments.resume = str(checkpoint_path)
                with self.subTest(mismatch=label), patch.object(
                    train_module, "_distributed_context",
                    return_value=(0, 1, 0, torch.device("cpu")),
                ), patch.object(
                    train_module, "_dataset_root", return_value=root
                ), patch.object(
                    train_module, "PFNNShardDataset", return_value=rows
                ), patch.object(
                    train_module.TorchG1ForwardKinematics,
                    "from_mjcf",
                    return_value=Kinematics(),
                ), patch.object(
                    train_module, "load_checkpoint", return_value=loaded
                ), patch.object(
                    PhaseFunctionedNetwork,
                    "load_state_dict",
                    side_effect=AssertionError("model restore was reached"),
                ) as model_restore, patch.object(
                    torch.optim.Adam,
                    "load_state_dict",
                    side_effect=AssertionError("Adam restore was reached"),
                ) as adam_restore:
                    with self.assertRaisesRegex(ValueError, "sequence sampler"):
                        train_module.train(arguments)
                    model_restore.assert_not_called()
                    adam_restore.assert_not_called()

    def test_checkpoint_rejects_inexact_nested_tensor_contracts(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0e-3, weight_decay=0.0
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pt"
            tampered = root / "tampered.pt"
            save_checkpoint(
                baseline,
                model,
                optimizer,
                _normalization(),
                dataset_digest="abc",
                kinematic_signature_sha256="def",
                runtime_seed=finite_runtime_seed(),
                step=1,
            )

            def rejected(label: str, mutation: object, match: str) -> None:
                payload = torch.load(baseline, map_location="cpu", weights_only=True)
                mutation(payload)
                torch.save(payload, tampered)
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, match):
                        load_checkpoint(
                            tampered,
                            expected_dataset_digest="abc",
                            expected_kinematic_signature_sha256="def",
                        )

            rejected(
                "normalization extra key",
                lambda value: value["normalization"].__setitem__("extra", torch.zeros(1)),
                "normalization",
            )
            rejected(
                "normalization missing key",
                lambda value: value["normalization"].pop("x_mean"),
                "normalization",
            )
            rejected(
                "normalization list",
                lambda value: value["normalization"].__setitem__(
                    "x_mean", value["normalization"]["x_mean"].tolist()
                ),
                "normalization",
            )
            rejected(
                "normalization dtype",
                lambda value: value["normalization"].__setitem__(
                    "x_mean", value["normalization"]["x_mean"].to(torch.float64)
                ),
                "normalization",
            )
            rejected(
                "normalization numpy",
                lambda value: value["normalization"].__setitem__(
                    "x_mean", value["normalization"]["x_mean"].numpy()
                ),
                "loaded safely|normalization",
            )
            rejected(
                "normalization stride",
                lambda value: value["normalization"].__setitem__(
                    "x_mean", torch.zeros(INPUT_LAYOUT.size, 2)[:, 0]
                ),
                "normalization",
            )
            rejected(
                "runtime seed extra key",
                lambda value: value["runtime_seed"].__setitem__("extra", None),
                "runtime seed",
            )
            rejected(
                "runtime seed missing key",
                lambda value: value["runtime_seed"].pop("world_xy"),
                "runtime seed",
            )
            rejected(
                "runtime seed list",
                lambda value: value["runtime_seed"].__setitem__(
                    "joint_position", value["runtime_seed"]["joint_position"].tolist()
                ),
                "runtime seed",
            )
            rejected(
                "runtime seed dtype",
                lambda value: value["runtime_seed"].__setitem__(
                    "joint_position",
                    value["runtime_seed"]["joint_position"].to(torch.float64),
                ),
                "runtime seed",
            )
            rejected(
                "runtime seed numpy",
                lambda value: value["runtime_seed"].__setitem__(
                    "joint_position",
                    value["runtime_seed"]["joint_position"].numpy(),
                ),
                "loaded safely|runtime seed",
            )
            rejected(
                "runtime seed stride",
                lambda value: value["runtime_seed"].__setitem__(
                    "body_position", torch.zeros(30, 6)[:, ::2]
                ),
                "runtime seed",
            )
            rejected(
                "runtime seed normalized input digest",
                lambda value: value["runtime_seed"]["normalized_input"].__setitem__(
                    0,
                    value["runtime_seed"]["normalized_input"][0] + 1.0,
                ),
                "normalized input digest",
            )
            rejected(
                "runtime provenance extra key",
                lambda value: value["runtime_seed"]["provenance"].__setitem__(
                    "extra", 0
                ),
                "runtime seed provenance",
            )
            rejected(
                "runtime predecessor lane",
                lambda value: value["runtime_seed"]["provenance"].__setitem__(
                    "predecessor_sequence_lane", "idle_phase_8"
                ),
                "runtime seed provenance",
            )
            rejected(
                "joint limits list",
                lambda value: value.__setitem__(
                    "joint_limits", value["joint_limits"].tolist()
                ),
                "joint limits",
            )
            rejected(
                "joint limits dtype",
                lambda value: value.__setitem__(
                    "joint_limits", value["joint_limits"].to(torch.float32)
                ),
                "joint limits",
            )
            rejected(
                "joint limits numpy",
                lambda value: value.__setitem__(
                    "joint_limits", value["joint_limits"].numpy()
                ),
                "loaded safely|joint limits",
            )
            rejected(
                "joint limits stride",
                lambda value: value.__setitem__(
                    "joint_limits", torch.zeros(29, 4, dtype=torch.float64)[:, ::2]
                ),
                "joint limits",
            )
            rejected(
                "sampler missing key",
                lambda value: value["sampler_state"].pop("global_offset"),
                "sampler state",
            )
            rejected(
                "sampler extra key",
                lambda value: value["sampler_state"].__setitem__("extra", 0),
                "sampler state",
            )

    def test_checkpoint_rejects_malformed_adam_group_before_restore(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0e-3, weight_decay=0.0
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pt"
            tampered = root / "tampered.pt"
            save_checkpoint(
                baseline,
                model,
                optimizer,
                _normalization(),
                dataset_digest="abc",
                kinematic_signature_sha256="def",
                runtime_seed=finite_runtime_seed(),
                step=1,
            )

            mutations = {
                "string lr": lambda group: group.__setitem__("lr", "0.001"),
                "tensor lr": lambda group: group.__setitem__("lr", torch.tensor(0.001)),
                "numpy lr": lambda group: group.__setitem__("lr", np.float32(0.001)),
                "nonfinite lr": lambda group: group.__setitem__("lr", float("nan")),
                "zero eps": lambda group: group.__setitem__("eps", 0.0),
                "negative weight decay": lambda group: group.__setitem__(
                    "weight_decay", -1.0
                ),
                "malformed betas container": lambda group: group.__setitem__(
                    "betas", {"first": 0.9, "second": 0.999}
                ),
                "malformed beta value": lambda group: group.__setitem__(
                    "betas", (0.9, 1.0)
                ),
                "extra group key": lambda group: group.__setitem__("extra", None),
                "invalid param id": lambda group: group.__setitem__(
                    "params", [0, 1, 2, 3, 4, 6]
                ),
                "duplicate param id": lambda group: group.__setitem__(
                    "params", [0, 1, 2, 3, 4, 4]
                ),
                "bool param id": lambda group: group.__setitem__(
                    "params", [False, 1, 2, 3, 4, 5]
                ),
                "wrong bool flag": lambda group: group.__setitem__("amsgrad", 0),
                "wrong foreach flag": lambda group: group.__setitem__("foreach", "no"),
                "wrong fused flag": lambda group: group.__setitem__(
                    "fused", torch.tensor(False)
                ),
            }
            for label, mutation in mutations.items():
                payload = torch.load(
                    baseline, map_location="cpu", weights_only=True
                )
                mutation(payload["optimizer_state"]["param_groups"][0])
                torch.save(payload, tampered)
                with self.subTest(label=label):
                    with patch.object(
                        torch.optim.Adam,
                        "load_state_dict",
                        side_effect=AssertionError("Adam restore was reached"),
                    ) as restore:
                        with self.assertRaises(ValueError):
                            load_checkpoint(
                                tampered,
                                expected_dataset_digest="abc",
                                expected_kinematic_signature_sha256="def",
                            )
                        restore.assert_not_called()

    def test_phase_recurrence_uses_shared_forward_clamp_and_required_cap(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=4, dropout_probability=0.0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            output_direction = model.b2[
                :, OUTPUT_LAYOUT["trajectory_direction"]
            ].reshape(model.b2.shape[0], 12, 2)
            output_direction[..., 0] = 1.0
            model.b2[:, OUTPUT_LAYOUT["phase_advance"]] = -0.25
        inputs = torch.zeros(2, 1, INPUT_LAYOUT.size)
        input_direction = inputs[
            ..., INPUT_LAYOUT["trajectory_direction"]
        ].reshape(2, 1, 12, 2)
        input_direction[..., 0] = 1.0
        input_semantic = inputs[
            ..., INPUT_LAYOUT["semantic_intent"]
        ].reshape(2, 1, 12, 2)
        input_semantic[..., 0] = 1.0
        targets = torch.zeros(2, 1, OUTPUT_LAYOUT.size)
        initial_phase = torch.tensor((0.7,))
        result = autoregressive_unroll(
            model,
            inputs,
            initial_phase,
            targets,
            normalization=_normalization(),
            phase_advance_cap=0.2,
        )
        self.assertLess(
            float(
                result.predictions[0][0, OUTPUT_LAYOUT["phase_advance"]][0].detach()
            ),
            0.0,
        )
        torch.testing.assert_close(result.phases[1], initial_phase)
        later_phase_only_loss = result.phases[1].square().sum()
        gradient = torch.autograd.grad(
            later_phase_only_loss, result.predictions[0], retain_graph=True
        )[0]
        phase_gradient = gradient[:, OUTPUT_LAYOUT["phase_advance"]]
        self.assertTrue(torch.isfinite(phase_gradient).all())
        torch.testing.assert_close(phase_gradient, torch.zeros_like(phase_gradient))

    def test_rollout_sequences_ignore_storage_order_and_reject_bad_metadata(self) -> None:
        rows = [
            {"clip_id": "walk4_subject1", "center_frame": 11, "sequence_lane": "motion", "terrain_class": "flat"},
            {"clip_id": "walk1_subject1", "center_frame": 2, "sequence_lane": "motion", "terrain_class": "flat"},
            {"clip_id": "walk4_subject1", "center_frame": 10, "sequence_lane": "motion", "terrain_class": "flat"},
            {"clip_id": "walk1_subject1", "center_frame": 1, "sequence_lane": "motion", "terrain_class": "flat"},
            {"clip_id": "walk1_subject1", "center_frame": 4, "sequence_lane": "motion", "terrain_class": "flat"},
            {"clip_id": "walk1_subject1", "center_frame": 5, "sequence_lane": "motion", "terrain_class": "flat"},
        ]
        rows.extend(
            {"clip_id": "walk1_subject1", "center_frame": 3, "sequence_lane": "motion", "terrain_class": "flat"}
            for _ in range(8)
        )

        class Rows:
            split = "train"

            def __init__(self, values: list[dict[str, object]]) -> None:
                self.values = values

            def __len__(self) -> int:
                return len(self.values)

            def __getitem__(self, index: int) -> dict[str, object]:
                value = dict(self.values[index])
                value.setdefault("split", "train")
                value.setdefault("split_identity", value["clip_id"])
                return value

        def metadata_sequences(values: list[dict[str, object]]) -> list[tuple[object, ...]]:
            sequences, _ = train_module._consecutive_starts(Rows(values), 2)
            return [
                tuple(
                    (values[index]["clip_id"], values[index]["center_frame"])
                    for index in sequence
                )
                for sequence in sequences
            ]

        expected = [
            (("walk1_subject1", 1), ("walk1_subject1", 2)),
            (("walk1_subject1", 4), ("walk1_subject1", 5)),
            (("walk4_subject1", 10), ("walk4_subject1", 11)),
        ]
        clean_rows = rows[:6]
        self.assertEqual(metadata_sequences(clean_rows), expected)
        shuffled = [clean_rows[index] for index in (0, 5, 3, 2, 1, 4)]
        self.assertEqual(metadata_sequences(shuffled), expected)
        with self.assertRaisesRegex(ValueError, "duplicate.*lane"):
            metadata_sequences(rows)

        mismatched = [dict(clean_rows[3]), dict(clean_rows[1])]
        mismatched[1]["split_identity"] = "different"
        with self.assertRaisesRegex(ValueError, "split/identity"):
            train_module._consecutive_starts(Rows(mismatched), 2)
        validation = Rows([clean_rows[3], clean_rows[1]])
        validation.split = "validation"
        with self.assertRaisesRegex(ValueError, "training split"):
            train_module._consecutive_starts(validation, 2)

    def test_rollout_sequences_are_exactly_lane_safe_and_reject_duplicates(self) -> None:
        lanes = ("motion", *(f"idle_phase_{index}" for index in range(8)))
        values: list[dict[str, object]] = []
        for clip, first_center in (
            ("walk1_subject1", 10),
            ("walk4_subject1", 12),
        ):
            for lane in lanes:
                for center in range(first_center, first_center + 3):
                    if clip == "walk4_subject1" and lane == "idle_phase_7" and center == 13:
                        continue
                    values.append({
                        "clip_id": clip,
                        "center_frame": center,
                        "sequence_lane": lane,
                        "terrain_class": "flat",
                        "split": "train",
                        "split_identity": clip,
                    })

        class Rows:
            split = "train"

            def __init__(self, rows: list[dict[str, object]]) -> None:
                self.rows = rows

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, object]:
                return self.rows[index]

        sequences, _ = train_module._consecutive_starts(Rows(values), 3)
        keys = [
            tuple(
                (
                    values[index]["clip_id"],
                    values[index]["sequence_lane"],
                    values[index]["center_frame"],
                )
                for index in sequence
            )
            for sequence in sequences
        ]
        self.assertEqual(len(keys), 17)
        self.assertEqual(len(set(keys)), 17)
        self.assertTrue(
            all(
                len({clip for clip, _, _ in sequence}) == 1
                and len({lane for _, lane, _ in sequence}) == 1
                and all(
                    right == left + 1
                    for left, right in zip(
                        [center for _, _, center in sequence],
                        [center for _, _, center in sequence][1:],
                    )
                )
                for sequence in keys
            )
        )
        self.assertFalse(
            any(
                clip == "walk4_subject1" and lane == "idle_phase_7"
                for sequence in keys
                for clip, lane, _ in sequence
            )
        )

        duplicate = values + [dict(values[0])]
        with self.assertRaisesRegex(ValueError, "duplicate.*lane|lane.*duplicate"):
            train_module._consecutive_starts(Rows(duplicate), 3)
        invalid_lane = [dict(value) for value in values]
        invalid_lane[0]["sequence_lane"] = "idle_phase_8"
        with self.assertRaisesRegex(ValueError, "lane"):
            train_module._consecutive_starts(Rows(invalid_lane), 3)

    def test_sampler_resume_reconstructs_identical_remaining_stream(self) -> None:
        classes = ["flat"] * 4 + ["ascent"] * 4 + ["descent"] * 4 + ["transition"] * 4
        global_epoch = train_module._balanced_epoch_indices(
            list(range(16)), classes, seed=23, epoch=5
        )
        uninterrupted, padding = train_module._padded_rank_epoch(
            global_epoch, batch_size=2, rank=1, world_size=2
        )
        consumed_local = 4
        resumed, resumed_offset, resumed_padding = (
            train_module._local_epoch_at_global_offset(
                global_epoch,
                global_offset=consumed_local * 2,
                batch_size=2,
                rank=1,
                world_size=2,
            )
        )
        self.assertEqual(resumed_padding, padding)
        self.assertEqual(resumed_offset, consumed_local)
        self.assertEqual(resumed[resumed_offset:], uninterrupted[consumed_local:])
        for invalid in (-1, 2, len(global_epoch) + 4):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "sampler"):
                    train_module._local_epoch_at_global_offset(
                        global_epoch,
                        global_offset=invalid,
                        batch_size=2,
                        rank=1,
                        world_size=2,
                    )

    def test_sequence_sampler_covers_before_replacement_resumes_and_partitions_ddp(self) -> None:
        sampler = train_module.DeterministicSequenceSampler(
            7, batch_size=3, seed=41
        )
        first_nine = [
            index
            for _ in range(3)
            for index in sampler.next_batch()
        ]
        self.assertEqual(set(first_nine[:7]), set(range(7)))
        self.assertEqual(len(set(first_nine[:7])), 7)
        expected_second_pass = train_module.DeterministicSequenceSampler(
            7, batch_size=1, seed=41
        )
        for _ in range(7):
            expected_second_pass.next_batch()
        self.assertEqual(first_nine[7], expected_second_pass.next_batch()[0])

        uninterrupted = train_module.DeterministicSequenceSampler(
            7, batch_size=3, seed=41
        )
        uninterrupted.next_batch()
        state = uninterrupted.state_dict()
        self.assertEqual(
            set(state),
            {
                "seed",
                "sequence_count",
                "permutation_number",
                "permutation",
                "cursor",
            },
        )
        expected_next = uninterrupted.next_batch()
        resumed = train_module.DeterministicSequenceSampler(
            7, batch_size=3, seed=41
        )
        resumed.load_state_dict(state)
        self.assertEqual(resumed.next_batch(), expected_next)
        malformed = dict(state)
        malformed["permutation"] = list(reversed(state["permutation"]))
        with self.assertRaisesRegex(ValueError, "sampler"):
            resumed.load_state_dict(malformed)

        reference = train_module.DeterministicSequenceSampler(
            7, batch_size=4, seed=41
        )
        rank_zero = train_module.DeterministicSequenceSampler(
            7, batch_size=2, seed=41, rank=0, world_size=2
        )
        rank_one = train_module.DeterministicSequenceSampler(
            7, batch_size=2, seed=41, rank=1, world_size=2
        )
        for _ in range(4):
            global_batch = reference.next_batch()
            partitioned = tuple(
                index
                for pair in zip(rank_zero.next_batch(), rank_one.next_batch())
                for index in pair
            )
            self.assertEqual(partitioned, global_batch)
            self.assertEqual(rank_zero.state_dict(), rank_one.state_dict())

    def test_normal_selection_requires_closed_loop_but_overfit_is_provisional(self) -> None:
        provisional = selection_metadata(
            one_step_score=1.25, pipeline_overfit=True
        )
        self.assertEqual(provisional["selection_mode"], "pipeline_overfit_one_step")
        self.assertIsNone(provisional["closed_loop_score"])
        self.assertTrue(provisional["provisional"])
        with self.assertRaisesRegex(RuntimeError, "closed-loop"):
            selection_metadata(one_step_score=1.25, pipeline_overfit=False)

    def test_small_model_overfits_and_reloads_bitwise_on_cpu(self) -> None:
        torch.manual_seed(17)
        model = PhaseFunctionedNetwork(hidden_size=32, dropout_probability=0.0)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0e-2, weight_decay=0.0
        )
        x = torch.zeros(64, INPUT_LAYOUT.size)
        phase = torch.zeros(64)
        target = torch.zeros(64, OUTPUT_LAYOUT.size)
        direction = target[:, OUTPUT_LAYOUT["trajectory_direction"]].reshape(64, 12, 2)
        direction[..., 0] = 1.0
        target[:, OUTPUT_LAYOUT["phase_advance"]] = 0.1
        target[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor((0.0, 1.0, 0.0, 1.0))
        normal = _normalization()
        model.train()
        with torch.no_grad():
            initial = float(
                pfnn_losses(
                    model(x, phase), target, model=model, normalization=normal
                )["total"]
            )
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            losses = pfnn_losses(
                model(x, phase), target, model=model, normalization=normal
            )
            losses["total"].backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x, phase)
            final = float(
                pfnn_losses(
                    prediction, target, model=model, normalization=normal
                )["total"]
            )
        self.assertLess(final, 0.1 * initial)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path, model, optimizer, normal, dataset_digest="abc",
                kinematic_signature_sha256="def",
                runtime_seed=finite_runtime_seed(), step=300,
            )
            loaded = load_checkpoint(
                path, expected_dataset_digest="abc",
                expected_kinematic_signature_sha256="def",
            )
            restored = loaded.build_model()
            restored.eval()
            with torch.no_grad():
                reloaded_prediction = restored(x, phase)
            self.assertTrue(torch.equal(prediction, reloaded_prediction))


if __name__ == "__main__":
    unittest.main()
