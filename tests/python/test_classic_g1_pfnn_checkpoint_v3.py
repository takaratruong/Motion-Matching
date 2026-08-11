from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mm_sonic.train_classic_g1_pfnn import (
    CLASSIC_CHECKPOINT_SCHEMA,
    load_classic_checkpoint,
    save_classic_checkpoint,
)
from mm_sonic.terrain_pfnn.dataset import pfnn_input_sha256
from mm_sonic.terrain_pfnn.layout import (
    CLASSIC_G1_INPUT_LAYOUT_V3,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
)
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.terrain_pfnn.training import finite_runtime_seed


def _runtime_seed() -> dict[str, object]:
    seed = finite_runtime_seed()
    normalized = np.zeros(CLASSIC_G1_INPUT_LAYOUT_V3.size, dtype=np.float32)
    normalized[: INPUT_LAYOUT.size] = np.asarray(seed["normalized_input"])
    seed["joint_velocity"] = torch.zeros(29, dtype=torch.float32)
    seed["normalized_input"] = torch.as_tensor(normalized)
    seed["normalized_input_sha256"] = pfnn_input_sha256(normalized)
    return seed


def _normalization() -> dict[str, torch.Tensor]:
    return {
        "x_mean": torch.zeros(CLASSIC_G1_INPUT_LAYOUT_V3.size),
        "x_std": torch.ones(CLASSIC_G1_INPUT_LAYOUT_V3.size),
        "y_mean": torch.zeros(OUTPUT_LAYOUT.size),
        "y_std": torch.ones(OUTPUT_LAYOUT.size),
    }


def _save(path: Path, model: PhaseFunctionedNetwork) -> None:
    save_classic_checkpoint(
        path,
        model=model,
        normalization=_normalization(),
        runtime_seed=_runtime_seed(),
        dataset_digest="a" * 64,
        kinematic_signature_sha256="b" * 64,
        joint_limits=torch.tensor([[-2.0, 2.0]] * 29),
        phase_advance_q99=0.2,
        epoch=3,
        validation_loss=0.125,
        seed=7,
        source_kind="released_pfnn",
        vertical_slice_receipt_sha256="c" * 64,
        terrain_receipt_set_sha256="d" * 64,
    )


class ClassicG1PFNNCheckpointV3Tests(unittest.TestCase):
    def test_round_trip_binds_exact_v3_input_contract(self) -> None:
        torch.manual_seed(7)
        model = PhaseFunctionedNetwork(
            hidden_size=16,
            dropout_probability=0.0,
            input_size=CLASSIC_G1_INPUT_LAYOUT_V3.size,
        ).eval()
        x = torch.randn(3, CLASSIC_G1_INPUT_LAYOUT_V3.size)
        phase = torch.tensor((0.0, 1.0, 2.0))
        expected = model(x, phase).detach()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _save(path, model)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            loaded = load_classic_checkpoint(path)

        self.assertEqual(payload["schema"], "classic-g1-pfnn/v3")
        self.assertEqual(CLASSIC_CHECKPOINT_SCHEMA, "classic-g1-pfnn/v3")
        self.assertEqual(payload["input_size"], 346)
        self.assertEqual(
            payload["input_layout"],
            [list(field) for field in CLASSIC_G1_INPUT_LAYOUT_V3.fields],
        )
        self.assertEqual(loaded.input_size, 346)
        actual = loaded.build_model().eval()(x, phase).detach()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_save_rejects_default_v2_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "input"):
                _save(
                    Path(directory) / "legacy.pt",
                    PhaseFunctionedNetwork(hidden_size=8),
                )

    def test_load_rejects_contract_tampering_and_relabeled_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "valid.pt"
            _save(
                path,
                PhaseFunctionedNetwork(
                    hidden_size=8,
                    input_size=CLASSIC_G1_INPUT_LAYOUT_V3.size,
                ),
            )
            valid = torch.load(path, map_location="cpu", weights_only=True)
            tampered = (
                ("input-size", lambda value: value.__setitem__("input_size", 288)),
                (
                    "input-layout",
                    lambda value: value.__setitem__(
                        "input_layout", [list(field) for field in INPUT_LAYOUT.fields]
                    ),
                ),
                (
                    "schema",
                    lambda value: value.__setitem__(
                        "schema", "classic-g1-pfnn/v2"
                    ),
                ),
                (
                    "joint-order",
                    lambda value: value["canonical_joint_order"].reverse(),
                ),
                (
                    "runtime-seed-qdot",
                    lambda value: value["runtime_seed"].pop("joint_velocity"),
                ),
                (
                    "runtime-seed-q-binding",
                    lambda value: value["runtime_seed"].__setitem__(
                        "joint_position",
                        value["runtime_seed"]["joint_position"] + 0.1,
                    ),
                ),
            )
            for label, mutate in tampered:
                with self.subTest(label=label):
                    payload = copy.deepcopy(valid)
                    mutate(payload)
                    candidate = root / f"tampered-{label}.pt"
                    torch.save(payload, candidate)
                    with self.assertRaisesRegex(
                        ValueError, "schema|contract|runtime seed"
                    ):
                        load_classic_checkpoint(candidate)

            relabeled_v2 = copy.deepcopy(valid)
            relabeled_v2["schema"] = "classic-g1-pfnn/v3"
            relabeled_v2["input_size"] = INPUT_LAYOUT.size
            relabeled_v2["input_layout"] = [list(field) for field in INPUT_LAYOUT.fields]
            relabeled_v2["model_state"]["W0"] = relabeled_v2["model_state"]["W0"][
                :, :, : INPUT_LAYOUT.size
            ]
            candidate = root / "relabeled-v2.pt"
            torch.save(relabeled_v2, candidate)
            with self.assertRaisesRegex(ValueError, "contract"):
                load_classic_checkpoint(candidate)


if __name__ == "__main__":
    unittest.main()
