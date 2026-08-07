import math
import unittest

import torch

from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from mm_sonic.terrain_pfnn.model import (
    PhaseFunctionedNetwork,
    catmull_rom_phase_banks,
)


class TerrainPFNNModelTest(unittest.TestCase):
    def test_layout_sizes_are_frozen(self) -> None:
        self.assertEqual(INPUT_LAYOUT.size, 288)
        self.assertEqual(OUTPUT_LAYOUT.size, 268)
        self.assertEqual(INPUT_LAYOUT["terrain_height"].stop - INPUT_LAYOUT["terrain_height"].start, 36)
        self.assertEqual(OUTPUT_LAYOUT["contact_logit"].stop - OUTPUT_LAYOUT["contact_logit"].start, 4)

    def test_phase_banks_hit_control_points_and_wrap(self) -> None:
        banks = torch.arange(4, dtype=torch.float64)[:, None]
        phase = torch.tensor(
            [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
            dtype=torch.float64,
        )
        actual = catmull_rom_phase_banks(banks, phase)[:, 0]
        torch.testing.assert_close(actual, torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0], dtype=torch.float64))

    def test_phase_interpolation_matches_equation_seven_scalar_reference(self) -> None:
        banks = torch.tensor([[-0.4], [0.2], [0.9], [0.1]], dtype=torch.float64)
        amount = 0.37
        phase = torch.tensor([(1.0 + amount) * 0.5 * math.pi], dtype=torch.float64)
        p0, p1, p2, p3 = (value.item() for value in banks)
        expected = (
            p1
            + 0.5 * amount * (p2 - p0)
            + amount**2 * (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3)
            + amount**3 * (1.5 * p1 - 1.5 * p2 + 0.5 * p3 - 0.5 * p0)
        )
        torch.testing.assert_close(
            catmull_rom_phase_banks(banks, phase)[0, 0],
            torch.tensor(expected, dtype=torch.float64),
        )

    def test_phase_wrap_has_matching_value_and_derivative(self) -> None:
        banks = torch.tensor(
            [[0.0, 0.1, -0.1], [0.2, 0.0, 0.05], [0.3, -0.1, 0.1], [0.1, 0.05, 0.0]],
            dtype=torch.float64,
        )
        epsilon = 1.0e-5
        left = catmull_rom_phase_banks(banks, torch.tensor([2.0 * math.pi - epsilon], dtype=torch.float64))
        zero = catmull_rom_phase_banks(banks, torch.tensor([0.0], dtype=torch.float64))
        right = catmull_rom_phase_banks(banks, torch.tensor([epsilon], dtype=torch.float64))
        torch.testing.assert_close(left, zero, atol=2.0e-5, rtol=0.0)
        torch.testing.assert_close(right, zero, atol=2.0e-5, rtol=0.0)
        torch.testing.assert_close((zero - left) / epsilon, (right - zero) / epsilon, atol=2.0e-4, rtol=0.0)

    def test_network_contract_and_gradient(self) -> None:
        model = PhaseFunctionedNetwork()
        x = torch.randn(5, 288, requires_grad=True)
        phase = torch.linspace(0.0, 2.0 * math.pi, 5)
        output = model(x, phase)
        self.assertEqual(tuple(output.shape), (5, 268))
        output.square().mean().backward()
        self.assertIsNotNone(model.W1.grad)
        self.assertTrue(torch.isfinite(model.W1.grad).all())


if __name__ == "__main__":
    unittest.main()
