import unittest
from unittest.mock import patch

import torch

from sampling.Sampler78rpm import GramophoneDegradationModel


class GramophoneDegradationModelTests(unittest.TestCase):
    def setUp(self):
        self.noises = [
            torch.ones(8),
            torch.arange(1, 9, dtype=torch.float32),
            torch.arange(2, 18, 2, dtype=torch.float32),
        ]
        self.signal = torch.arange(8, dtype=torch.float32)

    def test_requires_step_before_evaluation(self):
        degradation = GramophoneDegradationModel(
            self.noises,
            num_steps=2,
            gain=2.0,
        )

        with self.assertRaises(RuntimeError):
            degradation(self.signal)

    def test_reuses_one_noise_realization_within_step(self):
        degradation = GramophoneDegradationModel(
            self.noises,
            num_steps=2,
            gain=2.0,
        )
        degradation.setup_degradation(
            step=0, x_i=self.signal, t_i=torch.tensor(1.0)
        )

        first = degradation(self.signal)
        second = degradation(self.signal)

        torch.testing.assert_close(first, second)

    def test_uses_the_precomputed_noise_sequence(self):
        with patch(
            "sampling.Sampler78rpm.random.randrange",
            side_effect=[0, 1],
        ):
            degradation = GramophoneDegradationModel(
                self.noises,
                num_steps=2,
                gain=0.0,
            )

        degradation.setup_degradation(
            step=0, x_i=self.signal, t_i=torch.tensor(1.0)
        )
        first = degradation(self.signal)
        degradation.setup_degradation(
            step=1, x_i=self.signal, t_i=torch.tensor(1.0)
        )
        second = degradation(self.signal)

        torch.testing.assert_close(first, self.signal / 2)
        torch.testing.assert_close(second, self.signal / 2)
        self.assertEqual(degradation.noise_indices, [0, 1])
        self.assertFalse(torch.equal(degradation.noise, self.noises[0]))

    def test_affine_operator_products(self):
        degradation = GramophoneDegradationModel(
            self.noises,
            num_steps=1,
            gain=2.0,
        )
        degradation.setup_degradation(
            step=0, x_i=self.signal, t_i=torch.tensor(1.0)
        )
        vector = torch.arange(8, dtype=torch.float32)

        torch.testing.assert_close(degradation.jvp(self.signal, vector), vector / 2)
        torch.testing.assert_close(degradation.vjp(self.signal, vector), vector / 2)

    def test_rejects_out_of_range_step(self):
        degradation = GramophoneDegradationModel(
            self.noises,
            num_steps=1,
            gain=1.0,
        )

        with self.assertRaises(IndexError):
            degradation.setup_degradation(
                step=1, x_i=self.signal, t_i=torch.tensor(1.0)
            )

    def test_updates_conditioner_xi_when_enabled(self):
        conditioner = type("Conditioner", (), {"xi": 0.0})()
        with patch("sampling.Sampler78rpm.random.uniform", return_value=10.0):
            degradation = GramophoneDegradationModel(
                self.noises,
                num_steps=1,
                snr_range=[0.0, 20.0],
                update_xi=True,
                conditioner=conditioner,
            )
            degradation.setup_degradation(
                step=0, x_i=self.signal, t_i=torch.tensor(1.0)
            )

        self.assertAlmostEqual(conditioner.xi, 0.35)


if __name__ == "__main__":
    unittest.main()
