# Run inside container: python -m unittest discover -s tests -p "test_*.py" -v

import math
import unittest

import torch

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.exponential import (
    ExponentialSurvivalFunction,
    ExponentialSurvivalFunctionGenerator,
)


def _make_fn(lam: float) -> ExponentialSurvivalFunction:
    return ExponentialSurvivalFunction(lam=torch.tensor(lam, dtype=torch.float64))


class TestExponentialSurvivalFunction(unittest.TestCase):

    def test_boundary_scalar_zero(self):
        self.assertAlmostEqual(_make_fn(10.0)(torch.tensor(0.0)).item(), 1.0)

    def test_boundary_scalar_negative(self):
        self.assertAlmostEqual(_make_fn(10.0)(torch.tensor(-5.0)).item(), 1.0)

    def test_at_mean(self):
        # S(lam) = exp(-1) for any lam
        fn = _make_fn(7.0)
        self.assertAlmostEqual(fn(torch.tensor(7.0)).item(), math.exp(-1.0), places=6)

    def test_known_values(self):
        lam = 5.0
        fn = _make_fn(lam)
        for t in [1.0, 5.0, 10.0, 20.0]:
            self.assertAlmostEqual(fn(torch.tensor(t)).item(), math.exp(-t / lam), places=6)

    def test_survival_is_decreasing(self):
        fn = _make_fn(15.0)
        times = torch.tensor([1.0, 5.0, 10.0, 20.0, 50.0])
        values = fn(times)
        for i in range(len(values) - 1):
            self.assertGreater(values[i].item(), values[i + 1].item())

    def test_survival_in_unit_interval(self):
        fn = _make_fn(20.0)
        times = torch.tensor([0.001, 1.0, 10.0, 100.0, 1000.0])
        values = fn(times)
        self.assertTrue((values >= 0).all())
        self.assertTrue((values <= 1).all())

    def test_log_survival_consistent_with_call(self):
        fn = _make_fn(10.0)
        times = torch.tensor([0.5, 1.0, 5.0, 10.0, 50.0])
        torch.testing.assert_close(torch.exp(fn.log_survival(times)), fn(times))

    def test_log_survival_zero_at_boundary(self):
        fn = _make_fn(10.0)
        self.assertAlmostEqual(fn.log_survival(torch.tensor(0.0)).item(), 0.0)
        self.assertAlmostEqual(fn.log_survival(torch.tensor(-1.0)).item(), 0.0)

    def test_batched_input(self):
        lam = 10.0
        fn = _make_fn(lam)
        times = torch.tensor([-1.0, 0.0, 5.0, 10.0, 20.0])
        result = fn(times)
        self.assertEqual(result.shape, times.shape)
        self.assertAlmostEqual(result[0].item(), 1.0)
        self.assertAlmostEqual(result[1].item(), 1.0)
        self.assertAlmostEqual(result[2].item(), math.exp(-0.5), places=6)
        self.assertAlmostEqual(result[3].item(), math.exp(-1.0), places=6)
        self.assertAlmostEqual(result[4].item(), math.exp(-2.0), places=6)


class TestExponentialSurvivalFunctionGenerator(unittest.TestCase):

    def test_from_mean(self):
        estimate = PersistenceParameterEstimate(params={"mean": 10.0})
        fn = ExponentialSurvivalFunctionGenerator().build(estimate)
        # S(mean) = exp(-1)
        self.assertAlmostEqual(fn(torch.tensor(10.0)).item(), math.exp(-1.0), places=6)

    def test_from_t50(self):
        estimate = PersistenceParameterEstimate(params={"t50": 10.0})
        fn = ExponentialSurvivalFunctionGenerator().build(estimate)
        # S(t50) = 0.5 by definition
        self.assertAlmostEqual(fn(torch.tensor(10.0)).item(), 0.5, places=6)

    def test_mean_takes_priority_over_t50(self):
        # 'mean' and 't50' give different lam values; mean should win
        mean = 20.0
        estimate = PersistenceParameterEstimate(params={"mean": mean, "t50": 5.0})
        fn = ExponentialSurvivalFunctionGenerator().build(estimate)
        self.assertAlmostEqual(fn(torch.tensor(mean)).item(), math.exp(-1.0), places=6)

    def test_mean_and_t50_consistent(self):
        # If t50 = mean * ln(2), both derivations should give the same lam
        mean = 15.0
        t50 = mean * math.log(2)
        fn_mean = ExponentialSurvivalFunctionGenerator().build(
            PersistenceParameterEstimate(params={"mean": mean})
        )
        fn_t50 = ExponentialSurvivalFunctionGenerator().build(
            PersistenceParameterEstimate(params={"t50": t50})
        )
        for t in [1.0, 10.0, 30.0]:
            self.assertAlmostEqual(
                fn_mean(torch.tensor(t)).item(),
                fn_t50(torch.tensor(t)).item(),
                places=5,
            )

    def test_missing_keys_raises(self):
        estimate = PersistenceParameterEstimate(params={"t90": 30.0})
        with self.assertRaises(IncompatibleEstimateError):
            ExponentialSurvivalFunctionGenerator().build(estimate)


if __name__ == "__main__":
    unittest.main()
