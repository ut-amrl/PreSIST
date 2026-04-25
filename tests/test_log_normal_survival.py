# Run inside container: python -m unittest discover -s tests -p "test_*.py" -v

import math
import unittest

import torch

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.log_normal import (
    LogNormalSurvivalFunction,
    LogNormalSurvivalFunctionGenerator,
)


def _make_fn(mu: float, sigma: float) -> LogNormalSurvivalFunction:
    return LogNormalSurvivalFunction(
        mu=torch.tensor(mu, dtype=torch.float64),
        sigma=torch.tensor(sigma, dtype=torch.float64),
    )


class TestLogNormalSurvivalFunction(unittest.TestCase):

    def test_boundary_scalar_zero(self):
        self.assertAlmostEqual(_make_fn(1.0, 0.5)(torch.tensor(0.0)).item(), 1.0)

    def test_boundary_scalar_negative(self):
        self.assertAlmostEqual(_make_fn(1.0, 0.5)(torch.tensor(-5.0)).item(), 1.0)

    def test_at_median(self):
        # The median of LogNormal(mu, sigma) is exp(mu), so S(exp(mu)) = 0.5
        mu = 2.0
        fn = _make_fn(mu=mu, sigma=0.8)
        self.assertAlmostEqual(fn(torch.tensor(math.exp(mu))).item(), 0.5, places=6)

    def test_survival_is_decreasing(self):
        fn = _make_fn(mu=2.0, sigma=0.5)
        times = torch.tensor([1.0, 5.0, 10.0, 20.0, 100.0])
        values = fn(times)
        for i in range(len(values) - 1):
            self.assertGreater(values[i].item(), values[i + 1].item())

    def test_survival_in_unit_interval(self):
        fn = _make_fn(mu=2.0, sigma=0.5)
        times = torch.tensor([0.001, 1.0, 10.0, 100.0, 1000.0])
        values = fn(times)
        self.assertTrue((values >= 0).all())
        self.assertTrue((values <= 1).all())

    def test_log_survival_consistent_with_call(self):
        fn = _make_fn(mu=2.0, sigma=0.5)
        times = torch.tensor([0.5, 1.0, 5.0, 10.0, 50.0])
        torch.testing.assert_close(torch.exp(fn.log_survival(times)), fn(times))

    def test_log_survival_zero_at_boundary(self):
        fn = _make_fn(mu=2.0, sigma=0.5)
        self.assertAlmostEqual(fn.log_survival(torch.tensor(0.0)).item(), 0.0)
        self.assertAlmostEqual(fn.log_survival(torch.tensor(-1.0)).item(), 0.0)

    def test_batched_input(self):
        fn = _make_fn(mu=0.0, sigma=1.0)
        times = torch.tensor([-1.0, 0.0, 1.0])
        result = fn(times)
        self.assertEqual(result.shape, times.shape)
        self.assertAlmostEqual(result[0].item(), 1.0)
        self.assertAlmostEqual(result[1].item(), 1.0)
        # S(exp(0)) = S(1) = 0.5 for LogNormal(0, 1)
        self.assertAlmostEqual(result[2].item(), 0.5, places=6)


class TestLogNormalSurvivalFunctionGenerator(unittest.TestCase):

    def _build(self, t50: float, t90: float) -> LogNormalSurvivalFunction:
        estimate = PersistenceParameterEstimate(params={"t50": t50, "t90": t90})
        return LogNormalSurvivalFunctionGenerator().build(estimate)

    def test_roundtrip_t50(self):
        fn = self._build(t50=10.0, t90=30.0)
        self.assertAlmostEqual(fn(torch.tensor(10.0)).item(), 0.5, places=5)

    def test_roundtrip_t90(self):
        fn = self._build(t50=10.0, t90=30.0)
        self.assertAlmostEqual(fn(torch.tensor(30.0)).item(), 0.1, places=5)

    def test_roundtrip_different_times(self):
        fn = self._build(t50=100.0, t90=500.0)
        self.assertAlmostEqual(fn(torch.tensor(100.0)).item(), 0.5, places=5)
        self.assertAlmostEqual(fn(torch.tensor(500.0)).item(), 0.1, places=5)

    def test_mu_equals_log_t50(self):
        # With t50, z_low = Phi_inv(0.5) = 0, so mu = log(t50)
        t50 = 20.0
        estimate = PersistenceParameterEstimate(params={"t50": t50, "t90": 60.0})
        fn = LogNormalSurvivalFunctionGenerator().build(estimate)
        self.assertAlmostEqual(fn._mu.item(), math.log(t50), places=6)

    def test_custom_keys(self):
        estimate = PersistenceParameterEstimate(params={"t25": 5.0, "t75": 20.0})
        gen = LogNormalSurvivalFunctionGenerator(low_key="t25", high_key="t75")
        fn = gen.build(estimate)
        self.assertAlmostEqual(fn(torch.tensor(5.0)).item(), 0.75, places=5)
        self.assertAlmostEqual(fn(torch.tensor(20.0)).item(), 0.25, places=5)

    def test_missing_key_raises(self):
        estimate = PersistenceParameterEstimate(params={"t50": 10.0})
        with self.assertRaises(IncompatibleEstimateError):
            LogNormalSurvivalFunctionGenerator().build(estimate)

    def test_percentile_from_key(self):
        parse = LogNormalSurvivalFunctionGenerator._percentile_from_key
        self.assertAlmostEqual(parse("t50"), 0.50)
        self.assertAlmostEqual(parse("t90"), 0.90)
        self.assertAlmostEqual(parse("t25"), 0.25)

    def test_percentile_from_key_invalid_raises(self):
        with self.assertRaises(ValueError):
            LogNormalSurvivalFunctionGenerator._percentile_from_key("tmedian")


if __name__ == "__main__":
    unittest.main()
