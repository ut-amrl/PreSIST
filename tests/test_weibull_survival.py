# Run inside container: python -m unittest discover -s tests -p "test_*.py" -v

import math
import unittest

import torch

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.weibull import (
    WeibullSurvivalFunction,
    WeibullSurvivalFunctionGenerator,
)


def _make_fn(k: float, lam: float) -> WeibullSurvivalFunction:
    return WeibullSurvivalFunction(
        k=torch.tensor(k, dtype=torch.float64),
        lam=torch.tensor(lam, dtype=torch.float64),
    )


class TestWeibullSurvivalFunction(unittest.TestCase):

    def test_boundary_scalar_zero(self):
        fn = _make_fn(k=1.5, lam=10.0)
        self.assertAlmostEqual(fn(torch.tensor(0.0)).item(), 1.0)

    def test_boundary_scalar_negative(self):
        fn = _make_fn(k=1.5, lam=10.0)
        self.assertAlmostEqual(fn(torch.tensor(-5.0)).item(), 1.0)

    def test_at_scale_parameter(self):
        # S(lam) = exp(-1) for any k
        fn = _make_fn(k=2.0, lam=7.0)
        expected = math.exp(-1.0)
        self.assertAlmostEqual(fn(torch.tensor(7.0)).item(), expected, places=6)

    def test_exponential_special_case(self):
        # k=1 reduces to exponential: S(t) = exp(-t/lam)
        lam = 5.0
        fn = _make_fn(k=1.0, lam=lam)
        for t in [1.0, 5.0, 10.0, 20.0]:
            expected = math.exp(-t / lam)
            self.assertAlmostEqual(fn(torch.tensor(t)).item(), expected, places=6)

    def test_survival_is_decreasing(self):
        fn = _make_fn(k=1.2, lam=15.0)
        times = torch.tensor([1.0, 5.0, 10.0, 20.0, 50.0])
        values = fn(times)
        for i in range(len(values) - 1):
            self.assertGreater(values[i].item(), values[i + 1].item())

    def test_survival_in_unit_interval(self):
        fn = _make_fn(k=0.8, lam=20.0)
        times = torch.tensor([0.001, 1.0, 10.0, 100.0, 1000.0])
        values = fn(times)
        self.assertTrue((values >= 0).all())
        self.assertTrue((values <= 1).all())

    def test_log_survival_consistent_with_call(self):
        fn = _make_fn(k=1.5, lam=10.0)
        times = torch.tensor([0.5, 1.0, 5.0, 10.0, 50.0])
        log_s = fn.log_survival(times)
        s = fn(times)
        torch.testing.assert_close(torch.exp(log_s), s)

    def test_log_survival_zero_at_boundary(self):
        fn = _make_fn(k=1.5, lam=10.0)
        self.assertAlmostEqual(fn.log_survival(torch.tensor(0.0)).item(), 0.0)
        self.assertAlmostEqual(fn.log_survival(torch.tensor(-1.0)).item(), 0.0)

    def test_batched_input(self):
        fn = _make_fn(k=1.0, lam=10.0)
        times = torch.tensor([-1.0, 0.0, 5.0, 10.0, 20.0])
        result = fn(times)
        self.assertEqual(result.shape, times.shape)
        # dt <= 0 → 1.0
        self.assertAlmostEqual(result[0].item(), 1.0)
        self.assertAlmostEqual(result[1].item(), 1.0)
        # k=1: S(t) = exp(-t/lam)
        self.assertAlmostEqual(result[2].item(), math.exp(-0.5), places=6)
        self.assertAlmostEqual(result[3].item(), math.exp(-1.0), places=6)
        self.assertAlmostEqual(result[4].item(), math.exp(-2.0), places=6)


class TestWeibullSurvivalFunctionGenerator(unittest.TestCase):

    def _build(self, t50: float, t90: float) -> WeibullSurvivalFunction:
        estimate = PersistenceParameterEstimate(params={"t50": t50, "t90": t90})
        return WeibullSurvivalFunctionGenerator().build(estimate)

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

    def test_custom_keys(self):
        estimate = PersistenceParameterEstimate(params={"t25": 5.0, "t75": 20.0})
        gen = WeibullSurvivalFunctionGenerator(low_key="t25", high_key="t75")
        fn = gen.build(estimate)
        self.assertAlmostEqual(fn(torch.tensor(5.0)).item(), 0.75, places=5)
        self.assertAlmostEqual(fn(torch.tensor(20.0)).item(), 0.25, places=5)

    def test_missing_key_raises(self):
        estimate = PersistenceParameterEstimate(params={"t50": 10.0})
        with self.assertRaises(IncompatibleEstimateError):
            WeibullSurvivalFunctionGenerator().build(estimate)

    def test_missing_both_keys_raises(self):
        estimate = PersistenceParameterEstimate(params={})
        with self.assertRaises(IncompatibleEstimateError):
            WeibullSurvivalFunctionGenerator().build(estimate)

    def test_percentile_from_key(self):
        parse = WeibullSurvivalFunctionGenerator._percentile_from_key
        self.assertAlmostEqual(parse("t50"), 0.50)
        self.assertAlmostEqual(parse("t90"), 0.90)
        self.assertAlmostEqual(parse("t25"), 0.25)
        self.assertAlmostEqual(parse("t75"), 0.75)

    def test_percentile_from_key_invalid_raises(self):
        with self.assertRaises(ValueError):
            WeibullSurvivalFunctionGenerator._percentile_from_key("tmedian")


if __name__ == "__main__":
    unittest.main()
