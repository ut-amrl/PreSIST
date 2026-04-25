# Run inside container: python -m unittest discover -s tests -p "test_*.py" -v

import math
import unittest

import scipy.special
import torch

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.general_purpose import (
    GeneralPurposeSurvivalFunction,
    GeneralPurposeSurvivalFunctionGenerator,
)


def _make_fn(lambda_l: float, lambda_u: float) -> GeneralPurposeSurvivalFunction:
    return GeneralPurposeSurvivalFunction(
        lambda_l=torch.tensor(lambda_l, dtype=torch.float64),
        lambda_u=torch.tensor(lambda_u, dtype=torch.float64),
    )


def _reference_survival(t: float, lambda_l: float, lambda_u: float) -> float:
    """Scalar reference using scipy directly, matching the original implementation."""
    if t <= 0:
        return 1.0
    return (scipy.special.exp1(t * lambda_l) - scipy.special.exp1(t * lambda_u)) / math.log(lambda_u / lambda_l)


class TestGeneralPurposeSurvivalFunction(unittest.TestCase):

    def test_boundary_scalar_zero(self):
        self.assertAlmostEqual(_make_fn(0.1, 1.0)(torch.tensor(0.0)).item(), 1.0)

    def test_boundary_scalar_negative(self):
        self.assertAlmostEqual(_make_fn(0.1, 1.0)(torch.tensor(-5.0)).item(), 1.0)

    def test_matches_reference_scalar(self):
        lambda_l, lambda_u = 0.1, 2.0
        fn = _make_fn(lambda_l, lambda_u)
        for t in [0.5, 1.0, 5.0, 10.0]:
            expected = _reference_survival(t, lambda_l, lambda_u)
            self.assertAlmostEqual(fn(torch.tensor(t)).item(), expected, places=6)

    def test_matches_reference_batched(self):
        lambda_l, lambda_u = 0.05, 0.5
        fn = _make_fn(lambda_l, lambda_u)
        times = torch.tensor([1.0, 2.0, 5.0, 10.0])
        result = fn(times)
        for i, t in enumerate([1.0, 2.0, 5.0, 10.0]):
            expected = _reference_survival(t, lambda_l, lambda_u)
            self.assertAlmostEqual(result[i].item(), expected, places=6)

    def test_survival_is_decreasing(self):
        fn = _make_fn(0.1, 1.0)
        times = torch.tensor([0.1, 1.0, 5.0, 10.0, 50.0])
        values = fn(times)
        for i in range(len(values) - 1):
            self.assertGreater(values[i].item(), values[i + 1].item())

    def test_survival_in_unit_interval(self):
        fn = _make_fn(0.1, 1.0)
        times = torch.tensor([0.001, 0.1, 1.0, 10.0, 100.0])
        values = fn(times)
        self.assertTrue((values >= 0).all())
        self.assertTrue((values <= 1).all())

    def test_log_survival_consistent_with_call(self):
        fn = _make_fn(0.1, 1.0)
        times = torch.tensor([0.5, 1.0, 5.0, 10.0])
        torch.testing.assert_close(torch.exp(fn.log_survival(times)), fn(times))

    def test_log_survival_zero_at_boundary(self):
        fn = _make_fn(0.1, 1.0)
        self.assertAlmostEqual(fn.log_survival(torch.tensor(0.0)).item(), 0.0)
        self.assertAlmostEqual(fn.log_survival(torch.tensor(-1.0)).item(), 0.0)

    def test_batched_boundary_mixed(self):
        fn = _make_fn(0.1, 1.0)
        times = torch.tensor([-1.0, 0.0, 1.0])
        result = fn(times)
        self.assertEqual(result.shape, times.shape)
        self.assertAlmostEqual(result[0].item(), 1.0)
        self.assertAlmostEqual(result[1].item(), 1.0)
        self.assertLess(result[2].item(), 1.0)

    def test_buffers_not_parameters(self):
        fn = _make_fn(0.1, 1.0)
        self.assertEqual(list(fn.parameters()), [])


class TestGeneralPurposeSurvivalFunctionGenerator(unittest.TestCase):

    def _build(self, lambda_l: float, lambda_u: float) -> GeneralPurposeSurvivalFunction:
        estimate = PersistenceParameterEstimate(params={"lambda_l": lambda_l, "lambda_u": lambda_u})
        return GeneralPurposeSurvivalFunctionGenerator().build(estimate)

    def test_build_matches_direct(self):
        lambda_l, lambda_u = 0.1, 2.0
        fn = self._build(lambda_l, lambda_u)
        expected = _reference_survival(1.0, lambda_l, lambda_u)
        self.assertAlmostEqual(fn(torch.tensor(1.0)).item(), expected, places=6)

    def test_missing_lambda_l_raises(self):
        estimate = PersistenceParameterEstimate(params={"lambda_u": 1.0})
        with self.assertRaises(IncompatibleEstimateError):
            GeneralPurposeSurvivalFunctionGenerator().build(estimate)

    def test_missing_lambda_u_raises(self):
        estimate = PersistenceParameterEstimate(params={"lambda_l": 0.1})
        with self.assertRaises(IncompatibleEstimateError):
            GeneralPurposeSurvivalFunctionGenerator().build(estimate)

    def test_lambda_l_ge_lambda_u_raises(self):
        for lambda_l, lambda_u in [(1.0, 0.5), (1.0, 1.0)]:
            with self.assertRaises(IncompatibleEstimateError):
                self._build(lambda_l, lambda_u)


if __name__ == "__main__":
    unittest.main()
