"""
Unit tests for core/frontier.py

Run with: pytest tests/
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import frontier as f  # noqa: E402
from core import models as m  # noqa: E402


def _profile(name, performance_value, resource_value):
    return m.ModelProfile(
        name=name,
        performance_metric_name="MSE",
        performance_value=performance_value,
        performance_is_approximate=False,
        n_parameters=1000,
        resource_metric_name="CO2 (kg)",
        resource_value=resource_value,
        resource_is_approximate=False,
    )


class TestComputeFrontier(unittest.TestCase):
    def test_dominated_model_is_flagged_inefficient(self):
        # A=(1,1) strictly beats B=(2,2) on both dimensions, so B is
        # dominated. C=(0.5,3) is worse than A on resource but better on
        # performance, so neither A nor C dominates the other: both stay
        # on the frontier.
        a = _profile("A", 1, 1)
        b = _profile("B", 2, 2)
        c = _profile("C", 0.5, 3)

        result = f.compute_frontier((a, b, c))

        self.assertEqual(
            result.is_efficient, {"A": True, "B": False, "C": True}
        )

    def test_a_single_model_is_trivially_efficient(self):
        a = _profile("A", 1, 1)
        result = f.compute_frontier((a,))
        self.assertEqual(result.is_efficient, {"A": True})

    def test_tied_models_do_not_dominate_each_other(self):
        a = _profile("A", 1, 1)
        b = _profile("B", 1, 1)
        result = f.compute_frontier((a, b))
        self.assertEqual(result.is_efficient, {"A": True, "B": True})

    def test_raises_on_empty_profiles(self):
        with self.assertRaises(ValueError):
            f.compute_frontier(())


if __name__ == "__main__":
    unittest.main()
