"""
Unit tests for shared/pareto.py

Run with: pytest shared/tests/ -v
"""

from __future__ import annotations

import unittest

from shared.pareto import ParetoPoint, compute_pareto_efficiency


class TestComputeParetoEfficiency(unittest.TestCase):
    def test_dominated_point_is_flagged_inefficient(self):
        # A=(1,1) strictly beats B=(2,2) on both axes, so B is dominated.
        # C=(0.5,3) is worse than A on y but better on x, so neither A
        # nor C dominates the other: both stay on the frontier.
        a = ParetoPoint("A", x_value=1, y_value=1)
        b = ParetoPoint("B", x_value=2, y_value=2)
        c = ParetoPoint("C", x_value=0.5, y_value=3)

        result = compute_pareto_efficiency((a, b, c))

        self.assertEqual(result, {"A": True, "B": False, "C": True})

    def test_a_single_point_is_trivially_efficient(self):
        a = ParetoPoint("A", x_value=1, y_value=1)
        result = compute_pareto_efficiency((a,))
        self.assertEqual(result, {"A": True})

    def test_tied_points_do_not_dominate_each_other(self):
        a = ParetoPoint("A", x_value=1, y_value=1)
        b = ParetoPoint("B", x_value=1, y_value=1)
        result = compute_pareto_efficiency((a, b))
        self.assertEqual(result, {"A": True, "B": True})

    def test_raises_on_empty_points(self):
        with self.assertRaises(ValueError):
            compute_pareto_efficiency(())

    def test_raises_on_duplicate_names(self):
        a = ParetoPoint("A", x_value=1, y_value=1)
        a_again = ParetoPoint("A", x_value=2, y_value=2)
        with self.assertRaises(ValueError):
            compute_pareto_efficiency((a, a_again))


if __name__ == "__main__":
    unittest.main()
